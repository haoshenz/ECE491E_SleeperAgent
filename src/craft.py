"""Global gradient matching with an exact, memory-bounded two-pass derivative."""
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .common import atomic_save, seed_all
from .data import augment, project, STD
from .patch import apply_patch
from .progress import report_progress


def flat_grad(model, x, y, create_graph=False):
    grads = torch.autograd.grad(F.cross_entropy(model(x), y), tuple(model.parameters()),
                                create_graph=create_graph)
    return torch.cat([g.reshape(-1) for g in grads])


def average_gradient(model, x, y, batch_size, patch=None, location='random', seed=0):
    model.eval()
    generator = torch.Generator(device=x.device).manual_seed(seed)
    total = None
    for start in range(0, len(x), batch_size):
        xb = x[start:start+batch_size]
        if patch is not None:
            xb = apply_patch(xb, patch, location, generator)
        g = flat_grad(model, xb, y[start:start+batch_size]).detach() * (len(xb) / len(x))
        total = g if total is None else total + g
    return total.detach()


def alignment_step(model, clean, delta, labels, adversarial, batch_size, seed):
    """Exact derivative of ONE global cosine, replaying identical augmentations.

    First obtain g without keeping activation graphs. Then d(1-cos(g,a))/dg
    is a detached coefficient for a second pass through each minibatch.
    Only one minibatch's double-backward graph is live at a time.
    BatchNorm stays in eval mode, so replay is the same function.
    """
    model.eval()
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    total = None
    for start in range(0, len(clean), batch_size):
        sl = slice(start, start+batch_size)
        xb = augment(clean[sl] + delta[sl].detach(), generator)
        g = flat_grad(model, xb, labels[sl]).detach() * (len(xb) / len(clean))
        total = g if total is None else total + g
    norm = total.norm().clamp_min(1e-12)
    anorm = adversarial.norm().clamp_min(1e-12)
    cosine = torch.dot(total, adversarial) / (norm * anorm)
    coefficient = (-adversarial/(norm*anorm) + cosine*total/norm.square()).detach()
    generator.manual_seed(seed)
    result = torch.zeros_like(delta)
    for start in range(0, len(clean), batch_size):
        sl = slice(start, start+batch_size)
        part = delta[sl].detach().requires_grad_(True)
        xb = augment(clean[sl] + part, generator)
        g = flat_grad(model, xb, labels[sl], create_graph=True)
        objective = torch.dot(g, coefficient) * (len(xb)/len(clean))
        result[sl] = torch.autograd.grad(objective, part)[0].detach()
    assert bool(torch.isfinite(result).all()), 'Non-finite double-backward gradient'
    return float(cosine), result


def signed_adam_step(optimizer, delta, gradient, units='pixel'):
    optimizer.zero_grad(set_to_none=True)
    delta.grad = gradient.sign()
    previous = delta.detach().clone() if units == 'normalized' else None
    if units not in ('pixel', 'normalized'):
        raise ValueError(f'Unknown Adam coordinate system: {units}')
    optimizer.step()
    if previous is not None:
        # sign(dL/d(delta/std)) == sign(dL/d(delta)); the Adam moments are
        # identical. Convert its normalized-coordinate update back to pixels.
        with torch.no_grad():
            delta.copy_(previous + (delta-previous)*delta.new_tensor(STD)[None, :, None, None])


def craft(model, x, y, indices, source_images, target, patch, cfg, directory, retrain):
    directory = Path(directory)
    path = directory / 'craft.pt'
    seed = cfg['seed']
    clean = x[indices]
    seed_all(seed + 300)
    delta = torch.empty_like(clean).uniform_(-cfg['epsilon'], cfg['epsilon'])
    project(delta, clean, cfg['epsilon'])
    delta.requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=cfg['craft_lr'])
    milestones = [int(cfg['craft_steps']//k) for k in (2.667, 1.6, 1.142)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=.1)
    events = {cfg['craft_steps'] * k // (cfg['retraining_factor']+1)
              for k in range(1, cfg['retraining_factor']+1)}
    labels = y[indices]
    source_labels = torch.full((len(source_images),), target, device=x.device, dtype=torch.long)
    history, start, elapsed, compute_seconds = [], 0, 0.0, 0.0
    adversarial = None
    if path.exists():
        state = torch.load(path, map_location=x.device, weights_only=False)
        model.load_state_dict(state['model'])
        with torch.no_grad():
            delta.copy_(state['delta'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history, start = state['history'], state['step']
        elapsed = state['seconds']
        adversarial = state['adversarial']
        compute_seconds = state.get('compute_seconds', 0.0)
    clock = time.perf_counter()
    report_progress(cfg, directory, 'craft', 'Crafting tiny image changes', start,
                    cfg['craft_steps'], compute_seconds, announce=True)
    for step in range(start, cfg['craft_steps']):
        step_clock = time.perf_counter()
        if adversarial is None:
            adversarial = average_gradient(model, source_images, source_labels,
                cfg['craft_batch_size'], patch, cfg['patch_location'], seed+400+step)
        cosine, gradient = alignment_step(model, clean, delta, labels, adversarial,
                                         cfg['craft_batch_size'], seed+500+step)
        signed_adam_step(optimizer, delta, gradient, cfg.get('craft_lr_units', 'pixel'))
        project(delta, clean, cfg['epsilon'])
        scheduler.step()
        history.append({'step': step+1, 'cosine': cosine, 'lr': optimizer.param_groups[0]['lr']})
        compute_seconds += time.perf_counter()-step_clock
        report_progress(cfg, directory, 'craft', 'Crafting tiny image changes', step+1,
                        cfg['craft_steps'], compute_seconds,
                        announce=step+1 == cfg['craft_steps'], cosine=cosine)
        print(f'craft {step+1}/{cfg["craft_steps"]} cosine={cosine:.5f}', flush=True)
        if step+1 in events:
            # Fresh random weights, standard training on current poisoned data.
            model = retrain(delta.detach(), step+1)
            adversarial = None
        if (step+1) % cfg['checkpoint_every'] == 0 or step+1 in events or step+1 == cfg['craft_steps']:
            atomic_save({'model': model.state_dict(), 'delta': delta.detach(),
                         'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                         'adversarial': adversarial, 'history': history, 'step': step+1,
                         'compute_seconds': compute_seconds,
                         'seconds': elapsed+time.perf_counter()-clock}, path)
    return delta.detach(), history
