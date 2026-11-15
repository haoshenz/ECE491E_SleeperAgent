"""Independent, resumable benchmark-style Saha and Turner poison crafting.

Provenance and deliberate protocol differences: docs/table6_baselines.md.
All tensors and constraints use raw [0,1] pixels, never normalized coordinates.
"""
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from .common import atomic_save, seed_all
from .data import project
from .models import penultimate
from .progress import report_progress


DEFAULTS = {
    'htba': {'steps': 5000, 'lr': .001, 'decay_every': 2000, 'decay': .95,
             'early_stop_loss': 10., 'patch_margin': 5, 'batch_size': 128},
    'clba': {'steps': 20, 'step_size': 2/255, 'patch_margin': 0, 'batch_size': 128,
             'trigger_mode': 'stamp_then_project', 'opacity': 1.},
}


def settings(cfg, attack):
    spec = dict(DEFAULTS[attack], **cfg.get('baseline_settings', {}))
    if spec['steps'] <= 0 or spec['batch_size'] <= 0:
        raise ValueError('Baseline steps and batch size must be positive')
    if attack == 'clba' and (spec['opacity'] != 1. or spec['trigger_mode'] != 'stamp_then_project'):
        raise ValueError('CLBA supports benchmark full stamping followed by projection only')
    return spec


def stamp(x, patch, margin=0):
    """Benchmark placement: bottom-right, optionally inset by five pixels."""
    out = x.clone()
    p = patch.shape[-1]
    row, col = x.shape[-2]-p-margin, x.shape[-1]-p-margin
    if min(row, col) < 0:
        raise ValueError('Patch does not fit')
    out[:, :, row:row+p, col:col+p] = patch
    return out


def clipped_trigger(clean, adversarial, patch, epsilon, margin=0):
    """Matches the benchmark's stamp followed by final pixel-space projection."""
    delta = stamp(adversarial, patch, margin) - clean
    project(delta, clean, epsilon)
    return clean + delta


def greedy_assignment(source_features, poison_features):
    """Globally closest available pair, with a true one-to-one assignment.

    Stable sorting avoids 500 serial GPU synchronizations per crafting step.
    Reference code masks only one matrix entry and can repeat rows/columns;
    we explicitly prevent that (documented correction).
    """
    distances = torch.cdist(source_features, poison_features).detach().cpu().numpy()
    if distances.shape[0] != distances.shape[1]:
        raise ValueError('HTBA requires one training source per poison')
    n = len(distances)
    order = np.argsort(distances.reshape(-1), kind='stable')
    rows, cols = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    assignment = np.empty(n, dtype=np.int64)
    count = 0
    for flat in order:
        row, col = divmod(int(flat), n)
        if not rows[row] and not cols[col]:
            assignment[col] = row
            rows[row] = cols[col] = True
            count += 1
            if count == n:
                break
    return torch.as_tensor(assignment, device=source_features.device)


@torch.no_grad()
def features(model, x, batch_size):
    return torch.cat([penultimate(model, part) for part in x.split(batch_size)])


def htba_step(model, clean, delta, source_features, batch_size, lr):
    current = features(model, clean + delta, batch_size)
    matched = source_features[greedy_assignment(source_features, current)]
    loss_sum = 0.
    result = delta.detach().clone()
    for start in range(0, len(clean), batch_size):
        sl = slice(start, start+batch_size)
        part = delta[sl].detach().requires_grad_(True)
        loss = (penultimate(model, clean[sl]+part) - matched[sl]).square().sum()
        grad = torch.autograd.grad(loss, part)[0]
        if not bool(torch.isfinite(grad).all()):
            raise RuntimeError('Non-finite HTBA gradient')
        result[sl] -= lr * grad.detach()
        loss_sum += float(loss.detach())
    return result, loss_sum


def craft_baseline(model, x, y, indices, source_images, patch, cfg, directory,
                   *, step_limit=None):
    """Train-only inputs. Checkpoints replay exactly; step_limit is timing-only."""
    attack = cfg['attack']
    spec = settings(cfg, attack)
    path = Path(directory)/'baseline_craft.pt'
    clean, labels = x[indices], y[indices]
    model.eval()
    seed_all(cfg['seed']+300)
    delta = (torch.empty_like(clean).uniform_(-cfg['epsilon'], cfg['epsilon'])
             if attack == 'clba' else torch.zeros_like(clean))
    # Reference CLBA random start is not projected until its first PGD update.
    # Its adversarial inputs are still clamped by every subsequent update.
    history, start, elapsed, completed = [], 0, 0., False
    if path.exists():
        state = torch.load(path, map_location=x.device, weights_only=False)
        if state['settings'] != spec or state['attack'] != attack:
            raise ValueError('Baseline checkpoint settings changed; choose a new directory')
        delta, history, start, elapsed = state['delta'], state['history'], state['step'], state['seconds']
        completed = state['completed']
    source_features = None
    if attack == 'htba' and not completed:
        if len(source_images) < len(clean):
            raise ValueError('Need at least one distinct source training image per poison')
        source_features = features(model, stamp(source_images[:len(clean)], patch,
                                               spec['patch_margin']), spec['batch_size'])
    limit = spec['steps'] if step_limit is None else min(spec['steps'], step_limit)
    clock = time.perf_counter()
    for step in range(start, limit):
        if completed:
            break
        tick = time.perf_counter()
        if attack == 'htba':
            lr = spec['lr'] * spec['decay']**(step//spec['decay_every'])
            delta, loss = htba_step(model, clean, delta, source_features, spec['batch_size'], lr)
            project(delta, clean, cfg['epsilon'])
            completed = loss < spec['early_stop_loss'] or step+1 == spec['steps']
        else:
            loss = 0.
            for start_batch in range(0, len(clean), spec['batch_size']):
                sl = slice(start_batch, start_batch+spec['batch_size'])
                part = delta[sl].detach().requires_grad_(True)
                objective = F.cross_entropy(model(clean[sl]+part), labels[sl], reduction='sum')
                grad = torch.autograd.grad(objective, part)[0]
                if not bool(torch.isfinite(grad).all()):
                    raise RuntimeError('Non-finite CLBA gradient')
                delta[sl] += spec['step_size'] * grad.detach().sign()
                loss += float(objective.detach())
            project(delta, clean, cfg['epsilon'])
            completed = step+1 == spec['steps']
        history.append({'step': step+1, 'loss': loss, 'step_seconds': time.perf_counter()-tick})
        seconds = elapsed+time.perf_counter()-clock
        report_progress(cfg, directory, 'craft', f'{attack.upper()} poison crafting', step+1,
                        spec['steps'], seconds, announce=step == 0 or completed, loss=loss)
        if (step+1) % cfg['checkpoint_every'] == 0 or completed or step+1 == limit:
            atomic_save({'attack': attack, 'settings': spec, 'delta': delta.detach(),
                         'step': step+1, 'history': history, 'seconds': seconds,
                         'completed': completed}, path)
    project(delta, clean, cfg['epsilon'])
    before_stamp = delta.detach().clone()
    if attack == 'clba':
        final = clipped_trigger(clean, clean+before_stamp, patch, cfg['epsilon'], spec['patch_margin'])
        delta = final-clean
    return delta.detach(), before_stamp, history
