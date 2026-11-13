from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .common import atomic_save, write_json, rng_state, restore_rng, seed_all
from .data import augment
from .models import make_model
from .patch import apply_patch
from .craft import average_gradient
from .progress import report_progress


@torch.no_grad()
def evaluate(model, x, y, source, target, patch, batch_size=128, seed=0):
    model.eval()
    predictions = torch.cat([model(b).argmax(1) for b in x.split(batch_size)])
    mask = y == source
    assert bool(mask.any()), 'No held-out source examples'
    generator = torch.Generator(device=x.device).manual_seed(seed)
    patched = torch.cat([model(apply_patch(b, patch, 'random', generator)).argmax(1)
                         for b in x[mask].split(batch_size)])
    correct = predictions[mask] == source
    return {'clean_accuracy': float((predictions == y).float().mean()),
            'source_accuracy': float(correct.float().mean()),
            'asr': float((patched == target).float().mean()),
            'asr_clean_correct': float((patched[correct] == target).float().mean()) if bool(correct.any()) else None,
            'source_test_count': int(mask.sum()), 'source_clean_correct_count': int(correct.sum()),
            'asr_definition': 'all held-out source images; uniformly random patch location'}


def alignment_telemetry(model, x, y, indices, source_x, target, patch, cfg):
    count = cfg['telemetry_count']
    if not len(indices):
        return None
    idx = indices[:count]
    sources = source_x[:count]
    labels = torch.full((len(sources),), target, dtype=torch.long, device=x.device)
    a = average_gradient(model, sources, labels, cfg['craft_batch_size'], patch, 'random', cfg['seed']+600)
    b = average_gradient(model, x[idx], y[idx], cfg['craft_batch_size'])
    return float(F.cosine_similarity(a[None], b[None]).item())


def train(cfg, x, y, test_x, test_y, epochs, seed, directory, source, target, patch,
          telemetry_indices=None, source_x=None):
    directory = Path(directory)
    checkpoint = directory / 'train.pt'
    seed_all(seed)
    model = make_model(cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=.9,
                                nesterov=True, weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg['milestones'], gamma=.1)
    amp = cfg['amp'] and x.device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    history, start, elapsed = [], 0, 0.0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=x.device, weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        history, start, elapsed = state['history'], state['epoch'], state['seconds']
        restore_rng(state['rng'])
    clock = time.perf_counter()
    label = {'clean_victim': 'Normal-model baseline', 'surrogate': 'Practice-model training',
             'victim': 'Victim training and attack evaluation'}.get(directory.name)
    if label is None:
        label = ('Noise-only control' if directory.name.startswith('random_control')
                 else 'Practice-model refresh '+directory.name.removeprefix('retrain_'))
    report_progress(cfg, directory, 'train', label, start, epochs, elapsed, announce=True)
    for epoch in range(start, epochs):
        model.train()
        order = torch.randperm(len(x), device=x.device)
        total_loss = 0.0
        for indices in order.split(cfg['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(augment(x[indices])), y[indices])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(indices)
        scheduler.step()
        metrics = evaluate(model, test_x, test_y, source, target, patch, cfg['batch_size'], cfg['seed']+700)
        metrics.update(epoch=epoch+1, train_loss=total_loss/len(x))
        if telemetry_indices is not None:
            metrics['alignment_cosine'] = alignment_telemetry(
                model, x, y, telemetry_indices, source_x, target, patch, cfg)
        history.append(metrics)
        print(f'{directory.name} epoch {epoch+1}/{epochs}: acc={metrics["clean_accuracy"]:.3f} ASR={metrics["asr"]:.3f}', flush=True)
        atomic_save({'epoch': epoch+1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                     'history': history, 'seconds': elapsed+time.perf_counter()-clock,
                     'rng': rng_state()}, checkpoint)
        try:
            write_json(history, directory/'history.json')
        except OSError as error:
            print(f'History display deferred (full history is in train.pt): {error}', flush=True)
        report_progress(cfg, directory, 'train', label, epoch+1, epochs,
                        elapsed+time.perf_counter()-clock, announce=epoch+1 == epochs,
                        accuracy=metrics['clean_accuracy'], asr=metrics['asr'])
    model.eval()
    return model, history, elapsed+time.perf_counter()-clock
