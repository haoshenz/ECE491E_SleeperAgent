import argparse
import copy
from pathlib import Path
import time
import platform
import torch
from .common import (configure, load_config, fingerprint, seed_all, atomic_save,
                     write_json, append_jsonl)
from .data import load_data, project, assert_integrity
from .models import make_model
from .patch import make_patch
from .selection import select_poisons
from .craft import craft
from .victim import train, evaluate
from .progress import report_progress


def image_grid(clean, poisoned, patch, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = min(6, len(clean))
    fig, axes = plt.subplots(3, n, figsize=(n*1.6, 5), squeeze=False)
    for j in range(n):
        for row, image in enumerate([clean[j], poisoned[j], (poisoned[j]-clean[j]) * 4 + .5]):
            axes[row, j].imshow(image.detach().cpu().permute(1, 2, 0).clamp(0, 1))
            axes[row, j].set_xticks([])
            axes[row, j].set_yticks([])
    for ax, label in zip(axes[:, 0], ['Clean', 'Poisoned', 'Difference x4 + 0.5']):
        ax.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


ARM_ONLY_KEYS = ('selection', 'patch_location', 'retraining_factor',
                 'craft_steps', 'craft_lr', 'craft_lr_units', 'output')


def shared_directory(cfg):
    """Checkpoints (surrogate, clean victim, noise controls) shared by every arm of one pair/seed."""
    cache_cfg = {k: v for k, v in cfg.items() if k not in ARM_ONLY_KEYS}
    return Path(cfg['output'])/'shared'/fingerprint(cache_cfg)


def run_trial(cfg, source, target, seed, arm='custom', data=None, reference_shared=None):
    assert source != target and source in cfg['classes'] and target in cfg['classes']
    cfg = copy.deepcopy(cfg)
    cfg.update(source=source, target=target, seed=seed)
    trial_id = f'{source}-{target}_s{seed}_{arm}_{fingerprint(cfg)}'
    directory = Path(cfg['output']) / trial_id
    result_path = directory/'result.json'
    if result_path.exists():
        import json
        return json.loads(result_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    write_json(cfg, directory/'config.json')
    start = time.perf_counter()
    x, y, test_x, test_y = load_data(cfg) if data is None else data
    original_y = y.clone()
    assert cfg['poison_count'] <= .01*len(x)+1e-6 or cfg['name'] == 'smoke', 'Full-run poison budget >1%'
    patch = make_patch(cfg['patch_size'], seed+1, x.device)
    source_indices = torch.where(y == source)[0]
    order = torch.randperm(len(source_indices), generator=torch.Generator().manual_seed(seed+2))
    source_x = x[source_indices[order[:cfg['source_count']].to(x.device)]]
    # Only training data enters source gradient and selection; test data is evaluation-only.
    shared = shared_directory(cfg)
    if reference_shared is not None:
        shared = Path(reference_shared)
        for phase in ('surrogate', 'clean_victim'):
            if not (shared/phase/'train.pt').exists():
                raise RuntimeError(f'Required reusable checkpoint missing: {shared/phase}')
    surrogate, _, surrogate_seconds = train(cfg, x, y, test_x, test_y,
        cfg['epochs_surrogate'], seed+1000, shared/'surrogate', source, target, patch)
    selection_file = directory/'selection.pt'
    if selection_file.exists():
        saved = torch.load(selection_file, map_location=x.device, weights_only=False)
        indices, selection = saved['indices'], saved['metadata']
    else:
        def selection_progress(done, total, seconds):
            report_progress(cfg, directory, 'selection', 'Choosing training pictures to change',
                            done, total, seconds, announce=done in (0, total))
        indices, selection = select_poisons(surrogate, x, y, target, cfg['poison_count'],
                                           cfg['selection'], seed+3, selection_progress)
        atomic_save({'indices': indices, 'metadata': selection}, selection_file)
    clean_metrics, random_metrics = None, None
    clean_history = []
    if cfg['controls']:
        clean_model, clean_history, _ = train(cfg, x, y, test_x, test_y,
            cfg['epochs_victim'], seed+2000, shared/'clean_victim', source, target, patch,
            torch.where(y == target)[0][:cfg['telemetry_count']], source_x)
        clean_metrics = evaluate(clean_model, test_x, test_y, source, target, patch, cfg['batch_size'], seed+700)
        if reference_shared is not None:
            clean_history = []  # Historical telemetry belongs to the original pair.
        del clean_model
        if clean_metrics['clean_accuracy'] < cfg['baseline_min_accuracy']:
            raise RuntimeError(f'Clean accuracy {clean_metrics["clean_accuracy"]:.4f} below baseline gate '
                               f'{cfg["baseline_min_accuracy"]:.4f}. Fix training before attacking.')
    def retrain(delta, step):
        poisoned = x.clone()
        poisoned[indices] += delta
        model, _, _ = train(cfg, poisoned, y, test_x, test_y, cfg['epochs_retrain'],
            seed+3000+step, directory/f'retrain_{step}', source, target, patch)
        return model
    delta, alignment = craft(surrogate, x, y, indices, source_x, target, patch, cfg, directory, retrain)
    del surrogate
    poisoned = x.clone()
    poisoned[indices] += delta
    integrity = assert_integrity(x, poisoned, y, original_y, indices, target, cfg['epsilon'], patch)
    atomic_save({'indices': indices.cpu(), 'delta': delta.cpu(), 'patch': patch.cpu(), 'config': cfg}, directory/'poisons.pt')
    image_grid(x[indices], poisoned[indices], patch, directory/'poison_grid.png')
    model, victim_history, victim_seconds = train(cfg, poisoned, y, test_x, test_y,
        cfg['epochs_victim'], seed+2000, directory/'victim', source, target, patch, indices, source_x)
    metrics = evaluate(model, test_x, test_y, source, target, patch, cfg['batch_size'], seed+700)
    del model, poisoned
    if cfg['controls']:
        seed_all(seed+800)
        noise = torch.empty_like(delta).uniform_(-cfg['epsilon'], cfg['epsilon'])
        project(noise, x[indices], cfg['epsilon'])
        noisy = x.clone()
        noisy[indices] += noise
        assert_integrity(x, noisy, y, original_y, indices, target, cfg['epsilon'], patch)
        # Arms sharing the exact selected indices have the same random-noise
        # control. Reuse its checkpoint instead of spending another 80 epochs.
        control_key = fingerprint({'indices': indices.tolist(), 'epsilon': cfg['epsilon']})
        write_json({'directory': str(shared/f'random_control_{control_key}')},
                   directory/'random_control_ref.json')
        model, _, _ = train(cfg, noisy, y, test_x, test_y, cfg['epochs_victim'],
            seed+2000, shared/f'random_control_{control_key}', source, target, patch)
        random_metrics = evaluate(model, test_x, test_y, source, target, patch, cfg['batch_size'], seed+700)
        del model, noisy
    result = {'trial_id': trial_id, 'arm': arm, 'source': source, 'target': target, 'seed': seed,
              'config': cfg, 'metrics': metrics, 'clean_control': clean_metrics,
              'random_control': random_metrics, 'integrity': integrity, 'selection': selection,
              'alignment': alignment, 'victim_history': victim_history, 'clean_history': clean_history,
              'surrogate_seconds': surrogate_seconds, 'victim_seconds': victim_seconds,
              'invocation_seconds': time.perf_counter()-start, 'directory': str(directory),
              'environment': {'torch': torch.__version__, 'python': platform.python_version(),
                              'gpu': torch.cuda.get_device_name() if x.is_cuda else 'CPU'}}
    if reference_shared is not None:
        result['reference_shared'] = str(reference_shared)
    write_json(result, result_path)
    append_jsonl(result, Path(cfg['output'])/'results.jsonl')
    print(f'COMPLETED {trial_id}: {metrics}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/smoke.yaml')
    parser.add_argument('--source', type=int, default=0)
    parser.add_argument('--target', type=int, default=1)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    configure()
    run_trial(load_config(args.config), args.source, args.target, args.seed)


if __name__ == '__main__':
    main()
