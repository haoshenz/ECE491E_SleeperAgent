"""Train the full clean-model gate without crafting poisons."""
import argparse
import copy
from pathlib import Path
import torch
from .common import configure, load_config, write_json
from .data import load_data
from .patch import make_patch
from .victim import train
from .run_trial import shared_directory


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/cifar10_full.yaml')
    args = p.parse_args()
    configure()
    cfg = copy.deepcopy(load_config(args.config))
    source, target = cfg['pairs'][0]
    seed = cfg['seeds'][0]
    cfg.update(source=source, target=target, seed=seed)
    x, y, tx, ty = load_data(cfg)
    directory = shared_directory(cfg)/'clean_victim'
    patch = make_patch(cfg['patch_size'], seed+1, x.device)
    candidates = torch.where(y == source)[0]
    order = torch.randperm(len(candidates), generator=torch.Generator().manual_seed(seed+2))
    source_x = x[candidates[order[:cfg['source_count']].to(x.device)]]
    _, history, seconds = train(cfg, x, y, tx, ty, cfg['epochs_victim'], seed+2000,
                               directory, source, target, patch,
                               torch.where(y == target)[0][:cfg['telemetry_count']], source_x)
    result = {'metrics': history[-1], 'seconds': seconds, 'baseline_gate': cfg['baseline_min_accuracy']}
    write_json(result, Path(cfg['output'])/'baseline.json')
    if history[-1]['clean_accuracy'] < cfg['baseline_min_accuracy']:
        raise RuntimeError(f"Baseline gate failed; see {Path(cfg['output'])/'baseline.json'}")
    print(f'BASELINE PASSED in {seconds:.0f}s', flush=True)


if __name__ == '__main__':
    main()
