import argparse
import copy
from pathlib import Path
from .common import configure, load_config, append_jsonl, read_jsonl
from .data import load_data
from .run_trial import run_trial

ARMS = {
    'htba': {'attack': 'htba', 'selection': 'random', 'retraining_factor': 0},
    'clba': {'attack': 'clba', 'selection': 'random', 'retraining_factor': 0},
    'fixed': {'patch_location': 'fixed', 'selection': 'random', 'retraining_factor': 0},
    'random': {'patch_location': 'random', 'selection': 'random', 'retraining_factor': 0},
    'retrain': {'patch_location': 'random', 'selection': 'random', 'retraining_factor': 4},
    'select': {'patch_location': 'random', 'selection': 'gradient', 'retraining_factor': 0},
    'full': {'patch_location': 'random', 'selection': 'gradient', 'retraining_factor': 4},
}


def main():
    parser = argparse.ArgumentParser(description='Paired, resumable CIFAR-10 Table 10 ablation')
    parser.add_argument('--config', default='configs/cifar10_full.yaml')
    parser.add_argument('--arms', nargs='+', choices=list(ARMS),
                        default=['fixed', 'random', 'retrain', 'select', 'full'])
    parser.add_argument('--pairs', type=int, help='Use the first N predefined paired trials')
    args = parser.parse_args()
    configure()
    base = load_config(args.config)
    assert len(base['pairs']) == len(base['seeds'])
    data = load_data(base)
    for pair, seed in list(zip(base['pairs'], base['seeds']))[:args.pairs]:
        for arm in args.arms:
            cfg = copy.deepcopy(base)
            cfg.update(ARMS[arm])
            result = run_trial(cfg, *pair, seed, arm, data)
            # Recover a completed result if interrupted between result.json and JSONL.
            path = Path(cfg['output'])/'results.jsonl'
            if result['trial_id'] not in {r['trial_id'] for r in read_jsonl(path)}:
                append_jsonl(result, path)


if __name__ == '__main__':
    main()
