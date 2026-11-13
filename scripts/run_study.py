"""Run the main multi-pair study and write its frozen report data.

Every arm in configs/main_study.json runs on every pair, then the measurements,
summary statistics and control values are written to reports/main_study/.
Completed trials are skipped, so re-running the same command resumes.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import configure, load_config, write_json  # noqa: E402
from src.run_ablation import ARMS  # noqa: E402

REPORT = ROOT/'reports/main_study'


def load_plan(path=ROOT/'configs/main_study.json'):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def trial_config(plan, arm):
    cfg = copy.deepcopy(load_config(ROOT/plan['base_config']))
    cfg.update(ARMS[arm], output=plan['output'])
    return cfg


def reference_checkpoints(plan):
    """All pairs reuse the first pair's surrogate and clean model, as in the reported study."""
    from src.run_trial import shared_directory
    first = plan['pairs'][0]
    cfg = trial_config(plan, plan['arms'][0])
    cfg.update(source=first['source'], target=first['target'], seed=first['seed'])
    return shared_directory(cfg)


def run(plan):
    from src.data import load_data
    from src.run_trial import run_trial
    configure()
    data = load_data(trial_config(plan, plan['arms'][0]))
    reference = reference_checkpoints(plan)
    rows = []
    for i, pair in enumerate(plan['pairs']):
        for arm in plan['arms']:
            print(f'=== {pair["source"]} -> {pair["target"]} (seed {pair["seed"]}), arm {arm}', flush=True)
            rows.append(run_trial(trial_config(plan, arm), pair['source'], pair['target'], pair['seed'],
                                  arm, data, reference_shared=reference if i else None))
    return rows


def controls(rows):
    """Patched ASR without poisoning: clean model, random-index noise, selected-index noise."""
    by_pair = {}
    for r in rows:
        by_pair.setdefault((r['source'], r['target'], r['seed']), {})[r['arm']] = r
    out = []
    for (source, target, seed), arms in by_pair.items():
        out.append({'source': source, 'target': target, 'seed': seed,
                    'clean_model': arms['full']['clean_control'],
                    'random_noise': arms['fixed']['random_control'],
                    'selected_noise': arms['full']['random_control']})
    return out


def write_report(rows):
    from analysis.summarize import summarize
    summary = summarize(rows, ROOT/'results/main_study/analysis')
    write_json(rows, REPORT/'measurements.json')
    write_json(summary, REPORT/'summary.json')
    write_json(controls(rows), REPORT/'controls.json')
    print(f'Wrote {len(rows)} trials to {REPORT.relative_to(ROOT)}. '
          'Rebuild figures and README with: python -m analysis.build_readme', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--plan', default=str(ROOT/'configs/main_study.json'))
    args = parser.parse_args()
    write_report(run(load_plan(args.plan)))


if __name__ == '__main__':
    main()
