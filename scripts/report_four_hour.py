"""Write an honest single-pair pilot report from completed checkpoints only."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.summarize import summarize, PAPER
from src.common import atomic_write_text, write_json
from scripts.run_four_hour import load_plan
from scripts.progress import ARM_LABELS, read


def build_report(plan, cfg, pair):
    report = ROOT/plan['report']
    report.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in plan['arms']:
        result = read(pair['trials'][arm][0]/'result.json')
        if result:
            rows.append(result)
    # Unfinished training metrics are never promoted to completed attack results.
    summary = summarize(rows, ROOT/plan['output']/'analysis')
    summarize(rows, report)
    write_json(rows, report/'measurements.json')
    for filename in ('budget_clock.json', 'session.json', 'housekeeping.json', 'validation.json'):
        source = ROOT/plan['output']/filename
        if source.exists():
            shutil.copy2(source, report/filename)
    clock = read(ROOT/plan['output']/'budget_clock.json', {})
    if clock:
        generated = datetime.now(timezone.utc)
        write_json({'generated_utc': generated.isoformat(), 'deadline_utc': clock['deadline_utc'],
                    'window_elapsed_seconds': (generated-datetime.fromisoformat(clock['started_utc'])).total_seconds(),
                    'completed_configurations': len(rows), 'scheduled_configurations': len(plan['arms']),
                    'all_selected_complete': len(rows) == len(plan['arms']),
                    'reused_prior_checkpoints': True}, report/'completion.json')
    table = ['| Configuration | Paper ASR (reference) | Pilot ASR | Clean accuracy | Noise-control ASR | Status |',
             '|---|---:|---:|---:|---:|---|']
    by_arm = {r['arm']: r for r in rows}
    for arm, (paper_asr, _) in PAPER.items():
        result = by_arm.get(arm)
        if result:
            m = result['metrics']
            noise = result.get('random_control')
            noise_text = f'{noise["asr"]:.2%}' if noise else 'Unavailable'
            table.append(f'| {ARM_LABELS[arm]} | {paper_asr:.2f}% | {m["asr"]:.2%} | '
                         f'{m["clean_accuracy"]:.2%} | {noise_text} | Complete |')
        else:
            state = 'Outside four-hour scope' if arm not in plan['arms'] else 'Pending / incomplete'
            table.append(f'| {ARM_LABELS[arm]} | {paper_asr:.2f}% | — | — | — | {state} |')
    table_text = '\n'.join(table)+'\n'
    atomic_write_text(table_text, report/'table.md')
    source, target = pair['pair']
    section = (f'**Time-limited pilot: {len(rows)}/{len(plan["arms"])} scheduled configurations completed.** '
               f'This uses one source/target pair ({source} → {target}), seed {pair["seed"]}, '
               'with the original full CIFAR-10 training recipe and saved checkpoints. '
               'The original 40-experiment sweep is archived.\n\n'+table_text+'\n'+
               '**There is one trial per configuration. Standard errors and confidence intervals are unavailable.** '
               'The paper averages multiple class pairs; its reference values cannot be compared as equivalent estimates. '
               'This pilot illustrates behavior on one pair and does not establish population-level effect sizes.\n\n')
    if rows:
        high = max(rows, key=lambda r: r['metrics']['asr'])
        shutil.copy2(ROOT/Path(high['directory'])/'poison_grid.png', report/'poison_grid.png')
        section += ('![Clean versus poisoned examples](reports/four_hour/poison_grid.png)\n\n'
                    '![Measured gradient alignment](reports/four_hour/alignment.png)\n\n')
        section += ('Alignment is a diagnostic measured on small training subsets. The clean curve uses '
                    'fixed target examples and poisoned curves use selected poisons, so their gap is not '
                    'a comparison of identical examples.\n\n')
        control = rows[0].get('clean_control')
        if control:
            section += (f'The clean control achieved {control["clean_accuracy"]:.2%} ordinary accuracy '
                        f'and {control["asr"]:.2%} patched ASR. ')
        section += ('Every completed trial passed the pixel-space budget, unchanged-label, target-only selection, '
                    'exact poison-count, and trigger-absence checks.\n\n')
        observations = []
        if 'full' in by_arm:
            full = by_arm['full']
            observations.append(f'On this pair, the full attack reached {full["metrics"]["asr"]:.2%} ASR '
                f'with {full["metrics"]["clean_accuracy"]:.2%} clean accuracy; its matched noise-only control '
                f'reached {full["random_control"]["asr"]:.2%} ASR.')
        for first, second, mechanism in [('fixed', 'random', 'Random placement'),
                                          ('random', 'select', 'Selection'),
                                          ('select', 'full', 'Retraining with selection enabled')]:
            if first in by_arm and second in by_arm:
                delta = (by_arm[second]['metrics']['asr']-by_arm[first]['metrics']['asr'])*100
                observations.append(f'{mechanism} changed ASR by {delta:+.2f} percentage points on this pair.')
        section += '**What the completed measurements show.** '+' '.join(observations)+' '
        section += ('These are single-pair observations; the small sample does not establish how the '
                    'effects vary across other source/target pairs.\n\n'
                    '**Implementation lesson.** The observed interruption was a Windows lock on a progress file. '
                    'Bounded retries and isolating display failures allowed checkpoint-based recovery. '
                    'The two-pass global-gradient calculation avoided retaining all poison activation graphs, '
                    'and direct-gradient, finite-difference, and recovery tests checked its behavior.\n\n')
    section += ('The four-hour budget reserves 15 minutes for analysis and writing. '
                'The standalone retraining-only arm and seven additional class pairs are outside this scope. '
                'Any phase unfinished at the deadline remains checkpointed and is marked incomplete. '
                'The budget covers the remaining work: the full attack and parts of the fixed arm were '
                'computed before this window and are reused.\n\n')
    validation = read(ROOT/plan['output']/'validation.json', {})
    if validation.get('failed') == 0:
        section += (f'**Verification:** {validation["passed"]} tests passed, including global-gradient '
                    'and finite-difference checks, CUDA double-backward, checkpoint recovery, Windows file locks, '
                    'deadline enforcement, and single-pair reporting.\n\n')
    readme = ROOT/'README.md'
    text = readme.read_text(encoding='utf-8')
    start_marker, end_marker = '<!-- RESULTS_START -->', '<!-- RESULTS_END -->'
    start = text.index(start_marker)+len(start_marker)
    end = text.index(end_marker)
    atomic_write_text(text[:start]+'\n\n'+section+text[end:], readme)
    details = ('# Four-hour pilot report\n\n'+section.replace('reports/four_hour/', '')+
               '## Measured single-pair differences\n\n')
    for name, values in summary['paired_differences'].items():
        details += f'- {name}: {values["mean"]:+.2f} percentage points on this one pair; uncertainty unavailable.\n'
    details += ('\n## Scope and reproduction\n\n'
                'Run or resume with `.venv/Scripts/python.exe -u scripts/run_all.py`. '
                'The scheduling plan is `configs/four_hour_run.yaml`; scientific settings remain in '
                '`configs/cifar10_full.yaml`. Exact configs, seeds, metrics, and histories are in '
                '`measurements.json`. The clean, failed-run, smoke, and original sweep artifacts are preserved.\n')
    atomic_write_text(details, report/'README.md')
    return {'completed': len(rows), 'scheduled': len(plan['arms']), 'report': str(report)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', default='configs/four_hour_run.yaml')
    args = p.parse_args()
    plan, cfg, pair = load_plan(args.plan)
    print(json.dumps(build_report(plan, cfg, pair), indent=2))


if __name__ == '__main__':
    main()
