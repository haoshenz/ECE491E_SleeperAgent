"""Refresh an IDE-readable progress dashboard without loading model checkpoints."""
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import statistics
import sys
import time
from zoneinfo import ZoneInfo
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import fingerprint, write_json, atomic_write_text
from src.run_ablation import ARMS

HAWAII = ZoneInfo('Pacific/Honolulu')
SHARED_EXCLUSIONS = ('selection', 'patch_location', 'retraining_factor', 'craft_steps',
                     'craft_lr', 'craft_lr_units', 'output')
ARM_LABELS = {'fixed': 'Fixed patch location', 'random': 'Random patch location',
              'retrain': 'Random patch + retraining', 'select': 'Random patch + selection',
              'full': 'Full attack: random patch + retraining + selection'}


def read(path, default=None):
    import json
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def training(directory, total):
    history = read(directory/'history.json', [])
    activity = read(directory/'train_progress.json', {})
    done = max(history[-1]['epoch'] if history else 0, activity.get('done', 0))
    return {'done': min(done, total), 'total': total, 'directory': str(directory),
            'seconds': activity.get('seconds', 0), 'metrics': history[-1] if history else {}}


def paths(cfg):
    pairs = []
    output = ROOT/cfg['output']
    for pair, seed in zip(cfg['pairs'], cfg['seeds']):
        base = dict(cfg, source=pair[0], target=pair[1], seed=seed)
        shared = output/'shared'/fingerprint({k: v for k, v in base.items() if k not in SHARED_EXCLUSIONS})
        trials = {}
        for arm, overrides in ARMS.items():
            trial = dict(base, **overrides)
            name = f'{pair[0]}-{pair[1]}_s{seed}_{arm}_{fingerprint(trial)}'
            trials[arm] = (output/name, trial)
        pairs.append({'pair': pair, 'seed': seed, 'shared': shared, 'trials': trials})
    return pairs


def snapshot(cfg, now=None):
    now = now or datetime.now(HAWAII)
    output = ROOT/cfg['output']
    status = read(ROOT/'results/run_status.json', {})
    activity = read(output/'activity.json', {})
    pairs = paths(cfg)
    scope = status.get('scope') if status.get('mode') == 'four_hour' else None
    if scope:
        pairs = [pair for i, pair in enumerate(pairs) if i in scope['pair_indices']]
        for pair in pairs:
            pair['trials'] = {arm: value for arm, value in pair['trials'].items() if arm in scope['arms']}
    models, crafts, results = [], [], []
    pair_rows = []
    for pair in pairs:
        shared = pair['shared']
        models.append(training(shared/'surrogate', cfg['epochs_surrogate']))
        if cfg['controls']:
            models.append(training(shared/'clean_victim', cfg['epochs_victim']))
            for directory in shared.glob('random_control_*'):
                if directory.is_dir():
                    models.append(training(directory, cfg['epochs_victim']))
        completed = 0
        for arm, (directory, trial) in pair['trials'].items():
            models.append(training(directory/'victim', cfg['epochs_victim']))
            for k in range(1, trial['retraining_factor']+1):
                step = cfg['craft_steps']*k//(trial['retraining_factor']+1)
                models.append(training(directory/f'retrain_{step}', cfg['epochs_retrain']))
            result = read(directory/'result.json')
            craft = read(directory/'craft_progress.json', {})
            if result:
                completed += 1
                results.append(result)
            craft['done'] = max(craft.get('done', 0), len(result['alignment']) if result else 0)
            crafts.append(craft)
        pair_rows.append({'source': pair['pair'][0], 'target': pair['pair'][1],
                          'seed': pair['seed'], 'completed': completed, 'total': len(pair['trials'])})
    observed = [m['seconds']/m['done'] for m in models if m['done'] and m['seconds'] > 0]
    # The already-running baseline predates progress records. Its history timestamp
    # and the runner's baseline-start timestamp provide a live measurement.
    baseline = training(pairs[0]['shared']/'clean_victim', cfg['epochs_victim'])
    baseline_history = pairs[0]['shared']/'clean_victim/history.json'
    if status.get('stage') == 5 and 0 < baseline['done'] < baseline['total'] and baseline_history.exists():
        started = datetime.fromisoformat(status['updated_utc']).timestamp()
        elapsed = baseline_history.stat().st_mtime-started
        if elapsed > 0:
            observed.append(elapsed/baseline['done'])
    epoch_rate = statistics.median(observed) if observed else 28.4
    measured_craft = [c['seconds']/c['done'] for c in crafts
                      if c.get('done', 0) >= 10 and c.get('seconds', 0) > 0]
    step_range = ((statistics.median(measured_craft)*.8, statistics.median(measured_craft)*1.5)
                  if measured_craft else (2.4, 7.2))
    train_total = sum(cfg['epochs_surrogate'] + len(pair['trials'])*cfg['epochs_victim']
                  + sum(t['retraining_factor'] for _, t in pair['trials'].values())*cfg['epochs_retrain']
                  + ((1+len({t['selection'] for _, t in pair['trials'].values()}))*cfg['epochs_victim']
                     if cfg['controls'] else 0) for pair in pairs)
    train_done = sum(m['done'] for m in models)
    trial_total = sum(len(pair['trials']) for pair in pairs)
    craft_total = trial_total*cfg['craft_steps']
    craft_done = sum(c.get('done', 0) for c in crafts)
    selected_pair = pairs[0]
    selected_arm = 'full' if 'full' in selected_pair['trials'] else next(iter(selected_pair['trials']))
    if status.get('stage', 5) >= 7:
        for pair in pairs:
            pending = [arm for arm, (directory, _) in pair['trials'].items()
                       if not (directory/'result.json').exists()]
            if pending:
                selected_pair, selected_arm = pair, pending[0]
                break
    activity_path = Path(activity.get('directory', '')).resolve()
    for pair in pairs:
        for arm, (directory, _) in pair['trials'].items():
            if activity_path == directory.resolve() or directory.resolve() in activity_path.parents:
                selected_pair, selected_arm = pair, arm
    milestones = trial_milestones(cfg, selected_pair, selected_arm, epoch_rate, step_range, now)
    if status.get('stage') == 5 and status.get('state') == 'running':
        active = {'label': 'Normal-model baseline', 'done': baseline['done'],
                  'total': baseline['total'], 'metrics': baseline['metrics']}
    else:
        active = activity
    # Check runner liveness so an interrupted process cannot appear to be running.
    state = status.get('state', 'unknown')
    pid_text = (ROOT/'results/runner.pid').read_text().strip() if (ROOT/'results/runner.pid').exists() else ''
    if state == 'running' and pid_text.isdigit() and not process_alive(int(pid_text)):
        state = 'stopped'
    if state == 'paused':
        for milestone in milestones:
            if milestone['state'] == 'running':
                milestone['state'] = 'paused'
    budget = None
    if scope:
        clock = read(ROOT/status.get('budget_output', 'results/four_hour')/'budget_clock.json', {})
        if clock.get('deadline_utc'):
            deadline = datetime.fromisoformat(clock['deadline_utc']).astimezone(HAWAII)
            start = datetime.fromisoformat(clock['started_utc']).astimezone(HAWAII)
            budget = {'started_hawaii': start.isoformat(), 'deadline_hawaii': deadline.isoformat(),
                      'compute_cutoff_hawaii': (deadline-timedelta(minutes=15)).isoformat(),
                      'seconds_remaining': max(0, (deadline-now).total_seconds()),
                      'seconds_elapsed': max(0, (now-start).total_seconds())}
    remaining_train = max(0, train_total-train_done)*epoch_rate
    eta = [(now+timedelta(seconds=remaining_train+max(0, craft_total-craft_done)*r)).isoformat()
           for r in step_range] if state == 'running' else None
    experiments = []
    for pair_index, pair in enumerate(pairs):
        # The runner measures one full trial first, then visits the usual arms.
        order = ['full', *[arm for arm in ARMS if arm != 'full']] if pair_index == 0 else list(ARMS)
        order = [arm for arm in order if arm in pair['trials']]
        for arm in order:
            directory, _ = pair['trials'][arm]
            rows = trial_milestones(cfg, pair, arm, epoch_rate, step_range, now)
            if state == 'paused':
                for milestone in rows:
                    if milestone['state'] == 'running':
                        milestone['state'] = 'paused'
            saved = (directory/'result.json').exists()
            rows.append({'label': 'Save experiment result', 'done': int(saved), 'total': 1,
                         'unit': 'result', 'state': 'done' if saved else 'pending'})
            experiments.append({'number': len(experiments)+1, 'source': pair['pair'][0],
                'target': pair['pair'][1], 'seed': pair['seed'], 'arm': arm,
                'complete': saved, 'milestones_done': sum(m['state'] == 'done' for m in rows),
                'milestones_total': len(rows), 'milestones': rows})
    all_results = len(results) == trial_total
    analysis = ROOT/status.get('budget_output', str(output))/'analysis'
    final_report = ROOT/('reports/four_hour' if scope else 'reports/full')
    final_milestones = [
        {'label': 'Complete and save every selected experiment', 'done': all_results},
        {'label': ('Summarize single-pair differences; mark uncertainty unavailable' if scope
                   else 'Calculate paired differences and confidence intervals'),
         'done': all_results and (analysis/'summary.json').exists()},
        {'label': 'Generate the final ablation table',
         'done': all_results and (analysis/'table.md').exists()},
        {'label': 'Generate the full-sweep alignment plot',
         'done': all_results and (analysis/'alignment.png').exists()},
        {'label': 'Export presentation figures and measured results',
         'done': all_results and (final_report/'poison_grid.png').exists()
                 and (final_report/'measurements.json').exists()},
        {'label': 'Update the README with the completed results', 'done': state == 'complete'},
    ]
    if scope:
        metadata = ROOT/status.get('budget_output', 'results/four_hour')
        final_milestones = [
            {'label': 'Organize project files and archive the original plan',
             'done': read(metadata/'housekeeping.json', {}).get('complete', False)},
            {'label': 'Pass correctness and deadline-recovery checks',
             'done': read(metadata/'validation.json', {}).get('failed') == 0},
        ]+final_milestones
    return {'updated_hawaii': now.isoformat(), 'state': state, 'runner': status, 'active': active,
            'training_epochs': {'done': train_done, 'total': train_total},
            'craft_updates': {'done': craft_done, 'total': craft_total},
            'trials': {'done': len(results), 'total': trial_total},
            'seconds_per_epoch': epoch_rate, 'craft_speed_measured': bool(measured_craft),
            'estimated_finish_hawaii': eta, 'milestones': milestones, 'pairs': pair_rows,
            'experiments': experiments, 'final_milestones': final_milestones, 'scope': scope, 'budget': budget,
            'milestone_context': {'source': selected_pair['pair'][0], 'target': selected_pair['pair'][1],
                                  'seed': selected_pair['seed'], 'arm': selected_arm}}


def trial_milestones(cfg, pair, arm, epoch_rate, step_range, now):
    directory, trial = pair['trials'][arm]
    shared = pair['shared']
    rows = []
    def model(label, path, epochs):
        m = training(path, epochs)
        rows.append(dict(label=label, done=m['done'], total=epochs, unit='rounds',
                         remaining=[(epochs-m['done'])*epoch_rate]*2))
    model('Normal-model baseline', shared/'clean_victim', cfg['epochs_victim'])
    model('Train the practice model', shared/'surrogate', cfg['epochs_surrogate'])
    selection = read(directory/'selection_progress.json', {})
    selected = (directory/'selection.pt').exists()
    selection_total = selection.get('total', 5000)
    done = selection_total if selected else selection.get('done', 0)
    fraction = max(0, 1-done/selection_total)
    rows.append(dict(label='Choose the training pictures to change', done=done, total=selection_total,
                     unit='pictures checked', remaining=[120*fraction, 600*fraction]))
    craft = read(directory/'craft_progress.json', {})
    result = read(directory/'result.json', {})
    craft_done = max(craft.get('done', 0), len(result.get('alignment', [])))
    boundaries = [cfg['craft_steps']*k//(trial['retraining_factor']+1)
                  for k in range(trial['retraining_factor']+2)]
    for block, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
        done = max(0, min(end-start, craft_done-start))
        rows.append(dict(label=f'Crafting block {block} of {len(boundaries)-1}', done=done,
                         total=end-start, unit='updates', remaining=[(end-start-done)*r for r in step_range]))
        if block <= trial['retraining_factor']:
            model(f'Refresh practice model {block} of {trial["retraining_factor"]}',
                  directory/f'retrain_{end}', cfg['epochs_retrain'])
    model('Train victim and test the backdoor', directory/'victim', cfg['epochs_victim'])
    control = read(directory/'random_control_ref.json', {})
    control_path = ROOT/control['directory'] if control else directory/'not_started_control'
    if cfg['controls']:
        model('Run the noise-only comparison', control_path, cfg['epochs_victim'])
    cumulative = [0., 0.]
    for row in rows:
        cumulative = [a+b for a, b in zip(cumulative, row.pop('remaining'))]
        row['state'] = 'done' if row['done'] >= row['total'] else ('running' if row['done'] else 'pending')
        row['estimated_finish_hawaii'] = ([(now+timedelta(seconds=s)).isoformat() for s in cumulative]
                                          if row['state'] != 'done' else None)
    return rows


def process_alive(pid):
    import os
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = wintypes.DWORD()
        try:
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def time_range(values):
    if not values:
        return 'Done'
    a, b = [datetime.fromisoformat(v).strftime('%b %d, %I:%M %p') for v in values]
    return a if a == b else f'{a} to {b}'


def render(s):
    a = s['active']
    done, total = a.get('done', 0), a.get('total', 0)
    percent = done/total*100 if total else 0
    text = ['# Live experiment progress', '',
            f'Updated: {datetime.fromisoformat(s["updated_hawaii"]):%b %d, %I:%M:%S %p} Hawaii time. Refreshes every 10 seconds.', '',
            f'**Runner: {s["state"]}. Current task: {a.get("label", "Starting next stage")}.**', '',
            f'Current task: **{done}/{total} ({percent:.1f}%)**.']
    if s.get('budget'):
        b = s['budget']
        left = int(b['seconds_remaining'])
        deadline = datetime.fromisoformat(b['deadline_hawaii'])
        cutoff = datetime.fromisoformat(b['compute_cutoff_hawaii'])
        text += ['', f'**Hard deadline: {deadline:%I:%M %p} HST on {deadline:%B %d}.** '
                 f'Time left: **{left//3600}h {(left%3600)//60:02d}m {left%60:02d}s**. '
                 f'Compute stops by **{cutoff:%I:%M %p}**; the final 15 minutes are reserved for the report.']
    metrics = a.get('metrics', {})
    accuracy = metrics.get('clean_accuracy', metrics.get('accuracy'))
    if accuracy is not None:
        text += [f'Latest normal-picture accuracy: **{accuracy:.2%}** (accuracy is different from progress).']
    text += ['', '| Overall counter | Progress |', '|---|---:|']
    for label, key in [('Selected experiment results', 'trials'), ('Training passes across all models', 'training_epochs'),
                       ('Poison-crafting updates', 'craft_updates')]:
        v = s[key]
        text.append(f'| {label} | {v["done"]}/{v["total"]} ({100*v["done"]/v["total"]:.1f}%) |')
    finish = time_range(s['estimated_finish_hawaii'])+' HST' if s['estimated_finish_hawaii'] else (
        'Complete' if s['state'] == 'complete' else 'Unavailable while runner is '+s['state'])
    context = s['milestone_context']
    if s.get('scope'):
        text += ['', '**Four-hour pilot:** one source/target pair, four configurations. '
                 'The original multi-day sweep is archived. Existing checkpoints are reused. '
                 + ('The strict four-hour window is active and persists across restarts. ' if s.get('budget')
                    else 'The four-hour window starts on resume. ')+
                 'The retraining-only arm and additional pairs are outside this budget.']
        if s['state'] == 'complete' and not s['runner'].get('all_selected_experiments_complete', False):
            text += ['', '**The budgeted run ended with incomplete experiments. The report includes only completed results.**']
    text += ['', f'**Estimated selected-run finish: {finish}.**', '',
             'This is a rough compute estimate, not a deadline. It includes all training and crafting; '
             'selection, checkpoint overhead, thermal slowdowns, and reruns can add time. '
             + ('Crafting speed is measured.' if s['craft_speed_measured'] else 'Full-scale crafting speed is not measured yet; its estimate is 10–30 minutes per trial.'), '',
             f'Measured training speed: {s["seconds_per_epoch"]:.1f} seconds per round.', '',
             f'## Experiment milestones: {context["source"]} → {context["target"]}, '
             f'{context["arm"]}, seed {context["seed"]}', '',
             '| Milestone | Progress | Estimated finish (Hawaii time) |', '|---|---|---|']
    for m in s['milestones']:
        estimate = time_range(m['estimated_finish_hawaii']) if s['state'] == 'running' or m['state'] == 'done' else 'Waiting for runner'
        text.append(f'| {m["label"]} | {m["state"]}: {m["done"]}/{m["total"]} {m["unit"]} | {estimate} |')
    text += ['', 'Finishing the last comparison produces this experiment’s complete result. '
             'Then the runner completes the remaining paired experiments, generates the statistical table and plots, and updates the README.', '',
             '## Paired sweep', '', '| Source → target | Seed | Finished configurations |', '|---|---:|---:|']
    for p in s['pairs']:
        text.append(f'| {p["source"]} → {p["target"]} | {p["seed"]} | {p["completed"]}/{p["total"]} |')
    text += ['', '## All selected experiment milestones — complete checklist', '',
             'Every experiment is listed below in execution order, including every crafting block '
             'and practice-model refresh. Shared baseline, practice-model, and noise-control work '
             'can satisfy milestones for several experiments. Milestone counts do not measure elapsed time.', '',
             '| Order | Source → target | Configuration | Milestones finished | Result saved |',
             '|---|---|---|---:|---|']
    for e in s['experiments']:
        text.append(f'| {e["number"]}/{len(s["experiments"])} | {e["source"]} → {e["target"]} | {ARM_LABELS[e["arm"]]} | '
                    f'{e["milestones_done"]}/{e["milestones_total"]} | {"Yes" if e["complete"] else "Pending"} |')
    for e in s['experiments']:
        text += ['', f'### Experiment {e["number"]}/{len(s["experiments"])}: '
                 f'{e["source"]} → {e["target"]}, {ARM_LABELS[e["arm"]]}, seed {e["seed"]}', '',
                 '| Milestone | Status | Progress |', '|---|---|---|']
        for m in e['milestones']:
            text.append(f'| {m["label"]} | {m["state"]} | {m["done"]}/{m["total"]} {m["unit"]} |')
    text += ['', '## Final analysis and presentation milestones', '']
    for m in s['final_milestones']:
        text.append(f'- [{"x" if m["done"] else " "}] {m["label"]}')
    if s['state'] == 'failed':
        text += ['', 'Runner error: '+s['runner'].get('error', 'Check run.err.log')]
    text += ['', 'Raw details: `run.log`, `run.err.log`, `progress.json`, and `run_status.json` in this folder.', '']
    return '\n'.join(text)


def publish_snapshot(s):
    """Keep watching if an editor temporarily locks either display file."""
    published = True
    for path, writer in (
        (ROOT/'results/progress.json', lambda p: write_json(s, p)),
        (ROOT/'results/PROGRESS.md', lambda p: atomic_write_text(render(s), p)),
    ):
        try:
            writer(path)
        except OSError as error:
            print(f'Dashboard update deferred for {path}: {error}', flush=True)
            published = False
    return published


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--watch', action='store_true')
    p.add_argument('--config', default='configs/cifar10_full.yaml')
    args = p.parse_args()
    cfg = yaml.safe_load((ROOT/args.config).read_text(encoding='utf-8'))
    while True:
        try:
            s = snapshot(cfg)
            published = publish_snapshot(s)
        except OSError as error:
            print(f'Progress read deferred: {error}', flush=True)
            s, published = {}, False
        if not args.watch or (published and s.get('state') in ('complete', 'failed', 'stopped')):
            break
        time.sleep(10)


if __name__ == '__main__':
    main()
