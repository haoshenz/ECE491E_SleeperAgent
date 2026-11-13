"""Checkpoint-preserving pilot with a wall-clock limit and reporting reserve."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import atomic_write_text, write_json
from scripts.progress import paths, read, training, process_alive
from scripts.run_all import start_progress_monitor


def load_plan(path='configs/four_hour_run.yaml'):
    plan = yaml.safe_load((ROOT/path).read_text(encoding='utf-8'))
    cfg = yaml.safe_load((ROOT/plan['base_config']).read_text(encoding='utf-8'))
    if plan['pair_indices'] != [0]:
        raise ValueError('This pilot reuses the first class pair; pair_indices must be [0].')
    if not 0 < plan['report_reserve_minutes']*60 < plan['hours']*3600:
        raise ValueError('Reporting reserve must fit inside the wall-clock budget.')
    pair = paths(cfg)[0]
    if not set(plan['arms']).issubset(pair['trials']):
        raise ValueError('Unknown experiment arm')
    return plan, cfg, pair


def estimate_remaining(plan, cfg, pair, arms=None):
    arms = plan['arms'] if arms is None else arms
    models = {}
    shared = pair['shared']
    models[shared/'surrogate'] = cfg['epochs_surrogate']
    if cfg['controls']:
        models[shared/'clean_victim'] = cfg['epochs_victim']
    known_controls = {}
    for directory, trial in pair['trials'].values():
        ref = read(directory/'random_control_ref.json', {})
        if ref:
            known_controls[trial['selection']] = ROOT/ref['directory']
    rates = []
    for path in (ROOT/cfg['output']).rglob('train_progress.json'):
        p = read(path, {})
        if p.get('done') and p.get('seconds', 0) > 0:
            rates.append(p['seconds']/p['done'])
    rate = statistics.median(rates) if rates else 28.4
    craft_rates = []
    for path in (ROOT/cfg['output']).glob('*/craft_progress.json'):
        p = read(path, {})
        if p.get('done', 0) >= 10 and p.get('seconds', 0) > 0:
            craft_rates.append(p['seconds']/p['done'])
    craft_rate = statistics.median(craft_rates) if craft_rates else 3.0
    craft_steps, selection_seconds, completed = 0, 0, []
    for arm in arms:
        directory, trial = pair['trials'][arm]
        if (directory/'result.json').exists():
            completed.append(arm)
            continue
        models[directory/'victim'] = cfg['epochs_victim']
        for k in range(1, trial['retraining_factor']+1):
            step = cfg['craft_steps']*k//(trial['retraining_factor']+1)
            models[directory/f'retrain_{step}'] = cfg['epochs_retrain']
        if cfg['controls']:
            control = known_controls.get(trial['selection'], shared/f'pending_control_{trial["selection"]}')
            models[control] = cfg['epochs_victim']
        craft_steps += max(0, cfg['craft_steps']-read(directory/'craft_progress.json', {}).get('done', 0))
        if not (directory/'selection.pt').exists():
            selection_seconds += 120 if trial['selection'] == 'gradient' else 2
    epochs = sum(max(0, total-training(path, total)['done']) for path, total in models.items())
    seconds = epochs*rate + craft_steps*craft_rate + selection_seconds
    return {'remaining_training_epochs': epochs, 'remaining_craft_steps': craft_steps,
            'seconds_per_epoch': rate, 'seconds_per_craft_step': craft_rate,
            'estimated_compute_minutes': seconds/60,
            'conservative_compute_minutes': seconds*1.25/60,
            'report_reserve_minutes': plan['report_reserve_minutes'],
            'conservative_total_minutes': seconds*1.25/60+plan['report_reserve_minutes'],
            'already_completed': completed}


def scope_status(plan, state, **details):
    return {'state': state, 'mode': 'four_hour',
            'scope': {'pair_indices': plan['pair_indices'], 'arms': plan['arms'], 'hours': plan['hours']},
            'budget_output': plan['output'], 'stage': 7,
            'updated_utc': datetime.now(timezone.utc).isoformat(), **details}


def stop_compute(child):
    """Stop the Windows Python launcher AND its actual GPU worker."""
    if child.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
    if child.poll() is None:
        child.kill()
    child.wait(timeout=15)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--plan', default='configs/four_hour_run.yaml')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    plan, cfg, pair = load_plan(args.plan)
    estimate = estimate_remaining(plan, cfg, pair)
    print(json.dumps(estimate, indent=2), flush=True)
    if args.dry_run:
        return
    pid_path = ROOT/'results/runner.pid'
    if pid_path.exists():
        value = pid_path.read_text().strip()
        if value.isdigit() and int(value) != os.getpid() and process_alive(int(value)):
            raise RuntimeError('Another experiment runner is active; stop it before resuming the pilot.')
    started = time.monotonic()
    clock_path = ROOT/plan['output']/'budget_clock.json'
    clock = read(clock_path)
    if not clock:
        wall_start = datetime.now(timezone.utc)
        clock = {'started_utc': wall_start.isoformat(),
                 'deadline_utc': (wall_start+timedelta(hours=plan['hours'])).isoformat()}
        write_json(clock, clock_path)
    # Restarting the process must not silently grant another four hours.
    wall_remaining = (datetime.fromisoformat(clock['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
    deadline = started+max(0, wall_remaining)
    compute_deadline = deadline-plan['report_reserve_minutes']*60
    atomic_write_text(str(os.getpid())+'\n', pid_path)
    status_path = ROOT/'results/run_status.json'
    write_json(scope_status(plan, 'running', phase='experiments', estimate=estimate), status_path)
    start_progress_monitor()
    outcome, child = [], None
    try:
        for arm in plan['arms']:
            directory, _ = pair['trials'][arm]
            if (directory/'result.json').exists():
                outcome.append({'arm': arm, 'state': 'reused'})
                continue
            remaining = compute_deadline-time.monotonic()
            required = estimate_remaining(plan, cfg, pair, [arm])['conservative_compute_minutes']*60
            if required > remaining:
                outcome.append({'arm': arm, 'state': 'not_started', 'reason': 'Reserved time for final report'})
                continue
            write_json(scope_status(plan, 'running', phase='experiments', arm=arm,
                                   elapsed_seconds=time.monotonic()-started), status_path)
            print(f'PILOT: resuming {arm}; {remaining/60:.0f} compute minutes left', flush=True)
            child = subprocess.Popen([sys.executable, '-u', '-m', 'src.run_ablation',
                '--config', plan['base_config'], '--arms', arm, '--pairs', '1'], cwd=ROOT,
                stdout=sys.stdout, stderr=sys.stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            try:
                code = child.wait(timeout=max(1, compute_deadline-time.monotonic()))
                outcome.append({'arm': arm, 'state': 'complete' if code == 0 else 'failed', 'exit_code': code})
            except subprocess.TimeoutExpired:
                stop_compute(child)
                outcome.append({'arm': arm, 'state': 'checkpointed_at_budget_limit'})
                break
            finally:
                child = None
        write_json({'outcomes': outcome, 'elapsed_seconds': time.monotonic()-started,
                    'plan': plan}, ROOT/plan['output']/'session.json')
        write_json(scope_status(plan, 'running', phase='report'), status_path)
        subprocess.run([sys.executable, 'scripts/report_four_hour.py', '--plan', args.plan],
                       cwd=ROOT, check=True, timeout=max(60, deadline-time.monotonic()),
                       stdout=sys.stdout, stderr=sys.stderr,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        complete = all((pair['trials'][arm][0]/'result.json').exists() for arm in plan['arms'])
        write_json(scope_status(plan, 'complete', phase='finished', all_selected_experiments_complete=complete,
                               elapsed_seconds=time.monotonic()-started, outcomes=outcome), status_path)
        print(f'PILOT FINISHED in {(time.monotonic()-started)/60:.1f} minutes; all selected results={complete}', flush=True)
    except KeyboardInterrupt:
        write_json(scope_status(plan, 'paused', reason='User interrupted pilot'), status_path)
        raise
    except Exception as error:
        write_json(scope_status(plan, 'failed', error=str(error), outcomes=outcome), status_path)
        raise
    finally:
        if child is not None and child.poll() is None:
            stop_compute(child)


if __name__ == '__main__':
    main()
