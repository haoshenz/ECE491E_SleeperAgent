import json
from pathlib import Path
import subprocess
import sys
import yaml
import pytest
import scripts.progress as progress
import scripts.run_four_hour as runner
import scripts.report_four_hour as report
from src.common import write_json
from tests.test_progress import setup_run


def test_four_hour_scope_preserves_configs_and_reduces_dashboard(tmp_path, monkeypatch):
    cfg, now = setup_run(tmp_path, monkeypatch, state='paused')
    original = progress.paths(cfg)[0]['trials']['full'][0]
    plan = yaml.safe_load(Path('configs/four_hour_run.yaml').read_text())
    write_json(runner.scope_status(plan, 'paused'), tmp_path/'results/run_status.json')
    s = progress.snapshot(cfg, now)
    assert s['state'] == 'paused' and s['estimated_finish_hawaii'] is None
    assert s['trials']['total'] == 4 and s['training_epochs']['total'] == 760
    assert s['craft_updates']['total'] == 1000 and len(s['experiments']) == 4
    assert {e['arm'] for e in s['experiments']} == {'fixed', 'random', 'select', 'full'}
    assert progress.paths(cfg)[0]['trials']['full'][0] == original
    assert 'Four-hour pilot' in progress.render(s)


def test_estimate_reuses_shared_controls_and_all_saved_work(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(progress, 'ROOT', tmp_path)
    cfg = yaml.safe_load(Path('configs/cifar10_full.yaml').read_text())
    plan = yaml.safe_load(Path('configs/four_hour_run.yaml').read_text())
    pair = progress.paths(cfg)[0]
    shared = pair['shared']
    def history(directory, epoch):
        write_json([{'epoch': epoch}], directory/'history.json')
    history(shared/'clean_victim', 80)
    history(shared/'surrogate', 40)
    history(shared/'noise_gradient', 80)
    history(shared/'noise_random', 15)
    for arm, name in [('full', 'noise_gradient'), ('fixed', 'noise_random')]:
        directory = pair['trials'][arm][0]
        write_json({'directory': str(shared/name)}, directory/'random_control_ref.json')
        write_json({'done': 250, 'seconds': 375}, directory/'craft_progress.json')
        (directory/'selection.pt').touch()
    write_json({'complete': True}, pair['trials']['full'][0]/'result.json')
    history(pair['trials']['fixed'][0]/'victim', 80)
    result = runner.estimate_remaining(plan, cfg, pair)
    assert result['remaining_training_epochs'] == 225
    assert result['remaining_craft_steps'] == 500
    assert result['already_completed'] == ['full']
    assert result['conservative_total_minutes'] < 240


def test_budget_timeout_kills_compute_and_still_runs_report(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    plan = dict(base_config='configs/base.yaml', pair_indices=[0], arms=['fixed'],
                hours=4, report_reserve_minutes=15, output='results/four_hour')
    pair = {'trials': {'fixed': (tmp_path/'trial', {})}}
    monkeypatch.setattr(runner, 'load_plan', lambda _: (plan, {}, pair))
    monkeypatch.setattr(runner, 'estimate_remaining', lambda *a: {'conservative_compute_minutes': 1})
    monkeypatch.setattr(runner, 'start_progress_monitor', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['run_four_hour.py'])
    child_state, reports = {}, []
    class Child:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.pid = 123456789
        def wait(self, timeout=None):
            if timeout is not None and not child_state.get('killed'):
                child_state['timeout'] = timeout
                raise subprocess.TimeoutExpired('test compute', timeout)
            return -9
        def kill(self):
            child_state['killed'] = True
        def poll(self):
            return -9 if child_state.get('killed') else None
    monkeypatch.setattr(runner.subprocess, 'Popen', Child)
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: reports.append((a, kw)))
    runner.main()
    assert child_state['killed'] and 0 < child_state['timeout'] <= 225*60
    assert any('scripts/report_four_hour.py' in item[0][0] for item in reports)
    if runner.os.name == 'nt':
        assert any(item[0][0][0] == 'taskkill' and '/T' in item[0][0] for item in reports)
    status = json.loads((tmp_path/'results/run_status.json').read_text())
    assert status['state'] == 'complete' and not status['all_selected_experiments_complete']
    assert status['outcomes'][0]['state'] == 'checkpointed_at_budget_limit'


def test_single_pair_report_does_not_invent_error_bars_or_finish_pending_arms(tmp_path, monkeypatch):
    sample = next(json.loads(line) for line in Path('reports/smoke/measurements.jsonl').read_text().splitlines()
                  if json.loads(line)['arm'] == 'full')
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    plan = yaml.safe_load(Path('configs/four_hour_run.yaml').read_text())
    dirs = {arm: (tmp_path/arm, {}) for arm in plan['arms']}
    directory = dirs['full'][0]
    directory.mkdir()
    sample['directory'] = str(directory)
    source_grid = Path('reports/smoke/poison_grid.png')
    (directory/'poison_grid.png').write_bytes(source_grid.read_bytes())
    write_json(sample, directory/'result.json')
    (tmp_path/'README.md').write_text('Before\n<!-- RESULTS_START -->\nPending\n<!-- RESULTS_END -->\nAfter')
    pair = {'pair': [0, 1], 'seed': 17, 'trials': dirs}
    result = report.build_report(plan, {}, pair)
    assert result['completed'] == 1 and result['scheduled'] == 4
    text = (tmp_path/'README.md').read_text(encoding='utf-8')
    assert '1/4 scheduled configurations' in text and 'confidence intervals are unavailable' in text
    assert 'Outside four-hour scope' in text and 'Pending / incomplete' in text
    summary = json.loads((tmp_path/'reports/four_hour/summary.json').read_text())
    assert summary['arms']['full']['sem'] is None


def test_restarting_budget_runner_does_not_reset_its_deadline(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    plan = dict(base_config='configs/base.yaml', pair_indices=[0], arms=['fixed'],
                hours=4, report_reserve_minutes=15, output='results/four_hour')
    pair = {'trials': {'fixed': (tmp_path/'trial', {})}}
    original_clock = {'started_utc': datetime.now(timezone.utc).isoformat(),
                      'deadline_utc': (datetime.now(timezone.utc)+timedelta(seconds=90)).isoformat()}
    clock_path = tmp_path/'results/four_hour/budget_clock.json'
    write_json(original_clock, clock_path)
    monkeypatch.setattr(runner, 'load_plan', lambda _: (plan, {}, pair))
    monkeypatch.setattr(runner, 'estimate_remaining', lambda *a: {'conservative_compute_minutes': 1})
    monkeypatch.setattr(runner, 'start_progress_monitor', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['run_four_hour.py'])
    def must_not_start_compute(*a, **kw):
        raise AssertionError('Reporting reserve must prevent new compute')
    monkeypatch.setattr(runner.subprocess, 'Popen', must_not_start_compute)
    reports = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: reports.append((a, kw)))
    runner.main()
    assert json.loads(clock_path.read_text()) == original_clock
    assert len(reports) == 1 and reports[0][1]['timeout'] <= 90
    status = json.loads((tmp_path/'results/run_status.json').read_text())
    assert status['outcomes'][0]['state'] == 'not_started'


def test_default_entrypoint_selects_pilot_and_requires_opt_in_for_full_sweep(monkeypatch):
    import scripts.run_all as entry
    calls = []
    monkeypatch.setattr(runner, 'main', lambda argv=None: calls.append(('pilot', argv)))
    monkeypatch.setattr(entry, 'run_original_sweep', lambda: calls.append(('full', None)))
    monkeypatch.setattr(sys, 'argv', ['run_all.py'])
    entry.main()
    monkeypatch.setattr(sys, 'argv', ['run_all.py', '--full-sweep'])
    entry.main()
    assert calls == [('pilot', []), ('full', None)]


def test_pilot_analysis_includes_retraining_effect_conditional_on_selection(tmp_path):
    from analysis.summarize import summarize
    rows = []
    for arm, asr in [('select', .2), ('full', .9)]:
        rows.append({'source': 0, 'target': 1, 'seed': 17, 'arm': arm, 'trial_id': arm,
                     'metrics': {'asr': asr, 'clean_accuracy': .92},
                     'victim_history': [], 'alignment': []})
    summary = summarize(rows, tmp_path)
    comparison = summary['paired_differences']['full minus select']
    assert comparison['mean'] == pytest.approx(70)
    assert comparison['n'] == 1 and comparison['sem'] is None and comparison['ci95'] is None
