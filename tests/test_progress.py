import datetime
import json
import os
from pathlib import Path
import yaml
import pytest
import scripts.progress as progress
from src.common import write_json, atomic_write_text
from src.progress import report_progress


def setup_run(tmp_path, monkeypatch, state='running'):
    cfg = yaml.safe_load(Path('configs/cifar10_full.yaml').read_text())
    monkeypatch.setattr(progress, 'ROOT', tmp_path)
    now = datetime.datetime(2026, 11, 12, 12, 0, tzinfo=progress.HAWAII)
    write_json({'state': state, 'stage': 5, 'error': 'example failure',
                'updated_utc': (now-datetime.timedelta(seconds=300)).isoformat()},
               tmp_path/'results/run_status.json')
    (tmp_path/'results/runner.pid').write_text(str(os.getpid()))
    pair = progress.paths(cfg)[0]
    path = pair['shared']/'clean_victim/history.json'
    write_json([{'epoch': 10, 'clean_accuracy': .9}], path)
    os.utime(path, (now.timestamp(), now.timestamp()))
    return cfg, now


def test_live_training_progress_uses_saved_rounds_and_measured_speed(tmp_path, monkeypatch):
    cfg, now = setup_run(tmp_path, monkeypatch)
    s = progress.snapshot(cfg, now)
    assert s['state'] == 'running'
    assert s['training_epochs'] == {'done': 10, 'total': 8000}
    assert s['craft_updates'] == {'done': 0, 'total': 10000}
    assert s['trials'] == {'done': 0, 'total': 40}
    assert s['seconds_per_epoch'] == pytest.approx(30)
    assert s['milestones'][0]['done'] == 10
    assert datetime.datetime.fromisoformat(s['milestones'][0]['estimated_finish_hawaii'][0]) == now+datetime.timedelta(minutes=35)
    assert len(s['milestones']) == 14
    assert '10/80 (12.5%)' in progress.render(s)
    assert len(s['experiments']) == 40
    assert s['experiments'][0]['arm'] == 'full'
    assert s['experiments'][1]['arm'] == 'fixed'
    assert len({(e['source'], e['target'], e['seed'], e['arm']) for e in s['experiments']}) == 40
    assert sum(len(e['milestones']) for e in s['experiments']) == 408
    assert len(s['final_milestones']) == 6
    assert not any(m['done'] for m in s['final_milestones'])
    text = progress.render(s)
    assert '### Experiment 40/40:' in text and 'Final analysis and presentation milestones' in text


def test_failure_stops_time_predictions_and_reports_error(tmp_path, monkeypatch):
    cfg, now = setup_run(tmp_path, monkeypatch, state='failed')
    s = progress.snapshot(cfg, now)
    assert s['estimated_finish_hawaii'] is None
    text = progress.render(s)
    assert 'example failure' in text and 'Waiting for runner' in text


def test_phase_record_and_heartbeat_are_written_consistently(tmp_path):
    cfg = {'output': str(tmp_path/'results'), 'source': 0, 'target': 1, 'seed': 17}
    directory = tmp_path/'trial'
    report_progress(cfg, directory, 'craft', 'Crafting', 50, 250, 120, cosine=.7)
    event = json.loads((directory/'craft_progress.json').read_text())
    assert event == json.loads((tmp_path/'results/activity.json').read_text())
    assert event['percent'] == 20 and event['metrics']['cosine'] == .7
    assert event['updated_hawaii'].endswith('-10:00')


def test_four_hour_dashboard_shows_persistent_deadline_and_countdown(tmp_path, monkeypatch):
    cfg, now = setup_run(tmp_path, monkeypatch)
    scope = {'pair_indices': [0], 'arms': ['full', 'fixed', 'random', 'select'], 'hours': 4}
    write_json({'state': 'running', 'stage': 7, 'mode': 'four_hour', 'scope': scope,
                'budget_output': 'results/four_hour'}, tmp_path/'results/run_status.json')
    write_json({'started_utc': (now-datetime.timedelta(hours=1)).isoformat(),
                'deadline_utc': (now+datetime.timedelta(hours=3)).isoformat()},
               tmp_path/'results/four_hour/budget_clock.json')
    s = progress.snapshot(cfg, now)
    assert s['budget']['seconds_remaining'] == 3*3600
    assert '3h 00m 00s' in progress.render(s)
    assert 'Hard deadline:' in progress.render(s) and s['trials']['total'] == 4


def test_atomic_write_recovers_from_a_temporary_lock(tmp_path, monkeypatch):
    import src.common as common
    path = tmp_path/'activity.json'
    write_json({'done': 1}, path)
    real_replace = common.os.replace
    attempts = []
    def locked_replace(source, destination):
        attempts.append(source)
        if len(attempts) <= 2:
            raise PermissionError('simulated Windows sharing lock')
        real_replace(source, destination)
    monkeypatch.setattr(common.os, 'replace', locked_replace)
    monkeypatch.setattr(common.time, 'sleep', lambda _: None)
    write_json({'done': 2}, path)
    assert json.loads(path.read_text()) == {'done': 2}
    assert len(attempts) == 3 and not list(tmp_path.glob('*.tmp'))


def test_progress_lock_never_stops_training_or_other_progress_file(tmp_path, monkeypatch, capsys):
    import src.progress as reporter
    real_write = reporter.write_json
    def locked_write(value, path):
        if path.name == 'train_progress.json':
            raise PermissionError('file left open')
        real_write(value, path)
    monkeypatch.setattr(reporter, 'write_json', locked_write)
    cfg = {'output': str(tmp_path/'results')}
    report_progress(cfg, tmp_path/'trial', 'train', 'Training', 1, 40)
    assert json.loads((tmp_path/'results/activity.json').read_text())['done'] == 1
    assert 'deferred' in capsys.readouterr().out


def test_dashboard_recovers_on_next_refresh_without_losing_last_good_view(tmp_path, monkeypatch):
    cfg, now = setup_run(tmp_path, monkeypatch)
    s = progress.snapshot(cfg, now)
    view = tmp_path/'results/PROGRESS.md'
    view.write_text('last good view')
    real_write = progress.atomic_write_text
    attempts = []
    def locked_view(value, path):
        attempts.append(path)
        if len(attempts) == 1:
            raise PermissionError('editor read lock')
        real_write(value, path)
    monkeypatch.setattr(progress, 'atomic_write_text', locked_view)
    assert progress.publish_snapshot(s) is False
    assert view.read_text() == 'last good view'
    assert progress.publish_snapshot(s) is True
    assert '10/80 (12.5%)' in view.read_text()


def test_permanently_locked_json_keeps_previous_record_and_cleans_temp(tmp_path, monkeypatch):
    import src.common as common
    path = tmp_path/'activity.json'
    write_json({'done': 1}, path)
    def deny_replace(*args):
        raise PermissionError('persistent sharing lock')
    monkeypatch.setattr(common.os, 'replace', deny_replace)
    monkeypatch.setattr(common.time, 'sleep', lambda _: None)
    with pytest.raises(PermissionError):
        write_json({'done': 2}, path)
    assert json.loads(path.read_text()) == {'done': 1}
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.skipif(os.name != 'nt', reason='Windows file sharing regression')
def test_actual_windows_read_handle_can_remain_open_during_retry(tmp_path, monkeypatch):
    import ctypes
    from ctypes import wintypes
    import threading
    import src.common as common
    path = tmp_path/'PROGRESS.md'
    atomic_write_text('old progress', path)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    # A reader permits reads/writes, but withholds FILE_SHARE_DELETE; atomic
    # replacement must wait until the reader releases this handle.
    handle = kernel.CreateFileW(str(path), 0x80000000, 1 | 2, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value
    real_replace = common.os.replace
    failures = []
    def observe_replace(source, destination):
        try:
            real_replace(source, destination)
        except PermissionError:
            failures.append(True)
            raise
    monkeypatch.setattr(common.os, 'replace', observe_replace)
    release = threading.Timer(.15, lambda: kernel.CloseHandle(handle))
    release.start()
    try:
        atomic_write_text('new progress', path)
    finally:
        release.join()
    assert failures and path.read_text() == 'new progress'
