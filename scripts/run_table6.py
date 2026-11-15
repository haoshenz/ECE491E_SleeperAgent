"""Run the gated, resumable nine-cell benchmark in an isolated output namespace."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import configure, atomic_write_text, write_json
from src.table6 import (ARCHITECTURES, cell_config, cell_directory, table_shared,
                        import_legacy, run_cell)
from src.table6_progress import Dashboard, read_json
from scripts.make_table6 import build, collect
from scripts.table6_preflight import configs, implementation_digest

OUTPUT = ROOT/'results/table6'
REPORT = ROOT/'reports/table6'
ORDER = [(attack, arch) for attack in ('htba', 'clba') for arch in ARCHITECTURES]
ORDER += [('full', 'vgg11'), ('full', 'mobilenetv2'), ('full', 'resnet18')]


def process_alive(pid):
    if os.name == 'nt':
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        code = ctypes.c_ulong()
        try:
            kernel.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
            return code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def take_lock():
    path = OUTPUT/'runner.lock'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            pid = int(path.read_text())
        except ValueError:
            raise RuntimeError('Unreadable runner lock; inspect results/table6/runner.lock')
        if process_alive(pid):
            raise RuntimeError(f'Table 6 runner PID {pid} is already active')
        path.unlink()
    fd = os.open(path, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd, 'w') as f:
        f.write(str(os.getpid()))
    return path


def stages_for(cfg, estimates, crafting):
    """Measured time weights; progress counters are read without loading models."""
    arch, attack = cfg['architecture'], cfg['attack']
    ep = estimates['epochs'][arch]['seconds_per_epoch']
    shared, directory = table_shared(cfg), cell_directory(cfg)
    stages = []
    def add(path, total, seconds, **extra):
        record = read_json(path, {})
        # Already-completed shared models cost no remaining work in this cell.
        if record.get('done', 0) >= total:
            seconds = 0.
        stages.append(dict(path=str(path), total=total, seconds=seconds, **extra))
    for phase, epochs in (('clean_victim', 80), ('surrogate', 40)):
        if arch == 'resnet18' and estimates['resnet_shared_reused']:
            continue
        add(shared/phase/'train_progress.json', epochs, epochs*ep)
    if attack == 'full':
        add(directory/'selection_progress.json', 5000, crafting[arch]['selection_seconds'])
        add(directory/'craft_progress.json', 250, 250*crafting[arch]['full_step_seconds'] +
            5*crafting[arch]['adversarial_gradient_seconds'])
        for step in (50, 100, 150, 200):
            add(directory/f'retrain_{step}'/'train_progress.json', 40, 40*ep)
    else:
        steps = 5000 if attack == 'htba' else 20
        add(directory/'craft_progress.json', steps, steps*crafting[arch][attack+'_step_seconds'],
            baseline_checkpoint=True)
    add(directory/'victim'/'train_progress.json', 80, 80*ep)
    return stages


def verify_gate():
    preflight = read_json(OUTPUT/'preflight.json', {})
    if not (preflight.get('passed') and preflight.get('tests_passed') and
            preflight.get('smoke_cells') == 9 and
            preflight.get('estimate', {}).get('estimated_hours', float('inf')) <= 22):
        raise RuntimeError('Preflight gate failed or is missing. Run scripts/table6_preflight.py first.')
    if preflight.get('implementation_digest') != implementation_digest():
        raise RuntimeError('Implementation/config/tests changed since preflight; rerun preflight.')
    return preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-file', help='Append stdout and stderr to one log (for pythonw background launch).')
    args = parser.parse_args()
    if args.log_file:
        log = Path(args.log_file)
        log.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = open(log, 'a', encoding='utf-8', buffering=1)
    os.chdir(ROOT)
    lock = None
    dashboard = None
    try:
        preflight = verify_gate()
        lock = take_lock()
        write_json({'pid': os.getpid()}, OUTPUT/'pid.json')
        print(f'TABLE6 START PID={os.getpid()}', flush=True)
        configure()
        bases = configs()
        reuse = import_legacy(bases['resnet18'])
        completed = {a+'/'+arch: record for (a, arch), record in collect(OUTPUT).items()}
        dashboard = Dashboard(ROOT, OUTPUT, preflight['estimate']['cells_seconds'], completed)
        dashboard.start()
        build(OUTPUT, REPORT)
        from src.data import load_data
        data = load_data(bases['resnet18'])
        timing = {'epochs': preflight['timings']['epochs'], 'resnet_shared_reused': reuse['reused']}
        stopped = False
        for attack, arch in ORDER:
            key = attack+'/'+arch
            if key in dashboard.completed:
                continue
            if (OUTPUT/'STOP').exists() or dashboard.seconds_used >= 22*3600:
                reason = ('Safe stop requested; last cell is checkpointed.' if (OUTPUT/'STOP').exists()
                          else '22-hour budget threshold reached; finished the previous cell and stopped.')
                dashboard.close('STOPPED', reason)
                (OUTPUT/'STOP').unlink(missing_ok=True)
                stopped = True
                break
            cfg = cell_config(bases[arch], attack)
            stages = stages_for(cfg, timing, preflight['timings']['crafting'])
            dashboard.set_current(key, cell_directory(cfg), stages)
            print(f'CELL START {key}', flush=True)
            record = run_cell(bases[arch], attack, data=data)
            dashboard.finished(key, record)
            build(OUTPUT, REPORT)
            print(f"CELL DONE {key} ASR={record['metrics']['asr']:.6f}", flush=True)
        if not stopped:
            build(OUTPUT, REPORT)
            dashboard.close('DONE')
        print(f'TABLE6 {dashboard.status}', flush=True)
    except BaseException as error:
        message = f'{type(error).__name__}: {error}'
        traceback.print_exc()
        if dashboard:
            dashboard.close('STOPPED' if isinstance(error, KeyboardInterrupt) else 'CRASHED', message)
        else:
            existing_lock = OUTPUT/'runner.lock'
            active_other = False
            if existing_lock.exists() and lock is None:
                try:
                    active_other = process_alive(int(existing_lock.read_text()))
                except ValueError:
                    active_other = True
            if not active_other:
                atomic_write_text('# Table 6 progress\n\nStatus: CRASHED\n\n## Problems\n\n'+message+'\n',
                                  ROOT/'progress.md')
        raise
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
