"""Run the four-hour pilot by default; opt into the original sweep explicitly."""
import argparse
import datetime
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import write_json, atomic_write_text


def start_progress_monitor():
    from scripts.progress import process_alive
    pid_path = ROOT/'results/progress.pid'
    try:
        if pid_path.exists():
            old_pid = pid_path.read_text().strip()
            if old_pid.isdigit() and process_alive(int(old_pid)):
                return
        with (ROOT/'results/progress.log').open('a', encoding='utf-8') as out, \
             (ROOT/'results/progress.err.log').open('a', encoding='utf-8') as err:
            monitor = subprocess.Popen([sys.executable, '-u', str(ROOT/'scripts/progress.py'), '--watch'],
                cwd=ROOT, stdout=out, stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        atomic_write_text(str(monitor.pid)+'\n', pid_path)
        print(f'Live dashboard monitor started (PID {monitor.pid})', flush=True)
    except OSError as error:
        print(f'Live dashboard could not start: {error}; training will continue.', flush=True)


def run_original_sweep():
    status = ROOT/'results/run_status.json'
    atomic_write_text(str(os.getpid())+'\n', ROOT/'results/runner.pid')
    commands = [
        ['-m', 'pytest', '-q'],
        ['scripts/download_data.py'],
        ['-m', 'src.run_ablation', '--config', 'configs/smoke.yaml'],
        ['-m', 'analysis.summarize', 'results/smoke/results.jsonl', '--output', 'results/smoke/analysis'],
        ['-m', 'src.run_baseline'],
        ['-m', 'src.run_ablation', '--arms', 'full', '--pairs', '1'],
        ['-m', 'src.run_ablation'],
        ['-m', 'analysis.summarize', 'results/full/results.jsonl', '--output', 'results/full/analysis'],
        ['scripts/update_report.py'],
    ]
    try:
        for index, command in enumerate(commands):
            write_json({'state': 'running', 'stage': index+1, 'stage_count': len(commands),
                        'command': command, 'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}, status)
            if index == 0:
                start_progress_monitor()
            print('\nRUNNING '+ ' '.join(command), flush=True)
            subprocess.run([sys.executable, *command], cwd=ROOT, check=True)
        write_json({'state': 'complete', 'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}, status)
    except Exception as e:
        write_json({'state': 'failed', 'command': command, 'error': str(e),
                    'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}, status)
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-sweep', action='store_true',
                        help='Opt into the original multi-day 40-experiment sweep')
    args = parser.parse_args()
    if args.full_sweep:
        run_original_sweep()
    else:
        from scripts.run_four_hour import main as run_pilot
        run_pilot([])


if __name__ == '__main__':
    main()
