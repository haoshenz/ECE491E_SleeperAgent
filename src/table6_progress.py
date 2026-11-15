"""Independent 15-second dashboard thread; never holds a training-file handle."""
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from zoneinfo import ZoneInfo
from .common import atomic_write_text, write_json
from .table6 import ARCHITECTURES, ATTACKS, LABELS


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


class Dashboard:
    def __init__(self, root, output, estimates, completed=None, interval=15):
        self.root, self.output = Path(root), Path(output)
        self.estimates = estimates
        self.completed = dict(completed or {})
        saved = read_json(self.output/'run_state.json', {})
        self.previous = saved.get('seconds_used', 0.)
        self.clock = time.perf_counter()
        self.status, self.problems, self.current = 'RUNNING', [], None
        self.interval = interval
        self.lock = threading.RLock()
        self.event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name='table6-dashboard', daemon=True)

    def start(self):
        self.write()
        self.thread.start()

    @property
    def seconds_used(self):
        return self.previous + time.perf_counter()-self.clock

    def set_current(self, key, directory, stages):
        with self.lock:
            self.current = {'key': key, 'directory': str(directory), 'stages': stages}
            self.write()

    def finished(self, key, record):
        with self.lock:
            self.completed[key] = record
            self.current = None
            self.write()

    def remaining(self):
        remaining = sum(value for key, value in self.estimates.items() if key not in self.completed)
        total, left = 0., 0.
        if self.current:
            for stage in self.current['stages']:
                record = read_json(stage['path'], {})
                fraction = min(1., record.get('done', 0)/max(1, stage['total']))
                # HTBA may finish early at the reference loss threshold.
                if stage.get('baseline_checkpoint'):
                    result = Path(self.current['directory'])/'poisons.pt'
                    if result.exists():
                        fraction = 1.
                total += stage['seconds']
                left += stage['seconds']*(1-fraction)
            remaining -= self.estimates.get(self.current['key'], 0.)
            remaining += left
        return max(0., remaining), total, left

    def write(self):
        with self.lock:
            now = datetime.now(ZoneInfo('Pacific/Honolulu'))
            remaining, total, left = self.remaining()
            lines = ['# Table 6 progress', '', f'Last updated: {now:%Y-%m-%d %H:%M:%S} HST (updates every {self.interval:g} s)',
                     f'Status: {self.status}', '', '## Right now', '']
            if self.current:
                attack, arch = self.current['key'].split('/')
                activity = read_json(self.output/'activity.json', {})
                label = activity.get('label', 'preparing this cell')
                done, count = activity.get('done'), activity.get('total')
                step = f'{label}, {done} of {count}' if count else label
                percent = 100*(1-left/total) if total else 0.
                lines.extend([f'Cell {len(self.completed)+1} of 9: {LABELS[attack]} on {arch}',
                              f'Step: {step}', f'This cell: {percent:.0f}% done, about {left/60:.0f} min left'])
            else:
                lines.append('All scheduled work finished.' if self.status == 'DONE' else 'No cell is currently running.')
            lines.extend(['', '## Overall', '', f'Done: {len(self.completed)} of 9 cells',
                          f'GPU time used: {self.seconds_used/3600:.2f} h of 24 h budget',
                          f'Estimated time left: about {remaining/3600:.1f} h',
                          'Time budget conservatively counts active runner wall time, including CPU work and checkpoint writes.',
                          '', '## Table so far (attack success rate)', '',
                          '| Attack | ResNet-18 | MobileNetV2 | VGG11 |', '|---|---:|---:|---:|'])
            for attack in ATTACKS:
                values = []
                for arch in ARCHITECTURES:
                    key = attack+'/'+arch
                    result = self.completed.get(key)
                    if result:
                        value = f"{100*result['metrics']['asr']:.2f}%"
                        if result.get('reused'):
                            value += ' (reused)'
                    elif self.current and key == self.current['key']:
                        value = 'running'
                    else:
                        value = 'waiting' if self.status == 'RUNNING' else 'not run'
                    values.append(value)
                lines.append('| '+' | '.join([LABELS[attack], *values])+' |')
            lines.extend(['', '## Problems', '', '\n'.join(self.problems) if self.problems else 'None', ''])
            atomic_write_text('\n'.join(lines), self.root/'progress.md')
            write_json({'status': self.status, 'seconds_used': self.seconds_used,
                        'completed': list(self.completed), 'current': self.current,
                        'updated_hawaii': now.isoformat(), 'problems': self.problems},
                       self.output/'run_state.json')

    def _loop(self):
        while not self.event.wait(self.interval):
            try:
                self.write()
            except OSError as error:
                print(f'Dashboard write deferred: {error}', flush=True)

    def close(self, status, problem=None):
        self.event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        with self.lock:
            self.status = status
            if problem:
                self.problems.append(problem)
            self.write()
