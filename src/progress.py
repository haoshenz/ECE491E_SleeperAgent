"""Small progress records; no changes to random state or model computation."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from .common import write_json


def report_progress(cfg, directory, phase, label, done, total, seconds=0, announce=False, **metrics):
    if not cfg.get('output'):
        return
    now = datetime.now(ZoneInfo('Pacific/Honolulu'))
    record = {'phase': phase, 'label': label, 'done': done, 'total': total,
              'percent': 100*done/total if total else 0,
              'seconds': seconds, 'directory': str(directory),
              'source': cfg.get('source'), 'target': cfg.get('target'), 'seed': cfg.get('seed'),
              'updated_hawaii': now.isoformat(), 'metrics': metrics}
    for path in (Path(directory)/f'{phase}_progress.json', Path(cfg['output'])/'activity.json'):
        try:
            write_json(record, path)
        except OSError as error:
            # Optional display telemetry must never discard an expensive experiment.
            # Still attempt the other display record if this one is locked.
            print(f'Progress display update deferred for {path}: {error}', flush=True)
    if announce:
        state = 'FINISHED' if done >= total else 'STARTED'
        print(f'MILESTONE {now:%I:%M %p} HST | {state} {label} | {done}/{total}', flush=True)
