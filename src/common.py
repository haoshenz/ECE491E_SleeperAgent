from pathlib import Path
import hashlib
import json
import random
import os
import tempfile
import time
import numpy as np
import torch
import yaml


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True


def load_config(path):
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if cfg['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Use --device cpu for development.')
    assert cfg['craft_steps'] > cfg['retraining_factor'] >= 0
    assert 0 < cfg['epsilon'] <= 16 / 255 + 1e-12
    return cfg


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def _replace_with_retry(source, destination):
    """Allow short Windows reader/antivirus locks to clear before failing."""
    for attempt in range(9):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 8:
                raise
            time.sleep(min(.01 * 2**attempt, .2))


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    temp = Path(name)
    try:
        writer(temp)
        _replace_with_retry(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_save(value, path):
    _atomic_write(path, lambda temp: torch.save(value, temp))


def atomic_write_text(value, path):
    _atomic_write(path, lambda temp: temp.write_text(value, encoding='utf-8'))


def write_json(value, path):
    atomic_write_text(json.dumps(value, indent=2, allow_nan=False), path)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # a process can die halfway through its last append
    return rows


def append_jsonl(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Repair a torn final line before appending another record.
    if path.exists():
        contents = path.read_bytes()
        if contents and not contents.endswith(b'\n'):
            with path.open('wb') as f:
                f.write(contents[:contents.rfind(b'\n') + 1])
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
