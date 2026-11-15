"""Isolated Table 6 cells. Never train in or update an existing study cache."""
import copy
import json
from pathlib import Path
import shutil
import time
import torch
from .baselines import craft_baseline, clipped_trigger, settings
from .common import atomic_save, fingerprint, seed_all, write_json
from .craft import craft
from .data import assert_integrity
from .patch import make_patch
from .progress import report_progress
from .selection import select_poisons
from .victim import train, evaluate

ARCHITECTURES = ('resnet18', 'mobilenetv2', 'vgg11')
ATTACKS = ('htba', 'clba', 'full')
LABELS = {'htba': 'Saha (HTBA)', 'clba': 'Turner (CLBA)', 'full': 'Sleeper Agent'}
PROTOCOL = 'table6-v1-clean-surrogate-clipped-trigger-greedy-bijection'
LEGACY = Path('results/full/0-1_s17_full_6e499e05ba36/result.json')


def cell_config(base, attack, source=0, target=1, seed=17):
    from .run_ablation import ARMS
    cfg = copy.deepcopy(base)
    cfg.update(ARMS[attack], source=source, target=target, seed=seed, attack=attack)
    if attack in ('htba', 'clba'):
        cfg['baseline_settings'] = settings(cfg, attack)
    return cfg


def cell_directory(cfg):
    return Path(cfg['output'])/'cells'/f"{cfg['architecture']}_{cfg['attack']}_{fingerprint(cfg)}"


def table_shared(cfg):
    from .run_trial import shared_directory
    # Do not alter ARM_ONLY_KEYS or any pre-existing config's fingerprint.
    canonical = {k: v for k, v in cfg.items() if k not in ('attack', 'baseline_settings')}
    return shared_directory(canonical)


def reuse_check(base, path=LEGACY):
    cfg = cell_config(base, 'full', base.get('source', 0), base.get('target', 1), base.get('seed', 17))
    if not Path(path).is_file():
        return False, ['Existing full-attack result is absent']
    record = json.loads(Path(path).read_text())
    old = record['config']
    wanted = {k: v for k, v in cfg.items() if k not in ('output', 'attack')}
    saved = {k: v for k, v in old.items() if k != 'output'}
    differences = [k for k in sorted(set(wanted)|set(saved)) if wanted.get(k) != saved.get(k)]
    for relative in ('poisons.pt', 'victim/train.pt'):
        if not (Path(path).parent/relative).is_file():
            differences.append('missing '+relative)
    from .run_trial import shared_directory
    for phase in ('surrogate', 'clean_victim'):
        if not (shared_directory(old)/phase/'train.pt').is_file():
            differences.append('missing shared '+phase)
    return not differences, differences


def import_legacy(base):
    """Copy result metadata, not old experiment files; old files remain read-only."""
    compatible, differences = reuse_check(base)
    if not compatible:
        return {'reused': False, 'differences': differences}
    cfg = cell_config(base, 'full')
    directory = cell_directory(cfg)
    old = json.loads(LEGACY.read_text())
    record = {k: copy.deepcopy(old[k]) for k in (
        'source', 'target', 'seed', 'metrics', 'integrity', 'clean_control')}
    record.update(config=cfg, attack='full', arm='full', architecture=cfg['architecture'],
                  protocol=PROTOCOL, reused=True, reused_from=str(LEGACY),
                  directory=str(directory), seconds_this_run=0.,
                  reuse_note='Every saved config key matches except output; attack is new routing metadata.')
    if not (directory/'result.json').exists():
        write_json(record, directory/'result.json')
    return {'reused': True, 'path': str(LEGACY), 'differences': []}


def copy_legacy_shared(cfg):
    """Reuse by copying into the new namespace, so train() cannot touch old files."""
    canonical = {k: v for k, v in cfg.items() if k not in ('attack', 'baseline_settings')}
    compatible, _ = reuse_check(canonical)
    if not compatible:
        return
    from .run_trial import shared_directory
    old = json.loads(LEGACY.read_text())['config']
    source = shared_directory(old)
    destination = table_shared(cfg)
    for phase in ('surrogate', 'clean_victim'):
        src, dst = source/phase/'train.pt', destination/phase/'train.pt'
        if src.is_file() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            write_json({'copied_from': str(src), 'reason': 'Identical configuration; protect legacy cache'},
                       dst.parent/'origin.json')


def run_cell(base, attack, source=0, target=1, seed=17, data=None):
    from .data import load_data
    cfg = cell_config(base, attack, source, target, seed)
    if not Path(cfg['output']).resolve().is_relative_to(Path('results/table6').resolve()):
        raise ValueError('New baseline outputs must be under results/table6; use a table6 config')
    directory = cell_directory(cfg)
    path = directory/'result.json'
    if path.exists():
        return json.loads(path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    write_json(cfg, directory/'config.json')
    started = time.perf_counter()
    x, y, test_x, test_y = load_data(cfg) if data is None else data
    original_y = y.clone()
    assert source != target
    assert cfg['name'] == 'smoke' or cfg['poison_count'] == 500 <= .01*len(x)
    patch = make_patch(cfg['patch_size'], seed+1, x.device)
    pool = torch.where(y == source)[0]
    order = torch.randperm(len(pool), generator=torch.Generator().manual_seed(seed+2))
    sources = x[pool[order[:cfg['source_count']].to(x.device)]]
    copy_legacy_shared(cfg)
    shared = table_shared(cfg)
    clean_metrics = None
    if cfg['controls']:
        model, _, _ = train(cfg, x, y, test_x, test_y, cfg['epochs_victim'], seed+2000,
                            shared/'clean_victim', source, target, patch)
        clean_metrics = evaluate(model, test_x, test_y, source, target, patch,
                                 cfg['batch_size'], seed+700)
        # The original 91.5% ResNet gate is not a paper baseline for Mobile/VGG.
        gate = cfg['baseline_min_accuracy'] if cfg['architecture'].startswith('resnet') else .80
        if cfg['name'] == 'smoke':
            gate = 0.
        if clean_metrics['clean_accuracy'] < gate:
            raise RuntimeError(f"Clean {cfg['architecture']} accuracy {clean_metrics['clean_accuracy']:.3%} "
                               f'below {gate:.1%}; refusing to report an unvalidated benchmark')
        write_json(clean_metrics, shared/'clean_metrics.json')
        del model
    model, _, _ = train(cfg, x, y, test_x, test_y, cfg['epochs_surrogate'], seed+1000,
                        shared/'surrogate', source, target, patch)
    selection_path = directory/'selection.pt'
    if selection_path.exists():
        saved = torch.load(selection_path, map_location=x.device, weights_only=False)
        indices, selection = saved['indices'], saved['metadata']
    else:
        def selection_progress(done, total, seconds):
            report_progress(cfg, directory, 'selection', 'Choosing target-class training images',
                            done, total, seconds)
        indices, selection = select_poisons(model, x, y, target, cfg['poison_count'],
                                           cfg['selection'], seed+3, selection_progress)
        atomic_save({'indices': indices, 'metadata': selection}, selection_path)
    if attack == 'full':
        def retrain(delta, step):
            poisoned = x.clone()
            poisoned[indices] += delta
            fresh, _, _ = train(cfg, poisoned, y, test_x, test_y, cfg['epochs_retrain'],
                                seed+3000+step, directory/f'retrain_{step}', source, target, patch)
            return fresh
        delta, history = craft(model, x, y, indices, sources, target, patch, cfg, directory, retrain)
        before_stamp = delta
    else:
        delta, before_stamp, history = craft_baseline(model, x, y, indices, sources,
                                                     patch, cfg, directory)
    del model
    poisoned = x.clone()
    poisoned[indices] += delta
    integrity = assert_integrity(x, poisoned, y, original_y, indices, target, cfg['epsilon'], patch,
                                 allow_trigger=attack == 'clba', arm=attack)
    if attack == 'clba':
        pre = x.clone()
        pre[indices] += before_stamp
        pre_integrity = assert_integrity(x, pre, y, original_y, indices, target, cfg['epsilon'], patch)
        expected = clipped_trigger(x[indices], pre[indices], patch, cfg['epsilon'],
                                   cfg['baseline_settings']['patch_margin'])
        torch.testing.assert_close(poisoned[indices], expected, rtol=0, atol=1e-7)
        assert bool((expected != pre[indices]).flatten(1).any(1).all()), 'Trigger stamping had no effect'
        integrity.update(pre_trigger_max_delta=pre_integrity['max_delta'], trigger_stamped=True,
                         trigger_mode='stamp then project; visible pattern clipped to epsilon')
        del pre
    atomic_save({'indices': indices.cpu(), 'delta': delta.cpu(), 'pre_trigger_delta': before_stamp.cpu(),
                 'patch': patch.cpu(), 'config': cfg, 'integrity': integrity}, directory/'poisons.pt')
    model, victim_history, victim_seconds = train(
        cfg, poisoned, y, test_x, test_y, cfg['epochs_victim'], seed+2000,
        directory/'victim', source, target, patch)
    metrics = evaluate(model, test_x, test_y, source, target, patch, cfg['batch_size'], seed+700)
    result = dict(config=cfg, attack=attack, arm=attack, architecture=cfg['architecture'],
                  source=source, target=target, seed=seed, metrics=metrics, integrity=integrity,
                  selection=selection, crafting_history=history, victim_history=victim_history,
                  victim_seconds=victim_seconds, clean_control=clean_metrics, protocol=PROTOCOL,
                  directory=str(directory), reused=False, seconds_this_run=time.perf_counter()-started)
    write_json(result, path)
    return result
