"""Run measured timing, nine smoke cells, and tests before enabling a long run."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import configure, load_config, write_json, atomic_write_text, seed_all
from src.table6 import ARCHITECTURES, ATTACKS, LEGACY, cell_config, run_cell, reuse_check

OUTPUT = ROOT/'results/table6'
REPORT = ROOT/'reports/table6'


def implementation_digest():
    paths = sorted([*list((ROOT/'src').glob('*.py')), *list((ROOT/'scripts').glob('*table6*.py')),
                    *list((ROOT/'configs').glob('table6_*.yaml')), *list((ROOT/'tests').glob('*.py'))])
    h = hashlib.sha256()
    for path in paths:
        h.update(path.relative_to(ROOT).as_posix().encode())
        h.update(path.read_bytes())
    h.update((ROOT/'pytest.ini').read_bytes())
    return h.hexdigest()


def configs():
    return {arch: load_config(ROOT/f'configs/table6_{arch}.yaml') for arch in ARCHITECTURES}


def smoke_config(base):
    cfg = copy.deepcopy(base)
    cfg.update(name='smoke', output=str(OUTPUT/'smoke'), classes=[0, 1],
               train_per_class=128, test_per_class=8, epochs_surrogate=3,
               epochs_retrain=1, epochs_victim=1, batch_size=8, craft_batch_size=2,
               poison_count=4, source_count=8, craft_steps=6, checkpoint_every=1,
               milestones=[1], telemetry_count=2, baseline_min_accuracy=0.)
    return cfg


def smoke():
    from src.data import load_data
    from src.models import make_model
    from src.craft import alignment_step, flat_grad
    import torch
    data = None
    records = []
    for arch, base in configs().items():
        cfg = smoke_config(base)
        if data is None:
            data = load_data(cfg)
        # Full architecture double backward, even when a cell is already cached.
        model = make_model(cfg).eval()
        a = flat_grad(model, data[0][:2], data[1][:2]).detach()
        cosine, gradient = alignment_step(model, data[0][-2:], torch.zeros_like(data[0][-2:]),
                                          data[1][-2:], a, 2, 177)
        assert bool(torch.isfinite(gradient).all()) and float(gradient.abs().max()) > 0
        del model
        for attack in ATTACKS:
            current = copy.deepcopy(cfg)
            if attack in ('htba', 'clba'):
                current['baseline_settings'] = {'steps': 3, 'batch_size': 2}
            result = run_cell(current, attack, data=data)
            assert result['integrity']['changed_images'] == 4
            assert result['victim_history'][-1]['epoch'] == 1
            records.append({'architecture': arch, 'attack': attack, 'passed': True,
                            'directory': result['directory']})
            write_json(records, OUTPUT/'smoke/checks.json')
        torch.cuda.empty_cache()
    return records


def timings():
    import torch
    from src.baselines import craft_baseline
    from src.craft import alignment_step, average_gradient
    from src.data import load_data
    from src.models import make_model
    from src.patch import make_patch
    from src.selection import select_poisons
    from src.victim import train
    bases = configs()
    data = load_data(bases['resnet18'])
    x, y, tx, ty = data
    patch = make_patch(8, 18, x.device)
    epochs, crafting = {}, {}
    for arch, base in bases.items():
        cfg = cell_config(base, 'full')
        location = OUTPUT/'timing'/arch
        model, history, seconds = train(cfg, x, y, tx, ty, 1, 1017, location/'surrogate', 0, 1, patch)
        epochs[arch] = {'seconds_per_epoch': seconds, 'batch_size': 128,
                        'train_images': len(x), 'test_images': len(tx), 'epochs': 1}
        write_json(epochs, OUTPUT/'timing/epochs.json')
        indices = torch.where(y == 1)[0][:500]
        sources = x[torch.where(y == 0)[0][:512]]
        crafting[arch] = {}
        for attack, cap in (('htba', 8), ('clba', 20)):
            probe = cell_config(base, attack)
            if attack == 'htba':
                probe['baseline_settings']['early_stop_loss'] = -1.
            delta, _, steps = craft_baseline(model, x, y, indices, sources, patch, probe,
                                             location/attack, step_limit=cap)
            values = [s['step_seconds'] for s in steps[1:]]
            crafting[arch][attack+'_step_seconds'] = statistics.mean(values)
            del delta
        targets = torch.ones(len(sources), dtype=torch.long, device=x.device)
        torch.cuda.synchronize()
        tick = time.perf_counter()
        a = average_gradient(model, sources, targets, cfg['craft_batch_size'], patch, 'random', 417)
        torch.cuda.synchronize()
        crafting[arch]['adversarial_gradient_seconds'] = time.perf_counter()-tick
        selection_file = location/'selection_timing.json'
        if selection_file.exists():
            selection = json.loads(selection_file.read_text())
        else:
            _, selection = select_poisons(model, x, y, 1, 500, 'gradient', 20)
            write_json(selection, selection_file)
        crafting[arch]['selection_seconds'] = selection['seconds']
        values = []
        for step in range(3):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            _, gradient = alignment_step(model, x[indices], torch.zeros_like(x[indices]),
                                         y[indices], a, cfg['craft_batch_size'], 517+step)
            torch.cuda.synchronize()
            values.append(time.perf_counter()-tick)
            del gradient
        crafting[arch]['full_step_seconds'] = statistics.mean(values[1:])
        crafting[arch]['note'] = 'Full 500-image workload; timing surrogate trained for one epoch only. HTBA timing disables early stop to measure maximum schedule.'
        write_json(crafting, OUTPUT/'timing/crafting.json')
        print('MEASURED', arch, epochs[arch], crafting[arch], flush=True)
        del model, a
        torch.cuda.empty_cache()
    return epochs, crafting


def estimate(epochs, crafting, reuse):
    # All refresh/victim schedules stay full. Max HTBA iterations; no early-stop discount.
    cells = {}
    shared = {}
    for arch in ARCHITECTURES:
        ep = epochs[arch]['seconds_per_epoch']
        shared[arch] = 0. if arch == 'resnet18' and reuse else 120*ep
        for attack in ATTACKS:
            extra = (160*ep + 250*crafting[arch]['full_step_seconds'] +
                     crafting[arch]['selection_seconds'] + 5*crafting[arch]['adversarial_gradient_seconds']
                     if attack == 'full' else (5000 if attack == 'htba' else 20)*crafting[arch][attack+'_step_seconds'])
            cells[attack+'/'+arch] = 0. if attack == 'full' and arch == 'resnet18' and reuse else 80*ep+extra
        # Shared training is done once, before the architecture's first HTBA cell.
        cells['htba/'+arch] += shared[arch]
    raw = sum(cells.values())
    return {'cells_seconds': cells, 'shared_seconds': shared, 'raw_hours': raw/3600,
            'margin_fraction': .15, 'estimated_hours': raw*1.15/3600,
            'assumptions': ['Includes 15% runtime margin; full 5,000 HTBA iterations even if loss threshold stops earlier.',
                            'One-epoch timing is an estimate, not a guarantee; laptop throttling can change speed.',
                            'Clean 80-epoch baseline and initial 40-epoch surrogate shared once per architecture.',
                            'Full-gradient selection and five source-gradient evaluations use measured timings.',
                            'Existing ResNet checkpoints are copied into the new namespace when compatible.']}


def run_preflight(timing_only=False):
    configure()
    if timing_only:
        return timings()
    started = time.perf_counter()
    checked_digest = implementation_digest()
    result = {'passed': False, 'problems': []}
    try:
        command = [sys.executable, '-m', 'pytest', '-q', '-o', 'cache_dir=results/table6/pytest_cache',
                   '--basetemp=results/table6/pytest_tmp']
        tested = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        atomic_write_text(tested.stdout+'\n'+tested.stderr, REPORT/'tests.txt')
        result['tests_passed'] = tested.returncode == 0
        if not result['tests_passed']:
            raise RuntimeError('Unit tests failed; see reports/table6/tests.txt')
        result['smoke_cells'] = len(smoke())
        epochs, crafting = timings()
        compatible, differences = reuse_check(configs()['resnet18'])
        result['reuse'] = {'reused': compatible, 'differences': differences,
                           'source_path': str(LEGACY),
                           'reason': 'Every saved config key matches except output; new attack key only routes the new runner.'
                           if compatible else 'Configuration or required files differ.'}
        result['timings'] = {'epochs': epochs, 'crafting': crafting}
        result['estimate'] = estimate(epochs, crafting, compatible)
        if result['estimate']['estimated_hours'] > 22:
            result['problems'].append(f"Measured estimate including margin is {result['estimate']['estimated_hours']:.2f} h, above the 22 h launch gate.")
        result['passed'] = not result['problems']
    except BaseException as error:
        result['problems'].append(f'{type(error).__name__}: {error}')
    result['implementation_digest'] = implementation_digest()
    if result['implementation_digest'] != checked_digest:
        result['passed'] = False
        result['problems'].append('Implementation changed during preflight; rerun checks before launch.')
    result['preflight_seconds'] = time.perf_counter()-started
    write_json(result, OUTPUT/'preflight.json')
    lines = ['# Table 6 preflight', '', f"Launch allowed: {result['passed']}", '',
             'Implemented CIFAR MobileNetV2/VGG11, checkpointed HTBA/CLBA, isolated runner, report, and dashboard.',
             'All old shared-directory fingerprint rules are unchanged.',
             'Baseline differences and source citations: docs/table6_baselines.md.', '',
             'Protocol differences: Turner uses the requested clean surrogate rather than the reference adversarially trained checkpoint; HTBA uses a corrected greedy bijection; all cells use the same seeded random patch and random test-time positions. CLBA stamping is finally projected, so the training trigger is clipped.', '',
             f"Unit tests passed: {result.get('tests_passed', False)}", f"Smoke cells passed: {result.get('smoke_cells', 0)}/9", '']
    for arch, value in result.get('timings', {}).get('epochs', {}).items():
        lines.append(f"- {arch}: {value['seconds_per_epoch']:.2f} seconds/epoch, batch 128, 50,000 train + 10,000 test.")
    if 'estimate' in result:
        lines.extend(['', f"Estimated new work: {result['estimate']['raw_hours']:.2f} h; with 15% margin: {result['estimate']['estimated_hours']:.2f} h.",
                      *result['estimate']['assumptions']])
    lines.extend(['', 'ResNet Sleeper reuse: '+json.dumps(result.get('reuse')), '', 'Problems:'])
    lines.extend(result['problems'] or ['None'])
    atomic_write_text('\n'.join(lines)+'\n', REPORT/'preflight.md')
    if not result['passed']:
        atomic_write_text('# Table 6 progress\n\nStatus: STOPPED\n\n## Problems\n\n'+
                          '\n'.join(result['problems'])+'\n\nThe full run was not launched.\n', ROOT/'progress.md')
    print(json.dumps({'passed': result['passed'], 'problems': result['problems'],
                      'estimated_hours': result.get('estimate', {}).get('estimated_hours')}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timing-only', action='store_true')
    args = parser.parse_args()
    outcome = run_preflight(args.timing_only)
    if not args.timing_only and not outcome['passed']:
        raise SystemExit(1)
