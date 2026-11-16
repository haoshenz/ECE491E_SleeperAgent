"""Inference-only fixed/random re-scoring of the nine completed Table 6 victims."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torchvision.datasets import CIFAR10
from src.common import configure, write_json, atomic_write_text
from src.models import make_model
from src.victim import evaluate
from src.table6 import ARCHITECTURES, ATTACKS, LABELS
from scripts.make_table6 import CAPTION, collect

FIXED_POSITIONS = {'htba': (19, 19), 'clba': (24, 24), 'full': (24, 24)}


def rooted(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def trained_directory(record):
    # The ninth result references a pre-existing trained victim; never copy or
    # retrain it just to change its evaluation protocol.
    return rooted(record['reused_from']).parent if record.get('reused') else rooted(record['directory'])


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def snapshot(paths):
    return {str(path): digest(path) for path in sorted(set(map(Path, paths))) if path.is_file()}


def verify_unchanged(before):
    changed = [name for name, sha in before.items() if not Path(name).is_file() or digest(name) != sha]
    if changed:
        raise RuntimeError('Protected input files changed during re-scoring: '+', '.join(changed))


def verify_random(record, measured):
    original = record['metrics']
    key = record['attack']+'/'+record['architecture']
    if measured['source_test_count'] != 1000 or original['source_test_count'] != 1000:
        raise RuntimeError(f'{key}: expected exactly 1,000 source-class test images')
    for metric in ('asr', 'clean_accuracy'):
        if abs(measured[metric]-original[metric]) > 1e-7:
            raise RuntimeError(
                f'{key}: RANDOM {metric} mismatch: saved={original[metric]:.10f}, '
                f'rescored={measured[metric]:.10f}. No re-scoring reports were written. '
                'Check the checkpoint, stored trigger, test ordering, batch size, seed, '
                'normalization, device, and numerical backend before proceeding.')


def load_test_images(cfg):
    if cfg['classes'] != list(range(10)) or cfg['test_per_class'] is not None:
        raise ValueError('Re-scoring requires the complete ten-class CIFAR-10 test set')
    dataset = CIFAR10(rooted(cfg['data_root']), train=False, download=False)
    labels = torch.tensor(dataset.targets)
    # Preserve the original loader's class-grouped ordering, including ordering
    # within a class. RANDOM draws depend on this ordering and batch size.
    order = torch.cat([torch.where(labels == c)[0] for c in cfg['classes']])
    images = torch.from_numpy(dataset.data.copy()).permute(0, 3, 1, 2)
    x = images[order].to(cfg['device'], dtype=torch.float32).div_(255)
    y = labels[order].to(cfg['device'])
    if len(x) != 10000 or int((y == 0).sum()) != 1000:
        raise RuntimeError('Unexpected CIFAR-10 test counts')
    return x, y


def tables(cells):
    by_cell = {(r['attack'], r['architecture']): r for r in cells}
    result = {}
    for protocol in ('native', 'random'):
        rows = []
        for attack in ATTACKS:
            mode = 'fixed' if protocol == 'native' and attack != 'full' else 'random'
            values = {arch: by_cell[attack, arch][mode]['asr'] for arch in ARCHITECTURES}
            rows.append({'attack': attack, 'label': LABELS[attack], 'test_placement': mode,
                         'asr': values, 'mean_asr': sum(values.values())/3})
        result[protocol] = rows
    return result


def render_table(rows):
    lines = ['| Attack | ResNet-18 | MobileNetV2 | VGG11 | Average |', '|---|---:|---:|---:|---:|']
    for row in rows:
        values = [row['asr'][arch] for arch in ARCHITECTURES]+[row['mean_asr']]
        lines.append('| '+' | '.join([row['label'], *[f'{100*v:.2f}%' for v in values]])+' |')
    return '\n'.join(lines)


def render_report(result):
    lines = ['# Table 6 comparison', '', CAPTION, '',
             'Inference-only re-scoring of the same saved victims; no retraining. All nine RANDOM scores and clean accuracies matched their original cell results.', '',
             '## Main table: native test protocol', '',
             'HTBA: fixed inset bottom-right (row 19, column 19). CLBA: fixed bottom-right (24,24). Sleeper Agent: random position. Coordinates are zero-based.', '',
             render_table(result['tables']['native']), '',
             '## Extra table: all attacks, random trigger position', '',
             'All methods use uniformly random positions, seed 717, and the original batch size of 128.', '',
             render_table(result['tables']['random']), '']
    full = {r['architecture']: r for r in result['cells'] if r['attack'] == 'full'}
    fixed = ', '.join(f"{arch}: {100*full[arch]['fixed']['asr']:.2f}%" for arch in ARCHITECTURES)
    lines.extend(['Sleeper Agent with the trigger FIXED at (24,24): '+fixed+'.', '',
                  '## Clean test accuracy', '',
                  'All 10,000 test images, without a trigger. Clean accuracy is identical in both re-scoring passes.', '',
                  '| Attack | ResNet-18 | MobileNetV2 | VGG11 | Average |', '|---|---:|---:|---:|---:|'])
    for attack in ATTACKS:
        by_arch = {r['architecture']: r for r in result['cells'] if r['attack'] == attack}
        values = [by_arch[arch]['random']['clean_accuracy'] for arch in ARCHITECTURES]
        lines.append('| '+' | '.join([LABELS[attack], *[f'{100*v:.2f}%' for v in values],
                                     f'{100*sum(values)/3:.2f}%'])+' |')
    lines.extend(['', '## Protocol notes', '',
                  '- Both protocols use the same 1,000 held-out airplane images; ASR is the fraction classified as automobile.',
                  '- The main table uses the requested method-specific native test placement. HTBA/CLBA fixed coordinates follow the public benchmark. Other documented differences from the paper remain.',
                  '- CLBA used a clean surrogate, not an adversarially trained surrogate.',
                  '- HTBA uses the documented corrected greedy feature bijection; the training trigger for CLBA is clipped into the perturbation bound.',
                  '- The ResNet-18 Sleeper Agent checkpoint is reused from the original study, following the saved provenance.',
                  '- Averages are across architectures, not independent pair/seed repetitions. No uncertainty estimates are inferred.',
                  '- Original result.json files and reports/table6/summary.json are unchanged; the latter retains the original random-position results.',
                  '', f"Re-scoring time: {result['elapsed_seconds']:.2f} seconds. Completed {result['completed_hawaii']}.", '',
                  'Detailed scores and SHA-256 input provenance: [rescored.json](rescored.json).',
                  'Attack settings and remaining benchmark differences: [baseline documentation](../../docs/table6_baselines.md).', ''])
    return '\n'.join(lines)


def rescore(output=ROOT/'results/table6', report=ROOT/'reports/table6', device=None):
    tick = time.perf_counter()
    configure()
    output, report = Path(output), Path(report)
    if (output/'runner.lock').exists():
        raise RuntimeError('A Table 6 runner lock exists; do not re-score while training may be active')
    saved = collect(output)
    expected = {(attack, arch) for attack in ATTACKS for arch in ARCHITECTURES}
    if set(saved) != expected:
        raise RuntimeError('Expected exactly nine completed Table 6 cells')
    protected = list((ROOT/'reports/main_study').rglob('*'))
    protected.extend((output/'cells').glob('*/result.json'))
    protected.extend([report/'summary.json', ROOT/'progress.md', output/'run.log'])
    for record in saved.values():
        directory = trained_directory(record)
        for relative in ('victim/train.pt', 'poisons.pt'):
            path = directory/relative
            if not path.is_file():
                raise FileNotFoundError(f'Missing saved input: {path}')
            protected.append(path)
        if record.get('reused'):
            protected.append(rooted(record['reused_from']))
    before = snapshot(protected)
    cells, data_cache, common_patch = [], {}, None
    for attack in ATTACKS:
        for arch in ARCHITECTURES:
            record = saved[attack, arch]
            cfg = dict(record['config'])
            if device is not None:
                cfg['device'] = device
            if (record['source'], record['target'], record['seed']) != (0, 1, 17):
                raise ValueError('Expected airplane->automobile, seed 17')
            if cfg['batch_size'] != 128 or cfg['patch_size'] != 8 or cfg['architecture'] != arch:
                raise ValueError(f'Unexpected saved evaluation configuration: {attack}/{arch}')
            directory = trained_directory(record)
            checkpoint_path = directory/'victim/train.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if checkpoint['epoch'] != cfg['epochs_victim'] or checkpoint['epoch'] != 80:
                raise ValueError(f'{attack}/{arch}: saved victim is not the completed 80-epoch checkpoint')
            model = make_model(cfg)
            model.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            poison = torch.load(directory/'poisons.pt', map_location='cpu', weights_only=False)
            patch_cpu = poison['patch']
            if patch_cpu.shape != (3, 8, 8):
                raise ValueError('Expected the saved RGB 8x8 trigger')
            if common_patch is None:
                common_patch = patch_cpu.clone()
            if not torch.equal(patch_cpu, common_patch):
                raise ValueError('Saved cells do not share the same trigger')
            patch = patch_cpu.to(cfg['device'])
            del poison
            data_key = (cfg['device'], str(rooted(cfg['data_root'])))
            if data_key not in data_cache:
                data_cache[data_key] = load_test_images(cfg)
            x, y = data_cache[data_key]
            seed = cfg['seed']+700
            random = evaluate(model, x, y, 0, 1, patch, cfg['batch_size'], seed, location='random')
            verify_random(record, random)
            fixed = evaluate(model, x, y, 0, 1, patch, cfg['batch_size'], seed,
                             location=FIXED_POSITIONS[attack])
            if fixed['clean_accuracy'] != random['clean_accuracy']:
                raise RuntimeError('Clean accuracy changed between fixed and random evaluation')
            cells.append({'attack': attack, 'architecture': arch, 'source': 0, 'target': 1,
                          'seed': 17, 'evaluation_seed': seed, 'batch_size': cfg['batch_size'],
                          'checkpoint': str(checkpoint_path), 'result_json': str(rooted(record['directory'])/'result.json'),
                          'reused': record.get('reused', False), 'fixed_position_row_col': list(FIXED_POSITIONS[attack]),
                          'original_random_asr': record['metrics']['asr'],
                          'random_matches_original': True, 'random': random, 'fixed': fixed})
            print(f'{attack}/{arch}: RANDOM={100*random["asr"]:.2f}% (matches), '
                  f'FIXED={100*fixed["asr"]:.2f}%, clean={100*random["clean_accuracy"]:.2f}%', flush=True)
            del model, patch
    # No reports are written until every RANDOM sanity check has passed.
    verify_unchanged(before)
    result = {'caption': CAPTION, 'evaluation_only': True, 'source': 0, 'target': 1, 'seed': 17,
              'evaluation_seed': 717, 'test_images': 10000, 'source_test_images': 1000,
              'all_random_scores_match_original': True, 'protected_input_sha256': before,
              'cells': cells, 'tables': tables(cells), 'elapsed_seconds': time.perf_counter()-tick,
              'completed_hawaii': datetime.now(ZoneInfo('Pacific/Honolulu')).isoformat(),
              'notes': ['No retraining; original cell result.json files are not modified.',
                        'CLBA used a clean, not adversarially trained, surrogate.',
                        'Native means HTBA/CLBA fixed and Sleeper Agent random for this re-scoring.',
                        'Other benchmark differences remain documented in docs/table6_baselines.md.']}
    write_json(result, report/'rescored.json')
    atomic_write_text(render_report(result), report/'table6.md')
    print(f'All 9 sanity checks passed; evaluation-only elapsed time: {result["elapsed_seconds"]:.2f} s', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/table6')
    parser.add_argument('--report', type=Path, default=ROOT/'reports/table6')
    parser.add_argument('--device', choices=('cpu', 'cuda'), help='Default: each checkpoint\'s saved device')
    args = parser.parse_args()
    try:
        rescore(args.output, args.report, args.device)
    except Exception as error:
        print(f'STOPPED: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
