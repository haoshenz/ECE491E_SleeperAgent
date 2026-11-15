"""Build only measured Table 6 cells; never substitute reference-paper numbers."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import write_json, atomic_write_text
from src.table6 import ARCHITECTURES, ATTACKS, LABELS

CAPTION = 'one pair (airplane->automobile), one seed; not a statistical reproduction'


def collect(output):
    cells = {}
    for path in sorted((Path(output)/'cells').glob('*/result.json')):
        r = json.loads(path.read_text(encoding='utf-8'))
        key = (r['attack'], r['architecture'])
        if key in cells:
            raise ValueError(f'Multiple results for {key}; refusing to choose silently')
        if (r['source'], r['target'], r['seed']) != (0, 1, 17):
            raise ValueError(f'Unexpected pair/seed: {path}')
        cells[key] = r
    return cells


def build(output=ROOT/'results/table6', report=ROOT/'reports/table6'):
    cells = collect(output)
    rows = []
    for attack in ATTACKS:
        values = {arch: (cells[(attack, arch)]['metrics'] if (attack, arch) in cells else None)
                  for arch in ARCHITECTURES}
        complete = all(values.values())
        rows.append({'attack': attack, 'label': LABELS[attack], 'cells': values,
                     'mean_asr': sum(v['asr'] for v in values.values())/3 if complete else None,
                     'mean_clean_accuracy': sum(v['clean_accuracy'] for v in values.values())/3
                     if complete else None})
    summary = {'caption': CAPTION, 'source': 0, 'target': 1, 'seed': 17,
               'completed_cells': len(cells), 'expected_cells': 9, 'rows': rows,
               'provenance': [{'attack': a, 'architecture': arch, 'directory': r['directory'],
                               'reused': r.get('reused', False), 'reused_from': r.get('reused_from'),
                               'protocol': r.get('protocol')}
                              for (a, arch), r in cells.items()],
               'notes': ['ASR uses all 1,000 source-class test images with uniformly random patch locations.',
                         'Turner uses the requested clean surrogate; the reference benchmark supplies an adversarially trained surrogate.',
                         'HTBA closest-feature assignment enforces a bijection; see docs/table6_baselines.md.',
                         'Turner training trigger is clipped; both pre-trigger and final perturbations are bounded.',
                         'Row average is across architectures, not independent pair/seed repetitions.']}
    lines = ['# Table 6 comparison', '', CAPTION, '']
    for metric, title, mean_key in (('asr', 'Attack success rate', 'mean_asr'),
                                     ('clean_accuracy', 'Clean test accuracy', 'mean_clean_accuracy')):
        lines.extend([f'## {title}', '', '| Attack | ResNet-18 | MobileNetV2 | VGG11 | Average |',
                      '|---|---:|---:|---:|---:|'])
        for row in rows:
            entries = [f"{100*row['cells'][arch][metric]:.2f}%" if row['cells'][arch] else 'not run'
                       for arch in ARCHITECTURES]
            average = f"{100*row[mean_key]:.2f}%" if row[mean_key] is not None else 'not run'
            lines.append('| '+ ' | '.join([row['label'], *entries, average])+' |')
        lines.append('')
    lines.extend(['## Protocol notes', '', *['- '+note for note in summary['notes']], '',
                  'Exact settings and source citations: [baseline documentation](../../docs/table6_baselines.md).', ''])
    write_json(summary, Path(report)/'summary.json')
    atomic_write_text('\n'.join(lines), Path(report)/'table6.md')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT/'results/table6'))
    parser.add_argument('--report', default=str(ROOT/'reports/table6'))
    args = parser.parse_args()
    summary = build(args.output, args.report)
    print(f"Reported {summary['completed_cells']}/9 measured cells")
