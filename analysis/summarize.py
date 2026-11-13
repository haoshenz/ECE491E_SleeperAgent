import argparse
from pathlib import Path
import numpy as np
from scipy.stats import t
from src.common import read_jsonl, write_json

PAPER = {'fixed': (19.25, 3.01), 'random': (33.95, 4.57), 'retrain': (59.42, 5.78),
         'select': (63.49, 6.13), 'full': (85.27, 5.90)}


def stats(values):
    a = np.asarray(values, dtype=float)
    n = len(a)
    sem = float(a.std(ddof=1)/np.sqrt(n)) if n > 1 else None
    radius = float(t.ppf(.975, n-1)*sem) if sem is not None else None
    return {'n': n, 'mean': float(a.mean()), 'sem': sem,
            'ci95': [float(a.mean()-radius), float(a.mean()+radius)] if radius is not None else None}


def summarize(rows, output):
    # One observation per arm / class pair / seed. Mixing config variants is an error.
    trials = {}
    for r in rows:
        key = (r['source'], r['target'], r['seed'], r['arm'])
        if key in trials and trials[key]['trial_id'] != r['trial_id']:
            raise ValueError(f'Mixed configurations for {key}; use separate output directories')
        trials[key] = r
    rows = list(trials.values())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {'arms': {}, 'paired_differences': {}, 'asr_definition': 'all held-out source images'}
    lines = ['| Configuration | Paper ASR ± SE | Measured ASR ± SE | n | Clean accuracy |',
             '|---|---:|---:|---:|---:|']
    for arm, paper in PAPER.items():
        matched = [r for r in rows if r['arm'] == arm]
        if not matched:
            lines.append(f'| {arm} | {paper[0]:.2f} ± {paper[1]:.2f}% | pending | 0 | pending |')
            continue
        s = stats([r['metrics']['asr']*100 for r in matched])
        result['arms'][arm] = s
        error = f'{s["sem"]:.2f}' if s['sem'] is not None else 'unavailable'
        acc = np.mean([r['metrics']['clean_accuracy'] for r in matched])*100
        lines.append(f'| {arm} | {paper[0]:.2f} ± {paper[1]:.2f}% | {s["mean"]:.2f} ± {error}% | {s["n"]} | {acc:.2f}% |')
    for a, b in [('fixed', 'random'), ('random', 'retrain'), ('random', 'select'), ('random', 'full'),
                 ('select', 'full'), ('retrain', 'select')]:
        first = {k[:3]: r['metrics']['asr'] for k, r in trials.items() if k[3] == a}
        second = {k[:3]: r['metrics']['asr'] for k, r in trials.items() if k[3] == b}
        keys = sorted(first.keys() & second.keys())
        if keys:
            result['paired_differences'][f'{b} minus {a}'] = stats([(second[k]-first[k])*100 for k in keys])
    write_json(result, output/'summary.json')
    (output/'table.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if rows:
        plot(rows, output)
    return result


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    by_success = sorted(rows, key=lambda r: r['metrics']['asr'])
    selected = [by_success[0]] if len(rows) == 1 else [by_success[0], by_success[-1]]
    for r in selected:
        label = f'{r["arm"]}, ASR {r["metrics"]["asr"]:.1%}'
        h = r['victim_history']
        axes[0].plot([v['epoch'] for v in h], [v.get('alignment_cosine', np.nan) for v in h], label=label)
        a = r['alignment']
        axes[1].plot([v['step'] for v in a], [v['cosine'] for v in a], label=label)
    if rows[0].get('clean_history'):
        h = rows[0]['clean_history']
        axes[0].plot([v['epoch'] for v in h], [v.get('alignment_cosine', np.nan) for v in h], label='Clean control', linestyle='--')
    for ax, xlabel in zip(axes, ['Victim epoch', 'Craft step']):
        ax.set(xlabel=xlabel, ylabel='Global gradient cosine')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'alignment.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results')
    parser.add_argument('--output', default='results/analysis')
    args = parser.parse_args()
    result = summarize(read_jsonl(args.results), args.output)
    print((Path(args.output)/'table.md').read_text(encoding='utf-8'))
    print(result['paired_differences'])


if __name__ == '__main__':
    main()
