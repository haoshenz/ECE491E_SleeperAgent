"""Insert measured full-scale results only after a complete paired sweep."""
import json
from pathlib import Path
import sys
import shutil
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import read_jsonl
from analysis.summarize import summarize, PAPER


def main():
    root = Path('results/full')
    rows = read_jsonl(root/'results.jsonl')
    report = Path('reports/full')
    summary = summarize(rows, report)
    if set(summary['arms']) != set(PAPER) or any(s['n'] < 8 for s in summary['arms'].values()):
        raise RuntimeError('Full report requires eight completed paired trials in every arm')
    pair_sets = [{(r['source'], r['target'], r['seed']) for r in rows if r['arm'] == arm}
                 for arm in PAPER]
    if any(pairs != pair_sets[0] for pairs in pair_sets[1:]):
        raise RuntimeError('The completed arms must use identical source/target pairs and seeds')
    table = (report/'table.md').read_text(encoding='utf-8')
    high = max(rows, key=lambda r: r['metrics']['asr'])
    shutil.copy2(Path(high['directory'])/'poison_grid.png', report/'poison_grid.png')
    write_path = report/'measurements.json'
    write_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    differences = []
    for name, s in summary['paired_differences'].items():
        ci = s['ci95']
        resolved = 'excludes zero' if ci[0] > 0 or ci[1] < 0 else 'includes zero; the direction remains unresolved'
        differences.append(f'{name}: {s["mean"]:+.2f} percentage points, paired 95% CI '
                           f'[{ci[0]:+.2f}, {ci[1]:+.2f}] ({resolved}).')
    clean = sum(r['clean_control']['clean_accuracy'] for r in rows)/len(rows)
    delta = sum(r['metrics']['clean_accuracy']-r['clean_control']['clean_accuracy'] for r in rows)/len(rows)
    section = ('The five-arm, eight-pair sweep finished. Values below are measured ASR on held-out CIFAR-10 images. '
               'Error bars are standard errors over paired trials.\n\n'+table+'\n'+
               f'Mean clean-control accuracy: {clean:.2%}. Mean poisoned-minus-clean accuracy: {delta*100:+.2f} percentage points.\n\n'+
               ' '.join(differences)+'\n\n'+
               '![Clean and poisoned examples](reports/full/poison_grid.png)\n\n'+
               '![Measured gradient alignment](reports/full/alignment.png)\n\n'+
               'The results quantify this implementation and its recorded choices; they do not by themselves establish exact numerical reproduction of every paper setting.\n\n')
    path = Path('README.md')
    text = path.read_text(encoding='utf-8')
    marker = '<!-- RESULTS_START -->'
    start = text.index(marker)+len(marker)
    end = text.index('<!-- RESULTS_END -->')
    text = text[:start]+'\n\n'+section+text[end:]
    path.write_text(text, encoding='utf-8')
    print('README updated with completed full-scale measurements')


if __name__ == '__main__':
    main()
