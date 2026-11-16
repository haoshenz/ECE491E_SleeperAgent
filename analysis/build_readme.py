"""Rebuild the presentation README and figures from frozen experiment outputs.

No training, result rewriting, or execution of reference-repository code.
"""
import argparse
from pathlib import Path
import json
import textwrap
import re
from string import Template
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/main_study'
FIG = OUT/'figures'
ARMS = ('fixed', 'random', 'select', 'full')
LABELS = ('Fixed', 'Random', 'Random +\nselection', 'Full\n(+ retraining)')
PAIRS = ((0, 1), (9, 5), (4, 6))
PAIR_LABELS = ('Airplane → automobile', 'Truck → dog', 'Deer → frog')
COLORS = ('#7b3294', '#2166ac', '#1b7837')
ARM_COLORS = ('#7b3294', '#2166ac', '#d95f02', '#1b7837')
_ABLATION_REFERENCE = json.loads((OUT/'presentation_data.json').read_text(encoding='utf-8'))
PAPER = np.array(_ABLATION_REFERENCE['paper_means_percent'])
# Keep the archived JSON key; its name does not establish the paper's uncertainty type.
PAPER_SE = np.array(_ABLATION_REFERENCE['paper_se_percent'])
TABLE6 = ROOT/'reports/table6'
ARCHITECTURES = ('resnet18', 'mobilenetv2', 'vgg11')
ARCH_LABELS = ('ResNet-18', 'MobileNetV2', 'VGG11')


def load_table6():
    data = json.loads((TABLE6/'rescored.json').read_text(encoding='utf-8'))
    assert data['all_random_scores_match_original']
    assert (data['source'], data['target'], data['seed']) == (0, 1, 17)
    assert data['test_images'] == 10000 and data['source_test_images'] == 1000
    cells = {(r['attack'], r['architecture']): r for r in data['cells']}
    expected = {(attack, arch) for attack in ('htba', 'clba', 'full') for arch in ARCHITECTURES}
    assert len(data['cells']) == len(cells) == 9 and set(cells) == expected
    for protocol in ('native', 'random'):
        for row in data['tables'][protocol]:
            mode = 'fixed' if protocol == 'native' and row['attack'] != 'full' else 'random'
            values = [cells[row['attack'], arch][mode]['asr'] for arch in ARCHITECTURES]
            assert row['test_placement'] == mode
            assert all(np.isclose(row['asr'][arch], value) for arch, value in zip(ARCHITECTURES, values))
            assert np.isclose(row['mean_asr'], np.mean(values))
    return data


def attack_table(rows):
    lines = ['| Attack | ResNet-18 | MobileNetV2 | VGG11 | Average |', '|---|---:|---:|---:|---:|']
    for row in rows:
        values = [row['asr'][arch] for arch in ARCHITECTURES]+[row['mean_asr']]
        lines.append('| '+' | '.join([row['label'], *[f'{100*v:.2f}%' for v in values]])+' |')
    return '\n'.join(lines)


def clean_table(data):
    cells = {(r['attack'], r['architecture']): r for r in data['cells']}
    rows = []
    for row in data['tables']['native']:
        values = {arch: cells[row['attack'], arch]['random']['clean_accuracy'] for arch in ARCHITECTURES}
        rows.append({'label': row['label'], 'asr': values, 'mean_asr': np.mean(list(values.values()))})
    return attack_table(rows)


def paper_comparison_table(data):
    reference = json.loads((TABLE6/'paper_reference.json').read_text(encoding='utf-8'))
    paper = {r['attack']: r for r in reference['rows']}
    lines = ['| Attack | ResNet-18: ours / paper | MobileNetV2: ours / paper | VGG11: ours / paper | Average: ours / paper |',
             '|---|---:|---:|---:|---:|']
    for row in data['tables']['native']:
        published = paper[row['attack']]
        values = [f"{100*row['asr'][arch]:.2f}% / {published['asr'][arch]:.2f}%" for arch in ARCHITECTURES]
        values.append(f"{100*row['mean_asr']:.2f}% / {published['average']:.2f}%")
        lines.append('| '+' | '.join([row['label'], *values])+' |')
    fixed = {r['architecture']: r['fixed']['asr']*100 for r in data['cells'] if r['attack'] == 'full'}
    values = [f'{fixed[arch]:.2f}% / —' for arch in ARCHITECTURES]
    values.append(f'{np.mean(list(fixed.values())):.2f}% / —')
    lines.append('| '+' | '.join(['Sleeper Agent, trigger fixed at (24,24)', *values])+' |')
    return '\n'.join(lines)


def main_body_word_count(text):
    """Count rendered words before Appendix, excluding HTML details and code."""
    main = text.split('## Appendix', 1)[0]
    main = re.sub(r'<details\b[^>]*>.*?</details>', '', main, flags=re.S)
    main = re.sub(r'```.*?```', '', main, flags=re.S)
    main = re.sub(r'`[^`]*`', '', main)
    main = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', main)
    main = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', main)
    main = re.sub(r'<[^>]+>', ' ', main)
    return len(re.findall(r"\b[\w]+(?:[’'→.-][\w]+)*\b", main))


def make_table6_chart(data):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3), sharey=True)
    x = np.arange(3)
    colors = ('#7570b3', '#d95f02', '#1b7837')
    for ax, protocol, title in zip(axes, ('native', 'random'),
                                   ('Native test protocol', 'All attacks: random trigger position')):
        for i, (row, color) in enumerate(zip(data['tables'][protocol], colors)):
            values = [100*row['asr'][arch] for arch in ARCHITECTURES]
            bars = ax.bar(x+(i-1)*.25, values, .23, color=color, label=row['label'])
            ax.bar_label(bars, labels=[f'{v:.1f}' for v in values], padding=3, fontsize=8)
        ax.set(title=title, xticks=x, xticklabels=ARCH_LABELS, ylim=(0, 110),
               yticks=np.arange(0, 101, 20))
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Attack success rate (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, .015))
    fig.suptitle('Airplane → automobile, seed 17', fontsize=14, y=.98)
    fig.subplots_adjust(left=.07, right=.985, top=.85, bottom=.17, wspace=.13)
    destination = TABLE6/'figures/01_native_and_random_asr.png'
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def load():
    rows = json.loads((OUT/'measurements.json').read_text(encoding='utf-8'))
    table = {(r['source'], r['target'], r['arm']): r for r in rows}
    assert len(rows) == len(table) == 12, 'Expected one completed run per pair and arm'
    ordered = [[table[(*pair, arm)] for arm in ARMS] for pair in PAIRS]
    # Pool per arm only after checking the actual scientific settings.
    keys = ('architecture', 'width', 'batch_size', 'epochs_surrogate', 'epochs_retrain',
            'epochs_victim', 'lr', 'milestones', 'weight_decay', 'amp', 'epsilon',
            'poison_count', 'patch_size', 'source_count', 'craft_steps', 'craft_lr',
            'craft_lr_units', 'selection', 'patch_location', 'retraining_factor')
    for col in range(4):
        settings = [{k: row[col]['config'][k] for k in keys} for row in ordered]
        assert all(s == settings[0] for s in settings)
    asr = np.array([[r['metrics']['asr']*100 for r in row] for row in ordered])
    acc = np.array([[r['metrics']['clean_accuracy']*100 for r in row] for row in ordered])
    controls = {(c['source'], c['target']): c
                for c in json.loads((OUT/'controls.json').read_text(encoding='utf-8'))}
    controls = [controls[pair] for pair in PAIRS]
    clean = np.array([c['clean_model']['asr']*100 for c in controls])
    noise = np.array([[c['random_noise']['asr']*100, c['selected_noise']['asr']*100] for c in controls])
    return ordered, asr, acc, clean, noise


def sem(a):
    return np.std(a, axis=0, ddof=1)/np.sqrt(len(a))


def save(fig, name):
    fig.savefig(FIG/f'{name}.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 14, 'axes.labelsize': 11, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.axisbelow': True, 'figure.facecolor': 'white',
        'legend.frameon': False, 'savefig.facecolor': 'white'})


def make_charts(rows, asr, acc, clean, noise):
    FIG.mkdir(parents=True, exist_ok=True)
    style()
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(9.5, 5.0), layout='constrained')
    for values, errors, offset, color, label in [
        (PAPER, PAPER_SE, -.19, '#a6bddb', 'Paper Table 10 (mean ± as reported)'),
        (asr.mean(0), sem(asr), .19, '#e08214', 'Our runs (mean ± SE, n = 3 pairs)')]:
        bars = ax.bar(x+offset, values, .36, color=color, yerr=errors, capsize=4, label=label)
        for b, v, error in zip(bars, values, errors):
            ax.text(b.get_x()+b.get_width()/2, v+error+1.4, f'{v:.1f}', ha='center', fontsize=10)
    ax.set(xticks=x, xticklabels=LABELS, ylim=(0, 111), yticks=np.arange(0, 101, 20),
           ylabel='Attack success rate (%)', title='Sleeper Agent ablation: our measurements and the paper')
    ax.grid(axis='y', alpha=.22)
    ax.legend(loc='upper left', fontsize=9)
    save(fig, '01_reproduction_vs_paper')

    fig, ax = plt.subplots(figsize=(9.5, 4.7), layout='constrained')
    for values, color, label in zip(asr, COLORS, PAIR_LABELS):
        ax.plot(x, values, '-o', lw=2.3, ms=6, color=color, label=label)
    ax.set(xticks=x, xticklabels=LABELS, ylim=(0, 105), yticks=np.arange(0, 101, 20),
           ylabel='Attack success rate (%)', title='The same four configurations on three pair/seed trials')
    ax.grid(alpha=.22); ax.legend(loc='lower right', fontsize=10)
    save(fig, '02_pairwise_asr')

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), layout='constrained')
    ax = axes[0]; xp = np.arange(3)
    for values, offset, color, label in [(asr[:, 3], -.24, '#1b7837', 'Full attack'),
            (clean, 0, '#2166ac', 'Clean model + patch'),
            (noise[:, 1], .24, '#969696', 'Noise-only training + patch')]:
        bars = ax.bar(xp+offset, values, .22, color=color, label=label)
        for b, value in zip(bars, values):
            ax.text(b.get_x()+b.get_width()/2, value+1.7, f'{value:.1f}', ha='center', fontsize=9)
    ax.set(title='Does the patch work without poisoning?', ylabel='Patched ASR (%)',
           xticks=xp, xticklabels=('0 → 1', '9 → 5', '4 → 6'), ylim=(0, 112), yticks=np.arange(0, 101, 20))
    ax.grid(axis='y', alpha=.22); ax.legend(loc='center', bbox_to_anchor=(.5, .52), fontsize=9)
    ax = axes[1]
    ax.axhline(92.18, color='#555555', linestyle='--', lw=1.4, label='Shared clean baseline: 92.18%')
    for i, (color, label) in enumerate(zip(COLORS, PAIR_LABELS)):
        ax.scatter(x+(i-1)*.09, acc[i], color=color, s=36, zorder=3, label=label)
    ax.errorbar(x, acc.mean(0), yerr=sem(acc), color='black', fmt='D', capsize=4, ms=4,
                label='Mean ± SE', zorder=4)
    ax.set(title='Ordinary classification remains accurate', ylabel='Clean test accuracy (%)',
           xticks=x, xticklabels=LABELS, ylim=(91.2, 93.15))
    ax.grid(alpha=.22); ax.legend(loc='upper left', fontsize=8)
    save(fig, '03_controls_and_accuracy')

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), layout='constrained')
    ax = axes[0]
    for row, color, label in zip(rows, COLORS, PAIR_LABELS):
        history = row[3]['alignment']
        ax.plot([h['step'] for h in history], [h['cosine'] for h in history], lw=1.7, color=color, label=label)
    for k in (50, 100, 150, 200): ax.axvline(k, color='#555555', lw=1, linestyle=':', alpha=.6)
    ax.set(title='Crafting the full attack', xlabel='Craft update (dotted: retraining events)',
           ylabel='Global gradient cosine', ylim=(0, 1.03))
    ax.grid(alpha=.18); ax.legend(loc='lower right', fontsize=8)
    ax = axes[1]
    for r, color, label in zip(rows[0], ARM_COLORS, ('Fixed', 'Random', 'Random + selection', 'Full')):
        h = r['victim_history']
        ax.plot([v['epoch'] for v in h], [v['alignment_cosine'] for v in h], lw=1.7,
                color=color, label=f'{label}: ASR {r["metrics"]["asr"]:.1%}')
    h = rows[0][3]['clean_history']
    ax.plot([v['epoch'] for v in h], [v['alignment_cosine'] for v in h], color='#777777',
            linestyle='--', lw=1.5, label='Clean control (different target subset)')
    ax.set(title='Victim training: airplane → automobile', xlabel='Victim epoch',
           ylabel='Gradient cosine on telemetry subset', ylim=(-.03, 1.03))
    ax.grid(alpha=.18); ax.legend(loc='upper right', fontsize=8)
    save(fig, '05_gradient_alignment')


def make_examples(rows):
    import torch
    from torchvision.datasets import CIFAR10
    from src.models import make_model
    from src.patch import apply_patch
    torch.set_num_threads(4)
    ds = CIFAR10(str(ROOT/'data'), train=True, download=False)
    test = CIFAR10(str(ROOT/'data'), train=False, download=False)
    labels = np.asarray(ds.targets); test_labels = np.asarray(test.targets)
    order = np.concatenate([np.flatnonzero(labels == c) for c in range(10)])
    names = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    fig, axes = plt.subplots(3, 5, figsize=(12, 7.2))
    for i, row in enumerate(rows):
        r = row[3]; cfg = dict(r['config'], device='cpu')
        directory = ROOT/Path(r['directory'].replace('\\', '/'))
        poison = torch.load(directory/'poisons.pt', map_location='cpu', weights_only=False)
        idx = int(poison['indices'][0])
        clean = torch.from_numpy(ds.data[order[idx]].copy()).permute(2, 0, 1).float()/255
        changed = (clean+poison['delta'][0]).clamp(0, 1)
        source, target = PAIRS[i]
        ids = np.flatnonzero(test_labels == source)[:12]
        examples = torch.from_numpy(test.data[ids].copy()).permute(0, 3, 1, 2).float()/255
        generator = torch.Generator().manual_seed(r['seed']+12345)
        patched = apply_patch(examples, poison['patch'], 'random', generator)
        model = make_model(cfg)
        state = torch.load(directory/'victim/train.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model']); model.eval()
        with torch.no_grad():
            pred = model(examples).argmax(1); attack = model(patched).argmax(1)
        possible = torch.where((pred == source) & (attack == target))[0]
        assert len(possible), 'No clean-correct/flipped demonstration among first 12 images'
        j = int(possible[0])
        images = [clean, changed, (changed-clean)*4+.5, examples[j], patched[j]]
        subtitles = [f'Label: {names[target]}', f'Same label: {names[target]}',
                     'Difference ×4 + 0.5', f'Predicts: {names[int(pred[j])]}', f'Predicts: {names[int(attack[j])]}']
        for col, (image, subtitle) in enumerate(zip(images, subtitles)):
            ax = axes[i, col]
            ax.imshow(image.permute(1, 2, 0).clamp(0, 1)); ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(subtitle, fontsize=9)
        axes[i, 0].set_ylabel(PAIR_LABELS[i].replace(' → ', '\n→ '), fontsize=10)
        del model, state
    for ax, title in zip(axes[0], ('Clean training', 'Poisoned training', 'Amplified change', 'Clean held-out', 'Patched held-out')):
        ax.set_title(title, fontsize=11)
    fig.subplots_adjust(left=.12, right=.99, top=.93, bottom=.06, hspace=.38, wspace=.12)
    save(fig, '04_poison_examples')


def build_readme(rows, asr, acc, clean, noise, write_presentation_data=True):
    mean = asr.mean(0); error = sem(asr)
    table = ['| Configuration | Paper ASR (± as reported) | Our ASR ± SE | Clean acc. |',
             '|---|---:|---:|---:|']
    for i, label in enumerate(('Fixed patch', 'Random patch', 'Random + selection', 'Full attack')):
        table.append(f'| {label} | {PAPER[i]:.2f} ± {PAPER_SE[i]:.2f}% | '
                     f'{mean[i]:.2f} ± {error[i]:.2f}% | {acc.mean(0)[i]:.2f}% |')
    pairs_table = ['| Source → target | Seed | Fixed | Random | Random + selection | Full |', '|---|---:|---:|---:|---:|---:|']
    controls_table = ['| Source → target | Clean model + patch | Random-noise control | Selected-noise control | Full attack |',
                      '|---|---:|---:|---:|---:|']
    for i, row in enumerate(rows):
        pairs_table.append(f'| {PAIR_LABELS[i]} | `{row[0]["seed"]}` | '+' | '.join(f'{v:.1f}%' for v in asr[i])+' |')
        controls_table.append(f'| {PAIR_LABELS[i]} | {clean[i]:.1f}% | {noise[i,0]:.1f}% | {noise[i,1]:.1f}% | {asr[i,3]:.1f}% |')
    difference = asr[:, 3]-asr[:, 0]
    radius = float(t.ppf(.975, 2)*sem(difference))
    lo, hi = difference.mean()-radius, difference.mean()+radius
    comparison = load_table6()
    fixed_cells = {r['architecture']: r for r in comparison['cells'] if r['attack'] == 'full'}
    fixed_note = ', '.join(f"{100*fixed_cells[arch]['fixed']['asr']:.2f}% on {label}"
                           for arch, label in zip(ARCHITECTURES, ARCH_LABELS))
    baseline = json.loads((OUT/'controls.json').read_text(encoding='utf-8'))[0]['clean_model']['clean_accuracy']*100
    victim_minutes = np.mean([r['victim_seconds']/60 for row in rows for r in row])
    template = Template((ROOT/'analysis/README.template.md').read_text(encoding='utf-8'))
    text = template.substitute(
        full_asr=f'{mean[3]:.2f}', full_clean=f'{acc.mean(0)[3]:.2f}',
        ablation_table='\n'.join(table), pair_table='\n'.join(pairs_table),
        controls_table='\n'.join(controls_table),
        native_table=attack_table(comparison['tables']['native']),
        random_table=attack_table(comparison['tables']['random']),
        paper_comparison_table=paper_comparison_table(comparison),
        architecture_clean_table=clean_table(comparison),
        clean_baseline=f'{baseline:.2f}', victim_minutes=f'{victim_minutes:.0f}',
        sleeper_fixed_note=fixed_note, rescore_seconds=f"{comparison['elapsed_seconds']:.1f}")
    (ROOT/'README.md').write_text(text, encoding='utf-8')
    data = {'arms': list(ARMS), 'pairs': [list(p) for p in PAIRS], 'pair_labels': list(PAIR_LABELS),
            'asr_percent': asr.tolist(), 'clean_accuracy_percent': acc.tolist(),
            'clean_baseline_asr_percent': clean.tolist(),
            'noise_asr_percent_random_then_gradient': noise.tolist(),
            'paper_means_percent': PAPER.tolist(), 'paper_se_percent': PAPER_SE.tolist(),
            'full_minus_fixed': {'n': 3, 'mean_pp': float(difference.mean()), 'ci95_pp': [float(lo), float(hi)]}}
    if write_presentation_data:
        (OUT/'presentation_data.json').write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip-examples', action='store_true')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--table6-only', action='store_true',
                   help='Update README and the Table 6 figure; preserve all original main-study report files.')
    modes.add_argument('--readme-only', action='store_true', help='Regenerate README without rewriting figures or reports.')
    args = p.parse_args()
    rows, asr, acc, clean, noise = load()
    if not args.table6_only and not args.readme_only:
        make_charts(rows, asr, acc, clean, noise)
        if not args.skip_examples:
            make_examples(rows)
        elif not (FIG/'04_poison_examples.png').exists():
            raise FileNotFoundError('Image examples are not yet generated; run once without --skip-examples.')
    if not args.readme_only:
        make_table6_chart(load_table6())
    text = build_readme(rows, asr, acc, clean, noise,
                        write_presentation_data=not (args.table6_only or args.readme_only))
    print(f'Rebuilt README from {len(rows)*4} ablation trials and 9 comparison cells. '
          f'Main body: {main_body_word_count(text)} words (excluding Appendix and code).')


if __name__ == '__main__':
    main()
