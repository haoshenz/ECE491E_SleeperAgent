"""Rebuild the presentation README and figures from frozen experiment outputs.

No training, result rewriting, or execution of reference-repository code.
"""
import argparse
from pathlib import Path
import json
import textwrap
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
PAPER = np.array([19.25, 33.95, 63.49, 85.27])
PAPER_SE = np.array([3.01, 4.57, 6.13, 5.90])


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
        (PAPER, PAPER_SE, -.19, '#a6bddb', 'Paper Table 10 (reported mean ± SE)'),
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


def build_readme(rows, asr, acc, clean, noise):
    mean = asr.mean(0); error = sem(asr)
    table = ['| Configuration | Patch during crafting | Selection | Retrains | Paper ASR ± SE | Our ASR ± SE | n | Clean accuracy |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for i, label in enumerate(('Fixed patch', 'Random patch', 'Random + selection', 'Full attack')):
        table.append(f'| {label} | {"Bottom-right" if i == 0 else "Random"} | '
                     f'{"Largest full-gradient norms" if i >= 2 else "Random target images"} | '
                     f'{4 if i == 3 else 0} | {PAPER[i]:.2f} ± {PAPER_SE[i]:.2f}% | '
                     f'{mean[i]:.2f} ± {error[i]:.2f}% | 3 | {acc.mean(0)[i]:.2f}% |')
    pairs_table = ['| Source → target | Seed | Fixed | Random | Random + selection | Full |', '|---|---:|---:|---:|---:|---:|']
    controls_table = ['| Source → target | Clean model + patch | Random-noise control | Selected-noise control | Full attack |',
                      '|---|---:|---:|---:|---:|']
    for i, row in enumerate(rows):
        pairs_table.append(f'| {PAIR_LABELS[i]} | `{row[0]["seed"]}` | '+' | '.join(f'{v:.1f}%' for v in asr[i])+' |')
        controls_table.append(f'| {PAIR_LABELS[i]} | {clean[i]:.1f}% | {noise[i,0]:.1f}% | {noise[i,1]:.1f}% | {asr[i,3]:.1f}% |')
    difference = asr[:, 3]-asr[:, 0]
    radius = float(t.ppf(.975, 2)*sem(difference))
    lo, hi = difference.mean()-radius, difference.mean()+radius
    text = f'''# Sleeper Agent: Clean-Label Backdoors on CIFAR-10

From-scratch reproduction of Souri et al., *Sleeper Agent: Scalable Hidden Trigger Backdoors for Neural Networks Trained from Scratch* (NeurIPS 2022), for the ECE 491E empiricist role. [Paper](https://arxiv.org/abs/2106.08970) · [reference implementation](https://github.com/hsouri/Sleeper-Agent).

The attack subtly changes **500 correctly labeled training images (1%)**. A victim trained from scratch behaves normally on ordinary images, but an **8×8 sticker** on a source-class test image makes it predict the target class. The sticker is never inserted into the victim's training images.

**Headline:** the full attack reached **{mean[3]:.2f}% mean attack success rate (ASR)** with **{acc.mean(0)[3]:.2f}% clean accuracy** across **three source/target pairs** (12 attack trials + all noise controls completed).

## Setup

Requires Python 3.12 and a CUDA 12.8-compatible NVIDIA GPU (tested on an RTX 5070 Ti Laptop, 12 GB VRAM, 32 GB RAM). From the repository root:

```powershell
# 1. Create the environment
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt

# 2. Check the GPU is visible
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 3. Download CIFAR-10 (official source, checksum-verified) into data/
.venv/Scripts/python.exe scripts/download_data.py

# 4. Optional: run the correctness tests
.venv/Scripts/python.exe -m pytest -q
```

`requirements-lock.txt` pins the exact tested versions; `requirements.txt` lists direct dependencies. `results/` and `data/` are not tracked by Git; only load PyTorch checkpoints you trust.

## Run

**Quick smoke test** (small model, few classes; checks that everything executes, *not* a measure of attack strength):

```powershell
.venv/Scripts/python.exe -m src.run_ablation --config configs/smoke.yaml
```

**One full-scale attack** (airplane → automobile, seed 17; all four ablation arms):

```powershell
.venv/Scripts/python.exe -m src.run_ablation --config configs/cifar10_full.yaml --pairs 1 --arms fixed random select full
```

Use `--arms full` to run only the complete attack. Arms: `fixed` (fixed patch), `random` (random patch), `select` (+ gradient-norm selection), `full` (+ 4 surrogate retrains), `retrain` (retraining without selection). Runs checkpoint every epoch / every 5 crafting steps and skip completed trials, so re-running the same command resumes. Outputs go to `results/full/`.

**The full three-pair study** shown in the results (12 attack trials plus controls; pairs and seeds in [configs/main_study.json](configs/main_study.json)). It writes `reports/main_study/measurements.json`, `summary.json` and `controls.json`:

```powershell
.venv/Scripts/python.exe -u scripts/run_study.py
```

**Rebuild the figures and this README** from saved measurements (no training):

```powershell
.venv/Scripts/python.exe -m analysis.build_readme --skip-examples
```

Omit `--skip-examples` to also regenerate the example images (needs CIFAR-10 and the saved victim checkpoints in `results/`).

**Cost:** one standard epoch took about **28 s**, so an 80-epoch victim took about **38 min** on the laptop above.

## Results

ASR is the percentage of **all 1,000 held-out source-class test images** classified as the target after the patch is pasted at a random location. Clean accuracy uses all 10,000 CIFAR-10 test images.

### 1. Ablation vs. the paper

![Paper and reproduction ASR comparison](reports/main_study/figures/01_reproduction_vs_paper.png)

{chr(10).join(table)}

Error bars are **one standard error across three pair/seed trials** (n = 3, same pairs in every arm). The table contains the four configurations evaluated in this study. The paper's values come from a different set of trials and are reference points, not matched comparisons. Our ordering matches the paper (fixed < random < random + selection < full).

### 2. Per-pair results

![ASR by configuration for each source/target pair](reports/main_study/figures/02_pairwise_asr.png)

{chr(10).join(pairs_table)}

The random-patch arm ranges from **34.0% to 97.9%**, selection lifts truck → dog to **75.2%**, and the full attack reaches **99.7–99.9%** on all three pairs. Each pair has one seed, so pair difficulty and initialization effects are not separated.

Mean gains along the sequence: **+{mean[1]-mean[0]:.2f} points** for random placement, **+{mean[2]-mean[1]:.2f}** for selection, **+{mean[3]-mean[2]:.2f}** for retraining. Each incremental paired 95% t-interval includes zero at n = 3. The full-minus-fixed gain is **+{difference.mean():.2f} points**, 95% interval **[{lo:.2f}, {hi:.2f}]**. These are exploratory, unadjusted small-sample intervals.

### 3. Controls: is poisoning doing the work?

![Clean and noise controls, and clean classification accuracy](reports/main_study/figures/03_controls_and_accuracy.png)

{chr(10).join(controls_table)}

A clean model with the patch gets only **0.3–3.2%** ASR, and models trained with random or selected *noise* of the same budget stay at **0.1–2.3%**, while the full attack exceeds 99% on every pair. The patch alone does not explain the attack. The shared clean model has **92.18%** accuracy and every attacked victim stays within **0.5 points** of it, so the backdoor is invisible to ordinary accuracy checks.

### 4. What the poisons look like

![Clean, poisoned, amplified differences, and patched held-out examples](reports/main_study/figures/04_poison_examples.png)

Columns 1–2: the **same target-class training image** before and after poisoning (label unchanged). Column 3: the difference, amplified for visibility. Columns 4–5: a **source-class test image** without and with the trigger, labeled with the full-attack victim's predictions. One clean-correct, trigger-flipped example per pair is shown for illustration; the ASR numbers above use all 1,000 source images.

Every run passed checks for exactly 500 changed images, unchanged labels, target-class-only poisons, valid pixel range, `max(abs(delta)) <= 16/255 + 1e-6`, and absence of the exact trigger in training data.

### 5. Gradient alignment

![Crafting and victim-training gradient alignment](reports/main_study/figures/05_gradient_alignment.png)

Left: alignment during crafting using the full poison gradient; dotted lines mark the four surrogate refreshes. Right: alignment while training the victim on the original pair's **16-example telemetry subset** (so its scale differs from the 500-image objective). Alignment is a diagnostic; held-out ASR and the controls are the behavioral evidence.

## How the attack works

1. **Train a clean surrogate** (shared across pairs, also used as the clean baseline).
2. **Pick 500 target-class images**: random, or those with the largest parameter-gradient norms.
3. **Craft perturbations** (≤ 16/255) so the training gradient of the poisons aligns with the gradient of *patched source images labeled as the target*.
4. **Refresh the surrogate** (full arm): retrain from scratch on the current poisoned data 4 times during crafting.
5. **Train a fresh victim** on the poisoned dataset and evaluate on held-out images.

<details>
<summary>Core objective (explanatory excerpt; the real code is in src/craft.py)</summary>

```python
params = tuple(surrogate.parameters())
grad_adv = torch.autograd.grad(
    cross_entropy(surrogate(patched_source_train), target_labels), params
)
a = torch.cat([g.reshape(-1) for g in grad_adv]).detach()

grad_train = torch.autograd.grad(
    cross_entropy(surrogate(augment(clean + delta)), true_labels),
    params, create_graph=True,
)
g = torch.cat([v.reshape(-1) for v in grad_train])
alignment_loss = 1 - cosine_similarity(g[None], a[None])[0]

for step in range(250):
    cosine, grad = alignment_step(...)  # exact two-pass derivative
    signed_adam_step(optimizer, delta, grad, units="normalized")
    project(delta, clean, epsilon=16 / 255)
    if step + 1 in (50, 100, 150, 200):  # full arm only
        surrogate = train_fresh_surrogate(current_poisoned_data, epochs=40)
        a = recompute_adversarial_gradient(surrogate).detach()
```

The implementation evaluates the derivative in two passes to avoid holding double-backward graphs for all 500 images; direct-gradient and finite-difference tests verify it. Crop/flip augmentation is differentiable, and BatchNorm statistics are fixed during crafting.

</details>

| Setting | Value |
|---|---|
| Data / architecture | Full CIFAR-10 / CIFAR ResNet-18 (3×3 stem, no maxpool) |
| Poison budget / bound | 500 target-class images / 16/255 in pixel space |
| Trigger / source gradient | Seeded Bernoulli 8×8 patch / 512 source training images |
| Training | SGD, Nesterov 0.9, LR 0.1, weight decay 4e-4, batch 128 |
| Epochs | 40 initial surrogate, 40 per refresh, 80 per victim/control |
| LR drops | ×0.1 at epochs 14, 24, 35 |
| Crafting | 250 signed-Adam steps, normalized-coordinate LR 0.025, ×0.1 at steps 93, 156, 218 |
| Surrogate retraining | T = 0 or 4; fresh initialization at each refresh |
| Precision | Mixed precision for training; FP32 for crafting |

Full settings: [configs/cifar10_full.yaml](configs/cifar10_full.yaml).

## Limitations

- Only 3 pair/seed trials per arm, so the small incremental gaps between arms are not statistically settled.
- This study covers four selected configurations from Table 10. More pairs/seeds, ImageNet, cross-architecture transfer, defenses, and epsilon/retraining sweeps were not run.
- Surrogate initial weights are shared across pairs, which saves compute but does not sample surrogate-initialization variance.
- Reference code was consulted only for the signed-Adam schedule and the fresh-initialization retraining choice; everything else is an independent implementation.

## Repository layout

```text
src/                          data, model, selection, crafting, victim training/evaluation
scripts/                      CIFAR-10 download, main-study runner
analysis/build_readme.py      saved measurements → README and figures
configs/                      full recipe, smoke config, main-study pairs
reports/main_study/           frozen measurements, summary, controls, figures used above
reports/smoke/                development-only evidence
tests/                        correctness tests
results/, data/               local checkpoints, logs, dataset (not tracked by Git)
```

Primary data: [measurements.json](reports/main_study/measurements.json), [summary.json](reports/main_study/summary.json), [controls.json](reports/main_study/controls.json), and [presentation_data.json](reports/main_study/presentation_data.json) (compact export of the plotted values). The original eight-pair study design is in [docs/original_study_plan.md](docs/original_study_plan.md).
'''
    (ROOT/'README.md').write_text(text, encoding='utf-8')
    data = {'arms': list(ARMS), 'pairs': [list(p) for p in PAIRS], 'pair_labels': list(PAIR_LABELS),
            'asr_percent': asr.tolist(), 'clean_accuracy_percent': acc.tolist(),
            'clean_baseline_asr_percent': clean.tolist(),
            'noise_asr_percent_random_then_gradient': noise.tolist(),
            'paper_means_percent': PAPER.tolist(), 'paper_se_percent': PAPER_SE.tolist(),
            'full_minus_fixed': {'n': 3, 'mean_pp': float(difference.mean()), 'ci95_pp': [float(lo), float(hi)]}}
    (OUT/'presentation_data.json').write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip-examples', action='store_true')
    args = p.parse_args()
    rows, asr, acc, clean, noise = load()
    make_charts(rows, asr, acc, clean, noise)
    if not args.skip_examples:
        make_examples(rows)
    elif not (FIG/'04_poison_examples.png').exists():
        raise FileNotFoundError('Image examples are not yet generated; run once without --skip-examples.')
    build_readme(rows, asr, acc, clean, noise)
    print(f'Rebuilt README and 5 figures from {len(rows)*4} completed trials.')


if __name__ == '__main__':
    main()
