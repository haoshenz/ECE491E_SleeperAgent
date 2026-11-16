# Sleeper Agent: Clean-Label Backdoors on CIFAR-10

This repository contains an independent implementation of Souri et al.'s *Sleeper Agent: Scalable Hidden Trigger Backdoors for Neural Networks Trained from Scratch* (NeurIPS 2022), for ECE 491E. [Paper](https://arxiv.org/abs/2106.08970) · [reference implementation](https://github.com/hsouri/Sleeper-Agent).

Sleeper Agent changes 500 correctly labeled CIFAR-10 training images, or 1% of the dataset. The changes are designed to make a model learn a hidden error: when an 8×8 trigger is added to a source-class test image, the model predicts the chosen target class. The visible trigger is absent from the victim's training images.

The project includes an earlier three-pair ResNet-18 ablation study and an attack comparison run over November 14–15, 2026. In the ablation, the full attack averaged ${full_asr}% attack success with ${full_clean}% clean accuracy. The later comparison tested ResNet-18, MobileNetV2, and VGG11 on one class pair. All completed results are reported below, including the lower MobileNetV2 result.

## Setup

Requires Python 3.12, `uv`, and an NVIDIA GPU with CUDA 12.8 support. Run from the repository root:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt
.venv/Scripts/python.exe scripts/download_data.py
```

## Run

```powershell
# Three-pair ResNet-18 study
.venv/Scripts/python.exe -u scripts/run_study.py

# Three-model attack comparison
.venv/Scripts/python.exe scripts/table6_preflight.py
.venv/Scripts/python.exe scripts/run_table6.py --log-file results/table6/run.log

# Re-score saved models without training
.venv/Scripts/python.exe scripts/rescore_table6.py

# Rebuild the README and comparison figure
.venv/Scripts/python.exe -m analysis.build_readme --table6-only
```

The comparison runner resumes from checkpoints and updates [progress.md](progress.md). Re-scoring requires the saved local checkpoints.

## Results

Attack success rate (ASR) is the percentage of all 1,000 held-out source-class test images classified as the target after adding the trigger. Clean accuracy is measured on all 10,000 unmodified CIFAR-10 test images. Trigger position is stated separately for each table.

### 1. Attack comparison across architectures

The weekend run compared Hidden-Trigger Backdoor (HTBA, Saha et al.), Clean-Label Backdoor (CLBA, Turner et al.), and Sleeper Agent. Each attack was crafted using the same architecture as its fresh victim. All cells used airplane → automobile (`0→1`), seed `17`, 500 target-class poisons, an 8×8 trigger, and an 80-epoch victim.

*one pair (airplane->automobile), one seed; not a statistical reproduction*

![Attack success with method-specific and random test positions](reports/table6/figures/01_native_and_random_asr.png)

**Native test protocol.** HTBA uses a fixed position inset five pixels from the bottom-right, `(19,19)`. CLBA uses the bottom-right corner, `(24,24)`. Sleeper Agent uses uniformly random positions. Coordinates are zero-based `(row,column)`. This is a method-specific placement comparison; the remaining differences from the reference benchmark are listed below.

${native_table}

**All attacks with a random trigger position.** Every method uses the same evaluation seed, `717`, and batch size, `128`.

${random_table}

The row average is across architectures. It is not an average over repeated class pairs or seeds, and there are no error bars for this one-trial-per-cell comparison.

The fixed and random scores were obtained by loading the same saved victims, without retraining. All nine random ASRs and clean accuracies matched the original cell results. The re-scoring script reported ${rescore_seconds} seconds. With Sleeper Agent fixed at `(24,24)`, ASR was ${sleeper_fixed_note}. These results show that trigger position affected the outcome in this trial.

**Clean accuracy for the same victims:**

${architecture_clean_table}

Sleeper Agent reached high random-position ASR on ResNet-18 and VGG11, but 27.90% on MobileNetV2. MobileNetV2's clean accuracy was 88.54%. One pair and one seed are not enough to determine whether that gap is typical or to identify its cause.

The baseline implementations also differ from an exact replication: CLBA used a clean surrogate instead of the benchmark's adversarially trained surrogate. HTBA uses a corrected one-to-one greedy feature assignment. The same seeded random trigger was used across cells, and CLBA's stamped training trigger was projected back into the pixel-change bound. Full settings and sources are in [table6_baselines.md](docs/table6_baselines.md).

Saved outputs: [re-scored results](reports/table6/rescored.json), [comparison report](reports/table6/table6.md), and [original random-position summary](reports/table6/summary.json).

### 2. ResNet-18 ablation

The earlier study tested four Sleeper Agent configurations across three source/target pairs. All four were evaluated with random test-time trigger positions. The fixed/random distinction in this table refers to placement during poison crafting.

![ResNet-18 ablation compared with the paper](reports/main_study/figures/01_reproduction_vs_paper.png)

${ablation_table}

Values are means ± one standard error across three pair/seed trials. The same pairs were used in every configuration. The paper's Table 10 values are reference points from a different set of trials. This study did not run the separate retraining-without-selection configuration.

Mean ASR increased from fixed placement through random placement, selection, and retraining. With only three pair/seed trials, the smaller differences between configurations remain uncertain.

### 3. Results by class pair

![ResNet-18 ASR for each pair and configuration](reports/main_study/figures/02_pairwise_asr.png)

${pair_table}

The random-patch configuration varied from 34.0% to 97.9% ASR across pairs. The full attack reached 99.7–99.9%. Since each pair has one seed, class-pair difficulty and initialization effects are not separated.

### 4. Clean and noise controls

![Clean and noise controls alongside clean accuracy](reports/main_study/figures/03_controls_and_accuracy.png)

${controls_table}

The clean-model control uses the trigger without poisoning. The noise controls replace optimized poison changes with random noise of the same bound, on either randomly chosen or gradient-selected images. Their low ASRs support a poisoning-induced effect in the three tested pairs. The full attack's mean clean accuracy was ${full_clean}%, compared with 92.18% for the shared clean baseline.

These controls belong to the three-pair ResNet-18 study. They were not repeated for every cell of the architecture comparison.

### 5. Poison examples

![Original and poisoned training images, differences, and triggered test images](reports/main_study/figures/04_poison_examples.png)

The first two columns show the same target-class training image before and after poisoning, with its label unchanged. The third column amplifies the difference. The last two show a held-out source image before and after adding the trigger, with the victim's predictions. One clean-correct, trigger-flipped example was selected per pair for illustration; the reported ASR uses all 1,000 source images.

The main-study runs checked that exactly 500 target-class images changed, labels stayed unchanged, pixel changes stayed within `16/255`, pixels remained valid, and the exact trigger was absent from the training data. CLBA intentionally includes a clipped training trigger, so its checks allow that exception while still checking the perturbation bound before and after stamping.

### 6. Gradient alignment

![Gradient alignment during crafting and victim training](reports/main_study/figures/05_gradient_alignment.png)

The left plot tracks the full poison gradient during crafting; dotted lines mark surrogate refreshes. The right plot uses a 16-example subset during victim training on airplane → automobile. The scales differ because the subsets differ. Alignment checks the crafting objective; held-out ASR and the controls measure the resulting behavior.

## How Sleeper Agent makes the poison

1. Train a clean surrogate for 40 epochs.
2. Choose 500 target-class training images, randomly or by full-parameter gradient norm.
3. Adjust their pixels so learning from them pushes the model in a similar direction to learning that patched source images belong to the target class. This is gradient matching. The poisoned images keep their true labels.
4. In the full attack, train a new surrogate on the current poisoned dataset four times during crafting, then continue optimizing the images.
5. Train a separate victim from scratch on the final poisoned dataset for 80 epochs and evaluate it on held-out test images.

The clean control is a separate 80-epoch model. The initial surrogate and clean control were each reused across the three ablation pairs to reduce compute.

<details>
<summary>Core objective and settings</summary>

The following is an explanatory excerpt. The implementation is in [src/craft.py](src/craft.py).

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
    cosine, grad = alignment_step(...)
    signed_adam_step(optimizer, delta, grad, units="normalized")
    project(delta, clean, epsilon=16 / 255)
    if step + 1 in (50, 100, 150, 200):
        surrogate = train_fresh_surrogate(current_poisoned_data, epochs=40)
        a = recompute_adversarial_gradient(surrogate).detach()
```

The two-pass derivative avoids keeping second-derivative graphs for all 500 images in memory at once. Tests compare it with direct differentiation and finite differences. Crop/flip augmentation is differentiable, and BatchNorm statistics are frozen during crafting.

| Setting | Value |
|---|---|
| Dataset | Full CIFAR-10 |
| Poison budget / bound | 500 target-class images / 16/255 in pixel space |
| Trigger / source-gradient pool | Seeded Bernoulli RGB 8×8 patch / 512 source training images |
| Training | SGD, Nesterov 0.9, weight decay 4e-4, batch 128 |
| Initial learning rate | ResNet-18: 0.1; MobileNetV2 and VGG11: 0.01 |
| Epochs | 40 initial surrogate, 40 per refresh, 80 per victim/control |
| Learning-rate drops | Factor 0.1 at epochs 14, 24, 35 |
| Sleeper crafting | 250 signed-Adam steps, normalized-coordinate LR 0.025 |
| Crafting LR drops | Factor 0.1 at steps 93, 156, 218 |
| Surrogate refreshes | None in the reduced configurations; four fresh initializations in the full attack |
| Precision | Mixed precision for training; FP32 for crafting and evaluation |

Configs: [ResNet ablation](configs/cifar10_full.yaml), [ResNet comparison](configs/table6_resnet18.yaml), [MobileNetV2](configs/table6_mobilenetv2.yaml), [VGG11](configs/table6_vgg11.yaml).

</details>

## Limitations

- The ablation has three pair/seed trials per configuration. The architecture comparison has one pair and one seed per cell.
- Each architecture comparison cell crafts and tests on the same architecture with a fresh victim initialization. Cross-architecture transfer was not tested.
- The comparison retains the CLBA surrogate, HTBA matching, trigger, and placement differences described above. It is not an exact numerical reproduction of Table 6.
- Fixed/random re-scoring changes evaluation only. It does not show how the models would behave after crafting a different poison or using another seed.
- ImageNet, defenses, more pairs/seeds, and perturbation-budget or retraining-factor sweeps were not run.
- The Sleeper Agent implementation is independent. The baseline algorithms and new architecture definitions were checked against the cited benchmark source; those adaptations are documented separately.

## Repository layout

```text
src/                          data, models, poison crafting, victim training and evaluation
scripts/run_study.py           three-pair ResNet-18 ablation
scripts/run_table6.py          nine-cell attack comparison
scripts/rescore_table6.py      fixed/random evaluation of saved victims
analysis/build_readme.py       report data → README and figures
analysis/README.template.md    README wording and structure
configs/                      training recipes and study pairs
reports/main_study/            saved ablation measurements and original figures
reports/table6/                comparison, re-scoring, and figure
tests/                        correctness checks
results/, data/               local checkpoints, logs, and dataset (not tracked by Git)
```

The ablation's source data are [measurements.json](reports/main_study/measurements.json), [summary.json](reports/main_study/summary.json), and [controls.json](reports/main_study/controls.json). The weekend comparison and both test placements are recorded in [rescored.json](reports/table6/rescored.json). The earlier eight-pair proposal is retained in [original_study_plan.md](docs/original_study_plan.md); it describes the original plan, not the completed experiment count.
