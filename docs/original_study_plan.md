# Sleeper Agent — Reimplementation Plan

Handoff spec for a coding agent. Target hardware: RTX 5070 Ti laptop (Blackwell).

Paper: *Sleeper Agent: Scalable Hidden Trigger Backdoors for Neural Networks Trained from Scratch* (Souri, Fowl, Chellappa, Goldblum, Goldstein — NeurIPS 2022), arXiv 2106.08970v3. Reference implementation: `https://github.com/hsouri/Sleeper-Agent`.

**Write the attack from scratch.** Consult the reference repo only for the two details the paper omits (noted in §4.5 and §4.8). A from-scratch implementation is the deliverable being presented; a wrapper around someone else's repo is not.

---

## 1. Goal

Reproduce the **Table 10 ablation** on CIFAR-10 — the claim that random patch placement, surrogate retraining, and gradient-norm poison selection each contribute, and that combined they roughly quadruple attack success.

| Configuration | Paper ASR |
|---|---|
| Fixed patch location (bottom-right) | 19.25% (±3.01) |
| Random patch location | 33.95% (±4.57) |
| Random patch + retraining | 59.42% (±5.78) |
| Random patch + data selection | 63.49% (±6.13) |
| Random patch + retraining + data selection | 85.27% (±5.90) |

Secondary targets if time allows: retraining-factor sweep (T = 0/2/4/8 → 63.49/70.66/85.27/86.48) and the ε sweep (8→16/255 → 37.32/55.75/63.31/78.03/85.27).

**Non-goals:** ImageNet, black-box transfer across architectures, defense evaluation. All are in the paper; none fit the compute budget.

---

## 2. Threat model (what the code must respect)

- **Clean-label.** Poison images keep their true labels. No label flipping anywhere.
- **Bounded perturbation.** ‖δ‖∞ ≤ ε = 16/255 on poisoned training images.
- **Limited budget.** Only M = 1% of the training set may be perturbed (500 of 50,000).
- **Poisons come from the target class only.** All M poisons carry label `y_t`.
- **The trigger never appears in training data.** The patch is applied only at test time, and (during crafting) to source-class images used to estimate the adversarial gradient — never written into the poisoned training set.
- **Victim trains from scratch.** Fresh random init, standard recipe, no knowledge of the attack.

A correct implementation must make it impossible to violate these by accident. Assert them (§7).

---

## 3. Repository layout

```
sleeper-agent/
  README.md
  requirements.txt
  configs/
    smoke.yaml          # 2-class subset, ResNet-9, minutes — for development
    cifar10_full.yaml   # paper settings — for the real sweep
  src/
    data.py             # CIFAR-10 loading, normalization, differentiable augmentation
    models.py           # ResNet-18 (CIFAR variant), ResNet-9 for smoke tests
    patch.py            # trigger construction + application (fixed / random location)
    select.py           # poison index selection: gradient-norm and random
    craft.py            # the attack: alignment objective, signed Adam, retraining loop
    victim.py           # from-scratch victim training + ASR evaluation
    run_trial.py        # one (source, target, config) trial end to end
    run_ablation.py     # sweeps trials, writes results.jsonl
  analysis/
    summarize.py        # paired statistics + the final table
  results/
```

---

## 4. Algorithm specification

Notation follows the paper. `F(·; θ)` is the surrogate, `L` cross-entropy, `p` the trigger patch, `y_s` the source class, `y_t` the target class, `δ` the perturbations.

### 4.1 Surrogate training

Train `F` on **clean** CIFAR-10 to convergence using the victim recipe (§5). This model is the attacker's only knowledge of the victim.

### 4.2 Poison selection

Pick M = 500 indices from the training set.

- **Gradient-norm selection (the paper's method):** among images with label `y_t`, compute ‖∇θ L(F(x_i; θ), y_t)‖ for each at the trained surrogate's parameters, take the M largest.
- **Random selection (ablation control):** M uniformly at random from label `y_t`.

Per-example gradient norms are the expensive part. Computing the full parameter gradient per image is wasteful; the norm of the last-layer gradient is a standard cheap proxy and is what makes this tractable. **Implement the full-gradient version first** (it is unambiguous), measure its cost, and only switch to a proxy if it dominates runtime — and if you do, verify rank correlation against the full version on a few hundred images before trusting it.

### 4.3 The adversarial gradient

Sample K source-class **training** images. Apply the patch, relabel them `y_t`, and compute

```
grad_adv = ∇θ (1/K) Σ L(F(x + p; θ), y_t)
```

This is a constant with respect to δ — **detach it**. Recompute it whenever θ changes (i.e. after every retraining step), never otherwise.

Use K = 512 as a starting point. The paper does not specify K; it is a variance-vs-cost knob.

### 4.4 The alignment objective

```
grad_train = ∇θ (1/M) Σ L(F(x_i + δ_i; θ), y_i)      # y_i = y_t for all poisons
A = 1 - cosine_similarity(grad_train, grad_adv)
```

Minimize `A` with respect to δ.

**This is the one genuinely tricky piece.** `grad_train` must be built with `torch.autograd.grad(..., create_graph=True)` so that backpropagating `A` reaches δ through a second derivative. Flatten and concatenate all parameter gradients into a single vector before taking the cosine — a per-layer cosine averaged across layers is a *different and wrong* objective.

Memory note: the double-backward graph holds activations for all M poisons. 500 CIFAR images through ResNet-18 will not fit comfortably on a laptop GPU in one shot. Accumulate `grad_train` over minibatches of poisons while retaining the graph, or use gradient checkpointing. Get this right early — it determines whether the whole thing runs.

### 4.5 Optimizing δ

R = 250 steps of **signed Adam**: run Adam on δ but replace the gradient with its element-wise sign before the optimizer step.

After every step:
1. Clamp δ to the ε-ball: `δ.clamp_(-ε, +ε)`.
2. Clamp `x + δ` to the valid image range, then recover δ. Do this **in pixel space, before normalization** — a frequent and silent bug is enforcing the constraint in normalized space, which yields a perturbation that is not actually ε-bounded in the image.

> **Paper omits:** the learning rate and its decay schedule for signed Adam. Witches' Brew (Geiping et al. 2021), which this attack builds on, uses a base step size around 0.1 with step decay partway through crafting. Check the reference repo for the exact values and record what you used.

### 4.6 Differentiable data augmentation

Apply random crop (32×32, padding 4) and horizontal flip **inside** the crafting forward pass, differentiably, so gradients flow through the augmentation. The paper notes this follows Geiping et al. and improves poison stability. Implement crops as differentiable grid sampling or as index-shifts that preserve the graph; a non-differentiable torchvision transform silently breaks the gradient path to δ.

### 4.7 Patch application

- CIFAR-10 patch size: **8×8** (6.25% of pixels). Use a fixed random Bernoulli pattern, generated once per trial from the trial seed.
- **Random location** (default): sample a fresh uniform position every time the patch is applied during crafting.
- **Fixed location** (ablation row 1): always bottom-right.
- At **evaluation** time the location is random per image, independent of crafting.

The patch overwrites pixels; it is not blended.

### 4.8 Retraining loop

```
for r in 1..R:
    compute A, step signed Adam on δ
    if r mod floor(R / (T+1)) == 0 and r != R:
        retrain F on {(x_i + δ_i, y_i)}_{i≤M} ∪ {(x_i, y_i)}_{i>M}
        recompute grad_adv at the new θ
```

T = 4 for the full attack, T = 0 for the no-retraining ablation.

> **Paper ambiguity:** whether retraining continues from current weights or reinitializes. Appendix B.2 says CIFAR-10 models "train and retrain for 40 epochs," which reads as a fresh 40-epoch run. Check the reference repo; **this materially affects both results and runtime** (4 retrains × 40 epochs dominates the cost of a T=4 trial). Record the choice.

### 4.9 Victim evaluation

1. Freshly initialize a model (different seed from the surrogate).
2. Train 80 epochs on the poisoned training set, standard recipe, no attacker knowledge.
3. Report:
   - **Clean validation accuracy** — should match a clean baseline within ~0.5 points. If the attack tanks clean accuracy it is not stealthy and the result is uninteresting.
   - **Clean source-class accuracy** — unpatched source images, should stay high.
   - **ASR**: of held-out **validation** source-class images with the patch applied, the fraction classified as `y_t`. Use ~1000 patched source images.

ASR must be measured on **held-out** images. Measuring on the training-set source images used to compute `grad_adv` inflates the number and is the single easiest way to produce a meaningless result.

---

## 5. Hyperparameters (Appendix B.2)

| | |
|---|---|
| Architecture | ResNet-18 (CIFAR variant: 3×3 stem, no maxpool) |
| Optimizer | SGD, Nesterov momentum 0.9 |
| LR | 0.1 (ResNet); drops ×0.1 at epochs 14, 24, 35 |
| Weight decay | 4e-4 |
| Batch size | 128 |
| Epochs | 40 surrogate / 40 retrain / 80 victim |
| Augmentation | horizontal flip p=0.5, random crop 32 with padding 4 |
| ε | 16/255 |
| Poison budget | 500 images (1%) |
| Patch | 8×8 |
| Craft steps R | 250 |
| Retraining factor T | 4 |

---

## 6. Experiment design — read this before launching the sweep

**The variance is enormous and will dominate your conclusions if you ignore it.** The paper reports ±5.90 standard error over ~24 trials, which implies a standard deviation near 29 percentage points. Success depends heavily on which source/target class pair is drawn — the paper says so explicitly in §4.1 and again in §5.

Two consequences:

**Use a paired design.** Run every ablation arm on the *same* set of source/target pairs with the *same* seeds, and analyze paired differences. The paper does this ("Results averaged over the same source/target pairs"). Class-pair difficulty is the dominant variance component, and pairing removes it. Unpaired comparison at n=10 would tell you nothing.

**Do not try to resolve the small gaps.** Rows 3 and 4 differ by 4 points against a ~29-point spread. Separating them would take hundreds of trials. The honest findings available at this budget are the large effects:

- fixed → random patch location (+15)
- no retraining → retraining (+25)
- neither → both (+51)

Report the small gap as unresolved rather than over-claiming. *That is a finding*, and a better talking point than a number that matches the paper by luck.

### Compute budget

Estimate per trial on a 5070 Ti, assuming retraining runs fresh 40-epoch cycles:

| Phase | Cost |
|---|---|
| Surrogate (40 ep) | ~5 min |
| Crafting, 250 steps | ~5–10 min |
| Retraining, T=4 | ~20 min |
| Victim (80 ep) | ~10 min |
| **Total, T=4** | **~40 min** |
| **Total, T=0** | **~20 min** |

Measure this on trial one before committing to a sweep size. With 8 paired class pairs across 5 arms, expect roughly 20 hours — an overnight run plus a morning. Trim to 6 pairs, or drop the fixed-location arm (it is the least interesting), if that is too long.

Enable AMP (`torch.autocast` + `GradScaler`) for the victim and surrogate training loops. Be cautious about AMP during crafting — fp16 and double-backward through a cosine similarity interact badly; keep crafting in fp32 until it works, then try bf16.

### Checkpoint aggressively

Write `results.jsonl` incrementally, one line per completed trial, with the full config and all seeds. A sweep that dies at hour 14 and leaves nothing is the worst outcome here. Make `run_ablation.py` resumable.

---

## 7. Correctness checks

Build these **before** the sweep. Each catches a specific bug that otherwise produces plausible-looking wrong numbers.

1. **Clean-model control.** Train a clean model, measure ASR with the patch. Should be near zero (roughly the base misclassification rate into `y_t`). The paper confirms this in Appendix A.1. If a clean model shows meaningful "ASR," your evaluation is broken — most likely the patch is large or salient enough to act as a plain adversarial patch, which is a different phenomenon entirely.
2. **Perturbation budget assertion.** After crafting, assert `max|x_poisoned - x_clean| ≤ ε + 1e-6` in **pixel space**, and that exactly M images differ from clean.
3. **Label integrity assertion.** Assert no training label changed. This is what makes the attack clean-label.
4. **Trigger absence assertion.** Assert the patch pattern appears nowhere in the poisoned training set.
5. **Gradient-alignment telemetry.** Log cosine similarity between `grad_train` and `grad_adv` each craft step, and across victim epochs. It should rise during crafting and stay elevated during victim training; Figure 2 of the paper shows exactly this, and a flat curve means the objective is not doing its job. **This is your best single diagnostic** — it tells you whether the attack is failing because of a bug or because the class pair is hard.
6. **Random-δ control.** Train a victim on poisons that are random noise within the same ε-ball. ASR should be near zero. This separates "gradient matching works" from "any perturbation at this budget works."
7. **Visual check.** Save a grid of clean vs. poisoned images. They should be nearly indistinguishable (paper Figures 8, 11). If perturbations are obvious, the constraint is being applied in the wrong space.

---

## 8. Known pitfalls

- **Normalized vs. pixel space** for the ε-constraint. Covered above; the most common silent failure.
- **Non-differentiable augmentation** during crafting, severing the gradient to δ.
- **Per-layer instead of global cosine similarity.**
- **Forgetting to detach `grad_adv`**, producing a nonsense objective and possibly an OOM.
- **Stale `grad_adv` after retraining** — recompute at the new θ.
- **Evaluating ASR on training-set source images.**
- **Reusing the surrogate's seed for the victim**, which quietly converts a gray-box experiment into a same-model experiment and inflates ASR.
- **Blackwell/CUDA:** sm_120 needs a recent PyTorch built against CUDA 12.8+. Verify `torch.randn(10, device='cuda')` before a long run; older wheels install cleanly and then fail at the first kernel launch.
- **Counting ASR over all source images instead of the ones the clean model got right.** Pick one definition, state it, keep it consistent. The paper's "patched source val accuracy" and ASR are complementary views of the same evaluation.

---

## 9. Milestones

1. **Scaffolding.** Data, models, patch, victim training, evaluation. Verify a clean ResNet-18 hits ~92% on CIFAR-10 (paper: 92.31%). Nothing downstream is trustworthy until this baseline matches.
2. **Smoke-scale attack.** 2-class subset, ResNet-9, small budget, T=0. Get the double-backward working and the alignment cosine rising. Minutes per iteration — do all debugging here.
3. **Checks from §7** on the smoke config.
4. **One full-scale trial**, T=4, one class pair. Confirm plausible ASR and measure real runtime.
5. **Paired sweep** over class pairs × ablation arms. Resumable, checkpointed.
6. **Analysis.** Paired differences, confidence intervals, the table, the alignment-over-training plot.

---

## 10. The README is the deliverable

**The README is what gets presented, in 10 minutes, as the entire submission.** It is not setup documentation with results appended — it is the writeup, and the repo is its supporting evidence. Write it accordingly: someone should be able to read it top to bottom and understand what was built, what happened, and what was learned, without opening another file.

Setup and install instructions go at the **bottom**, or in a separate `SETUP.md`. They are the least interesting thing in the repo and must not occupy the top of the page.

### Required structure

**1. What this is — 3 sentences.** The attack in plain terms: a small fraction of training images are perturbed imperceptibly and keep their correct labels; a model trained on them behaves normally until a patch appears at test time, at which point source-class images are classified as the target class. Name the paper and link it.

**2. The result, immediately.** The ablation table, your numbers beside the paper's, with error bars. This is the headline and it goes above the fold. Do not make the reader scroll for it.

**3. Two figures.**
   - Clean vs. poisoned image grid — makes the stealth claim concrete in one glance, and takes no explanation.
   - Alignment cosine over victim training epochs, for clean / unsuccessful / successful runs. This is the mechanism picture (paper Figure 2) and the single most explanatory image available.

**4. Core code, inline.** Paste the ~30 lines that matter — the alignment objective (§4.4) and the retraining loop (§4.8) — directly into the README as a fenced block. Everything else is a standard training loop and should be linked, not shown. The assignment asks to "share core code"; reviewers will not go spelunking in `src/`, so bring it to them.

**5. Empirical intuitions — the section that carries the presentation.** Prose, not bullets of numbers. It should answer:
   - What actually made the attack work, and what turned out not to matter much.
   - How bad the variance was, and what that does to the comparisons. State plainly which effects were resolvable at this budget and which were not.
   - What broke during implementation and how it was diagnosed. The ε-constraint-in-the-wrong-space bug, the double-backward memory wall, whatever else came up. Debugging stories are the most interesting content here and the most likely to be asked about.
   - What the paper leaves unspecified (§4.5, §4.8) and what was chosen instead.
   - What you would do differently with more compute.

**6. What was not reproduced.** ImageNet, black-box transfer, defenses — scoped out, say so explicitly. An honest boundary reads as competence; an unstated gap reads as an oversight.

**7. Reproduction instructions.** Commands, config files, seeds, runtime. Brief.

### Tone

Claims should be sized to the evidence. "Retraining helped substantially (+25 points, paired across 8 class pairs)" is defensible. "We reproduced the paper" is not, and invites the one question you cannot answer. The variance finding in §6 is a genuine contribution — the paper acknowledges the instability but does not quantify how it constrains the ablation — so present it as a result, not an apology.

### Write it last, but outline it first

Have the README skeleton in place before the sweep launches, with the figures and table as placeholders. It determines what the experiments need to log — the alignment telemetry and the paired seeds exist *because* the README needs them. A sweep that finishes without the data its writeup requires has to be run again.
