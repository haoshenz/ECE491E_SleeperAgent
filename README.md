# Sleeper Agent: an independent CIFAR-10 reproduction

We study whether tiny, correctly labeled changes to 1% of training images can teach a classifier a hidden response to a patch. The classifier should recognize ordinary images normally, while patched source-class images are predicted as a chosen target class. This repository implements the attack from scratch, following [Souri et al., NeurIPS 2022](https://arxiv.org/abs/2106.08970).

## Results and current evidence

<!-- RESULTS_START -->

# Overnight multi-pair ablation

Clean baseline patched ASR on original pair: **1.50% (15/1,000)**. New pairs reuse the same clean baseline/surrogate weights, with newly evaluated pair-specific clean controls.

| Configuration | Paper ASR Â± SE | Measured ASR Â± SE | n | Clean accuracy |
|---|---:|---:|---:|---:|
| fixed | 19.25 Â± 3.01% | 4.00 Â± 0.93% | 3 | 92.34% |
| random | 33.95 Â± 4.57% | 56.17 Â± 20.88% | 3 | 92.20% |
| retrain | 59.42 Â± 5.78% | pending | 0 | pending |
| select | 63.49 Â± 6.13% | 90.97 Â± 7.89% | 3 | 92.37% |
| full | 85.27 Â± 5.90% | 99.77 Â± 0.07% | 3 | 92.41% |

**Unequal n:** each row reports its own completed pair count; arms with different n cover different pair sets. Only matched pairs enter each difference below. One pair has no across-pair error bar.

- random minus fixed: +52.17 pp; matched n=3; 95% CI=[-38.96643525106311, 143.29977252327006].
- select minus random: +34.80 pp; matched n=3; 95% CI=[-40.98533419030911, 110.585340134879].
- full minus random: +43.60 pp; matched n=3; 95% CI=[-46.374838821383655, 133.57484467058612].
- full minus select: +8.80 pp; matched n=3; 95% CI=[-25.027171568361467, 42.627171472994036].

## Control status

- Clean baseline, pair 0â†’1, seed 17: patched ASR **1.50%**, clean accuracy 92.18%.
- Clean baseline, pair 9â†’5, seed 368152593: patched ASR **0.30%**, clean accuracy 92.18%.
- Clean baseline, pair 4â†’6, seed 1231470895: patched ASR **3.20%**, clean accuracy 92.18%.
- Pair A, gradient noise control: ASR 0.20%.
- Pair A, random noise control: ASR 0.10%.
- Pair B, gradient noise control: ASR 0.10%.
- Pair B, random noise control: ASR 0.60%.

## Queue outcomes

- 1: pair A, full, attack: complete.
- 2: pair B, full, attack: complete.
- 3: pair A, fixed, attack: complete.
- 4: pair A, random, attack: complete.
- 5: pair A, select, attack: complete.
- 6: pair B, fixed, attack: complete.
- 7: pair B, random, attack: complete.
- 8: pair B, select, attack: complete.
- 9: pair A, full, noise: complete.
- 10: pair A, fixed, noise: complete.
- 11: pair B, full, noise: complete.
- 12: pair B, fixed, noise: complete.

![Gradient alignment](reports/overnight/alignment.png)

Completed artifacts from the earlier pilot are preserved. No reduced-fidelity runs were added. The 28-test pilot verification is preserved; overnight changes have separate validation.


<!-- RESULTS_END -->

## What was built

The pipeline trains a clean surrogate, selects only target-class training images, crafts bounded pixel perturbations, and trains an independently initialized victim. The trigger is never written into victim training data. Its only uses are source-gradient estimation during crafting and held-out evaluation.

The selected configurations share the source/target pair, seeds, training orders, and evaluation patch positions. The clean-label constraint, exact number of changed images, pixel range, perturbation bound, and absence of the exact trigger pattern are checked before victim training. Clean and random-noise controls are included. The active four-hour pilot evaluates four configurations on one pair; the original five-arm, eight-pair design is archived.

## Two diagnostic figures

Each completed trial writes `poison_grid.png`: clean images, their poisoned counterparts, and a magnified difference. The analysis command writes `alignment.png`, showing global gradient alignment during crafting and victim training. These figures are generated from actual measurements; no illustrative result curves are substituted.

## Core code

The objective is one cosine over the concatenation of **all** parameter gradients:

```python
grad_train = torch.autograd.grad(
    cross_entropy(model(clean + delta), target_labels),
    tuple(model.parameters()), create_graph=True,
)
g = torch.cat([v.reshape(-1) for v in grad_train])
a = adversarial_gradient.detach()
loss = 1 - cosine_similarity(g[None], a[None])[0]
delta_gradient = torch.autograd.grad(loss, delta)[0]

for step in range(craft_steps):
    # Production code computes this exact derivative in two memory-bounded passes.
    cosine, delta_gradient = alignment_step(...)
    delta.grad = delta_gradient.sign()
    adam.step()
    project(delta, clean, epsilon)  # raw pixels, before normalization
    if step + 1 in retraining_events:
        surrogate = train_fresh_model(clean_training_data_with_delta)
        adversarial_gradient = recompute_source_gradient(surrogate).detach()
```

The executable implementation is [src/craft.py](src/craft.py). Standard training and evaluation are in [src/victim.py](src/victim.py).

## Implementation choices and empirical questions

**Memory.** Retaining double-backward graphs for 500 images can exceed laptop memory. Simply summing minibatch gradients with `create_graph=True` retains all those graphs. We instead compute the global training gradient without retaining graphs, differentiate the cosine analytically with respect to that gradient, then replay the exact same augmentations one minibatch at a time. A direct double-backward reference and directional finite differences verify this derivative. BatchNorm uses frozen running statistics during crafting, and standard training behavior during training.

**Differentiable augmentation.** Crop and flip use tensor indexing. The commonly suggested `grid_sample` does not generally supply the second derivative required here; index operations preserve the path to the perturbations.

**Reference checks, limited to the two omissions allowed by the plan.** In the [reference optimizer setup](https://github.com/hsouri/Sleeper-Agent/blob/master/forest/witchcoven/witch_base.py), the effective Adam rate is `tau * pbatch / 512`, with default `tau=0.1`; milestones are `R // 2.667`, `R // 1.6`, and `R // 1.142`, with factor 0.1. We choose normalized-coordinate LR **0.025** for the full experiment (nominal pbatch 128), independent of the memory minibatch size. Each Adam update is multiplied by its channel standard deviation to recover the equivalent pixel-space update; delta and all projection constraints remain in raw pixels. A test compares this against running Adam directly on normalized perturbations. The smoke configuration uses pixel-coordinate LR 0.1. The same reference file and [options](https://github.com/hsouri/Sleeper-Agent/blob/master/forest/options.py) confirm fresh initialization for the default retraining scenario. No reference attack code is imported or wrapped.

**Other choices.** K=512 training-source examples estimate the detached adversarial gradient, which refreshes only after surrogate weights change. Evaluation patch positions are random in every arm, including fixed-location crafting. ASR uses **all held-out source images**; ASR conditional on correct clean classification is recorded separately. Alignment telemetry during victim training uses a fixed 16-image training subset in full runs to limit overhead, whereas crafting uses the complete poison set. The training LR milestones remain 14, 24, 35 even for the 80-epoch victim, following the supplied plan.

**Variance.** The pilot has one paired trial per configuration, so standard errors and confidence intervals across class pairs are unavailable. Differences describe this one pair. The original eight-pair design is preserved for future work; its analysis code supports Student-t intervals when repeated paired trials are available. The present results do not resolve variation across class pairs.

**More compute.** Additional paired class pairs would give stronger evidence about variance and effect sizes before expanding to retraining-factor or epsilon sweeps.

## Scope

This is a time-limited pilot of Table 10 on CIFAR-10. The standalone retraining-only arm, additional class pairs, ImageNet, transfer between architectures, defenses, and epsilon/retraining-factor sweeps are outside this submission's measurements. A successful attack on one pair does not establish numerical reproduction of the paper's averages.

## Reproduction

Run `.venv/Scripts/python.exe -u scripts/run_all.py` to run or resume the pilot. See [SETUP.md](SETUP.md) for environment setup and checkpoints, [the four-hour plan](docs/OVERNIGHT_PLAN.md) for the schedule, and [the original specification](docs/archive/SLEEPER_AGENT_PLAN.md) for the archived full study. Exact configurations and measurements are exported under `reports/four_hour/`; local model checkpoints remain under `results/`.
