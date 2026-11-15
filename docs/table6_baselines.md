# Table 6 benchmark implementation and limits

This extension compares three attacks on CIFAR-10 ResNet-18, MobileNetV2, and
VGG11, using one pair (airplane->automobile), one seed; not a statistical reproduction.
All nine cells use source 0, target 1, seed 17, and the same Bernoulli RGB 8x8
patch generated with seed 18. Outputs are isolated in `results/table6/` and
`reports/table6/`. The older main-study results are not changed.

## Sources

The implementations are independently written for this repository, using the
benchmark's architecture definitions and attack algorithms/settings as references.
They do not invoke the benchmark repository as a library or wrap its scripts.
The local reference checkout is pinned to commit
`97a67ae5065b9ef1df946ed10ff5094fcee91bbc` of
[Schwarzschild et al.'s official poisoning benchmark](https://github.com/aks2203/poisoning-benchmark).

- [Sleeper Agent, section 4.2 and Appendix B.3](https://arxiv.org/html/2106.08970v3):
  competing-attack benchmark, 1% poison budget, epsilon 16/255, and adjusted
  Turner opacity/perturbation constraints.
- [Saha implementation](https://github.com/aks2203/poisoning-benchmark/blob/97a67ae5065b9ef1df946ed10ff5094fcee91bbc/poison_crafting/craft_poisons_htbd.py):
  `adjust_learning_rate`, feature matching, gradient descent, patch placement,
  early stopping, and parser defaults.
- [Turner implementation](https://github.com/aks2203/poisoning-benchmark/blob/97a67ae5065b9ef1df946ed10ff5094fcee91bbc/poison_crafting/craft_poisons_clbd.py):
  `AttackPGD`, patch stamping, final perturbation projection, and parser defaults.
- [MobileNetV2 topology](https://github.com/aks2203/poisoning-benchmark/blob/97a67ae5065b9ef1df946ed10ff5094fcee91bbc/models/mobilenetv2.py):
  stride-one CIFAR stem and second group, projected stride-one shortcuts, ReLU,
  expansion groups, 4x4 pooling, and 1280->10 head.
- [VGG11 topology](https://github.com/aks2203/poisoning-benchmark/blob/97a67ae5065b9ef1df946ed10ff5094fcee91bbc/models/vgg.py):
  eight biased convolutions with batch normalization, five max pools, and
  512->10 head. Our non-in-place ReLU preserves second derivatives.

The benchmark is MIT licensed, copyright (c) 2020 Avi Schwarzschild, Arjun Gupta.
See its [license](https://github.com/aks2203/poisoning-benchmark/blob/97a67ae5065b9ef1df946ed10ff5094fcee91bbc/LICENSE.txt).

## Exact attack settings

| Setting | Saha / HTBA | Turner / CLBA | Sleeper Agent |
|---|---|---|---|
| Poison count / labels | 500 target-class images; unchanged labels | Same | Same |
| Perturbation bound | 16/255 | 16/255 before **and after** stamping | 16/255 |
| Poison selection | Random target-class training images | Same random indices as HTBA | Largest full-parameter-gradient norms |
| Initial perturbation | Zero | Uniform in [-epsilon, epsilon] | Uniform in [-epsilon, epsilon] |
| Objective | Sum of squared penultimate feature differences | Maximize true-label cross entropy | One global gradient cosine |
| Maximum steps | 5,000 | 20 | 250 |
| Update | Ordinary input-gradient descent, LR .001 | PGD sign ascent, step 2/255 | Existing signed Adam, normalized-coordinate LR .025 |
| LR drops | .95^(floor(step/2000)), zero-based step | None | Existing steps 93, 156, 218, factor .1 |
| Early stop | Global summed feature loss below 10 | None | None |
| Crafting patch position | Inset 5 pixels from bottom-right (19,19) | Bottom-right (24,24) | Uniform random |
| Training trigger | Not stamped | Stamped then projected into epsilon ball | Not stamped |
| Surrogate refreshes | None | None | Four fresh 40-epoch models at steps 50,100,150,200 |
| Craft microbatch | 128 | 128 | 16 (two-pass second derivative) |

The HTBA source pool consists of 500 distinct source-class **training** images
drawn from the same seeded 512-image pool used by Sleeper Agent. Source and poison
features are greedily matched by closest available pair each optimization step.
The squared-feature loss is a global **sum**, not a mean over microbatches.
BatchNorm is frozen during crafting, so microbatching does not change the
mathematical objective. The microbatch size is an implementation memory choice;
the benchmark HTBA script actually uses all bases as a batch, and its CLBA script
uses batches of up to 1,000. It is not a reduction in steps or data.

CLBA requests full-opacity stamping (opacity 1), **then clips the complete final
perturbation**, as the reference code does. Consequently training patches are
image-dependent clipped versions of the trigger, not exact binary 8x8 copies.
There is no undocumented scalar opacity taken from the paper: the paper says it
adjusts opacity but does not give its precise value. Both pre-trigger PGD and
final stamped images are checked against epsilon; we do not allow a large visible
patch to evade the final perturbation bound. Only CLBA skips the exact-trigger
absence scan, and its final patch pixels are checked against the stamping formula.

## Training and evaluation

Each architecture shares its own clean 40-epoch surrogate and clean 80-epoch
baseline across the three attacks. All attacked victims start from fresh weights
and train for 80 epochs. Learning rates: ResNet-18 .1; MobileNetV2 and VGG11 .01.
Other training settings follow `configs/cifar10_full.yaml`, including SGD/Nesterov,
weight decay, augmentation, and learning-rate milestones. Standard training uses
AMP; poison crafting uses FP32. Inputs are normalized once in the model, using
the existing main study's CIFAR statistics.

All cells evaluate all 10,000 unmodified test images for clean accuracy and all
1,000 airplane test images for ASR, with a uniformly random patch position and
fixed evaluation seed 717. No test images enter selection or crafting. This is a
class-wide evaluation, not the benchmark repository's single targeted test image.
For clean baselines the existing 91.5% accuracy gate is retained for ResNet-18;
the new architectures use an 80% minimum gate, below the paper's approximately
88-89% clean accuracy, to catch failed training. Actual accuracy is always reported.
No extra random-noise controls are scheduled in this nine-cell extension; the
original study's controls and results remain separate.

## Explicit differences from an exact Table 6 reproduction

1. **One pair and one seed**, not the paper's collection of independent runs.
   Row averages are across the three architectures; they are not uncertainty estimates.
2. The requested **shared clean surrogate** is used for CLBA. The benchmark
   CLBA default points to `ResNet18_CIFAR10_adv.pth`, an adversarially trained
   checkpoint. We do not silently substitute it or claim an exact Turner replication.
   An architecture-specific adversarial-surrogate replication would need separate
   models, code, and compute beyond this approved clean-surrogate protocol.
3. HTBA uses a **true greedy bijection**. The reference loop masks only one
   distance-matrix entry after assignment and can repeatedly reuse rows/columns;
   this implementation masks used source/poison indices. This correction is
   documented rather than presented as byte-for-byte reference behavior.
4. The same seeded random patch is used for all cells, retaining compatibility
   with the existing airplane->automobile trial. The paper's primary experiments
   use a reference patch; Appendix A.2 also studies random patches.
5. Final CLBA projection follows the public benchmark; the paper's exact scalar
   opacity adjustment is not specified. ResNet and normalization preserve this
   repository's prior implementation; the added architectures match benchmark
   topology (checked numerically after transferring module weights).
6. Uniform-random test-time patch positions preserve the main study's evaluation.
   The public baseline scripts store a fixed patch coordinate for their single
   test target. Fixed-position crafting remains method-specific as listed above.

## Running, stopping, and resuming

Use `.venv/Scripts/python.exe scripts/table6_preflight.py` for unit tests, all nine
small real-architecture smoke cells, full-data one-epoch timings, crafting timings,
reuse comparison, and a conservative compute estimate. No full run is permitted
unless every check passes and the estimate, including a 15% margin, is <=22 hours.
Preflight results are bound to a hash of implementation, tests, and configurations.

Run `.venv/Scripts/python.exe scripts/run_table6.py` after preflight. It runs six
baseline cells, then Sleeper Agent VGG11, MobileNetV2, and ResNet-18 if not reused.
It updates root `progress.md` every 15 seconds and rebuilds the partial report
after every completed cell. Missing cells are explicitly `not run`; incomplete
rows have no average. Reused cells retain their original provenance.

For a detached Windows launch, use `pythonw.exe` and
`--log-file results/table6/run.log`. This appends stdout and stderr to one log.
Only one runner can own the process lock. Create `results/table6/STOP` to request
a safe stop **after the current cell finishes**. The request is consumed when
acknowledged. Rerun the same command to skip finished cells and resume checkpoints.
Stopping the foreground process with Ctrl+C also preserves existing checkpoints;
an interrupted current epoch is repeated. Baseline crafting checkpoints every five
steps; training checkpoints every epoch. The existing Sleeper crafting checkpoint
also records model, optimizer, and refresh state.

The 22-hour threshold is checked between cells: the current cell completes, then
the runner stops before starting another. This is a soft boundary, not a guarantee
against exceeding 24 hours if a cell unexpectedly slows down. Budget usage is
conservative active runner wall time, including CPU work, checkpointing, and
resumed invocations; preflight is separately reported and not charged to the
24-hour full-run allowance. Catchable failures write CRASHED plus the error;
power loss or force-killing a process cannot execute a final dashboard write.

The old fingerprint sentinel remains `results/full/shared/527ca756e205` for the
unchanged full config with source 0, target 1, seed 17. Existing shared checkpoint
files, when compatible, are **copied** into `results/table6/shared/` before any
training routine loads them, preventing accidental writes to old caches.
