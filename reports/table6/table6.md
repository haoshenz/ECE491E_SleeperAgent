# Table 6 comparison

one pair (airplane->automobile), one seed; not a statistical reproduction

## Attack success rate

| Attack | ResNet-18 | MobileNetV2 | VGG11 | Average |
|---|---:|---:|---:|---:|
| Saha (HTBA) | 2.50% | 2.40% | 11.20% | 5.37% |
| Turner (CLBA) | 0.90% | 3.70% | 5.00% | 3.20% |
| Sleeper Agent | 99.90% | 27.90% | 98.40% | 75.40% |

## Clean test accuracy

| Attack | ResNet-18 | MobileNetV2 | VGG11 | Average |
|---|---:|---:|---:|---:|
| Saha (HTBA) | 92.51% | 88.80% | 89.37% | 90.23% |
| Turner (CLBA) | 92.36% | 88.79% | 88.96% | 90.04% |
| Sleeper Agent | 92.22% | 88.54% | 88.99% | 89.92% |

## Protocol notes

- ASR uses all 1,000 source-class test images with uniformly random patch locations.
- Turner uses the requested clean surrogate; the reference benchmark supplies an adversarially trained surrogate.
- HTBA closest-feature assignment enforces a bijection; see docs/table6_baselines.md.
- Turner training trigger is clipped; both pre-trigger and final perturbations are bounded.
- Row average is across architectures, not independent pair/seed repetitions.

Exact settings and source citations: [baseline documentation](../../docs/table6_baselines.md).
