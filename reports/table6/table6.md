# Table 6 comparison

one pair (airplane->automobile), one seed; not a statistical reproduction

## Attack success rate

| Attack | ResNet-18 | MobileNetV2 | VGG11 | Average |
|---|---:|---:|---:|---:|
| Saha (HTBA) | not run | not run | not run | not run |
| Turner (CLBA) | not run | not run | not run | not run |
| Sleeper Agent | 99.90% | not run | not run | not run |

## Clean test accuracy

| Attack | ResNet-18 | MobileNetV2 | VGG11 | Average |
|---|---:|---:|---:|---:|
| Saha (HTBA) | not run | not run | not run | not run |
| Turner (CLBA) | not run | not run | not run | not run |
| Sleeper Agent | 92.22% | not run | not run | not run |

## Protocol notes

- ASR uses all 1,000 source-class test images with uniformly random patch locations.
- Turner uses the requested clean surrogate; the reference benchmark supplies an adversarially trained surrogate.
- HTBA closest-feature assignment enforces a bijection; see docs/table6_baselines.md.
- Turner training trigger is clipped; both pre-trigger and final perturbations are bounded.
- Row average is across architectures, not independent pair/seed repetitions.

Exact settings and source citations: [baseline documentation](../../docs/table6_baselines.md).
