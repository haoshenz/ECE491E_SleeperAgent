# Table 6 comparison

one pair (airplane->automobile), one seed; not a statistical reproduction

Inference-only re-scoring of the same saved victims; no retraining. All nine RANDOM scores and clean accuracies matched their original cell results.

## Main table: native test protocol

HTBA: fixed inset bottom-right (row 19, column 19). CLBA: fixed bottom-right (24,24). Sleeper Agent: random position. Coordinates are zero-based.

| Attack | ResNet-18 | MobileNetV2 | VGG11 | Average |
|---|---:|---:|---:|---:|
| Saha (HTBA) | 4.90% | 2.70% | 10.60% | 6.07% |
| Turner (CLBA) | 0.90% | 1.10% | 6.60% | 2.87% |
| Sleeper Agent | 99.90% | 27.90% | 98.40% | 75.40% |

## Extra table: all attacks, random trigger position

All methods use uniformly random positions, seed 717, and the original batch size of 128.

| Attack | ResNet-18 | MobileNetV2 | VGG11 | Average |
|---|---:|---:|---:|---:|
| Saha (HTBA) | 2.50% | 2.40% | 11.20% | 5.37% |
| Turner (CLBA) | 0.90% | 3.70% | 5.00% | 3.20% |
| Sleeper Agent | 99.90% | 27.90% | 98.40% | 75.40% |

Sleeper Agent with the trigger FIXED at (24,24): resnet18: 96.50%, mobilenetv2: 4.00%, vgg11: 39.10%.

## Clean test accuracy

All 10,000 test images, without a trigger. Clean accuracy is identical in both re-scoring passes.

| Attack | ResNet-18 | MobileNetV2 | VGG11 | Average |
|---|---:|---:|---:|---:|
| Saha (HTBA) | 92.51% | 88.80% | 89.37% | 90.23% |
| Turner (CLBA) | 92.36% | 88.79% | 88.96% | 90.04% |
| Sleeper Agent | 92.22% | 88.54% | 88.99% | 89.92% |

## Protocol notes

- Both protocols use the same 1,000 held-out airplane images; ASR is the fraction classified as automobile.
- The main table uses the requested method-specific native test placement. HTBA/CLBA fixed coordinates follow the public benchmark. Other documented differences from the paper remain.
- CLBA used a clean surrogate, not an adversarially trained surrogate.
- HTBA uses the documented corrected greedy feature bijection; the training trigger for CLBA is clipped into the perturbation bound.
- The ResNet-18 Sleeper Agent checkpoint is reused from the original study, following the saved provenance.
- Averages are across architectures, not independent pair/seed repetitions. No uncertainty estimates are inferred.
- Original result.json files and reports/table6/summary.json are unchanged; the latter retains the original random-position results.

Re-scoring time: 24.80 seconds. Completed 2026-11-15T17:27:16.538618-10:00.

Detailed scores and SHA-256 input provenance: [rescored.json](rescored.json).
Attack settings and remaining benchmark differences: [baseline documentation](../../docs/table6_baselines.md).
