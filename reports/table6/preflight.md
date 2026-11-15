# Table 6 preflight

Launch allowed: True

Implemented CIFAR MobileNetV2/VGG11, checkpointed HTBA/CLBA, isolated runner, report, and dashboard.
All old shared-directory fingerprint rules are unchanged.
Baseline differences and source citations: docs/table6_baselines.md.

Protocol differences: Turner uses the requested clean surrogate rather than the reference adversarially trained checkpoint; HTBA uses a corrected greedy bijection; all cells use the same seeded random patch and random test-time positions. CLBA stamping is finally projected, so the training trigger is clipped.

Unit tests passed: True
Smoke cells passed: 9/9

- resnet18: 28.50 seconds/epoch, batch 128, 50,000 train + 10,000 test.
- mobilenetv2: 93.12 seconds/epoch, batch 128, 50,000 train + 10,000 test.
- vgg11: 8.24 seconds/epoch, batch 128, 50,000 train + 10,000 test.

Estimated new work: 18.38 h; with 15% margin: 21.14 h.
Includes 15% runtime margin; full 5,000 HTBA iterations even if loss threshold stops earlier.
One-epoch timing is an estimate, not a guarantee; laptop throttling can change speed.
Clean 80-epoch baseline and initial 40-epoch surrogate shared once per architecture.
Full-gradient selection and five source-gradient evaluations use measured timings.
Existing ResNet checkpoints are copied into the new namespace when compatible.

ResNet Sleeper reuse: {"reused": true, "differences": [], "source_path": "results\\full\\0-1_s17_full_6e499e05ba36\\result.json", "reason": "Every saved config key matches except output; new attack key only routes the new runner."}

Problems:
None
