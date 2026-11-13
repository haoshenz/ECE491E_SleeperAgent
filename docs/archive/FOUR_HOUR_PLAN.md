# Four-hour completion plan

The original 40-experiment sweep is archived. Its checkpoints and measurements are preserved. The smaller run started at **5:16 PM Hawaii time on November 12, 2026**, with a strict deadline of **9:16 PM**. New compute stops by **9:01 PM** to reserve reporting time.

## What will be delivered

A presentation-ready **single-pair pilot**, using airplane → automobile (classes 0 → 1), seed 17. It retains full CIFAR-10, the original model and training recipe, the 1% poison budget, the epsilon bound, held-out evaluation, and clean/random-noise controls.

| Configuration | State at start of window | Work scheduled in the window |
|---|---|---|
| Full attack | Complete | Reuse the measured result and controls |
| Fixed patch | Crafting and victim complete | Finish the noise control from epoch 15/80 |
| Random patch | Pending | Craft and train a victim; reuse the fixed arm's noise control |
| Random patch + selection | Pending | Select/craft and train a victim; reuse the full arm's noise control |

These four arms cover fixed versus random placement, selection, and the contribution of retraining when selection is enabled. The standalone retraining-only arm and the seven additional class pairs are outside the four-hour scope.

## Expected timeline within the window

| Elapsed time | Milestone |
|---|---|
| About 0–35 minutes | Finish fixed arm and shared noise control |
| About 35–85 minutes | Finish random-patch arm |
| About 85–135 minutes | Finish selection arm |
| About 135–165 minutes | Analysis, figures, and README; includes a safety margin |
| At 3 hours 45 minutes | Stop any remaining compute and finalize from completed results |
| By about 4 hours | Report ready, with unfinished phases explicitly marked |

The measured estimate when preparing this plan was about 121 minutes of remaining compute. A 25% compute margin plus 15 minutes for reporting gives approximately 166 minutes. These are estimates; the runner enforces a compute cutoff and preserves saved checkpoints if a phase cannot finish.

## Scientific limits

There is one source/target pair and one seed. Report measured ASR, clean accuracy, noise-control ASR, and single-pair differences. Standard errors and confidence intervals across class pairs are unavailable. This is a working pilot and a mechanism comparison, not the original eight-pair statistical reproduction. No completed results are discarded or invented to fit the deadline.

## Run or resume

The pilot is running. To resume after an interruption, use:

```powershell
.venv/Scripts/python.exe -u scripts/run_all.py
```

Scheduling is in `configs/four_hour_run.yaml`. Scientific settings remain in `configs/cifar10_full.yaml` so existing caches and checkpoints are reused. The four-hour wall-clock deadline is created on first resume and retained across process restarts. The runner reserves the last 15 minutes for the report, starts the live dashboard automatically, and marks skipped or interrupted experiments honestly.

Read `results/PROGRESS.md` for the revised four-experiment checklist and countdown. The report is in `reports/four_hour/README.md`; it is rebuilt from completed results when the pilot finishes. Returning to the original multi-day sweep requires the explicit `--full-sweep` option.
