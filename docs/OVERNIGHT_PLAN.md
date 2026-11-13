# Overnight multi-pair ablation

Hard stop: **Friday November 13, 2026, 8:30 AM Hawaii time**. Compute stops at **8:15 AM**; the final 15 minutes are reserved for analysis and reporting. The previous evening deadline is removed.

The current four-arm pilot on 0→1 finishes first without interruption. The overnight controller waits for its results and report before starting any new compute.

## Fixed random draw

Uniform sampling without replacement from the 90 ordered source/target pairs, excluding 0→1; sampling seed **1536078799**:

- A: **9→5 (truck→dog)**, trial seed **368152593**.
- B: **9→7 (truck→horse)**, trial seed **1231470895**.

Pair B was redrawn before any new experiment started, at the user?s request to avoid repeating truck. Conditional redraw seed: **2158728332**; eligible classes: [2, 3, 4, 6, 7, 8], disjoint from the original pair and A. The original draw is retained in the configuration history.

The draw is recorded in `configs/overnight_run.json` and is not selected using outcomes.

## Exact queue

1. Finish existing pilot, including experiments 3 and 4.
2. Full attack on A.
3. Full attack on B.
4. Fixed, random, selection arms on A, in that order.
5. Fixed, random, selection arms on B, in that order.
6. Noise controls last: A gradient-selected, A random-selected, B gradient-selected, B random-selected. Each selection type supplies the corresponding two arms.

No reordering for efficiency or reduction of fidelity. New full arms use 250 updates, T=4, four fresh 40-epoch surrogate refreshes, 80-epoch victims, 16/255 epsilon, 8×8 patch, and 500 poisons. Other arms retain their original ablation settings. Clean and initial surrogate checkpoints are loaded read-only from `results/full/shared/527ca756e205`; clean ASR is reevaluated for each new pair and patch.

Existing completed trials and model checkpoints are preserved. New trials live under `results/overnight/trials`, with separate noise-control results. Attacks completed before the noise stage are explicitly marked as having deferred controls. Every training epoch and crafting checkpoint remains resumable. Display callbacks are disabled in overnight compute workers; the dashboard runs separately.

## Reporting

Pool only identical per-arm settings. State n for every arm; report matched n for each paired difference. Unequal arm coverage is explicit. Incomplete jobs and unrun controls remain incomplete. The original clean baseline patched ASR is **1.50% (15/1,000)** and is already surfaced in the README.

Current status: `results/PROGRESS.md`. Overnight log: `results/overnight/run.log`. Final report: `reports/overnight/README.md`. The controller waits for the current pilot automatically:

```powershell
.venv/Scripts/python.exe -u scripts/overnight.py
```
