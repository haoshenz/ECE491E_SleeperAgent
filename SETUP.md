# Setup and reproduction

The active schedule is now [the overnight plan](docs/OVERNIGHT_PLAN.md), ending Friday at 8:30 AM Hawaii time with compute stopping at 8:15 AM. Run `.venv/Scripts/python.exe -u scripts/overnight.py` to resume its exact queue. It waits for the current pilot to finish first. The earlier four-hour runner remains available for the original pilot; do not start a second controller while one is active.

## Environment

Use Python 3.12 and a CUDA 12.8-compatible NVIDIA GPU. Run commands at the repository root:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-lock.txt
.venv/Scripts/python.exe -c "import torch; print(torch.__version__); print(torch.randn(10, device='cuda'))"
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/download_data.py
```

`requirements-lock.txt` records the exact tested environment. `requirements.txt` lists direct dependencies. The verified hardware is an RTX 5070 Ti Laptop GPU with 12 GB GPU memory and 32 GB system RAM. Dataset files are downloaded from the official CIFAR-10 source and checksum-verified.

## Run and resume

```powershell
.venv/Scripts/python.exe -u scripts/run_all.py
```

This defaults to `scripts/run_four_hour.py`, which reads `configs/four_hour_run.yaml`. Its scientific settings come from `configs/cifar10_full.yaml`. Keeping those settings unchanged allows the existing full-attack result and model checkpoints to be reused.

The first run creates `results/four_hour/budget_clock.json`. The deadline persists across process restarts. No new compute starts unless its conservative estimate fits before the reporting reserve; active compute stops at the cutoff. The final 15 minutes are reserved for analysis and the README. If an experiment is incomplete, the report says so and retains its last checkpoint.

Training saves model, optimizer, scheduler, scaler, and random states every epoch. Crafting saves perturbations, the surrogate, adversarial gradient, Adam state, and step history every five updates and after each retrain. Completed trials are skipped. Shared clean and noise-control models are reused only when their settings/data agree. Do not run two experiment writers simultaneously.

## Live progress

Open `results/PROGRESS.md` or its Markdown preview and leave it open. The monitor starts with the runner and refreshes every 10 seconds. It shows the four selected experiments, every phase, the budget deadline/countdown, and final reporting tasks. `results/progress.json` provides the same data programmatically.

Logs are `results/run.log`, `results/run.err.log`, and `results/progress.err.log`. PIDs are recorded in `results/runner.pid` and `results/progress.pid`. The current job was launched hidden. Leave the laptop awake and connected to power. To stop it manually, stop the runner and its child processes in Task Manager, then use the same command to resume.

Short Windows file locks are retried. Display failures are isolated from training; the dashboard retries on its next refresh. Essential checkpoint write failures remain visible.

## Outputs and organization

| Location | Contents |
|---|---|
| `README.md` | Main presentation and measured result table |
| `reports/four_hour/` | Pilot report, figures, exact measurements and configs |
| `docs/FOUR_HOUR_PLAN.md` | Current scope and timeline |
| `results/four_hour/` | Budget clock, session outcomes, analysis and completion metadata |
| `results/full/` | Existing scientific trials and shared checkpoints reused by the pilot |
| `results/archive/` | Preserved logs from earlier attempts |
| `reports/smoke/` | Clearly labeled development evidence |

`results/`, `data/`, and `.venv/` are ignored by Git. Copy `results/` and optionally `data/` manually when moving machines; recreate `.venv` on the destination. Only load trusted local PyTorch checkpoints.

To rebuild the report from completed measurements:

```powershell
.venv/Scripts/python.exe scripts/report_four_hour.py
```

## Development and original study

The smoke configuration uses fewer classes, a smaller model, fewer epochs, and a 1.56% poison fraction. Its numbers test execution and cannot estimate paper-scale attack success:

```powershell
.venv/Scripts/python.exe -m src.run_ablation --config configs/smoke.yaml
```

The original 40-experiment study requires an explicit opt-in and is outside the four-hour budget:

```powershell
.venv/Scripts/python.exe -u scripts/run_all.py --full-sweep
```
