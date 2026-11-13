# Original study design

`SLEEPER_AGENT_PLAN.md` is the preserved original specification for the five-arm, eight-pair Table 10 study. It requires several days on this laptop. The active submission is the smaller pilot described in [the four-hour plan](../FOUR_HOUR_PLAN.md).

The original code paths and checkpoints remain available. To explicitly opt into the original study:

```powershell
.venv/Scripts/python.exe -u scripts/run_all.py --full-sweep
```

This command is outside the four-hour budget. The default runner executes the four-hour pilot instead.
