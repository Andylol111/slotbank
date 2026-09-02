# Paper verification trail

This directory is the methodology log for the systems manuscript.
It is meant to be read with the paper, not instead of it.

| File | What it is |
|---|---|
| `METHODOLOGY.md` | Standing protocol: research questions, what a run may claim, what it may not |
| `campaign_latest.md` | Latest dated run (human) |
| `campaign_latest.json` | Same run (machine) |
| `suite_latest.json` | Envelope suite rows from that run |
| `campaign_<UTC>.*` | Immutable copies of each campaign |
| `pytest_*_latest.txt` | Raw pytest tails |

## Reproduce

```
PYTHONPATH=src python3 scripts/verify_paper.py
```

## How to read a claim

- `verified_here` — reproduced on the campaign host.
- `air_pinned` — measured 2026-08-31 on the author's M4 Air; tests fail if the pin drifts.
- `cannot_retime` — needs the 27B Metal process or a soak. Named so the paper cannot quietly promote it.

Live Metal tok/s are not retimed on the Linux workspace. Task quality of the condensed prompt versus the full OMP harness is not measured here.
