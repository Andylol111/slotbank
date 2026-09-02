# Conference evaluation protocol

This directory is the measurement plan the manuscript cites.
It does **not** invent Metal tok/s. The Linux workspace has no 27B weights.

| File | Role |
|---|---|
| `prompts.json` | 48-prompt speculation suite (code, prose, reasoning, repetitive, structured) |
| `pilot.json` | Existing Air measurements, labeled as a **pilot** (n=1 machine, small n) |
| `conference.py` | Stats over the pilot; refuses to fabricate Metal rows |
| `../scripts/conference_eval.py` | CLI: validate suite, print pilot tables |

## What a Metal run must collect

On the author's M4 Air (or any other unified-memory host):

1. **Speculation suite.** All 48 prompts. Greedy and the documented instruct pair separately. 256–1024 generated tokens. 3–5 reps. Record tok/s, accepted draft length, target forwards per emitted token, TTFT, decode-only.
2. **MoE $C$ sweep.** $C \in \{8,16,32,48,64,96\}$, 5+ alternating cold reps, report mean and range.
3. **Thermal.** Same greedy prefix at cool, 5 min, 10 min, 20 min soak.
4. **Second machine (optional).** Any other unified-memory capacity. The claim is that $C^\star$ moves when the leftover page cache can hold the expert bank.

Until those logs exist, `pilot.json` is a characterization, not the conference evaluation.

```
PYTHONPATH=src python3 scripts/conference_eval.py
```
