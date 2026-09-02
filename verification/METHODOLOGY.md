# Verification methodology

This file is the standing protocol. Dated runs live next to it as
`campaign_<UTC>.md` / `.json`. The manuscript cites `campaign_latest.md`.

## What a run is allowed to claim

| Kind | Allowed claim in the paper |
|---|---|
| `verified_here` | Reproduced on the campaign host. |
| `air_pinned` | Measured on the author's M4 Air; pins must not drift. |
| `cannot_retime` | Named gap. Do not upgrade to a result. |

## What a run is not

- Not a Metal retime of 27B tok/s.
- Not a task-quality study of the envelope versus the full OMP harness.
- Not a fanless soak.
- Not permission to mark a rejected catalog id as adopted.

## Research questions

### RQ1. Decode and prefill clocks

On a fanless 24 GB M4 Air, what named clocks does Qwen3.8-27B-4bit actually hit, and how do they sit against bandwidth-scaled Max numbers and the 2026 +200% Mac headline?

### RQ2. Hybrid-safe speculation

Can speculative decode raise tok/s without changing 27B text when 48/64 layers are untrimmable Gated DeltaNet?

### RQ3. Working-set fit

What leave-free and slot-capacity rules admit 27B dense plus sidecar, or 35B-A3B experts, on 24 GB without pretending the page cache is VRAM?

### RQ4. Fail-closed policy

Can admission, catalog, and OMP policy be checked on a machine that cannot pin 15 GiB, and do those checks refuse routes that change tokens?

### RQ5. Envelope structure

Does the local envelope keep system≤256, packed ids≤8192, and a history-stable prefix on OMP-shaped dumps, without claiming task quality?

## Independent / dependent variables

- IV: prompt class (short / file dump / cwd dump / 26k / 39k), turn (first vs follow-up), leave-free (6g vs 8g), catalog status, draft trained K, temperature relative to 0.99
- DV: system tokens after condense, packed id length, stable_prefix_n, admit.ok, catalog_sound, pin equality, session_ok
- Held out: task quality of condensed vs full OMP harness (C-quality); fanless soak tok/s (C-soak); live Metal 27B tok/s on this VM (C-metal-toks)

## Command

```
PYTHONPATH=src python3 scripts/verify_paper.py
```
