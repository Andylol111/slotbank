# Verification campaign 20260902T001357Z

Traceability log for the slotbank systems manuscript.
Air tok/s are pinned constants from `M4_AIR_24G`, measured 2026-08-31.
This host has no Metal 27B weights; live decode is not retimed here.

## Host

- OS: Linux x86_64
- Python: 3.12.3
- Metal 27B: absent
- UTC: 2026-09-02T00:14:02.145422+00:00

## Method

- Independent variables: prompt class (short / file dump / cwd dump / 26k / 39k), turn (first vs follow-up), leave-free (6g vs 8g), catalog status, draft trained K, temperature relative to 0.99
- Dependent variables: system tokens after condense, packed id length, stable_prefix_n, admit.ok, catalog_sound, pin equality, session_ok
- Not measured: task quality of condensed vs full OMP harness (C-quality); fanless soak tok/s (C-soak); live Metal 27B tok/s on this VM (C-metal-toks)
- Oracle: Bit-identity is not claimed here. Structural oracles are token-budget caps, exact prefix equality, catalog fail-closed, and pin drift.

Three pytest invocations:

1. **Paper + fence** — every claim id; writes suite JSON.
2. **Filtered tree** — project CI filter (optional deps dropped).
3. **Full tree** — records the seven optional-dep failures so they are not silent.

## Research questions

- **RQ1 (Decode and prefill clocks).** On a fanless 24 GB M4 Air, what named clocks does Qwen3.8-27B-4bit actually hit, and how do they sit against bandwidth-scaled Max numbers and the 2026 +200% Mac headline?
- **RQ2 (Hybrid-safe speculation).** Can speculative decode raise tok/s without changing 27B text when 48/64 layers are untrimmable Gated DeltaNet?
- **RQ3 (Working-set fit).** What leave-free and slot-capacity rules admit 27B dense plus sidecar, or 35B-A3B experts, on 24 GB without pretending the page cache is VRAM?
- **RQ4 (Fail-closed policy).** Can admission, catalog, and OMP policy be checked on a machine that cannot pin 15 GiB, and do those checks refuse routes that change tokens?
- **RQ5 (Envelope structure).** Does the local envelope keep system≤256, packed ids≤8192, and a history-stable prefix on OMP-shaped dumps, without claiming task quality?

## pytest

| Run | passed | failed | skipped | deselected | returncode |
|---|---:|---:|---:|---:|---:|
| paper+fence | 29 | 0 | 0 | 0 | 0 |
| filtered CI | 221 | 0 | 23 | 7 | 0 |
| full tree | 221 | 7 | 23 | 0 | 1 |

Full-tree failures are the named optional-dep tests:

- `tests/test_cli.py::test_resolve_passes_through_explicit_ids`
- `tests/test_cli.py::test_hub_progress_receives_the_name_kwarg`
- `tests/test_cli.py::test_hub_progress_forces_disable_off`
- `tests/test_cli.py::test_plain_progress_has_no_escape_codes_and_is_throttled`
- `tests/test_cli.py::test_download_view_never_paints_over_a_pipe`
- `tests/test_kv_quant.py::test_shed_keeps_dflash_session`
- `tests/test_kv_quant.py::test_strip_unused_drops_vision`

```
............................                                             [100%]
=============================== warnings summary ===============================
../home/ubuntu/.local/lib/python3.12/site-packages/fastapi/testclient.py:1
  /home/ubuntu/.local/lib/python3.12/site-packages/fastapi/testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
221 passed, 23 skipped, 7 deselected, 1 warning in 1.53s
```

## Claim matrix

| Id | RQ | Kind | Campaign | Test | Statement |
|---|---|---|---|---|---|
| `C-fence` | RQ4 | verified_here | verified_here | `tests/test_fence.py::test_only_three_files_import_mlx` | Only runtime.py, expert_slots.py, offload_cache.py import MLX. |
| `C-catalog` | RQ4 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_catalog` | Fail-closed catalog: banned ids cannot be adopted; adopted routes keep 27B text. |
| `C-clocks` | RQ1 | air_pinned | pin_checked | `tests/test_paper_verification.py::test_C_clocks` | Named Air clocks: 5.71 greedy, 13.47/9.95 MTP, 11.76/9.10 DFlash, 17.0s/0.88s prefill. |
| `C-pack-ceil` | RQ1 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_pack_ceil` | 4-bit pack-read ceiling on 120 GB/s is in [7, 9] tok/s. |
| `C-mac200` | RQ1 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_mac200` | No official Qwen Mac tok/s row; +200% dense-27B Mac hit is ANE prefill, not this Air. |
| `C-scale` | RQ1 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_scale` | oMLX M3 Max decode scaled by 120/400 is 5.94 / 9.78, near the pinned Air rates. |
| `C-hybrid-kv` | RQ2 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_hybrid_kv` | Qwen3.8 hybrid full-attn KV is 64 KiB/token; can_trim path is refused without dflash verify. |
| `C-trained-k` | RQ2 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_trained_k` | Draft block is trained K (MTP 3, DFlash 8), not a forced 5. |
| `C-leave-free` | RQ3 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_leave_free` | 24 GB default leave-free is 8 GiB; 27B 4-bit + sidecar needs 6 GiB leave-free (18 GiB WS). |
| `C-slots-c` | RQ3 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_slots_c` | 35B-A3B on 24 GB picks C in (16, 256); OLMoE stays at the 2×top-k floor. |
| `C-sys-cap` | RQ5 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_sys_cap` | Envelope system head is 256 tokens on every turn, including follow-up. |
| `C-history` | RQ5 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_history` | User-dump condense is the same recipe as last-ask and as history. |
| `C-prefix` | RQ5 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_prefix` | Follow-up encode shares stable_prefix_n with the first encode (pre /no_think body). |
| `C-pack-8k` | RQ5 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_pack_8k` | After envelope, Metal prompt ids are ≤ 8192; 26k/39k dumps do not refuse. |
| `C-pyramid` | RQ5 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_pyramid` | keep_token_ids is sink + pyramid middle + tail; 8k head lands on 2048. |
| `C-slim` | RQ5 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_slim` | slim_tools drops JSON schemas and stays inside the 256-token tool budget. |
| `C-inject` | RQ5 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_inject` | Serve envelope does not compile CONTEXT_DIR back into the prefix unless inject=1. |
| `C-context` | RQ5 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_context` | Context OS log is append-only; compile is newest verbatim spans, not a paraphrase. |
| `C-session` | RQ2 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_session` | dflash_session_ok refuses attention offset past \|fed\|. |
| `C-sampling` | RQ2 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_sampling` | OMP temperature ≥ 0.99 remaps to Qwen's documented instruct pair; T<0.99 is left alone. |
| `C-suite` | RQ5 | verified_here | verified_here | `tests/test_paper_verification.py::test_C_suite` | Envelope prompt suite: every case keeps system≤256, packed≤8192, history-stable prefix. |
| `C-quality` | RQ5 | cannot_retime | documented_gap | `tests/test_paper_verification.py::test_C_cannot_retime_are_documented` | Task quality of the condensed 8k prompt vs the full OMP harness (needs 27B + agent tasks). |
| `C-soak` | RQ1 | cannot_retime | documented_gap | `tests/test_paper_verification.py::test_C_cannot_retime_are_documented` | Fanless soak 60–70% of cool decode after 8–10 min (needs the Air under load). |
| `C-metal-toks` | RQ1 | air_pinned | pin_checked | `tests/test_paper_verification.py::test_C_air_pinned_are_documented` | Live Metal tok/s on Qwen3.8-27B-4bit; this VM has no weights. |

## How to read kind / campaign status

- `verified_here` — reproduced on this host by the paper tests.
- `air_pinned` / `pin_checked` — measured 2026-08-31 on the author's M4 Air; tests fail if the pin drifts.
- `cannot_retime` / `documented_gap` — needs the 27B process or a soak; listed so the paper does not pretend otherwise.

## Envelope suite (structural, not task quality)

n = 29. Classes: cwd, edge, file, omp-26k, omp-39k, short.
Every case asserts system ≤ 256 tokens, packed ids ≤ 8192, and a history-stable prefix across first / follow-up / third turn.
This is not SWE-bench. `C-quality` remains a documented gap.

| Id | class | raw user tok | sys tok | prefix n | first ids | packed |
|---|---|---:|---:|---:|---:|---:|
| `short-0` | short | 1 | 40 | 162 | 175 | 8192 |
| `file-0` | file | 1007 | 244 | 5002 | 5015 | 8192 |
| `cwd-0` | cwd | 787 | 124 | 560 | 573 | 8192 |
| `short-1` | short | 6 | 40 | 181 | 194 | 8192 |
| `file-1` | file | 1262 | 244 | 6022 | 6035 | 8192 |
| `cwd-1` | cwd | 1112 | 124 | 579 | 592 | 8192 |
| `short-2` | short | 5 | 40 | 180 | 193 | 8192 |
| `file-2` | file | 1512 | 244 | 7021 | 7034 | 8192 |
| `cwd-2` | cwd | 1432 | 124 | 578 | 591 | 8192 |
| `short-3` | short | 7 | 40 | 185 | 198 | 8192 |
| `file-3` | file | 1763 | 244 | 8026 | 8039 | 8192 |
| `cwd-3` | cwd | 1753 | 124 | 583 | 596 | 8192 |
| `short-4` | short | 6 | 40 | 181 | 194 | 8192 |
| `file-4` | file | 2012 | 244 | 9022 | 9035 | 8192 |
| `cwd-4` | cwd | 2072 | 124 | 579 | 592 | 8192 |
| `short-5` | short | 12 | 40 | 206 | 219 | 8192 |
| `file-5` | file | 2268 | 244 | 10047 | 10060 | 8192 |
| `cwd-5` | cwd | 2398 | 124 | 604 | 617 | 8192 |
| `short-6` | short | 3 | 40 | 172 | 185 | 8192 |
| `file-6` | file | 2510 | 244 | 11013 | 11026 | 8192 |
| `cwd-6` | cwd | 2710 | 124 | 570 | 583 | 8192 |
| `short-7` | short | 8 | 40 | 189 | 202 | 8192 |
| `file-7` | file | 2764 | 244 | 12030 | 12043 | 8192 |
| `cwd-7` | cwd | 3034 | 124 | 587 | 600 | 8192 |
| `cwd-26k` | omp-26k | 8132 | 263 | 1192 | 1205 | 8192 |
| `tmp-39k` | omp-39k | 11640 | 263 | 1130 | 1143 | 8192 |
| `empty-ask` | edge | 0 | 40 | 160 | 172 | 8192 |
| `follow-is-dump` | edge | 1509 | 40 | 6194 | 6207 | 8192 |
| `think-tags` | edge | 1 | 40 | 162 | 175 | 8192 |

## Reproduction

```
PYTHONPATH=src python3 scripts/verify_paper.py
```

Outputs: `verification/campaign_latest.md`, `verification/campaign_latest.json`,
`verification/suite_latest.json`, and timestamped copies of each.

