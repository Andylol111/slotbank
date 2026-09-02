"""Claim IDs for the slotbank technical report.

Each claim is either:
  verified_here  — this VM can reproduce it (no Metal weights)
  air_pinned     — measured on the author's M4 Air; tests pin the recorded numbers
  cannot_retime  — needs the 27B Metal process; stated as such in the paper

Do not invent Air tok/s here. Pins live in slotbank.tps.M4_AIR_24G.

Traceability: every claim has a test name and a research-question id.
The campaign script writes verification/campaign_<stamp>.{json,md}.
"""

from __future__ import annotations

CLAIMS: tuple[dict[str, str], ...] = (
    {
        "id": "C-fence",
        "kind": "verified_here",
        "rq": "RQ4",
        "test": "tests/test_fence.py::test_only_three_files_import_mlx",
        "also": "tests/test_paper_verification.py::test_C_fence",
        "statement": "Only runtime.py, expert_slots.py, offload_cache.py import MLX.",
    },
    {
        "id": "C-catalog",
        "kind": "verified_here",
        "rq": "RQ4",
        "test": "tests/test_paper_verification.py::test_C_catalog",
        "statement": "Fail-closed catalog: banned ids cannot be adopted; adopted routes keep 27B text.",
    },
    {
        "id": "C-clocks",
        "kind": "air_pinned",
        "rq": "RQ1",
        "test": "tests/test_paper_verification.py::test_C_clocks",
        "statement": "Named Air clocks: 5.71 greedy, 13.47/9.95 MTP, 11.76/9.10 DFlash, 17.0s/0.88s prefill.",
    },
    {
        "id": "C-pack-ceil",
        "kind": "verified_here",
        "rq": "RQ1",
        "test": "tests/test_paper_verification.py::test_C_pack_ceil",
        "statement": "4-bit pack-read ceiling on 120 GB/s is in [7, 9] tok/s.",
    },
    {
        "id": "C-mac200",
        "kind": "verified_here",
        "rq": "RQ1",
        "test": "tests/test_paper_verification.py::test_C_mac200",
        "statement": "No official Qwen Mac tok/s row; +200% dense-27B Mac hit is ANE prefill, not this Air.",
    },
    {
        "id": "C-scale",
        "kind": "verified_here",
        "rq": "RQ1",
        "test": "tests/test_paper_verification.py::test_C_scale",
        "statement": "oMLX M3 Max decode scaled by 120/400 is 5.94 / 9.78, near the pinned Air rates.",
    },
    {
        "id": "C-hybrid-kv",
        "kind": "verified_here",
        "rq": "RQ2",
        "test": "tests/test_paper_verification.py::test_C_hybrid_kv",
        "statement": "Qwen3.8 hybrid full-attn KV is 64 KiB/token; can_trim path is refused without dflash verify.",
    },
    {
        "id": "C-trained-k",
        "kind": "verified_here",
        "rq": "RQ2",
        "test": "tests/test_paper_verification.py::test_C_trained_k",
        "statement": "Draft block is trained K (MTP 3, DFlash 8), not a forced 5.",
    },
    {
        "id": "C-leave-free",
        "kind": "verified_here",
        "rq": "RQ3",
        "test": "tests/test_paper_verification.py::test_C_leave_free",
        "statement": "24 GB default leave-free is 8 GiB; 27B 4-bit + sidecar needs 6 GiB leave-free (18 GiB WS).",
    },
    {
        "id": "C-slots-c",
        "kind": "verified_here",
        "rq": "RQ3",
        "test": "tests/test_paper_verification.py::test_C_slots_c",
        "statement": "35B-A3B on 24 GB picks C in (16, 256); OLMoE stays at the 2×top-k floor.",
    },
    {
        "id": "C-sys-cap",
        "kind": "verified_here",
        "rq": "RQ5",
        "test": "tests/test_paper_verification.py::test_C_sys_cap",
        "statement": "Envelope system head is 256 tokens on every turn, including follow-up.",
    },
    {
        "id": "C-history",
        "kind": "verified_here",
        "rq": "RQ5",
        "test": "tests/test_paper_verification.py::test_C_history",
        "statement": "User-dump condense is the same recipe as last-ask and as history.",
    },
    {
        "id": "C-prefix",
        "kind": "verified_here",
        "rq": "RQ5",
        "test": "tests/test_paper_verification.py::test_C_prefix",
        "statement": "Follow-up encode shares stable_prefix_n with the first encode (pre /no_think body).",
    },
    {
        "id": "C-pack-8k",
        "kind": "verified_here",
        "rq": "RQ5",
        "test": "tests/test_paper_verification.py::test_C_pack_8k",
        "statement": "After envelope, Metal prompt ids are ≤ 8192; 26k/39k dumps do not refuse.",
    },
    {
        "id": "C-pyramid",
        "kind": "verified_here",
        "rq": "RQ5",
        "test": "tests/test_paper_verification.py::test_C_pyramid",
        "statement": "keep_token_ids is sink + pyramid middle + tail; 8k head lands on 2048.",
    },
    {
        "id": "C-slim",
        "kind": "verified_here",
        "rq": "RQ5",
        "test": "tests/test_paper_verification.py::test_C_slim",
        "statement": "slim_tools drops JSON schemas and stays inside the 256-token tool budget.",
    },
    {
        "id": "C-inject",
        "kind": "verified_here",
        "rq": "RQ5",
        "test": "tests/test_paper_verification.py::test_C_inject",
        "statement": "Serve envelope does not compile CONTEXT_DIR back into the prefix unless inject=1.",
    },
    {
        "id": "C-context",
        "kind": "verified_here",
        "rq": "RQ5",
        "test": "tests/test_paper_verification.py::test_C_context",
        "statement": "Context OS log is append-only; compile is newest verbatim spans, not a paraphrase.",
    },
    {
        "id": "C-session",
        "kind": "verified_here",
        "rq": "RQ2",
        "test": "tests/test_paper_verification.py::test_C_session",
        "statement": "dflash_session_ok refuses attention offset past |fed|.",
    },
    {
        "id": "C-sampling",
        "kind": "verified_here",
        "rq": "RQ2",
        "test": "tests/test_paper_verification.py::test_C_sampling",
        "statement": "OMP temperature ≥ 0.99 remaps to Qwen's documented instruct pair; T<0.99 is left alone.",
    },
    {
        "id": "C-suite",
        "kind": "verified_here",
        "rq": "RQ5",
        "test": "tests/test_paper_verification.py::test_C_suite",
        "statement": "Envelope prompt suite: every case keeps system≤256, packed≤8192, history-stable prefix.",
    },
    {
        "id": "C-quality",
        "kind": "cannot_retime",
        "rq": "RQ5",
        "test": "tests/test_paper_verification.py::test_C_cannot_retime_are_documented",
        "statement": "Task quality of the condensed 8k prompt vs the full OMP harness (needs 27B + agent tasks).",
    },
    {
        "id": "C-soak",
        "kind": "cannot_retime",
        "rq": "RQ1",
        "test": "tests/test_paper_verification.py::test_C_cannot_retime_are_documented",
        "statement": "Fanless soak 60–70% of cool decode after 8–10 min (needs the Air under load).",
    },
    {
        "id": "C-metal-toks",
        "kind": "air_pinned",
        "rq": "RQ1",
        "test": "tests/test_paper_verification.py::test_C_air_pinned_are_documented",
        "statement": "Live Metal tok/s on Qwen3.8-27B-4bit; this VM has no weights.",
    },
)

RESEARCH_QUESTIONS: tuple[dict[str, str], ...] = (
    {
        "id": "RQ1",
        "title": "Decode and prefill clocks",
        "question": (
            "On a fanless 24 GB M4 Air, what named clocks does Qwen3.8-27B-4bit "
            "actually hit, and how do they sit against bandwidth-scaled Max numbers "
            "and the 2026 +200% Mac headline?"
        ),
    },
    {
        "id": "RQ2",
        "title": "Hybrid-safe speculation",
        "question": (
            "Can speculative decode raise tok/s without changing 27B text when "
            "48/64 layers are untrimmable Gated DeltaNet?"
        ),
    },
    {
        "id": "RQ3",
        "title": "Working-set fit",
        "question": (
            "What leave-free and slot-capacity rules admit 27B dense plus sidecar, "
            "or 35B-A3B experts, on 24 GB without pretending the page cache is VRAM?"
        ),
    },
    {
        "id": "RQ4",
        "title": "Fail-closed policy",
        "question": (
            "Can admission, catalog, and OMP policy be checked on a machine that "
            "cannot pin 15 GiB, and do those checks refuse routes that change tokens?"
        ),
    },
    {
        "id": "RQ5",
        "title": "Envelope structure",
        "question": (
            "Does the local envelope keep system≤256, packed ids≤8192, and a "
            "history-stable prefix on OMP-shaped dumps, without claiming task quality?"
        ),
    },
)

# Optional-dep tests the cloud CI filter drops. Not paper-claim bugs.
OPTIONAL_DEP_FILTER = (
    "not huggingface and not hub_progress and not rich and not "
    "download_view and not plain_progress and not resolve_passes and not "
    "shed_keeps and not strip_unused"
)

OPTIONAL_DEP_FAILURES = (
    "tests/test_cli.py::test_resolve_passes_through_explicit_ids",
    "tests/test_cli.py::test_hub_progress_receives_the_name_kwarg",
    "tests/test_cli.py::test_hub_progress_forces_disable_off",
    "tests/test_cli.py::test_plain_progress_has_no_escape_codes_and_is_throttled",
    "tests/test_cli.py::test_download_view_never_paints_over_a_pipe",
    "tests/test_kv_quant.py::test_shed_keeps_dflash_session",
    "tests/test_kv_quant.py::test_strip_unused_drops_vision",
)

PROTOCOL = {
    "name": "slotbank paper verification campaign",
    "independent": (
        "prompt class (short / file dump / cwd dump / 26k / 39k), "
        "turn (first vs follow-up), leave-free (6g vs 8g), catalog status, "
        "draft trained K, temperature relative to 0.99"
    ),
    "dependent": (
        "system tokens after condense, packed id length, stable_prefix_n, "
        "admit.ok, catalog_sound, pin equality, session_ok"
    ),
    "not_measured": (
        "task quality of condensed vs full OMP harness (C-quality); "
        "fanless soak tok/s (C-soak); live Metal 27B tok/s on this VM (C-metal-toks)"
    ),
    "oracle": (
        "Bit-identity is not claimed here. Structural oracles are token-budget "
        "caps, exact prefix equality, catalog fail-closed, and pin drift."
    ),
}


def claims_by_kind(kind: str) -> tuple[dict[str, str], ...]:
    return tuple(c for c in CLAIMS if c["kind"] == kind)
