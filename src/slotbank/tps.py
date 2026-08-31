"""Speculative-decode routes that can raise tok/s without changing 27B text.

The long prompt-chat list (MTPLX brew, Medusa trees, unquantized 27B in 16 GB,
async Metal queues, reverse-spec small writers) is scored here against this
Air: Qwen3.8-27B 4-bit, 24 GB, 120 GB/s, hybrid Gated DeltaNet, mlx-vlm verify.

Status:
  adopted   — daily path, measured
  attempted — measured, not daily
  in_tree   — already in slotbank, inert or not the 27B door
  deferred  — real paper/engine, not wired here
  rejected  — unsafe, impossible, or changes the 27B's tokens

Filesystem only. Nothing here imports MLX (tests/test_fence.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ADOPTED = "adopted"
ATTEMPTED = "attempted"
IN_TREE = "in_tree"
DEFERRED = "deferred"
REJECTED = "rejected"

_VALID = frozenset({ADOPTED, ATTEMPTED, IN_TREE, DEFERRED, REJECTED})

# Cool-machine, this Air, 2026-08-31, 64-token warm window, thinking/vision off.
# Same greedy prefix on parse_iso_dates for MTP vs DFlash (verify intact).
M4_AIR_24G = {
    "greedy_toks": 5.71,
    "mtp_k3_count": 13.47,
    "mtp_k3_code": 9.95,
    "dflash_k8_count": 11.76,
    "dflash_k8_code": 9.10,
    "bandwidth_bytes_s": 120 << 30,
    "weight_bytes_4bit": 15 << 30,
}


@dataclass(frozen=True)
class Strategy:
    id: str
    status: str
    summary: str
    evidence: str
    changes_target_weights: bool = False
    needs_trim_cache: bool = False
    extra_drafter: bool = False


STRATEGIES: tuple[Strategy, ...] = (
    Strategy(
        "sidecar-mtp-k3",
        ADOPTED,
        "Native MTP sidecar at trained block_size=3, mlx-vlm exact verify.",
        "M4 Air 24 GB 2026-08-31: 13.47 count / 9.95 code tok/s, peak 14.9–15.1 GiB. "
        "2.36× greedy. Same parse_iso_dates prefix as DFlash.",
        extra_drafter=True,
    ),
    Strategy(
        "dflash2-k8",
        ATTEMPTED,
        "z-lab DFlash2 student at trained block_size=8.",
        "Same session: 11.76 count / 9.10 code, peak 16.5–17.0 GiB. Slower and heavier than MTP.",
        extra_drafter=True,
    ),
    Strategy(
        "forced-k5",
        REJECTED,
        "One K=5 for every drafter.",
        "Flattened DFlash (trained 8) and MTP (trained 3) onto the same wrong length.",
        extra_drafter=True,
    ),
    Strategy(
        "mtp-plus-dflash",
        REJECTED,
        "Run MTP and DFlash2 in one generate.",
        "mlx-vlm takes one draft_model and one draft_kind. Two guessers share one verify. "
        "DFlash lost to MTP on the high-accept count prompt, so a longer second guesser "
        "is not a 2× layer.",
        extra_drafter=True,
    ),
    Strategy(
        "unquantized-bf16-27b",
        REJECTED,
        "Full BF16 27B in 16–24 GB.",
        "~54 GiB of weights. Does not load. 4-bit is the fit that keeps greedy text.",
        changes_target_weights=True,
    ),
    Strategy(
        "extra-quant-3bit",
        REJECTED,
        "Go below 4-bit to chase bandwidth.",
        "Mixed 3.5bpw measured 0.94–1.11 tok/s on this Air. Hurts kernels, not just bits.",
        changes_target_weights=True,
    ),
    Strategy(
        "dais-trained-cap",
        ADOPTED,
        "Shrink K when accept rate is poor; never exceed the drafter's trained block.",
        "mlx-vlm already backs off DFlash depth. slotbank.tps.scale_draft_block is the "
        "same rule for MTP (cap 3). Growing past trained K is how block 6 lost on 4B MTP.",
        extra_drafter=True,
    ),
    Strategy(
        "suffix-prefix-reuse",
        ADOPTED,
        "Append-only DFlash/MTP cache: prefill only the new suffix.",
        "819-token reuse 0.88 s vs 17.0 s cold (19×). Editing history is a full prefill.",
    ),
    Strategy(
        "context-os",
        ADOPTED,
        "Disk log + verbatim excerpts instead of paging hybrid KV.",
        "can_trim_prompt_cache is false on GDN. Working set is excerpts, not a 1M tensor.",
    ),
    Strategy(
        "vlm-rollback",
        IN_TREE,
        "mlx-vlm rollback_speculative_cache (the real 'innovation tape').",
        "Rewinds Gated DeltaNet on reject. Required for lossless MTP/DFlash on this hybrid.",
        extra_drafter=True,
    ),
    Strategy(
        "prompt-lookup",
        IN_TREE,
        "SLOTBANK_LOOKAHEAD n-gram proposals from context.",
        "_can_speculate requires a trimmable cache, so this is inert on Qwen3.8 hybrid.",
        needs_trim_cache=True,
    ),
    Strategy(
        "mtplx-engine",
        DEFERRED,
        "youssofal/MTPLX: in-model MTP heads, custom verify kernels.",
        "Real Apache-2.0 Mac app. Flagship 27B speed pack wants 32 GB+. Published 2.24× "
        "is M5 Max class (~28→63), not this 120 GB/s Air. mlx-community 4-bit already "
        "stripped mtp.* tensors; sidecar MTP is the matching slotbank door. Do not replace "
        "the measured 13.5 tok/s path with an unmeasured brew install.",
        extra_drafter=False,
    ),
    Strategy(
        "tree-medusa-eagle",
        DEFERRED,
        "Draft trees (Medusa / EAGLE-3 / DDTree / TreeWY).",
        "TreeWY (arXiv 2608.20961): trees raise accept a little and are not a throughput "
        "win yet; GDN needs per-node state or a WY solve. DDTree-mlx claims ~10–15% over "
        "DFlash, which would still lose to sidecar MTP here. Needs extra Metal kernels.",
        extra_drafter=True,
    ),
    Strategy(
        "leap-mtp",
        DEFERRED,
        "L-MTP skip-ahead heads.",
        "No Qwen3.8-27B L-MTP checkpoint on this machine. Would still verify through 27B.",
        extra_drafter=True,
    ),
    Strategy(
        "ssd-double-speculate",
        DEFERRED,
        "Speculate the verify outcome (SSD).",
        "Not in mlx-vlm generate_step. One drafter, one verify per round.",
        extra_drafter=True,
    ),
    Strategy(
        "sliding-window-kv",
        REJECTED,
        "Page or LRU hybrid KV / 4k sliding window mid-decode.",
        "48 GDN layers cannot trim. Silent wrong text. Context OS is the history path.",
        needs_trim_cache=True,
    ),
    Strategy(
        "hybrid-kv-dynamic-page",
        REJECTED,
        "Dynamically page/offload hybrid KV so OMP's harness fits in 24 GB.",
        "Load + `run hi` MEMORY PRESSURE 2 is ~15 GiB 4-bit weights + page cache, "
        "not KV: 50 tokens × 64 KiB ≈ 3 MiB. 48 GDN layers are already O(1) "
        "(~150 MiB ArraysCache). Only 16 full-attn layers grow (64 KiB/token). "
        "mlx-lm ArraysCache cannot trim (you cannot drop N tokens from a hidden "
        "state; PR 1254 reset-to-empty was refused). Daily --draft forbids "
        "SLOTBANK_KV_BITS (verifier keys.shape). Prefill activations, not KV, "
        "are the spike that grows with context (3.61→5.06 GiB, 256→4096). "
        "Levers that exist: shrink the prompt, SLOTBANK_PREFILL_STEP, "
        "SLOTBANK_CACHE_LIMIT_MIB, append-only suffix reuse, context OS, "
        "8-bit KV only when not drafting. A pager would change 27B text.",
        needs_trim_cache=True,
    ),
    Strategy(
        "layer-stream-gdn",
        REJECTED,
        "Keep attention in RAM, stream GDN from SSD.",
        "4-bit + MTP already fits leave-free 6g. SSD GDN would be the 1 tok/s path.",
    ),
    Strategy(
        "async-metal-queues",
        REJECTED,
        "DraftQueue + VerifyQueue double buffer.",
        "slotbank owns one Metal worker thread. Arrays cannot cross threads. mlx-vlm "
        "already overlaps draft compute inside that thread.",
        extra_drafter=True,
    ),
    Strategy(
        "reverse-spec-small-writer",
        REJECTED,
        "27B thinks, 4B writes the answer tokens.",
        "The small model emits tokens. That is not 27B greedy. User asked not to cripple "
        "the model's own operations. 4B is a separate speed door (~46–54 tok/s).",
        changes_target_weights=True,
    ),
    Strategy(
        "kv-quant-with-draft",
        REJECTED,
        "SLOTBANK_KV_BITS on the DFlash/MTP path.",
        "Verifier reads keys.shape. Quantised cache breaks that.",
        extra_drafter=True,
    ),
    Strategy(
        "pflash-drop-prompt",
        REJECTED,
        "PFlash / drop mid-prompt tokens.",
        "Changes the prompt the 27B sees. Incompatible with context OS verbatim excerpts.",
    ),
    Strategy(
        "skip-full-attn-on-draft",
        REJECTED,
        "Skip every fourth (full-attention) layer while drafting.",
        "Changes the draft distribution. Verify stays 27B, accept rate falls.",
        extra_drafter=True,
    ),
    Strategy(
        "harness-structure-for-tps",
        REJECTED,
        "Oh My Pi / long structured briefs as a tok/s multiplier.",
        "Harness does not raise Metal tok/s. Count vs code was 13.47 vs 9.95, not 13 vs 27. "
        "Longer briefs spend more tokens. Thinking stays on; it uses tokens, it does not slow GEMM.",
    ),
    Strategy(
        "harness-temp-1",
        ADOPTED,
        "Anthropic/OMP temperature 1.0 with thinking. Rejection sampling keeps 27B.",
        "messages.py already defaults temp to 1 when the client omits it. mlx-vlm "
        "verify is Leviathan-Chen: output distribution is the 27B at that temperature. "
        "Accept rate (hence tok/s) is lower than the greedy 13.47 bench. Do not force "
        "temp 0 to chase the bench number — the harness uses 1 for thinking.",
        extra_drafter=True,
    ),
    Strategy(
        "qwen35-4b-as-27b-drafter",
        REJECTED,
        "Load Qwen3.5-4B (+ DFlash) as the 27B's speculative drafter.",
        "Same vocab (248320) is a trap. The 4B is a different width (~2560 vs 5120) "
        "and a full LM, not an MTP/DFlash student. Classic small-draft/large-verify "
        "uses mlx-lm trim, which is silently wrong on Gated DeltaNet. 24 GB also "
        "cannot hold 27B 4-bit (~15 GiB) and 4B+DFlash (~3 GiB) beside vision/KV. "
        "4B+DFlash remains the separate ~46–54 tok/s door, not a 27B booster.",
        extra_drafter=True,
        changes_target_weights=True,
    ),
    Strategy(
        "omp-models-yml",
        ADOPTED,
        "Write current-schema ~/.omp/agent/models.yml on serve so OMP lists the 27B.",
        "Old examples used type: anthropic / base_url; one invalid custom file disables "
        "all custom providers. OMP 18 /model local pane is llama.cpp + lm-studio. "
        "discovery: openai-models-list on those ids replaces the implicit probe and "
        "F5 keeps a cached empty list. YAML lists static models on both engines "
        "(no discovery key) plus providers.slotbank. Then: omp models refresh.",
    ),
    Strategy(
        "auto-sidecar-mtp",
        ADOPTED,
        "If --draft is omitted, attach a sibling MTP-4bit (else DFlash) that admits.",
        "Serve without --draft was greedy 5.71 tok/s. Same 27B weights; mlx-vlm verify.",
        extra_drafter=True,
    ),
)


def get(strategy_id: str) -> Strategy:
    for s in STRATEGIES:
        if s.id == strategy_id:
            return s
    raise KeyError(strategy_id)


def by_status(status: str) -> tuple[Strategy, ...]:
    if status not in _VALID:
        raise ValueError(status)
    return tuple(s for s in STRATEGIES if s.status == status)


def daily_draft() -> str:
    """The --draft that belongs on the 27B serve line."""
    return "sidecar-mtp-k3"


def pack_read_ceiling_toks(
    weight_bytes: int,
    bandwidth_bytes_s: int = M4_AIR_24G["bandwidth_bytes_s"],
) -> float:
    """Tokens/s if every decode rereads the whole pack once. Speculative can beat this."""
    if weight_bytes <= 0 or bandwidth_bytes_s <= 0:
        return 0.0
    return bandwidth_bytes_s / weight_bytes


def draft_accept_rate(
    accept_lens: Iterable[Any] | None,
    draft_lens: Iterable[Any] | None,
) -> float | None:
    """Accepted draft tokens / proposed draft tokens. None if there is no sample."""
    acc = [float(a) for a in (accept_lens or [])]
    drf = [int(d) for d in (draft_lens or [])]
    n = min(len(acc), len(drf))
    if n <= 0:
        return None
    proposed = sum(drf[:n])
    if proposed <= 0:
        return None
    return sum(acc[:n]) / proposed


def scale_draft_block(
    *,
    cap: int,
    accept_rate: float | None,
    current: int | None = None,
    low: float = 0.40,
    high: float = 0.80,
) -> int:
    """DAIS: move K by 1 toward the trained cap. Never exceed cap. Floor 1.

    Growing past the drafter's trained block_size is how MTP block 6 measured
    0.85× on the 4B. Shrinking on low accept avoids a fat verify that then rolls back.
    """
    cap_n = max(1, int(cap))
    cur = cap_n if current is None else max(1, int(current))
    cur = min(cur, cap_n)
    if accept_rate is None:
        return cur
    if accept_rate < low:
        return max(1, cur - 1)
    if accept_rate > high:
        return min(cap_n, cur + 1)
    return cur


def catalog_sound() -> None:
    """Invariants the unit tests pin. Fail closed if a rejected idea is marked adopted."""
    ids = [s.id for s in STRATEGIES]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate strategy id")
    for s in STRATEGIES:
        if s.status not in _VALID:
            raise ValueError(f"{s.id}: bad status {s.status}")
    if get("sidecar-mtp-k3").status != ADOPTED:
        raise ValueError("daily MTP must stay adopted until a cooler A/B beats it")
    if daily_draft() != "sidecar-mtp-k3":
        raise ValueError("daily_draft mismatch")
    banned_adopted = {
        "unquantized-bf16-27b",
        "mtp-plus-dflash",
        "sliding-window-kv",
        "hybrid-kv-dynamic-page",
        "kv-quant-with-draft",
        "pflash-drop-prompt",
        "reverse-spec-small-writer",
        "forced-k5",
        "layer-stream-gdn",
        "async-metal-queues",
        "skip-full-attn-on-draft",
        "extra-quant-3bit",
        "harness-structure-for-tps",
        "qwen35-4b-as-27b-drafter",
    }
    for sid in banned_adopted:
        if get(sid).status == ADOPTED:
            raise ValueError(f"{sid} cannot be adopted")
    for s in STRATEGIES:
        if s.status == ADOPTED and s.needs_trim_cache:
            raise ValueError(f"{s.id}: hybrid 27B cannot adopt a trim-cache route")
        if s.status == ADOPTED and s.changes_target_weights:
            raise ValueError(f"{s.id}: adopted route changed 27B weights")
    ceil = pack_read_ceiling_toks(M4_AIR_24G["weight_bytes_4bit"])
    if not (7.0 <= ceil <= 9.0):
        raise ValueError(f"4-bit pack-read ceiling drifted: {ceil}")
    if M4_AIR_24G["mtp_k3_count"] <= M4_AIR_24G["dflash_k8_count"]:
        raise ValueError("catalog still claims DFlash beats MTP")
    if M4_AIR_24G["mtp_k3_count"] < 2 * M4_AIR_24G["greedy_toks"] - 0.2:
        raise ValueError("MTP speedup vs greedy drifted below ~2×")


def log_path() -> Path:
    override = os.environ.get("SLOTBANK_TPS_LOG", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".local" / "share" / "slotbank" / "tps-attempts.jsonl"


def register_attempt(
    strategy_id: str,
    *,
    outcome: str,
    evidence: str,
    toks: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Append one attempt to the machine-local JSONL. Creates the file if needed."""
    get(strategy_id)  # unknown id is a bug
    if outcome not in _VALID:
        raise ValueError(outcome)
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "id": strategy_id,
        "outcome": outcome,
        "evidence": evidence,
    }
    if toks is not None:
        rec["toks"] = toks
    if extra:
        rec["extra"] = extra
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return path


def read_attempts() -> list[dict[str, Any]]:
    path = log_path()
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def seed_local_log() -> Path:
    """Register the measured catalog once so the Air (or this VM) has a local paper trail."""
    path = log_path()
    if path.is_file() and path.stat().st_size > 0:
        return path
    for s in STRATEGIES:
        toks = None
        if s.id == "sidecar-mtp-k3":
            toks = M4_AIR_24G["mtp_k3_count"]
        elif s.id == "dflash2-k8":
            toks = M4_AIR_24G["dflash_k8_count"]
        register_attempt(s.id, outcome=s.status, evidence=s.evidence, toks=toks)
    return path
