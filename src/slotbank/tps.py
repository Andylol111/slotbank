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
    # Cold 819-token prefill vs live suffix reuse, same Air, 2026-08-31.
    "prefill_819_s": 17.0,
    "prefill_819_reuse_s": 0.88,
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
        "omp-defer-weight-pin",
        ADOPTED,
        "OMP /models/load returns after mmap; mx.eval of the 15 GiB pack runs before the first job.",
        "The picker spinner was waiting on _pin_dense, not Python graph construction. "
        "Engine sets ready after lazy load + tokenizer + draft mmap, then pins on the "
        "Metal thread. First generate waits on that pin if you send immediately. "
        "Restarting serve re-pays it. Omitting --vision saves ~0.4 GiB, not the 30s. "
        "Does not page hybrid KV or change 27B weights.",
    ),
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
        "ttft-is-prefill",
        ADOPTED,
        "Time-to-first-token is hybrid GDN prefill, not MTP. ~48 tok/s on this Air.",
        "819-token cold 17.0 s (48 tok/s). Same prefix reused 0.88 s (19×). "
        "Qwen3.8-27B is 48 Gated DeltaNet + 16 full-attn. Prefill should beat "
        "decode (~5.7 tok/s greedy) because one weight read covers T tokens; "
        "48 tok/s is only ~8× decode, far below fused-GDN machines (vMLX ~10k "
        "tok in 14 s cold on an M3 Ultra 397B; mlx-swift GDN pipeline thousands "
        "tok/s on M5 Max 35B-A3B). MTP K=3 raises decode tok/s; it does not "
        "shrink TTFT. Three levers: shrink N (envelope), stop stalling Metal "
        "between tiles (async-prefill-pipeline), kernels (gdn-chunked-cuda-prefill "
        "does not transfer to Metal). Pin (~30 s) is first-request after boot, "
        "not prefill.",
    ),
    Strategy(
        "skip-lm-head-prefill",
        ADOPTED,
        "Prefill tiles pass skip_logits so Qwen's 248k lm_head is not built.",
        "mlx-vlm LanguageModel.__call__ already has skip_logits. Gemma 3 on "
        "mlx-swift saw 2.6× prefill from skipping a 262k head that was actually "
        "evaluated. Here we only mx.eval cache states; lazy MLX may already drop "
        "the head. Passing skip_logits is the explicit path and avoids building "
        "the [T, 248320] graph. Does not change 27B tokens.",
    ),
    Strategy(
        "spec-prefill-sparse",
        REJECTED,
        "SpecPrefill / GemFilter / SwiftKV: skip prompt tokens or layers.",
        "Train-free token dropping and distilled skip-layers quote 2–7× TTFT "
        "on CUDA. The 27B would not see the packed prompt, or its weights "
        "would change. Same class as pflash-drop-prompt. A 4B importance "
        "scorer is qwen35-4b-as-27b-drafter. User forbids changing 27B "
        "tokens/verify.",
    ),
    Strategy(
        "async-prefill-pipeline",
        ADOPTED,
        "async_eval tile N while building N+1; one mx.eval + clear_cache at the end.",
        "mlx-vlm #945 (open): per-chunk mx.eval+clear_cache is the server vs "
        "vMLX TTFT gap on the same Mac (vMLX 10k tok 14 s cold / 0.24 s warm). "
        "LM Studio 1.5× from chunk 512→8192. mlx-swift-lm #225: asyncEval per "
        "chunk, one terminal eval — Qwen3.6-35B-A3B-4bit M5 Max 512 tok "
        "235→2201 tok/s (9.4×), 2k 270→3937 (14.6×); M2 Mini 16 GB only "
        "1.15–1.3× (GPU already saturated). This Air is M4 24 GB / 10 GPU "
        "cores — expect Mini-class if compute-bound, #945-class if the "
        "eval+clear stall dominated. PrefixCache copies still block that tile "
        "(GDN cannot trim). Does not change 27B tokens. Unmeasured on the Air.",
    ),
    Strategy(
        "gdn-chunked-cuda-prefill",
        REJECTED,
        "Port mlx-vlm #1423 chunked GDN (8.5× CUDA TTFT) into slotbank.",
        "On CUDA, T>1 GDN was Python for t in range(T) (~49k graph nodes at "
        "2048 tok). gated_delta_chunked (parallel intra-chunk matmul + "
        "triangular solve + inter-chunk scan) took Qwen3.5-4B bf16 2048 tok "
        "from 2090→247 ms on RTX PRO 6000. PR text: macOS/Metal unchanged — "
        "Metal already runs gated_delta_kernel with the T-loop inside the "
        "shader when Dk%32==0. Qwen3.8-27B is Dk=Dv=128, 16 QK / 48 V heads, "
        "so the packed Metal kernel is eligible. mlx-node #68 then measured "
        "chunked GDN as 2.5–3.5× *slower* than per-step on M5 (24–31× per "
        "isolated GDN call) and ~2× slower on M3; per-step is the M1–M4 "
        "reference. CUDA 8.5× was ops-loop → chunked, not 'chunked beats "
        "fused scan'. README already forbids the T>1 chunked scan for MTP "
        "verify (~4% argmax flips vs T=1). Not a 27B TTFT lever here.",
    ),
    Strategy(
        "vllm-gdn-block-apc",
        DEFERRED,
        "vLLM mamba_cache_mode=all: checkpoint GDN SSM at every block 64.",
        "vLLM #36649 / #54637 expose intermediate h from the CUDA/Triton "
        "chunk_gated_delta_rule so APC can restore mid-prefix without "
        "recompute. Hybrid MTP+APC has been a year of correctness bugs "
        "(tool-call leak, needle miss, hit-rate collapse) and still costs "
        "warm TTFT when rollback crosses a block. Metal gated_delta_kernel "
        "does not return per-block states. Our PrefixCache is the analogue: "
        "stop-and-copy at 128 (short) / 2048 (packed sink). Cannot copy "
        "SGLang Radix page_size>1 (sgl-project/sglang#12867 still struggles "
        "storing GDN at branches). Do not page hybrid KV.",
    ),
    Strategy(
        "distserve-pd",
        REJECTED,
        "Prefill/decode disaggregation (DistServe, vLLM P/D).",
        "Cluster technique: extra hop often helps ITL and hurts TTFT. One "
        "Metal worker; MLX arrays cannot cross threads (async-metal-queues). "
        "No second machine on this Air.",
    ),
    Strategy(
        "gdn-cache-contiguous",
        DEFERRED,
        "mx.contiguous on GDN conv/state slices so multi-chunk prefill does not leak.",
        "mlx-lm #1077: cache slices aliased parent graphs (~540 KB/tok), "
        "24k ctx OOM on 128 GB. mlx-vlm Qwen3.5 already contiguous's conv "
        "state; cache[1] (GDN state) is still a kernel output. Envelope is "
        "8k on 24 GB (~4.4 GiB if the leak is live). Do not reimplement "
        "GatedDeltaNet here; confirm the Air's mlx-vlm includes the conv "
        "contiguous and whether long OMP prefills still grow RSS.",
    ),
    Strategy(
        "qwen-chat-prefix-stable",
        ADOPTED,
        "PrefixCache stops at the Qwen generation-prompt boundary, not 128.",
        "QwenLM/Qwen3#1826 and lmstudio mlx-engine#176: enable_thinking=false "
        "injects empty think tags on add_generation_prompt, but historical "
        "assistant turns omit them, so prefix_n is not a prefix of the next "
        "OMP encode. mlx-engine's template fix got 25× follow-up TTFT "
        "(4.96 s → 0.20 s). We do not patch jinja; encode_chat records "
        "stable_prefix_n from a second apply_chat_template without the "
        "generation prompt. Short prompts snap there (near the tail) instead "
        "of 128 in the middle. Packed 8k still snaps at 2048. Does not change "
        "27B tokens.",
    ),
    Strategy(
        "metal-qmm-prefill",
        DEFERRED,
        "4-bit QMM of the 15 GiB pack is most of remaining prefill time.",
        "atomgradient mlx-inference-bench on Qwen3.5-9B: quantized matmul "
        "57.6% of prefill, GDN recurrence 29%, attention 6.7%. Hybrid prefill "
        "only 5–9% slower than dense Qwen3. After async tiles, the Air's "
        "floor is mlx's affine-4 QMM, not another GDN algorithm. mlx-node "
        "sym8 W8A8 was +67% TTFT at 1024 but is a new checkpoint mlx-lm "
        "cannot load (extra-quant-3bit class). Do not requantize 27B.",
        changes_target_weights=True,
    ),
    Strategy(
        "ane-npu-prefill",
        REJECTED,
        "Core ML / ANE prefill, Metal decode (Yetter / SqueezeBits).",
        "ANE can beat MLX on TTFT for small models. 27B 4-bit is ~15 GiB; a "
        "second Core ML copy does not fit leave-free 6g on 24 GB. One Metal "
        "worker already owns the weights.",
    ),
    Strategy(
        "warm-prefix-at-load",
        DEFERRED,
        "After pin, prefill a synthetic system head into PrefixCache.",
        "Helps only if that head is an exact prefix of the first OMP encode. "
        "OMP re-encodes cwd+harness each run; --direct is stable but short. "
        "The first real turn already fills PrefixCache for follow-ups (19×). "
        "Warming a dummy that misses is a 17s tax at boot.",
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
        "Never exceed the drafter's trained block. Daily 27B uses that cap every request.",
        "mlx-vlm already backs off DFlash depth. scale_draft_block is the same cap for "
        "MTP (3). Growing past trained K is how block 6 lost on 4B MTP. generate_step "
        "reads K once, so shrinking after a round cannot help that round — see "
        "dais-reset-each-request.",
        extra_drafter=True,
    ),
    Strategy(
        "dais-reset-each-request",
        ADOPTED,
        "Arm MTP/DFlash at the trained K at the start of every generate, not last-turn DAIS.",
        "DAIS persist left follow-up OMP turns at K=1 after a low-accept think dump "
        "(~greedy 5.7 instead of MTP ~13). generate_step cannot retune mid-round. "
        "_arm_draft_block restores the cap (3 on sidecar MTP) each _iter_draft. "
        "Does not change 27B weights or page hybrid KV.",
        extra_drafter=True,
    ),
    Strategy(
        "skip-vlm-rope-prime",
        ADOPTED,
        "Do not prime Qwen mRoPE on the full prompt before mlx-vlm generate_step.",
        "mlx-vlm's helper writes mRoPE onto the LM and into kwargs. generate_step then "
        "nulls _position_ids/_rope_deltas, and we never passed the kwargs in. On the 8k "
        "envelope it was get_rope_index of the whole prompt, then discard. Pyramid "
        "tiles + last-token generate_step continue from cache offset. Does not page "
        "hybrid KV.",
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
        "named-kv-survey",
        REJECTED,
        "PyramidKV, Ada-KV, KVTC, DeltaKV, TriAttention, BUZZ, Sparse-vLLM/KVStream.",
        "None attempted. They assume a uniform full-attn KV you can top-k evict, "
        "PCA-code, or page. 48/64 layers here are ArraysCache (already O(1)); "
        "evicting the 16 attn layers without GDN rollback changes 27B text. "
        "KVTC/DeltaKV/BUZZ/Sparse-vLLM are CUDA kernels and page tables, not "
        "the one Metal worker. Daily --draft forbids even SLOTBANK_KV_BITS. "
        "Closest in-tree: 8-bit KV when not drafting (KIVI residual is a comment, "
        "not code), context OS, append-only suffix. vLLM/SGLang hybrid managers "
        "separate a GDN pool from a KV pool — they still do not Pyramid the "
        "recurrent state. The iGPU analogue is igpu-pyramid-tiles (prompt pack + "
        "Metal tiles), not these kernels.",
        needs_trim_cache=True,
    ),
    Strategy(
        "igpu-pyramid-tiles",
        IN_TREE,
        "Pyramid/BUZZ/TriAttention analogue on prompt ids + prefill tiles.",
        "Does not evict or recode hybrid KV. keep_token_ids (SLOTBANK_PROMPT_PACK=1) "
        "keeps a sink prefix, a dense-early middle, and the tail. _pyramid_step "
        "sizes chunk×(offset+chunk) so early Metal tiles stay large. "
        "One-shot CLI still 400s overlong dumps. Serve envelope packs leftovers "
        "after condense, not as the first cut.",
    ),
    Strategy(
        "two-stage-harness",
        IN_TREE,
        "OMP full prompt on a cloud subscription; condensed 27B locally.",
        "The harness blob is non-negotiable for OMP. 27B cannot prefill it on "
        "24 GB. SLOTBANK_CONDENSE / serve --condense keeps the last ask and "
        "file citations, logs the raw user blob when CONTEXT_DIR is set. "
        "upsert does not invent Anthropic/OpenAI keys; it keeps sibling "
        "providers. Serve envelope is the daily local door (omp-serve-envelope).",
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
        "PrefixCache on MTP (draft-prefix-cache), 8-bit KV only when not "
        "drafting. A pager would change 27B text.",
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
        "temp 0 to chase the bench number. OMP still sends 1.0; the 27B envelope maps "
        "that onto Qwen's documented pair (instruct 0.7/0.8/20 with /no_think, thinking "
        "0.6/0.95/20 with /think on the last ask). Not an OMP yaml temperature.",
        extra_drafter=True,
    ),
    Strategy(
        "omp-serve-envelope",
        ADOPTED,
        "Serve absorbs OMP's harness: condense, slim tools, pack leftovers, prefix cache.",
        "OMP still failed on 400/jetsam when the client sent 26k–39k tokens. "
        "slotbank serve now defaults SLOTBANK_ENVELOPE=1. Condense keeps ask + "
        "citations; slim_tools drops JSON schemas (the catalog alone is >16k); "
        "keep_token_ids packs any leftover to 8192. Prefix cache now runs on "
        "the MTP path (draft-prefix-cache) so later turns skip the system-head "
        "prefill. Does not page hybrid KV and does not change 27B weights. "
        "--no-envelope restores 400.",
    ),
    Strategy(
        "omp-session-vs-metal",
        ADOPTED,
        "Wide OMP session (64k) so compaction does not loop; Metal envelope stays 8k.",
        "OMP compacting on its own bar: a 39k cwd dump at contextWindow 32k is ~80% "
        "and 'the most recent turn alone is too large'. Lying in prompt_tokens does "
        "not shrink that bar. Advertise 65536 so the dump fits; serve still prefills "
        "8192. PrefixCache snaps at most 2048 tokens / 384 MiB (a 10k copy was ~790 MiB "
        "and jetsamed 24 GB). YAML maxTokens 2048. Qwen /no_think (qwen-no-think-prompt) "
        "stops the think dump on hi. SpecialHoldback strips streamed <|im_end|>. "
        "Does not page hybrid KV. ~30s is 27B 4-bit pin into Metal; "
        "OMP /models/load now returns after mmap (omp-defer-weight-pin).",
    ),
    Strategy(
        "qwen-no-think-prompt",
        ADOPTED,
        "Qwen /no_think + instruct sampling on the 27B OMP door; stream the answer only.",
        "thinkingFormat: qwen-chat-template does not see <think> in the stream — the "
        "template opens it in the prompt — so OMP printed analysis + </think> + answer "
        "+ <|im_end|>. Envelope appends /no_think (Qwen's own switch), sets "
        "enable_thinking false, remaps OMP temp 1.0 to 0.7/0.8/20, holds think/EOS in "
        "the stream, and names the pane Qwen3.8-27B not …-agent (slotbank). Put /think "
        "on the last ask to reason. Does not change 27B weights or page hybrid KV.",
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
    Strategy(
        "draft-prefix-cache",
        ADOPTED,
        "PrefixCache + pyramid tiles on the MTP/DFlash path. Exact prefix only; no GDN trim.",
        "Daily serve uses --draft, so PrefixCache used to be greedy-only and every OMP "
        "turn re-encoded the chat (not an append of _fed_ids) and paid a cold 10k prefill. "
        "_iter_draft now restores the longest exact snapshot into a new cache, pyramid-tiles "
        "the gap, and leaves mlx-vlm generate_step the last token + rollback. Live append "
        "still wins when the client is append-only. Snap stops are the Qwen "
        "generation-prompt boundary on short prompts (qwen-chat-prefix-stable) "
        "and 2048 on long ones (packed sink). A hardcoded 128 used to split "
        "every short hi into two 27B forwards. 256/512/1024 crumbs used to split "
        "the first 2k of an 8k envelope; eviction already dropped them first. "
        "SLOTBANK_PREFIX_CACHE_MIB defaults to 384 so a 2048-token head fits "
        "(~278 MiB attn KV + GDN), not a 10k envelope copy (~790 MiB) that jetsams 24 GB. "
        "put() refuses snaps past PrefixCache.MAX_SNAP=2048. Does not page hybrid KV.",
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
    if get("async-prefill-pipeline").status != ADOPTED:
        raise ValueError("prefill pipeline must stay adopted; per-tile eval+clear is the TTFT stall")
    if get("qwen-chat-prefix-stable").status != ADOPTED:
        raise ValueError("Qwen generation-prompt boundary must stay the short-prompt snap")
    if daily_draft() != "sidecar-mtp-k3":
        raise ValueError("daily_draft mismatch")
    banned_adopted = {
        "unquantized-bf16-27b",
        "mtp-plus-dflash",
        "sliding-window-kv",
        "hybrid-kv-dynamic-page",
        "named-kv-survey",
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
        "spec-prefill-sparse",
        "gdn-chunked-cuda-prefill",
        "distserve-pd",
        "ane-npu-prefill",
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
    if M4_AIR_24G["prefill_819_reuse_s"] >= M4_AIR_24G["prefill_819_s"] / 10:
        raise ValueError("suffix reuse no longer beats cold prefill by ~10×")


def prefill_seconds(n_tokens: int, reuse: int = 0) -> float:
    """TTFT prefill estimate from the measured 819-token cold rate.

    Does not include pin-on-first-request or the last-token generate_step.
    """
    cold = float(M4_AIR_24G["prefill_819_s"])
    if cold <= 0:
        return 0.0
    rate = 819.0 / cold
    work = max(0, int(n_tokens) - max(0, int(reuse)))
    return work / rate


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
