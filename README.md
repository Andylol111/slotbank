# slotbank

Run Mixture-of-Experts models on Apple Silicon without wiring the whole expert bank into Metal. A C-slot LRU keeps a compact pack resident; misses read from a file-backed bank; `gather_qmm` sees only the pack. macOS keeps a leave-free reserve so the machine still multitasks.

No PyTorch. Inference is MLX. One process owns the model.

**What it is for:** models whose expert bank exceeds the Metal working set. On a 24 GB Mac that is roughly 18–40 GiB of weights. Below that line, stock mlx-lm is faster and you should use stock. Above it, stock and llama.cpp do not run at all. See [Benchmarks](#benchmarks).

## About

slotbank is a macOS-native MoE runtime. The job is the same idea as a discrete-GPU expert cache, rewritten for unified memory:

1. Steal the stacked `(E, out, in)` weights off Switch linears so mlx-lm cannot `gather_qmm` the full bank.
2. Remap routing ids → slot ids on Metal. Decode does not `.tolist()` the router.
3. Fill only misses into a C-slot pack, in place, with batched readahead. GEMM on that pack.
4. Size `C` once at load. Freeze it for a generate. Do not softmax live RAM every token.

There are three memory tiers, and the whole design is about which one holds what:

| Tier | On a 24 GB M4 | Holds |
|---|---|---|
| Metal working set (wired) | 17.8 GiB | the C-slot pack — 3.46 GiB for a 35B-A3B |
| OS page cache (evictable) | ~12–16 GiB usable | as much of the expert bank as fits |
| SSD | everything | the rest |

Stock mlx-lm and llama.cpp both put the whole bank in tier 1 and fail at 19 GiB. slotbank puts 3.46 GiB in tier 1 and lets tiers 2 and 3 serve misses.

Decode speed is bandwidth over bytes touched:

```
tok/s ≈ bytes_available_per_second / bytes_touched_per_token
```

Which tier serves those bytes decides everything. Measured on this machine:

| Source | Throughput |
|---|---|
| page cache (warm) | ~10 GB/s |
| SSD via `pread` / batched readahead | 1000–3300 MiB/s |
| SSD via one-at-a-time mmap page faults | 358 MiB/s |

That last row is why `SliceStore` issues `madvise(MADV_WILLNEED)` for the whole miss set before it blocks on any of it.

## Features

- **Expert-slot cache** — file bank + C-slot LRU + in-place miss fill + GEMM on the pack
- **Batched readahead** — the ensure kernel knows every miss row before the first read, so they fault in parallel
- **Leave-free RAM** — `total − leave_free` is the working set
- **Admit before load** — refuse a model whose active weights + KV do not fit
- **Bit-identical** — default output matches stock mlx-lm exactly (maxabs 0.0)
- **Agent APIs** — OpenAI Chat Completions (Cursor), Anthropic Messages (Claude Code), OpenAI Responses (Codex)
- **CLI** — `generate`, `serve`, `admit`

Not included, on purpose: CUDA graphs, PCIe overlap, hybrid CPU experts, tensor parallelism, radix/semantic cache, a second draft model, a desktop GUI.

## Benchmarks

Measured 2026-08-23 on a fanless **M4 Air, 10 GPU cores, 24 GB unified, 16 KB pages**, `mlx==0.32.1`, `mlx-lm==0.31.3`. Metal reports a 17.76 GiB max recommended working set. Every number is from a script in this repo's history; each config ran cold first, then repeated, and both are shown because the spread is the story.

### Headline: a model nothing else on this machine can run

`Qwen3.5-35B-A3B` 4-bit — 40 layers, E=256, top-k 8, **19.02 GiB of weights** (16.9 GiB of it experts).

| Runtime | Quant | Result | Active memory | Machine |
|---|---|---|---|---|
| **llama.cpp** (5 configs) | Q4_K_S, 19.25 GiB | **cannot run** | Metal OOM | drove to critical pressure twice |
| **stock mlx-lm** (full bank) | MLX 4-bit, 19.02 GiB | 0.005 tok/s (208 s/token) | 18.225 GiB | 13.2 GB compressed, thrashing |
| **slotbank** `C=32` (default) | MLX 4-bit, 19.02 GiB | **~3.9–4.7 tok/s sustained** (6–8 in short bursts) | **3.459 GiB wired** | responsive |

> **3.459 GiB is the wired Metal working set, not total RAM.** slotbank still wants the OS page cache to hold as much of the 16.9 GiB expert bank as it can — that is the whole point of the tier table above, and it is why cold is 2.24 and warm is 7.96 at an identical wired footprint. RAM did not stop mattering; it moved from a hard wall to a speed knob. On a machine with little spare RAM this model would still *run*, but nearly every miss would reach the SSD.

llama.cpp failures in full — this is not a tuning oversight, it is the working-set wall:

```
-ngl 99                    failed to decode prompt batch, res = -3
-ngl 32                    memory pressure level 4 (critical), killed by watchdog
-ngl 24                    res = -3
-ngl 99 -c 512 -fa on      kIOGPUCommandBufferCallbackErrorOutOfMemory
-ngl 99 -ot exps=CPU       memory pressure level 4 (critical), killed by watchdog
```

Caveat, stated plainly: quantization was matched for fairness. llama.cpp also ships `UD-IQ4_XS` at 16.29 GiB, which would likely fit. The honest claim is *llama.cpp cannot run this model at this quality level on this machine* — its route is a smaller, lower-quality quant.

### The other side: models that already fit

`OLMoE-1B-7B` 4-bit — 16 layers, E=64, top-k 8, 3.6 GB of weights. This fits in the working set, so stock wins outright:

| Runtime | tok/s (3 reps) | Active memory |
|---|---|---|
| stock mlx-lm | **56.04 / 54.63 / 53.48** | 3.656 GiB |
| slotbank `C=16` | 14.38 / 18.04 / 18.94 | **1.130 GiB** |

**Use stock for models that fit.** slotbank trades ~3× throughput for ~3× memory here, which is a bad deal when the machine had the RAM anyway.

### Choosing C

35B-A3B, same prompt, C swept. The optimum is interior, and it is *not* "fill available RAM":

| C | C/E | pack | active | tok/s |
|---|---|---|---|---|
| 16 | 6.2% | 1.055 GiB | 2.404 GiB | 2.63 |
| **32** | **12.5%** | **2.109 GiB** | **3.459 GiB** | **8.12** |
| 64 | 25.0% | 4.219 GiB | 5.569 GiB | 8.07 |
| 128 | 50.0% | 8.438 GiB | 9.787 GiB | 2.22 |
| 227 | 88.7% | 14.963 GiB | 16.313 GiB | 0.50 |

A pack that does not fit competes with the page cache holding the rest of the bank, so past a point **more slots are slower**. C=227 was an earlier default; the policy now targets `max(2·top_k, E/8)` clamped by budget.

### Where the wins came from

35B-A3B at C=64, decode, cold and warm:

| Change | cold | warm |
|---|---|---|
| private-queue `DeviceCopy`, one `waitUntilCompleted` per layer per token | 1.16 | 5.93 |
| in-place slice fill, no pack eval inside `copy_missing` | 2.50 | 2.89 |
| **mmap + batched `madvise(WILLNEED)`** | **7.66** | **8.22** |

The first change removed 16 cross-queue stalls per token (~3.5 ms each, moving 1.4 MiB — ~300× the time those bytes deserve). The second removed the serialization of cold page faults. Cold start improved **6.7×** and the cold/warm gap effectively closed.

Prefill, 865-token prompt:

| Change | prefill |
|---|---|
| baseline | 126.4 s |
| + batched readahead | 49.4 s |
| + `sorted_indices` threaded through | ~43 s |
| + prefill waves (`SLOTBANK_WAVES=1`, opt-in) | 37.3 s |

### Context scaling

35B-A3B at C=32. **Decode throughput is flat in context**; the cost of long context is prefill, in time and in peak memory:

| ctx | prefill | tok/s | miss/layer-call | active | peak |
|---|---|---|---|---|---|
| 128 | 19.6 s | 1.99 | 4.079 / 8 | 3.459 GiB | 5.567 GiB |
| 512 | 28.4 s | 2.42 | 3.490 / 8 | 3.469 GiB | 6.053 GiB |
| 2048 | 57.5 s | 2.13 | 3.635 / 8 | 3.498 GiB | 7.134 GiB |
| 4096 | 105.6 s | 1.82 | 2.700 / 8 | 3.537 GiB | 9.310 GiB |

Active memory barely moves because only 10 of 40 layers are full-attention (`full_attention_interval: 4`); the other 30 are GatedDeltaNet with constant-size state. KV is ~20 KiB/token, so 128k context is ~2.5 GiB. **KV is not the constraint.** The prefill peak is: extrapolating the last column, ~16 GiB around 32k context, which is where it meets the working set.

### Decode time budget

35B-A3B, C=64, cold, measured by stubbing stages out:

| Stage | ms/token | share |
|---|---|---|
| file read + pack write | 136.9 | 73% |
| per-layer meta sync (40×) | 23.4 | 12% |
| gather + attention + sampler (irreducible) | 30.0 | 15% |

Pinned down separately at the default `C=32` with the copy path stubbed out, 8 repeats:

```
compute-bound ceiling   median 19.23 tok/s   min 18.40   max 21.25
```

So the ceiling is **~19–21 tok/s**, and from 7.96 warm there is roughly **2.4× of headroom** left. (An earlier single run at C=64 reported 33.4; two other runs of the same measurement gave 21.0 and 20.1, so treat 33 as an outlier.) The remaining gap is I/O shape, not bandwidth — and it cannot be pushed past the ceiling without attacking the attention/GEMM stack itself, which is MLX's domain, not slotbank's.

### Accuracy

Default output is **bit-identical to stock mlx-lm** — maxabs 0.0 on real-model logits, including through mlx-lm's sorted-routing path. Expert row reads were verified bit-identical for both `U32` and `BF16` tensors. 51 tests pass.

`SLOTBANK_WAVES=1` is the one exception and is off by default; see [Environment](#environment).

## Install

Apple Silicon macOS. Python 3.10+. [uv](https://docs.astral.sh/uv/) recommended.

```bash
cd ~/Desktop/slotbank
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick start

Admit first (no weights are downloaded):

```bash
slotbank admit --model /path/to/Qwen3.5-35B-A3B-4bit --leave-free 8g
```

One-shot generate:

```bash
slotbank generate --model /path/to/model --prompt "hello" --max-tokens 64 --leave-free 8g
```

Serve the three agent doors on localhost:

```bash
slotbank serve --model /path/to/model --host 127.0.0.1 --port 8080 --leave-free 8g
```

Optional `--api-key` is checked on `Authorization: Bearer` and `x-api-key`.

## Agent APIs

The server is local HTTP. Point the client at this process. Do not send your weights to anyone else.

| Client | Protocol | Base URL |
|---|---|---|
| Cursor Chat / OpenAI SDK | `POST /v1/chat/completions` | `http://127.0.0.1:8080/v1` |
| Claude Code | `POST /v1/messages` | `http://127.0.0.1:8080` |
| Codex | `POST /v1/responses` | `http://127.0.0.1:8080/v1` |

Also: `GET /v1/models`, `POST /v1/completions`, `POST /v1/messages/count_tokens`, `GET /health`.

Responses is stateless. Codex must resend full `input`. `previous_response_id` and `background` return 400.

### Cursor Chat

Settings → Models → add a custom OpenAI-compatible model:

- API base: `http://127.0.0.1:8080/v1`
- API key: whatever you passed to `--api-key`, or any string if you omitted it
- Model id: the folder name (also listed on `GET /v1/models`)

### Claude Code

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_API_KEY=local
claude
```

Claude Code talks to `/v1/messages`. System text, tool_use / tool_result blocks, and `count_tokens` are implemented. Images are ignored.

### Codex

`~/.codex/config.toml`:

```toml
[model_providers.slotbank]
name = "slotbank"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"

[profiles.slotbank]
model_provider = "slotbank"
model = "local"
```

Then `codex --profile slotbank`.

### Running several agents at once

**Point every agent at one `slotbank serve` and let the queue serialize them.** Do not run one process per agent.

Two concurrent slotbank processes on the same 35B model, measured:

| | one process | two processes |
|---|---|---|
| per-process tok/s | 2.24 → 7.96 | 1.15 → 1.82 |
| **combined tok/s** | **7.96** | **3.61** |
| active memory | 3.459 GiB | 6.918 GiB |

Total throughput nearly halves while memory doubles. The page cache is the bottleneck and it is shared, so a second process buys nothing and evicts the first one's hot experts. The single Metal-owning thread with a job queue is the correct design, not a limitation to work around.

Practical setup: Cursor, Claude Code, and Codex can all be configured against `127.0.0.1:8080` simultaneously. Requests queue; each completes at full speed in turn.

## Memory

The heap is one LPDDR pool. There is no host↔GPU expert copy in the CUDA sense.

Leave-free (when you omit `--leave-free`):

| Installed RAM | Left free |
|---|---|
| < 12 GB | half, cap 4 GB |
| 16 GB | 6 GB |
| 24 GB | 8 GB |
| 36 GB | 10 GB |
| 48 GB | 12 GB |
| 64 GB+ | 16 GB |

Admission uses **active** bytes for MoE (shared + top-k experts), **stored** bytes for dense.

`C` at load (`SLOTBANK_SLOTS`, default `auto`):

| Situation | `C` |
|---|---|
| Small MoE that already fits (OLMoE) | `2 × top-k` |
| Large MoE that does **not** fit stored (35B on 24 GB) | `max(2 × top-k, E/8)`, clamped by budget |
| Large MoE that **does** fit stored (35B on 36 GB+) | `C = E` |
| `ram` | always `2 × top-k` |
| `full` | `C = E` if stored fits, else the budget `C` |

Do not fill leftover RAM with slots. The measured optimum is far below the budget, because the pack competes with the page cache holding the rest of the bank.

Pressure (`vm_stat` + `kern.memorystatus_vm_pressure_level`) stops **new** L2 inserts and can drop KV. It does not shed-and-refault L2 experts. L2 is off unless `SLOTBANK_L2=1`.

### Which Macs unlock which models

Metal's recommended working set is ~74–78% of installed RAM. Below that line, use stock. The band above it is what slotbank is for:

| Mac RAM | Working set | Band slotbank unlocks |
|---|---|---|
| 24 GB | ~17.8 GiB | ~18–40 GiB models |
| 36 GB | ~27 GiB | ~27–60 GiB |
| 64 GB | ~48 GiB | ~48–100 GiB |
| 128 GB | ~96 GiB | ~96–200 GiB |

Only the 24 GB row is measured. The rest follows from the same arithmetic and has not been tested.

## Models

Any mlx-lm checkpoint with a `config.json` that yields a memory card. MoE Switch* modules get slots; dense models load as-is. Sharded checkpoints and `BF16` tensors are supported.

- `mlx-community/Qwen3.5-35B-A3B-4bit` — the reference big-MoE target; measured above
- a local OLMoE 4-bit folder — good for measuring the slot path cheaply

`slotbank` will not guess a card. If it cannot read bits and a parameter count, it refuses.

## CLI

```
slotbank admit    --model PATH [--leave-free 8g]
slotbank generate --model PATH --prompt TEXT [--max-tokens N] [--temp 0] [--leave-free 8g]
slotbank serve    --model PATH [--host 127.0.0.1] [--port 8080] [--api-key KEY] [--leave-free 8g]
```

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `SLOTBANK_SLOTS` | `auto` | `ram` / `auto` / `full` |
| `SLOTBANK_L2` | off | `1` to keep evicted experts in an in-process L2 |
| `SLOTBANK_WAVES` | off | `1` for prefill in waves of `C` experts |
| `SLOTBANK_WARM` | on | `0` to skip the hot-expert warm pass at load |
| `SLOTBANK_WARM_GIB` | adaptive | warm-budget override in GiB; default scales with memory pressure |
| `SLOTBANK_WIRED_MIB` | auto | wired-limit override in MiB; `0` keeps the pack-only limit |
| `SLOTBANK_PREFIX_CACHE` | off | `1` to reuse KV across requests sharing a prefix (agentic workloads) |

`SLOTBANK_WAVES=1` makes prefill ~1.3× faster (49.4 s → 37.3 s at 865 tokens). It splits one gather into several, which changes float16 split-K rounding, so output stops being bit-identical to stock — 1 ULP per layer, ~4% relative on final logits. Greedy tokens matched 48/48 when measured, but bit-identical is the accuracy contract, so it is off by default. It does **not** reduce the prefill peak.

## Limits

- Apple Silicon only. No CUDA objects, no Intel Mac.
- One in-flight generate. The Metal thread is exclusive. This is deliberate — see [Running several agents at once](#running-several-agents-at-once).
- **Prefill is the context ceiling**: 105.6 s and 9.3 GiB peak at 4096 tokens. Extrapolating, peak meets the working set somewhere around 32k. Decode itself is flat in context.
- **Cold start still costs ~3.5×** versus warm (2.24 vs 7.96 tok/s). The expert bank is 16.9 GiB against ~12–16 GiB of usable page cache, so it nearly fits but not quite.
- Slower than stock mlx-lm on any model that fits in the working set.
- Tool calls: Qwen `<tool_call>{...}</tool_call>` and the Qwen3 tool markers. Other markup stays in the text.
- No multimodal, no JSON-schema constrained decoding, no stored Responses.
- Benchmarks are one machine, two models, one session. The mechanism is architectural and should generalize; that has not been verified.

## Known next steps

Ordered by measured headroom:

1. **Offline expert-frequency reordering** — rewrite the checkpoint so hot experts are contiguous. Now mostly a load-time win (faster warm pass), not a decode win, since warm start already captures the residency benefit.
2. **Device-side scatter** — worth 12% of decode, but no working design: a device-side scatter needs the source rows device-resident, and the architecture is that they live in a file. Every formulation either copies the whole pack (the bytes-touched bug) or needs the miss set on the host anyway, which is the sync being removed. The per-layer sync looks inherent to "routing on GPU, expert bytes on disk".
3. **Smaller quantization** — only with a *calibrated* 3-bit (DWQ/AWQ). Naive 3-bit is measured below and is not worth it.

### Burst vs sustained — quote the sustained number

Short benchmarks on this machine overstate throughput badly, for two compounding reasons: the page cache is still holding the previous run's experts, and **the GPU clock ramps under sustained load** (a pure GEMM loop goes 0.77 → 2.01 TFLOPS fp16 over 20 s; the compute ceiling measured 14.2 → 20.4 tok/s across 8 consecutive reps).

A 240-token generation on an otherwise quiet machine, in 20-token windows:

```
3.15  3.30  3.90  3.16  3.85  3.83  4.21  4.16  4.33  4.71  4.14  3.99
```

**Sustained: ~3.9 tok/s, rising to ~4.7** as the GPU clock ramps. (An earlier measurement, before the wired-limit fix and warm start, gave ~2.7.) Short 30-token runs on the same machine report 2.5–3.0, and occasionally 6–8 when the cache and clock both happen to be favourable. Only the sustained figure describes what a user generating a real answer experiences — and it is the figure that matters most for reasoning models, which emit long chains of thought.

Prompt content is a minor effect by comparison: 2.87 tok/s on a trivial prompt vs 2.57 on a technical paragraph (~12%).

### Adaptive warm budget

The warm pass claims page cache for hot experts. A fixed fraction of the working set is antisocial on a busy machine — it evicts other applications' cache and is evicted straight back — so the budget scales with live memory pressure:

| condition | share of ceiling |
|---|---|
| pressure normal | 100% |
| `should_shed` (free < 256 MiB) | 50% |
| pressure warn (2) | 25% |
| pressure critical (4) | 0% — claim nothing |

Ceiling is `max_working_set / 3`. `SLOTBANK_WARM_GIB` overrides it outright.

Measured: tok/s is flat across budgets on a warm machine (1 GiB -> 3.32, 2 GiB -> 3.88, 3 GiB -> 4.15, 4.2 GiB -> 3.81), so claiming less costs nothing there — while a genuinely cold start still gets the full 3.4x when the machine is idle enough to warrant it. `C` is still frozen at load and never resized mid-generate.

### Warm start

slotbank records which experts each layer kept resident (`~/.cache/slotbank/hot-<hash>.json`, written on close) and faults them into the page cache at load, before the first token. Measured against a genuinely cold cache — evicted by streaming 14 GiB of an unrelated file between every run:

| | `SLOTBANK_WARM=0` | warm on (default) |
|---|---|---|
| rep 1 | 2.42 tok/s | **7.63** |
| rep 2 | 2.21 tok/s | **8.86** |
| rep 3 | 2.44 tok/s | **7.61** |
| load time | 3.5–3.8 s | 4.6–5.3 s |

**~3.4× on cold start for ~1.2 s of extra load.** This is the single largest win available, because the cold/warm gap was never compute — it was page-cache residency. The warm pass touches one byte per page rather than copying rows, so it costs no heap and no MLX allocation, and it is bounded by a third of the working set.

It also makes offline expert reordering much less valuable than the skew numbers below suggest: reordering would mainly make the *warm pass* sequential (~1.2 s → maybe 0.4 s), since the decode benefit it targeted is already captured here.

### Expert routing is heavily skewed

Sampled over three prompts (code, history, arithmetic), 40 layers, `Qwen3.5-35B-A3B`:

| | measured | uniform routing would be |
|---|---|---|
| distinct experts used per layer | 169 / 256 | 256 |
| traffic taken by top 32 experts (C=32) | **53.7%** | 12.5% |
| traffic taken by top 64 experts | **76.2%** | 25.0% |

This is why the LRU works: the 56% hit rate at C=32 tracks the 53.7% concentration almost exactly. It also means an offline reordering of the checkpoint by expert frequency would not change the miss *rate* — the LRU already captures the head — but would make the cold tail contiguous, which is where the scattered reads are.

### Quantization: 4-bit vs 3-bit

Measured by quantizing a bf16 model directly (`Llama-3.2-1B`, dense, group size 64), so this is the shape of the loss rather than the exact MoE figure:

| config | perplexity | vs bf16 | weights | vs bf16 |
|---|---|---|---|---|
| bf16 | 3.269 | 1.000× | 2.302 GiB | 1.00× |
| 4-bit | 3.381 | 1.034× | 0.647 GiB | 0.28× |
| **3-bit** | **4.319** | **1.321×** | 0.504 GiB | 0.22× |
| 2-bit | 19872 | 6079× | 0.360 GiB | 0.16× |

**3-bit costs ~28% worse perplexity than 4-bit to save ~22% of the size.** Dropping the 16.9 GiB bank to ~12.7 GiB would fit page cache and roughly double throughput, but the quality cost exceeds the reason you wanted the larger model. Naive 3-bit is not recommended; a calibrated 3-bit (DWQ/AWQ) is the only version worth testing.

### Running more than one model

| scenario | model A | model B | combined |
|---|---|---|---|
| 35B alone | 7.96 tok/s | — | 7.96 |
| 35B + 35B (two processes) | 1.82 | 1.82 | **3.61** |
| 35B + OLMoE | **7.48** | **11.52** | coexist |

One large model alongside small ones is nearly free — OLMoE's whole 3.6 GB bank fits beside the 35B's working set, costing it ~6%. Two large models thrash: same contested bank, double the wired pack, and neither keeps its page cache. **Pair one large model with small ones; do not run two large ones.**

### Wiring the working set, not just the pack

`HotResidency` sizes `mx.set_wired_limit` from expert-pack bytes alone (2.11 GiB for the 35B), which leaves ~1.4 GiB of dense weights and KV in evictable memory — exactly what another application will push out. The wired limit is a **cap, not a reservation**, so raising it to the admitted working set costs nothing when the model is smaller than the cap.

Interleaved A/B, 4 pairs, alternating order:

```
pack-only limit:   4.59  7.60  5.87  3.30     median 5.2
working-set limit: 6.36  8.35  6.02  3.08     median 6.2
```

**~+19%**, consistent with an independent +18% measured earlier under load. `SLOTBANK_WIRED_MIB` overrides it (`0` restores the pack-only behaviour).

### GPU allocation: what is and is not a lever

Three things worth recording, because they look promising and two of them are not:

- **Dynamic Caching (M3+)** does not apply. It allocates *on-chip* memory — registers, threadgroup and tile memory — in hardware for shader occupancy. It is automatic, has no API, and never touches unified-memory residency or file I/O, which is where 75% of decode time goes.
- **There is no Metal priority/QoS API for compute.** A GPU-using application cannot be preempted, which is why contention cannot be engineered around.
- **MLX's allocator cache is not the problem.** Measured during decode: **74.6 MiB**, against 3542 MiB active. `mx.set_cache_limit` has nothing meaningful to reclaim.

The one knob that mattered was `set_wired_limit`, above.

### Contention, not thermal

An earlier version of this file claimed thermal throttling after the compute-bound ceiling fell from 19.23 to 8.30 tok/s. **That was wrong.** A pure GEMM benchmark ramps *up* under sustained load (0.77 -> 2.01 TFLOPS fp16 over 20 s) rather than decaying, which rules throttling out.

The real cause was another application using the GPU. Measured with a game running:

| | GPU to itself | sharing with a game |
|---|---|---|
| compute-bound ceiling | 19.23 tok/s | 8.30 tok/s |
| decode | 7.6 – 8.9 | **1.8 – 2.4** |

**Anything else using the GPU roughly quarters throughput.** On a 10-core fanless Air the GPU is one shared resource and there is no priority control to work around it.

The consolation is that the split still favours optimisation work: even while contended, decode is **75% file I/O and only 14% GPU**, so the bottleneck is not the thing being fought over.

**Benchmark methodology.** Never compare a code change across sessions. Interleave A/B within one session, alternating order. Three separate conclusions in this project's history were misattributed to code before the machine state was checked.

### Predictive expert prefetch does not beat a static profile

The router computes a softmax over all 256 experts and discards everything below top-k, so the "nearly selected" experts are free information. Testing whether they forecast the next token's routing (40 layers, 30 decode steps):

| warm set at token t | covers token t+1 | static global-frequency set, same size |
|---|---|---|
| top-10 (the selection itself) | 37.2% | — |
| top-16 | 46.6% | — |
| top-32 | 58.8% | 53.7% |
| top-64 | 62.3% | **76.2%** |

Only **37% of a token's experts survive to the next token** — routing churns fast, which is why the LRU misses ~44% of the time even when working correctly. Router-weight prediction helps slightly at small cache sizes but saturates near 62%, while a static frequency profile keeps improving to 76%.

**The hot region of the routing distribution is stable globally, not locally.** A recorded profile captures it better than per-token forecasting, at zero inference cost. This is why `warm_from_profile` is the right design and a predictive prefetcher would be a regression.

### Batching amortises I/O, and warm start already removed the I/O

With `_is_decode` fixed to recognise batched decode (see below), B sequences share one ensure and one copy per layer. Distinct experts needed per layer: 15.1 at B=1, 21.6 at B=2, 28.9 at B=4 — sublinear, so the reads do amortise. Against a **cold** baseline, B=2 measured 2.28x aggregate throughput.

But strictly alternated on a **warm** cache:

```
B=1:  3.87  3.97  2.90     median 3.87
B=2:  3.37  3.18  2.95  2.81   median 3.10
```

B=1 wins. Batching amortises I/O, and warm start already eliminated the I/O; what remains scales linearly with B because it is compute. Memory is not the obstacle — B=2 costs only +66 MiB, since KV is ~20 KiB/token across 10 of 40 layers.

**Bug fixed regardless:** `_is_decode` tested `shape[0] == 1`, hardcoding batch size 1. Any batched call fell silently into the *prefill* path — no LRU, a `.tolist()` of routing ids on the decode path, and ~8x slower steps. It now tests the token axis (`shape[-2] == 1`).

### Prefix cache (opt-in)

`SLOTBANK_PREFIX_CACHE=1` caches KV/recurrent state at 128-token block boundaries, so agents sharing a system prompt prefill it once. Two architectural constraints shape the design:

- **The cache cannot be trimmed.** 30 of 40 layers are `ArraysCache` holding recurrent linear-attention state, which cannot be rolled back — `can_trim_prompt_cache` is `False`. So a reusable prefix must land exactly where prefill stopped, which is why snapshots are taken on block boundaries rather than by trimming a longer entry.
- **Each snapshot costs ~63 MiB**, nearly all fixed recurrent state regardless of prefix length.

Measured on 4 agent spawns sharing a 179-token system prompt:

| spawn | no cache | with cache |
|---|---|---|
| 1 | 24.00 s | 39.12 s (snapshot taken) |
| 2 | 26.54 s | 16.90 s |
| 3 | 26.78 s | 19.33 s |
| 4 | 26.76 s | 18.32 s |
| **total** | **104.08 s** | **93.67 s** → **1.11x** |

Output is identical in every case. Break-even is ~3 spawns, improving with N. **Off by default**, because a one-off generation pays the ~15 s snapshot for nothing. Boundaries already cached are not re-snapshotted, so repeat requests pay no split cost.

### Multi-token decode passes amortise ~3x (and unlock speculative decoding)

`_is_decode` treated any `T > 1` as prefill, so a multi-token pass took the throwaway-stack route — every unique expert restacked and re-read, no LRU. Verifying 8 candidate tokens cost 7x one token instead of ~2x.

The slot path is safe for `T > 1` whenever **`n_ids <= capacity`**, which bounds the unique expert count to the pack size. Without that bound the LRU could evict an expert before the gather reads it, so the limit is correctness, not just speed. Larger passes still fall back to prefill.

With the path enabled (C=48), per-token cost:

| T | OLMoE ms/token | vs T=1 | 35B-A3B ms/token | vs T=1 |
|---|---|---|---|---|
| 1 | 61.2 | 1.00x | 441.9 | 1.00x |
| 2 | 26.3 | **2.32x** | 187.9 | **2.35x** |
| 4 | 23.7 | **2.59x** | 142.8 | **3.09x** |
| 6 | 14.9 | **4.10x** | 184.8 | 2.39x |

Bit-identical to stock at every T tested. Before the fix, T=2 measured **0.31x** — worse than single-token.

**This is what makes speculative decoding / MTP viable here.** Verification is a `T=k` pass, not a batched one, so it uses this path. End-to-end speculative speedup is roughly `acceptance_rate x amortisation`; at T=4 the amortisation is ~3x, so a draft model with ~60% acceptance would put the 35B near 8-12 tok/s. The acceptance rate is unmeasured — a draft model or an MTP checkpoint is needed to close that loop.

Note this is the **sequence** dimension. The batch dimension (B>1) remains pathologically slow for reasons not attributable to the copy path, `gather_qmm`, or the ensure kernel — see below.

### Compatibility checks

Two paths produce silently wrong output rather than an error, so `admit.py` exposes
checks for both. Neither imports MLX (see the import fence).

- `check_draft_compatible(target, draft)` — speculative decoding needs an identical
  vocabulary. `Qwen3` (151936) and `Qwen3.5` (248320) look interchangeable and are not.
- `check_speculative_supported(cache)` — rejects untrimmable caches. Duck-typed on
  `is_trimmable()`, mirroring `can_trim_prompt_cache`.

`docs/mlx-lm-speculative-bug.md` writes up the upstream defect these guard against.

### Speculative decoding is unsafe on hybrid linear-attention models

The `T>1` amortisation above is the verification path for speculative decoding, but **neither speculative route currently works on `Qwen3.5-35B-A3B`**.

**Draft-model route — silently wrong.** `speculative_generate_step` rewinds the target cache when draft tokens are rejected:

```python
def _rewind_cache(num_draft, num_accept):
    cache.trim_prompt_cache(model_cache, num_draft - num_accept)
```

`trim_prompt_cache` guards itself (`if not can_trim_prompt_cache(cache): return 0`), so nothing is corrupted — but nothing is trimmed either, and `speculative_generate_step` has **no `can_trim` check of its own**. The cache keeps the state of rejected tokens, and everything generated afterwards is conditioned on tokens that were never emitted. No crash, no warning.

| model | cache composition | trimmable | speculative safe? |
|---|---|---|---|
| OLMoE-1B-7B | 16 KVCache | **True** | yes |
| Qwen3.5-35B-A3B | 30 ArraysCache + 10 KVCache | **False** | **no — silently wrong** |

Any model with linear-attention or recurrent layers is affected, because that state is a running summary that cannot be rolled back. This is the same constraint that limits the prefix cache to block granularity.

**MTP route — not available either.** MLX "MTP" checkpoints contain **zero MTP tensors** (`AX-Qwen3.6-35B-A3B-MLX-4bit-MTP`: 2090 tensors, 0 mtp) because they were converted with mlx-lm, whose `sanitize` drops `mtp.*`. The originals do have them — `Qwen/Qwen3.6-35B-A3B` carries 19 MTP tensors forming a complete extra MoE layer (256 experts, gate, shared expert, norms) plus `mtp.fc`. Using MTP therefore needs a re-conversion from the ~70 GB original **and** an MTP layer implementation in the model class; mlx-lm has neither.

**What this means:** the `T>1` slot path is correct and fast, but it is infrastructure waiting for a compatible source of candidate tokens. On a pure-attention MoE it can be fed by a vocab-matched draft model today. On hybrid models it needs MTP, which is a build.

Note also that draft models require an **identical vocabulary**: `Qwen3.5-35B-A3B` uses vocab 248320, while `Qwen3-0.6B`/`1.7B` use 151936 — a silent incompatibility worth checking at admit time.

### Batching does not scale here — on either model

Tested on both regimes. `Qwen3.5-35B-A3B` (bank exceeds page cache, I/O-bound) and `OLMoE` (bank fits, compute-bound):

```
OLMoE, warm:   B=1  53.6 ms/step  18.64 tok/s aggregate
               B=2 261.4 ms/step   7.65
               B=4 555.1 ms/step   7.21
               B=6 736.3 ms/step   8.15
```

Per-step cost grows **superlinearly** in B (2x work costs 3-5x), so aggregate throughput falls. This holds after removing the ensure kernel's O(n^2) dedup — which was a real inefficiency scaling as `(batch * top_k)^2` on a single GPU thread, but not the binding one.

The one regime where batching wins is a **cold** cache, where it amortises expert reads: B=2 measured 2.28x aggregate against a cold B=1. Warm start already removes that I/O, so the win does not survive.

**For agentic throughput, queue rather than batch.** N agents on one `slotbank serve` each get roughly 1/N of single-stream throughput, and single-stream is the number to maximise — via a model whose bank fits page cache, not via concurrency.

### On vLLM

- **PagedAttention** targets KV fragmentation at high batch counts. KV here is ~20 KiB/token over 10 of 40 layers; B=2 costs +66 MiB. Not the bottleneck.
- **Continuous batching** would matter only if batching paid off, and above it does not.
- **Prefix caching** is the one idea worth borrowing: N agents sharing a system prompt each pay full prefill (8-12 s) today, because `reuse_prefill_start` only reuses an exact prefix of the same sequence.

vLLM has no Metal backend, so this is idea-borrowing rather than a dependency.

### Tried and rejected

- **One span read covering min..max expert.** Fine for dense prefill sets (65.1 ms vs 64.7 for plain pread) but catastrophic for decode: 21.6 ms vs 0.6 ms, because it reads a 200-row range to fetch 3.
- **Expert-interleaved checkpoint.** A miss costs 9 reads (gate/up/down x weight/scales/biases) scattered across separate tensors. Rewriting the bank so one expert's 1.6875 MiB is contiguous measured **2.6x faster for identical bytes** (774 vs 2005 us/expert; 2180 vs 842 MiB/s). Built (16.88 GiB in 97 s), verified bit-identical on 63 rows, wired into the copy path as one `pread` per miss expert. End-to-end it was **1.7x slower** (1.52-1.80 vs 2.82-3.17 tok/s, six interleaved pairs, including after fixing the warm pass to target the new file). Reverted.

  Three read-pattern micro-benchmarks have now predicted large wins that reversed end-to-end (sidecar, threaded `pread`, interleaving). **Micro-benchmarks of read shape do not predict decode throughput on this system** — measure end-to-end, interleaved, before believing any of them.
- **Hot-expert sidecar.** The profile identifies the ~64 experts per layer carrying 76% of traffic — only 4.2 GiB. Copying those into one contiguous file should have let them be read sequentially and stay in page cache. Built in 11.2 s, verified bit-identical. Per-row reads did get faster (570-685 us vs 1116 us from the checkpoint) — but end-to-end decode was **2.5x slower** (0.57-0.70 vs 1.62-1.68 tok/s, three interleaved pairs). Best explanation: the sidecar duplicates bytes that already exist in the checkpoint, so page cache holds two copies and the effective cache halves, while two separate mappings defeat the kernel's readahead on both. A version that *replaced* rather than duplicated (i.e. full checkpoint reordering) would not have this flaw.
- **Parallel per-layer decode reads.** A layer misses ~3 experts x 9 banks = ~31 rows, previously read serially on one core. `pread` releases the GIL (mmap slices do not), so a pool should have used more than one core. Interleaved: serial median 2.94, 4 threads 2.79, 8 threads 2.98 tok/s — no effect. The kernel's readahead is already providing the parallelism.
- **One array + one scatter per bank** (instead of one `pack[s] = row` per expert per bank). The per-row form builds ~1100 MLX graph nodes per token, so this looked like obvious dispatch savings. Interleaved A/B: median 2.55 vs 2.69 tok/s — slightly *worse*. Joining the rows into one buffer costs more than the graph nodes it saves. A `mx.stack` variant was worse still and tripped the resident-memory test.
- **Threaded `pread` for batched reads.** 3.6× on a cold micro-benchmark (96.5 → 26.6 ms for 200 rows), but a **regression end-to-end** — prefill at 4096 tokens went 105.6 s → 112.7 s. Real prefill hits substantially warm pages, where mmap is a memcpy and pread is a syscall plus a copy, and ~120 batched calls per chunk each dispatch hundreds of pool tasks. Reverted. The micro-benchmark did not predict the real workload because it used fresh tensors, so every read was cold.

## Tests

```bash
python -m pytest tests -q
```

51 tests. `tests/test_offload_cache.py` and `tests/test_expert_slots.py` need `mlx` and `mlx-lm`. The rest is CPU.

Only `runtime.py`, `expert_slots.py`, and `offload_cache.py` may import MLX; `tests/test_fence.py` enforces this with an AST check.

## License

Apache License 2.0. See `LICENSE`.
