# slotbank

Run Mixture-of-Experts models on Apple Silicon without wiring the whole expert bank into Metal. A C-slot LRU keeps a compact pack resident; misses read from a file-backed bank; `gather_qmm` sees only the pack. macOS keeps a leave-free reserve so the machine still multitasks.

No PyTorch. Inference is MLX. One process owns the model.

**What it is for:** models whose expert bank exceeds the Metal working set. On a 24 GB Mac that is roughly 18–40 GiB of weights. Below that line, stock mlx-lm is faster and you should use stock. Above it, stock and llama.cpp do not run at all. A dense checkpoint that **fits** the working set (Qwen3.8-27B mixed 3.5bpw on this Air) is also served here: same HTTP door, stock mlx-vlm load, no expert slots. See [Benchmarks](#benchmarks) and [Qwen3.8-27B on 24 GB](#qwen38-27b-on-24-gb).

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
- **CLI** — `generate`, `serve`, `admit`, `context`

Not included, on purpose: CUDA graphs, PCIe overlap, hybrid CPU experts, tensor parallelism, radix/semantic cache, a second draft model, a desktop GUI, a 1M SSD KV pager, layer-streaming dense weights.

## Qwen3.8-27B on 24 GB

Dense 27B has no expert subset. The slotbank move is the same one MoE already makes: **shrink the bytes that must stay hot until they fit in RAM**, then run stock mlx-vlm. Layer streaming from SSD is the fallback if even 3-bit OOMs; it was not needed here.

**~1 tok/s vs ~10 tok/s.** Mixed 3.5bpw on this Air is **~1 tok/s** — stock mlx-lm, not a slotbank wrapper tax. The 7–10 tok/s RAM-bandwidth estimate assumes a uniform 4-bit GEMM; this pack is mostly 3-bit overrides and lands at the kernel ceiling. **Daily 27B is 4-bit + native MTP** (mlx-vlm exact verify, not mlx-lm trim). DFlash2 is still valid; on this Air MTP is faster and lighter.

| Pack | Path | This Air (measured 2026-08-31) | Published same-class |
|---|---|---|---|
| `~/Models/Qwen3.8-27B-3.5bpw` | stock mlx-lm / slotbank text-only | **0.94–1.11 tok/s**, peak 12.8–14.2 GiB | ~11 tok/s on M2 Pro 32 GB (rapid-mlx card) |
| `~/Models/Qwen3.8-27B-4bit` | greedy | **5.71 tok/s**, peak 14.4 GiB | 5–6.5 tok/s on M4 mini 24 GB |
| 4-bit + `Qwen3.8-27B-DFlash2-4bit` | `--draft` trained K=8 | **11.76** count / **9.10** code, peak 16.5–17.0 GiB | **11.7–12.2 tok/s** on M4 mini 24 GB |
| 4-bit + `Qwen3.8-27B-MTP-4bit` | `--draft` trained K=3 | **13.47** count / **9.95** code, peak 14.9–15.1 GiB | native MTP sidecar ~228 MB |

```
hf download mlx-community/Qwen3.8-27B-4bit --local-dir ~/Models/Qwen3.8-27B-4bit
hf download z-lab/Qwen3.8-27B-DFlash2 --local-dir ~/Models/Qwen3.8-27B-DFlash2
python -m mlx_vlm.convert --hf-path ~/Models/Qwen3.8-27B-DFlash2 \
  --mlx-path ~/Models/Qwen3.8-27B-DFlash2-4bit --quantize --q-bits 4 --q-group-size 64

PYTHONPATH=src .venv/bin/python -m slotbank.cli serve \
  --model ~/Models/Qwen3.8-27B-4bit \
  --draft ~/Models/Qwen3.8-27B-MTP-4bit \
  --leave-free 6g --thinking --vision
```

`--leave-free 6g` (18 GiB working set) is required: ~15 GiB 4-bit + a drafter does not fit the default 16 GiB (leave-free 8). Native MTP is the measured daily drafter (~228 MB, trained K=3). DFlash2 still works (trained K=8, ~1 GiB). mlx-lm trim speculation stays **off** on this hybrid (48 Gated DeltaNet layers). `--draft` uses mlx-vlm's `rollback_speculative_cache`, which can rewind GDN state. KV quant is off on the draft path (the verifier reads `keys.shape`).

**3.5bpw** (`rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX`) still fits at leave-free 8 and is the quality pack (MMLU-Pro 61.0 vs 54.2 for uniform 3-bit). It is not the local-use speed pack.

**Context.** 16 full-attention layers × 4 KV heads × 256 × 2 × 2 bytes ≈ **64 KiB/token**, plus ~150 MiB fixed Gated DeltaNet state. Practical window **8k–16k**. 32k is a stretch (and what Oh My Pi needs advertised: its tool catalog already blows a 16k cap, which is why a first `hi` showed `159.5%/16K` and stalled). 128k+ does not fit. `can_trim_prompt_cache` is false — do not page or LRU the hybrid cache. The DFlash path keeps the live cache across turns and prefills only an **appended** suffix (new prompt starts with every token already cached). Measured on this Air: 819-token prefix reuse **0.88 s vs 17.0 s cold** (19×), greedy `bit_match=True`. Editing or reordering history is a full prefill, not a trim. Longer history than the window is the [context OS](#context-os): append-only disk log + verbatim excerpts, not a 1M KV tensor.

**Thinking** is off unless `SLOTBANK_THINKING=1`. **Vision** is off unless `SLOTBANK_VISION=1` or `--vision`. Latency tests should keep both off.

**Serve + OMP.** One process, existing Anthropic / OpenAI doors:

```
PYTHONPATH=src .venv/bin/python -m slotbank.cli serve \
  --model ~/Models/Qwen3.8-27B-4bit \
  --draft ~/Models/Qwen3.8-27B-MTP-4bit \
  --leave-free 6g --thinking --vision
```

Point Oh My Pi at `http://127.0.0.1:8080`. Run `slotbank omp --model ~/Models/Qwen3.8-27B-4bit --thinking --vision` (no weight load) or `slotbank serve` — both write static `llama.cpp` / `lm-studio` rows into `~/.omp/agent/models.yml`. Then **`omp models refresh`**. Start omp from `~` or `/tmp`, not this checkout: OMP dumps the project tree into the prompt, which is why a first `hi` from `~/Desktop/slotbank` sat at ~80% of 33K and prefills for minutes. Picker `llama.cpp/Qwen3.8-27B-4bit` has tools off (fast hi); `llama.cpp/Qwen3.8-27B-4bit-agent` is the tool/thinking door. Snippet: [`examples/omp-qwen38.yml`](examples/omp-qwen38.yml). A sibling `Qwen3.8-27B-MTP-4bit` is auto-attached if you omit `--draft`. For 30+ tok/s / less RAM, use the 4B snippet below.

**Not built, on purpose.** No `layer_slots.py` (resident 4-bit is the speed path). No mlx-lm trim speculation on hybrid. No Apple Neural Engine: MLX 0.32.1 exposes only Metal GPU, and an ANE/Core ML copy of 27B is fp16 with a compile spike that does not fit beside 15 GiB of 4-bit weights on 24 GB. Drafters stay on GPU so verify stays bit-identical to the target. The `mtp/` folder in the 3.5bpw pack is unused; daily MTP is `~/Models/Qwen3.8-27B-MTP-4bit`.

### 30+ tok/s (less RAM)

27B 4-bit cannot hit 30 on 120 GB/s: a full-pack read is ~10 tok/s, native MTP measured **13.47** count / **9.95** code. Qwen3.5-9B is still GDN-compute-bound here (**6.4** raw, **13.8** DFlash). **Qwen3.5-4B 4-bit** is the pack that clears 30 on this Air (greedy, thinking/vision off, 2026-08-29 afternoon):

| Pack | Path | tok/s | peak | `parse_iso_dates` to EOS |
|---|---|---|---|---|
| `Qwen3.5-4B-4bit` | stock mlx-vlm | **32.68** | 2.69 GiB | — |
| + `Qwen3.5-4B-DFlash-4bit` | `--draft` block 5 | **46.39** raw / **54.32** slotbank | 2.87 GiB | **102** tokens, `bit_match=True` |
| `Qwen3.5-9B-4bit` + DFlash | same | 13.75 | 5.70 GiB | 162 tokens |
| `Qwen3.8-27B-4bit` + MTP | trained K=3 | **13.47** count / **9.95** code | 14.9–15.1 GiB | same 64-token window |
| `Qwen3.8-27B-4bit` + DFlash2 | trained K=8 | **11.76** count / **9.10** code | 16.5–17.0 GiB | 156 tokens (older to-EOS run) |

4B finished the same closed task in **fewer** tokens than 27B. Output is bit-identical to 4B greedy, not to 27B — keep the 27B serve line when you want that model.

```
hf download mlx-community/Qwen3.5-4B-4bit --local-dir ~/Models/Qwen3.5-4B-4bit
hf download z-lab/Qwen3.5-4B-DFlash --local-dir ~/Models/Qwen3.5-4B-DFlash
python -m mlx_vlm.convert --hf-path ~/Models/Qwen3.5-4B-DFlash \
  --mlx-path ~/Models/Qwen3.5-4B-DFlash-4bit --quantize --q-bits 4 --q-group-size 64

PYTHONPATH=src .venv/bin/python -m slotbank.cli serve \
  --model ~/Models/Qwen3.5-4B-4bit \
  --draft ~/Models/Qwen3.5-4B-DFlash-4bit
```

Default leave-free 8g is enough (~3 GiB hot). Omit `--draft` if you only need ~33 tok/s and 2.7 GiB. OMP: [`examples/omp-qwen35-4b.yml`](examples/omp-qwen35-4b.yml).

### Context OS

`slotbank context` keeps an append-only `log.jsonl`. `compile` selects newest verbatim spans that fit a token budget (default **4096**) and expands `file:path:lo-hi` pointers only when the whole span fits `SLOTBANK_CONTEXT_EXPAND` (default **1024** tokens). Oversized files stay on disk as citations. It does not paraphrase. One directory per local issue so tasks do not share one KV. Optional `SLOTBANK_CONTEXT_COMPILER_URL` POSTs `{log, repo, budget}` and must return `{working_set}`; offline falls back to the local extract.

Serve injects the compiled working set as a system prefix when `SLOTBANK_CONTEXT_DIR` is set. `--leave-free 6g` is the multitask reserve; a 4k working set is ~256 MiB of KV plus a bounded prefill, so that reserve stays for the rest of the machine. Prefix KV reuse (`SLOTBANK_PREFIX_CACHE=1`) only hits when that prefix is byte-identical — the existing hybrid snapshot path. Reordering excerpts is a new prefill, not a trim. `SLOTBANK_CONTEXT_EXPAND=0` is citations only (smallest context-stage RAM).

```
slotbank context init --dir ~/.slotbank/sessions/job
slotbank context append --dir ~/.slotbank/sessions/job --role user --content "..." --pointer file:src/foo.py:10-40
slotbank context compile --dir ~/.slotbank/sessions/job --repo . --budget 4096
SLOTBANK_CONTEXT_DIR=~/.slotbank/sessions/job \
SLOTBANK_CONTEXT_REPO=. \
SLOTBANK_CONTEXT_BUDGET=4096 \
  slotbank serve --model ~/Models/Qwen3.8-27B-4bit --draft ~/Models/Qwen3.8-27B-DFlash2-4bit --leave-free 6g
```

## Benchmarks

Measured 2026-08-23 on a fanless **M4 Air, 10 GPU cores, 24 GB unified, 16 KB pages**, `mlx==0.32.1`, `mlx-lm==0.31.3`. Metal reports a 17.76 GiB max recommended working set. Every number is from a script in this repo's history; each config ran cold first, then repeated, and both are shown because the spread is the story.

### Headline: a model no other runtime tested here can run at 4-bit

`Qwen3.5-35B-A3B` 4-bit — 40 layers, E=256, top-k 8, **19.02 GiB of weights** (16.9 GiB of it experts).

**Three throughput numbers appear throughout this document and they measure
different things.** Every figure below carries one of these labels; an unlabelled
number is an editing mistake, not a fourth kind.

| Label | Clock starts | Includes | Answers |
|---|---|---|---|
| **cold end-to-end** | process launch | load + prefill + decode | what you actually wait for |
| **warm sustained decode** | first token | decode only, over the whole run | steady-state serving |
| **compute-only ceiling** | — | GPU math with I/O removed | how much headroom is left |

They diverge by design. The same C=32 configuration on the 35B is **4.2 cold
end-to-end** to 256 tokens, **6.1 warm sustained decode**, and short warm windows
reach 6.5–7.8 because a brief burst can run almost entirely out of experts that
are already cached. Quoting the largest of those as "the" throughput is the
usual way these comparisons become dishonest.

| Runtime | Quant | Result | Active memory | Machine |
|---|---|---|---|---|
| **llama.cpp** (5 configs) | Q4_K_S, 19.25 GiB | **cannot run** | Metal OOM | drove to critical pressure twice |
| **stock mlx-lm** (full bank) | MLX 4-bit, 19.02 GiB | 0.005 tok/s (208 s/token) | 18.225 GiB | 13.2 GB compressed, thrashing |
| **mlx-vlm** 0.6.15 (full bank) | MLX 4-bit, 19.02 GiB | **no tokens** — killed loading | n/a (66% -> 7% free) | swap 8.76 / 10.24 GB |
| **slotbank** `C=32` (default) | MLX 4-bit, 19.02 GiB | **4.2 cold end-to-end** to 256 tokens (**6.1 warm sustained decode**) | **~3.8 GiB footprint** | responsive |

> **Counted from cold start, because that is what you actually wait for.**
> A run pays ~19 s of fixed cost before the second token: 8.7 s to load and
> ~10 s of prefill. Quoting only the warm decode rate hides it.
>
> | N tokens | wall from process start | cold end-to-end | warm sustained decode |
> |---|---|---|---|
> | 32 | 18.7 s | 1.71 | 5.5 |
> | 64 | 32.8 s | 1.96 | 4.7 |
> | 128 | 42.1 s | 3.05 | 5.6 |
> | **256** | **61.2 s** | **4.19** | **6.07** |
>
> Effective throughput climbs the whole way because the fixed cost amortises;
> it only approaches the decode rate for long answers. The same run on the
> pre-`preadv`, pre-`memoryview` path took **98 s** for those 256 tokens at
> 2.61 cold end-to-end -- so today's two changes are **1.61x measured from
> cold start**, and 1.88x on decode alone. Short answers see less: 1.20x at
> 32 tokens, because 19 s of load and prefill dominate and neither change
> touches them.

> **On the mlx-vlm row.** It spent 126 s loading and drove free memory from 66%
> to 7% without emitting a token, at which point a watchdog killed it to protect
> the machine. That is *not* proof it cannot run: left alone it would most likely
> have limped like stock mlx-lm (same MLX backend, same absence of offloading)
> rather than hard-failing like llama.cpp. The measurement was stopped by choice,
> and the stock mlx-lm row is the fair stand-in for "no expert offloading". What
> the run does establish is that memory is exhausted **during load**, before
> throughput is even a question.

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

### Choosing C: why more RAM makes it slower

Re-measured after the read-path work, 35B-A3B, cold start to 256 tokens:

| C | active | cold end-to-end (2 reps) | warm sustained decode |
|---|---|---|---|
| 8 | 1.88 GiB | 3.32 / 4.19 | 4.54 / 5.78 |
| 16 | 2.41 GiB | 4.06 / 4.30 | 5.62 / 6.02 |
| **32** | **3.46 GiB** | **4.10 / 4.88** | **5.97 / 7.15** |
| 48 | 4.52 GiB | 3.55 / 4.57 | 5.20 / 6.80 |
| 64 | 5.57 GiB | 3.36 / 3.78 | 4.84 / 5.27 |
| 96 | 7.68 GiB | 3.32 | 4.82 |

The optimum is interior and it is *not* "use more RAM". **Two caches compete
for the same physical memory**: the slot pack (wired Metal) and the OS page
cache holding the 16.9 GiB bank. Every GiB the pack takes is a GiB the cache
loses -- and they buy different things. A slot avoids a *copy*; the page cache
decides whether a miss costs a memcpy or an SSD read.

Slots have sharply diminishing returns because routing is skewed: the top 32
experts carry 53.7% of traffic and the top 64 carry 76.2%, so doubling C from
32 to 64 buys only 22 more points of hit rate. Page cache value is closer to
linear. Direct evidence, alternating the two configs three times against a
fully warm cache so neither benefits from ordering:

| C | tok/s | miss/call | **disk read per token** |
|---|---|---|---|
| **32** | 8.60 / 8.74 / 8.53 | 3.012 | **17.1 / 15.5 / 22.0 MiB** |
| 64 | 7.35 / 6.31 / 5.93 | 2.089 | 24.2 / 39.0 / 41.5 MiB |

**C=64 takes 31% fewer misses and still reads about twice as much from disk**,
because its extra 2.1 GiB of pack came straight out of the page cache serving
the misses it still takes. Fewer, more expensive misses lose to more, cheaper
ones.

This is machine-relative, not universal. The rule is that the pack should grow
only while the rest of the bank still fits in whatever page cache remains. On a
machine where 16.9 GiB of experts fits in cache outright, the trade reverses and
a bigger pack wins. On this 24 GB laptop it does not.

**Measurement warning.** Page-cache state dominates absolute numbers here. Run
C=32 then C=64 and C=64 looks 1.9x faster -- purely because the first run warmed
the cache for the second. Only alternating, repeated runs are meaningful; a
single ordered A/B on this system measures the order, not the config.

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

### The prefetch cost 3x the read it was hinting

Measured 2026-08-24. Profiling the miss path by phase, rather than by stubbing
whole stages out, found the cost was not where the stage-level numbers implied
(35B-A3B, `C=32`, 48 tokens):

| Phase | ms/token | share |
|---|---|---|
| `madvise(MADV_WILLNEED)` prefetch | 179.9 | **45.9%** |
| `mx.eval` + `meta[0].item()` sync (40x) | 124.2 | 31.7% |
| read I/O | 54.6 | 13.9% |
| scatter into pack | 1.5 | 0.4% |
| attention + GEMM + sampler | 31.6 | 8.1% |

Two things fall out. The reads are **not** the problem — 54.6 ms matches what
this SSD should deliver for the bytes involved, within a few percent. And the
*hint* costs three times the read it hints, because `MADV_WILLNEED` is per-page
VM bookkeeping: a layer advises 27 rows x 32 pages, ~34,500 pages per token.

Deleting the advise is not an option — it is worth 2x on its own (2.43 -> 1.20
tok/s), because without it the mmap faults are taken one at a time. The advise
buys fault batching, and pays for it in VM work.

**What this SSD actually wants is request size and queue depth, not locality.**
Uncached (`F_NOCACHE`) reads, MiB/s:

| size | T=1 | T=4 | T=8 | T=16 |
|---|---|---|---|---|
| 16K | 134 | 505 | 856 | 942 |
| 64K | 532 | 1365 | 1944 | 2469 |
| 192K | 995 | 2214 | 3188 | 3573 |
| **512K** | **1342** | 3136 | **4015** | 4046 |

Random and sequential differ by <10% above 64K. Expert rows are 512 KiB
(weights) and 32 KiB (scales/biases), 9 reads per expert — so the serial mmap
path was leaving 3x on the table, and *locality-based* fixes were attacking a
variable this device does not price.

Since `pread` releases the GIL and does no VM work, the fix is to take the
pread path **and skip the advise**, which no earlier experiment had tried —
every previous threaded-read run still called `prefetch` first and so paid both.
`SLOTBANK_READ_THREADS` now defaults to 8, and `_host_copy` advises only on the
serial fallback.

Controlled A/B over a 400-token generation, greedy so both arms route
identically (the per-window `miss/call` sequence is equal to three decimals):

| window | new (pread, 8) | old (mmap + advise) | ratio |
|---|---|---|---|
| 0–50 | 4.45 | 2.79 | 1.60x |
| 50–100 | 5.56 | 3.63 | 1.53x |
| 100–150 | 6.01 | 3.81 | 1.58x |
| 150–200 | 5.09 | 3.21 | 1.59x |
| 200–250 | 5.25 | 3.33 | 1.58x |
| 250–300 | 6.10 | 3.65 | 1.67x |
| **sustained** | **5.2** | **3.3** | **1.58x** |
| best 50-token window | 6.69 | 4.00 | 1.67x |

Output is bit-identical: 40/40 token ids match the mmap path exactly.

Short benchmarks understate this badly. The warm ramp completes around token 50
and is worth ~35%, so a 64-token run reports ~3.9 where the sustained figure is
~5.2. Note that `miss/call` *rises* slightly as the run warms (3.15 -> 3.40)
while tokens get faster — warming is the page cache making each miss cheaper,
not the slot cache taking fewer of them.

### Reading straight into the pack

A miss used to travel `pread` -> `bytes` -> `mx.array` -> `.view()` ->
`.reshape()` -> `pack[slot] = row`. The bytes on disk are *already* in the
pack's layout, so that entire chain is overhead: ~218 MiB/token of transient
host copies and ~1137 MLX graph nodes to place data the kernel can deposit
directly. `os.preadv` writes into `memoryview(pack)` instead.

Seven paired runs, 120 warm tokens each:

| rep | chain | `preadv` | ratio |
|---|---|---|---|
| 1 | 5.920 | 7.989 | 1.35x |
| 2 | 5.468 | 7.436 | 1.36x |
| 3 | 5.909 | 5.858 | 0.99x |
| 4 | 5.562 | 7.295 | 1.31x |
| 5 | 5.811 | 7.261 | 1.25x |
| 6 | 5.751 | 7.625 | 1.33x |
| 7 | 6.317 | 8.171 | 1.29x |
| **mean** | **5.820** | **7.376** | **1.27x** |

Rep 3 is contention, not signal; the median ratio is 1.31x. Output is
bit-identical to the independent mmap path (100/100 token ids), 3771 preadv
calls with zero fallbacks. Guarded by `test_preadv_lands_rows_in_pack_slots`,
which fails if a row offset is wrong.

Writing into an evaluated array's buffer is safe here only because each pack is
written once per token and its gather has been evaluated by the time we return
to it. The helper bails out to the old path if a slot is not exactly one row
wide, so a layout change degrades rather than corrupts.

**Budget after this change** (6.68 tok/s, 149.7 ms/token warm):

| phase | ms/token | share |
|---|---|---|
| **GPU sync (40x)** | **84.2** | **56.3%** |
| `preadv` | 53.5 | 35.8% |
| attention + GEMM + gather | 11.9 | 8.0% |

The scatter, previously 16.4 ms, is gone. Sync now dominates, and with a ~50
ms compute floor roughly 34 ms of it is stall.

#### Why overlapping host and GPU work is not the answer

The obvious next move is to overlap the 53.5 ms of reads with the 84.2 ms of
GPU work — `SwitchGLU` needs `gate`/`up` before `down`, so `down`'s rows could
load while the GPU runs the first GEMM. Measured first, with `mx.async_eval` on
balanced layer-sized work (4.14 ms GPU against 0.72 ms host):

| | ms |
|---|---|
| serial | 5.05 |
| `async_eval` overlapped | 4.83 |
| ideal (`max`, not sum) | 4.14 |

**Overlap efficiency: 24%.** Against the ~18 ms of `down_proj` reads that are
eligible, that is ~4 ms — 2.7% — for a change that would have to split
`copy_missing` by bank group. Not taken.

The same ceiling applies to replacing the 40 blocking waits with Metal
completion handlers: a command-buffer round trip measures ~286 us, so the
round-trip component is ~11 ms/token, and MLX owns command-buffer submission —
slotbank cannot inject a completion handler without patching MLX itself.

### The sync that was half a Python idiom

`mx.eval` makes an array host-readable. Reading a scalar out of it afterwards
with `meta[0].item()` does **not** read that host memory — it builds a fresh
lazy slice on the Metal stream and blocks on a second command buffer:

| route (array already evaluated) | us/call |
|---|---|
| `int(self.meta[0].item())` | **224.7** |
| `int(memoryview(self.meta)[0])` | **0.8** |
| `int(self.meta.tolist()[0])` | 0.6 |

All three return the same value. The copy path runs this once per MoE layer, so
it was **40 x 224 us of avoidable command-buffer latency per token**, hiding
inside what the phase profile attributed to genuine GPU synchronisation.

Reading through the buffer protocol instead, paired runs at 120 warm tokens:

| rep | `memoryview` | `.item()` | saved |
|---|---|---|---|
| 1 | 152.1 ms | 160.5 ms | 8.4 ms |
| 2 | 165.2 ms | 176.9 ms | 11.7 ms |
| 3 | 151.4 ms | 163.9 ms | 12.5 ms |
| **mean** | **6.411 tok/s** | 5.996 tok/s | **+6.9%** |

10.9 ms measured against 9.0 ms predicted from the microbenchmark — the rare
case in this project where a microbenchmark and the end-to-end result agree.
Output is bit-identical (100/100 token ids). Pinned by
`test_host_copy_does_not_item_meta`.

The general lesson is worth more than the 6.9%: **`.item()` after `eval` is a
GPU round trip, not a host read.** Any `.item()` on a per-layer path is a
latency bug. `slot_stats` still uses it four times, which is why instrumented
benchmark harnesses in this repo read slower than production.

#### Is Python the bottleneck? No.

A full attribution of the ~167 ms token budget puts the **CPython interpreter
proper at 3-5 ms, or 2-3%**:

| Term | ms/token | What it is |
|---|---|---|
| GPU compute | ~50 | Metal (matches the 19-21 tok/s stubbed ceiling) |
| Device I/O, saturated at depth 8 | ~46-48 | kernel + SSD |
| Host memcpy + Metal alloc in scatter | ~14 | libc / MLX allocator |
| Metal command-buffer latency and stall | ~15-25 | driver |
| Redundant `meta[0].item()` round trip | ~7-11 | Metal, **caused by a Python idiom** |
| `ThreadPoolExecutor` dispatch | ~7 | Python |
| MLX graph-node construction | ~5-8 | MLX C++, called from Python |
| **CPython bytecode** | **~3-5** | **Python** |

So rewriting the decode loop in C++/Rust is projected at **6-9%** — it deletes
essentially all Python and touches none of the compute, device I/O, or driver
latency — for months of work, and it would forfeit mlx-lm compatibility and the
bit-identical accuracy contract. **Not worth it.** The one-line fix above beat
that entire rewrite's ceiling on its own.

What actually caps the system is structural, not linguistic: the per-layer
sequence GPU ensure -> host sync -> device read -> host scatter -> GPU gather is
fully serialised 40 times per token, with the GPU idle during reads and the SSD
idle during compute. Overlapping them is worth up to ~45 ms/token, an order of
magnitude more than every Python cost combined — and is blocked by the fact
that layer *L*'s expert set is unknown until layer *L-1* has run.

### Phase 3 (MTP): the ordinary T>1 path is blocked on numerics

> **Two separate questions, answered separately.** This document reaches what
> look like opposite verdicts on MTP, in two places, and they are about
> different axes:
>
> | Axis | Question | Verdict | Where |
> |---|---|---|---|
> | **Correctness** | does MTP through ordinary T>1 MLX execution produce stock output? | **No.** The GatedDeltaNet chunked scan diverges from the recurrent step; ~4% of argmax decisions flip on the 35B. | this section |
> | **Correctness** | does a specialised MTP path produce stock output? | **Yes** — mlx-vlm's is byte-identical, using per-token GEMV verification kernels and `rollback_speculative_cache`. | this section |
> | **Economics** | is MTP *worth* its cost here? | **Was no, is now yes.** At 94.3% copy the verify pass could not pay for itself at any acceptance rate; at 43.3% it projects 1.19x. | [Native MTP](#native-mtp-viable-now-worth-12x-not-23x) |
>
> So "MTP flipped from rejected to viable" later in this document is an
> *economic* statement. It does not mean the pread fix made MTP correct — the
> numerics were never a function of the copy path. Reaching a working MTP still
> requires adopting the specialised implementation; the pread fix only made
> doing so worth the effort.

Building MTP inside slotbank means verifying `[confirmed, drafted]` in one T=2
backbone pass. On this hybrid model that pass **does not compute the same
logits as two T=1 passes**. Measured directly on `Qwen3.5-4B` through mlx-lm:

```
max |logit diff| T=1 vs T=2 : 1.562e-01
mean|logit diff|            : 2.563e-02
cache kinds                 : ArraysCache, KVCache
can_trim_prompt_cache       : False
```

The chunked scan used for T>1 in `GatedDeltaNet` differs numerically from the
recurrent step used at T=1. On the 35B this flips ~4% of argmax decisions. So a
drafter head plus perfect state rollback still yields output that differs from
non-speculative decoding — which is exactly what the `_can_speculate` guard was
added to prevent, and exactly what "do not lose reasoning on the model" rules
out. This also finally explains that guard's docstring: snapshot-and-restore was
never the bug, the T>1 numerics were.

**mlx-vlm gets this right, and the reason is expensive.** Its MTP output is
byte-identical (verified: 273 bytes exact at draft blocks 2 and 4 on the same
4B). It achieves that with *custom Metal verification kernels* --
`qwen3_5_target_verify_gemv`, `_target_verify_qmv_kernel`,
`_target_verify_qargmax_kernel` -- which verify per token with GEMV rather than
a batched GEMM, avoiding the split-K reduction differences measured above, plus
`rollback_speculative_cache` for the GDN/SSM state.

So "add MTP to slotbank" is not "add a drafter head". It is: reimplement
per-token quantized verification kernels in Metal, plus hybrid state rollback,
and then prove bit-identity. Weeks of work whose whole purpose is to re-derive a
correctness property another package already has.

**The path that preserves the accuracy contract** is therefore to run slotbank
*under* mlx-vlm's model classes rather than reimplement its kernels. That is
already compatible: slotbank patches by class *name*, and
`QuantizedSwitchLinear`, `SwitchLinear`, `SwitchGLU` and `SwitchMLP` all exist
under both packages. Two things still gate it: mlx-vlm is a vision package
(+372 MiB, opencv/mlx-audio), and **no MTP drafter exists for
`Qwen3.5-35B-A3B`** -- `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` does, but needs
its matching 19 GiB Qwen3.6 target, and this disk has 23 GiB free.

Projected payoff if all of that lands: **1.26x** at draft block 2, against the
current budget where sync is 56.3%.

### Phase 4 (removing the per-layer stall): what it would take

Compute-only ceiling on the current code, copy path stubbed, 80 warm tokens:

| arm | tok/s | ms/token |
|---|---|---|
| full | 6.68 / 6.61 | 149.7 / 151.2 |
| copy stubbed | 16.15 / 13.84 | 61.9 / 72.3 |

So ~67 ms is compute-plus-sync and ~83 ms is the host read path: **2.2x of
headroom, all of it in overlapping those two**. Three things would have to be
true, and only the third is in slotbank's control:

1. **MLX would have to overlap host and GPU work efficiently.** It does not
   today: `mx.async_eval` on balanced layer-sized work recovered 0.22 ms of a
   possible 0.91 ms, **24% efficiency**. Until that improves, every overlap
   scheme is discounted by 4x.
2. **MLX would have to expose command-buffer completion callbacks.** A round
   trip measures ~286 us, so 40 blocking waits cost ~11 ms/token in pure
   latency. MLX owns submission; slotbank cannot inject a handler without
   patching MLX or standing up its own command queue. `DeviceCopy`'s objc
   plumbing reaches residency sets, not submission. **This is an upstream
   feature request, not a slotbank change.**
3. **The dependency chain caps what could overlap anyway.** Layer *L+1*'s expert
   set is unknown until layer *L* has run, so no read can be issued a layer
   ahead without speculating -- and predictive prefetch already lost to the
   static profile. The one legitimate intra-layer overlap is `down_proj`'s rows
   loading during the `gate`/`up` GEMM, worth ~18 ms gross, ~4 ms at the
   measured 24% efficiency.

Honest conclusion: Phase 4 is worth ~11 ms (round-trip latency) plus ~4 ms
(intra-layer overlap) at today's MLX behaviour -- about **10%** -- and the
larger prize behind it is gated on upstream MLX capabilities rather than on
anything this repo can implement.

### Warming is now conditional, because it stopped being free

The hot-expert warm pass used to run at load unconditionally and was worth
~3.4x on a cold start. Reading rows [straight into the
pack](#reading-straight-into-the-pack) made cold misses much cheaper, which
shrank what warming buys. Re-measured on 35B-A3B:

| | warm off | warm on |
|---|---|---|
| model load | 3.8 s | 10.2 s |
| warm sustained decode | 4.64 tok/s | 6.03 tok/s (1.30x) |

The pass costs **6.47 s** and returns **49.5 ms/token**, so it breaks even at
**131 tokens**. At 128 tokens it was a wash; at 256 it won by only 1.04x. Below
that it is a straight loss -- every short request paying for a speedup it never
amortises.

So the pass is deferred out of `load()` and gated on whether the work will pay
it back. Two signals, because neither is honest alone: `max_tokens` is an upper
bound that HTTP callers default to 1024, and cumulative output proves a
long-lived session -- a server answering many short requests still wants warm
experts. Either crossing `SLOTBANK_WARM_MIN_TOKENS` (default 128) triggers it,
and only between requests, never mid-stream.

| request | before (warm at load) | after (deferred) |
|---|---|---|
| 32 tokens | 25.5 s wall, 1.27 cold end-to-end | **18.7 s, 1.71** (1.34x) |
| load time | 11.35 s | **3.43 s** |
| 256 tokens | 4.29-4.45 cold end-to-end | **4.55** |

Set `SLOTBANK_WARM_MIN_TOKENS=0` to restore eager warming, or `SLOTBANK_WARM=0`
to disable the pass entirely.

**Keeping experts warm is mostly about not throwing them away.** Three
sequential requests in one process, 96 tokens each:

| request | prefill | decode tok/s | wall |
|---|---|---|---|
| 1 | 6.74 s | 5.31 | 24.6 s |
| 2 | 5.98 s | 6.89 | 19.8 s |
| 3 | 6.61 s | 6.36 | 21.6 s |

Requests 2+ never pay the load again and decode ~25% faster because slots and
page cache stay populated. A long-lived server is worth more than any flag
here; relaunching the CLI per query discards all of it.

One sharp edge: the hot profile is keyed on a **SHA1 of the model path
string**. `.models/Qwen3.5-35B-A3B-4bit` and its absolute HF snapshot path hash
differently, so loading by a different spelling silently warms nothing and
reports no error.

### Behaviour under GPU contention

Decode is highly sensitive to *external* GPU load, far more than its own
utilisation suggests. Sampled from `IOAccelerator PerformanceStatistics`, an
idle desktop already costs 14.6% mean device utilisation (WindowServer plus a
browser GPU helper); slotbank decoding takes that to ~41% mean, 61% peak — so
on a quiet machine the GPU is **not** the constraint.

Put a real competitor on the GPU and that reverses (64 tokens, 35B-A3B, default
path, synthetic `float16` matmul load):

| external GPU load | device util (p50) | tok/s | vs quiet |
|---|---|---|---|
| none | 24% | 4.16 | 1.00x |
| moderate (1024²) | 96% | 2.20 | **0.53x** |
| heavy (3072²) | 99% | 0.62 | **0.15x** |

The mechanism is the per-layer synchronisation. Decode blocks on the GPU 40
times per token, and each of those waits queues behind whatever else holds the
device — so contention multiplies through every layer rather than costing a
fixed slice. Warm phase profile of the default path, past the ramp:

| phase | ms/token | share |
|---|---|---|
| **GPU sync (40x)** | **82.7** | **49.5%** |
| threaded `pread` | 55.9 | 33.4% |
| scatter into pack | 16.4 | 9.8% |
| gather + attention + sampler | 12.1 | 7.2% |

Roughly 50 ms of that sync is genuine compute (the ~19–21 tok/s compute ceiling
implies ~50 ms/token); the rest is wait. The practical advice is unglamorous:
this workload wants the GPU to itself. Quitting a browser is worth more than
any tuning flag in this repo.

### Accuracy

Default output is **bit-identical to stock mlx-lm** — maxabs 0.0 on real-model logits, including through mlx-lm's sorted-routing path. Expert row reads were verified bit-identical for both `U32` and `BF16` tensors.

**Scope of that claim.** It was measured on the models marked *measured* in
[Model support](#model-support) — `Qwen3.5-35B-A3B`, `Qwen3-30B-A3B`, `OLMoE`.
The *argument* for it generalises: slotbank changes which rows sit in the GEMM's
operand and in what order they were fetched, not the arithmetic, so a family
whose `SwitchLinear` the patch attaches to should behave identically. But an
argument is not a measurement, and "bit-identical" is not asserted for any family
listed as *supported by class*. The one deliberate exception is
`SLOTBANK_WAVES=1`, which is off by default precisely because it breaks this.

`SLOTBANK_WAVES=1` is the one exception and is off by default; see [Environment](#environment).

## Install

Apple Silicon macOS, Python 3.10+.

**As a tool** — puts `slotbank` on your `PATH` in an isolated environment, so it
is available in any terminal and its dependencies cannot collide with a project:

```bash
uv tool install slotbank          # or: pipx install slotbank
slotbank                          # shows what to do next
```

Add `slotbank[vlm]` for `qwen4_exp` (Qwen3.8-Flash-Next), `kimi_k3`,
`deepseek_v4`, and `minimax_m3` — model classes `mlx-lm` does not carry.
`qwen4_exp` is experimental and needs mlx-vlm git main (2026-08-27+):
`pip install git+https://github.com/Blaizzy/mlx-vlm.git`. The extra pulls
~370 MB, so it is not the default.

**From source**, for development:

```bash
git clone <repo> && cd slotbank
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
uv build --wheel               # produces dist/slotbank-*.whl
```

## Quick start

Models are managed for you. Nothing here needs a filesystem path.

```bash
slotbank check Qwen3.5-35B-A3B-4bit    # will it run here? (no download)
slotbank pull  Qwen3.5-35B-A3B-4bit    # check first, then download
slotbank run   Qwen3.5-35B-A3B-4bit    # chat, model stays loaded
slotbank list                          # what is downloaded, and its expert share
slotbank rm    Qwen3.5-4B-4bit         # free the disk
```

`check` reads the safetensors headers over HTTP range requests — a few MB
against a checkpoint of any size, ~6 s for an 86 GiB model across 42 shards —
and reports the two figures that decide viability:

```
mlx-community/Qwen3.5-35B-A3B-4bit  (qwen3_5_moe, 4 shards)
  layers 40  experts 256  top-k 8
  total on disk       18.99 GiB
    routed experts    16.88 GiB  (88.9%, streamed)
    resident floor     2.12 GiB  (must fit in RAM)
  per expert           1.69 MiB
  touched / token      0.53 GiB  (sets throughput)

  this machine: 24 GiB RAM, 25 GiB disk free
  slotbank would pick C=32: 4.23 GiB wired (4x smaller than the checkpoint)

  runs here. 19 GiB of weights, 4.2 GiB wired.
```

The resident floor is what must stay in RAM no matter how large the checkpoint
is; total parameter count is not the metric that decides whether a model loads.
`pull` runs this first and refuses a model that cannot run, because discovering
that after downloading 86 GiB is the waste `check` exists to prevent
(`--force` overrides).

**Model storage is the Hugging Face cache**, not a private blob store. A model
pulled by mlx-lm or any other MLX tool is already visible to `slotbank list`, and
nothing is duplicated — which matters when the checkpoint is larger than your
free disk.

Short names resolve against the cache first, then against known namespaces, so
`Qwen3.5-35B-A3B-4bit` and `mlx-community/Qwen3.5-35B-A3B-4bit` and a local
directory all work. GGUF repos are never matched: that is llama.cpp's format.

Serve the three agent doors on localhost:

```bash
slotbank serve --model Qwen3.5-35B-A3B-4bit --port 8080
```
```
slotbank serving mlx-community/Qwen3.5-35B-A3B-4bit on http://127.0.0.1:8080
  OpenAI / Codex / OpenCode  OPENAI_BASE_URL=http://127.0.0.1:8080/v1
  Claude Code                ANTHROPIC_BASE_URL=http://127.0.0.1:8080
```

Optional `--api-key` is checked on `Authorization: Bearer` and `x-api-key`.

The lower-level `slotbank generate --model <path>` and `slotbank admit --model
<path>` still take explicit paths, for scripting against a checkpoint outside the
cache.

## Agent APIs

The server is local HTTP. Point the client at this process. Do not send your weights to anyone else.

### What `serve` is for

Not concurrency — the engine is single-model and processes one request at a
time. `serve` exists to **stop paying the fixed cost**, which on a file-backed
runtime is most of a short request.

Measured on the 35B, same prompt, same 48 tokens, `temperature 0`:

| | Per request | tok/s |
|---|---|---|
| fresh `slotbank run` process, twice | 18.2 s, 18.7 s | 2.57–2.64 cold end-to-end |
| warm server, request 1 | 15.9 s | 3.01 |
| warm server, requests 2–4 | 12.2–13.4 s | **3.58–3.94** |

Two separate effects, and it is worth separating them:

1. **Load elision, ~6 s.** The process already holds the model. This is constant
   per request and is the larger effect.
2. **Warm slots, ~20%.** Request 1 through the server is still slower than
   requests 2–4, because the C-slot pack and the page cache behind it start cold
   and fill with the experts this workload actually routes to. This effect is
   specific to a runtime whose cost is dominated by expert reads — it is much
   weaker for a runtime holding all weights resident, which has nothing to warm.

Net **~1.45x on a 48-token request**. That ratio is not fixed: the saving is
roughly constant in seconds, so it dominates short requests and vanishes into the
noise on long ones. An agent doing many small tool-call round trips is the case
where it matters most, which is the case `serve` is aimed at.

What `serve` does **not** currently do: run more than one model, unload on idle,
or survive as a daemon. It holds one model for the lifetime of the process.

### Viability of a multi-model daemon

Measured, not projected — the question is whether slotbank's design makes an
Ollama-style daemon cheap or expensive, and the answer is unusually favourable
for a reason specific to this runtime.

**Do two engines coexist?** Yes. MLX arrays are thread-bound and each `Engine`
owns its loader thread, so this was the gate. Two engines loaded, generated,
alternated, and unloaded in one process with no Metal or threading conflict.

**What does residency cost?** Almost nothing, because slotbank does not hold
weights resident. A second model adds its resident floor plus its pack, not its
checkpoint:

| Configuration | Process RSS | 35B warm sustained decode |
|---|---|---|
| 35B alone | 3.66 GiB | 6.00 tok/s |
| 35B + Qwen3.5-4B both loaded | **4.78 GiB** | 5.26 tok/s (**0.88x**) |
| after unloading the 4B | 3.17 GiB | 6.76 tok/s |

A 19 GiB model and a 2.8 GiB model, both live and both answering, in 4.78 GiB.
The 4B ran at 14.31 tok/s concurrently. The cost to the large model is **~12%**,
and it is bandwidth contention rather than capacity — the same effect as any
other neighbour.

This is the inverse of the usual situation. For a runtime that holds weights
resident, a second model costs a second full residency and is usually impossible;
here the marginal cost of an idle-but-loaded model is a pack, which is why
`C` being small is what makes multi-model cheap.

**Does unload reclaim?** It does now. It did not before this measurement:
`Runtime.close` called `mx.clear_cache()` *before* dropping the model reference,
so the clear freed nothing and the buffers then landed in MLX's allocator cache
as they released — **1.09 GiB retained per unloaded model**, which would make an
idle-unload policy do nothing at all. Reordering the drops before the clear takes
reclaim to immediate and complete, and the remaining model recovers to 0.97–1.13x
of its solo rate.

**What is actually left to build**, given the above: a process that owns several
engines, an LRU over them keyed on last use, and routing by the `model` field the
API layer already parses. Load cost is 3.4 s for the 35B and 1.8 s for the 4B, so
an eviction is cheap to undo. None of this is a runtime change — the hard parts
(coexistence, reclaim, per-model wired cost) are measured and settled.

| Client | Protocol | Base URL |
|---|---|---|
| Cursor Chat / OpenAI SDK | `POST /v1/chat/completions` | `http://127.0.0.1:8080/v1` |
| Claude Code | `POST /v1/messages` | `http://127.0.0.1:8080` |
| Codex | `POST /v1/responses` | `http://127.0.0.1:8080/v1` |

Also: `GET /v1/models`, `GET /models`, `GET /props` (llama.cpp-shaped, so Oh My Pi 18's `/model` local pane is not empty), `POST /v1/completions`, `POST /v1/messages/count_tokens`, `GET /health`.

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

## Model support

### What gets slotted, and what does not

Only **routed expert weights** are streamed. Everything else — attention,
embeddings, norms, router gates, and any always-active shared expert — stays
resident. That split decides whether a model is viable here, and it is a
property of the architecture, not of its parameter count.

Measured on the reference checkpoint: 88.9% experts, **11.1% resident**.

Two rules follow:

1. **The resident floor must fit in RAM.** It scales with `layers x hidden^2`
   plus embeddings, so a model with wide hidden dimensions is expensive even
   when its expert bank is streamable. `DeepSeek-V3` needs ~39.6 GiB resident
   and cannot run on 24 GB; `DeepSeek-V4-Flash`, four times larger in total,
   needs **6.25 GiB** and fits.
2. **Throughput tracks activated expert bytes**, `top_k x layers x 3 x hidden x
   moe_intermediate`, not total size. The shape that runs well is **high total,
   small experts, few active** — the A3B family exactly.

### Architectures

Expert slotting attaches to any module named `QuantizedSwitchLinear` or
`SwitchLinear` inside a `SwitchGLU` / `SwitchMLP` container. Matching on class
*name* rather than identity means it follows a model class wherever it lives.
In `mlx-lm` 0.31.3 that is **34 architectures**:

`afmoe`, `bailing_moe`, `bailing_moe_linear`, `deepseek`, `deepseek_v2`,
`deepseek_v3`, `deepseek_v32`, `dots1`, `ernie4_5_moe`, `exaone_moe`,
`gemma4_text`, `glm4_moe`, `glm4_moe_lite`, `gpt_oss`, `granitemoe`,
`granitemoehybrid`, `hunyuan`, `jamba`, `kimi_linear`, `Klear`, `lfm2_moe`,
`llama4`, `longcat_flash`, `mimo_v2_flash`, `minimax`, `mixtral`, `nemotron_h`,
`olmoe`, `phimoe`, `phixtral`, `qwen2_moe`, `qwen3_moe`, `qwen3_next`,
`step3p5`

Several more inherit a covered base class: **`kimi_k25`** (via `deepseek_v3`),
**`glm_moe_dsa`** (via `deepseek_v32`), **`qwen3_5_moe`** (via `qwen3_5`).

Three status levels, and the difference between them is the whole point of this
table. **Only "measured" means a model has produced tokens here.**

| Level | What was verified | What was not |
|---|---|---|
| **measured** | loaded, generated, output compared against stock | — |
| **supported by class** | the patch attaches to this family's `SwitchLinear`; the arithmetic is unchanged | that the model loads, runs, or fits |
| **not loadable** | `mlx-lm` has no class for it | — |

Attaching to a `SwitchLinear` successfully proves one layer type is handled. It
does not prove an architecture works: tensor-name prefixes, tokenizer quirks, and
non-MoE layers all break independently of anything slotbank touches.

| Family | Status |
|---|---|
| Qwen 2/3/3.5/3.6 MoE, Qwen3-Next | supported; 3.5-35B-A3B and 3-30B-A3B **measured** |
| DeepSeek V2 / V3 / V3.2 | supported by class; resident floor decides viability |
| GLM4-MoE, GLM4-MoE-Lite, GLM-MoE-DSA | supported by class, untested |
| Kimi Linear, Kimi K2.5 | supported by class, untested |
| Mixtral, OLMoE, PhiMoE, GPT-OSS, Llama4, MiniMax, Jamba | supported by class; OLMoE **measured** |
| **DeepSeek-V4-Flash** | **mlx-vlm**: `mlx-lm` has no `deepseek_v4` |
| **Qwen3.8-Flash-Next (`qwen4_exp`)** | **mlx-vlm git main**: experimental class; SwitchLinear attaches; n-gram table is not slotted |
| Dense models that **fit** the working set | **served resident** (stock mlx-vlm / mlx-lm); Qwen3.8-27B 3.5bpw **measured** on 24 GB |
| Dense models that do not fit | **no benefit** from expert slots; do not layer-stream unless admit fails |

"Supported by class" means the patching mechanism attaches and the arithmetic is
unchanged. It is not a claim that the model has been run. Two things break
independently of the MoE layer: a checkpoint whose tensors carry an unexpected
prefix (`Qwen3.5` uses `language_model.`, which some tools reject), and any
architecture `mlx-lm` cannot load at all.

### Checking a model before committing to a download

`slotbank admit` reads `config.json` and refuses rather than loading blind:

```
ok=True kind=moe stored=18.99GiB active=2.64GiB leave_free=8GiB working_set=16GiB
moe: E=256 top_k=8 -> C=32 (12.5% of bank)
speculative decoding: UNSAFE -- mixed layer_types: full_attention, linear_attention
```

A remote checkpoint's exact layout can be read without downloading it, by range-
requesting the safetensors headers — about 2 MB against an 86 GiB model. That is
enough to compute the resident floor and the per-expert row size, which together
determine both viability and slot cost.

## Clients

Three API surfaces on one port, all verified with streaming:

```bash
slotbank serve --model <path> --port 8080 --effort high
```

| Surface | Endpoints |
|---|---|
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/models` |
| Anthropic | `/v1/messages`, `/v1/messages/count_tokens` |
| OpenAI Responses | `/v1/responses`, `/v1/responses/{id}` |

Anything speaking those protocols connects directly — Claude Code, Codex,
Cursor, OpenCode, or a plain `curl`. Point the client's base URL at
`http://127.0.0.1:8080` and give it any non-empty API key unless `--api-key` is
set.

Requests **queue**: one generation runs at a time because the Metal thread is
exclusive. Two *processes* on one model measured worse than one (3.61 vs 7.96
combined tok/s) because each holds its own pack and they evict each other's page
cache. Batched decode within one process is a measured 2.3x per agent and is
**not yet implemented** — see `docs/DECISIONS.md`.

## Tuning

Three presets cover most uses. Each value is the one that measured best for that
intent; nothing here is a guess about what "more" should mean.

```bash
slotbank generate --model <path> --effort low    --prompt "..."
slotbank serve    --model <path> --effort high
```

| `--effort` | Footprint | Warm pass | Prefill chunk | Use when |
|---|---|---|---|---|
| `low` | capped at 2 GiB | off | 512 | something else needs the machine |
| `medium` *(default)* | policy picks | after 128 tokens | 2048 | general use |
| `high` | policy picks | eager | 4096 | dedicated machine, long generations |

Measured on `Qwen3.5-35B-A3B`, cold process:

| Preset | 60 tokens | 300 tokens |
|---|---|---|
| `low` | 4.71 tok/s, 18.5 s | — |
| `medium` | **5.75 tok/s, 15.9 s** | 4.93 tok/s, 69.5 s |
| `high` | 5.21 tok/s, 19.5 s | **5.25 tok/s, 68.7 s** |

`high` is slower on short answers and faster on long ones, which is the warm pass
paying for itself: it costs ~6.5 s and returns ~49.5 ms/token, so it breaks even
at **131 tokens**. `medium` defers it until the work justifies it.

**`high` does not raise the slot count, and that is deliberate.** Past the
capacity the policy picks, more slots measured *slower* here — C=32 gives
8.6 tok/s against C=64 at 6.5 — because the extra pack is taken from the page
cache that serves its own misses. A preset that raised C would look like a
performance setting while being a regression. A test pins this.

### Individual flags

Every knob is also a flag, and a flag beats a preset, which beats the
environment. Presets are a starting point, not a lock.

| Flag | Environment | Default |
|---|---|---|
| `--budget-gib` | `SLOTBANK_BUDGET_GIB` | policy |
| `--slots` | `SLOTBANK_SLOTS_OVERRIDE` | policy |
| `--read-threads` | `SLOTBANK_READ_THREADS` | 8 |
| `--prefill-step` | `SLOTBANK_PREFILL_STEP` | 2048 |
| `--warm-min-tokens` | `SLOTBANK_WARM_MIN_TOKENS` | 128 |
| `--no-warm` | `SLOTBANK_WARM=0` | on |
| `--quiet` | — | progress on stderr |

Progress and the stats line go to **stderr**, so `stdout` stays a clean
completion that pipes. The line reports what actually happened:

```
200 tokens - 7.58 tok/s decode - first token 7.1s - 33.4s total
```

### What the footprint dial is, and is not

It is a dial **down**. On a machine where the expert bank exceeds the page
cache, spending more memory on slots is a measured loss, so `--budget-gib`
exists to give memory *back* when something else needs it — not to buy speed.

| `--budget-gib` | Wired | Decode |
|---|---|---|
| 2 | 2.40 GiB | ~6.06 tok/s |
| default | 3.46 GiB | ~6.55 tok/s |

The rule underneath: the pack should grow only while the rest of the bank still
fits in whatever page cache remains. Where the whole bank fits in cache, the
trade reverses and more slots win — so this is machine-relative, not a law.

### Multitasking

The wired footprint is ~3.5 GiB for a 19 GiB model and is nearly independent of
context: 16x the context costs 80 MB. Everything above that is OS page cache,
which the kernel reclaims on demand — so under memory pressure this degrades in
speed rather than failing, and it does not swap the machine out from under your
other work.

It is not free concurrency. Measured against an adversarial neighbour (a
4096&times; float32 matmul loop, itself memory-bandwidth-bound):

| | Alone | Concurrent |
|---|---|---|
| neighbour | 7.3 s | 12.9–17.5 s (**~2x**) |
| slotbank | ~6.7–7.5 tok/s | 4.64 tok/s (**~1.5x**) |
| RAM free | 75% | 61% |

Both slow down, because on unified memory the scarce resource is **bandwidth,
not capacity**, and no caching strategy fixes that. A GPU-heavy neighbour is far
worse: decode drops 6.7x under GPU contention, since it blocks on the GPU 40
times per token. A neighbour that is I/O-light and GPU-free — editing, browsing,
most development — contends much less than the figure above, which is close to a
worst case.

## Memory

The heap is one LPDDR pool. There is no host↔GPU expert copy in the CUDA sense.

### Why unified memory is not just "a big GPU"

The appeal of unified memory is that a 24 GB Mac appears to have 24 GB of VRAM.
The reality is that three different subsystems have claims on that pool, and only
one of them can refuse.

Queried on this machine (`mx.device_info()`):

```
memory_size                       24.00 GiB   installed
max_recommended_working_set_size  17.76 GiB   what Metal will let you hold (74%)
max_buffer_length                 13.32 GiB   the largest SINGLE allocation
```

Those are separate walls, and it is worth being precise about which one binds,
because it is easy to overstate the second. **In practice it does not bind at
this scale.** Checkpoints already arrive split per layer and per projection: the
largest single tensor in the 35B is a stacked `(256, 512, 256)` U32 expert weight
at **128 MiB**, about 106x below the cap. Neither slotbank's per-layer packs nor
stock mlx-lm's stacked banks come close.

Where it does bind is the one design nobody should reach for anyway: the entire
16.88 GiB expert bank as a single `mx.array`. That exceeds 13.32 GiB and is
therefore not merely slow, it is unallocatable. Worth knowing as a hard floor
under the design space — not as the reason anything here fails.

**The wall that actually binds is the working-set limit, and it binds in
aggregate.** No individual allocation is too large; the sum of what is
simultaneously wired is. That is why the interesting question is never "how big
is this tensor" but "how many of them are resident at once" — which is exactly
what `C` controls.

The working-set limit is not the OS being conservative. Anything a GPU is
actively reading must be **wired** — physically resident, exempt from paging,
because a GPU cannot take a page fault the way a CPU thread can. Wired pages are
the one tier macOS cannot reclaim under pressure. The 74% figure is the kernel
reserving enough unwireable headroom to keep WindowServer, the compositor, and
your other processes alive.

### The three tiers, and what each one costs

| Tier | Reclaimable? | Cost to read | What slotbank puts here |
|---|---|---|---|
| **wired** (Metal working set) | **no** | — (already there) | the C-slot pack, ~3.5 GiB |
| **OS page cache** (file-backed, clean) | yes, instantly | ~10 GB/s | as much of the bank as fits |
| **SSD** | n/a | 1000–3300 MiB/s via `pread` | the rest |
| **compressor / swap** | — | catastrophic | *nothing, by design* |

The fourth row is the point. A clean, file-backed page is **free to evict**: the
kernel drops it and re-reads from disk if needed. A dirty anonymous page — which
is what a full weight residency looks like — cannot be dropped. It must be
compressed or swapped, and both cost far more than the re-read would have.

This is the entire difference between the runtimes benchmarked above. Stock
mlx-lm holds 18.2 GiB of *anonymous* pages and reaches 0.005 tok/s with 13.2 GB
in the compressor. slotbank holds 3.5 GiB wired and lets the other 15 GiB be
*file-backed*, where eviction is a no-op. Same model, same machine, same total
bytes touched — different tier, four orders of magnitude in throughput.

### Why `free` is always near zero, and why that is fine

macOS does not leave RAM idle. On this machine right now, `vm_stat` reports
~140,000 file-backed pages and only ~4,200 speculative — free memory is a rounding
error, and that is the design working correctly. Every byte not otherwise claimed
is holding cache that costs nothing to give back.

So `free` is the wrong scarcity signal, and reading it as one is how you conclude
a healthy machine is about to die. slotbank uses three signals instead
(`src/slotbank/um.py`):

| Signal | Source | Means |
|---|---|---|
| **pressure level** | `kern.memorystatus_vm_pressure_level` | 1 normal, 2 warn, 4 critical — the kernel's own verdict |
| **reclaimable** | free + speculative + inactive + purgeable | what could be handed back without swapping |
| **compressor growth** | `Pages occupied by compressor` over time | the machine is already paying, not about to |

Only the third distinguishes "the cache is full because caching works" from "the
cache is full because we are in trouble." At the time of writing this machine has
2,243,829 pages held in the compressor occupying 541,761 — a **4.1:1** ratio,
meaning ~34 GiB of logical pages squeezed into ~8 GiB. That is the compressor
succeeding; it is also the last thing standing between the machine and swap.

### How it degrades under strain

Degradation is not graceful in one smooth curve. There are three distinct
regimes, and slotbank is engineered to stay in the first:

**1. Cache eviction (normal).** Wired footprint is fixed; the page cache shrinks
under pressure. A miss that would have been served at ~10 GB/s is served from SSD
at 1000–3300 MiB/s instead. **Throughput falls; nothing fails.** Measured under an
adversarial bandwidth-bound neighbour: 6.7–7.5 → 4.64 warm sustained decode
(~1.5x), while the neighbour itself slowed ~2x. Both lose, because on unified
memory the contended resource is **bandwidth, not capacity** — and no caching
strategy fixes bandwidth.

**2. GPU contention (bad, recoverable).** A GPU-heavy neighbour is far worse than
a memory-heavy one: decode blocks on the GPU ~40 times per token, so queueing
behind another process's command buffers costs **6.7x**. This is a scheduling
problem, not a memory problem, and no amount of RAM helps.

**3. Wired overcommit (fatal).** If wired demand approaches the working-set limit,
there is no reclaim path — the kernel cannot evict wired pages, so it compresses
and then swaps *everything else*, including the UI. This is what a full-residency
run does, and it was observed twice during this work: once at 571 MB free during a
64k prefill, and once as a hard freeze from a competing runtime configured for
full residency. The machine did not recover on its own.

Slotbank's defence against regime 3 is structural rather than reactive: `C` is
frozen at load, so the wired footprint cannot grow mid-generate no matter what
the prompt does. Pressure can shrink the KV cache and the warm budget, but it
**never resizes `C`**. A dial that could grow under load is a dial that can reach
the cliff, and the cliff has no recovery path.

### What this buys in practice

The wired footprint is ~3.5 GiB for a 19 GiB model and is nearly independent of
context — 16x the context costs 80 MB. That is why the machine stays usable: your
browser and editor are competing with 3.5 GiB of unreclaimable memory, not 18.2.
An I/O-light, GPU-free neighbour (editing, browsing, most development) contends
far less than the adversarial figures above, which are close to worst case.

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

Pressure (`vm_stat` + `kern.memorystatus_vm_pressure_level`) can drop KV and scales back the warm budget. It never resizes `C` mid-generate.

**There is no in-process L2 cache.** An earlier version kept evicted experts in MLX memory; it was removed because it never ran (the copy path prefers the file store, so it was bypassed on every file-backed model) and because it is the wrong tier: it holds bytes in *wired* memory competing with the pack, while the OS page cache does the same job in *evictable* memory. Warm start exploits that tier deliberately.

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
| `SLOTBANK_WAVES` | off | `1` for prefill in waves of `C` experts |
| `SLOTBANK_WARM` | on | `0` to skip the hot-expert warm pass entirely |
| `SLOTBANK_WARM_GIB` | adaptive | warm-budget override in GiB; default scales with memory pressure |
| `SLOTBANK_WARM_MIN_TOKENS` | `128` | tokens the warm pass must pay back before it runs; `0` warms eagerly at load |
| `SLOTBANK_PREFILL_STEP` | `2048` | tokens per prefill chunk; lower caps the memory peak |
| `SLOTBANK_BUDGET_GIB` | unset | cap on resident expert bytes; capacity is solved from it |
| `SLOTBANK_SLOTS_OVERRIDE` | unset | force a slot count, bypassing the capacity policy |
| `SLOTBANK_WIRED_MIB` | auto | wired-limit override in MiB; `0` keeps the pack-only limit |
| `SLOTBANK_PREFIX_CACHE` | off | `1` to reuse KV across requests sharing a prefix (agentic workloads) |
| `SLOTBANK_LOOKAHEAD` | `0` | `k` candidate tokens per verify pass; needs `(k+1)*top_k <= C` and a trimmable cache |
| `SLOTBANK_READ_THREADS` | `8` | miss reads per layer via threaded `pread`; `0` falls back to the mmap + `madvise` path (1.58x slower) |
| `SLOTBANK_KV_BITS` | off | `8` to quantise the KV cache past `SLOTBANK_KV_START`; `4` accepted but not recommended |
| `SLOTBANK_KV_START` | `4096` | context length past which the KV cache is quantised |

### KV cache quantisation

The KV cache, not the model, sets the ceiling on context. On Qwen3-30B-A3B it is
96 KiB per token — 48 layers × 4 kv heads × 128 head dim × 2 for K and V × 2
bytes — so 150k tokens of chat costs 13.7 GiB, against a 16 GiB working set that
already holds 3.2 GiB of shared weights. Admission checks once at load and never
rechecks, so the ceiling is currently discovered by thrashing.

`SLOTBANK_KV_BITS=8` shrinks the cache 1.88× (measured 120.0 → 63.8 MiB over
1055 tokens), moving the ceiling from roughly 122k to 230k tokens on 24 GB.
Measured on 1024 tokens of real text, six layers sampled:

| bits | KiB/token | K rel err | V rel err | top-1 attention preserved |
| --- | --- | --- | --- | --- |
| fp16 | 96.0 | — | — | 100.0% |
| 8-bit | 51.0 | 0.0138 | 0.0084 | 96.9% |
| 4-bit | 27.0 | 0.1462 | 0.1040 | 78.1% |

`top-1` is the share of positions whose highest-attention token is unchanged
after quantisation, probed with keys as queries.

**This buys memory, not speed.** mlx dispatches a quantised cache to
`quantized_scaled_dot_product_attention`, which is *not* fused: two
`quantized_matmul` calls with the full N-wide score matrix materialised between
them, where the fp16 path uses `mx.fast.scaled_dot_product_attention` and never
materialises it. Fewer bytes therefore does not mean less time. Attention cost
per generated token, ×48 layers, medians of two runs of the kernel alone:

| ctx | fp16 | 8-bit | 4-bit |
| --- | --- | --- | --- |
| 4,096 | 46 ms | 52 ms | 40 ms |
| 16,384 | 44 ms | 52 ms | 45 ms |
| 65,536 | 165 ms | 341 ms | 126 ms |
| 150,016 | 370 ms | 873 ms | 393 ms |

At 150k, 8-bit costs +504 ms per token — 6.1× more than the entire 82 ms/token
of expert reads that the freed 6.43 GiB could buy back in the best case. Use
8-bit to make a context *fit*, not to make it fast. 4-bit is roughly speed-neutral
and frees more, but see the quality table above before reaching for it.

That table also shows where the real long-context bottleneck is: at 65k and
beyond, attention alone exceeds the 154 ms/token measured for a whole decode
step at ~1k context. Expert I/O is a short-context problem. Attention is
O(N) per token, so the only lever that makes long windows *faster* is dropping
tokens (eviction), not shrinking them (quantisation).

4-bit is accepted but not recommended for quality, and the reason is structural.
`mx.quantize` groups scales along `head_dim`, so each token's key is scaled
across its own channels. K's outliers sit in a *fixed* channel — measured, the
same channel is the largest for 93.6% of tokens, against 29.2% for V — so a
per-token group pays for an outlier it could have isolated. Quantising K
per-channel instead measures 3.5× lower error at an identical scale budget, and
the ratio holds flat from 128 to 2048 tokens. That is not an mlx defect:
per-channel groups span 64 *tokens*, and `mx.quantize` refuses a token count that
is not a multiple of the group size, so a decoder appending one token per step
cannot form them. Closing the gap needs a KIVI-style fp16 residual buffer for
the partial group.

Quantisation is one-way and whole-cache, which is why it waits for
`SLOTBANK_KV_START`: below a few thousand tokens the cache is a small share of
the bytes read per token, and all of the quality cost lands on long-context
retrieval. `SLOTBANK_PREFIX_CACHE` is disabled while quantising — see the note in
`runtime.py`. Within-session KV reuse is unaffected.

### Why C is a dial down

Cache hit rate is the wrong objective. A bigger pack removes reads, but it
evicts the OS page cache that was serving them, so each surviving miss gets
more expensive. Measured on Qwen3-30B-A3B (E=128, top-8), 512-token context,
one interleaved round:

| C | hit rate | read/token | I/O wall | effective BW | tok/s |
| --- | --- | --- | --- | --- | --- |
| 8 | 41.1% | 0.559 GiB | 2.93 s | 12.21 GiB/s | 10.26 |
| 16 | 62.3% | 0.358 GiB | 3.19 s | 7.16 GiB/s | 8.70 |
| 32 | 87.0% | 0.123 GiB | 1.85 s | 4.27 GiB/s | 7.93 |

Going from C=8 to C=16 reads **36% fewer bytes and spends 9% more time doing
it**. 12.21 GiB/s is far above this SSD's ~2.9 GiB/s, so at C=8 the reads were
being served from page cache; by C=32 effective bandwidth is converging on real
disk. Both terms move against you — the gather also runs over a larger array,
so GPU time rose 2.97 → 3.60 → 5.29 s across the same sweep.

This is corroborated independently by [MawForge](https://arxiv.org/html/2607.09686),
validated on a 24 GB unified-memory Mac with an 18 GiB serving budget: Gemma Q8
at 4K went from 13.86 tok/s at a 35% cache to **1.30 tok/s at 65%**, while hit
rate *improved* from 86.73% to 95.03%. Their collapse fires near 84% system-used
— an absolute-footprint threshold, not a cache fraction.

Caveat on the table above: three points, one prompt length, one round. It is
enough to establish the mechanism and the direction, not enough to retune the
default — `C = max(floor, E // 8)` is a ratio of `E`, while the evidence says the
binding variable is absolute footprint against free memory. Those coincide only
by accident.

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

### What "3.5 GiB" actually means

`mx.get_active_memory()` counts MLX allocations only. Four accountings of the same run
(`Qwen3.5-35B-A3B`, C=32, after 50 tokens):

| measure | value | what it is |
|---|---|---|
| `mlx_active` | 3.46 GiB | MLX allocations — the figure usually quoted |
| **`phys_footprint`** | **3.77 GiB** | **what Activity Monitor shows** — adds Python, tokenizer, runtime |
| RSS (`ps`) | 5.99 GiB | includes resident mmap'd file pages; overcounts, they are clean and reclaimable |
| system wired delta | +3.9 GiB | the part that genuinely constrains other applications |

**Quote ~3.8 GiB, not 3.46.** The gap up to 5.99 is evictable page cache holding expert
rows — real memory, but instantly reclaimable and not denied to anything else. That
distinction is the whole design, and it is also why `ps` makes this look worse than it is.

For comparison, stock mlx-lm on the same model reports 18.2 GiB of `mlx_active`, all of
it wired.

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

### On-demand expert quantization

Tempting idea: demote experts to 3-bit while they are in flight to cut per-call
cost, and promote them back later. Reversibility is not the problem — the 4-bit
checkpoint on disk is never mutated, so any demotion is a derived copy and
"returning to steady state" is just a re-read. The problem is that it cannot
pay for itself.

Measured on real expert tensors (`gate_proj`, 256 experts, group size 64):

| target | rel L1 error vs stored 4-bit | KiB/expert | vs 4-bit |
|---|---|---|---|
| 3-bit | 20.4% | 448 | 78% |
| 2-bit | 42.2% | 320 | 56% |

`dequantize` + `quantize` costs **493 us per expert per bank**. A token misses
~120 experts across 9 banks, so demoting on the fly adds **~532 ms/token of GPU
work** against a total warm budget of ~167 ms — 3x slower than everything decode
does today, and loaded onto the exact resource that collapses under contention.

The deeper objection is structural: **on-demand quantization saves no I/O at
all.** The bytes on disk are 4-bit, so a miss must read the full 4-bit row
before there is anything to requantize. Reads are the bottleneck; requantizing
after the read shrinks only the pack, and the pack is not what is slow.

Storing a lower precision is a different proposal, and there the arithmetic is
merely unattractive rather than fatal. Reads are 33% of the warm budget, and
misses come mostly from the cold tail, so demoting the tail to 3-bit offline
would buy roughly **7%** throughput in exchange for 20% relative error on the
24% of traffic the tail carries. 2-bit is worse on both axes.

The conclusion generalises: quantization is a lever for **fitting**, not for
speed — and fitting is the problem slotbank already solves. Spend the bits on
quality and let the slot cache handle the size.

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

### Prompt-lookup speculative decoding (opt-in)

`SLOTBANK_LOOKAHEAD=k` proposes `k` candidate tokens by finding the current suffix earlier
in the context and reusing what followed, then verifies them in **one `T=k+1` pass**. No
draft model, no MTP head, no extra memory: the candidate source is the context itself.

Greedy-equivalent **in practice, not by construction**. A proposal is kept only where it
matches the token the verify pass predicts — but a `T=k` verify pass is not numerically
identical to `k` sequential `T=1` steps, so "the token the model would have produced" is
itself path-dependent. Measured logit difference between one `T=2` pass and two `T=1`
passes:

| model | logit diff | argmax flips |
|---|---|---|
| OLMoE-1B-7B | 3.1e-02 | **0 / 24 positions** |
| Qwen3.5-35B-A3B | **6.9e-01** | **1 / 24 (4%)** |

The hybrid model is ~20x worse because linear attention uses a **chunked scan for T>1 and a
recurrent step for T=1** — genuinely different code paths, not float noise. On OLMoE the
difference is small enough that the argmax never moved in testing and output stayed
identical; on the 35B it moves about 4% of the time.

This is why the trimmable-cache guard is doing double duty: the models it excludes are also
the ones where batch verification is least faithful.

`OLMoE`, k=4, C=48, interleaved against k=0:

| workload | k=0 | k=4 | speedup | acceptance | identical |
|---|---|---|---|---|---|
| repetitive (quote back) | 27.23 | **41.99** | **1.54x** | 75% | yes |
| code-like (edit a function) | 18.11 | 18.75 | 1.03x | 64% | yes |
| open prose | 15.75 | **26.38** | **1.67x** | 31% | yes |

**Two hard constraints:**

- **`(k+1) x top_k <= C`.** The multi-token slot path is only safe while every routed id
  can be resident at once. At OLMoE's default C=16 with top_k 8 that caps `k` at 1; asking
  for k=4 pushes every verify into the prefill path and measured **0.5-0.67x — slower than
  not speculating.** Raise `C` before raising `k`.
- **The cache must be trimmable.** `Qwen3.5-35B-A3B` has 30 recurrent layers, so
  `_can_speculate()` returns `False` and it falls back to normal decoding rather than
  generating conditioned on rejected tokens. This is the guard that mlx-lm's own
  `speculative_generate_step` is missing (see `docs/mlx-lm-speculative-bug.md`).

So this helps pure-attention MoE models and is correctly inert on hybrid ones.

**Snapshot-rewind: cheap, correct, and still not enough.** A recurrent cache cannot be
trimmed, but it can be copied before the verify pass and restored after. The cost is not the
obstacle — **~3 ms to snapshot 61.9 MiB, ~0 ms to restore, against a ~300 ms token, about
1%** — and the mechanism is provably faithful:

- restore vs a cache that never ran the verify pass: **0 mismatches across all 40 layers**, offsets included
- restore-then-re-run vs a clean run: **identical logits (maxabs 0.0), 0 differing cache arrays**

The divergence was never the rewind. It is the `T>1` verify pass itself (above). Snapshot
rewind is therefore correct and reusable, but it does not make speculation lossless on a
hybrid model, so the guard stays.

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

### Native MTP: viable now, worth ~1.2x, not 2-3x

Qwen3.5 checkpoints ship a Multi-Token Prediction head (`mtp_num_hidden_layers: 1`)
that predicts token *t+2* from the backbone hidden state at *t*. mlx-lm strips it,
but [mlx-lm#990](https://github.com/ml-explore/mlx-lm/pull/990) adds native support,
and it independently fixes both blockers documented above:

- the `+1.0` norm shift now keys on **unsanitized conv1d, not on the presence of
  MTP weights**, so re-adding `mtp.*` to a converted checkpoint no longer
  double-shifts every RMSNorm;
- `cache.py` gains a `rollback_state` slot, and `n_confirmed=1` makes
  `GatedDeltaNet` snapshot conv/SSM state after the confirmed token — which is
  exactly the hybrid-rollback problem that makes ordinary speculation unsafe here.

So the upstream story is solved. The question is whether it helps *this* runtime.

**It does not, and the reason is structural.** MTP verifies `[t+1, t+2]` in one
`T=2` backbone pass. On a resident machine that pass re-reads the *union* of the
two tokens' expert sets — cheaper per token than two passes, which is where the
speedup comes from. On slotbank, decode is 94.3% expert copy, so the payoff
depends entirely on how much that union overlaps.

Measured directly: 48 tokens x 40 layers of real routing ids, replayed through
the LRU with the same capacity, `T=1` vs `T=k`.

| k | misses per pass | avg union | overlap | copy per pass | **copy per token** |
|---|---|---|---|---|---|
| 1 | 160.2 | 8.00 | 0% | 1.00x | **1.00x** |
| 2 | 324.7 | 12.79 | 20% | 2.03x | **1.01x** |
| 3 | 494.1 | 16.73 | 30% | 3.08x | **1.03x** |
| 4 | 656.7 | 20.52 | 36% | 4.10x | **1.02x** |

Adjacent tokens *do* share ~20% of their routed experts — but batching them
recovers none of it, because **the LRU has already harvested that overlap**: a
shared expert was resident from the previous token anyway. Copy-per-token is flat
at 1.00x, and it stays flat at **every capacity from C=8 to C=256**, so this is
not an artifact of the slot budget.

On the **old** copy path this capped MTP at `(1 + a) / 2.03` — break-even only at
perfect acceptance. That verdict was correct for a runtime spending 94.3% of
decode on copy, and it is worth keeping because it validated the cost model:
run for a *resident* machine (cost tracking union size rather than misses) the
same model predicts **1.11x** at 80% acceptance, and upstream measured **1.11x**
on this exact model (35B-A3B 8-bit, M2 Ultra) — a number it was not fitted to.
It also explains the one anomaly upstream: fp16 is **0.61x** with MTP, slower,
because speculation pays in compute and charges in bandwidth.

**The gating condition, and crossing it.** Writing *f* for the copy fraction of
decode, speedup is `(1 + a) / (2.03f + 1.05(1 - f))`. It crosses 1.0 near
f = 0.85. The [pread fix](#the-prefetch-cost-3x-the-read-it-was-hinting) took
f from **0.943 to 0.433**, so MTP flipped from economically rejected to
economically viable — not because anything about MTP changed, but because the
denominator did.

**This is a statement about cost, not correctness.** The numerics problem in
[Phase 3](#phase-3-mtp-the-ordinary-t1-path-is-blocked-on-numerics) is untouched
by any of this: ordinary T>1 execution on a hybrid model still diverges from
stock, at every copy fraction. What changed is that adopting the *specialised*
MTP path — which is correct — now buys something, where before it bought nothing.
Both conditions must hold, and only the second one moved.

#### Measured, not projected

`mlx-lm` still has no MTP in any release (PR #990 open), but **`mlx-vlm` ships
it today**: MTP arrives as a *separate drafter artifact* (`mtp_num_hidden_layers:
1`, ~0.46 GiB at 4-bit for 35B-A3B) loaded alongside an unmodified target. That
design also sidesteps the norm-shift trap entirely — the base checkpoint is never
touched, so the `+1.0` double-shift cannot fire.

Measured here on `Qwen3.5-4B` 4-bit (dense, but hybrid like our 35B: 24 linear /
8 full attention), M4 Air, greedy, 160 tokens:

| draft block | avg drafts | accepted | tok/s (3 reps) | mean | vs baseline |
|---|---|---|---|---|---|
| none | — | — | 14.94 / 14.34 / 15.14 | 14.81 | 1.00x |
| **2** | 1.00 | **75.4%** | 18.59 / 19.26 / 15.47 | **17.77** | **1.20x** |
| 4 | 3.00 | 48.6% | 17.58 / 15.41 / 13.39 | 15.46 | 1.04x |

A single-run sweep over more block sizes put the peak in the same place —
block 2 at 1.27x, block 3 at 1.22x, block 6 at **0.85x** (a net loss). Acceptance
decays sharply with position: 76.7% at the first drafted token, 32.5% by the
fifth. Block 2 also used *less* peak memory than the baseline (3.390 vs 3.574
GB); block 4 slightly more (3.598 GB).

**Greedy output is byte-identical** to the non-speculative run at blocks 2 and 4
(273 bytes, exact match) — so mlx-vlm's `rollback_speculative_cache` is correctly
restoring GDN/SSM state on rejection. That is the hybrid-rollback problem solved
in practice, on the same architecture family as our 35B.

These figures are **not comparable to slotbank's 5.98 tok/s**: this is a 4B that
fits in RAM with room to spare, not a 19 GiB bank that does not fit. The 4B was
chosen only to measure the acceptance rate, which was the missing input to the
projection below.

**A cache-bound runtime wants a shorter draft block than a resident one.** Its
marginal cost per drafted token is the expert union (2.03x copy for one draft,
4.10x for three), where a resident runtime's marginal cost is near zero. Feeding
the measured acceptance into the warm budget:

| block | tokens/round | copy | ms/token | projected |
|---|---|---|---|---|
| **2** | 1.77 | 2.03x | 140.6 | **1.19x** |
| 3 | 2.18 | 3.08x | 149.6 | 1.12x |
| 4 | 2.52 | 4.10x | 159.3 | 1.05x |

Break-even acceptance is **49%**; the 4B measures 76.7%. The 4B independently
agrees on the ordering (block 2 fastest, block 6 a net loss).

#### If we ever wanted MTP *in* slotbank

Three paths, and the constraint is not where it looks. `mlx-lm`'s `qwen3_5`
does **not** implement `rollback_speculative_cache`; only `mlx-vlm`'s does. That
hook is what restores GDN/SSM state after a rejected draft, so it is the thing
that makes speculation safe on a hybrid model — the exact problem documented in
[Speculative decoding is unsafe on hybrid linear-attention models](#speculative-decoding-is-unsafe-on-hybrid-linear-attention-models).

1. **Depend on `mlx-vlm`.** Works today, but it is a *vision* package: +372 MiB
   over slotbank's venv, dragging `opencv-python`, `mlx-audio`, `miniaudio`,
   `llguidance`. MTP lives there only because the Qwen3.5 model class does.
2. **Vendor the drafter.** `speculative/mtp.py` + `drafters/qwen3_5_mtp/` is
   small, but it is useless without the model class that owns the rollback hook,
   and that is the bulk of the code — a fork of fast-moving upstream model files.
3. **Port the hook.** Add `rollback_speculative_cache` to the path slotbank
   already uses, and implement the drafter head in `runtime.py`.

Worth recording: **slotbank's expert slotting is already compatible with
`mlx-vlm`.** It patches by class *name*, and `QuantizedSwitchLinear`,
`SwitchLinear`, `SwitchGLU` and `SwitchMLP` all exist under both packages with
the same names, so `install_expert_slots` would attach unchanged. One caveat —
mlx-vlm routes MoE through `_target_verify_switch_glu` during verification, and
at `C=32`/top-k 8 a block-2 verify presents 16 routed ids (fits `_is_decode`)
while block 4 presents 32, exactly on the boundary. Another reason block 2 is
the right size here.

So the honest expectation for slotbank is **~1.2x, not the 2–3x** the original
issue speculated — contingent on the 35B MoE showing first-position acceptance
near the 4B's. Untested, because no MTP drafter exists for `Qwen3.5-35B-A3B`;
`mlx-community/Qwen3.6-35B-A3B-MTP-4bit` does, but needs its matching 19 GiB
Qwen3.6 target.


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
- **Expert-interleaved checkpoint — retested, and the earlier verdict was half wrong.** A miss costs 9 reads (gate/up/down x weight/scales/biases) scattered across 9 tensors. Rewriting the bank so one expert's 1.6875 MiB is contiguous makes a miss **one** `pread`. First attempt measured **1.7x slower** and was reverted; that run still called `madvise` first, the same confound that made threaded `pread` look worthless, so it was retested after the pread fix.

  Built 10 of 40 layers spread across depth (4.22 GiB), verified bit-exact on 60 random (module, expert, piece) rows, and compared interleaved against plain layers **inside a single decode run** — which removes the run-to-run variance that has misled this project repeatedly. Three reps: **1.04x, 1.03x, 1.05x** faster per layer-call. Cutting preads 23x (753/token to 33/token) bought 3-5%.

  So the regression was indeed the `madvise` confound, but the *win* was never there: **read count is not the cost, bytes are.** Reads already run at 3637 MiB/s against a measured 4015 MiB/s ceiling, and 63% of them are page-cache hits where a `pread` is a memcpy. Not worth 16.9 GiB of disk. One caveat kept for honesty: the partial file duplicates pages the original checkpoint also holds, which handicaps it slightly; a full 40-layer replacement would not, but a 23x syscall cut returning 4% caps how much that could be worth.

  Three read-pattern micro-benchmarks have now predicted large wins that reversed end-to-end (sidecar, threaded `pread`, interleaving). **Micro-benchmarks of read shape do not predict decode throughput on this system** — measure end-to-end, interleaved, before believing any of them.
- **Hot-expert sidecar.** The profile identifies the ~64 experts per layer carrying 76% of traffic — only 4.2 GiB. Copying those into one contiguous file should have let them be read sequentially and stay in page cache. Built in 11.2 s, verified bit-identical. Per-row reads did get faster (570-685 us vs 1116 us from the checkpoint) — but end-to-end decode was **2.5x slower** (0.57-0.70 vs 1.62-1.68 tok/s, three interleaved pairs). Best explanation: the sidecar duplicates bytes that already exist in the checkpoint, so page cache holds two copies and the effective cache halves, while two separate mappings defeat the kernel's readahead on both. A version that *replaced* rather than duplicated (i.e. full checkpoint reordering) would not have this flaw.
- **Parallel per-layer decode reads.** A layer misses ~3 experts x 9 banks = ~31 rows, previously read serially on one core. `pread` releases the GIL (mmap slices do not), so a pool should have used more than one core. Interleaved: serial median 2.94, 4 threads 2.79, 8 threads 2.98 tok/s — no effect. The kernel's readahead is already providing the parallelism.
- **One array + one scatter per bank** (instead of one `pack[s] = row` per expert per bank). The per-row form builds ~1100 MLX graph nodes per token, so this looked like obvious dispatch savings. Interleaved A/B: median 2.55 vs 2.69 tok/s — slightly *worse*. Joining the rows into one buffer costs more than the graph nodes it saves. A `mx.stack` variant was worse still and tripped the resident-memory test.
- **Native MTP — re-opened, not rejected.** On the old copy path (94.3% copy) the `T=2` verify pass copied 2.03x a `T=1` decode and capped MTP below 1.0x. The pread fix took copy to 43.3% and flipped it: projected **1.19x** at draft block 2, break-even at 49% acceptance vs 76.7% measured. See *Native MTP* above.
- **Parallelising `MADV_WILLNEED`, four ways.** The advise is 46% of decode, so making it concurrent looked like the whole game. Threading CPython's `mmap.madvise` does nothing — it does not release the GIL. Calling `madvise` through `ctypes` (which does) and threading *that* also does nothing: the kernel serialises on the VM object lock, since all rows of a bank share one mapping. Serial ctypes was worse than the builtin (2.06 vs 2.41 tok/s) on call overhead. The advise cost is per-page VM work and it does not parallelise.
- **Skipping the advise for rows already resident.** Half the advised rows are already in page cache — `mincore` confirms 46–55%. Filtering them out with `mincore` cost as much as the advise it skipped (2.46 vs 2.56 tok/s). A bounded LRU of recently-advised rows was worse still as a filter: only a **1%** hit rate, because a slot-cache miss almost by definition means the row has not been touched recently.
- **Threaded `pread` for batched reads.** 3.6× on a cold micro-benchmark (96.5 → 26.6 ms for 200 rows), but a **regression end-to-end** — prefill at 4096 tokens went 105.6 s → 112.7 s. Real prefill hits substantially warm pages, where mmap is a memcpy and pread is a syscall plus a copy, and ~120 batched calls per chunk each dispatch hundreds of pool tasks. Reverted. The micro-benchmark did not predict the real workload because it used fresh tensors, so every read was cold.

## Tests

```bash
python -m pytest tests -q
```

51 tests. `tests/test_offload_cache.py` and `tests/test_expert_slots.py` need `mlx` and `mlx-lm`. The rest is CPU.

Only `runtime.py`, `expert_slots.py`, and `offload_cache.py` may import MLX; `tests/test_fence.py` enforces this with an AST check.

## License

Apache License 2.0. See `LICENSE`.
