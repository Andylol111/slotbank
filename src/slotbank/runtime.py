from __future__ import annotations

import gc

from slotbank.types import GenerationStep, SamplingParams


def _prefix_disabled() -> bool:
    import os

    # Off by default: the first request pays ~15 s to snapshot, which is pure
    # cost unless several requests share a prefix. Worth it from ~3 spawns.
    return os.environ.get("SLOTBANK_PREFIX_CACHE", "0").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    )


def reuse_prefill_start(fed: list[int], ids: list[int]) -> int:
    if not fed or len(ids) <= 1:
        return 0
    n = len(fed)
    if n >= len(ids) or ids[:n] != fed:
        return 0
    return n


def draft_feed(fed: list[int], ids: list[int], has_cache: bool) -> tuple[int, list[int]]:
    """Append-only suffix for DFlash. Never trim — hybrid GDN cannot rewind.

    Reuse only when the new prompt starts with every token already in the
    cache. A shorter shared prefix would need trim_prompt_cache, which is a
    no-op on ArraysCache and would silently condition on extra tokens.
    """
    reuse = reuse_prefill_start(fed, ids) if has_cache and fed else 0
    if reuse <= 0:
        return 0, ids
    feed = ids[reuse:]
    return (reuse, feed) if feed else (0, ids)


def draft_reuse(
    fed: list[int],
    ids: list[int],
    has_cache: bool,
    stored_prefix: list[int] | None = None,
) -> tuple[int, list[int]]:
    """Live append first; else an exact PrefixCache snapshot. Never a partial trim.

    OMP re-encodes the whole chat each turn, so `_fed_ids` (prompt + generated
    ids) is usually not a prefix of the new prompt. The system head is. Restore
    that snapshot into a *new* cache; the live cache still holds extra tokens.
    """
    reuse, feed = draft_feed(fed, ids, has_cache)
    if reuse > 0:
        return reuse, feed
    if not stored_prefix:
        return 0, ids
    n = len(stored_prefix)
    if n < PrefixCache.MIN_PREFIX or n >= len(ids) or ids[:n] != stored_prefix:
        return 0, ids
    feed = ids[n:]
    return (n, feed) if feed else (0, ids)


def dflash_session_ok(cache, fed_n: int) -> bool:
    """False when attention-layer offset disagrees with tokens we recorded.

    ArraysCache (Gated DeltaNet) has no offset. If DFlash accepted tokens
    past EOS, KVCache.offset runs ahead of `_fed_ids`; reusing that cache
    would condition the next turn on tokens the client never saw.
    """
    offs = [
        int(c.offset)
        for c in (cache or [])
        if getattr(c, "offset", None) is not None
    ]
    if not offs:
        return True
    return min(offs) == max(offs) == fed_n


# Architectures mlx-lm has no class for. qwen4_exp is experimental and lives
# in mlx-vlm git main (2026-08-27+); the others are in the vlm extra already.
_VLM_ONLY = frozenset({"qwen4_exp", "deepseek_v4", "kimi_k3", "minimax_m3"})
# Dense Qwen3.8 VLMs are qwen3_5 with a vision tower; mlx-lm can load text
# but mlx-vlm owns the multimodal graph. Prefer vlm when vision_config is set.
_VLM_LOAD_KEYS = ("lazy", "adapter_path", "revision", "strict")


def _config_model_type(model_path: str) -> str:
    from slotbank.admit import load_hf_config

    cfg = load_hf_config(model_path)
    raw = cfg.get("model_type") or (cfg.get("text_config") or {}).get("model_type") or ""
    return str(raw).lower()


def _model_sources(model_path: str | None = None):
    """Loaders to try, in order. SLOTBANK_LOADER pins one.

    Expert slotting attaches by class *name*, so it works against any package
    whose MoE is built from SwitchGLU/SwitchLinear -- verified against both
    mlx-lm and mlx-vlm. Keeping the loader pluggable is what decouples model
    coverage from a single package's release cadence: as of mlx-lm 0.31.3,
    `deepseek_v4`, `kimi_k3` and `minimax_m3` exist only in mlx-vlm.
    `qwen4_exp` (Qwen3.8-Flash-Next) is the same, and is experimental: it
    needs mlx-vlm git main, not a released mlx-lm.
    """
    import os

    pinned = os.environ.get("SLOTBANK_LOADER", "").strip().lower()
    order = ["mlx_lm", "mlx_vlm"]
    if model_path:
        from slotbank.admit import load_hf_config

        cfg = load_hf_config(model_path)
        kind = str(
            cfg.get("model_type")
            or (cfg.get("text_config") or {}).get("model_type")
            or ""
        ).lower()
        # Text-only is the 24 GB default: mlx-lm drops the vision tower
        # (~0.4 GiB on Qwen3.8-27B) and the vlm wrapper graph. Opt in with
        # SLOTBANK_VISION=1 or --vision. Architectures mlx-lm cannot load
        # still go through vlm first.
        if _draft_on():
            # DFlash verify lives in mlx-vlm (rollback_speculative_cache).
            order = ["mlx_vlm"]
        elif kind in _VLM_ONLY or (_vision_on() and cfg.get("vision_config")):
            order = ["mlx_vlm", "mlx_lm"]
    if pinned in order:
        order = [pinned]
    return order


def _loader_kwargs(name: str, kwargs: dict) -> dict:
    """mlx-lm takes tokenizer_config; mlx-vlm.load forwards extras into
    load_model. A TypeError retry without kwargs would drop lazy=True and
    eager-materialise a 100 GB Flash-Next. So vlm gets only its real keys."""
    if name != "mlx_vlm":
        return kwargs
    return {k: kwargs[k] for k in _VLM_LOAD_KEYS if k in kwargs}


def _as_tokenizer(obj):
    """mlx-lm returns a tokenizer; mlx-vlm returns a ProcessorMixin.

    encode/decode live on processor.tokenizer. Using the processor as a
    tokenizer 500s the chat path the first time a VLM loads.
    """
    if callable(getattr(obj, "encode", None)):
        return obj
    inner = getattr(obj, "tokenizer", None)
    if callable(getattr(inner, "encode", None)):
        return inner
    raise ValueError("loader returned no tokenizer.encode")


def _vlm_hint(kind: str) -> str:
    base = "install with: pip install 'slotbank[vlm]'"
    if kind == "qwen4_exp":
        return (
            f"{base}; qwen4_exp is experimental and needs mlx-vlm git main "
            "(2026-08-27+): pip install git+https://github.com/Blaizzy/mlx-vlm.git"
        )
    return base


def _load_model(model_path: str, kwargs: dict):
    """Load through the first source that supports this architecture.

    An unsupported model_type raises ValueError deep inside a loader; that is a
    reason to try the next source, not to fail. Anything else -- a missing file,
    a corrupt checkpoint -- is re-raised immediately so real errors are not
    masked by a fallback attempt.
    """
    kind = _config_model_type(model_path)
    errors = []
    for name in _model_sources(model_path):
        try:
            mod = __import__(name, fromlist=["load"])
        except ImportError:
            errors.append(f"{name}: not installed")
            if name == "mlx_vlm":
                errors.append(_vlm_hint(kind))
            continue
        try:
            model, tok = mod.load(model_path, **_loader_kwargs(name, kwargs))
            return model, _as_tokenizer(tok)
        except (ValueError, KeyError, AttributeError) as exc:
            # unsupported architecture, or tensor names this package rejects
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            if name == "mlx_vlm" and kind == "qwen4_exp":
                errors.append(_vlm_hint(kind))
            continue
        except TypeError as exc:
            if name == "mlx_vlm":
                # Do not retry without lazy=True: that eval's the whole bank.
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
            try:
                model, tok = mod.load(model_path)
                return model, _as_tokenizer(tok)
            except Exception:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
    raise ValueError(
        "no installed loader could load this model:\n  "
        + "\n  ".join(errors)
    )


def _vision_on() -> bool:
    import os

    return os.environ.get("SLOTBANK_VISION", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _draft_on() -> bool:
    import os

    return bool(os.environ.get("SLOTBANK_DRAFT", "").strip())


def _is_dense(um) -> bool:
    card = getattr(um, "card", None) if um is not None else None
    return getattr(card, "kind", None) == "dense"


def _dense_tight(um) -> bool:
    """Dense weights eating most of the working set — 27B on 24 GB."""
    if um is None or not _is_dense(um):
        return False
    card, profile = um.card, getattr(um, "profile", None)
    ws = getattr(profile, "max_working_set_bytes", 0) or 0
    stored = int(getattr(card, "stored_bytes", 0) or 0)
    return bool(ws and stored * 10 > ws * 7)


def _kv_bits(um=None) -> int | None:
    """SLOTBANK_KV_BITS: quantise the KV cache to 8 or 4 bits.

    Off by default on MoE. Dense packs that already fill the working set get
    8-bit automatically so 8k–16k context does not push peak into the Metal
    buffer cap (13.3 GiB on this M4 Air). Set SLOTBANK_KV_BITS=0 to refuse.

    The cache is 96 KiB per token on Qwen3-30B-A3B (48 layers x 4 kv heads x
    128 head dim x 2 for K and V x 2 bytes), so it -- not the model -- sets the
    ceiling on context. Measured on that model over 1024 tokens of real text:

        bits   KiB/token   K rel err   attention top-1 preserved
        fp16        96.0           -                      100.0%
        8-bit       51.0      0.0138                       96.9%
        4-bit       27.0      0.1462                       78.1%

    8-bit is the supported setting. 4-bit is accepted but not recommended: mlx
    groups scales along head_dim, and K's outliers sit in a fixed channel
    (measured: the same channel is the largest for 93.6% of tokens), so a
    per-token group takes 3.5x the error a per-channel group would. That is a
    deliberate mlx trade -- per-channel groups span 64 tokens and cannot be
    formed one token at a time -- not a defect, and fixing it needs a KIVI-style
    fp16 residual buffer.
    """
    import os

    # DFlash verify indexes keys.shape; QuantizedKVCache stores (w, scales).
    if _draft_on():
        return None
    raw = os.environ.get("SLOTBANK_KV_BITS", "").strip()
    if raw.lower() in {"0", "off", "none", "fp16"}:
        return None
    if raw:
        try:
            bits = int(raw)
        except ValueError:
            return None
        return bits if bits in (4, 8) else None
    return 8 if _dense_tight(um) else None


def _kv_quant_start(um=None) -> int:
    """SLOTBANK_KV_START: context length past which the cache is quantised.

    Conversion is one-way and whole-cache, so short chats are left exact: the
    cache is only ~16% of the bytes read per token at 8k context, and all of the
    quality cost lands on long-context retrieval, which is where the memory is
    needed anyway. Dense-tight packs start at 0 so the first tokens already
    use the smaller cache.
    """
    import os

    raw = os.environ.get("SLOTBANK_KV_START", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return 0 if _dense_tight(um) else 4096


def _prefill_step(default: int) -> int:
    """SLOTBANK_PREFILL_STEP: tokens per prefill chunk.

    Prefill activations are the only part of the footprint that grows sharply
    with context -- measured 3.61 -> 5.06 GiB peak going from 256 to 4096
    tokens, while wired memory moved only 3.46 -> 3.54. Smaller chunks trade
    prefill speed for a lower peak, which is what matters when something else
    on the machine needs the RAM.
    """
    import os

    raw = os.environ.get("SLOTBANK_PREFILL_STEP", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return int(default)


def _prefill_budget() -> int:
    """SLOTBANK_PREFILL_BUDGET: cap on chunk x context, in token-pairs.

    The transient peak during prefill is dominated by the attention score
    matrix of the full-attention layers, which is proportional to
    chunk x context -- so a *fixed* chunk cannot bound it. Measured: at chunk
    2048 the peak was 5.49 GiB at 8k context but 7.93 GiB at 32k, and a 64k
    prefill drove this machine to 571 MB of free swap.
    Capping the product instead keeps the peak flat as context grows, at the
    cost of more chunks. The default is the 2048 x 8192 point, which measured a
    5.49 GiB peak.
    """
    import os

    raw = os.environ.get("SLOTBANK_PREFILL_BUDGET", "").strip()
    if raw:
        try:
            return max(1 << 16, int(raw))
        except ValueError:
            pass
    return 2048 * 8192


def _adaptive_step(step: int, prefix_n: int) -> int:
    """Shrink the prefill chunk on long prompts so chunk x context stays capped."""
    if prefix_n <= 0:
        return step
    allowed = max(64, _prefill_budget() // prefix_n)
    return max(1, min(step, allowed))


def _pyramid_step(step: int, offset: int, prefix_n: int) -> int:
    """Size this prefill tile from the *current* prefix, not the final length.

    Attention peak is chunk × (offset + chunk). A uniform budget//prefix_n
    throttles the first 8k of a 32k prompt to the last-chunk size. Early
    tiles stay large; later tiles shrink. Same peak, fewer Metal launches.
    mlx-vlm generate_step still gets _adaptive_step — it takes one size.
    """
    remain = prefix_n - offset
    if remain <= 0:
        return 1
    budget = _prefill_budget()
    hi = max(1, min(int(step), remain))
    lo, best = 1, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if mid * (offset + mid) <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _tune_metal() -> None:
    """Cap the allocator cache so freed activations do not sit at peak.

    Peak on the 3.5bpw pack was 14.22 GiB against a 13.3 GiB max_buffer_length.
    A large MLX cache is how a 12.5 GiB language model grows another 1+ GiB
    after the first prefill. SLOTBANK_CACHE_LIMIT_MIB overrides (default 512).
    """
    import os

    import mlx.core as mx

    if not hasattr(mx, "metal") or not mx.metal.is_available():
        return
    raw = os.environ.get("SLOTBANK_CACHE_LIMIT_MIB", "512").strip()
    try:
        mib = int(raw)
    except ValueError:
        mib = 512
    if mib <= 0:
        return
    setter = getattr(mx, "set_cache_limit", None) or mx.metal.set_cache_limit
    setter(mib << 20)


def _generation_stream():
    import mlx.core as mx

    stream = getattr(_generation_stream, "_stream", None)
    if stream is None:
        stream = mx.new_thread_local_stream(mx.default_device())
        _generation_stream._stream = stream
    return stream


def _strip_unused(model) -> list[str]:
    """Drop vision / MTP modules when this process is text-only."""
    if _vision_on() or model is None:
        return []
    dropped = []
    for name in ("vision_tower", "visual", "vision_model", "mtp"):
        if getattr(model, name, None) is not None:
            setattr(model, name, None)
            dropped.append(name)
    if dropped:
        import gc

        import mlx.core as mx

        gc.collect()
        mx.clear_cache()
    return dropped


def _pin_dense(model) -> int:
    """Materialise lazy weights so decode does not fault them from SSD."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    leaves = [v for _, v in tree_flatten(model.parameters()) if hasattr(v, "nbytes")]
    for i in range(0, len(leaves), 64):
        mx.eval(*leaves[i : i + 64])
    mx.clear_cache()
    return sum(int(v.nbytes) for v in leaves)


def _model_bytes(model) -> int:
    from mlx.utils import tree_reduce

    return int(
        tree_reduce(
            lambda acc, x: acc + x.nbytes if hasattr(x, "nbytes") else acc,
            model,
            0,
        )
    )


def _copy_state(st):
    """Deep-copy a cache state, preserving container type.

    ArraysCache assigns into its state by index, so a tuple breaks restore.
    QuantizedKVCache nests `(w, scales, biases)` inside `(keys, values)` —
    one-level copy would alias the inner arrays to the live cache.
    """
    import mlx.core as mx

    if isinstance(st, (tuple, list)):
        items = [_copy_state(a) for a in st]
        return tuple(items) if isinstance(st, tuple) else items
    return mx.array(st) if hasattr(st, "shape") else st


def _state_bytes(st) -> int:
    if isinstance(st, (tuple, list)):
        return sum(_state_bytes(a) for a in st)
    return int(st.nbytes) if hasattr(st, "nbytes") else 0


def _prefix_budget_bytes() -> int:
    """SLOTBANK_PREFIX_CACHE_MIB: RAM for copied prefix snapshots.

    16 full-attn layers are ~64 KiB/token; GDN ArraysCache is ~150 MiB O(1).
    A 2048-token head is ~278 MiB (attn KV + GDN). Default 384 MiB so we
    keep that head, not a 10k envelope copy (~790 MiB) that jetsams 24 GB.
    """
    import os

    raw = os.environ.get("SLOTBANK_PREFIX_CACHE_MIB", "").strip()
    if raw:
        try:
            return max(64, int(raw)) << 20
        except ValueError:
            pass
    return 384 << 20


class PrefixCache:
    """KV/recurrent state keyed by token prefix.

    Agents sharing a system prompt otherwise each pay full prefill. Entries are
    copied out of the live cache because the model mutates those arrays in place
    as generation continues.
    """

    MIN_PREFIX = 32
    # Don't snapshot the full 10k envelope — that copy is ~790 MiB extra RAM.
    MAX_SNAP = 2048

    def __init__(self, max_entries: int = 2, max_bytes: int | None = None):
        self.max_entries = max_entries
        self.max_bytes = _prefix_budget_bytes() if max_bytes is None else max_bytes
        self._entries: list = []          # [(ids, states, nbytes)]
        self.hits = 0
        self.saved_tokens = 0

    def has(self, ids: list[int]) -> bool:
        return any(e_ids == ids for e_ids, _s, _n in self._entries)

    def find(self, ids: list[int]):
        best = None
        for e_ids, states, _ in self._entries:
            n = len(e_ids)
            if n < self.MIN_PREFIX or n >= len(ids):
                continue
            if ids[:n] == e_ids and (best is None or n > len(best[0])):
                best = (e_ids, states)
        return best

    def put(self, ids: list[int], cache) -> int:
        if cache is None or len(ids) < self.MIN_PREFIX:
            return 0
        if len(ids) > self.MAX_SNAP:
            return 0
        import mlx.core as mx
        for e_ids, _s, _n in self._entries:
            if e_ids == ids:
                return 0
        states, nbytes = [], 0
        for c in cache:
            st = _copy_state(c.state)
            states.append((st, getattr(c, "meta_state", None)))
            nbytes += _state_bytes(st)
        if nbytes > self.max_bytes:
            return 0
        flat = []
        stack = [st for st, _m in states]
        while stack:
            cur = stack.pop()
            if isinstance(cur, (tuple, list)):
                stack.extend(cur)
            elif hasattr(cur, "shape"):
                flat.append(cur)
        if flat:
            mx.eval(*flat)
        self._entries.append((list(ids), states, nbytes))
        self._evict()
        return nbytes

    def _evict(self) -> None:
        """Drop the shortest snapshot first so a 10k system head outlives 128-token crumbs."""
        while (
            len(self._entries) > self.max_entries
            or sum(e[2] for e in self._entries) > self.max_bytes
        ):
            if len(self._entries) <= 1:
                return
            i = min(
                range(len(self._entries)),
                key=lambda j: (len(self._entries[j][0]), self._entries[j][2], j),
            )
            self._entries.pop(i)

    def restore(self, cache, states) -> None:
        for c, (st, meta) in zip(cache, states):
            c.state = st
            if meta is not None and hasattr(type(c), "meta_state"):
                try:
                    c.meta_state = meta
                except (ValueError, TypeError):
                    pass


def propose_from_context(ids: list[int], k: int, ngram: int = 3) -> list[int]:
    """Prompt-lookup draft: find the most recent earlier occurrence of the last
    `ngram` tokens and propose what followed.

    No draft model and no MTP head; the candidate source is the context itself.
    Works well when output quotes or repeats the input (code, editing,
    summarising) and simply proposes nothing otherwise.
    """
    if k <= 0 or len(ids) <= ngram:
        return []
    tail = ids[-ngram:]
    for start in range(len(ids) - ngram - 1, -1, -1):
        if ids[start : start + ngram] == tail:
            cand = ids[start + ngram : start + ngram + k]
            if cand:
                return list(cand)
    return []


class Cancelled(Exception):
    pass


def _as_host_token(token) -> int:
    if hasattr(token, "item"):
        return int(token.item())
    return int(token)


def _cache_states(cache):
    if not cache:
        return None
    if all(hasattr(c, "state") for c in cache):
        return [c.state for c in cache]
    return cache


def prefill_forward(fwd, chunk, cache) -> None:
    """One prefill tile. Skip lm_head — we only need the cache for TTFT.

    Qwen3.8 vocab is 248320. mlx-vlm LanguageModel.__call__(skip_logits=True)
    omits that matmul. Older wrappers TypeError; fall back to a full call.
    """
    try:
        fwd(chunk, cache=cache, skip_logits=True)
        return
    except TypeError:
        pass
    try:
        fwd(chunk, cache=cache)
    except TypeError:
        fwd(inputs=chunk, cache=cache)


def _compile_logprobs():
    import mlx.core as mx

    def logprobs(logits):
        return logits - mx.logsumexp(logits, keepdims=True)

    return mx.compile(logprobs, shapeless=True)


class Runtime:
    def __init__(self, args, *, eos_token_ids: set[int] | None = None, um=None):
        self._model_path = args.model_path
        default_step = getattr(args, "prefill_step_size", 2048) or 2048
        # Pyramid + PREFILL_BUDGET cap the attention-score peak. A fixed 512
        # on dense 27B used to throttle every tile, including the first 8k.
        self._prefill_step_size = _prefill_step(default_step)
        self._eos_token_ids: set[int] = set(eos_token_ids or ())
        self.um = um
        self._model = None
        self._tokenizer = None
        self._logprobs = None
        self._cache = None
        self._sampler = None
        self._cancelled = False
        self._prompt_ids: list[int] = []
        self._generated: list[int] = []
        self._last_token = None
        self._sampling_params: SamplingParams | None = None
        self._fed_ids: list[int] = []
        self._pending: list[int] = []
        self._spec_proposed = 0
        self._spec_accepted = 0
        self._spec_ok = None
        self._warmed = 0
        self._total_generated = 0
        # PrefixCache copies via _copy_state, which recurses into nested
        # (w, scales, biases) so 8-bit KV snapshots do not alias the live cache.
        self._kv_bits = _kv_bits(um)
        self._kv_start = _kv_quant_start(um)
        self._prefix = None if _prefix_disabled() else PrefixCache()
        self._wired = 0
        self._draft = None
        self._draft_kind = "dflash"
        self._draft_block = None
        self._draft_cap = None
        self._dflash_cache = None
        self._pinned = False

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_path(self) -> str:
        return self._model_path

    def load(self, progress=None, *, pin: bool = True) -> None:
        """Map the graph. ``pin=False`` leaves mx.eval of the 15 GiB pack for ``pin()``.

        OMP's picker waits on /models/load, which waits for Engine ready.
        The ~30s is _pin_dense, not safetensors mmap. Serve sets ready after
        this returns so the spinner can drop; the worker then pins before
        the first generate.
        """
        if progress is not None:
            progress("load", 0, 1)
        _tune_metal()
        kwargs = self.um.load_kwargs() if self.um is not None else {
            "lazy": True,
            "tokenizer_config": {"trust_remote_code": False},
        }
        model, tokenizer = _load_model(self._model_path, kwargs)
        from slotbank.expert_slots import install_expert_slots

        install_expert_slots(model, model_path=self._model_path, um=self.um)
        _strip_unused(model)
        self._wired = self._raise_wired_limit(model)
        self._model = model
        # The warm pass is deferred to the first request that can pay for it.
        # See _maybe_warm: it costs ~6.5 s and returns ~49.5 ms/token, so it is
        # a loss below ~131 tokens.
        self._tokenizer = tokenizer
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            try:
                self._eos_token_ids.add(int(eos))
            except (TypeError, ValueError):
                pass
        extra = getattr(tokenizer, "eos_token_ids", None) or ()
        try:
            for t in extra:
                self._eos_token_ids.add(int(t))
        except TypeError:
            try:
                self._eos_token_ids.add(int(extra))
            except (TypeError, ValueError):
                pass
        self._logprobs = _compile_logprobs()
        self._load_draft()
        if pin:
            self.pin()
        if progress is not None:
            progress("load", 1, 1)

    def pin(self) -> None:
        """Fault 4-bit leaves into Metal so decode does not stall on SSD.

        Safe to call twice. Dense 27B + sidecar MTP are both pinned; MoE
        stays lazy until the warm pass.
        """
        if self._pinned:
            return
        if self._model is not None and _is_dense(self.um):
            _pin_dense(self._model)
        if self._draft is not None:
            _pin_dense(self._draft)
        self._pinned = True

    def _load_draft(self) -> None:
        import os
        from pathlib import Path

        path = os.environ.get("SLOTBANK_DRAFT", "").strip()
        if not path:
            return
        path = str(Path(path).expanduser())
        kind = os.environ.get("SLOTBANK_DRAFT_KIND", "").strip() or None
        block = os.environ.get("SLOTBANK_DRAFT_BLOCK", "").strip()
        from slotbank.admit import draft_block_from_config

        trained = draft_block_from_config(path)
        if block:
            self._draft_block = int(block)
            self._draft_cap = int(block)
        else:
            self._draft_block = trained
            self._draft_cap = trained
        from mlx_vlm.speculative.drafters import (
            load_drafter,
            validate_drafter_compatibility,
        )

        draft, resolved = load_drafter(path, kind=kind)
        if not hasattr(self._model, "language_model"):
            raise ValueError(
                "DFlash verify needs an mlx-vlm model with language_model"
            )
        validate_drafter_compatibility(self._model, draft, resolved)
        self._draft = draft
        self._draft_kind = resolved

    def draft_report(self) -> tuple[str | None, int | None, float | None]:
        """Last-round drafter kind, K, and accept rate. None without a draft."""
        if self._draft is None:
            return None, None, None
        from slotbank.tps import draft_accept_rate

        rate = draft_accept_rate(
            getattr(self._draft, "accept_lens", None),
            getattr(self._draft, "draft_lens", None),
        )
        return self._draft_kind, self._draft_block, rate

    def _raise_wired_limit(self, model=None) -> int:
        """Wire model bytes plus slop, not the whole working set.

        HotResidency sizes the limit from pack bytes alone, which leaves dense
        weights evictable. Wiring the entire 16 GiB working set for a 13 GiB
        model also leaves the Metal allocator no room under max_buffer_length
        (13.3 GiB on this M4). Cap at model + 2 GiB, the recommended WS, and
        the admitted working set.
        """
        import os

        import mlx.core as mx

        if not hasattr(mx, "set_wired_limit"):
            return 0
        override = os.environ.get("SLOTBANK_WIRED_MIB", "").strip()
        if override:
            mib = int(override)
            if mib <= 0:
                return 0
            try:
                mx.set_wired_limit(mib << 20)
            except (ValueError, RuntimeError):
                return 0
            return mib << 20
        info = mx.device_info() if hasattr(mx, "device_info") else {}
        cap = int(info.get("max_recommended_working_set_size") or 0)
        want = cap - (256 << 20) if cap else 0
        if self.um is not None and getattr(self.um, "profile", None) is not None:
            want = min(want or (1 << 62), int(self.um.profile.max_working_set_bytes))
        if model is not None:
            try:
                want = min(want or (1 << 62), _model_bytes(model) + (2 << 30))
            except (TypeError, ValueError):
                pass
        if want <= 0:
            return 0
        try:
            mx.set_wired_limit(want)
        except (ValueError, RuntimeError):
            return 0
        return want

    def _warm_budget(self) -> int:
        """How much page cache to claim for hot experts, scaled to live pressure.

        A fixed fraction of the working set is antisocial on a busy machine: it
        evicts other applications' cache and is evicted straight back. Warming
        is worth ~3.4x on a cold start, so it is worth claiming when the machine
        is idle and worth skipping when it is not.
        """
        ceiling = 4 << 30
        if self.um is not None and getattr(self.um, "profile", None) is not None:
            ceiling = max(1 << 30, int(self.um.profile.max_working_set_bytes // 3))
        share = 1.0
        if self.um is not None:
            try:
                snap = self.um.snapshot()
            except (OSError, ValueError):
                snap = None
            if snap is not None:
                if snap.pressure >= 4:
                    share = 0.0          # critical: claim nothing
                elif snap.pressure >= 2:
                    share = 0.25         # warn: a quarter
                # NOT keyed on free_bytes: macOS drives free memory to ~0 by
                # design and holds the rest as cache, so "low free" is the
                # normal steady state, not pressure. Keying on it halved the
                # budget on every idle machine and erased the warm-start win.
        return int(ceiling * share)

    def _warm(self, model) -> None:
        """Pull previously-hot experts into the page cache before the first
        token. Cold decode runs ~3.5x slower than warm, and the gap is entirely
        page-cache residency."""
        import os

        if os.environ.get("SLOTBANK_WARM", "1").strip().lower() in ("0", "false", "no", "off"):
            return
        from slotbank.expert_slots import warm_from_profile

        budget = self._warm_budget()
        override = os.environ.get("SLOTBANK_WARM_GIB", "").strip()
        if override:
            try:
                budget = max(0, int(float(override) * (1 << 30)))
            except ValueError:
                pass
        try:
            self._warmed = warm_from_profile(model, self._model_path, budget)
        except (OSError, ValueError):
            self._warmed = 0

    def save_profile(self) -> int:
        """Record the resident expert set so the next process starts warm."""
        if self._model is None:
            return 0
        from slotbank.expert_slots import save_hot_profile

        try:
            return save_hot_profile(self._model, self._model_path)
        except (OSError, ValueError):
            return 0

    def metadata(self) -> dict:
        import mlx.core as mx

        meta = {
            "runtime": "mlx",
            "active_memory_bytes": mx.get_active_memory(),
        }
        if self.um is not None:
            meta["um"] = self.um.note()
        return meta

    def shed_if_needed(self) -> bool:
        if self.um is None or not self.um.should_shed():
            return False
        if self._draft is not None:
            # The DFlash cache is the session. Dropping it on 24 GB pressure
            # (the normal steady state) forces a full re-prefill every turn.
            import mlx.core as mx

            mx.clear_cache()
            return True
        self._cache = None
        self._fed_ids = []
        import mlx.core as mx

        mx.clear_cache()
        return True

    def _warm_min_tokens(self) -> int:
        """Tokens the warm pass must pay back before it is worth running.

        Measured on 35B-A3B: the pass costs 6.47 s of load and lifts decode
        from 4.64 to 6.03 tok/s, saving 49.5 ms/token -- break-even at 131.
        It used to be worth ~3.4x unconditionally; reading rows straight into
        the pack made cold misses much cheaper, which shrank what warming buys.
        """
        import os

        raw = os.environ.get("SLOTBANK_WARM_MIN_TOKENS", "").strip()
        if raw:
            try:
                return max(0, int(raw))
            except ValueError:
                pass
        return 128

    def _maybe_warm(self, max_tokens: int) -> None:
        """Warm before a request that will amortise it, never mid-stream.

        Two signals, because neither alone is honest: ``max_tokens`` is an upper
        bound that HTTP callers default to 1024, and cumulative output proves a
        long-lived session (a server serving many short requests still wants
        warm experts). Either crossing the threshold is enough.
        """
        if self._warmed or self._model is None:
            return
        if max(int(max_tokens or 0), self._total_generated) < self._warm_min_tokens():
            return
        self._warm(self._model)

    def start_request(self, input_ids: list[int], sampling_params: SamplingParams) -> None:
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        self.pin()
        ids = [int(x) for x in input_ids]
        if not ids:
            raise ValueError("empty prompt")
        self._cancelled = False
        self._prompt_ids = ids
        self._sampling_params = sampling_params
        self._generated = []
        self._pending = []
        self._spec_ok = None
        if self._draft is not None:
            # mlx-vlm generate_step owns decode + GDN rollback.
            # Prefill, PrefixCache, and pyramid tiles run in _iter_draft.
            self._maybe_warm(getattr(sampling_params, "max_tokens", 0))
            return
        self._maybe_warm(getattr(sampling_params, "max_tokens", 0))
        reuse = reuse_prefill_start(self._fed_ids, ids) if self._cache is not None else 0
        if reuse == 0:
            self._cache = make_prompt_cache(self._model)
            self._fed_ids = []
            hit = self._prefix.find(ids) if self._prefix is not None else None
            if hit is not None:
                try:
                    self._prefix.restore(self._cache, hit[1])
                    self._fed_ids = list(hit[0])
                    reuse = len(hit[0])
                    self._prefix.hits += 1
                    self._prefix.saved_tokens += reuse
                except (ValueError, TypeError, RuntimeError):
                    self._cache = make_prompt_cache(self._model)
                    self._fed_ids = []
                    reuse = 0
        self._sampler = make_sampler(
            temp=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )
        prefix_n = len(ids) - 1
        if reuse < prefix_n:
            self._prefill_ids(ids, self._cache, reuse, prefix_n, commit_fed=True)
        self._quantize_kv()
        self._last_token = ids[-1]

    def _prefill_model(self):
        """Draft caches are sized for language_model; greedy uses the wrapper."""
        if self._draft is not None:
            return getattr(self._model, "language_model", self._model)
        return self._model

    def _prefill_ids(
        self,
        ids: list[int],
        cache,
        start: int,
        end: int,
        *,
        commit_fed: bool,
    ) -> None:
        """Pyramid-tile ids[start:end] into cache. Snapshot PrefixCache at heads.

        generate_step on the MTP path used to own this prefill as one chunk
        (and older mlx-vlm disabled chunking whenever a drafter was set).
        Owning the tiles keeps early Metal launches large and lets us copy
        the system head before decode mutates the arrays.
        """
        import mlx.core as mx

        if cache is None or start >= end:
            return
        prompt = mx.array(ids, dtype=mx.int32)
        snaps = self._snap_points(start, end, ids)
        stream = _generation_stream()
        fwd = self._prefill_model()
        offset = start
        while offset < end:
            if self._cancelled:
                raise Cancelled()
            step = _pyramid_step(self._prefill_step_size, offset, end)
            chunk_end = min(offset + step, end)
            for sp in snaps:
                if offset < sp < chunk_end:
                    chunk_end = sp
                    break
            with mx.stream(stream):
                chunk = prompt[offset:chunk_end][None]
                prefill_forward(fwd, chunk, cache)
                states = _cache_states(cache)
                if states is not None:
                    mx.eval(states)
            if commit_fed:
                self._fed_ids.extend(ids[offset:chunk_end])
            mx.clear_cache()
            offset = chunk_end
            if self._prefix is not None and offset in snaps:
                try:
                    self._prefix.put(ids[:offset], cache)
                except (ValueError, TypeError, RuntimeError):
                    pass
        if self._prefix is not None and end >= PrefixCache.MIN_PREFIX:
            try:
                self._prefix.put(ids[:end], cache)
            except (ValueError, TypeError, RuntimeError):
                pass

    def _quantize_kv(self) -> None:
        """Convert the KV cache to `SLOTBANK_KV_BITS` once it passes the start.

        mlx-lm's helper swaps each entry for a QuantizedKVCache in place. The
        swap is one-way and the replacement has no `to_quantized`, so repeat
        calls are a cheap hasattr check rather than repeated work.
        """
        if self._kv_bits is None or not self._cache:
            return
        from mlx_lm.generate import maybe_quantize_kv_cache

        maybe_quantize_kv_cache(
            self._cache,
            quantized_kv_start=self._kv_start,
            kv_group_size=64,
            kv_bits=self._kv_bits,
        )

    def _snap_points(self, start: int, prefix_n: int, ids: list[int] | None = None) -> set:
        """Geometric heads to stop prefill on so the state can be restored.

        Hybrid GDN cannot trim, so a reusable prefix must land exactly where
        prefill stopped. Packed OMP turns share the 25% sink head (~2048
        tokens at the 8k envelope). Short follow-ups share only the system
        head; 128/256 sit inside that when the full prefix_n includes a
        generation-prompt token the next turn does not start with. Keep the
        largest few boundaries; put() stores prefix_n only when it is
        <= MAX_SNAP (2048) so 8k copies do not jetsam 24 GB.
        """
        if self._prefix is None:
            return set()
        src = ids if ids is not None else self._prompt_ids
        pts: list[int] = []
        n = 128
        while n < prefix_n and n <= PrefixCache.MAX_SNAP:
            if n > start:
                pts.append(n)
            n *= 2
        pts = [p for p in pts if not self._prefix.has(src[:p])]
        return set(pts[-3:])

    def _lookahead(self) -> int:
        import os

        try:
            return max(0, int(os.environ.get("SLOTBANK_LOOKAHEAD", "0")))
        except ValueError:
            return 0

    def _can_trim(self) -> bool:
        if self._spec_ok is None:
            from mlx_lm.models.cache import can_trim_prompt_cache

            self._spec_ok = bool(self._cache) and can_trim_prompt_cache(self._cache)
        return self._spec_ok

    def _can_speculate(self) -> bool:
        """Only where a rejected draft can be rewound by trimming.

        Snapshot-and-restore was tried as a way to support recurrent caches --
        the copy is cheap (~3 ms against a ~300 ms token) -- but it produced
        output that differed from non-speculative decoding, and an unexplained
        correctness failure is exactly what this guard exists to prevent.
        See docs/mlx-lm-speculative-bug.md.
        """
        return self._can_trim()

    def _speculate(self) -> list[int]:
        """Verify k proposals in one pass. Returns the accepted tokens.

        Greedy-equivalent: a proposal is kept only where it equals the token the
        model would have produced anyway, so the output is identical.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import trim_prompt_cache

        k = self._lookahead()
        if k <= 0 or not self._can_speculate():
            return []
        ctx = self._prompt_ids + self._generated
        draft = propose_from_context(ctx, k)
        if not draft:
            return []
        toks = mx.array([[self._last_token] + draft], dtype=mx.int32)
        with mx.stream(_generation_stream()):
            logits = self._model(toks, cache=self._cache)
            states = _cache_states(self._cache)
            mx.eval(logits, states) if states is not None else mx.eval(logits)
        picked = [int(v) for v in mx.argmax(logits[0], axis=-1).tolist()]
        self._spec_proposed += len(draft)
        out = [picked[0]]
        for i, d in enumerate(draft):
            if d != out[-1]:
                break
            out.append(picked[i + 1])
        accepted_draft = len(out) - 1
        self._spec_accepted += accepted_draft
        # the pass advanced the cache by len(draft)+1; keep only what we used
        extra = len(draft) - accepted_draft
        if extra > 0:
            trim_prompt_cache(self._cache, extra)
        self._fed_ids.append(int(self._last_token))
        self._fed_ids.extend(out[:accepted_draft])
        return out

    def iter_steps(self):
        """Yield GenerationStep until finished. DFlash uses mlx-vlm verify."""
        if self._draft is not None:
            yield from self._iter_draft()
            return
        while True:
            step = self.step()
            yield step
            if step.finished:
                return

    def _iter_draft(self):
        import mlx.core as mx
        from mlx_vlm.generate.ar import generate_step
        from mlx_vlm.models import cache as vlm_cache

        if self._cancelled:
            raise Cancelled()
        sp = self._sampling_params
        if sp is None:
            raise ValueError("start_request first")
        # generate_step reads K once. A previous low-accept turn used to leave
        # this at 1 and the next OMP "hi" ran at greedy speed.
        self._arm_draft_block()
        prompt_ids = self._prompt_ids
        live_n, _live_feed = draft_feed(
            self._fed_ids, prompt_ids, self._dflash_cache is not None
        )
        hit = self._prefix.find(prompt_ids) if self._prefix is not None else None
        stored = hit[0] if hit is not None else None
        reuse, _feed = draft_reuse(
            self._fed_ids,
            prompt_ids,
            self._dflash_cache is not None,
            stored,
        )
        if live_n > 0:
            reuse = live_n
        elif reuse > 0 and hit is not None and self._prefix is not None:
            self._dflash_cache = vlm_cache.make_prompt_cache(self._model.language_model)
            try:
                self._prefix.restore(self._dflash_cache, hit[1])
                self._prefix.hits += 1
                self._prefix.saved_tokens += reuse
            except (ValueError, TypeError, RuntimeError):
                self._dflash_cache = vlm_cache.make_prompt_cache(
                    self._model.language_model
                )
                reuse = 0
        else:
            self._dflash_cache = vlm_cache.make_prompt_cache(self._model.language_model)
            reuse = 0
        prefix_n = len(prompt_ids) - 1
        try:
            if reuse < prefix_n:
                self._prefill_ids(
                    prompt_ids,
                    self._dflash_cache,
                    reuse,
                    prefix_n,
                    commit_fed=False,
                )
            # Do not prime Qwen mRoPE on the full prompt before decode.
            # mlx-vlm's helper writes deltas onto the LM and into kwargs;
            # generate_step then nulls them, and we never forwarded the
            # kwargs. On an 8k envelope that was a full-prompt rope index,
            # then discard. Pyramid tiles + last-token generate_step
            # continue from cache offset.
            ids = mx.array([[prompt_ids[-1]]], dtype=mx.int32)
            top_k = sp.top_k if sp.top_k and sp.top_k > 0 else 0
            # Last prompt token only: pyramid tiles already filled the prefix.
            kwargs: dict = {
                "max_tokens": int(sp.max_tokens),
                "temperature": float(sp.temperature),
                "top_p": float(sp.top_p),
                "top_k": int(top_k),
                "draft_model": self._draft,
                "draft_kind": self._draft_kind,
                "draft_block_size": self._draft_block,
                "prefill_step_size": _adaptive_step(self._prefill_step_size, 1),
                "prompt_cache": self._dflash_cache,
            }
            for token, _lp in generate_step(ids, self._model, None, None, **kwargs):
                if self._cancelled:
                    raise Cancelled()
                if isinstance(token, (list, tuple)):
                    token = token[0]
                step = self._finish_token(_as_host_token(token))
                # Engine closes this generator on step.finished; commit
                # before yield or the next turn cannot reuse the cache.
                self._fed_ids = list(self._prompt_ids) + list(self._generated)
                try:
                    yield step
                finally:
                    if step.finished and not dflash_session_ok(
                        self._dflash_cache, len(self._fed_ids)
                    ):
                        self._dflash_cache = None
                        self._fed_ids = []
                if step.finished:
                    return
        except Exception:
            self._dflash_cache = None
            self._fed_ids = []
            raise

    def _arm_draft_block(self) -> None:
        """Start every request at the trained/user K.

        generate_step takes one draft_block_size. Shrinking after a round
        cannot help that round, and persisting it made the next OMP turn
        inherit K=1. Daily 27B always uses the MTP/DFlash cap (3 or 8).
        """
        if self._draft_cap is not None:
            self._draft_block = int(self._draft_cap)

    def _retune_draft_block(self) -> None:
        """Move K by 1 from last-round accept. Never past the trained/user cap.

        Not used on the daily MTP door: generate_step cannot change K mid-round,
        and _arm_draft_block resets to the cap for the next request. Kept so
        SLOTBANK_DAIS can still be measured in isolation.
        """
        import os

        if self._draft is None or self._draft_cap is None:
            return
        if os.environ.get("SLOTBANK_DAIS", "1").strip().lower() in {
            "0", "false", "no", "off",
        }:
            return
        from slotbank.tps import draft_accept_rate, scale_draft_block

        rate = draft_accept_rate(
            getattr(self._draft, "accept_lens", None),
            getattr(self._draft, "draft_lens", None),
        )
        self._draft_block = scale_draft_block(
            cap=int(self._draft_cap),
            accept_rate=rate,
            current=self._draft_block,
        )

    def step(self) -> GenerationStep:
        import mlx.core as mx

        if self._cancelled:
            raise Cancelled()
        if self._pending:
            return self._finish_token(self._pending.pop(0))
        if self._lookahead() > 0 and self._sampling_params is not None \
                and self._sampling_params.is_greedy:
            got = self._speculate()
            if got:
                self._pending = got[1:]
                return self._finish_token(got[0])
        if self._logprobs is None:
            self._logprobs = _compile_logprobs()
        token = mx.array([self._last_token], dtype=mx.int32)
        with mx.stream(_generation_stream()):
            logits = self._model(token[None], cache=self._cache)
            logits = logits[:, -1, :]
            logprobs = self._logprobs(logits)
            sampled = self._sampler(logprobs)
            states = _cache_states(self._cache)
            if states is not None:
                mx.eval(sampled, states)
            else:
                mx.eval(sampled)
        self._fed_ids.append(int(self._last_token))
        token_id = _as_host_token(sampled)
        return self._finish_token(token_id)

    def _finish_token(self, token_id: int) -> GenerationStep:
        """Shared bookkeeping: stop checks and the step record.

        Used by both the one-token path and the speculative path, so a token
        emitted either way is accounted identically.
        """
        self._quantize_kv()
        self._generated.append(token_id)
        self._total_generated += 1
        self._last_token = token_id
        matched = self._match_stop()
        ignore_eos = bool(self._sampling_params and self._sampling_params.ignore_eos)
        hit_eos = (not ignore_eos) and token_id in self._eos_token_ids
        max_tokens = self._sampling_params.max_tokens if self._sampling_params else 0
        hit_length = len(self._generated) >= max_tokens
        finished = hit_eos or matched is not None or hit_length
        if finished:
            self._pending = []           # stop means stop, drop unused proposals
        if hit_eos or matched is not None:
            reason = "stop"
        elif hit_length:
            reason = "length"
        else:
            reason = None
        # Do not mx.get_active_memory() per token: nothing reads
        # GenerationStep.active_memory_bytes, and the query sits on the
        # decode loop.
        return GenerationStep(
            token_id=token_id,
            finished=finished,
            finish_reason=reason,
            matched_stop=matched,
        )

    def cancel(self) -> None:
        self._cancelled = True

    def close(self) -> None:
        self.save_profile()
        # Drop every reference first. Clearing the cache while the model is
        # still referenced frees nothing, and the buffers then land back in
        # MLX's allocator cache as they are released -- measured at 1.09 GiB
        # retained per unloaded model, which is exactly what makes an idle
        # unload useless in a multi-model process.
        self._model = None
        self._draft = None
        self._dflash_cache = None
        self._tokenizer = None
        self._cache = None
        self._fed_ids = []
        self._pending = []
        self._logprobs = None
        gc.collect()
        # Do not mx.clear_cache() here. After a Metal-thread generate,
        # clear_cache on Python 3.13 fatal-aborts (PyThreadState_Get /
        # GIL released). Dropped refs + process exit reclaim the allocator.

    def _match_stop(self) -> str | None:
        stops = (self._sampling_params.stop_strs if self._sampling_params else None) or []
        if not stops:
            return None
        decode = getattr(self._tokenizer, "decode", None)
        if decode is None:
            return None
        window = max((len(s) for s in stops if s), default=0)
        if not window:
            return None
        text = decode(self._generated[-window:])
        for s in stops:
            if s and s in text:
                return s
        return None
