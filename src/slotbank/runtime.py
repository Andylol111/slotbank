from __future__ import annotations

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


def _model_sources():
    """Loaders to try, in order. SLOTBANK_LOADER pins one.

    Expert slotting attaches by class *name*, so it works against any package
    whose MoE is built from SwitchGLU/SwitchLinear -- verified against both
    mlx-lm and mlx-vlm. Keeping the loader pluggable is what decouples model
    coverage from a single package's release cadence: as of mlx-lm 0.31.3,
    `deepseek_v4`, `kimi_k3` and `minimax_m3` exist only in mlx-vlm.
    """
    import os

    pinned = os.environ.get("SLOTBANK_LOADER", "").strip().lower()
    order = ["mlx_lm", "mlx_vlm"]
    if pinned in order:
        order = [pinned]
    return order


def _load_model(model_path: str, kwargs: dict):
    """Load through the first source that supports this architecture.

    An unsupported model_type raises ValueError deep inside a loader; that is a
    reason to try the next source, not to fail. Anything else -- a missing file,
    a corrupt checkpoint -- is re-raised immediately so real errors are not
    masked by a fallback attempt.
    """
    errors = []
    for name in _model_sources():
        try:
            mod = __import__(name, fromlist=["load"])
        except ImportError:
            errors.append(f"{name}: not installed")
            continue
        try:
            return mod.load(model_path, **kwargs)
        except (ValueError, KeyError, AttributeError) as exc:
            # unsupported architecture, or tensor names this package rejects
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        except TypeError as exc:
            # signature drift between packages: retry without the extras
            try:
                return mod.load(model_path)
            except Exception:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
    raise ValueError(
        "no installed loader could load this model:\n  "
        + "\n  ".join(errors)
    )


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


def _copy_state(st):
    """Deep-copy a cache state, preserving container type.

    ArraysCache assigns into its state by index, so a tuple breaks restore.
    """
    import mlx.core as mx

    if isinstance(st, (tuple, list)):
        items = [mx.array(a) if hasattr(a, "shape") else a for a in st]
        return tuple(items) if isinstance(st, tuple) else items
    return mx.array(st) if hasattr(st, "shape") else st


def _state_bytes(st) -> int:
    if isinstance(st, (tuple, list)):
        return sum(int(a.nbytes) for a in st if hasattr(a, "nbytes"))
    return int(st.nbytes) if hasattr(st, "nbytes") else 0


class PrefixCache:
    """KV/recurrent state keyed by token prefix.

    Agents sharing a system prompt otherwise each pay full prefill. Entries are
    copied out of the live cache because the model mutates those arrays in place
    as generation continues.
    """

    MIN_PREFIX = 32
    BLOCK = 128          # snapshot boundary; reuse is block-granular

    def __init__(self, max_entries: int = 4, max_bytes: int = 512 << 20):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
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
        import mlx.core as mx

        if cache is None or len(ids) < self.MIN_PREFIX:
            return 0
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
        flat = [a for st, _m in states for a in (st if isinstance(st, tuple) else (st,))
                if hasattr(a, "shape")]
        if flat:
            mx.eval(*flat)
        self._entries.append((list(ids), states, nbytes))
        while len(self._entries) > self.max_entries or \
                sum(e[2] for e in self._entries) > self.max_bytes:
            self._entries.pop(0)
        return nbytes

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


def _compile_logprobs():
    import mlx.core as mx

    def logprobs(logits):
        return logits - mx.logsumexp(logits, keepdims=True)

    return mx.compile(logprobs, shapeless=True)


class Runtime:
    def __init__(self, args, *, eos_token_ids: set[int] | None = None, um=None):
        self._model_path = args.model_path
        self._prefill_step_size = _prefill_step(
            getattr(args, "prefill_step_size", 2048) or 2048
        )
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
        self._prefix = None if _prefix_disabled() else PrefixCache()
        self._wired = 0

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_path(self) -> str:
        return self._model_path

    def load(self, progress=None) -> None:
        if progress is not None:
            progress("load", 0, 1)
        kwargs = self.um.load_kwargs() if self.um is not None else {
            "lazy": True,
            "tokenizer_config": {"trust_remote_code": False},
        }
        model, tokenizer = _load_model(self._model_path, kwargs)
        from slotbank.expert_slots import install_expert_slots

        install_expert_slots(model, model_path=self._model_path, um=self.um)
        self._wired = self._raise_wired_limit()
        self._model = model
        # The warm pass is deferred to the first request that can pay for it.
        # See _maybe_warm: it costs ~6.5 s and returns ~49.5 ms/token, so it is
        # a loss below ~131 tokens.
        self._tokenizer = tokenizer
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            self._eos_token_ids.add(int(eos))
        self._logprobs = _compile_logprobs()
        if progress is not None:
            progress("load", 1, 1)

    def _raise_wired_limit(self) -> int:
        """Let MLX wire the whole admitted working set, not just the expert pack.

        HotResidency sizes the limit from pack bytes alone, which leaves the
        dense weights and KV evictable -- and another app using memory will push
        exactly those out. The limit is a cap, not a reservation, so raising it
        costs nothing when the model is smaller than the cap.
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
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        ids = [int(x) for x in input_ids]
        if not ids:
            raise ValueError("empty prompt")
        self._cancelled = False
        self._prompt_ids = ids
        self._sampling_params = sampling_params
        self._generated = []
        self._pending = []
        self._spec_ok = None
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
        prompt = mx.array(ids, dtype=mx.int32)
        self._sampler = make_sampler(
            temp=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )
        prefix_n = len(ids) - 1
        offset = reuse
        step = _adaptive_step(self._prefill_step_size, prefix_n)
        snaps = self._snap_points(offset, prefix_n)   # uses self._prompt_ids
        while offset < prefix_n:
            if self._cancelled:
                raise Cancelled()
            end = min(offset + step, prefix_n)
            for sp in snaps:
                if offset < sp < end:
                    end = sp          # stop on the boundary so it can be cached
                    break
            self._model(prompt[offset:end][None], cache=self._cache)
            self._fed_ids.extend(ids[offset:end])
            states = _cache_states(self._cache)
            if states is not None:
                mx.eval(states)
            mx.clear_cache()
            offset = end
            if self._prefix is not None and offset in snaps:
                try:
                    self._prefix.put(ids[:offset], self._cache)
                except (ValueError, TypeError, RuntimeError):
                    pass
        if self._prefix is not None and prefix_n >= PrefixCache.MIN_PREFIX:
            try:
                self._prefix.put(ids[:prefix_n], self._cache)
            except (ValueError, TypeError, RuntimeError):
                pass
        self._last_token = ids[-1]

    def _snap_points(self, start: int, prefix_n: int) -> set:
        """Block boundaries to stop prefill on so the state can be cached.

        The cache cannot be trimmed on hybrid models (30 of 40 layers hold
        recurrent state), so a reusable prefix must land exactly where prefill
        stopped. Splitting the first chunk costs one extra chunk; a hit saves
        the whole block every time.
        """
        if self._prefix is None:
            return set()
        b = PrefixCache.BLOCK
        pts = [n for n in range(b, prefix_n, b) if n > start]
        # Skip boundaries already cached: re-splitting the chunk costs ~50% of
        # prefill and buys nothing when the snapshot already exists.
        pts = [n for n in pts if not self._prefix.has(self._prompt_ids[:n])]
        return set(pts[:2])                  # cap: each snapshot costs ~63 MiB

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
        logits = self._model(toks, cache=self._cache)
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
        logits = self._model(token[None], cache=self._cache)
        self._fed_ids.append(int(self._last_token))
        logits = logits[:, -1, :]
        logprobs = self._logprobs(logits)
        sampled = self._sampler(logprobs)
        token_id = _as_host_token(sampled)
        return self._finish_token(token_id)

    def _finish_token(self, token_id: int) -> GenerationStep:
        """Shared bookkeeping: stop checks and the step record.

        Used by both the one-token path and the speculative path, so a token
        emitted either way is accounted identically.
        """
        import mlx.core as mx

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
        return GenerationStep(
            token_id=token_id,
            finished=finished,
            finish_reason=reason,
            matched_stop=matched,
            active_memory_bytes=int(mx.get_active_memory() or 0),
        )

    def cancel(self) -> None:
        self._cancelled = True

    def close(self) -> None:
        import mlx.core as mx

        self.save_profile()
        mx.clear_cache()
        self._model = None
        self._tokenizer = None
        self._cache = None
        self._fed_ids = []
        self._pending = []
        self._logprobs = None

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
