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
        self._prefill_step_size = int(getattr(args, "prefill_step_size", 2048) or 2048)
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
        self._warmed = 0
        self._prefix = None if _prefix_disabled() else PrefixCache()
        self._wired = 0

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_path(self) -> str:
        return self._model_path

    def load(self, progress=None) -> None:
        from mlx_lm import load

        if progress is not None:
            progress("load", 0, 1)
        kwargs = self.um.load_kwargs() if self.um is not None else {
            "lazy": True,
            "tokenizer_config": {"trust_remote_code": False},
        }
        model, tokenizer = load(self._model_path, **kwargs)
        from slotbank.expert_slots import apply_um_pressure, install_expert_slots

        install_expert_slots(model, model_path=self._model_path, um=self.um)
        apply_um_pressure(model, self.um)
        self._warm(model)
        self._wired = self._raise_wired_limit()
        self._model = model
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

    def start_request(self, input_ids: list[int], sampling_params: SamplingParams) -> None:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        from slotbank.expert_slots import apply_um_pressure

        ids = [int(x) for x in input_ids]
        if not ids:
            raise ValueError("empty prompt")
        self._cancelled = False
        self._prompt_ids = ids
        self._sampling_params = sampling_params
        self._generated = []
        apply_um_pressure(self._model, self.um)
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
        step = self._prefill_step_size
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

    def step(self) -> GenerationStep:
        import mlx.core as mx

        if self._cancelled:
            raise Cancelled()
        if self._logprobs is None:
            self._logprobs = _compile_logprobs()
        token = mx.array([self._last_token], dtype=mx.int32)
        logits = self._model(token[None], cache=self._cache)
        self._fed_ids.append(int(self._last_token))
        logits = logits[:, -1, :]
        logprobs = self._logprobs(logits)
        sampled = self._sampler(logprobs)
        token_id = _as_host_token(sampled)
        self._generated.append(token_id)
        self._last_token = token_id
        matched = self._match_stop()
        ignore_eos = bool(self._sampling_params and self._sampling_params.ignore_eos)
        hit_eos = (not ignore_eos) and token_id in self._eos_token_ids
        max_tokens = self._sampling_params.max_tokens if self._sampling_params else 0
        hit_length = len(self._generated) >= max_tokens
        finished = hit_eos or matched is not None or hit_length
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
