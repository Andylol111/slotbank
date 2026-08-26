"""Steal stacked Switch* weights; GEMM on a C-slot pack."""

from __future__ import annotations

import json
import mmap
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from slotbank.offload_cache import read_safetensors_layout

DEFAULT_CAPACITY = 16  # decode LRU = 2× top-8; prefill does not pin
PIN_AFTER = 8  # decode calls before frequency pins freeze
_SWITCH_TYPES = frozenset({"QuantizedSwitchLinear", "SwitchLinear"})
_CONTAINER_TYPES = frozenset({"SwitchGLU", "SwitchMLP"})
_KINDS = ("weight", "scales", "biases", "bias")
_ST_TO_MX = {
    "U8": "uint8", "U16": "uint16", "U32": "uint32", "U64": "uint64",
    "I8": "int8", "I16": "int16", "I32": "int32", "I64": "int64",
    "F16": "float16", "BF16": "bfloat16", "F32": "float32", "BOOL": "bool_",
}
_SLOTTED_CLS: dict[type, type] = {}
_CONTAINER_CLS: dict[type, type] = {}
_CONTAINER_CALL: dict[type, object] = {}


def _slots_mode() -> str:
    raw = os.environ.get("SLOTBANK_SLOTS", "auto").strip().lower()
    return raw if raw in {"ram", "auto", "full"} else "auto"


def _budget_bytes() -> int:
    """SLOTBANK_BUDGET_GIB: cap resident expert bytes, letting tok/s float."""
    raw = os.environ.get("SLOTBANK_BUDGET_GIB", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(float(raw) * (1 << 30)))
    except ValueError:
        return 0


def _capacity_from_model(model, capacity: int | None, um=None) -> int:
    if capacity is not None:
        return int(capacity)
    override = os.environ.get("SLOTBANK_SLOTS_OVERRIDE", "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    args = getattr(model, "args", None)
    if args is None:
        args = getattr(getattr(model, "model", None), "args", None)
    top_k = getattr(args, "num_experts_per_tok", None) if args is not None else None
    n_e = getattr(args, "num_experts", None) if args is not None else None
    floor = DEFAULT_CAPACITY
    if top_k:
        from slotbank.layout import slot_floor

        floor = slot_floor(int(n_e or 2 * int(top_k)), int(top_k))
    profile = getattr(um, "profile", None)
    card = getattr(um, "card", None)
    if profile is None or card is None or not getattr(card, "n_routed_experts", 0):
        return floor
    budget = _budget_bytes()
    if budget:
        from slotbank.layout import capacity_for_budget

        # A hard cap on wired bytes, letting throughput float. This is a dial
        # DOWN, not up: raising C removes reads but makes the survivors more
        # expensive, because the pack evicts the page cache that was serving
        # them. Measured on Qwen3-30B-A3B (E=128, top-8), 512-token context:
        #
        #    C    hit   read/tok   I/O wall   effective BW   tok/s
        #    8   41.1%   0.559 G     2.93 s    12.21 GiB/s   10.26
        #   16   62.3%   0.358 G     3.19 s     7.16 GiB/s    8.70
        #   32   87.0%   0.123 G     1.85 s     4.27 GiB/s    7.93
        #
        # C=8 -> C=16 reads 36% fewer bytes and spends 9% MORE time doing it.
        # 12.21 GiB/s is well above this SSD's ~2.9 GiB/s, so those reads were
        # page-cache hits; by C=32 the bandwidth is converging on real disk.
        # Hit rate is an actively misleading objective here.
        return capacity_for_budget(
            int(card.stored_bytes),
            int(card.n_routed_experts or n_e or floor),
            int(card.top_k or top_k or 8),
            budget,
            expert_param_frac=float(card.expert_param_frac or 0.8),
        )
    from slotbank.layout import MIN_KV_BYTES, slot_capacity

    return slot_capacity(
        int(card.n_routed_experts or n_e or floor),
        int(card.top_k or top_k or 8),
        stored_bytes=int(card.stored_bytes),
        working_set_bytes=int(profile.max_working_set_bytes),
        kv_bytes=MIN_KV_BYTES,
        expert_param_frac=float(card.expert_param_frac or 0.8),
        mode=_slots_mode(),
    )


def _waves_enabled() -> bool:
    """Prefill waves: ~1.3x prefill, but splitting one gather into several
    changes float16 split-K rounding, so output stops being bit-identical to
    stock (1 ULP per layer; greedy tokens matched 48/48 when measured).
    Off by default because bit-identical is the accuracy contract.
    """
    return os.environ.get("SLOTBANK_WAVES", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _profile_path(model_path) -> Path | None:
    """Cache key for the hot-expert profile.

    Keyed on the resolved path, not the spelling. `.models/Qwen` and the
    Hugging Face snapshot it links to are the same checkpoint; hashing the
    string gave them different keys, so loading by the other spelling warmed
    nothing and reported no error.
    """
    if not model_path:
        return None
    import hashlib
    import os

    try:
        resolved = os.path.realpath(str(model_path))
    except OSError:
        resolved = str(model_path)
    tag = hashlib.sha1(resolved.encode()).hexdigest()[:16]
    return Path.home() / ".cache" / "slotbank" / f"hot-{tag}.json"


def _packs_with_store(model):
    modules = getattr(model, "modules", None)
    if modules is None:
        return
    for mod in modules():
        pack = getattr(mod, "_expert_slots", None)
        if pack is None or pack.cache is None or pack._store is None:
            continue
        key = pack._keys.get("weight")
        if key is not None:
            yield pack, key


def save_hot_profile(model, model_path) -> int:
    """Record which experts each layer kept resident, so the next run can warm
    the page cache before decode instead of faulting them one at a time."""
    path = _profile_path(model_path)
    if path is None:
        return 0
    import mlx.core as mx

    prof: dict[str, list] = {}
    if path.exists():
        try:
            prof = json.loads(path.read_text())
        except (ValueError, OSError):
            prof = {}
    n = 0
    for pack, key in _packs_with_store(model):
        mx.eval(pack.cache.id_of_slot)
        live = [int(v) for v in pack.cache.id_of_slot.tolist() if int(v) >= 0]
        if not live:
            continue
        # most-recent-first, capped so the profile cannot grow without bound
        merged = list(dict.fromkeys(live + list(prof.get(key) or [])))
        prof[key] = merged[: max(2 * pack.capacity, 32)]
        n += 1
    if not n:
        return 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prof))
    except OSError:
        return 0
    return n


def warm_from_profile(model, model_path, budget_bytes: int = 4 << 30) -> int:
    """Pull the recorded hot experts into the page cache. Bounded by budget."""
    path = _profile_path(model_path)
    if path is None or not path.exists():
        return 0
    try:
        prof = json.loads(path.read_text())
    except (ValueError, OSError):
        return 0
    read = 0
    for pack, wkey in _packs_with_store(model):
        ids = prof.get(wkey)
        if not ids:
            continue
        for key in pack._keys.values():
            if read >= budget_bytes:
                return read
            read += pack._store.warm(key, ids)
    return read


def install_expert_slots(
    model,
    capacity: int | None = None,
    model_path: str | None = None,
    um=None,
) -> int:
    """Wrap every Switch* linear. No-op (0) on dense."""
    named = getattr(model, "named_modules", None)
    if named is None:
        return 0
    capacity = _capacity_from_model(model, capacity, um=um)
    store = SliceStore.from_model_path(model_path) if model_path else None
    n = 0
    for name, mod in named():
        if type(mod).__name__ not in _SWITCH_TYPES:
            continue
        if getattr(mod, "_expert_slots", None) is not None:
            continue
        keys = store.prefix_keys(name) if store is not None else None
        wrap_switch(
            mod,
            capacity,
            store=store if keys else None,
            keys=keys,
        )
        n += 1
    for _name, mod in named():
        if type(mod).__name__ in _CONTAINER_TYPES or getattr(
            type(mod), "_slotbank_layer_slots", False
        ):
            _patch_container(mod)
    if n:
        from slotbank.offload_cache import request_model_residency

        request_model_residency(model)
    return n


def slot_stats(model) -> dict[str, int | float]:
    wrapped = filled = cap = 0
    decode_calls = decode_hits = decode_misses = prefill_calls = 0
    file_misses = l1_hits = 0
    modules = getattr(model, "modules", None)
    empty = {
        "wrapped": 0,
        "filled_slots": 0,
        "capacity_slots": 0,
        "decode_calls": 0,
        "decode_hits": 0,
        "decode_misses": 0,
        "prefill_calls": 0,
        "file_misses": 0,
        "l1_hits": 0,
        "misses_per_decode": 0.0,
    }
    if modules is None:
        return empty
    seen = []
    for mod in modules():
        pack = getattr(mod, "_expert_slots", None)
        if pack is None:
            continue
        cache = getattr(pack, "cache", None)
        if cache is not None and not any(c is cache for c in seen):
            seen.append(cache)
            pack._pull_stats()
        wrapped += 1
        filled += pack.filled
        cap += pack.capacity
        decode_calls += pack.decode_calls
        decode_hits += pack.decode_hits
        decode_misses += pack.decode_misses
        prefill_calls += pack.prefill_calls
        file_misses += pack.file_misses
        l1_hits += pack.l1_hits
    return {
        "wrapped": wrapped,
        "filled_slots": filled,
        "capacity_slots": cap,
        "decode_calls": decode_calls,
        "decode_hits": decode_hits,
        "decode_misses": decode_misses,
        "prefill_calls": prefill_calls,
        "file_misses": file_misses,
        "l1_hits": l1_hits,
        "misses_per_decode": (
            decode_misses / decode_calls if decode_calls else 0.0
        ),
    }


def _is_decode(indices, capacity: int | None = None) -> bool:
    """Eligible for the slot path if every routed id can be resident at once.

    T=1 always qualifies (checks the token axis, not the batch axis: B
    sequences decoding together are (B, 1, k), still decode).

    Short multi-token passes -- speculative verification of k candidates --
    also qualify when n_ids <= capacity, which bounds the unique expert count
    to the pack size. Without that bound the LRU would evict an expert before
    the gather reads it, so the limit is correctness, not just speed. Larger
    passes still take the prefill path.
    """
    shape = tuple(int(s) for s in indices.shape)
    if not shape:
        return False
    if len(shape) >= 2 and shape[-2] == 1:
        return True
    n = 1
    for d in shape:
        n *= d
    if capacity is not None and n <= int(capacity):
        return True
    return n <= 8 if len(shape) == 1 else False


def wrap_switch(
    mod,
    capacity: int = DEFAULT_CAPACITY,
    store=None,
    keys=None,
    pin_after: int = PIN_AFTER,
):
    pack = ExpertSlotPack.steal(mod, capacity)
    pack.pin_after = int(pin_after)
    if store is not None and keys:
        pack._store = store
        pack._keys = keys
        pack.cache.set_store(store, keys)
        pack.drop_stack()
    mod._expert_slots = pack
    cls = type(mod)
    if not getattr(cls, "_slotbank_slotted", False):
        slotted = _SLOTTED_CLS.get(cls)
        if slotted is None:

            def _call(self, x, indices, sorted_indices=False):
                return self._expert_slots(x, indices, sorted_indices)

            slotted = type(cls.__name__, (cls,), {
                "__call__": _call,
                "_slotbank_slotted": True,
            })
            _SLOTTED_CLS[cls] = slotted
        mod.__class__ = slotted
    return pack


def _patch_container(mod) -> None:
    packs = []
    for attr in ("gate_proj", "up_proj", "down_proj", "fc1", "fc2"):
        child = getattr(mod, attr, None)
        pack = getattr(child, "_expert_slots", None) if child is not None else None
        if pack is not None:
            packs.append(pack)
    if not packs:
        return
    lead = packs[0]
    for p in packs[1:]:
        if p.cache is None or p.cache is lead.cache:
            continue
        mapping = p.cache.merge_into(lead.cache)
        if "weight" in mapping:
            p.bank_weight = mapping["weight"]
        if "scales" in mapping:
            p.bank_scales = mapping["scales"]
        if "biases" in mapping:
            p.bank_biases = mapping["biases"]
        if "bias" in mapping:
            p.bank_bias = mapping["bias"]
        p.cache = lead.cache
        p._adopt_cache_packs()
    mod._layer_slots = packs
    cls = type(mod)
    if getattr(cls, "_slotbank_layer_slots", False):
        return
    orig = _CONTAINER_CALL.get(cls)
    if orig is None:
        orig = cls.__call__
        _CONTAINER_CALL[cls] = orig
    slotted = _CONTAINER_CLS.get(cls)
    if slotted is None:

        def _call(self, x, indices):
            _ensure_layer(self._layer_slots, indices)
            return orig(self, x, indices)

        slotted = type(cls.__name__, (cls,), {
            "__call__": _call,
            "_slotbank_layer_slots": True,
        })
        _CONTAINER_CLS[cls] = slotted
    mod.__class__ = slotted


def _as_routing(slot_ids, indices):
    """Give slot ids the routing tensor's shape.

    ``ensure_experts`` returns them flat. At batch 1 that is (top_k,), which
    broadcasts against (1, 1, top_k) by luck; at batch B it is (B*top_k,) and
    ``gather_qmm`` rejects it. Reshaping makes the batch axis explicit instead
    of relying on the accident.
    """
    if slot_ids is None:
        return slot_ids
    shape = tuple(int(d) for d in indices.shape)
    n = 1
    for d in shape:
        n *= d
    if int(slot_ids.size) != n:
        return slot_ids
    return slot_ids.reshape(shape)


def _ensure_layer(packs, indices) -> None:
    """One Metal ensure + copy per proj. Prefill does not pin."""
    lead = packs[0]
    if not _is_decode(indices, lead.capacity):
        return
    lead.prepare_decode(indices)
    slot_ids = lead._ready_slot_ids
    for pack in packs:
        pack._adopt_cache_packs()
        pack._ready_slot_ids = slot_ids
        pack._layer_hit = True
        if pack is lead:
            continue
        pack.decode_calls += 1


def _steal_key(mod, key):
    if key in mod:
        val = mod[key]
        del mod[key]
        return val
    return getattr(mod, key, None)


class SliceStore:
    """Per-expert row reads from safetensors. Does not create a stacked mlx bank."""

    def __init__(self, weight_map: dict[str, str]):
        self._map = weight_map
        self._handles: dict[str, object] = {}
        self._layouts: dict[str, dict] = {}
        self._fds: dict[str, object] = {}
        self._rawfds: dict[str, int] = {}

    @classmethod
    def from_model_path(cls, model_path: str | None) -> SliceStore | None:
        if not model_path:
            return None
        root = Path(model_path)
        if not root.exists():
            return None
        index = root / "model.safetensors.index.json"
        if index.exists():
            data = json.loads(index.read_text())
            mapping = {k: str(root / v) for k, v in (data.get("weight_map") or {}).items()}
            return cls(mapping) if mapping else None
        mapping: dict[str, str] = {}
        from safetensors import safe_open

        for path in sorted(root.glob("*.safetensors")):
            with safe_open(str(path), framework="numpy") as handle:
                for key in handle.keys():
                    mapping[key] = str(path)
        return cls(mapping) if mapping else None

    @classmethod
    def from_file(cls, path: str) -> SliceStore:
        from safetensors import safe_open

        mapping: dict[str, str] = {}
        with safe_open(path, framework="numpy") as handle:
            for key in handle.keys():
                mapping[key] = path
        return cls(mapping)

    def prefix_keys(self, module_path: str) -> dict[str, str] | None:
        keys = {}
        prefix = f"{module_path}." if module_path else ""
        for kind in _KINDS:
            name = f"{prefix}{kind}"
            if name in self._map:
                keys[kind] = name
        return keys or None

    def layout(self, tensor_key: str) -> dict | None:
        path = self._map.get(tensor_key)
        if not path:
            return None
        table = self._layouts.get(path)
        if table is None:
            table = read_safetensors_layout(path)
            self._layouts[path] = table
        return table.get(tensor_key)

    def read(self, tensor_key: str, expert: int):
        spec = self.layout(tensor_key)
        mx_dtype = _ST_TO_MX.get(spec["dtype"]) if spec else None
        if mx_dtype is None:
            return self._read_via_safetensors(tensor_key, expert)
        import mlx.core as mx

        off, row_bytes = self._row_span(spec, expert)
        mm = self._mm(spec["path"])
        raw = mx.array(memoryview(mm[off : off + row_bytes]))
        return raw.view(getattr(mx, mx_dtype)).reshape(spec["shape"][1:])

    def warm(self, tensor_key: str, experts, advise: bool = True) -> int:
        """Fault rows into the page cache. Touches one byte per page rather than
        copying the row, so this costs no heap and no MLX allocation.

        The touch is what makes the pages *stick*. Measured 2026-08-25 on a
        0.75 GiB range, two rotations: pages populated by `pread` alone fall out
        of the page cache inside 15-60 s even with the fd open, a mapping held and
        4.5 GiB reclaimable; the same pages touched through the mapping were
        still 92.7%/96.0% resident at 180 s. Residency follows the mapping's
        resident set, not the read.

        ``advise=False`` skips the MADV_WILLNEED pass. Use it when the bytes
        were just read by `pread` and are already in the page cache, so the
        touch is a soft fault and the advise would only add VM bookkeeping.
        """
        spec = self.layout(tensor_key)
        if spec is None:
            return 0
        if advise:
            self.prefetch(tensor_key, experts)
        mm = self._mm(spec["path"])
        total = 0
        for e in experts:
            off, row_bytes = self._row_span(spec, e)
            for o in range(off, off + row_bytes, mmap.PAGESIZE):
                _ = mm[o]
            total += row_bytes
        return total

    def prefetch(self, tensor_key: str, experts) -> None:
        """Ask the kernel to fault these rows in parallel before we block on them.

        Cold mmap faults taken one at a time run ~3x slower than the same bytes
        with readahead requested up front; warm reads stay a plain memcpy.
        """
        spec = self.layout(tensor_key)
        if spec is None:
            return
        mm = self._mm(spec["path"])
        for e in experts:
            off, row_bytes = self._row_span(spec, e)
            start = off - (off % mmap.PAGESIZE)
            try:
                mm.madvise(mmap.MADV_WILLNEED, start, (off + row_bytes) - start)
            except (OSError, ValueError):
                return

    def raw_fd(self, path: str) -> int:
        """Cached read-only fd for pread/preadv. One per shard, never closed."""
        fd = self._rawfds.get(path)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._rawfds[path] = fd
        return fd

    def pread_row(self, tensor_key: str, expert: int):
        """Raw row bytes via pread. Unlike an mmap slice this releases the GIL,
        so a pool of these actually runs on more than one core."""
        spec = self.layout(tensor_key)
        if spec is None:
            return None
        off, row_bytes = self._row_span(spec, expert)
        fd = self.raw_fd(spec["path"])
        out = os.pread(fd, row_bytes, off)
        while len(out) < row_bytes:
            more = os.pread(fd, row_bytes - len(out), off + len(out))
            if not more:
                break
            out += more
        return out

    def row_spec(self, tensor_key: str):
        spec = self.layout(tensor_key)
        dt = _ST_TO_MX.get(spec["dtype"]) if spec else None
        return (dt, tuple(spec["shape"][1:])) if dt else (None, None)

    @staticmethod
    def _row_span(spec: dict, expert: int) -> tuple[int, int]:
        row_bytes = spec["nbytes"] // int(spec["shape"][0])
        return spec["offset"] + int(expert) * row_bytes, row_bytes

    def _mm(self, path: str):
        mm = self._fds.get(path)
        if mm is None:
            fd = os.open(path, os.O_RDONLY)
            try:
                mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            finally:
                os.close(fd)
            self._fds[path] = mm
        return mm

    def _read_via_safetensors(self, tensor_key: str, expert: int):
        path = self._map[tensor_key]
        handle = self._handles.get(path)
        if handle is None:
            from safetensors import safe_open

            handle = safe_open(path, framework="numpy")
            self._handles[path] = handle
        return handle.get_slice(tensor_key)[int(expert)]


class ExpertSlotPack:
    """Per-proj LRU of detached expert slices. Never gather_qmm the stacked bank."""

    def __init__(self):
        self.capacity = DEFAULT_CAPACITY
        self.num_experts = 0
        self.quantized = False
        self.group_size = 64
        self.bits = 4
        self.mode = "affine"
        self._store: SliceStore | None = None
        self._keys: dict[str, str] = {}
        self._src_weight = None
        self._src_scales = None
        self._src_biases = None
        self._src_bias = None
        self._slot_of: list[int] = []
        self._expert_of: list[int] = []
        self._last: list[int] = []
        self._tick = 0
        self._n = 0
        self._slot_w: list = []
        self._slot_s: list = []
        self._slot_qb: list = []
        self._slot_b: list = []
        self._pack_w = None
        self._pack_s = None
        self._pack_qb = None
        self._pack_b = None
        self._dirty = False
        self.dropped_stack = False
        self._slot_map = None
        self._layer_hit = False
        self.decode_calls = 0
        self.decode_hits = 0
        self.decode_misses = 0
        self.prefill_calls = 0
        self.file_misses = 0
        self.l1_hits = 0
        self.use_waves = _waves_enabled()
        self.pin_after = PIN_AFTER
        self._pins_ready = False
        self._pinned: set[int] = set()
        self._freq: list[int] = []
        self.cache = None
        self._ready_slot_ids = None
        self.bank_weight = "weight"
        self.bank_scales = "scales"
        self.bank_biases = "biases"
        self.bank_bias = "bias"

    def _pull_stats(self) -> None:
        cache = self.cache
        if cache is None:
            return
        cache.sync_stats()
        if cache.filled:
            self._n = cache.filled
        self.decode_misses = cache.stat_miss_host
        self.file_misses = cache.stat_miss_host

    @property
    def filled(self) -> int:
        self._pull_stats()
        return self._n

    @classmethod
    def steal(cls, mod, capacity: int = DEFAULT_CAPACITY) -> ExpertSlotPack:
        pack = cls()
        pack._src_weight = _steal_key(mod, "weight")
        pack._src_scales = _steal_key(mod, "scales")
        pack._src_biases = _steal_key(mod, "biases")
        pack._src_bias = _steal_key(mod, "bias")
        if pack._src_weight is None:
            raise ValueError("switch linear has no weight")
        pack.num_experts = int(pack._src_weight.shape[0])
        pack.capacity = min(int(capacity), pack.num_experts)
        pack.quantized = pack._src_scales is not None
        pack.group_size = int(getattr(mod, "group_size", 64) or 64)
        pack.bits = int(getattr(mod, "bits", 4) or 4)
        pack.mode = getattr(mod, "mode", "affine") or "affine"
        pack._slot_of = [-1] * pack.num_experts
        pack._expert_of = [-1] * pack.capacity
        pack._last = [0] * pack.capacity
        pack._slot_w = [None] * pack.capacity
        pack._slot_s = [None] * pack.capacity
        pack._slot_qb = [None] * pack.capacity
        pack._slot_b = [None] * pack.capacity
        pack._freq = [0] * pack.num_experts
        from slotbank.offload_cache import OffloadMoeCache

        cache = OffloadMoeCache(pack.num_experts, pack.capacity)
        cache.add_bank("weight", source=pack._src_weight)
        if pack._src_scales is not None:
            cache.add_bank("scales", source=pack._src_scales)
        if pack._src_biases is not None:
            cache.add_bank("biases", source=pack._src_biases)
        if pack._src_bias is not None:
            cache.add_bank("bias", source=pack._src_bias)
        pack.cache = cache
        pack._adopt_cache_packs()
        return pack

    def drop_stack(self) -> None:
        self._src_weight = None
        self._src_scales = None
        self._src_biases = None
        self._src_bias = None
        self.dropped_stack = True
        if self.cache is not None:
            self.cache.drop_sources()

    def _adopt_cache_packs(self) -> None:
        cache = self.cache
        if cache is None:
            return
        self._pack_w = cache.banks[self.bank_weight].pack
        self._pack_s = cache.banks[self.bank_scales].pack if self.bank_scales in cache.banks else None
        self._pack_qb = cache.banks[self.bank_biases].pack if self.bank_biases in cache.banks else None
        self._pack_b = cache.banks[self.bank_bias].pack if self.bank_bias in cache.banks else None
        self._slot_map = cache.slot_for_id
        self._n = cache.filled

    def _pinned_experts(self):
        return self._pinned

    def prepare_decode(self, indices):
        """Metal ensure + copy_missing. Routing ids stay on GPU."""
        self.decode_calls += 1
        cache = self.cache
        if cache is None:
            return None
        if self._pins_ready:
            cache.set_pinned(self._pinned)
        slot_ids = cache.ensure_experts(indices)
        n = cache.copy_missing(loader=self._detach)
        self._adopt_cache_packs()
        if n == 0:
            self.decode_hits += 1
            self.l1_hits += 1
        elif n > 0:
            self.decode_misses += n
            self.file_misses += n
        self._ready_slot_ids = slot_ids
        self._maybe_refresh_pins()
        return slot_ids

    def __call__(self, x, indices, sorted_indices=False):
        if not _is_decode(indices, self.capacity):
            return self._prefill(x, indices, sorted_indices)
        return self._decode(x, indices)

    def _decode(self, x, indices):
        if self._layer_hit and self._ready_slot_ids is not None:
            si = self._ready_slot_ids
            self._ready_slot_ids = None
            self._layer_hit = False
            return self._gather(x, _as_routing(si, indices))
        n_ids = 1
        for s in indices.shape:
            n_ids *= int(s)
        if n_ids > self.capacity:
            ids = [int(v) for v in indices.reshape(-1).tolist()]
            unique = list(dict.fromkeys(ids))
            return self._temp_gather(x, indices, ids, unique, pin=True)
        si = self.prepare_decode(indices)
        self._ready_slot_ids = None
        return self._gather(x, _as_routing(si, indices))

    def _prefill(self, x, indices, sorted_indices=False):
        self.prefill_calls += 1
        ids = [int(v) for v in indices.reshape(-1).tolist()]
        unique = list(dict.fromkeys(ids))
        # dict.fromkeys keeps first-appearance order, so sorted ids remap to a
        # sorted compact index and the caller's sorted_indices still holds.
        return self._temp_gather(
            x, indices, ids, unique, pin=False, sorted_indices=sorted_indices
        )

    def _publish_map(self) -> None:
        import mlx.core as mx

        self._slot_map = mx.array(self._slot_of, dtype=mx.int32)

    def _prefetch(self, experts) -> None:
        """Batched readahead for a known expert set. Prefill knows all of its
        uniques up front, so it must not take these faults one at a time."""
        if self._store is None:
            return
        for key in self._keys.values():
            self._store.prefetch(key, experts)

    def _stack_kind(self, kind: str, unique):
        """Stack one bank's rows for a set of experts, reading them in parallel.

        Prefill knows every expert it needs up front, so the rows can be pulled
        with threaded `pread` instead of advised and then faulted one at a time.
        `MADV_WILLNEED` measured 63.8% of prefill against 11.7% for the reads it
        was hinting -- the same defect already removed from the decode path.
        Returns None when there is no file-backed store, so the caller falls
        back to the in-memory path.
        """
        import mlx.core as mx

        store = self._store
        key = self._keys.get(kind) if self._keys else None
        if store is None or key is None:
            return None
        spec = store.row_spec(key)
        if spec is None or spec[0] is None:
            return None
        dt, shape = spec
        from slotbank.offload_cache import _parallel_reads, _pool

        workers = _parallel_reads()
        if workers and len(unique) > 1:
            bufs = list(_pool(workers).map(lambda e: store.pread_row(key, e), unique))
        else:
            bufs = [store.pread_row(key, e) for e in unique]
        if any(b is None for b in bufs):
            return None
        return mx.stack([
            mx.array(memoryview(b)).view(getattr(mx, dt)).reshape(shape)
            for b in bufs
        ])

    def _temp_gather(self, x, indices, ids, unique, pin: bool, sorted_indices=False):
        # Prefill: throwaway pack, do not pin the decode LRU.
        import mlx.core as mx

        if self.use_waves and sorted_indices and not pin and len(unique) > self.capacity:
            return self._wave_gather(x, ids, unique)
        w = self._stack_kind("weight", unique)
        threaded = w is not None
        if not threaded:
            # no file-backed store: advise, then take the faults
            self._prefetch(unique)
            w = mx.stack([self._detach("weight", e) for e in unique])
        scales = qbiases = bias = None
        if self.quantized:
            scales = (self._stack_kind("scales", unique) if threaded else None)
            if scales is None:
                scales = mx.stack([self._detach("scales", e) for e in unique])
            if self._has("biases"):
                qbiases = (self._stack_kind("biases", unique) if threaded else None)
                if qbiases is None:
                    qbiases = mx.stack([self._detach("biases", e) for e in unique])
        if self._has("bias"):
            bias = (self._stack_kind("bias", unique) if threaded else None)
            if bias is None:
                bias = mx.stack([self._detach("bias", e) for e in unique])
        mx.eval(*[t for t in (w, scales, qbiases, bias) if t is not None])
        inv = {e: i for i, e in enumerate(unique)}
        si = mx.array([inv[e] for e in ids], dtype=mx.int32).reshape(indices.shape)
        y = self._gather_raw(x, si, w, scales, qbiases, bias, sorted_indices)
        if pin:
            for e in unique:
                self._pin(e)
            self._sync_pack()
            self._publish_map()
        return y

    def _wave_gather(self, x, ids, unique):
        """Prefill in waves of C experts instead of stacking every unique one.

        Sorted routing puts each expert's rows in one contiguous run, so a wave
        is a slice. A gather is a sum over experts, so this is a reordering, not
        a maths change. Each wave is evaluated before the next is built, which
        is what actually caps the peak.
        """
        import mlx.core as mx

        n = len(ids)
        outs = []
        pos = 0
        for i in range(0, len(unique), self.capacity):
            chunk = unique[i : i + self.capacity]
            last = chunk[-1]
            start = pos
            while pos < n and ids[pos] <= last:
                pos += 1
            if start >= pos:
                continue
            self._prefetch(chunk)
            w = mx.stack([self._detach("weight", e) for e in chunk])
            scales = qbiases = bias = None
            if self.quantized:
                scales = mx.stack([self._detach("scales", e) for e in chunk])
                if self._has("biases"):
                    qbiases = mx.stack([self._detach("biases", e) for e in chunk])
            if self._has("bias"):
                bias = mx.stack([self._detach("bias", e) for e in chunk])
            inv = {e: j for j, e in enumerate(chunk)}
            si = mx.array([inv[ids[p]] for p in range(start, pos)], dtype=mx.int32)
            y = self._gather_raw(x[start:pos], si, w, scales, qbiases, bias, True)
            mx.eval(y)
            outs.append(y)
        return mx.concatenate(outs, axis=0) if len(outs) > 1 else outs[0]

    def _has(self, kind: str) -> bool:
        if kind in self._keys:
            return True
        return getattr(self, f"_src_{kind}") is not None

    def _read_kind(self, kind: str, e: int):
        import mlx.core as mx

        key = self._keys.get(kind)
        if key is not None and self._store is not None:
            arr = mx.array(self._store.read(key, e))
            mx.eval(arr)
            return arr
        src = getattr(self, f"_src_{kind}")
        if src is None:
            raise RuntimeError("expert source dropped; cannot fault a cold expert")
        sl = mx.contiguous(src[e])
        mx.eval(sl)
        return sl

    def _detach(self, kind: str, e: int):
        return self._read_kind(kind, e)

    def _maybe_refresh_pins(self) -> None:
        # Once after PIN_AFTER decode calls. Not per token. Does not change C.
        if self._pins_ready or self.decode_calls < self.pin_after:
            return
        n_pin = max(1, self.capacity // 2)
        if self.cache is not None:
            import mlx.core as mx

            mx.eval(self.cache.id_of_slot, self.cache.usage)
            ranked = []
            for i in range(self.cache.cache_size):
                e = int(self.cache.id_of_slot[i].item())
                if e >= 0:
                    ranked.append((int(self.cache.usage[i].item()), e))
            ranked.sort(reverse=True)
            self._pinned = {e for _u, e in ranked[:n_pin]}
            self._pins_ready = True
            return
        ranked = sorted(
            range(self.num_experts),
            key=lambda e: self._freq[e] if e < len(self._freq) else 0,
            reverse=True,
        )
        self._pinned = {e for e in ranked[:n_pin] if e < len(self._freq) and self._freq[e] > 0}
        self._pins_ready = True

    def _victim(self) -> int:
        unpinned_s = None
        unpinned_t = None
        any_s = 0
        any_t = self._last[0]
        for i in range(self._n):
            t = self._last[i]
            if t < any_t:
                any_t = t
                any_s = i
            e = self._expert_of[i]
            if e in self._pinned:
                continue
            if unpinned_t is None or t < unpinned_t:
                unpinned_t = t
                unpinned_s = i
        return unpinned_s if unpinned_s is not None else any_s

    def _pin(self, e: int) -> int:
        if e < len(self._freq):
            self._freq[e] += 1
        self._maybe_refresh_pins()
        s = self._slot_of[e]
        if s >= 0:
            self.l1_hits += 1
            self.decode_hits += 1
            self._tick += 1
            self._last[s] = self._tick
            return s
        self.decode_misses += 1
        self.file_misses += 1
        if self._n < self.capacity:
            s = self._n
            self._n += 1
        else:
            s = self._victim()
            old = self._expert_of[s]
            if old >= 0:
                self._slot_of[old] = -1
        w = self._detach("weight", e)
        s_w = self._detach("scales", e) if self.quantized else None
        qb = self._detach("biases", e) if self.quantized and self._has("biases") else None
        b = self._detach("bias", e) if self._has("bias") else None
        pack_n = 0 if self._pack_w is None else int(self._pack_w.shape[0])
        if self._pack_w is not None and s < pack_n and self._n == pack_n:
            self._pack_w[s] = w
            evals = [self._pack_w]
            if s_w is not None:
                self._pack_s[s] = s_w
                evals.append(self._pack_s)
            if qb is not None:
                self._pack_qb[s] = qb
                evals.append(self._pack_qb)
            if b is not None:
                self._pack_b[s] = b
                evals.append(self._pack_b)
            import mlx.core as mx

            mx.eval(*evals)
            self._dirty = False
        else:
            self._slot_w[s] = w
            self._slot_s[s] = s_w
            self._slot_qb[s] = qb
            self._slot_b[s] = b
            self._dirty = True
        self._slot_of[e] = s
        self._expert_of[s] = e
        self._tick += 1
        self._last[s] = self._tick
        return s

    def _row(self, slots, pack, i):
        if slots[i] is not None:
            return slots[i]
        return pack[i]

    def _sync_pack(self) -> None:
        import mlx.core as mx

        if not self._dirty and self._pack_w is not None:
            return
        n = self._n
        self._pack_w = mx.stack([self._row(self._slot_w, self._pack_w, i) for i in range(n)])
        evals = [self._pack_w]
        if self.quantized:
            self._pack_s = mx.stack([self._row(self._slot_s, self._pack_s, i) for i in range(n)])
            evals.append(self._pack_s)
            if self._has("biases"):
                self._pack_qb = mx.stack([self._row(self._slot_qb, self._pack_qb, i) for i in range(n)])
                evals.append(self._pack_qb)
        if self._has("bias"):
            self._pack_b = mx.stack([self._row(self._slot_b, self._pack_b, i) for i in range(n)])
            evals.append(self._pack_b)
        mx.eval(*evals)
        # Drop per-slot refs so Metal does not hold 2C (individuals + stack).
        for i in range(n):
            self._slot_w[i] = None
            self._slot_s[i] = None
            self._slot_qb[i] = None
            self._slot_b[i] = None
        self._dirty = False

    def _gather(self, x, slot_idx):
        return self._gather_raw(
            x, slot_idx, self._pack_w, self._pack_s, self._pack_qb, self._pack_b
        )

    def _gather_raw(self, x, slot_idx, w, scales, qbiases, bias, sorted_indices=False):
        import mlx.core as mx

        if self.quantized:
            y = mx.gather_qmm(
                x,
                w,
                scales,
                qbiases,
                rhs_indices=slot_idx,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
                sorted_indices=sorted_indices,
            )
        else:
            y = mx.gather_mm(
                x,
                w.swapaxes(-1, -2),
                rhs_indices=slot_idx,
                sorted_indices=sorted_indices,
            )
        if bias is not None:
            y = y + mx.expand_dims(bias[slot_idx], -2)
        return y
