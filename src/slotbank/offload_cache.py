"""Metal LRU remap and miss copy into a compact pack."""

from __future__ import annotations

import ctypes
import json
import mmap
import os
import struct
from ctypes import CFUNCTYPE, POINTER, Structure, c_char_p, c_ulong, c_void_p
from dataclasses import dataclass, field

_ST_DTYPE_NBYTES = {
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "BOOL": 1,
}
_PAGE = 16384


_ENSURE = None
_COPY = None
_OBJC = None


_READ_POOL = None


def _parallel_reads() -> int:
    """SLOTBANK_READ_THREADS: fan the per-layer miss reads across cores.

    A layer misses ~3 experts x 9 banks = ~31 rows of 512 KiB (weights) and
    32 KiB (scales/biases). This SSD serves 512 KiB reads at 1342 MiB/s on one
    thread and 4015 MiB/s at depth 8, and `pread` releases the GIL, so the
    depth is real. Set 0 to fall back to the mmap path.
    """
    try:
        return max(0, int(os.environ.get("SLOTBANK_READ_THREADS", "8")))
    except ValueError:
        return 0


def _retain_reads() -> bool:
    """SLOTBANK_RETAIN: touch pread-populated rows through the file mapping.

    macOS drops `pread`-populated pages fast. Measured 2026-08-25, no model,
    two shard rotations: a 0.75 GiB range read with `pread` was 0% resident
    within 15-60 s with the fd open and 4.5 GiB reclaimable; holding an `mmap`
    over it did not help (0% by 60 s). Only pages *faulted through* the mapping
    stayed -- 92.7% and 96.0% resident at 180 s.

    So the retention step is a soft-fault touch after the read, costing a
    measured 95-148 ms per GiB touched (n=2). Off by default: the touch is
    pure page-cache policy and changes no numerics, but its end-to-end
    throughput effect is not yet established.
    """
    return os.environ.get("SLOTBANK_RETAIN", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _pool(n: int):
    global _READ_POOL
    if _READ_POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _READ_POOL = ThreadPoolExecutor(max_workers=n, thread_name_prefix="sb-read")
    return _READ_POOL


def _preadv_one(job) -> bool:
    fd, dst, off = job
    want = len(dst)
    got = os.preadv(fd, [dst], off)
    while 0 < got < want:
        more = os.preadv(fd, [dst[got:]], off + got)
        if not more:
            break
        got += more
    return got == want


def _mx():
    import mlx.core as mx

    return mx


def _ensure_kernel():
    global _ENSURE
    if _ENSURE is not None:
        return _ENSURE
    mx = _mx()
    _ENSURE = mx.fast.metal_kernel(
        name="sb_lru_ensure",
        input_names=[
            "expert_ids",
            "slot_for_id",
            "id_of_slot",
            "usage",
            "step",
            "pinned",
            "stat_miss",
            "stat_calls",
        ],
        output_names=[
            "slot_ids",
            "new_slot_for_id",
            "new_id_of_slot",
            "new_usage",
            "new_step",
            "src_indices",
            "evict_slots",
            "meta",
            "new_stat_miss",
            "new_stat_calls",
        ],
        source="""
    uint tid = thread_position_in_grid.x;
    if (tid != 0) { return; }
    int n = int(expert_ids_shape[0]);
    int E = int(slot_for_id_shape[0]);
    int C = int(id_of_slot_shape[0]);
    for (int i = 0; i < E; i++) { new_slot_for_id[i] = slot_for_id[i]; }
    for (int i = 0; i < C; i++) {
        new_id_of_slot[i] = id_of_slot[i];
        new_usage[i] = usage[i];
        src_indices[i] = -1;
        evict_slots[i] = -1;
    }
    int n_filled = 0;
    for (int i = 0; i < C; i++) { if (new_id_of_slot[i] >= 0) n_filled++; }
    int step1 = step[0] + 1;
    new_step[0] = step1;
    int n_miss = 0;
    // No O(n^2) dedup: a repeat of e finds new_slot_for_id[e] >= 0 below and
    // skips, so the outcome is identical. n = batch * top_k, so the old scan
    // cost 36x more single-threaded work at batch 6 than at batch 1.
    for (int i = 0; i < n; i++) {
        int e = expert_ids[i];
        int s = (e >= 0 && e < E) ? new_slot_for_id[e] : -1;
        if (s >= 0) {
            new_usage[s] = step1;
            continue;
        }
        if (n_filled < C) { s = n_filled; n_filled++; }
        else {
            int best = 2147483647;
            s = 0;
            for (int k = 0; k < C; k++) {
                int owner = new_id_of_slot[k];
                int pin = (owner >= 0 && owner < E && pinned[owner] != 0);
                if (pin) continue;
                if (new_usage[k] < best) { best = new_usage[k]; s = k; }
            }
            if (best == 2147483647) {
                best = new_usage[0]; s = 0;
                for (int k = 1; k < C; k++) {
                    if (new_usage[k] < best) { best = new_usage[k]; s = k; }
                }
            }
            int old = new_id_of_slot[s];
            if (old >= 0 && old < E) { new_slot_for_id[old] = -1; }
        }
        if (e >= 0 && e < E) { new_slot_for_id[e] = s; }
        new_id_of_slot[s] = e;
        new_usage[s] = step1;
        src_indices[n_miss] = e;
        evict_slots[n_miss] = s;
        n_miss++;
    }
    for (int i = 0; i < n; i++) {
        int e = expert_ids[i];
        slot_ids[i] = (e >= 0 && e < E) ? new_slot_for_id[e] : -1;
    }
    meta[0] = n_miss;
    meta[1] = n_filled;
    new_stat_miss[0] = stat_miss[0] + n_miss;
    new_stat_calls[0] = stat_calls[0] + 1;
""",
    )
    return _ENSURE


def _copy_kernel():
    global _COPY
    if _COPY is not None:
        return _COPY
    mx = _mx()
    _COPY = mx.fast.metal_kernel(
        name="sb_copy_missing",
        input_names=["pack", "src", "src_indices", "evict_slots", "num_indices"],
        output_names=["out"],
        source="""
    uint gid = thread_position_in_grid.x;
    int n_miss = num_indices[0];
    int feat = 1;
    for (uint d = 1; d < src_ndim; d++) { feat *= int(src_shape[d]); }
    int pack_n = int(pack_shape[0]) * feat;
    if (gid < uint(pack_n)) { out[gid] = pack[gid]; }
    int total = n_miss * feat;
    if (n_miss > 0 && gid < uint(total)) {
        int i = gid / feat;
        int j = gid % feat;
        int e = src_indices[i];
        int s = evict_slots[i];
        out[s * feat + j] = src[e * feat + j];
    }
""",
    )
    return _COPY


def lru_ensure_cpu(ids, slot_for_id, id_of_slot, usage, step, pinned=None):
    """Bit-identical host reference for the Metal ensure kernel."""
    ids = [int(x) for x in ids]
    slot_for_id = [int(x) for x in slot_for_id]
    id_of_slot = [int(x) for x in id_of_slot]
    usage = [int(x) for x in usage]
    E = len(slot_for_id)
    C = len(id_of_slot)
    pin = [0] * E
    if pinned is not None:
        pin = [int(x) for x in pinned]
    n_filled = sum(1 for x in id_of_slot if x >= 0)
    step = int(step) + 1
    src_indices = []
    evict_slots = []
    seen: list[int] = []
    for e in ids:
        if e in seen:
            continue
        seen.append(e)
        s = slot_for_id[e] if 0 <= e < E else -1
        if s >= 0:
            usage[s] = step
            continue
        if n_filled < C:
            s = n_filled
            n_filled += 1
        else:
            best = 2147483647
            s = 0
            for k in range(C):
                owner = id_of_slot[k]
                if 0 <= owner < E and pin[owner]:
                    continue
                if usage[k] < best:
                    best = usage[k]
                    s = k
            if best == 2147483647:
                s = min(range(C), key=lambda k: usage[k])
            old = id_of_slot[s]
            if 0 <= old < E:
                slot_for_id[old] = -1
        if 0 <= e < E:
            slot_for_id[e] = s
        id_of_slot[s] = e
        usage[s] = step
        src_indices.append(e)
        evict_slots.append(s)
    slot_ids = [slot_for_id[e] if 0 <= e < E else -1 for e in ids]
    return {
        "slot_ids": slot_ids,
        "slot_for_id": slot_for_id,
        "id_of_slot": id_of_slot,
        "usage": usage,
        "step": step,
        "src_indices": src_indices,
        "evict_slots": evict_slots,
        "n_miss": len(src_indices),
        "n_filled": n_filled,
    }


def _objc():
    global _OBJC
    if _OBJC is not None:
        return _OBJC
    libobjc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.dylib")
    libobjc.sel_registerName.restype = c_void_p
    libobjc.sel_registerName.argtypes = [c_char_p]
    libobjc.objc_getClass.restype = c_void_p
    libobjc.objc_getClass.argtypes = [c_char_p]
    metal = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/Metal.framework/Metal"
    )
    metal.MTLCreateSystemDefaultDevice.restype = c_void_p
    _OBJC = (libobjc, metal)
    return _OBJC


def _sel(libobjc, name: str):
    return libobjc.sel_registerName(name.encode())


def _send(libobjc, restype, obj, selector, argtypes=(), args=()):
    fn = CFUNCTYPE(restype, c_void_p, c_void_p, *argtypes)(
        ("objc_msgSend", libobjc)
    )
    return fn(obj, _sel(libobjc, selector), *args)


class _MTLSize(Structure):
    _fields_ = [("width", c_ulong), ("height", c_ulong), ("depth", c_ulong)]


def read_safetensors_layout(path: str) -> dict[str, dict]:
    """Tensor offset/shape/dtype from a safetensors header. No weight load."""
    with open(path, "rb") as handle:
        n = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(n))
    data0 = 8 + n
    out = {}
    for key, spec in header.items():
        if key == "__metadata__" or not isinstance(spec, dict):
            continue
        lo, hi = spec["data_offsets"]
        out[key] = {
            "path": path,
            "offset": data0 + int(lo),
            "nbytes": int(hi) - int(lo),
            "shape": tuple(int(x) for x in spec["shape"]),
            "dtype": spec["dtype"],
        }
    return out


class _MmapTensor:
    __slots__ = ("path", "offset", "nbytes", "shape", "row_bytes", "mm", "buf", "delta", "keep")

    def __init__(self, path: str, offset: int, nbytes: int, shape: tuple):
        self.path = path
        self.offset = int(offset)
        self.nbytes = int(nbytes)
        self.shape = shape
        e = int(shape[0]) if shape else 1
        self.row_bytes = self.nbytes // e
        self.mm = None
        self.buf = None
        self.delta = 0
        self.keep = None


class DeviceCopy:
    """In-place Metal row copy. Reads n_miss from GPU meta. No host .item()."""

    _src = r"""
#include <metal_stdlib>
using namespace metal;
kernel void sb_copy_bytes(
    device const uchar* src [[buffer(0)]],
    device uchar* pack [[buffer(1)]],
    device const int* src_indices [[buffer(2)]],
    device const int* evict_slots [[buffer(3)]],
    device const int* meta [[buffer(4)]],
    constant uint& row_bytes [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    int n_miss = meta[0];
    if (n_miss <= 0) return;
    uint total = uint(n_miss) * row_bytes;
    if (gid >= total) return;
    int i = gid / row_bytes;
    int j = gid % row_bytes;
    int e = src_indices[i];
    int s = evict_slots[i];
    pack[uint(s) * row_bytes + uint(j)] = src[uint(e) * row_bytes + uint(j)];
}
"""
    _pipe = None
    _dev = None
    _queue = None
    _keep: list = []

    @classmethod
    def _ready(cls) -> bool:
        if cls._pipe is not None:
            return True
        try:
            libobjc, metal = _objc()
        except OSError:
            return False
        dev = metal.MTLCreateSystemDefaultDevice()
        if not dev:
            return False
        ns_cls = libobjc.objc_getClass(b"NSString")
        src = _send(
            libobjc,
            c_void_p,
            ns_cls,
            "stringWithUTF8String:",
            argtypes=(c_char_p,),
            args=(cls._src.encode(),),
        )
        err = c_void_p()
        lib = _send(
            libobjc,
            c_void_p,
            dev,
            "newLibraryWithSource:options:error:",
            argtypes=(c_void_p, c_void_p, POINTER(c_void_p)),
            args=(src, None, ctypes.byref(err)),
        )
        if not lib:
            return False
        name = _send(
            libobjc,
            c_void_p,
            ns_cls,
            "stringWithUTF8String:",
            argtypes=(c_char_p,),
            args=(b"sb_copy_bytes",),
        )
        fn = _send(
            libobjc,
            c_void_p,
            lib,
            "newFunctionWithName:",
            argtypes=(c_void_p,),
            args=(name,),
        )
        perr = c_void_p()
        pipe = _send(
            libobjc,
            c_void_p,
            dev,
            "newComputePipelineStateWithFunction:error:",
            argtypes=(c_void_p, POINTER(c_void_p)),
            args=(fn, ctypes.byref(perr)),
        )
        if not pipe:
            return False
        cls._dev = dev
        cls._pipe = pipe
        cls._queue = _send(libobjc, c_void_p, dev, "newCommandQueue")
        cls._keep = [lib, fn, src, name]
        return True

    @classmethod
    def buffer_from_mx(cls, arr, offset: int = 0):
        if not cls._ready():
            return None, None
        mx = _mx()
        mx.eval(arr)
        mv = memoryview(arr)
        raw = (ctypes.c_char * mv.nbytes).from_buffer(mv)
        addr = ctypes.addressof(raw) + int(offset)
        libobjc, _metal = _objc()
        buf = _send(
            libobjc,
            c_void_p,
            cls._dev,
            "newBufferWithBytesNoCopy:length:options:deallocator:",
            argtypes=(c_void_p, c_ulong, c_ulong, c_void_p),
            args=(addr, mv.nbytes - int(offset), 0, None),
        )
        return buf, (arr, mv, raw)

    @classmethod
    def map_tensor(cls, layout: dict) -> _MmapTensor:
        if not cls._ready():
            raise RuntimeError("Metal device copy unavailable")
        t = _MmapTensor(layout["path"], layout["offset"], layout["nbytes"], layout["shape"])
        aligned = t.offset - (t.offset % _PAGE)
        t.delta = t.offset - aligned
        need = t.delta + t.nbytes
        length = ((need + _PAGE - 1) // _PAGE) * _PAGE
        file_size = os.path.getsize(t.path)
        fd = os.open(t.path, os.O_RDONLY)
        if aligned + length <= file_size:
            t.mm = mmap.mmap(fd, length, offset=aligned, access=mmap.ACCESS_COPY)
            os.close(fd)
        else:
            os.close(fd)
            with open(t.path, "rb") as handle:
                handle.seek(t.offset)
                blob = handle.read(t.nbytes)
            t.mm = mmap.mmap(-1, max(_PAGE, need))
            t.mm[t.delta : t.delta + t.nbytes] = blob
        addr = ctypes.addressof(ctypes.c_char.from_buffer(t.mm))
        libobjc, _metal = _objc()
        t.buf = _send(
            libobjc,
            c_void_p,
            cls._dev,
            "newBufferWithBytesNoCopy:length:options:deallocator:",
            argtypes=(c_void_p, c_ulong, c_ulong, c_void_p),
            args=(addr, length, 0, None),
        )
        t.keep = t.mm
        if not t.buf:
            raise RuntimeError("mmap MTLBuffer failed")
        return t

    @classmethod
    def copy_rows(cls, src_buf, src_delta, pack_arr, src_idx, evict, meta, row_bytes: int, cap: int) -> bool:
        return cls.copy_banks(
            [(src_buf, src_delta, pack_arr, row_bytes)],
            src_idx,
            evict,
            meta,
            cap,
        )

    @classmethod
    def copy_banks(cls, jobs: list, src_idx, evict, meta, cap: int) -> bool:
        """One command buffer, one wait. n_miss is meta[0] on GPU."""
        if not cls._ready() or not jobs:
            return False
        libobjc, _metal = _objc()
        b_idx, keep_i = cls.buffer_from_mx(src_idx)
        b_ev, keep_e = cls.buffer_from_mx(evict)
        b_meta, keep_m = cls.buffer_from_mx(meta)
        if not all((b_idx, b_ev, b_meta)):
            return False
        mx = _mx()
        keeps = [keep_i, keep_e, keep_m]
        q, pipe = cls._queue, cls._pipe
        cb = _send(libobjc, c_void_p, q, "commandBuffer")
        enc = _send(libobjc, c_void_p, cb, "computeCommandEncoder")
        if not enc:
            return False
        n_ok = 0
        for src_buf, src_delta, pack_arr, row_bytes in jobs:
            if not src_buf or pack_arr is None or row_bytes <= 0:
                continue
            b_pack, keep_p = cls.buffer_from_mx(pack_arr)
            rb = mx.array([int(row_bytes)], dtype=mx.uint32)
            b_rb, keep_r = cls.buffer_from_mx(rb)
            if not b_pack or not b_rb:
                continue
            keeps.extend([keep_p, keep_r, rb])
            _send(libobjc, None, enc, "setComputePipelineState:", argtypes=(c_void_p,), args=(pipe,))
            bufs = (src_buf, b_pack, b_idx, b_ev, b_meta, b_rb)
            offs = (int(src_delta), 0, 0, 0, 0, 0)
            for i, (b, off) in enumerate(zip(bufs, offs)):
                _send(
                    libobjc,
                    None,
                    enc,
                    "setBuffer:offset:atIndex:",
                    argtypes=(c_void_p, c_ulong, c_ulong),
                    args=(b, off, i),
                )
            total = max(1, int(cap) * int(row_bytes))
            tg = _MTLSize(64, 1, 1)
            groups = _MTLSize((total + 63) // 64, 1, 1)
            _send(
                libobjc,
                None,
                enc,
                "dispatchThreadgroups:threadsPerThreadgroup:",
                argtypes=(_MTLSize, _MTLSize),
                args=(groups, tg),
            )
            n_ok += 1
        _send(libobjc, None, enc, "endEncoding")
        _send(libobjc, None, cb, "commit")
        _send(libobjc, None, cb, "waitUntilCompleted")
        cls._keep.extend(keeps)
        if len(cls._keep) > 64:
            cls._keep = cls._keep[-32:]
        return n_ok > 0


class HotResidency:
    """MTLResidencySet on pack buffers + mx.set_wired_limit(hot bytes).

    llama.cpp PR 11427: without a residency set, macOS collects GPU memory
    after ~1s. We add only the compact pack (the hot set), not the file bank.
    """

    def __init__(self):
        self.residency_ok = False
        self.wired_limit = 0
        self.allocated_size = 0
        self._arrays: list = []
        self._keep: list = []
        self._rs = None
        self._dev = None

    def attach(self, arrays: list) -> None:
        mx = _mx()
        live = [a for a in arrays if a is not None]
        self._arrays = live
        hot = int(sum(int(a.nbytes) for a in live))
        if hot > 0 and hasattr(mx, "set_wired_limit"):
            info = mx.device_info() if hasattr(mx, "device_info") else {}
            cap = int(info.get("max_recommended_working_set_size") or hot)
            # Leave headroom under the system wired cap.
            limit = min(hot, max(0, cap - (256 << 20)))
            if limit > 0:
                self.wired_limit = int(mx.set_wired_limit(limit) or limit)
        self._bind_residency_set(live, hot)

    def _bind_residency_set(self, arrays: list, hot: int) -> None:
        try:
            libobjc, metal = _objc()
        except OSError:
            return
        dev = metal.MTLCreateSystemDefaultDevice()
        if not dev:
            return
        desc_cls = libobjc.objc_getClass(b"MTLResidencySetDescriptor")
        if not desc_cls:
            return
        desc = _send(libobjc, c_void_p, desc_cls, "new")
        err = c_void_p()
        rs = _send(
            libobjc,
            c_void_p,
            dev,
            "newResidencySetWithDescriptor:error:",
            argtypes=(c_void_p, POINTER(c_void_p)),
            args=(desc, ctypes.byref(err)),
        )
        if not rs:
            return
        keep = []
        added = 0
        for arr in arrays:
            mx = _mx()
            mx.eval(arr)
            try:
                mv = memoryview(arr)
            except TypeError:
                continue
            nbytes = int(mv.nbytes)
            if nbytes <= 0:
                continue
            raw = (ctypes.c_char * nbytes).from_buffer(mv)
            addr = ctypes.addressof(raw)
            if addr % 16384:
                continue
            mbuf = _send(
                libobjc,
                c_void_p,
                dev,
                "newBufferWithBytesNoCopy:length:options:deallocator:",
                argtypes=(c_void_p, c_ulong, c_ulong, c_void_p),
                args=(addr, nbytes, 0, None),
            )
            if not mbuf:
                continue
            _send(
                libobjc,
                None,
                rs,
                "addAllocation:",
                argtypes=(c_void_p,),
                args=(mbuf,),
            )
            keep.append((arr, mv, raw, mbuf))
            added += nbytes
        if not keep:
            return
        _send(libobjc, None, rs, "commit")
        _send(libobjc, None, rs, "requestResidency")
        self._rs = rs
        self._dev = dev
        self._keep = keep
        self.allocated_size = added
        self.residency_ok = True

    def request(self) -> None:
        if self._rs is None:
            return
        try:
            libobjc, _metal = _objc()
            _send(libobjc, None, self._rs, "requestResidency")
        except OSError:
            return


@dataclass
class _Bank:
    name: str
    pack: object
    source: object = None
    store: object = None
    key: str | None = None
    mmap: _MmapTensor | None = None
    row_bytes: int = 0


@dataclass(eq=False)
class OffloadMoeCache:
    """Per-proj (or merged per-layer) slot cache."""

    num_experts: int
    cache_size: int
    banks: dict[str, _Bank] = field(default_factory=dict)
    dropped: bool = False
    residency: HotResidency | None = None
    last_n_miss: int = 0
    filled: int = 0
    _slot_ids: object = None

    def __post_init__(self) -> None:
        mx = _mx()
        e, c = int(self.num_experts), int(self.cache_size)
        self.slot_for_id = mx.full((e,), -1, dtype=mx.int32)
        self.id_of_slot = mx.full((c,), -1, dtype=mx.int32)
        self.usage = mx.zeros((c,), dtype=mx.int32)
        self.step = mx.zeros((1,), dtype=mx.int32)
        self.src_indices = mx.full((c,), -1, dtype=mx.int32)
        self.evict_slots = mx.full((c,), -1, dtype=mx.int32)
        self.meta = mx.zeros((2,), dtype=mx.int32)
        self.pinned = mx.zeros((e,), dtype=mx.int32)
        self.stat_miss = mx.zeros((1,), dtype=mx.int32)
        self.stat_calls = mx.zeros((1,), dtype=mx.int32)
        self.stat_miss_host = 0
        self.stat_calls_host = 0
        self._stats_fresh = False

    def add_bank(self, name: str, source=None, store=None, key: str | None = None):
        mx = _mx()
        if source is None and store is None:
            raise ValueError(f"bank {name!r} needs a source or a store")
        if source is not None:
            row = source.shape[1:]
            pack = mx.zeros((self.cache_size, *row), dtype=source.dtype)
        else:
            sample = store.read(key, 0)
            arr = mx.array(sample)
            pack = mx.zeros((self.cache_size, *arr.shape), dtype=arr.dtype)
        mx.eval(pack)
        row_bytes = int(pack[0].nbytes) if self.cache_size else 0
        self.banks[name] = _Bank(
            name, pack, source, store, key, row_bytes=row_bytes
        )
        return name

    def drop_sources(self) -> None:
        for bank in self.banks.values():
            bank.source = None
        self.dropped = True

    def set_store(self, store, keys: dict[str, str]) -> None:
        layout_fn = getattr(store, "layout", None)
        for name, key in keys.items():
            if name not in self.banks and store is not None:
                self.add_bank(name, store=store, key=key)
            bank = self.banks[name]
            bank.store = store
            bank.key = key
            if layout_fn is None:
                continue
            spec = layout_fn(key)
            if spec is None:
                continue
            try:
                bank.mmap = DeviceCopy.map_tensor(spec)
                bank.row_bytes = bank.mmap.row_bytes
            except (OSError, RuntimeError, ValueError, TypeError):
                bank.mmap = None

    def pack(self, name: str):
        return self.banks[name].pack

    def hot_arrays(self) -> list:
        return [b.pack for b in self.banks.values() if b.pack is not None]

    def hot_bytes(self) -> int:
        return int(sum(int(a.nbytes) for a in self.hot_arrays()))

    def set_pinned(self, experts) -> None:
        mx = _mx()
        mask = [0] * self.num_experts
        for e in experts:
            if 0 <= int(e) < self.num_experts:
                mask[int(e)] = 1
        self.pinned = mx.array(mask, dtype=mx.int32)

    def ensure_experts(self, expert_ids, pinned=None) -> object:
        """Rewrite routing ids → slot ids on Metal. No host .tolist()."""
        mx = _mx()
        if pinned is not None:
            self.set_pinned(pinned)
        flat = expert_ids.reshape((-1,))
        outs = _ensure_kernel()(
            inputs=[
                flat,
                self.slot_for_id,
                self.id_of_slot,
                self.usage,
                self.step,
                self.pinned,
                self.stat_miss,
                self.stat_calls,
            ],
            output_shapes=[
                flat.shape,
                self.slot_for_id.shape,
                self.id_of_slot.shape,
                self.usage.shape,
                (1,),
                (self.cache_size,),
                (self.cache_size,),
                (2,),
                (1,),
                (1,),
            ],
            output_dtypes=[mx.int32] * 10,
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
        )
        (
            slot_ids,
            self.slot_for_id,
            self.id_of_slot,
            self.usage,
            self.step,
            self.src_indices,
            self.evict_slots,
            self.meta,
            self.stat_miss,
            self.stat_calls,
        ) = outs
        self._slot_ids = slot_ids.reshape(expert_ids.shape)
        self._stats_fresh = False
        return self._slot_ids

    def sync_stats(self) -> None:
        """Host read of GPU counters. Call after a request, not per layer."""
        if self._stats_fresh:
            return
        mx = _mx()
        mx.eval(self.meta, self.stat_miss, self.stat_calls)
        self.last_n_miss = int(self.meta[0].item())
        self.filled = int(self.meta[1].item())
        self.stat_miss_host = int(self.stat_miss.item())
        self.stat_calls_host = int(self.stat_calls.item())
        self._stats_fresh = True

    def copy_missing(self, loader=None, device: bool = False) -> int:
        """In-place slice fill. Returns n_miss, or -1 on the parked device path.

        ``pack[s] = row`` stays a lazy MLX write so it fuses into the single
        gather eval. Do not eval the packs here, and do not take the ``device``
        path on decode: its private-queue commit + waitUntilCompleted costs
        ~3.5 ms per layer per token, ~300x the time the bytes deserve.
        """
        if device and self._device_copy():
            if self.residency is not None:
                self.residency.request()
            return -1
        return self._host_copy_counted(loader)

    def _device_copy(self) -> bool:
        """In-place Metal copy from mmap or mx source. No meta.item()."""
        mx = _mx()
        packs = [b.pack for b in self.banks.values() if b.pack is not None]
        mx.eval(self.src_indices, self.evict_slots, self.meta, *packs)
        jobs = []
        keeps = []
        for bank in self.banks.values():
            if bank.pack is None or bank.row_bytes <= 0:
                continue
            if bank.mmap is not None:
                jobs.append((bank.mmap.buf, bank.mmap.delta, bank.pack, bank.row_bytes))
                continue
            if bank.source is None:
                continue
            src_buf, keep = DeviceCopy.buffer_from_mx(bank.source)
            if not src_buf:
                continue
            keeps.append(keep)
            jobs.append((src_buf, 0, bank.pack, bank.row_bytes))
        if not jobs:
            return False
        DeviceCopy._keep.extend(keeps)
        return DeviceCopy.copy_banks(
            jobs, self.src_indices, self.evict_slots, self.meta, self.cache_size
        )

    def _host_copy_counted(self, loader) -> int:
        mx = _mx()
        mx.eval(self.meta, self.src_indices, self.evict_slots)
        # meta is host-readable after the eval above, so read it through the
        # buffer protocol. `self.meta[0].item()` instead builds a fresh lazy
        # slice on the Metal stream and waits on a SECOND command buffer:
        # 224.7 us versus 0.8 us measured, x40 layers per token.
        n = int(memoryview(self.meta)[0])
        self.last_n_miss = n
        if n <= 0:
            return 0
        self._host_copy(n, loader)
        return n

    def _metal_copy(self, n: int) -> None:
        mx = _mx()
        num = self.meta[0:1]
        for bank in self.banks.values():
            if bank.source is None or bank.pack is None:
                continue
            feat = 1
            for d in bank.source.shape[1:]:
                feat *= int(d)
            need = max(int(bank.pack.size), n * feat)
            tg = 64 if need >= 64 else max(1, need)
            grid = ((need + tg - 1) // tg) * tg
            bank.pack = _copy_kernel()(
                inputs=[
                    bank.pack,
                    bank.source,
                    self.src_indices,
                    self.evict_slots,
                    num,
                ],
                output_shapes=[bank.pack.shape],
                output_dtypes=[bank.pack.dtype],
                grid=(grid, 1, 1),
                threadgroup=(tg, 1, 1),
            )[0]
        mx.eval(*[b.pack for b in self.banks.values()])

    def _preadv_into_packs(self, src, evict, banks, workers) -> bool:
        """Read each missing row straight into its pack slot. False if unusable.

        The row bytes on disk are already in the pack's layout, so the whole
        chain of bytes -> mx.array -> view -> reshape -> pack[slot] = row is
        pure overhead: it copies ~218 MiB/token on the host and builds ~1137
        MLX graph nodes to place data that ``preadv`` can deposit directly.
        Measured 1.27x end-to-end over seven paired runs, bit-identical.

        Writing into an evaluated array's buffer is only safe because each
        pack is written once per token and its gather has been evaluated by
        the time we return to it; ``mx.eval`` below settles anything pending.
        """
        mx = _mx()
        packs = [b.pack for b in banks if b.pack is not None]
        if len(packs) != len(banks):
            return False
        mx.eval(*packs)
        jobs = []
        for bank in banks:
            store = bank.store
            spec = store.layout(bank.key)
            if spec is None:
                return False
            view = memoryview(bank.pack).cast("B")
            _, row_bytes = store._row_span(spec, 0)
            # a slot must be exactly one row wide, or the offsets below are wrong
            if row_bytes <= 0 or view.nbytes % row_bytes:
                return False
            fd = store.raw_fd(spec["path"])
            for e, slot in zip(src, evict):
                if (slot + 1) * row_bytes > view.nbytes:
                    return False
                off, nb = store._row_span(spec, e)
                if nb != row_bytes:
                    return False
                jobs.append((fd, view[slot * row_bytes:(slot + 1) * row_bytes], off))
        for got in _pool(workers).map(_preadv_one, jobs):
            if not got:
                return False
        if _retain_reads():
            # The rows are in the page cache right now but will not stay there.
            # Touching them through the shard mapping moves them into that
            # mapping's resident set, which is what actually survives. Soft
            # faults only -- the bytes were just read, so no disk I/O here.
            for bank in banks:
                try:
                    bank.store.warm(bank.key, src, advise=False)
                except (OSError, ValueError):
                    break
        return True

    def _host_copy(self, n: int, loader) -> None:
        mx = _mx()
        src = [int(x) for x in self.src_indices.tolist()[:n]]
        evict = [int(x) for x in self.evict_slots.tolist()[:n]]
        workers = _parallel_reads()
        banks = list(self.banks.values())
        all_stored = bool(banks) and all(
            b.store is not None and b.key is not None for b in banks
        )
        if workers and n >= 2 and all_stored:
            if self._preadv_into_packs(src, evict, banks, workers):
                return
        # mmap path only. MADV_WILLNEED costs ~180 ms/token of per-page VM
        # bookkeeping -- 3x the read it is hinting -- but without it the faults
        # below serialize and decode halves. The pread path above skips both.
        for bank in banks:
            if bank.store is not None and bank.key is not None:
                bank.store.prefetch(bank.key, src)
        for e, s in zip(src, evict):
            for bank in self.banks.values():
                row = None
                if bank.store is not None and bank.key is not None:
                    row = mx.array(bank.store.read(bank.key, e))
                elif loader is not None:
                    kind = bank.name.split("_", 1)[-1]
                    row = loader(kind, e)
                elif bank.source is not None:
                    row = bank.source[e]
                if row is None:
                    continue
                bank.pack[s] = row

    def request_residency(self) -> HotResidency:
        if self.residency is None:
            self.residency = HotResidency()
        self.residency.attach(self.hot_arrays())
        return self.residency

    def merge_into(self, other: OffloadMoeCache) -> dict[str, str]:
        """Move banks into ``other``. Maps stay on ``other`` (one LRU)."""
        mapping = {}
        prefix = f"{len(other.banks)}_"
        for name, bank in self.banks.items():
            new = prefix + name
            other.banks[new] = bank
            mapping[name] = new
        self.banks = {}
        return mapping


def request_model_residency(model) -> HotResidency | None:
    arrays = []
    caches = []
    modules = getattr(model, "modules", None)
    if modules is None:
        return None
    for mod in modules():
        pack = getattr(mod, "_expert_slots", None)
        cache = getattr(pack, "cache", None) if pack is not None else None
        if cache is None:
            continue
        if any(c is cache for c in caches):
            continue
        caches.append(cache)
        arrays.extend(cache.hot_arrays())
    if not arrays:
        return None
    hot = HotResidency()
    hot.attach(arrays)
    for cache in caches:
        cache.residency = hot
    if hasattr(model, "__dict__"):
        model._hot_residency = hot
    return hot
