from __future__ import annotations

import pytest

from slotbank.offload_cache import OffloadMoeCache, lru_ensure_cpu


def _require_mlx():
    pytest.importorskip("mlx.core")


def test_lru_ensure_cpu_hit_and_evict():
    got = lru_ensure_cpu(
        ids=[0, 1, 0],
        slot_for_id=[-1] * 8,
        id_of_slot=[-1] * 3,
        usage=[0, 0, 0],
        step=0,
    )
    assert got["slot_ids"] == [0, 1, 0]
    assert got["n_miss"] == 2
    assert got["n_filled"] == 2
    got2 = lru_ensure_cpu(
        ids=[0, 2],
        slot_for_id=got["slot_for_id"],
        id_of_slot=got["id_of_slot"],
        usage=got["usage"],
        step=got["step"],
    )
    assert got2["slot_ids"] == [0, 2]
    assert got2["src_indices"] == [2]
    assert got2["n_filled"] == 3


def test_metal_ensure_matches_cpu():
    _require_mlx()
    import mlx.core as mx

    cache = OffloadMoeCache(num_experts=8, cache_size=3)
    ids = mx.array([0, 1, 0], dtype=mx.int32)
    slot_ids = cache.ensure_experts(ids)
    mx.eval(slot_ids, cache.meta, cache.slot_for_id, cache.id_of_slot)
    ref = lru_ensure_cpu(
        ids=[0, 1, 0],
        slot_for_id=[-1] * 8,
        id_of_slot=[-1] * 3,
        usage=[0, 0, 0],
        step=0,
    )
    assert slot_ids.tolist() == ref["slot_ids"]
    assert int(cache.meta[0].item()) == ref["n_miss"]
    assert cache.slot_for_id.tolist() == ref["slot_for_id"]

    ids2 = mx.array([0, 2], dtype=mx.int32)
    slot2 = cache.ensure_experts(ids2)
    mx.eval(slot2, cache.slot_for_id, cache.id_of_slot)
    ref2 = lru_ensure_cpu(
        ids=[0, 2],
        slot_for_id=ref["slot_for_id"],
        id_of_slot=ref["id_of_slot"],
        usage=ref["usage"],
        step=ref["step"],
    )
    assert slot2.tolist() == ref2["slot_ids"]
    assert cache.id_of_slot.tolist() == ref2["id_of_slot"]


def test_metal_copy_missing_writes_src_rows():
    _require_mlx()
    import mlx.core as mx

    cache = OffloadMoeCache(num_experts=4, cache_size=2)
    src = mx.arange(4 * 8, dtype=mx.int32).reshape(4, 8)
    cache.add_bank("weight", source=src)
    ids = mx.array([3, 1], dtype=mx.int32)
    cache.ensure_experts(ids)
    cache.copy_missing()
    cache.sync_stats()
    assert cache.stat_miss_host == 2
    mx.eval(cache.pack("weight"))
    assert cache.pack("weight")[0].tolist() == src[3].tolist()
    assert cache.pack("weight")[1].tolist() == src[1].tolist()


def test_hot_residency_adds_pack_not_source():
    _require_mlx()
    import mlx.core as mx

    cache = OffloadMoeCache(num_experts=4, cache_size=2)
    src = mx.arange(4 * 32, dtype=mx.float32).reshape(4, 32)
    cache.add_bank("weight", source=src)
    ids = mx.array([0, 1], dtype=mx.int32)
    cache.ensure_experts(ids)
    cache.copy_missing()
    hot = cache.request_residency()
    assert cache.hot_bytes() == int(cache.pack("weight").nbytes)
    assert cache.hot_bytes() < int(src.nbytes)
    assert hot.residency_ok is True or hot.wired_limit >= 0
    if hot.residency_ok:
        assert hot.allocated_size == cache.hot_bytes()


def test_mmap_copy_does_not_item_meta(tmp_path):
    _require_mlx()
    import mlx.core as mx

    from slotbank.expert_slots import SliceStore
    from slotbank.offload_cache import OffloadMoeCache

    src = mx.arange(4 * 8, dtype=mx.uint32).reshape(4, 8)
    path = str(tmp_path / "qsl_mmap_copy.safetensors")
    mx.save_safetensors(path, {"weight": src})
    store = SliceStore.from_file(path)
    cache = OffloadMoeCache(num_experts=4, cache_size=2)
    cache.add_bank("weight", source=src)
    cache.set_store(store, {"weight": "weight"})
    cache.drop_sources()
    assert cache.banks["weight"].mmap is not None
    ids = mx.array([3, 1], dtype=mx.int32)
    cache.ensure_experts(ids)
    orig = type(cache.meta).item
    hits = []

    def _boom(self):
        hits.append(1)
        return orig(self)

    type(cache.meta).item = _boom
    try:
        cache.copy_missing(device=True)
    finally:
        type(cache.meta).item = orig
    assert hits == [], "device copy must not meta.item() the miss count"
    # This guards the parked DeviceCopy path only. The default in-place path
    # deliberately reads meta[0] + the two index vectors on the host.
    mx.eval(cache.pack("weight"))
    assert cache.pack("weight")[0].tolist() == src[3].tolist()
    assert cache.pack("weight")[1].tolist() == src[1].tolist()
    cache.sync_stats()
    assert cache.stat_miss_host == 2
