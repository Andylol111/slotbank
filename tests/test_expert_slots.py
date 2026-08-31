from __future__ import annotations

from types import SimpleNamespace

import pytest

from slotbank.expert_slots import (
    DEFAULT_CAPACITY,
    _capacity_from_model,
    install_expert_slots,
    slot_stats,
    wrap_switch,
)


def test_install_noop_without_switch_modules():
    assert install_expert_slots(SimpleNamespace()) == 0
    assert install_expert_slots(None) == 0
    empty = slot_stats(SimpleNamespace())
    assert empty["wrapped"] == 0
    assert empty["filled_slots"] == 0
    assert empty["decode_misses"] == 0


def test_capacity_from_model_is_two_times_top_k():
    model = SimpleNamespace(args=SimpleNamespace(num_experts_per_tok=8, num_experts=64))
    assert _capacity_from_model(model, None) == 16
    assert _capacity_from_model(model, 32) == 32
    vlm = SimpleNamespace(
        language_model=SimpleNamespace(
            args=SimpleNamespace(num_experts_per_tok=10, num_experts=512)
        )
    )
    assert _capacity_from_model(vlm, None) == 20


def test_capacity_from_um_keeps_olmoe_floor_and_grows_35b():
    from slotbank.layout import GIB, device_profile, model_memory_card

    model = SimpleNamespace(args=SimpleNamespace(num_experts_per_tok=8, num_experts=64))
    um = SimpleNamespace(
        profile=device_profile(24 * GIB),
        card=model_memory_card(
            7_000_000_000, 4.0, kind="moe", n_routed_experts=64, top_k=8
        ),
    )
    assert _capacity_from_model(model, None, um=um) == 16
    um35 = SimpleNamespace(
        profile=device_profile(24 * GIB),
        card=model_memory_card(
            35_000_000_000, 4.0, kind="moe", n_routed_experts=256, top_k=8
        ),
    )
    c = _capacity_from_model(model, None, um=um35)
    assert 16 < c < 256


def test_install_skips_already_wrapped():
    class _Fake:
        def named_modules(self):
            return []

    assert install_expert_slots(_Fake()) == 0


def _require_mlx():
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm.models.switch_layers")


def test_slot_forward_memory_below_full_bank(tmp_path):
    _require_mlx()
    import gc

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    from slotbank.expert_slots import SliceStore

    inn, out, e = 256, 256, 64
    qsl = QuantizedSwitchLinear(inn, out, e, bias=False, group_size=64, bits=4)
    mx.eval(qsl.weight, qsl.scales, qsl.biases)
    path = str(tmp_path / "qsl.safetensors")
    mx.save_safetensors(
        path, {"weight": qsl.weight, "scales": qsl.scales, "biases": qsl.biases}
    )
    del qsl
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()

    loaded = mx.load(path)
    tiny = nn.Module()
    tiny.weight = loaded["weight"]
    tiny.scales = loaded["scales"]
    tiny.biases = loaded["biases"]
    tiny.group_size = 64
    tiny.bits = 4
    tiny.mode = "affine"
    store = SliceStore.from_file(path)
    wrap_switch(tiny, capacity=16, store=store, keys=store.prefix_keys(""))
    del loaded
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()

    x = mx.random.normal((1, 1, 1, inn))
    idx = mx.array([[0, 1, 2, 3]], dtype=mx.int32)
    y = tiny(x, idx)
    mx.eval(y)
    slot_mem = int(mx.get_active_memory() or 0)
    full = mx.load(path)["weight"]
    mx.eval(full)
    bank_mem = int(mx.get_active_memory() or 0)
    parent = bank_mem - slot_mem
    assert slot_mem > 0
    assert parent > 0
    assert slot_mem * 2 < parent, (slot_mem, parent, bank_mem)
    assert tiny._expert_slots.filled == 4
    assert tiny._expert_slots.dropped_stack is True
    assert DEFAULT_CAPACITY == 16


def test_slot_matches_stock_gather():
    _require_mlx()
    import mlx.core as mx
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    inn, out, e = 64, 64, 8
    qsl = QuantizedSwitchLinear(inn, out, e, bias=False, group_size=64, bits=4)
    x = mx.random.normal((2, 1, 1, inn))
    idx = mx.array([[0, 3], [1, 2]], dtype=mx.int32)
    stock = qsl(x, idx)
    mx.eval(stock)
    wrap_switch(qsl, capacity=4)
    slotted = qsl(x, idx)
    mx.eval(slotted)
    assert bool(mx.allclose(stock, slotted, atol=1e-4, rtol=1e-4).item())


def test_overflow_unique_above_capacity_still_runs():
    _require_mlx()
    import mlx.core as mx
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    inn, out, e = 64, 64, 8
    qsl = QuantizedSwitchLinear(inn, out, e, bias=False, group_size=64, bits=4)
    wrap_switch(qsl, capacity=3)
    x = mx.random.normal((1, 1, 1, inn))
    idx = mx.array([[0, 1, 2, 3, 4]], dtype=mx.int32)
    y = qsl(x, idx)
    mx.eval(y)
    assert qsl._expert_slots.filled == 3
    assert y.shape[-1] == out


def test_prefill_does_not_saturate_decode_lru():
    _require_mlx()
    import mlx.core as mx
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    inn, out, e = 64, 64, 8
    qsl = QuantizedSwitchLinear(inn, out, e, bias=False, group_size=64, bits=4)
    wrap_switch(qsl, capacity=4)
    # Prefill-shaped: T=8 tokens, enough unique to fill C if we wrongly pin.
    x_pre = mx.random.normal((8, 1, 1, inn))
    idx_pre = mx.array([[i % e] for i in range(8)], dtype=mx.int32)
    y = qsl(x_pre, idx_pre)
    mx.eval(y)
    pack = qsl._expert_slots
    assert pack.filled == 0, "prefill must not pin the decode LRU"
    assert pack.prefill_calls == 1
    assert pack.decode_misses == 0
    # Decode-shaped: T=1, two experts.
    x_dec = mx.random.normal((1, 1, 1, inn))
    idx_dec = mx.array([[0, 1]], dtype=mx.int32)
    y = qsl(x_dec, idx_dec)
    mx.eval(y)
    assert pack.filled == 2
    assert pack.decode_misses == 2
    y = qsl(x_dec, idx_dec)
    mx.eval(y)
    assert pack.filled == 2
    assert pack.decode_misses == 2



def test_switch_glu_shared_layer_pin():
    _require_mlx()
    import mlx.core as mx
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

    glu = SwitchGLU(64, 64, 4, bias=False)
    glu.gate_proj = QuantizedSwitchLinear(64, 64, 4, bias=False, group_size=64, bits=4)
    glu.up_proj = QuantizedSwitchLinear(64, 64, 4, bias=False, group_size=64, bits=4)
    glu.down_proj = QuantizedSwitchLinear(64, 64, 4, bias=False, group_size=64, bits=4)
    assert install_expert_slots(glu, capacity=3) == 3
    x = mx.random.normal((2, 64))
    idx = mx.array([[0, 1], [2, 0]], dtype=mx.int32)
    y = glu(x, idx)
    mx.eval(y)
    assert y.shape[-1] == 64
    assert glu.gate_proj._expert_slots.filled == 0
    y = glu(x[:1], idx[:1])
    mx.eval(y)
    assert glu.gate_proj._expert_slots.filled >= 1


def test_install_wraps_quantized_switch_and_dense_name_misses():
    _require_mlx()
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

    class _Bag(nn.Module):
        def __init__(self):
            super().__init__()
            self.switch_mlp = SwitchGLU(64, 64, 4, bias=False)
            self.switch_mlp.gate_proj = QuantizedSwitchLinear(
                64, 64, 4, bias=False, group_size=64, bits=4
            )
            self.dense = nn.Linear(8, 8)

    bag = _Bag()
    n = install_expert_slots(bag, capacity=2)
    assert n >= 1
    stats = slot_stats(bag)
    assert stats["wrapped"] == n
    assert install_expert_slots(bag, capacity=2) == 0




def test_freq_pin_skips_hot_expert():
    _require_mlx()
    import mlx.core as mx
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    qsl = QuantizedSwitchLinear(64, 64, 8, bias=False, group_size=64, bits=4)
    wrap_switch(qsl, capacity=2, pin_after=1)
    pack = qsl._expert_slots
    x = mx.random.normal((1, 1, 1, 64))
    y = qsl(x, mx.array([[0, 1]], dtype=mx.int32))
    mx.eval(y)
    y = qsl(x, mx.array([[0, 1]], dtype=mx.int32))
    mx.eval(y)
    assert 0 in pack._pinned or 1 in pack._pinned
    y = qsl(x, mx.array([[2, 3]], dtype=mx.int32))
    mx.eval(y)


def test_decode_does_not_tolist_routing_ids():
    _require_mlx()
    import mlx.core as mx
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    qsl = QuantizedSwitchLinear(64, 64, 8, bias=False, group_size=64, bits=4)
    wrap_switch(qsl, capacity=4)
    x = mx.random.normal((1, 1, 1, 64))
    idx = mx.array([[0, 1]], dtype=mx.int32)
    y = qsl(x, idx)
    mx.eval(y)

    # tolist() cannot be patched per instance on an mx.array, so attribute each
    # call to its caller: the copy path may read src_indices / evict_slots,
    # the routing path may not read indices.
    import traceback

    orig = type(idx).tolist
    routing_reads = []

    def _boom(self):
        frame = traceback.extract_stack()[-2]
        if frame.filename.endswith("expert_slots.py"):
            routing_reads.append(f"{frame.lineno}: {frame.line}")
        return orig(self)

    type(idx).tolist = _boom
    try:
        y = qsl(x, idx)
        mx.eval(y)
        y = qsl(x, mx.array([[0, 2]], dtype=mx.int32))
        mx.eval(y)
        y = qsl(x, mx.array([[1, 3]], dtype=mx.int32))
        mx.eval(y)
    finally:
        type(idx).tolist = orig
    assert routing_reads == [], f"decode must remap on Metal, not indices.tolist(): {routing_reads}"


def test_offload_cache_residency_marks_hot_pack():
    _require_mlx()
    import mlx.core as mx
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    from slotbank.offload_cache import request_model_residency

    qsl = QuantizedSwitchLinear(64, 64, 8, bias=False, group_size=64, bits=4)
    wrap_switch(qsl, capacity=4)
    x = mx.random.normal((1, 1, 1, 64))
    y = qsl(x, mx.array([[0, 1]], dtype=mx.int32))
    mx.eval(y)
    hot = request_model_residency(qsl)
    assert hot is not None
    assert hot.allocated_size > 0 or hot.wired_limit >= 0
    assert qsl._expert_slots.cache.hot_bytes() > 0
    assert qsl._expert_slots.cache.last_n_miss >= 0


def test_hot_profile_key_follows_the_symlink(tmp_path):
    """Two spellings of one checkpoint must share a profile.

    Models are commonly reached through a symlink into the Hugging Face cache.
    Keying on the path string gave the link and its target different profiles,
    so warming silently did nothing and reported success.
    """
    from slotbank.expert_slots import _profile_path

    real = tmp_path / "snapshot"
    real.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(real)

    assert _profile_path(str(link)) == _profile_path(str(real))
    assert _profile_path(str(real) + "/") == _profile_path(str(real))
    assert _profile_path(None) is None
    assert _profile_path("") is None


def test_prefill_threaded_rows_match_the_mmap_path(tmp_path):
    """Prefill's parallel reads must produce exactly the mmap path's bytes.

    Prefill computes the prompt's KV, so a wrong row here corrupts everything
    downstream rather than degrading it. `MADV_WILLNEED` was 63.8% of prefill
    against 11.7% for the reads it hinted; replacing it with threaded `pread`
    measured ~2.2x, but only matters if the bytes are identical.
    """
    _require_mlx()
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    from slotbank.expert_slots import SliceStore

    inn, out, e = 128, 128, 32
    qsl = QuantizedSwitchLinear(inn, out, e, bias=False, group_size=64, bits=4)
    mx.eval(qsl.weight, qsl.scales, qsl.biases)
    path = str(tmp_path / "prefill_rows.safetensors")
    mx.save_safetensors(
        path, {"weight": qsl.weight, "scales": qsl.scales, "biases": qsl.biases}
    )
    loaded = mx.load(path)
    tiny = nn.Module()
    tiny.weight, tiny.scales, tiny.biases = (
        loaded["weight"], loaded["scales"], loaded["biases"])
    tiny.group_size, tiny.bits, tiny.mode = 64, 4, "affine"
    store = SliceStore.from_file(path)
    pack = wrap_switch(tiny, capacity=8, store=store, keys=store.prefix_keys(""))

    unique = [7, 0, 31, 12, 3]
    for kind in ("weight", "scales", "biases"):
        threaded = pack._stack_kind(kind, unique)
        assert threaded is not None, f"{kind}: store-backed path should engage"
        serial = mx.stack([pack._detach(kind, i) for i in unique])
        mx.eval(threaded, serial)
        assert threaded.dtype == serial.dtype
        assert threaded.shape == serial.shape
        assert mx.array_equal(threaded, serial), f"{kind} rows differ"

    # no store -> None, so the caller keeps its in-memory fallback
    pack._store = None
    assert pack._stack_kind("weight", unique) is None
