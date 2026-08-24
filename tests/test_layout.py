from __future__ import annotations

import pytest

from slotbank.layout import (
    GIB,
    M4_AIR_UNIFIED_BANDWIDTH,
    MIN_KV_BYTES,
    admit,
    decode_toks_ceiling,
    detect_device_profile,
    device_profile,
    model_memory_card,
    parse_byte_size,
    quantized_bytes,
    recommended_leave_free,
    resident_expert_bytes,
    slot_capacity,
    slot_floor,
)


def test_parse_byte_size_g_and_raw():
    assert parse_byte_size("10g") == 10 * GIB
    assert parse_byte_size("10GiB") == 10 * GIB
    assert parse_byte_size("512m") == 512 << 20
    assert parse_byte_size("4096") == 4096


def test_recommended_leave_free_keeps_multitasking():
    assert recommended_leave_free(8 * GIB) == 4 * GIB
    assert recommended_leave_free(16 * GIB) == 6 * GIB
    assert recommended_leave_free(24 * GIB) == 8 * GIB
    assert recommended_leave_free(36 * GIB) == 10 * GIB
    assert recommended_leave_free(48 * GIB) == 12 * GIB
    assert recommended_leave_free(64 * GIB) == 16 * GIB


def test_slot_floor_is_two_times_top_k():
    assert slot_floor(64, 8) == 16
    assert slot_floor(8, 8) == 8
    assert slot_floor(256, 8) == 16


def test_slot_capacity_ram_is_always_floor():
    assert slot_capacity(256, 8, stored_bytes=18 * GIB, working_set_bytes=16 * GIB, mode="ram") == 16


def test_slot_capacity_olmoe_on_24g_stays_floor():
    card = model_memory_card(7_000_000_000, 4.0, kind="moe", n_routed_experts=64, top_k=8)
    p = device_profile(24 * GIB)
    c = slot_capacity(64, 8, stored_bytes=card.stored_bytes, working_set_bytes=p.max_working_set_bytes)
    assert c == 16
    assert card.stored_bytes < 8 * GIB


def test_slot_capacity_35b_on_24g_grows_but_stays_under_budget():
    card = model_memory_card(35_000_000_000, 4.0, kind="moe", n_routed_experts=256, top_k=8)
    p = device_profile(24 * GIB)
    c = slot_capacity(256, 8, stored_bytes=card.stored_bytes, working_set_bytes=p.max_working_set_bytes)
    assert 16 < c < 256
    assert resident_expert_bytes(card.stored_bytes, c, 256) + MIN_KV_BYTES <= p.max_working_set_bytes


def test_slot_capacity_35b_on_64g_is_full_bank():
    card = model_memory_card(35_000_000_000, 4.0, kind="moe", n_routed_experts=256, top_k=8)
    p = device_profile(64 * GIB)
    c = slot_capacity(256, 8, stored_bytes=card.stored_bytes, working_set_bytes=p.max_working_set_bytes)
    assert c == 256


def test_device_profile_max_working_set_is_total_minus_leave_free():
    p = device_profile(24 * GIB, leave_free_bytes=10 * GIB)
    assert p.max_working_set_bytes == 14 * GIB
    assert p.fits(14 * GIB, 0) is True
    assert p.fits(14 * GIB, 1) is False


def test_detect_device_profile_reads_this_machine():
    p = detect_device_profile()
    assert p.total_bytes > 0
    assert p.max_working_set_bytes == p.total_bytes - p.leave_free_bytes


def test_moe_card_active_is_smaller_than_stored():
    dense = model_memory_card(70_000_000_000, 2.0)
    moe = model_memory_card(
        70_000_000_000, 2.0, kind="moe", n_routed_experts=64, top_k=2, expert_param_frac=0.8
    )
    assert moe.stored_bytes == dense.stored_bytes
    assert moe.active_bytes < dense.stored_bytes


def test_admit_24g_accepts_4b_4bit():
    result = admit(device_profile(24 * GIB), model_memory_card(4_000_000_000, 4.0), MIN_KV_BYTES)
    assert result.ok is True


def test_admit_active_path_can_fit_moe_when_stored_cannot():
    profile = device_profile(24 * GIB, leave_free_bytes=10 * GIB)
    moe = model_memory_card(
        70_000_000_000, 2.0, kind="moe", n_routed_experts=64, top_k=2, expert_param_frac=0.8
    )
    assert admit(profile, moe, MIN_KV_BYTES, use_active=False).ok is False
    assert admit(profile, moe, MIN_KV_BYTES, use_active=True).ok is True


def test_decode_toks_ceiling_is_bandwidth_over_bytes():
    raw = quantized_bytes(70_000_000_000, 2.0)
    assert decode_toks_ceiling(M4_AIR_UNIFIED_BANDWIDTH, raw) == pytest.approx(120e9 / raw)


def test_params_from_name_rejects_hashes_and_prefers_total():
    from slotbank.admit import _params_from_name, _params_from_stored

    # a Hugging Face snapshot dir is a commit hash; hex digits + 'b' must not
    # parse as a parameter count
    assert _params_from_name("23511b9407e0d69bebdcb091ef5353a59f464a99") is None
    assert _params_from_name("1e20fd8d42056f870933bf98ca6211024744f7ec") is None
    # MoE names carry active and total; the total is the larger
    assert _params_from_name("OLMoE-1B-7B-0125-4bit") == 7_000_000_000
    assert _params_from_name("Qwen3.5-35B-A3B-4bit") == 35_000_000_000
    assert _params_from_name("Qwen3-30B-A3B-Instruct-2507-4bit") == 30_000_000_000
    assert _params_from_name("no-size-here") is None
    # inverting quantized_bytes round-trips
    from slotbank.layout import quantized_bytes

    n = 7_000_000_000
    assert abs(_params_from_stored(quantized_bytes(n, 4.0), 4.0) - n) < n * 0.01


def test_draft_compatibility_rejects_vocab_mismatch(tmp_path):
    import json

    from slotbank.admit import check_draft_compatible

    a, b = tmp_path / "tgt", tmp_path / "dft"
    a.mkdir(); b.mkdir()
    (a / "config.json").write_text(json.dumps({"vocab_size": 248320}))
    (b / "config.json").write_text(json.dumps({"vocab_size": 151936}))
    msg = check_draft_compatible(str(a), str(b))
    assert msg and "vocab mismatch" in msg
    (b / "config.json").write_text(json.dumps({"vocab_size": 248320}))
    assert check_draft_compatible(str(a), str(b)) is None
    # missing vocab_size must refuse, not guess
    (b / "config.json").write_text(json.dumps({}))
    assert check_draft_compatible(str(a), str(b)) is not None


def test_speculative_check_rejects_untrimmable_cache():
    from slotbank.admit import check_speculative_supported

    class Trimmable:
        def is_trimmable(self): return True

    class Recurrent:
        def is_trimmable(self): return False

    assert check_speculative_supported([Trimmable(), Trimmable()]) is None
    msg = check_speculative_supported([Trimmable(), Recurrent()])
    assert msg and "not trimmable" in msg and "Recurrent" in msg
    assert check_speculative_supported([]) is not None
