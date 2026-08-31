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
    # DFlash2 often omits vocab_size; it emits target token ids
    (b / "config.json").write_text(json.dumps({"model_type": "dflash2"}))
    assert check_draft_compatible(str(a), str(b)) is None
    # Same vocab, different width: 4B is not a 27B drafter
    (a / "config.json").write_text(json.dumps({
        "vocab_size": 248320, "hidden_size": 5120,
    }))
    (b / "config.json").write_text(json.dumps({
        "vocab_size": 248320, "hidden_size": 2560, "model_type": "qwen3_5",
    }))
    msg = check_draft_compatible(str(a), str(b))
    assert msg and "hidden_size" in msg
    # Matching MTP sidecar is the same width and an mtp model_type
    (b / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_mtp",
        "vocab_size": 248320,
        "text_config": {"vocab_size": 248320, "hidden_size": 5120},
    }))
    assert check_draft_compatible(str(a), str(b)) is None


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


def test_kv_bytes_per_token_qwen38_hybrid():
    from slotbank.admit import kv_bytes_per_token, max_context_tokens

    cfg = {
        "text_config": {
            "layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
        }
    }
    assert kv_bytes_per_token(cfg) == 16 * 4 * 256 * 2 * 2
    profile = type("P", (), {"max_working_set_bytes": 16 << 30})()
    card = type("C", (), {"stored_bytes": 13 << 30})()
    # 16 - 13 - 1 GiB slop = 2 GiB / 64 KiB ≈ 32768
    assert max_context_tokens(profile, card, cfg) == (2 << 30) // (64 << 10)


def test_draft_viable_refuses_hybrid_and_tight_headroom():
    from slotbank.admit import draft_viable
    from slotbank.layout import MIN_KV_BYTES

    profile = type("P", (), {"max_working_set_bytes": 16 << 30})()
    card = type("C", (), {"stored_bytes": 13 << 30})()
    why = draft_viable(profile, card, "mixed layer_types: full_attention, linear_attention")
    assert why and "UNSAFE" in why
    roomy = type("P", (), {"max_working_set_bytes": 64 << 30})()
    assert draft_viable(roomy, card, None) is None
    tight = type("C", (), {"stored_bytes": 15 << 30})()
    why = draft_viable(profile, tight, None)
    assert why and "headroom" in why
    assert MIN_KV_BYTES > 0
    # mlx-vlm DFlash rolls GDN state back; hybrid is allowed when it fits
    assert draft_viable(
        profile, card, "mixed layer_types: full_attention, linear_attention",
        verify="dflash",
    ) is None
    why = draft_viable(
        profile, tight, "mixed layer_types", draft_bytes=2 << 30, verify="dflash",
    )
    assert why and "headroom" in why


def test_draft_block_from_config_reads_trained_k(tmp_path):
    from slotbank.admit import draft_block_from_config

    dflash = tmp_path / "dflash"
    dflash.mkdir()
    (dflash / "config.json").write_text(
        '{"dflash_config": {"block_size": 8}, "block_size": 4}'
    )
    assert draft_block_from_config(str(dflash)) == 8
    mtp = tmp_path / "mtp"
    mtp.mkdir()
    (mtp / "config.json").write_text('{"model_type": "qwen3_5_mtp", "block_size": 3}')
    assert draft_block_from_config(str(mtp)) == 3
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "config.json").write_text("{}")
    assert draft_block_from_config(str(empty)) == 5
    assert draft_block_from_config(str(tmp_path / "missing")) == 5


def test_refuse_draft_if_needed_uses_dflash_verify(tmp_path, monkeypatch):
    import json

    from slotbank.admit import refuse_draft_if_needed

    tgt, dft = tmp_path / "tgt", tmp_path / "dft"
    tgt.mkdir(); dft.mkdir()
    (tgt / "config.json").write_text(json.dumps({
        "vocab_size": 248320,
        "text_config": {
            "vocab_size": 248320,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        },
    }))
    (dft / "config.json").write_text(json.dumps({"model_type": "dflash2"}))
    profile = type("P", (), {"max_working_set_bytes": 18 << 30})()
    card = type("C", (), {"stored_bytes": 16 << 30})()
    monkeypatch.setenv("SLOTBANK_DRAFT", str(dft))
    refuse_draft_if_needed(str(tgt), profile, card)
    tight = type("C", (), {"stored_bytes": 17 << 30})()
    with pytest.raises(ValueError, match="headroom"):
        refuse_draft_if_needed(str(tgt), profile, tight)
    monkeypatch.delenv("SLOTBANK_DRAFT")
    refuse_draft_if_needed(str(tgt), profile, tight)
    huge = tmp_path / "bf16"
    huge.mkdir()
    (huge / "config.json").write_text(json.dumps({"model_type": "dflash2"}))
    monkeypatch.setenv("SLOTBANK_DRAFT", str(huge))
    monkeypatch.setattr(
        "slotbank.admit.stored_bytes_from_files", lambda p: 3 << 30,
    )
    with pytest.raises(ValueError, match="unquantized"):
        refuse_draft_if_needed(str(tgt), profile, card)


def test_hybrid_detected_from_config_without_loading():
    from slotbank.admit import hybrid_from_config

    # Qwen3.5-class: mixed layer types + full_attention_interval
    assert hybrid_from_config(
        {"text_config": {"layer_types": ["linear_attention"] * 3 + ["full_attention"],
                         "full_attention_interval": 4}}
    ) is not None
    # interval alone is enough
    assert hybrid_from_config({"full_attention_interval": 4}) is not None
    # recurrent config keys
    assert hybrid_from_config({"linear_conv_kernel_dim": 4}) is not None
    # a plain attention MoE (OLMoE-class) is not hybrid
    assert hybrid_from_config({"num_experts": 64, "num_experts_per_tok": 8}) is None
    # uniform layer_types must not trip it
    assert hybrid_from_config({"layer_types": ["full_attention"] * 16}) is None


def test_capacity_for_budget_inverts_resident_bytes():
    """A budget must translate to a capacity that actually fits it."""
    from slotbank.layout import capacity_for_budget, resident_expert_bytes

    stored, e, k = 19 * (1 << 30), 256, 8
    for gib in (2, 3, 4, 6, 8):
        c = capacity_for_budget(stored, e, k, gib << 30)
        got = resident_expert_bytes(stored, c, e)
        assert got <= (gib << 30) or c == 16, (gib, c, got)


def test_capacity_for_budget_clamps():
    """Below the non-expert floor, and above the whole bank, both stay sane."""
    from slotbank.layout import capacity_for_budget, slot_floor

    stored, e, k = 19 * (1 << 30), 256, 8
    assert capacity_for_budget(stored, e, k, 1) == slot_floor(e, k)
    assert capacity_for_budget(stored, e, k, 1 << 40) == e
    assert capacity_for_budget(stored, e, k, 0) == slot_floor(e, k)
    assert capacity_for_budget(0, e, k, 4 << 30) == slot_floor(e, k)


def test_budget_env_drives_capacity(monkeypatch):
    """SLOTBANK_BUDGET_GIB must actually reach the capacity policy.

    The budget branch needs a memory card (stored_bytes), so it sits after the
    um check; production always supplies one via UmManager. This pins that the
    env var is honoured and that clearing it restores the normal policy.
    """
    from types import SimpleNamespace

    from slotbank.expert_slots import _capacity_from_model

    card = SimpleNamespace(n_routed_experts=256, top_k=8,
                           stored_bytes=19 * (1 << 30), expert_param_frac=0.8)
    profile = SimpleNamespace(max_working_set_bytes=16 * (1 << 30))
    um = SimpleNamespace(card=card, profile=profile)
    model = SimpleNamespace(args=SimpleNamespace(num_experts_per_tok=8, num_experts=256))

    monkeypatch.delenv("SLOTBANK_BUDGET_GIB", raising=False)
    default_c = _capacity_from_model(model, None, um=um)

    monkeypatch.setenv("SLOTBANK_BUDGET_GIB", "2")
    tight = _capacity_from_model(model, None, um=um)
    monkeypatch.setenv("SLOTBANK_BUDGET_GIB", "8")
    loose = _capacity_from_model(model, None, um=um)

    assert tight < loose, (tight, loose)
    assert tight <= default_c, "a 2 GiB budget must not exceed the default policy"

    monkeypatch.setenv("SLOTBANK_BUDGET_GIB", "junk")
    assert _capacity_from_model(model, None, um=um) == default_c, \
        "a malformed budget must fall back, not crash"
