from __future__ import annotations

import json

import pytest

from slotbank.tps import (
    ADOPTED,
    M4_AIR_24G,
    REJECTED,
    STRATEGIES,
    catalog_sound,
    daily_draft,
    draft_accept_rate,
    get,
    pack_read_ceiling_toks,
    read_attempts,
    register_attempt,
    scale_draft_block,
    seed_local_log,
)


def test_catalog_sound():
    catalog_sound()
    assert daily_draft() == "sidecar-mtp-k3"
    assert get("sidecar-mtp-k3").status == ADOPTED
    assert get("dflash2-k8").status != ADOPTED
    assert get("mtp-plus-dflash").status == REJECTED
    assert get("unquantized-bf16-27b").status == REJECTED
    assert get("sliding-window-kv").status == REJECTED
    assert get("mtplx-engine").status != ADOPTED
    assert get("tree-medusa-eagle").status != ADOPTED
    assert get("harness-temp-1").status == ADOPTED
    assert get("qwen35-4b-as-27b-drafter").status == REJECTED
    assert len(STRATEGIES) >= 10


def test_adopted_routes_keep_27b_text():
    for s in STRATEGIES:
        if s.status != ADOPTED:
            continue
        assert not s.changes_target_weights, s.id
        assert not s.needs_trim_cache, s.id


def test_pack_read_ceiling_is_under_ten():
    ceil = pack_read_ceiling_toks(M4_AIR_24G["weight_bytes_4bit"])
    assert 7.5 < ceil < 8.5
    # Speculative can beat the pack-read ceiling; it cannot 2× the measured MTP
    # without extra accepted tokens per 27B forward, which DFlash@8 already lost.
    assert M4_AIR_24G["mtp_k3_count"] > ceil
    assert M4_AIR_24G["mtp_k3_count"] < 20


def test_scale_draft_block_never_exceeds_trained_k():
    assert scale_draft_block(cap=3, accept_rate=None) == 3
    assert scale_draft_block(cap=3, accept_rate=0.99, current=3) == 3
    assert scale_draft_block(cap=3, accept_rate=0.99, current=2) == 3
    assert scale_draft_block(cap=3, accept_rate=0.10, current=3) == 2
    assert scale_draft_block(cap=3, accept_rate=0.10, current=1) == 1
    assert scale_draft_block(cap=8, accept_rate=0.20, current=8) == 7
    assert scale_draft_block(cap=8, accept_rate=0.90, current=5) == 6
    assert scale_draft_block(cap=1, accept_rate=0.99, current=4) == 1
    assert scale_draft_block(cap=3, accept_rate=1.0, current=3) == 3


def test_draft_accept_rate():
    assert draft_accept_rate(None, None) is None
    assert draft_accept_rate([], [3]) is None
    assert draft_accept_rate([3, 2], [3, 3]) == pytest.approx(5 / 6)
    assert draft_accept_rate([0], [0]) is None


def test_register_and_seed_local_log(tmp_path, monkeypatch):
    log = tmp_path / "tps-attempts.jsonl"
    monkeypatch.setenv("SLOTBANK_TPS_LOG", str(log))
    path = seed_local_log()
    assert path == log
    rows = read_attempts()
    ids = [r["id"] for r in rows]
    assert "sidecar-mtp-k3" in ids
    assert "mtp-plus-dflash" in ids
    mtp = next(r for r in rows if r["id"] == "sidecar-mtp-k3")
    assert mtp["toks"] == pytest.approx(13.47)
    # Second seed must not duplicate.
    seed_local_log()
    assert len(read_attempts()) == len(rows)
    register_attempt(
        "dais-trained-cap",
        outcome="adopted",
        evidence="unit",
        extra={"cap": 3},
    )
    extra = read_attempts()[-1]
    assert extra["id"] == "dais-trained-cap"
    assert extra["extra"]["cap"] == 3
    with pytest.raises(KeyError):
        register_attempt("not-a-strategy", outcome="rejected", evidence="x")
    rec = json.loads(log.read_text().splitlines()[0])
    assert "ts" in rec and rec["id"]


def test_retune_draft_block_shrinks_on_low_accept(monkeypatch):
    from types import SimpleNamespace

    from slotbank.runtime import Runtime

    monkeypatch.delenv("SLOTBANK_DAIS", raising=False)
    rt = Runtime(SimpleNamespace(model_path="x", prefill_step_size=512))
    rt._draft = SimpleNamespace(accept_lens=[1, 1], draft_lens=[3, 3])
    rt._draft_cap = 3
    rt._draft_block = 3
    rt._retune_draft_block()
    assert rt._draft_block == 2
    monkeypatch.setenv("SLOTBANK_DAIS", "0")
    rt._draft_block = 3
    rt._retune_draft_block()
    assert rt._draft_block == 3


def test_draft_report_empty_without_drafter():
    from types import SimpleNamespace

    from slotbank.runtime import Runtime

    rt = Runtime(SimpleNamespace(model_path="x", prefill_step_size=512))
    assert rt.draft_report() == (None, None, None)
    rt._draft = SimpleNamespace(accept_lens=[2, 3], draft_lens=[3, 3])
    rt._draft_kind = "mtp"
    rt._draft_block = 3
    kind, block, rate = rt.draft_report()
    assert kind == "mtp" and block == 3
    assert rate == pytest.approx(5 / 6)
