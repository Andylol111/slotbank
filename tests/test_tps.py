from __future__ import annotations

import json

import pytest

from slotbank.tps import (
    ADOPTED,
    DEFERRED,
    IN_TREE,
    M4_AIR_24G,
    REJECTED,
    STRATEGIES,
    catalog_sound,
    daily_draft,
    draft_accept_rate,
    get,
    pack_read_ceiling_toks,
    prefill_seconds,
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
    assert get("hybrid-kv-dynamic-page").status == REJECTED
    assert get("named-kv-survey").status == REJECTED
    assert get("igpu-pyramid-tiles").status == IN_TREE
    assert get("igpu-pyramid-tiles").needs_trim_cache is False
    assert get("two-stage-harness").status == IN_TREE
    assert get("omp-serve-envelope").status == ADOPTED
    assert get("omp-serve-envelope").needs_trim_cache is False
    assert get("mtplx-engine").status != ADOPTED
    assert get("tree-medusa-eagle").status != ADOPTED
    assert get("harness-temp-1").status == ADOPTED
    assert get("qwen35-4b-as-27b-drafter").status == REJECTED
    assert get("omp-models-yml").status == ADOPTED
    assert get("auto-sidecar-mtp").status == ADOPTED
    assert get("draft-prefix-cache").status == ADOPTED
    assert get("draft-prefix-cache").needs_trim_cache is False
    assert get("omp-session-vs-metal").status == ADOPTED
    assert get("omp-session-vs-metal").needs_trim_cache is False
    assert get("qwen-no-think-prompt").status == ADOPTED
    assert get("qwen-no-think-prompt").needs_trim_cache is False
    assert get("dais-reset-each-request").status == ADOPTED
    assert get("dais-reset-each-request").needs_trim_cache is False
    assert get("skip-vlm-rope-prime").status == ADOPTED
    assert get("skip-vlm-rope-prime").needs_trim_cache is False
    assert get("omp-defer-weight-pin").status == ADOPTED
    assert get("omp-defer-weight-pin").needs_trim_cache is False
    assert get("ttft-is-prefill").status == ADOPTED
    assert get("skip-lm-head-prefill").status == ADOPTED
    assert get("spec-prefill-sparse").status == REJECTED
    assert get("async-prefill-pipeline").status == ADOPTED
    assert get("async-prefill-pipeline").needs_trim_cache is False
    assert get("gdn-chunked-cuda-prefill").status == REJECTED
    assert get("distserve-pd").status == REJECTED
    assert get("vllm-gdn-block-apc").status == DEFERRED
    assert get("gdn-cache-contiguous").status == DEFERRED
    assert get("qwen-chat-prefix-stable").status == ADOPTED
    assert get("qwen-chat-prefix-stable").needs_trim_cache is False
    assert get("metal-qmm-prefill").status == DEFERRED
    assert get("ane-npu-prefill").status == REJECTED
    assert get("warm-prefix-at-load").status != ADOPTED
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


def test_prefill_seconds_matches_measured_819():
    assert prefill_seconds(819) == pytest.approx(17.0, rel=0.02)
    assert prefill_seconds(819, reuse=819) == 0.0
    # Follow-up that hits the 2048 snap and prefills a short suffix.
    assert prefill_seconds(2100, reuse=2048) < 2.0


def test_prefill_forward_skips_logits():
    from slotbank.runtime import prefill_forward

    seen: dict = {}

    def fwd(chunk, cache=None, **k):
        seen["k"] = k
        seen["chunk"] = chunk
        seen["cache"] = cache

    prefill_forward(fwd, "ids", "kv")
    assert seen["k"].get("skip_logits") is True
    assert seen["chunk"] == "ids" and seen["cache"] == "kv"

    seen.clear()

    def old_fwd(chunk, cache=None):
        seen["plain"] = True

    prefill_forward(old_fwd, "ids", "kv")
    assert seen.get("plain") is True


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


def test_arm_draft_block_resets_poisoned_k():
    from types import SimpleNamespace

    from slotbank.runtime import Runtime

    rt = Runtime(SimpleNamespace(model_path="x", prefill_step_size=512))
    rt._draft = object()
    rt._draft_cap = 3
    rt._draft_block = 1
    rt._arm_draft_block()
    assert rt._draft_block == 3
    rt._draft_cap = None
    rt._draft_block = 2
    rt._arm_draft_block()
    assert rt._draft_block == 2


def test_iter_draft_starts_at_cap_and_skips_dead_rope_prime():
    import inspect

    from slotbank.runtime import Runtime

    src = inspect.getsource(Runtime._iter_draft)
    assert "_arm_draft_block" in src
    assert "mlx_vlm.generate.dispatch" not in src
    assert "_retune_draft_block" not in src


def test_pin_is_idempotent_and_covers_draft(monkeypatch):
    from types import SimpleNamespace

    from slotbank import runtime
    from slotbank.runtime import Runtime

    seen: list = []

    monkeypatch.setattr(runtime, "_is_dense", lambda um: True)
    monkeypatch.setattr(runtime, "_pin_dense", lambda m: seen.append(m) or 1)

    rt = Runtime(SimpleNamespace(model_path="x", prefill_step_size=512))
    rt._model = object()
    rt._draft = object()
    rt.pin()
    rt.pin()
    assert seen == [rt._model, rt._draft]


def test_engine_loop_ready_before_pin():
    import inspect

    from slotbank.engine import Engine

    src = inspect.getsource(Engine._loop)
    assert "pin=False" in src
    mapped = src.index("self._ready.set()")
    # skip the failure-path set inside the except
    mapped = src.index("self._ready.set()", mapped + 1)
    assert mapped < src.index("self.runtime.pin()")


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


def test_realize_prefill_prefers_async_until_wait():
    from slotbank.runtime import _realize_prefill

    class Mx:
        def __init__(self):
            self.calls: list[str] = []

        def eval(self, states):
            self.calls.append("eval")

        def async_eval(self, states):
            self.calls.append("async")

    mx = Mx()
    _realize_prefill(mx, ["st"], wait=False)
    assert mx.calls == ["async"]
    mx.calls.clear()
    _realize_prefill(mx, ["st"], wait=True)
    assert mx.calls == ["eval"]
    mx.calls.clear()
    _realize_prefill(mx, None, wait=False)
    assert mx.calls == []

    class OldMx:
        def __init__(self):
            self.calls: list[str] = []

        def eval(self, states):
            self.calls.append("eval")

    old = OldMx()
    _realize_prefill(old, ["st"], wait=False)
    assert old.calls == ["eval"]


def test_prefill_ids_pipelines_tiles():
    import inspect

    from slotbank.runtime import Runtime

    src = inspect.getsource(Runtime._prefill_ids)
    assert "_realize_prefill" in src
    assert "_cut_prefill_tile" in src
    assert src.count("clear_cache") == 1
    loop = src[src.index("while offset") : src.index("if prev is not None")]
    assert "clear_cache" not in loop
    assert "wait=False" in src
    assert "wait=True" in src


def test_start_request_reads_stable_prefix_before_copying_ids():
    import inspect

    from slotbank.runtime import Runtime

    src = inspect.getsource(Runtime.start_request)
    assert "stable_prefix_n" in src
    assert src.index("stable_prefix_n") < src.index("ids = [int(x) for x in input_ids]")
