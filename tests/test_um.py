from __future__ import annotations

from slotbank.um import PRESSURE_WARN, parse_vm_stat, snapshot_from_vm_stat


def test_parse_vm_stat():
    text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100.\nPages wired down: 200.\n"
    pages = parse_vm_stat(text)
    assert pages["page_size"] == 16384
    assert pages["Pages free"] == 100


def test_snapshot_sheds_on_warn():
    text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages wired down: 100.\n"
    snap = snapshot_from_vm_stat(text, pressure=PRESSURE_WARN)
    assert snap.should_shed is True
    assert snap.free_bytes == 10 * 16384


def test_warm_budget_scales_with_pressure():
    from types import SimpleNamespace

    from slotbank.runtime import Runtime

    args = SimpleNamespace(model_path="x", leave_free=None, prefill_step_size=2048)

    def um_at(pressure, shed=False):
        snap = SimpleNamespace(pressure=pressure, should_shed=shed)
        return SimpleNamespace(
            profile=SimpleNamespace(max_working_set_bytes=12 << 30),
            snapshot=lambda: snap,
        )

    ceiling = (12 << 30) // 3
    assert Runtime(args, um=um_at(1))._warm_budget() == ceiling
    # should_shed is keyed on free_bytes, which macOS drives to ~0 normally;
    # it must NOT reduce the warm budget or warm start never fires.
    assert Runtime(args, um=um_at(1, shed=True))._warm_budget() == ceiling
    assert Runtime(args, um=um_at(2))._warm_budget() == ceiling // 4
    assert Runtime(args, um=um_at(4))._warm_budget() == 0
    # no um: still bounded, never unbounded
    assert 0 < Runtime(args, um=None)._warm_budget() <= (4 << 30)


def test_prompt_lookup_proposer():
    from slotbank.runtime import propose_from_context

    # proposes what followed the last matching trigram
    assert propose_from_context([1, 2, 3, 4, 5, 1, 2, 3], 4) == [4, 5, 1, 2]
    # no earlier occurrence -> propose nothing rather than guess
    assert propose_from_context([9, 8, 7, 6], 4) == []
    # too short to form a trigram
    assert propose_from_context([1, 2], 4) == []
    # k=0 disables it
    assert propose_from_context([1, 2, 3, 1, 2, 3], 0) == []


def _rt():
    from types import SimpleNamespace

    from slotbank.runtime import Runtime

    rt = Runtime(SimpleNamespace(model_path="m", prefill_step_size=2048))
    rt._model = object()          # stand in for a loaded model
    calls = []
    rt._warm = lambda model: (calls.append(model), setattr(rt, "_warmed", 1))[0]
    return rt, calls


def test_warm_deferred_until_it_pays_back(monkeypatch):
    """The warm pass costs ~6.5 s and returns ~49.5 ms/token: break-even 131.

    Running it at load made every short request pay for a speedup it would
    never amortise, so it is deferred and gated.
    """
    monkeypatch.delenv("SLOTBANK_WARM_MIN_TOKENS", raising=False)

    rt, calls = _rt()
    rt._maybe_warm(32)
    assert calls == [], "a 32-token request must not pay for the warm pass"

    rt._maybe_warm(512)
    assert len(calls) == 1, "a long request should warm"

    rt._maybe_warm(512)
    assert len(calls) == 1, "warming is one-shot"


def test_warm_triggers_on_a_long_session(monkeypatch):
    """max_tokens is an upper bound HTTP callers default to 1024, so cumulative
    output is the second signal: many short requests still deserve warm experts.
    """
    monkeypatch.delenv("SLOTBANK_WARM_MIN_TOKENS", raising=False)
    rt, calls = _rt()
    rt._total_generated = 400
    rt._maybe_warm(16)
    assert len(calls) == 1, "a long-lived session should warm even on a short request"


def test_warm_min_tokens_override(monkeypatch):
    monkeypatch.setenv("SLOTBANK_WARM_MIN_TOKENS", "0")
    rt, calls = _rt()
    rt._maybe_warm(1)
    assert len(calls) == 1, "threshold 0 restores eager warming"

    monkeypatch.setenv("SLOTBANK_WARM_MIN_TOKENS", "nonsense")
    rt2, calls2 = _rt()
    rt2._maybe_warm(1)
    assert calls2 == [], "a bad override must fall back to the default, not crash"


def test_adaptive_prefill_step_caps_chunk_times_context(monkeypatch):
    """The prefill peak scales with chunk x context, not chunk alone.

    A fixed chunk therefore cannot bound it: measured 5.49 GiB peak at 8k
    context and 7.93 GiB at 32k with the same 2048 chunk, and a 64k prefill
    exhausted swap. Capping the product keeps the peak flat.
    """
    monkeypatch.delenv("SLOTBANK_PREFILL_BUDGET", raising=False)
    from slotbank.runtime import _adaptive_step, _prefill_budget

    budget = _prefill_budget()
    for ctx in (1024, 8192, 32768, 131072):
        step = _adaptive_step(2048, ctx)
        assert step * ctx <= budget or step == 64, (ctx, step)
        assert step <= 2048, "must never exceed the configured chunk"
    # short prompts keep the full chunk
    assert _adaptive_step(2048, 1024) == 2048
    # long prompts shrink it
    assert _adaptive_step(2048, 131072) < 2048
    # never below the floor, however long the prompt
    assert _adaptive_step(2048, 10_000_000) == 64
    # degenerate input is safe
    assert _adaptive_step(2048, 0) == 2048


def test_prefill_budget_override(monkeypatch):
    from slotbank.runtime import _adaptive_step

    monkeypatch.setenv("SLOTBANK_PREFILL_BUDGET", str(1024 * 1024))
    assert _adaptive_step(2048, 8192) == 128
    monkeypatch.setenv("SLOTBANK_PREFILL_BUDGET", "nonsense")
    assert _adaptive_step(2048, 8192) == 2048, "bad value falls back to the default"


def test_pyramid_step_keeps_early_tiles_large(monkeypatch):
    """Uniform budget//prefix_n throttles the first chunk of a 32k prefill.

    Attention peak is chunk x (offset + chunk). Early tiles can stay at the
    configured step; later tiles shrink. Same peak, fewer Metal launches.
    """
    monkeypatch.delenv("SLOTBANK_PREFILL_BUDGET", raising=False)
    from slotbank.runtime import _adaptive_step, _prefill_budget, _pyramid_step

    budget = _prefill_budget()
    prefix = 32768
    early = _pyramid_step(2048, 0, prefix)
    late = _pyramid_step(2048, 24576, prefix)
    conservative = _adaptive_step(2048, prefix)
    assert early == 2048, "first tile must not inherit the final-length throttle"
    assert early > conservative
    assert late <= early
    assert late * (24576 + late) <= budget or late == 1
    # 8k cap path stays the measured 2048 chunk
    assert _pyramid_step(2048, 0, 8192) == 2048
    assert _pyramid_step(2048, 8192, 8192) == 1


def _vm(free_mb, inactive_mb=6000, spec_mb=200, purge_mb=100, wired_mb=4000):
    p = 16384
    mb = lambda n: n * 1024 * 1024 // p
    return (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        f"Pages free: {mb(free_mb)}.\n"
        f"Pages speculative: {mb(spec_mb)}.\n"
        f"Pages inactive: {mb(inactive_mb)}.\n"
        f"Pages purgeable: {mb(purge_mb)}.\n"
        f"Pages wired down: {mb(wired_mb)}.\n"
        f"File-backed pages: {mb(2000)}.\n"
        "Pages occupied by compressor: 1000.\n"
        "Pages stored in compressor: 4000.\n"
    )


def test_shed_ignores_low_free_when_memory_is_reclaimable():
    """macOS drives free pages to ~0 by design; that is not pressure.

    Keying shed on free_bytes made it fire at normal pressure on every real
    machine, and shed_if_needed runs after every request -- so the prompt cache
    was being dropped constantly, defeating prefix reuse.
    """
    from slotbank.um import PRESSURE_NORMAL, snapshot_from_vm_stat

    snap = snapshot_from_vm_stat(_vm(free_mb=70), pressure=PRESSURE_NORMAL)
    assert snap.free_bytes < (256 << 20), "the old trigger condition holds"
    assert snap.reclaimable_bytes > (1 << 30)
    assert snap.should_shed is False, "low free with ample reclaimable is normal"


def test_shed_fires_on_real_shortage_and_on_kernel_pressure():
    from slotbank.um import PRESSURE_NORMAL, PRESSURE_WARN, snapshot_from_vm_stat

    tight = snapshot_from_vm_stat(
        _vm(free_mb=50, inactive_mb=200, spec_mb=20, purge_mb=10),
        pressure=PRESSURE_NORMAL,
    )
    assert tight.reclaimable_bytes < (1 << 30)
    assert tight.should_shed is True, "genuinely low reclaimable must shed"

    warned = snapshot_from_vm_stat(_vm(free_mb=8000), pressure=PRESSURE_WARN)
    assert warned.should_shed is True, "the kernel signal always wins"


def test_snapshot_exposes_the_signals_that_predict_throughput():
    """file-backed pages are the expert cache; correlation with tok/s is -0.866."""
    from slotbank.um import snapshot_from_vm_stat

    snap = snapshot_from_vm_stat(_vm(free_mb=100))
    assert snap.file_backed_bytes == 2000 * 1024 * 1024
    assert snap.compressor_stored_bytes == 4000 * 16384


def test_loader_falls_through_unsupported_architecture(monkeypatch):
    """An unsupported model_type must try the next source, not abort.

    mlx-lm 0.31.3 has no deepseek_v4, kimi_k3 or minimax_m3; mlx-vlm does.
    Slotting attaches by class name and works against both, so coverage should
    not be capped by one package's release cadence.
    """
    import sys, types as pytypes

    from slotbank import runtime

    monkeypatch.delenv("SLOTBANK_LOADER", raising=False)
    monkeypatch.delenv("SLOTBANK_DRAFT", raising=False)
    calls = []

    def make(name, fail):
        mod = pytypes.ModuleType(name)
        def load(path, **kw):
            calls.append(name)
            if fail:
                raise ValueError(f"Model type deepseek_v4 not supported.")
            return (f"model-from-{name}", "tok")
        mod.load = load
        return mod

    monkeypatch.setitem(sys.modules, "mlx_lm", make("mlx_lm", fail=True))
    monkeypatch.setitem(sys.modules, "mlx_vlm", make("mlx_vlm", fail=False))

    model, _ = runtime._load_model("some/model", {"lazy": True})
    assert model == "model-from-mlx_vlm"
    assert calls == ["mlx_lm", "mlx_vlm"], "must try mlx-lm first, then fall through"


def test_loader_can_be_pinned_and_reports_every_failure(monkeypatch):
    import sys, types as pytypes

    from slotbank import runtime

    def boom(name):
        mod = pytypes.ModuleType(name)
        def load(path, **kw):
            raise ValueError(f"{name} cannot load this")
        mod.load = load
        return mod

    monkeypatch.setitem(sys.modules, "mlx_lm", boom("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_vlm", boom("mlx_vlm"))

    monkeypatch.delenv("SLOTBANK_DRAFT", raising=False)
    monkeypatch.setenv("SLOTBANK_LOADER", "mlx_vlm")
    assert runtime._model_sources() == ["mlx_vlm"], "pin must select one source"

    monkeypatch.delenv("SLOTBANK_LOADER")
    try:
        runtime._load_model("x", {})
    except ValueError as exc:
        msg = str(exc)
        assert "mlx_lm" in msg and "mlx_vlm" in msg, "must name every attempt"
    else:
        raise AssertionError("should have raised")


def _qwen4_dir(tmp_path):
    d = tmp_path / "qwen4"
    d.mkdir()
    (d / "config.json").write_text('{"model_type": "qwen4_exp"}')
    return str(d)


def test_vision_config_stays_on_mlx_lm_unless_asked(tmp_path, monkeypatch):
    """Text-only is the 24 GB default: mlx-lm drops the vision tower."""
    from slotbank import runtime

    d = tmp_path / "qwen38"
    d.mkdir()
    (d / "config.json").write_text(
        '{"model_type": "qwen3_5", "vision_config": {"depth": 24}}'
    )
    monkeypatch.delenv("SLOTBANK_LOADER", raising=False)
    monkeypatch.delenv("SLOTBANK_VISION", raising=False)
    monkeypatch.delenv("SLOTBANK_DRAFT", raising=False)
    assert runtime._model_sources(str(d)) == ["mlx_lm", "mlx_vlm"]
    monkeypatch.setenv("SLOTBANK_VISION", "1")
    assert runtime._model_sources(str(d)) == ["mlx_vlm", "mlx_lm"]
    monkeypatch.delenv("SLOTBANK_VISION", raising=False)
    monkeypatch.setenv("SLOTBANK_DRAFT", "/tmp/dflash")
    assert runtime._model_sources(str(d)) == ["mlx_vlm"]


def test_qwen4_exp_prefers_mlx_vlm(tmp_path, monkeypatch):
    """qwen4_exp is experimental and only exists in mlx-vlm. Trying mlx-lm
    first wastes a ValueError and a confusing 'not supported' on the path
    that cannot work."""
    import sys, types as pytypes

    from slotbank import runtime

    monkeypatch.delenv("SLOTBANK_LOADER", raising=False)
    monkeypatch.delenv("SLOTBANK_DRAFT", raising=False)
    path = _qwen4_dir(tmp_path)
    assert runtime._model_sources(path) == ["mlx_vlm", "mlx_lm"]

    calls = []

    def make(name):
        mod = pytypes.ModuleType(name)
        def load(p, **kw):
            calls.append((name, kw))
            return (f"model-from-{name}", SimpleTokenizer())
        mod.load = load
        return mod

    monkeypatch.setitem(sys.modules, "mlx_lm", make("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_vlm", make("mlx_vlm"))
    model, tok = runtime._load_model(path, {"lazy": True, "tokenizer_config": {"trust_remote_code": False}})
    assert model == "model-from-mlx_vlm"
    assert calls == [("mlx_vlm", {"lazy": True})]
    assert tok.encode("x") == [1]


class SimpleTokenizer:
    def encode(self, text):
        return [1]
    def decode(self, ids):
        return "x"


class Processor:
    tokenizer = SimpleTokenizer()


def test_vlm_processor_unwraps_to_tokenizer():
    from slotbank.runtime import _as_tokenizer

    tok = SimpleTokenizer()
    assert _as_tokenizer(tok) is tok
    assert _as_tokenizer(Processor()).encode("hi") == [1]


def test_vlm_kwargs_drop_tokenizer_config():
    from slotbank.runtime import _loader_kwargs

    kw = {"lazy": True, "tokenizer_config": {"trust_remote_code": False}}
    assert _loader_kwargs("mlx_lm", kw) == kw
    assert _loader_kwargs("mlx_vlm", kw) == {"lazy": True}


def test_qwen4_exp_missing_vlm_names_the_git_install(tmp_path, monkeypatch):
    import sys, types as pytypes

    from slotbank import runtime

    monkeypatch.delenv("SLOTBANK_LOADER", raising=False)
    path = _qwen4_dir(tmp_path)
    lm = pytypes.ModuleType("mlx_lm")
    def boom(p, **kw):
        raise ValueError("Model type qwen4_exp not supported.")
    lm.load = boom
    monkeypatch.setitem(sys.modules, "mlx_lm", lm)
    monkeypatch.delitem(sys.modules, "mlx_vlm", raising=False)

    orig = __import__

    def fake_import(name, *a, **k):
        if name == "mlx_vlm":
            raise ImportError("No module named mlx_vlm")
        return orig(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", fake_import)
    try:
        runtime._load_model(path, {"lazy": True})
    except ValueError as exc:
        msg = str(exc)
        assert "mlx_vlm: not installed" in msg
        assert "git+https://github.com/Blaizzy/mlx-vlm.git" in msg
    else:
        raise AssertionError("should have raised")
