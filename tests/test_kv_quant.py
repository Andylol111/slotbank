import types

import pytest


def test_dflash_session_ok_refuses_overshoot():
    from slotbank.runtime import dflash_session_ok

    class KV:
        def __init__(self, n):
            self.offset = n

    class Arr:
        offset = None

    assert dflash_session_ok([Arr(), KV(5), KV(5)], 5)
    assert not dflash_session_ok([Arr(), KV(7)], 5)
    assert not dflash_session_ok([KV(5), KV(6)], 5)
    assert dflash_session_ok([Arr()], 5)
    assert dflash_session_ok(None, 5)


def test_draft_feed_is_append_only():
    from slotbank.runtime import draft_feed

    ids = [1, 2, 3, 4, 5]
    assert draft_feed([], ids, False) == (0, ids)
    assert draft_feed([1, 2, 3], ids, True) == (3, [4, 5])
    # shorter shared prefix would need a trim; refuse rather than corrupt GDN
    assert draft_feed([1, 2, 9], ids, True) == (0, ids)
    # equal length is not a suffix — re-prefill
    assert draft_feed(ids, ids, True) == (0, ids)
    assert draft_feed([1, 2, 3, 4], [1, 2], True) == (0, [1, 2])


def test_shed_keeps_dflash_session(monkeypatch):
    rt = _rt(monkeypatch, SLOTBANK_DRAFT="/tmp/dflash")
    rt._draft = object()
    rt._dflash_cache = ["keep"]
    rt._fed_ids = [1, 2, 3]
    rt.um = types.SimpleNamespace(should_shed=lambda: True)
    assert rt.shed_if_needed() is True
    assert rt._dflash_cache == ["keep"] and rt._fed_ids == [1, 2, 3]


def test_iter_draft_records_fed_when_consumer_stops(monkeypatch):
    """Engine._run_job breaks on step.finished and closes the generator.

    Bookkeeping after the generate_step loop never runs on that path, so
    _fed_ids must be committed before each yield or the next turn cold-prefills.
    """
    pytest.importorskip("mlx.core")
    ar = pytest.importorskip("mlx_vlm.generate.ar")
    vlm_cache = pytest.importorskip("mlx_vlm.models.cache")
    from slotbank.runtime import draft_feed
    from slotbank.types import SamplingParams

    def generate_step(*_a, **_k):
        yield 11, None
        yield 22, None
        yield 33, None

    monkeypatch.setattr(ar, "generate_step", generate_step)
    monkeypatch.setattr(vlm_cache, "make_prompt_cache", lambda *_a, **_k: ["new"])

    rt = _rt(monkeypatch, SLOTBANK_DRAFT="/tmp/dflash")
    rt._model = types.SimpleNamespace(language_model=object())
    rt._draft = object()
    rt._prompt_ids = [10, 20, 30]
    rt._generated = []
    rt._sampling_params = SamplingParams(temperature=0.0, max_tokens=8)
    rt._eos_token_ids = {22}
    rt._dflash_cache = ["keep"]
    rt._fed_ids = []
    rt._cancelled = False
    rt._total_generated = 0

    seen = []
    for step in rt.iter_steps():
        seen.append(step.token_id)
        if step.finished:
            break
    assert seen == [11, 22]
    assert rt._fed_ids == [10, 20, 30, 11, 22]
    nxt = rt._fed_ids + [40, 41]
    reuse, feed = draft_feed(rt._fed_ids, nxt, rt._dflash_cache is not None)
    assert reuse == 5 and feed == [40, 41]


def _rt(monkeypatch, **env):
    """A Runtime built without loading a model. __init__ only reads args."""
    for k in ("SLOTBANK_KV_BITS", "SLOTBANK_KV_START", "SLOTBANK_PREFIX_CACHE", "SLOTBANK_DRAFT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from slotbank.runtime import Runtime

    return Runtime(types.SimpleNamespace(model_path="/nonexistent"))


@pytest.mark.parametrize(
    "raw,want",
    [("", None), ("0", None), ("off", None), ("8", 8), ("4", 4), ("16", None), ("2", None), ("junk", None)],
)
def test_kv_bits_accepts_only_what_mlx_supports(monkeypatch, raw, want):
    """mx.quantize supports 4 and 8 here; anything else must fall back to off
    rather than reaching mlx and raising mid-generation."""
    from slotbank import runtime

    monkeypatch.delenv("SLOTBANK_KV_BITS", raising=False)
    monkeypatch.delenv("SLOTBANK_DRAFT", raising=False)
    if raw:
        monkeypatch.setenv("SLOTBANK_KV_BITS", raw)
    assert runtime._kv_bits() is want


def test_dense_tight_auto_kv8(monkeypatch):
    from types import SimpleNamespace

    from slotbank import runtime

    monkeypatch.delenv("SLOTBANK_KV_BITS", raising=False)
    monkeypatch.delenv("SLOTBANK_KV_START", raising=False)
    monkeypatch.delenv("SLOTBANK_DRAFT", raising=False)
    um = SimpleNamespace(
        card=SimpleNamespace(kind="dense", stored_bytes=13 << 30),
        profile=SimpleNamespace(max_working_set_bytes=16 << 30),
    )
    assert runtime._dense_tight(um) is True
    assert runtime._kv_bits(um) == 8
    assert runtime._kv_quant_start(um) == 0
    monkeypatch.setenv("SLOTBANK_KV_BITS", "0")
    assert runtime._kv_bits(um) is None
    monkeypatch.delenv("SLOTBANK_KV_BITS", raising=False)
    monkeypatch.setenv("SLOTBANK_DRAFT", "/tmp/dflash")
    assert runtime._kv_bits(um) is None


def test_strip_unused_drops_vision(monkeypatch):
    from slotbank import runtime

    monkeypatch.delenv("SLOTBANK_VISION", raising=False)

    class M:
        vision_tower = object()
        mtp = object()

    dropped = runtime._strip_unused(M())
    assert "vision_tower" in dropped and "mtp" in dropped


def test_kv_start_defaults_and_overrides(monkeypatch):
    from slotbank import runtime

    monkeypatch.delenv("SLOTBANK_KV_START", raising=False)
    assert runtime._kv_quant_start() == 4096
    monkeypatch.setenv("SLOTBANK_KV_START", "512")
    assert runtime._kv_quant_start() == 512
    monkeypatch.setenv("SLOTBANK_KV_START", "junk")
    assert runtime._kv_quant_start() == 4096


def test_prefix_cache_is_off_while_quantising(monkeypatch):
    """_copy_state copies one level deep. A QuantizedKVCache state is nested --
    (keys, values) each being (w, scales, biases) -- so the inner arrays would
    be stored by reference and then mutated by the live cache, serving a later
    request a prefix that has silently changed underneath it.
    """
    on = _rt(monkeypatch, SLOTBANK_PREFIX_CACHE="1")
    assert on._prefix is not None, "prefix cache should be available unquantised"

    both = _rt(monkeypatch, SLOTBANK_PREFIX_CACHE="1", SLOTBANK_KV_BITS="8")
    assert both._kv_bits == 8
    assert both._prefix is None, "prefix cache must be disabled when quantising"


def test_quantize_kv_is_a_noop_without_a_cache(monkeypatch):
    """_quantize_kv runs on every generated token, including before the first
    prefill has built a cache."""
    rt = _rt(monkeypatch, SLOTBANK_KV_BITS="8")
    rt._cache = None
    rt._quantize_kv()          # must not raise
    rt._cache = []
    rt._quantize_kv()


def test_quantize_kv_converts_past_the_start_only(monkeypatch):
    """The conversion is one-way and threshold-driven: below the start the
    cache must stay exact, past it every layer must be quantised."""
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    rt = _rt(monkeypatch, SLOTBANK_KV_BITS="8", SLOTBANK_KV_START="128")

    def cache_of(n):
        out = []
        for _ in range(3):
            c = KVCache()
            c.update_and_fetch(
                mx.zeros((1, 4, n, 128), mx.float16),
                mx.zeros((1, 4, n, 128), mx.float16),
            )
            out.append(c)
        return out

    rt._cache = cache_of(64)
    rt._quantize_kv()
    assert all(isinstance(c, KVCache) for c in rt._cache)

    rt._cache = cache_of(256)
    rt._quantize_kv()
    assert all(isinstance(c, QuantizedKVCache) for c in rt._cache)
    rt._quantize_kv()          # idempotent
    assert all(isinstance(c, QuantizedKVCache) for c in rt._cache)
