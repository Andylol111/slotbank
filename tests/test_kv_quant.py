import types

import pytest


def _rt(monkeypatch, **env):
    """A Runtime built without loading a model. __init__ only reads args."""
    for k in ("SLOTBANK_KV_BITS", "SLOTBANK_KV_START", "SLOTBANK_PREFIX_CACHE"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from slotbank.runtime import Runtime

    return Runtime(types.SimpleNamespace(model_path="/nonexistent"))


@pytest.mark.parametrize(
    "raw,want",
    [("", None), ("8", 8), ("4", 4), ("16", None), ("2", None), ("junk", None)],
)
def test_kv_bits_accepts_only_what_mlx_supports(monkeypatch, raw, want):
    """mx.quantize supports 4 and 8 here; anything else must fall back to off
    rather than reaching mlx and raising mid-generation."""
    from slotbank import runtime

    monkeypatch.delenv("SLOTBANK_KV_BITS", raising=False)
    if raw:
        monkeypatch.setenv("SLOTBANK_KV_BITS", raw)
    assert runtime._kv_bits() is want


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
