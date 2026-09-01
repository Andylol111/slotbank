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


def test_draft_reuse_prefers_live_append_over_stored_prefix():
    from slotbank.runtime import draft_reuse

    ids = list(range(50)) + [99]
    fed = list(range(50))
    stored = list(range(40))
    reuse, feed = draft_reuse(fed, ids, True, stored)
    assert reuse == 50 and feed == [99]


def test_draft_reuse_uses_stored_prefix_when_chat_is_reencoded():
    """OMP sends system+history+new ask, not prompt+raw generated ids."""
    from slotbank.runtime import draft_reuse

    stored = list(range(40))
    ids = stored + [7, 8, 9]
    fed = [1, 2, 3, 99]
    reuse, feed = draft_reuse(fed, ids, True, stored)
    assert reuse == 40 and feed == [7, 8, 9]
    # too short to snapshot
    assert draft_reuse([], [1, 2, 3, 4], False, [1, 2, 3]) == (0, [1, 2, 3, 4])


def test_prefix_cache_short_followup_hits_system_head():
    """Turn 1 snapshot of prompt[:-1] includes the gen-prompt token; turn 2 does not."""
    from slotbank.runtime import PrefixCache, draft_reuse

    pc = PrefixCache(max_bytes=1 << 20)
    turn1 = list(range(200)) + [90, 91, 92, 93]  # multi-token generation prompt
    turn2 = list(range(200)) + [7, 8, 9]  # assistant wrap, no gen prompt yet
    pc._entries = [
        (turn1[:-1], "full", 1),  # includes gen-prompt tokens 90..92
        (list(range(128)), "head", 1),
    ]
    hit = pc.find(turn2)
    assert hit is not None and hit[0] == list(range(128))
    reuse, feed = draft_reuse([1, 2, 9], turn2, True, hit[0])
    assert reuse == 128
    assert feed == turn2[128:]


def test_prefix_cache_finds_longest_exact_prefix():
    from slotbank.runtime import PrefixCache

    pc = PrefixCache(max_bytes=1 << 20)
    short = list(range(40))
    long = list(range(80))
    pc._entries = [(short, "s", 1), (long, "l", 1)]
    assert pc.find(long + [9])[0] == long
    assert pc.find(short + [1])[0] == short
    assert pc.find(list(range(10))) is None


def test_prefix_cache_evicts_shortest_first():
    from slotbank.runtime import PrefixCache

    pc = PrefixCache(max_entries=2, max_bytes=1 << 20)
    pc._entries = [
        (list(range(40)), "a", 10),
        (list(range(80)), "b", 10),
        (list(range(120)), "c", 10),
    ]
    pc._evict()
    kept = [len(e[0]) for e in pc._entries]
    assert kept == [80, 120]


def test_prefix_cache_put_skips_oversize_without_mlx():
    from slotbank.runtime import PrefixCache

    pc = PrefixCache(max_entries=2, max_bytes=1 << 20)
    assert PrefixCache.MAX_SNAP == 2048
    assert pc.put(list(range(3000)), object()) == 0
    assert pc._entries == []


def test_snap_points_keep_large_heads(monkeypatch):
    rt = _rt(monkeypatch, SLOTBANK_PREFIX_CACHE="1")
    rt._prompt_ids = list(range(8000))
    pts = rt._snap_points(0, 7999)
    # 256/512/1024 crumbs used to split the first 2k into extra 27B forwards.
    assert pts == {2048}
    packed_head = rt._snap_points(0, 3000)
    assert packed_head == {2048}
    assert 128 not in packed_head
    assert 256 not in packed_head
    assert 512 not in packed_head
    # Short OMP "hi": full prefix_n includes the generation-prompt token,
    # which the next turn does not start with. 128 sits inside the system head.
    short = rt._snap_points(0, 249)
    assert short == {128}
    # A short prompt that already fits in one tile still snaps at 128, not 256.
    mid = rt._snap_points(0, 1800)
    assert mid == {128}


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
    rt._prefill_ids = lambda *a, **k: None

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


def test_iter_draft_keeps_live_cache_on_append(monkeypatch):
    pytest.importorskip("mlx.core")
    ar = pytest.importorskip("mlx_vlm.generate.ar")
    vlm_cache = pytest.importorskip("mlx_vlm.models.cache")
    from slotbank.types import SamplingParams

    seen: dict = {}
    prefills: list = []

    def generate_step(ids, *_a, **k):
        seen["ids"] = [int(x) for x in ids.flatten().tolist()]
        seen["cache"] = k.get("prompt_cache")
        seen["draft_block_size"] = k.get("draft_block_size")
        yield 7, None

    def prefill(ids, cache, start, end, *, commit_fed):
        prefills.append((start, end, commit_fed, cache))

    monkeypatch.setattr(ar, "generate_step", generate_step)
    monkeypatch.setattr(vlm_cache, "make_prompt_cache", lambda *_a, **_k: ["new"])

    rt = _rt(monkeypatch, SLOTBANK_DRAFT="/tmp/dflash")
    rt._model = types.SimpleNamespace(language_model=object())
    rt._draft = object()
    rt._prompt_ids = [10, 20, 30, 11, 22, 40, 41]
    rt._generated = []
    rt._sampling_params = SamplingParams(temperature=0.0, max_tokens=8)
    rt._eos_token_ids = {7}
    rt._dflash_cache = ["keep"]
    rt._fed_ids = [10, 20, 30, 11, 22]
    rt._draft_cap = 3
    rt._draft_block = 1
    rt._cancelled = False
    rt._total_generated = 0
    rt._prefill_ids = prefill

    steps = list(rt.iter_steps())
    assert [s.token_id for s in steps] == [7]
    assert seen["cache"] == ["keep"]
    assert seen["ids"] == [41]
    assert seen["draft_block_size"] == 3
    assert prefills == [(5, 6, False, ["keep"])]


def test_iter_draft_restores_prefix_when_omp_reencodes(monkeypatch):
    pytest.importorskip("mlx.core")
    ar = pytest.importorskip("mlx_vlm.generate.ar")
    vlm_cache = pytest.importorskip("mlx_vlm.models.cache")
    from slotbank.runtime import PrefixCache
    from slotbank.types import SamplingParams

    class Cell:
        def __init__(self):
            self.state = "fresh"
            self.meta_state = None

    made = []

    def make_cache(*_a, **_k):
        c = [Cell()]
        made.append(c)
        return c

    seen: dict = {}
    prefills: list = []

    def generate_step(ids, *_a, **k):
        seen["ids"] = [int(x) for x in ids.flatten().tolist()]
        seen["cache"] = k.get("prompt_cache")
        yield 9, None

    def prefill(ids, cache, start, end, *, commit_fed):
        prefills.append((start, end, commit_fed))

    monkeypatch.setattr(ar, "generate_step", generate_step)
    monkeypatch.setattr(vlm_cache, "make_prompt_cache", make_cache)

    stored = list(range(40))
    pc = PrefixCache(max_bytes=1 << 20)
    pc._entries = [(stored, [("snap", None)], 8)]

    rt = _rt(monkeypatch, SLOTBANK_DRAFT="/tmp/dflash", SLOTBANK_PREFIX_CACHE="1")
    rt._prefix = pc
    rt._model = types.SimpleNamespace(language_model=object())
    rt._draft = object()
    rt._prompt_ids = stored + [100, 101, 102]
    rt._generated = []
    rt._sampling_params = SamplingParams(temperature=0.0, max_tokens=4)
    rt._eos_token_ids = {9}
    rt._dflash_cache = ["stale"]
    rt._fed_ids = [1, 2, 3, 99]
    rt._cancelled = False
    rt._total_generated = 0
    rt._prefill_ids = prefill

    steps = list(rt.iter_steps())
    assert [s.token_id for s in steps] == [9]
    assert seen["ids"] == [102]
    assert seen["cache"] is made[0]
    assert seen["cache"][0].state == "snap"
    assert prefills == [(40, 42, False)]
    assert pc.hits == 1
    assert pc.saved_tokens == 40


def _rt(monkeypatch, **env):
    """A Runtime built without loading a model. __init__ only reads args."""
    for k in (
        "SLOTBANK_KV_BITS",
        "SLOTBANK_KV_START",
        "SLOTBANK_PREFIX_CACHE",
        "SLOTBANK_PREFIX_CACHE_MIB",
        "SLOTBANK_DRAFT",
    ):
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


def test_prefix_cache_works_while_quantising(monkeypatch):
    """_copy_state recurses into nested (w, scales, biases), so 8-bit KV
    snapshots no longer alias the live cache."""
    on = _rt(monkeypatch, SLOTBANK_PREFIX_CACHE="1")
    assert on._prefix is not None, "prefix cache should be available unquantised"

    both = _rt(monkeypatch, SLOTBANK_PREFIX_CACHE="1", SLOTBANK_KV_BITS="8")
    assert both._kv_bits == 8
    assert both._prefix is not None, "nested copy makes prefix+quant safe"


def test_copy_state_recurses_into_nested_tuples():
    mx = pytest.importorskip("mlx.core")
    from slotbank.runtime import _copy_state, _state_bytes

    inner = mx.ones((2, 2))
    st = ((inner, inner), (inner, inner))
    copied = _copy_state(st)
    assert copied[0][0] is not inner
    assert copied[0][0].tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert _state_bytes(copied) == 4 * int(inner.nbytes)


def test_prefix_budget_bytes(monkeypatch):
    from slotbank.runtime import _prefix_budget_bytes

    monkeypatch.delenv("SLOTBANK_PREFIX_CACHE_MIB", raising=False)
    assert _prefix_budget_bytes() == 384 << 20
    monkeypatch.setenv("SLOTBANK_PREFIX_CACHE_MIB", "512")
    assert _prefix_budget_bytes() == 512 << 20
    monkeypatch.setenv("SLOTBANK_PREFIX_CACHE_MIB", "junk")
    assert _prefix_budget_bytes() == 384 << 20


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
