from __future__ import annotations

from fastapi.testclient import TestClient

from slotbank.api.app import create_app
from slotbank.types import GenResult


class FakeEngine:
    model_id = "toy"

    def tokenize_chat(self, messages, tools):
        return [1, 2, 3]

    def tokenize_text(self, text):
        return [1, 2]

    def generate(self, ids, sampling, on_token=None):
        if on_token is not None:
            on_token(7, "hi")
        return GenResult(content="hi", prompt_tokens=len(ids), completion_tokens=1)

    def stream(self, ids, sampling):
        yield ("delta", "hi")
        yield ("result", GenResult(content="hi", prompt_tokens=len(ids), completion_tokens=1))


def _client() -> TestClient:
    return TestClient(create_app(FakeEngine(), api_key="k"))


def test_health_skips_auth():
    r = _client().get("/health")
    assert r.status_code == 200
    assert r.json()["model"] == "toy"


def test_models_list_matches_omp_discovery():
    r = _client().get("/v1/models")
    assert r.status_code == 200
    row = r.json()["data"][0]
    assert row["id"] == "toy"
    assert row["owned_by"] == "slotbank"
    assert "anthropic" in row["supported_endpoint_types"]
    assert "openai" in row["supported_endpoint_types"]
    assert row["max_model_len"] == 16384
    assert row["meta"]["n_ctx"] == 16384


def test_llama_cpp_native_discovery_skips_auth(monkeypatch):
    monkeypatch.delenv("SLOTBANK_VISION", raising=False)
    c = _client()
    models = c.get("/models")
    assert models.status_code == 200
    body = models.json()
    assert body["data"][0]["id"] == "toy"
    assert body["data"][0]["meta"]["n_ctx"] == 16384
    assert body["data"][0]["architecture"]["input_modalities"] == ["text"]
    assert c.get("/v1/models").json()["data"][0]["id"] == "toy"
    props = c.get("/props")
    assert props.status_code == 200
    p = props.json()
    assert p["n_ctx"] == 16384
    assert p["default_generation_settings"]["n_ctx"] == 16384
    assert p["default_generation_settings"]["params"]["max_tokens"] == -1
    assert p["default_generation_settings"]["params"]["n_predict"] == -1
    assert p["modalities"]["vision"] is False


def test_health_reports_loading():
    from slotbank.api.app import EngineProxy, LoadingEngine, create_app
    from fastapi.testclient import TestClient

    loading = LoadingEngine("Qwen3.8-27B-4bit")
    client = TestClient(create_app(loading))
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "loading"
    assert r.json()["model"] == "Qwen3.8-27B-4bit"
    # OMP 18 probes GET /models in 250 ms while weights are still loading.
    listed = client.get("/models")
    assert listed.status_code == 200
    assert listed.json()["data"][0]["id"] == "Qwen3.8-27B-4bit"
    assert client.get("/props").status_code == 200
    proxy = EngineProxy(LoadingEngine("a"))
    assert proxy.model_id == "a"
    proxy.replace(FakeEngine())
    assert proxy.model_id == "toy"
    assert not getattr(proxy, "loading", False)


def test_chat_requires_key():
    c = _client()
    assert c.post("/v1/chat/completions", json={"model": "toy", "messages": [{"role": "user", "content": "x"}]}).status_code == 401
    r = c.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer k"},
        json={"model": "toy", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "hi"


def test_claude_messages():
    r = _client().post(
        "/v1/messages",
        headers={"x-api-key": "k"},
        json={"model": "toy", "max_tokens": 16, "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message"
    assert body["content"][0]["text"] == "hi"
    assert body["stop_reason"] == "end_turn"


def test_claude_count_tokens():
    r = _client().post(
        "/v1/messages/count_tokens",
        headers={"x-api-key": "k"},
        json={"model": "toy", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.json()["input_tokens"] == 3


def test_codex_responses():
    r = _client().post(
        "/v1/responses",
        headers={"Authorization": "Bearer k"},
        json={"model": "toy", "input": "hello"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"] == "hi"


def test_codex_rejects_previous_response_id():
    r = _client().post(
        "/v1/responses",
        headers={"Authorization": "Bearer k"},
        json={"model": "toy", "input": "hello", "previous_response_id": "resp_x"},
    )
    assert r.status_code == 400


def test_chat_stream_done():
    r = _client().post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer k"},
        json={"model": "toy", "messages": [{"role": "user", "content": "x"}], "stream": True},
    )
    assert r.status_code == 200
    assert "data: [DONE]" in r.text
    assert "chat.completion.chunk" in r.text


def test_quiet_status_accepts_the_same_call_shape():
    """--quiet must not change the emitter's signature.

    The suppressed emitter is a stub; when its parameter names drifted from the
    live one, every --quiet run crashed with TypeError at the final stats line
    while ordinary runs passed.
    """
    from slotbank.cli import _status

    for quiet in (True, False):
        say = _status(quiet)
        say()
        say("progress")
        say("final", end=True)


def test_cli_reports_a_missing_model_without_a_traceback(capsys):
    """A missing model is an ordinary outcome and should read as one."""
    from slotbank.cli import main

    code = main(["admit", "--model", "/nonexistent/model/path"])
    assert code == 2
    err = capsys.readouterr().err
    assert err.startswith("slotbank: ")
    assert "Traceback" not in err


def test_effort_presets_apply_and_flags_still_win(monkeypatch, request):
    """A preset sets defaults; an explicit flag alongside it must still win.

    Precedence is flag > preset > environment, so a preset is a starting point
    rather than a lock. Verified here because a preset that silently overrode
    an explicit flag would be worse than having no presets at all.
    """
    from types import SimpleNamespace

    from slotbank.cli import EFFORT, _apply_tuning

    # _apply_tuning writes os.environ directly, which monkeypatch cannot undo,
    # so snapshot and restore explicitly or the settings leak into other tests
    import os

    touched = ("SLOTBANK_BUDGET_GIB", "SLOTBANK_WARM", "SLOTBANK_PREFILL_STEP",
               "SLOTBANK_WARM_MIN_TOKENS", "SLOTBANK_READ_THREADS",
               "SLOTBANK_SLOTS_OVERRIDE")
    saved = {k: os.environ.get(k) for k in touched}
    for k in touched:
        os.environ.pop(k, None)
    def restore():
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    request.addfinalizer(restore)

    base = dict(budget_gib=None, slots=None, read_threads=None,
                prefill_step=None, warm_min_tokens=None, no_warm=False)

    _apply_tuning(SimpleNamespace(effort="low", **base))
    assert os.environ["SLOTBANK_BUDGET_GIB"] == EFFORT["low"]["SLOTBANK_BUDGET_GIB"]
    assert os.environ["SLOTBANK_WARM"] == "0"

    # an explicit flag alongside the preset overrides that one value only
    _apply_tuning(SimpleNamespace(effort="low", **{**base, "budget_gib": 9.0}))
    assert os.environ["SLOTBANK_BUDGET_GIB"] == "9.0"
    assert os.environ["SLOTBANK_WARM"] == "0", "the rest of the preset should stand"

    _apply_tuning(SimpleNamespace(effort="high", **base))
    assert os.environ["SLOTBANK_PREFILL_STEP"] == "4096"
    assert os.environ["SLOTBANK_WARM_MIN_TOKENS"] == "0"


def test_high_effort_does_not_raise_slot_count():
    """Guards a measured trap: more slots is slower when the bank does not fit.

    C=32 measured 8.6 tok/s against C=64 at 6.5, because the extra pack comes
    out of the page cache serving its own misses. A "high" preset that raised
    the slot count would be slower than the default while looking faster.
    """
    from slotbank.cli import EFFORT

    for name, preset in EFFORT.items():
        assert "SLOTBANK_SLOTS_OVERRIDE" not in preset, name
        assert "SLOTBANK_BUDGET_GIB" not in EFFORT["high"], \
            "high must leave capacity to the policy, not force it up"
