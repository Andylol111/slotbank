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
