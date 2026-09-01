from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from slotbank.load import EngineNotReady, is_loading, poll_until_ready
from slotbank.omp import DEFAULT_CONTEXT_WINDOW
from slotbank.prompt import enforce_prompt_cap
from slotbank.types import SamplingParams


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    ignore_eos: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    prompt: str | list[int]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None


def _stops(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return [s for s in stop if s]


def _sampling(req) -> SamplingParams:
    max_tokens = getattr(req, "max_completion_tokens", None) or req.max_tokens or 1024
    return SamplingParams(
        temperature=0.0 if req.temperature is None else float(req.temperature),
        top_p=1.0 if req.top_p is None else float(req.top_p),
        top_k=-1 if req.top_k is None else int(req.top_k),
        max_tokens=int(max_tokens),
        ignore_eos=bool(getattr(req, "ignore_eos", False)),
        stop_strs=_stops(req.stop),
    )


def _tools(req: ChatRequest) -> list[dict[str, Any]] | None:
    if req.tool_choice == "none" or not req.tools:
        return None
    return req.tools


def _openai_tools(calls) -> list[dict[str, Any]]:
    out = []
    for i, call in enumerate(calls):
        out.append({
            "id": call.call_id or f"call_{i}",
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        })
    return out


def _vision_on() -> bool:
    return os.environ.get("SLOTBANK_VISION", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _prompt_too_long(exc: ValueError) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": str(exc), "type": "invalid_request_error"}},
        400,
    )


def _not_ready(exc: EngineNotReady) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": str(exc), "type": "overloaded_error"}},
        503,
        headers={"Retry-After": "15"},
    )


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def llama_model_status(engine) -> dict[str, Any]:
    """llama.cpp router ``data[].status`` so OMP does not treat us as unloaded."""
    err = getattr(engine, "load_error", None)
    if err:
        return {"value": "failed", "message": str(err)}
    if getattr(engine, "loading", False):
        return {"value": "loading"}
    return {"value": "loaded"}


def models_payload(engine) -> dict[str, Any]:
    """OpenAI ``GET /v1/models`` plus llama.cpp native fields OMP 18 parses."""
    ctx = int(getattr(engine, "context_window", 0) or DEFAULT_CONTEXT_WINDOW)
    modalities = ["text", "image"] if _vision_on() else ["text"]
    return {
        "object": "list",
        "data": [{
            "id": engine.model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "slotbank",
            "supported_endpoint_types": ["anthropic", "openai"],
            "max_model_len": ctx,
            "context_length": ctx,
            "meta": {"n_ctx": ctx, "n_ctx_train": ctx},
            "architecture": {"input_modalities": modalities},
            "status": llama_model_status(engine),
        }],
    }


def llama_props_payload(engine) -> dict[str, Any]:
    """llama.cpp ``GET /props`` so OMP's 150 ms probe is not a miss."""
    ctx = int(getattr(engine, "context_window", 0) or DEFAULT_CONTEXT_WINDOW)
    return {
        "n_ctx": ctx,
        "modalities": {"vision": _vision_on()},
        "default_generation_settings": {
            "n_ctx": ctx,
            "params": {"max_tokens": -1, "n_predict": -1},
        },
    }


def register_chat(app: FastAPI, engine) -> None:
    @app.get("/v1/models")
    def models():
        return models_payload(engine)

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        if req.stream:
            return StreamingResponse(
                _stream_chat_when_ready(engine, req),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        try:
            # A base model with no chat template is a client error, not a 500.
            # /v1/messages and /v1/responses already guard this.
            ids = engine.tokenize_chat(req.messages, _tools(req))
        except EngineNotReady as exc:
            return _not_ready(exc)
        except ValueError as exc:
            return _prompt_too_long(exc)
        sampling = _sampling(req)
        result = engine.generate(ids, sampling)
        message: dict[str, Any] = {"role": "assistant", "content": result.content or None}
        if result.reasoning:
            message["reasoning_content"] = result.reasoning
        if result.tool_calls:
            message["tool_calls"] = _openai_tools(result.tool_calls)
            message["content"] = result.content or None
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason,
            }],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        if req.stream:
            return StreamingResponse(
                _stream_completion_when_ready(engine, req),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        try:
            ids = _prompt_ids(engine, req)
        except EngineNotReady as exc:
            return _not_ready(exc)
        except ValueError as exc:
            return _prompt_too_long(exc)
        sampling = _sampling(req)
        result = engine.generate(ids, sampling)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "text": result.content,
                "finish_reason": result.finish_reason,
            }],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _prompt_ids(engine, req: ChatRequest | CompletionRequest) -> list[int]:
    if isinstance(req, CompletionRequest):
        if isinstance(req.prompt, list):
            return enforce_prompt_cap([int(x) for x in req.prompt])
        return engine.tokenize_text(req.prompt)
    return engine.tokenize_chat(req.messages, _tools(req))


def _stream_chat_when_ready(engine, req: ChatRequest):
    """Ping while weights load. OMP llama.cpp streams this path, not Anthropic."""
    from slotbank.api.messages import LOAD_WAIT_S, STREAM_PING_S

    sampling = _sampling(req)
    uid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    yield _sse({
        "id": uid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": req.model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    })
    try:
        if is_loading(engine):
            for _ in poll_until_ready(
                engine, timeout=LOAD_WAIT_S, ping_s=STREAM_PING_S,
            ):
                yield ": ping\n\n"
        ids = _prompt_ids(engine, req)
    except EngineNotReady as exc:
        yield _sse({"error": {"message": str(exc), "type": "overloaded_error"}})
        yield "data: [DONE]\n\n"
        return
    except ValueError as exc:
        yield _sse({"error": {"message": str(exc), "type": "invalid_request_error"}})
        yield "data: [DONE]\n\n"
        return
    yield from _stream_chat_body(engine, ids, sampling, req.model, uid, created)


def _stream_chat(engine, ids, sampling, model: str):
    uid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    yield _sse({
        "id": uid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    })
    yield from _stream_chat_body(engine, ids, sampling, model, uid, created)


def _stream_chat_body(engine, ids, sampling, model: str, uid: str, created: int):
    from slotbank.api.messages import _with_pings

    result = None
    for kind, payload in _with_pings(engine.stream(ids, sampling)):
        if kind == "ping":
            yield ": ping\n\n"
            continue
        if kind == "delta" and payload:
            yield _sse({
                "id": uid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": payload}, "finish_reason": None}],
            })
        elif kind == "result":
            result = payload
        elif kind == "err":
            yield _sse({"error": {"message": payload}})
            yield "data: [DONE]\n\n"
            return
    finish = result.finish_reason if result else "stop"
    extra: dict[str, Any] = {}
    if result and result.tool_calls:
        extra["tool_calls"] = _openai_tools(result.tool_calls)
    yield _sse({
        "id": uid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": extra, "finish_reason": finish}],
    })
    yield "data: [DONE]\n\n"


def _stream_completion_when_ready(engine, req: CompletionRequest):
    from slotbank.api.messages import LOAD_WAIT_S, STREAM_PING_S

    sampling = _sampling(req)
    try:
        if is_loading(engine):
            for _ in poll_until_ready(
                engine, timeout=LOAD_WAIT_S, ping_s=STREAM_PING_S,
            ):
                yield ": ping\n\n"
        ids = _prompt_ids(engine, req)
    except EngineNotReady as exc:
        yield _sse({"error": {"message": str(exc), "type": "overloaded_error"}})
        yield "data: [DONE]\n\n"
        return
    except ValueError as exc:
        yield _sse({"error": {"message": str(exc), "type": "invalid_request_error"}})
        yield "data: [DONE]\n\n"
        return
    yield from _stream_completion(engine, ids, sampling, req.model)


def _stream_completion(engine, ids, sampling, model: str):
    uid = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    for kind, payload in engine.stream(ids, sampling):
        if kind == "delta":
            yield _sse({
                "id": uid,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "text": payload, "finish_reason": None}],
            })
        elif kind == "result":
            yield _sse({
                "id": uid,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "text": "", "finish_reason": payload.finish_reason}],
            })
        elif kind == "err":
            yield _sse({"error": {"message": payload}})
    yield "data: [DONE]\n\n"
