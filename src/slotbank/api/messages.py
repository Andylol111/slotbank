from __future__ import annotations

import json
import queue
import threading
import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from slotbank.types import SamplingParams


class MessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = 1024
    system: Any | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None
    tools: list[dict[str, Any]] | None = None


def _system_text(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts = []
    for block in system:
        if isinstance(block, dict):
            parts.append(block.get("text") or "")
        else:
            parts.append(str(block))
    return "\n".join(p for p in parts if p)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text") or "")
        elif btype == "tool_result":
            parts.append(_content_text(block.get("content")))
    return "".join(parts)


def to_chat_messages(req: MessagesRequest) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    system = _system_text(req.system)
    if system:
        out.append({"role": "system", "content": system})
    for msg in req.messages:
        role = msg.get("role") or "user"
        content = msg.get("content")
        if role == "assistant" and isinstance(content, list):
            texts = []
            tools = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    texts.append(block.get("text") or "")
                elif block.get("type") == "tool_use":
                    tools.append({
                        "id": block.get("id") or "",
                        "type": "function",
                        "function": {
                            "name": block.get("name") or "",
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    })
            row: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if tools:
                row["tool_calls"] = tools
            out.append(row)
            continue
        if role == "user" and isinstance(content, list):
            texts = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    texts.append(block.get("text") or "")
                elif block.get("type") == "tool_result":
                    out.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id") or "",
                        "content": _content_text(block.get("content")),
                    })
            if texts:
                out.append({"role": "user", "content": "".join(texts)})
            continue
        out.append({"role": role, "content": _content_text(content)})
    return out


def to_openai_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    out = []
    for tool in tools:
        out.append({
            "type": "function",
            "function": {
                "name": tool.get("name") or "",
                "description": tool.get("description"),
                "parameters": tool.get("input_schema") or tool.get("parameters") or {"type": "object"},
            },
        })
    return out


def _err(status: int, typ: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"type": "error", "error": {"type": typ, "message": message}},
        status_code=status,
    )


def _stop(finish: str | None, matched: str | None) -> tuple[str, str | None]:
    if matched:
        return "stop_sequence", matched
    return {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}.get(
        finish or "stop", "end_turn"
    ), None


def register_messages(app: FastAPI, engine) -> None:
    @app.post("/v1/messages")
    def messages(req: MessagesRequest):
        try:
            chat = to_chat_messages(req)
            if not chat:
                return _err(400, "invalid_request_error", "messages: no tokenizable content")
            ids = engine.tokenize_chat(chat, to_openai_tools(req.tools))
        except ValueError as exc:
            return _err(400, "invalid_request_error", str(exc))
        sampling = SamplingParams(
            temperature=1.0 if req.temperature is None else float(req.temperature),
            top_p=1.0 if req.top_p is None else float(req.top_p),
            top_k=-1 if req.top_k is None else int(req.top_k),
            max_tokens=int(req.max_tokens),
            stop_strs=list(req.stop_sequences or []),
        )
        mid = f"msg_{uuid.uuid4().hex}"
        if req.stream:
            return StreamingResponse(
                _stream(engine, ids, sampling, req.model, mid),
                media_type="text/event-stream",
            )
        result = engine.generate(ids, sampling)
        reason, seq = _stop(result.finish_reason, result.matched_stop)
        content: list[dict[str, Any]] = []
        if result.reasoning:
            content.append({"type": "thinking", "thinking": result.reasoning})
        if result.content:
            content.append({"type": "text", "text": result.content})
        for call in result.tool_calls:
            try:
                inp = json.loads(call.arguments)
            except json.JSONDecodeError:
                inp = {"raw": call.arguments}
            content.append({
                "type": "tool_use",
                "id": call.call_id or f"toolu_{uuid.uuid4().hex[:12]}",
                "name": call.name,
                "input": inp,
            })
        return {
            "id": mid,
            "type": "message",
            "role": "assistant",
            "model": req.model,
            "content": content,
            "stop_reason": reason,
            "stop_sequence": seq,
            "usage": {
                "input_tokens": result.prompt_tokens,
                "output_tokens": result.completion_tokens,
            },
        }

    @app.post("/v1/messages/count_tokens")
    def count_tokens(req: MessagesRequest):
        try:
            chat = to_chat_messages(req)
            ids = engine.tokenize_chat(chat, to_openai_tools(req.tools))
        except ValueError as exc:
            return _err(400, "invalid_request_error", str(exc))
        return {"input_tokens": len(ids)}


def _event(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_DONE = object()
# OMP aborts Anthropic streams that go silent during 27B prefill. A ping
# every few seconds is cheaper than a stall after "hi".
STREAM_PING_S = 5.0


def _with_pings(it, timeout: float | None = None):
    """Yield ``("ping", None)`` when ``it`` is quiet for ``timeout`` seconds."""
    wait = STREAM_PING_S if timeout is None else float(timeout)
    q: queue.Queue = queue.Queue()

    def run() -> None:
        try:
            for item in it:
                q.put(item)
        except Exception as exc:
            q.put(("err", str(exc)))
        finally:
            q.put(_DONE)

    threading.Thread(target=run, daemon=True, name="slotbank-sse-ping").start()
    while True:
        try:
            item = q.get(timeout=wait)
        except queue.Empty:
            yield ("ping", None)
            continue
        if item is _DONE:
            return
        yield item


def _stream(engine, ids, sampling, model: str, mid: str):
    yield _event("message_start", {
        "type": "message_start",
        "message": {
            "id": mid,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": len(ids), "output_tokens": 0},
        },
    })
    yield _event("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })
    result = None
    for kind, payload in _with_pings(engine.stream(ids, sampling)):
        if kind == "ping":
            yield _event("ping", {"type": "ping"})
            continue
        if kind == "delta" and payload:
            yield _event("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": payload},
            })
        elif kind == "result":
            result = payload
        elif kind == "err":
            yield _event("error", {"type": "error", "error": {"type": "api_error", "message": payload}})
            return
    yield _event("content_block_stop", {"type": "content_block_stop", "index": 0})
    reason, seq = _stop(result.finish_reason if result else "stop", result.matched_stop if result else None)
    yield _event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": reason, "stop_sequence": seq},
        "usage": {"output_tokens": result.completion_tokens if result else 0},
    })
    yield _event("message_stop", {"type": "message_stop"})
