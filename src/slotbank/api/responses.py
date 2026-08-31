from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from slotbank.load import EngineNotReady
from slotbank.prompt import _text_of
from slotbank.types import SamplingParams


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    previous_response_id: str | None = None
    background: bool = False


def _convert_item(item: dict[str, Any]) -> list[dict[str, Any]]:
    itype = item.get("type", "message")
    if itype == "message" or ("role" in item and "type" not in item):
        role = item.get("role", "user")
        if role == "developer":
            role = "system"
        return [{"role": role, "content": _text_of(item.get("content"))}]
    if itype == "function_call":
        return [{
            "role": "assistant",
            "tool_calls": [{
                "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "") or "",
                },
            }],
        }]
    if itype == "function_call_output":
        return [{
            "role": "tool",
            "tool_call_id": item.get("call_id", ""),
            "content": item.get("output") if isinstance(item.get("output"), str) else json.dumps(item.get("output") or ""),
        }]
    if itype == "reasoning":
        text = "".join(
            c.get("text") or ""
            for c in (item.get("content") or [])
            if isinstance(c, dict) and c.get("type") == "reasoning_text"
        )
        return [{"role": "assistant", "reasoning_content": text}] if text else []
    return []


def to_chat_messages(req: ResponsesRequest) -> list[dict[str, Any]]:
    system: list[str] = []
    if req.instructions:
        system.append(req.instructions)
    other: list[dict[str, Any]] = []
    if isinstance(req.input, str):
        other.append({"role": "user", "content": req.input})
    else:
        for item in req.input:
            for msg in _convert_item(item):
                if msg.get("role") == "system":
                    system.append(msg.get("content") or "")
                else:
                    other.append(msg)
    messages: list[dict[str, Any]] = []
    joined = "\n\n".join(t for t in system if t)
    if joined:
        messages.append({"role": "system", "content": joined})
    messages.extend(other)
    return messages


def to_openai_tools(tools: list[dict[str, Any]] | None, tool_choice: Any) -> list[dict[str, Any]] | None:
    if tool_choice == "none" or not tools:
        return None
    out = []
    for tool in tools:
        if tool.get("type") not in (None, "function"):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        out.append({
            "type": "function",
            "function": {
                "name": fn.get("name", ""),
                "description": fn.get("description"),
                "parameters": fn.get("parameters") or {"type": "object"},
            },
        })
    return out or None


def _err(status: int, message: str, code: str = "invalid_request") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error", "code": code}}, status)


def register_responses(app: FastAPI, engine) -> None:
    @app.post("/v1/responses")
    def responses(req: ResponsesRequest):
        if req.background:
            return _err(400, "background mode is not supported")
        if req.previous_response_id:
            return _err(400, "previous_response_id is not supported; resend full context in input")
        if req.max_output_tokens is not None and req.max_output_tokens < 1:
            return _err(400, "max_output_tokens must be a positive integer")
        try:
            chat = to_chat_messages(req)
            ids = engine.tokenize_chat(chat, to_openai_tools(req.tools, req.tool_choice))
        except EngineNotReady as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "overloaded_error", "code": "overloaded"}},
                503,
            )
        except ValueError as exc:
            return _err(400, str(exc))
        sampling = SamplingParams(
            temperature=0.0 if req.temperature is None else float(req.temperature),
            top_p=1.0 if req.top_p is None else float(req.top_p),
            top_k=-1 if req.top_k is None else int(req.top_k),
            max_tokens=int(req.max_output_tokens or 1024),
        )
        rid = f"resp_{uuid.uuid4().hex}"
        created = int(time.time())
        if req.stream:
            return StreamingResponse(
                _stream(engine, ids, sampling, req.model, rid, created),
                media_type="text/event-stream",
            )
        result = engine.generate(ids, sampling)
        return _full(result, req.model, rid, created)

    @app.get("/v1/responses/{response_id}")
    def get_response(response_id: str):
        return _err(404, f"response {response_id!r} not found (stateless server)")

    @app.post("/v1/responses/{response_id}/cancel")
    def cancel_response(response_id: str):
        return _err(404, f"response {response_id!r} not found (stateless server)")


def _usage(pt: int, ct: int) -> dict:
    return {
        "input_tokens": pt,
        "output_tokens": ct,
        "total_tokens": pt + ct,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


def _response(rid: str, created: int, model: str, status: str, output: list, usage=None, incomplete=None):
    body = {
        "id": rid,
        "object": "response",
        "created_at": created,
        "model": model,
        "status": status,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    if usage is not None:
        body["usage"] = usage
    if incomplete:
        body["incomplete_details"] = {"reason": incomplete}
    return body


def _full(result, model: str, rid: str, created: int) -> dict:
    output: list[dict[str, Any]] = []
    if result.reasoning:
        output.append({
            "id": f"rs_{uuid.uuid4().hex}",
            "type": "reasoning",
            "status": "completed",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": result.reasoning}],
        })
    if result.content:
        output.append({
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "status": "incomplete" if result.finish_reason == "length" else "completed",
            "content": [{"type": "output_text", "text": result.content, "annotations": []}],
        })
    for call in result.tool_calls:
        output.append({
            "type": "function_call",
            "id": f"fc_{uuid.uuid4().hex}",
            "call_id": call.call_id or f"call_{uuid.uuid4().hex[:24]}",
            "name": call.name,
            "arguments": call.arguments,
            "status": "completed",
        })
    truncated = result.finish_reason == "length"
    return _response(
        rid, created, model,
        "incomplete" if truncated else "completed",
        output,
        usage=_usage(result.prompt_tokens, result.completion_tokens),
        incomplete="max_output_tokens" if truncated else None,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream(engine, ids, sampling, model: str, rid: str, created: int):
    seq = 0

    def nxt() -> int:
        nonlocal seq
        seq += 1
        return seq

    empty = _response(rid, created, model, "in_progress", [])
    yield _sse("response.created", {"type": "response.created", "sequence_number": nxt(), "response": empty})
    yield _sse("response.in_progress", {"type": "response.in_progress", "sequence_number": nxt(), "response": empty})
    msg_id = f"msg_{uuid.uuid4().hex}"
    yield _sse("response.output_item.added", {
        "type": "response.output_item.added",
        "sequence_number": nxt(),
        "output_index": 0,
        "item": {"id": msg_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []},
    })
    yield _sse("response.content_part.added", {
        "type": "response.content_part.added",
        "sequence_number": nxt(),
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    })
    text = ""
    result = None
    for kind, payload in engine.stream(ids, sampling):
        if kind == "delta" and payload:
            text += payload
            yield _sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "sequence_number": nxt(),
                "item_id": msg_id,
                "output_index": 0,
                "content_index": 0,
                "delta": payload,
            })
        elif kind == "result":
            result = payload
        elif kind == "err":
            failed = _response(rid, created, model, "failed", [])
            failed["error"] = {"code": "server_error", "message": payload}
            yield _sse("response.failed", {"type": "response.failed", "sequence_number": nxt(), "response": failed})
            return
    yield _sse("response.output_text.done", {
        "type": "response.output_text.done",
        "sequence_number": nxt(),
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "text": (result.content if result and result.content else text),
    })
    yield _sse("response.content_part.done", {
        "type": "response.content_part.done",
        "sequence_number": nxt(),
        "item_id": msg_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": result.content if result else text, "annotations": []},
    })
    yield _sse("response.output_item.done", {
        "type": "response.output_item.done",
        "sequence_number": nxt(),
        "output_index": 0,
        "item": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": result.content if result else text, "annotations": []}],
        },
    })
    output = []
    if result:
        body = _full(result, model, rid, created)
        output = body["output"]
        usage = body["usage"]
        status = body["status"]
        event = "response.incomplete" if status == "incomplete" else "response.completed"
        yield _sse(event, {"type": event, "sequence_number": nxt(), "response": _response(rid, created, model, status, output, usage)})
    else:
        yield _sse("response.completed", {
            "type": "response.completed",
            "sequence_number": nxt(),
            "response": _response(rid, created, model, "completed", output),
        })
