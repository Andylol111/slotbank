from __future__ import annotations

import json
import re
import uuid

from slotbank.types import ToolCall

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_TOOL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_QWEN = re.compile(
    r"<\|tool_call_begin\|>(.*?)<\|tool_call_argument_end\|>",
    re.DOTALL,
)


def split_think(text: str) -> tuple[str, str]:
    blocks = _THINK.findall(text)
    if not blocks:
        return "", text
    reasoning = "\n".join(b.strip() for b in blocks if b.strip())
    rest = _THINK.sub("", text).strip()
    return reasoning, rest


def parse_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for raw in _TOOL.findall(text):
        call = _from_json(raw)
        if call is not None:
            calls.append(call)
    if calls:
        return calls
    for raw in _QWEN.findall(text):
        name = ""
        args = raw
        if "<|tool_sep|>" in raw:
            name, args = raw.split("<|tool_sep|>", 1)
        name = name.replace("<|tool_call_argument_begin|>", "").strip()
        args = args.replace("<|tool_call_argument_begin|>", "").strip()
        calls.append(ToolCall(name=name, arguments=args, call_id=_cid()))
    return calls


def strip_tool_markup(text: str) -> str:
    text = _TOOL.sub("", text)
    text = _QWEN.sub("", text)
    return text.strip()


def finish_text(text: str) -> tuple[str, str, list[ToolCall]]:
    reasoning, body = split_think(text)
    calls = parse_tool_calls(body)
    if calls:
        body = strip_tool_markup(body)
    return body, reasoning, calls


def _from_json(raw: str) -> ToolCall | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    name = obj.get("name") or obj.get("function") or ""
    args = obj.get("arguments") or obj.get("parameters") or {}
    if not isinstance(args, str):
        args = json.dumps(args)
    if not name:
        return None
    return ToolCall(name=str(name), arguments=args, call_id=_cid())


def _cid() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"
