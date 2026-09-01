from __future__ import annotations

import json
import os
import re
import uuid

from slotbank.types import ToolCall

_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_TOOL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_QWEN = re.compile(
    r"<\|tool_call_begin\|>(.*?)<\|tool_call_argument_end\|>",
    re.DOTALL,
)

# Chat template stops. Streaming them as content is how `<|im_end|>` showed
# up after "Hi. What do you need?" — the EOS token was generated, then decoded.
QWEN_STOPS: tuple[str, ...] = (
    "<|im_end|>",
    "<|endoftext|>",
    "<|im_start|>",
)


def strip_special(text: str) -> str:
    for s in QWEN_STOPS:
        text = text.replace(s, "")
    return text


def merge_stops(client: list[str] | None) -> list[str]:
    out: list[str] = []
    for s in list(client or []) + list(QWEN_STOPS):
        if s and s not in out:
            out.append(s)
    return out


def completion_cap(requested: int | None) -> int:
    """OMP yaml used to send maxTokens 8192; thinking then ate RAM on 'hi'."""
    raw = os.environ.get("SLOTBANK_MAX_COMPLETION", "2048").strip()
    try:
        cap = max(16, int(raw))
    except ValueError:
        cap = 2048
    if requested is None or int(requested) <= 0:
        return min(1024, cap)
    return min(int(requested), cap)


class SpecialHoldback:
    """Hold a suffix that might still become a stop token (`<|im` + `_end|>`)."""

    def __init__(self, stops: tuple[str, ...] = QWEN_STOPS):
        self.stops = stops
        self._buf = ""

    def push(self, piece: str) -> str:
        self._buf += piece or ""
        hold = 0
        for s in self.stops:
            for i in range(1, len(s)):
                if self._buf.endswith(s[:i]):
                    hold = max(hold, i)
        emit = self._buf[:-hold] if hold else self._buf
        self._buf = self._buf[-hold:] if hold else ""
        return strip_special(emit)

    def flush(self) -> str:
        out = strip_special(self._buf)
        self._buf = ""
        return out


def split_think(text: str) -> tuple[str, str]:
    blocks = _THINK.findall(text)
    if blocks:
        reasoning = "\n".join(b.strip() for b in blocks if b.strip())
        rest = _THINK.sub("", text).strip()
        return reasoning, strip_special(rest).strip()
    open_m = _THINK_OPEN.search(text)
    close_m = _THINK_CLOSE.search(text)
    if open_m and not close_m:
        return strip_special(text[open_m.end():]).strip(), ""
    if close_m and not open_m:
        return (
            strip_special(text[: close_m.start()]).strip(),
            strip_special(text[close_m.end():]).strip(),
        )
    return "", strip_special(text).strip()


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
    return strip_special(body).strip(), reasoning, calls


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
