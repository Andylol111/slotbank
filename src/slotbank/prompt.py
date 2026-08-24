from __future__ import annotations

from typing import Any


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") in {"text", "input_text", "output_text"} or "text" in part:
                parts.append(part.get("text") or "")
        else:
            parts.append(str(part))
    return "".join(parts)


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in messages:
        role = raw.get("role") or "user"
        if role == "developer":
            role = "system"
        msg: dict[str, Any] = {"role": role, "content": _text_of(raw.get("content"))}
        if raw.get("tool_call_id"):
            msg["tool_call_id"] = raw["tool_call_id"]
        if raw.get("name"):
            msg["name"] = raw["name"]
        if raw.get("tool_calls"):
            msg["tool_calls"] = raw["tool_calls"]
        if raw.get("reasoning_content"):
            msg["reasoning_content"] = raw["reasoning_content"]
        out.append(msg)
    return out


def encode_chat(tokenizer, messages: list[dict[str, Any]], tools: list[dict] | None) -> list[int]:
    msgs = normalize_messages(messages)
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        text = "\n".join(f"{m['role']}: {m.get('content') or ''}" for m in msgs)
        return list(tokenizer.encode(text))
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    if tools:
        kwargs["tools"] = tools
    try:
        ids = apply(msgs, **kwargs)
    except TypeError:
        kwargs.pop("tools", None)
        ids = apply(msgs, **kwargs)
    return [int(x) for x in ids]


def encode_text(tokenizer, text: str) -> list[int]:
    return [int(x) for x in tokenizer.encode(text)]
