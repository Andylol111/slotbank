from __future__ import annotations

import os
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


def _thinking_on() -> bool:
    return os.environ.get("SLOTBANK_THINKING", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _direct_on() -> bool:
    return os.environ.get("SLOTBANK_DIRECT", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


# Short on purpose: it sits in every prefix, so it must stay byte-stable.
_DIRECT_SYSTEM = (
    "You are a local assistant on this machine. Be direct, specific, and useful. "
    "Do not refuse ordinary adult, creative, or technical requests out of caution. "
    "Do not add unsolicited lectures or disclaimers. Follow the user's instructions."
)


def direct_system_text() -> str:
    raw = os.environ.get("SLOTBANK_DIRECT_PROMPT", "").strip()
    if raw:
        return raw
    return _DIRECT_SYSTEM if _direct_on() else ""


def with_direct(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepend a stable persona. Does not replace a client system message."""
    text = direct_system_text()
    if not text:
        return messages
    if messages and messages[0].get("role") == "system":
        head = dict(messages[0])
        body = _text_of(head.get("content"))
        if body.startswith(text):
            return messages
        head["content"] = text + ("\n\n" + body if body else "")
        return [head, *messages[1:]]
    return [{"role": "system", "content": text}, *messages]


def with_context_os(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prefix = compiled_system_message()
    if not prefix:
        return messages
    if messages and messages[0].get("role") == "system":
        head = dict(messages[0])
        head["content"] = prefix + "\n\n" + _text_of(head.get("content"))
        return [head, *messages[1:]]
    return [{"role": "system", "content": prefix}, *messages]


def compiled_system_message() -> str:
    if not os.environ.get("SLOTBANK_CONTEXT_DIR"):
        return ""
    from slotbank.context_os import compiled_system_message as compile_msg

    repo = os.environ.get("SLOTBANK_CONTEXT_REPO")
    return compile_msg(repo=repo)


# 27B 4-bit on 24 GB cannot prefill a repo-sized OMP dump (~26k tokens from
# cwd ~/Desktop/slotbank). Refuse before Metal alloc; 0 disables.
DEFAULT_MAX_PROMPT_TOKENS = 16384


def max_prompt_tokens() -> int:
    raw = os.environ.get("SLOTBANK_MAX_PROMPT", "").strip()
    if not raw:
        return DEFAULT_MAX_PROMPT_TOKENS
    try:
        n = int(raw)
    except ValueError:
        return DEFAULT_MAX_PROMPT_TOKENS
    if n < 0:
        return DEFAULT_MAX_PROMPT_TOKENS
    return n


def enforce_prompt_cap(ids: list[int]) -> list[int]:
    cap = max_prompt_tokens()
    if cap and len(ids) > cap:
        raise ValueError(
            f"prompt is {len(ids)} tokens (cap {cap}). "
            "27B on 24 GB cannot prefill that. "
            "OMP over ~50% of 33K is tools, MCP, or a project (footer ↳ name), "
            "not just cwd. New session: omp --tools read, no project, no MCP. "
            "Override: SLOTBANK_MAX_PROMPT=0"
        )
    return ids


def encode_chat(tokenizer, messages: list[dict[str, Any]], tools: list[dict] | None) -> list[int]:
    msgs = normalize_messages(with_context_os(with_direct(messages)))
    apply = getattr(tokenizer, "apply_chat_template", None)

    def plain() -> list[int]:
        text = "\n".join(f"{m['role']}: {m.get('content') or ''}" for m in msgs)
        return list(tokenizer.encode(text))

    if apply is None:
        return enforce_prompt_cap(plain())
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": _thinking_on(),
    }
    if tools:
        kwargs["tools"] = tools
    try:
        ids = apply(msgs, **kwargs)
    except TypeError:
        kwargs.pop("tools", None)
        try:
            ids = apply(msgs, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            ids = apply(msgs, **kwargs)
    except ValueError:
        # A base model ships the method but no template, and transformers
        # raises rather than returning None. Without this, every base
        # checkpoint 500s on /v1/chat/completions instead of falling back.
        return enforce_prompt_cap(plain())
    return enforce_prompt_cap(_token_ids(ids))


def _token_ids(ids) -> list[int]:
    """mlx-lm returns a list; mlx-vlm processors return a BatchEncoding.

    BatchEncoding is not a dict subclass; iterating it yields the keys.
    """
    getter = getattr(ids, "get", None)
    if callable(getter):
        got = getter("input_ids")
        if got is not None:
            ids = got
    if hasattr(ids, "tolist") and not isinstance(ids, (list, tuple)):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


def encode_text(tokenizer, text: str) -> list[int]:
    return enforce_prompt_cap(_token_ids(tokenizer.encode(text)))
