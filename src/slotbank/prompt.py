from __future__ import annotations

import os
import re
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


def _flag(name: str, default: str = "") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1", "true", "yes", "on",
    }


def _thinking_on() -> bool:
    return _flag("SLOTBANK_THINKING", "0")


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


_FILE_CITE = re.compile(
    r"(?:file:)?((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z0-9]{1,8})"
    r"(?::(\d+)(?:-(\d+))?)?"
)
_FOOTER_CHILD = re.compile(r"↳\s+(\S+)")


def _approx_tok(text: str) -> int:
    return max(0, (len(text) + 3) // 4)


def _condense_on() -> bool:
    return _flag("SLOTBANK_CONDENSE") or _envelope_on()


def _envelope_on() -> bool:
    """Serve's OMP runtime: condense + slim tools + pack leftovers.

    Off unless SLOTBANK_ENVELOPE=1 (slotbank serve sets that). Without it,
    overlong prompts still 400 so a one-shot CLI cannot jetsam the Air.
    """
    return _flag("SLOTBANK_ENVELOPE")


def condense_budget() -> int:
    raw = os.environ.get("SLOTBANK_CONDENSE_BUDGET", "").strip()
    if raw:
        try:
            return max(256, int(raw))
        except ValueError:
            pass
    cap = max_prompt_tokens()
    # Envelope keeps more of a real ask than the old 4k cloud-stage clip.
    ceiling = 8192 if _envelope_on() else 4096
    if cap:
        return min(ceiling, cap)
    return ceiling


def _cites(text: str, limit: int = 24) -> list[str]:
    seen: list[str] = []
    found: set[str] = set()
    for m in _FILE_CITE.finditer(text):
        path = m.group(1)
        if ".." in path.split("/"):
            continue
        label = f"file:{path}"
        lo, hi = m.group(2), m.group(3)
        if lo:
            label += f":{lo}" + (f"-{hi}" if hi else "")
        if label not in found:
            found.add(label)
            seen.append(f"[{label}]")
        if len(seen) >= limit:
            break
    for m in _FOOTER_CHILD.finditer(text):
        name = m.group(1)
        key = f"cwd-child:{name}"
        if key not in found:
            found.add(key)
            seen.append(f"[{key}]")
        if len(seen) >= limit:
            break
    return seen


def _clip_tok(text: str, n: int, *, tail: bool = False) -> str:
    if n <= 0:
        return ""
    chars = max(1, int(n) * 4)
    if len(text) <= chars:
        return text
    return text[-chars:] if tail else text[:chars]


def _last_ask(text: str, ask_n: int) -> tuple[str, str]:
    """Split a dumped user blob into (head, ask). Prefer the last short paragraph."""
    parts = re.split(r"\n\s*\n", text.rstrip())
    if len(parts) >= 2:
        last = parts[-1].strip()
        if last and _approx_tok(last) <= ask_n:
            idx = text.rfind(parts[-1])
            return text[:idx], last
    lines = text.rstrip().splitlines()
    if lines:
        last = lines[-1].strip()
        if last and _approx_tok(last) <= min(ask_n, 256):
            idx = text.rfind(lines[-1])
            return text[:idx], last
    ask = _clip_tok(text, ask_n, tail=True)
    return text[: max(0, len(text) - len(ask))], ask


def condense_harness_messages(
    messages: list[dict[str, Any]],
    *,
    budget: int | None = None,
) -> list[dict[str, Any]]:
    """Local 27B stage: keep the ask + citations, not OMP's full harness blob.

    The harness prompt stays intact on the cloud subscription side. This only
    shapes what the Air prefills. Verbatim full text can still land on the
    context-OS disk log when SLOTBANK_CONTEXT_DIR is set.
    """
    budget = condense_budget() if budget is None else max(64, int(budget))
    sys_n = max(64, budget * 15 // 100)
    cite_n = max(64, budget * 25 // 100)
    ask_n = max(128, budget - sys_n - cite_n)
    last_user = None
    for i, m in enumerate(messages):
        if (m.get("role") or "") == "user":
            last_user = i
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(messages):
        m = dict(raw)
        role = m.get("role") or "user"
        text = _text_of(m.get("content"))
        limit = ask_n if i == last_user else sys_n
        if _approx_tok(text) <= limit:
            out.append(m)
            continue
        if role == "system":
            m["content"] = _clip_tok(text, sys_n) + "\n\n[harness system truncated]"
            out.append(m)
            continue
        if role == "tool" or m.get("tool_call_id"):
            cites = _cites(text)
            m["content"] = "\n".join(cites) if cites else "[tool result omitted]"
            out.append(m)
            continue
        if i == last_user:
            if _FOOTER_CHILD.search(text) and _approx_tok(text) > 256:
                _, ask = _last_ask(text, ask_n)
                child_cites: list[str] = []
                seen: set[str] = set()
                for hit in _FOOTER_CHILD.finditer(text):
                    key = f"cwd-child:{hit.group(1)}"
                    if key not in seen:
                        seen.add(key)
                        child_cites.append(f"[{key}]")
                parts = ["[cwd nested git dump omitted]"]
                if child_cites:
                    parts.append("Citations:\n" + "\n".join(child_cites[:8]))
                parts.append((ask or "hi").lstrip())
                m["content"] = "\n\n".join(p for p in parts if p.strip())
                out.append(m)
                continue
            head, ask = _last_ask(text, ask_n)
            cites = _cites(head) or _cites(text)
            dump_head = _clip_tok(head, min(24, cite_n // 4))
            parts = []
            if cites:
                parts.append("Citations:\n" + "\n".join(cites))
            if dump_head.strip():
                parts.append(dump_head.rstrip())
            parts.append(ask.lstrip())
            m["content"] = "\n\n".join(parts)
            out.append(m)
            continue
        m["content"] = _clip_tok(text, sys_n, tail=True)
        out.append(m)
    return out


def _maybe_log_raw_user(messages: list[dict[str, Any]]) -> None:
    root = os.environ.get("SLOTBANK_CONTEXT_DIR", "").strip()
    if not root:
        return
    for m in reversed(messages):
        if (m.get("role") or "") != "user":
            continue
        text = _text_of(m.get("content"))
        if _approx_tok(text) <= condense_budget():
            return
        from slotbank.context_os import append

        append(root, "user", text)
        return


# 27B 4-bit on 24 GB cannot prefill a raw ~15k OMP dump (jetsam ~60s).
# Envelope condenses first; this cap is the Metal pack target after that.
# 0 disables.
DEFAULT_MAX_PROMPT_TOKENS = 8192
# Serve envelope: 8k tokens × 64 KiB attn KV ≈ 512 MiB. 10k was a 640 MiB
# KV plus a 790 MiB prefix copy; that is what skyrocketed RAM on 24 GB.
DEFAULT_ENVELOPE_MAX_PROMPT = 8192
TOOL_SLIM_BUDGET = 256


def max_prompt_tokens() -> int:
    raw = os.environ.get("SLOTBANK_MAX_PROMPT", "").strip()
    if not raw:
        if _envelope_on():
            return DEFAULT_ENVELOPE_MAX_PROMPT
        return DEFAULT_MAX_PROMPT_TOKENS
    try:
        n = int(raw)
    except ValueError:
        return DEFAULT_MAX_PROMPT_TOKENS
    if n < 0:
        return DEFAULT_MAX_PROMPT_TOKENS
    return n


def _prompt_pack_on() -> bool:
    return _flag("SLOTBANK_PROMPT_PACK") or _envelope_on()


def _pyramid_sample(src: list[int], k: int) -> list[int]:
    """Keep k tokens from src, denser at the start, order preserved."""
    n = len(src)
    if k <= 0:
        return []
    if n <= k:
        return list(src)
    seen: set[int] = set()
    picked: list[int] = []
    den = (k - 1) * (k - 1) if k > 1 else 1
    for t in range(k):
        idx = 0 if k == 1 else (t * t * (n - 1)) // den
        if idx not in seen:
            seen.add(idx)
            picked.append(idx)
    if len(picked) < k:
        for i in range(n):
            if i not in seen:
                seen.add(i)
                picked.append(i)
                if len(picked) == k:
                    break
    picked.sort()
    return [src[i] for i in picked]


def keep_token_ids(ids: list[int], cap: int) -> list[int]:
    """Pack an overlong prompt without touching hybrid KV.

    Head is the attention sink / system prefix. Tail is the current turn.
    The middle is a one-shot pyramidal sample (dense near the head). This is
    the PyramidKV + BUZZ + TriAttention idea applied to *what gets prefills*,
    not to Gated DeltaNet state.

    Head length is snapped to 512, 1024, 2048, ... so PrefixCache on the
    MTP path can restore it. 25% of the 8k envelope is 2048, a snap.
    """
    n = len(ids)
    if n <= cap or cap <= 0:
        return list(ids)
    raw_head = max(1, cap * 1 // 4)
    head_n = _snap_len(raw_head)
    tail_n = max(1, cap * 5 // 10)
    if head_n + tail_n > cap:
        tail_n = min(cap, max(1, cap - 1))
        head_n = cap - tail_n
    mid_n = cap - head_n - tail_n
    if head_n + tail_n >= n:
        return ids[:head_n] + ids[n - (cap - head_n) :]
    head = ids[:head_n]
    tail = ids[-tail_n:]
    mid = _pyramid_sample(ids[head_n : n - tail_n], mid_n)
    return head + mid + tail


def _snap_len(n: int) -> int:
    """Largest 512×2^k that still fits in n. Tiny heads stay as-is."""
    if n < 512:
        return n
    snap = 512
    while snap * 2 <= n:
        snap *= 2
    return snap


def maybe_pack_prompt(ids: list[int]) -> list[int]:
    """Pack sink+pyramid+tail when over cap. Envelope does this instead of 400."""
    cap = max_prompt_tokens()
    if not cap or len(ids) <= cap or not _prompt_pack_on():
        return ids
    return keep_token_ids(ids, cap)


def slim_tools(
    tools: list[dict[str, Any]] | None,
    *,
    budget: int | None = None,
) -> list[dict[str, Any]] | None:
    """Keep tool names, drop JSON schemas. OMP's catalog alone is >16k tokens."""
    if not tools:
        return None
    limit = TOOL_SLIM_BUDGET if budget is None else max(32, int(budget))
    out: list[dict[str, Any]] = []
    used = 0
    for raw in tools:
        fn = raw.get("function") if isinstance(raw.get("function"), dict) else raw
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or raw.get("name") or "tool").strip() or "tool"
        desc = str(fn.get("description") or raw.get("description") or "")
        desc = desc.strip().splitlines()[0][:120] if desc.strip() else ""
        cost = _approx_tok(name + " " + desc) + 8
        if out and used + cost > limit:
            break
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {"type": "object", "properties": {}},
            },
        })
        used += cost
    return out or None


def enforce_prompt_cap(ids: list[int]) -> list[int]:
    ids = maybe_pack_prompt(ids)
    cap = max_prompt_tokens()
    if cap and len(ids) > cap:
        raise ValueError(
            f"prompt is {len(ids)} tokens (cap {cap}). "
            "27B on 24 GB cannot prefill that. "
            "OMP footer `/tmp ↳ name` means a git repo that is a *child* of cwd "
            "was injected (e.g. /tmp/llama.cpp-dflash2). "
            "slotbank serve envelopes OMP dumps (condense + slim tools + pack). "
            "Override: SLOTBANK_MAX_PROMPT=0, SLOTBANK_ENVELOPE=1, "
            "SLOTBANK_CONDENSE=1, or SLOTBANK_PROMPT_PACK=1."
        )
    return ids


def encode_chat(tokenizer, messages: list[dict[str, Any]], tools: list[dict] | None) -> list[int]:
    msgs = normalize_messages(with_context_os(with_direct(messages)))
    if _condense_on():
        _maybe_log_raw_user(messages)
        msgs = condense_harness_messages(msgs)
    if _condense_on() or _envelope_on():
        tools = slim_tools(tools)
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
