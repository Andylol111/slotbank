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


def hide_think_from_client() -> bool:
    """OMP envelope streams the answer only — no reasoning_content / thinking block."""
    return _envelope_on()


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


_NO_THINK_SW = re.compile(r"/no_think\b", re.IGNORECASE)
_THINK_SW = re.compile(r"/think\b", re.IGNORECASE)

# Qwen3.8 documented instruct vs thinking sampling. Applied only when the
# client sent the harness default (~1.0), so greedy CLI and an explicit
# 0.2 stay put. This is the model's own pair, not an OMP yaml temperature.
_QWEN_INSTRUCT = (0.7, 0.8, 20)
_QWEN_THINKING = (0.6, 0.95, 20)


def _last_user_index(messages: list[dict[str, Any]]) -> int | None:
    last = None
    for i, m in enumerate(messages):
        if (m.get("role") or "") == "user":
            last = i
    return last


def _ask_for_switch(text: str) -> str:
    """Only the last short ask, not an OMP cwd dump that might mention /think."""
    if not text:
        return ""
    _, ask = _last_ask(text, 256)
    return (ask or text[-240:]).strip()


def qwen_mode_of(messages: list[dict[str, Any]] | None) -> str:
    """Qwen3.8 native switch: /think or /no_think on the last ask.

    Serve envelope defaults to no_think so OMP prints the answer, not a
    think dump. CLI ``--thinking`` without envelope still thinks.
    """
    if not messages:
        if _envelope_on():
            return "no_think"
        return "think" if _thinking_on() else "no_think"
    idx = _last_user_index(messages)
    hay = _ask_for_switch(_text_of(messages[idx].get("content"))) if idx is not None else ""
    if _NO_THINK_SW.search(hay):
        return "no_think"
    if _THINK_SW.search(hay):
        return "think"
    if _envelope_on():
        return "no_think"
    return "think" if _thinking_on() else "no_think"


def with_qwen_mode(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append /no_think on the last user turn for the OMP envelope.

    Qwen's template reads that switch. It is unique to this model, not an
    OMP harness temperature or thinking-effort field.
    """
    if not messages or not _envelope_on():
        return messages
    if qwen_mode_of(messages) == "think":
        return messages
    idx = _last_user_index(messages)
    if idx is None:
        return messages
    out = [dict(m) for m in messages]
    text = _text_of(out[idx].get("content"))
    if _NO_THINK_SW.search(text):
        return out
    out[idx]["content"] = (text.rstrip() + "\n/no_think") if text.strip() else "/no_think"
    return out


def apply_qwen_sampling(sp, mode: str | None = None):
    """Map OMP's default temp 1.0 onto Qwen's documented instruct/think pair."""
    from slotbank.types import SamplingParams

    if not isinstance(sp, SamplingParams):
        return sp
    if not _envelope_on():
        return sp
    raw = os.environ.get("SLOTBANK_QWEN_SAMPLING", "1").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return sp
    mode = mode or sp.qwen_mode or "no_think"
    if float(sp.temperature) < 0.99:
        return sp
    temp, top_p, top_k = _QWEN_THINKING if mode == "think" else _QWEN_INSTRUCT
    return SamplingParams(
        temperature=temp,
        top_p=top_p if float(sp.top_p) >= 0.99 else sp.top_p,
        top_k=top_k if int(sp.top_k) < 1 else sp.top_k,
        ignore_eos=sp.ignore_eos,
        max_tokens=sp.max_tokens,
        stop_strs=list(sp.stop_strs or []),
        qwen_mode=mode,
    )


def _context_inject_on() -> bool:
    """Whether to compile the session log back into the chat prefix.

    Serve envelope sets CONTEXT_DIR so oversized OMP dumps can be logged.
    Compiling that log into the system message changes every turn
    (newest-first) and puts back the dump condense just removed — PrefixCache
    then misses. Opt in with SLOTBANK_CONTEXT_INJECT=1. A CONTEXT_DIR the
    user set themselves, without the envelope, still injects.
    """
    raw = os.environ.get("SLOTBANK_CONTEXT_INJECT", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    if _envelope_on():
        return False
    return bool(os.environ.get("SLOTBANK_CONTEXT_DIR", "").strip())


def with_context_os(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not _context_inject_on():
        return messages
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


def _is_cwd_dump(text: str) -> bool:
    return bool(_FOOTER_CHILD.search(text)) and _approx_tok(text) > 256


def _condense_user_blob(text: str, ask_n: int, cite_n: int) -> str:
    """Deterministic dump → ask+cites. Same raw text must stay the same turn to turn.

    PrefixCache is exact GDN. If user1 is citations+ask on turn 1 and a
    sys_n tail-clip of the raw dump on turn 2, the follow-up misses and
    pays the cold prefill again.
    """
    if _is_cwd_dump(text):
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
        return "\n\n".join(p for p in parts if p.strip())
    head, ask = _last_ask(text, ask_n)
    cites = _cites(head) or _cites(text)
    dump_head = _clip_tok(head, min(24, cite_n // 4))
    parts = []
    if cites:
        parts.append("Citations:\n" + "\n".join(cites))
    if dump_head.strip():
        parts.append(dump_head.rstrip())
    parts.append(ask.lstrip())
    return "\n\n".join(parts)


def condense_harness_messages(
    messages: list[dict[str, Any]],
    *,
    budget: int | None = None,
) -> list[dict[str, Any]]:
    """Local 27B stage: keep the ask + citations, not OMP's full harness blob.

    The harness prompt stays intact on the cloud subscription side. This only
    shapes what the Air prefills. Verbatim full text can still land on the
    context-OS disk log when SLOTBANK_CONTEXT_DIR is set.

    User dumps use one recipe whether they are the last ask or history.
    A follow-up that tail-clipped user1 used to miss PrefixCache.
    """
    budget = condense_budget() if budget is None else max(64, int(budget))
    sys_n = max(64, budget * 15 // 100)
    cite_n = max(64, budget * 25 // 100)
    ask_n = max(128, budget - sys_n - cite_n)
    out: list[dict[str, Any]] = []
    for raw in messages:
        m = dict(raw)
        role = m.get("role") or "user"
        text = _text_of(m.get("content"))
        if role == "system":
            if _approx_tok(text) <= sys_n:
                out.append(m)
            else:
                m["content"] = _clip_tok(text, sys_n) + "\n\n[harness system truncated]"
                out.append(m)
            continue
        if role == "tool" or m.get("tool_call_id"):
            if _approx_tok(text) <= sys_n:
                out.append(m)
            else:
                cites = _cites(text)
                m["content"] = "\n".join(cites) if cites else "[tool result omitted]"
                out.append(m)
            continue
        if role == "user":
            if _approx_tok(text) <= ask_n and not _is_cwd_dump(text):
                out.append(m)
            else:
                m["content"] = _condense_user_blob(text, ask_n, cite_n)
                out.append(m)
            continue
        if _approx_tok(text) <= sys_n:
            out.append(m)
        else:
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

    Head length is snapped to 512, 1024, 2048, ... The 8k envelope's 25%
    sink is 2048, which is PrefixCache.MAX_SNAP on the long-prompt path.
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


class PromptIds(list):
    """Chat token ids plus the PrefixCache stop that survives the next OMP encode.

    Three things are not a prefix of the next turn:
    - add_generation_prompt think tags (QwenLM/Qwen3#1826, mlx-engine#176)
    - envelope /no_think on the last user (historical turns omit it)
    - a last-user-only condense that rewrites user1 on the follow-up

    stable_prefix_n is the common prefix of the full encode and the same
    messages *without* the generation prompt and *without* /no_think.
    """

    stable_prefix_n: int = 0


def _prompt_ids(ids: list[int], stable: int = 0) -> PromptIds:
    out = PromptIds(ids)
    out.stable_prefix_n = max(0, min(int(stable), len(out)))
    return out


def _common_prefix_n(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def encode_chat(tokenizer, messages: list[dict[str, Any]], tools: list[dict] | None) -> list[int]:
    msgs = normalize_messages(with_context_os(with_direct(messages)))
    if _condense_on():
        _maybe_log_raw_user(messages)
        msgs = condense_harness_messages(msgs)
    if _condense_on() or _envelope_on():
        tools = slim_tools(tools)
    # /no_think is only on the last ask. The next OMP encode will omit it
    # from this turn, so PrefixCache must snap the pre-switch body.
    history = msgs
    msgs = with_qwen_mode(msgs)
    apply = getattr(tokenizer, "apply_chat_template", None)

    def plain() -> list[int]:
        text = "\n".join(f"{m['role']}: {m.get('content') or ''}" for m in msgs)
        return list(tokenizer.encode(text))

    if apply is None:
        return _prompt_ids(enforce_prompt_cap(plain()))
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": qwen_mode_of(history) == "think",
    }
    if tools:
        kwargs["tools"] = tools

    def _apply(m: list[dict[str, Any]], k: dict[str, Any]):
        return apply(m, **k)

    try:
        raw = _apply(msgs, kwargs)
    except TypeError:
        kwargs.pop("tools", None)
        try:
            raw = _apply(msgs, kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            raw = _apply(msgs, kwargs)
    except ValueError:
        # A base model ships the method but no template, and transformers
        # raises rather than returning None. Without this, every base
        # checkpoint 500s on /v1/chat/completions instead of falling back.
        return _prompt_ids(enforce_prompt_cap(plain()))
    full = _token_ids(raw)
    body: list[int] | None = None
    try:
        body_kw = dict(kwargs)
        body_kw["add_generation_prompt"] = False
        body = _token_ids(_apply(history, body_kw))
    except (TypeError, ValueError):
        body = None
    capped = enforce_prompt_cap(full)
    stable = 0
    if body:
        n = _common_prefix_n(body, full)
        # Packing rewrites the middle; only keep a stop that is still a
        # prefix of what we actually prefill.
        if n >= 32 and capped[:n] == full[:n]:
            stable = n
    return _prompt_ids(capped, stable)


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
