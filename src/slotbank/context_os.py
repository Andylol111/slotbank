"""Lossless context OS: disk log is the source of truth; the model sees excerpts.

The working set is selected spans with pointers, never an abstractive summary
as the only copy. Cloud compile is optional (SLOTBANK_CONTEXT_COMPILER_URL).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

_FILE_PTR = re.compile(
    r"file:(?P<path>[^\s:]+)(?::(?P<lo>\d+)(?:-(?P<hi>\d+))?)?"
)
_DEFAULT_BUDGET = 4096
_DEFAULT_EXPAND = 1024
_LOG_NAME = "log.jsonl"


def context_dir(path: str | Path | None = None) -> Path:
    raw = path or os.environ.get("SLOTBANK_CONTEXT_DIR") or ""
    if not raw:
        raise ValueError("set --dir or SLOTBANK_CONTEXT_DIR")
    return Path(raw)


def init_session(dir_path: str | Path) -> Path:
    root = Path(dir_path)
    root.mkdir(parents=True, exist_ok=True)
    log = root / _LOG_NAME
    if not log.is_file():
        log.write_text("")
    return root


def append(
    dir_path: str | Path,
    role: str,
    content: str,
    *,
    pointers: list[str] | None = None,
) -> dict[str, Any]:
    root = init_session(dir_path)
    log = root / _LOG_NAME
    seq = sum(1 for _ in _read_log(root)) + 1
    rec = {
        "seq": seq,
        "role": role,
        "content": content,
        "pointers": list(pointers or []),
    }
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def _read_log(root: Path) -> list[dict[str, Any]]:
    path = root / _LOG_NAME
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _approx_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _expand_pointer(ptr: str, repo: Path | None) -> str | None:
    m = _FILE_PTR.fullmatch(ptr.strip())
    if m is None or repo is None:
        return None
    rel = Path(m.group("path"))
    if rel.is_absolute() or ".." in rel.parts:
        return None
    target = (repo / rel).resolve()
    try:
        target.relative_to(repo.resolve())
    except ValueError:
        return None
    if not target.is_file():
        return None
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    lo = int(m.group("lo") or 1)
    hi = int(m.group("hi") or lo)
    lo = max(1, lo)
    hi = min(len(lines), max(lo, hi))
    body = "\n".join(lines[lo - 1 : hi])
    return f"[file:{rel}:{lo}-{hi}]\n{body}"


def _expand_cap() -> int:
    """Max tokens of inlined file bodies. 0 = cite pointers only.

    The log and the files stay on disk. Inlining a whole span is what
    inflates the prompt — and the KV — during the context stage.
    """
    raw = os.environ.get("SLOTBANK_CONTEXT_EXPAND", "").strip()
    if not raw:
        return _DEFAULT_EXPAND
    if raw.lower() in {"off", "none"}:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_EXPAND


def _pointer_excerpt(ptr: str, repo: Path | None, expand_left: int) -> tuple[str, int]:
    """Citation always; file body only when the whole span fits expand_left."""
    cite = f"[{ptr}]"
    expanded = _expand_pointer(str(ptr), repo)
    if expanded is None or expand_left <= 0:
        return cite, 0
    cost = _approx_tokens(expanded)
    if cost > expand_left:
        return cite, 0
    return expanded, cost


def _local_compile(
    records: list[dict[str, Any]],
    budget: int,
    repo: Path | None,
) -> str:
    """Newest-first excerpts until the token budget is spent. Verbatim only."""
    chunks: list[str] = []
    used = 0
    expand_left = _expand_cap()
    for rec in reversed(records):
        seq = rec.get("seq")
        role = rec.get("role") or "user"
        content = rec.get("content") or ""
        block = f"[log:{seq} {role}]\n{content}"
        extras = []
        for ptr in rec.get("pointers") or []:
            excerpt, spent = _pointer_excerpt(str(ptr), repo, expand_left)
            extras.append(excerpt)
            expand_left -= spent
        if extras:
            block += "\n" + "\n".join(extras)
        cost = _approx_tokens(block)
        if chunks and used + cost > budget:
            break
        chunks.append(block)
        used += cost
    chunks.reverse()
    return "\n\n".join(chunks)


def _cloud_compile(payload: dict[str, Any], url: str) -> str:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ValueError(f"cloud compiler failed: {exc}") from exc
    text = body.get("working_set") or body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("cloud compiler returned no working_set")
    return text


def compile_working_set(
    dir_path: str | Path,
    *,
    budget: int | None = None,
    repo: str | Path | None = None,
    compiler_url: str | None = None,
) -> str:
    root = context_dir(dir_path)
    records = _read_log(root)
    budget = budget or int(os.environ.get("SLOTBANK_CONTEXT_BUDGET") or _DEFAULT_BUDGET)
    repo_path = Path(repo) if repo else None
    url = compiler_url or os.environ.get("SLOTBANK_CONTEXT_COMPILER_URL") or ""
    if url:
        try:
            return _cloud_compile(
                {
                    "log": records,
                    "budget": budget,
                    "repo": str(repo_path) if repo_path else None,
                },
                url,
            )
        except ValueError:
            pass
    return _local_compile(records, budget, repo_path)


def compiled_system_message(
    dir_path: str | Path | None = None,
    *,
    budget: int | None = None,
    repo: str | Path | None = None,
) -> str:
    raw = dir_path or os.environ.get("SLOTBANK_CONTEXT_DIR")
    if not raw:
        return ""
    text = compile_working_set(raw, budget=budget, repo=repo)
    if not text.strip():
        return ""
    return (
        "Working set compiled from the session log. Pointers are the source "
        "of truth; do not treat this as a summary that replaces the log.\n\n"
        + text
    )


def iter_log(dir_path: str | Path) -> Iterable[dict[str, Any]]:
    return _read_log(context_dir(dir_path))
