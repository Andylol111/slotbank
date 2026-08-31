"""Write Oh My Pi ``~/.omp/agent/models.yml`` without PyYAML.

OMP's current schema is ``baseUrl`` + ``api: anthropic-messages`` + ``auth: none``.
The examples that used ``type: anthropic`` / ``base_url`` fail validation, and
**one invalid custom file disables every custom provider for that run**. OMP's
implicit llama.cpp probe also hits ``http://127.0.0.1:8080/models`` (native), not
``/v1/models``, so a slotbank server on 8080 is invisible until this file is valid.

Filesystem only. Nothing here imports MLX (tests/test_fence.py).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable

PROVIDER_ID = "slotbank"
BEGIN = "# --- slotbank omp (managed) ---"
END = "# --- end slotbank omp ---"
LEGACY_IDS = frozenset({
    "slotbank-qwen38-27b",
    "slotbank-qwen38-27b-dflash",
    "slotbank-qwen35-4b",
})

_PROVIDER_KEY = re.compile(r"^  ([A-Za-z0-9_.-]+):\s*(?:#.*)?$")
_SAFE_SCALAR = re.compile(r"^[A-Za-z0-9._-]+$")


def models_yml_path() -> Path:
    override = os.environ.get("SLOTBANK_OMP_YML", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".omp" / "agent" / "models.yml"


def selector(model_id: str) -> str:
    return f"{PROVIDER_ID}/{model_id}"


def _y(value: str) -> str:
    if _SAFE_SCALAR.fullmatch(value or ""):
        return value
    return json.dumps(value)


def render_provider(
    *,
    model_id: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    thinking: bool = False,
    vision: bool = False,
    context_window: int = 16384,
    max_tokens: int = 8192,
) -> str:
    """Indent-2 provider block, including the ``slotbank:`` key line."""
    base = f"http://{host}:{int(port)}/v1"
    inputs = "[text, image]" if vision else "[text]"
    name = f"{model_id} (slotbank)"
    lines = [
        f"  {PROVIDER_ID}:",
        f"    baseUrl: {base}",
        "    api: anthropic-messages",
        "    auth: none",
        "    disableStrictTools: true",
        "    discovery:",
        "      type: openai-models-list",
        "      timeoutMs: 2000",
        "    models:",
        f"      - id: {_y(model_id)}",
        f"        name: {_y(name)}",
        f"        reasoning: {'true' if thinking else 'false'}",
    ]
    if thinking:
        lines.extend([
            "        thinking:",
            "          mode: effort",
            "          efforts: [low, medium, high, xhigh]",
            "          defaultLevel: high",
        ])
    lines.extend([
        f"        input: {inputs}",
        "        tokenizer: qwen3",
        f"        contextWindow: {int(context_window)}",
        f"        maxTokens: {int(max_tokens)}",
        "        supportsTools: true",
    ])
    return "\n".join(lines) + "\n"


def _legacy_block(block: str) -> bool:
    """True when this provider would fail OMP's current schema."""
    if re.search(r"^    type:\s", block, re.M):
        return True
    if re.search(r"^    base_url:\s", block, re.M):
        return True
    if re.search(r"^    model:\s", block, re.M) and not re.search(r"^    models:\s", block, re.M):
        return True
    return False


def _provider_blocks(text: str) -> list[tuple[str, str]]:
    """``(id, full block including key line)`` under a root ``providers:`` map."""
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^providers:\s*(?:#.*)?$", line):
            start = i + 1
            break
    if start is None:
        return []
    out: list[tuple[str, str]] = []
    i = start
    while i < len(lines):
        m = _PROVIDER_KEY.match(lines[i].rstrip("\n"))
        if not m:
            i += 1
            continue
        pid = m.group(1)
        j = i + 1
        while j < len(lines):
            raw = lines[j]
            if _PROVIDER_KEY.match(raw.rstrip("\n")):
                break
            if raw.startswith(" ") or raw.startswith("\t") or not raw.strip() \
                    or raw.lstrip().startswith("#"):
                j += 1
                continue
            break
        out.append((pid, "".join(lines[i:j])))
        i = j
    return out


def kept_providers(text: str, *, drop: Iterable[str] = ()) -> list[tuple[str, str]]:
    drop_ids = {PROVIDER_ID, *LEGACY_IDS, *drop}
    kept = []
    for pid, block in _provider_blocks(text):
        if pid in drop_ids:
            continue
        if _legacy_block(block):
            continue
        kept.append((pid, block if block.endswith("\n") else block + "\n"))
    return kept


def compose_models_yml(
    *,
    model_id: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    thinking: bool = False,
    vision: bool = False,
    existing: str = "",
    context_window: int = 16384,
    max_tokens: int = 8192,
) -> str:
    others = kept_providers(existing)
    body = render_provider(
        model_id=model_id,
        host=host,
        port=port,
        thinking=thinking,
        vision=vision,
        context_window=context_window,
        max_tokens=max_tokens,
    )
    parts = [
        f"# Written by slotbank serve. Selector: {selector(model_id)}",
        "# Refresh the picker: omp models slotbank",
        "providers:",
    ]
    for _, block in others:
        parts.append(block.rstrip("\n"))
    parts.append(BEGIN)
    parts.append(body.rstrip("\n"))
    parts.append(END)
    return "\n".join(parts) + "\n"


def upsert(
    *,
    model_id: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    thinking: bool = False,
    vision: bool = False,
    path: Path | None = None,
    context_window: int = 16384,
    max_tokens: int = 8192,
) -> Path:
    dest = path or models_yml_path()
    existing = ""
    if dest.is_file():
        try:
            existing = dest.read_text(encoding="utf-8")
        except OSError:
            existing = ""
    text = compose_models_yml(
        model_id=model_id,
        host=host,
        port=port,
        thinking=thinking,
        vision=vision,
        existing=existing,
        context_window=context_window,
        max_tokens=max_tokens,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, dest)
    return dest
