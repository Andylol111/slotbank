from __future__ import annotations

import json
import re
from pathlib import Path

from slotbank.admit import discover_sidecar_draft, public_model_id
from slotbank.omp import (
    BEGIN,
    compose_models_yml,
    render_provider,
    selector,
    upsert,
)


def _block(text: str, provider: str) -> str:
    key = f"  {provider}:"
    start = text.index(key)
    rest = text[start + len(key):]
    nxt = re.search(r"\n  [A-Za-z0-9_.-]+:", rest)
    return text[start: start + len(key) + (nxt.start() if nxt else len(rest))]


def _checkpoint(root: Path, name: str, cfg: dict) -> Path:
    d = root / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "weights.safetensors").write_bytes(b"x")
    return d


def test_public_model_id_folder_and_snapshot(tmp_path):
    pack = tmp_path / "Qwen3.8-27B-4bit"
    pack.mkdir()
    assert public_model_id(str(pack)) == "Qwen3.8-27B-4bit"
    snap = (
        tmp_path / "hub" / "models--mlx-community--Qwen3.8-27B-4bit"
        / "snapshots" / "23511b9407abcdef0123456789abcdef01234567"
    )
    snap.mkdir(parents=True)
    assert public_model_id(str(snap)) == "Qwen3.8-27B-4bit"
    assert public_model_id("mlx-community/Qwen3.8-27B-4bit") == "Qwen3.8-27B-4bit"
    assert public_model_id("") == "model"


def test_discover_sidecar_prefers_mtp(tmp_path):
    tgt = _checkpoint(tmp_path, "Qwen3.8-27B-4bit", {
        "vocab_size": 248320, "hidden_size": 5120,
    })
    _checkpoint(tmp_path, "Qwen3.8-27B-DFlash2-4bit", {
        "model_type": "dflash2", "vocab_size": 248320,
    })
    mtp = _checkpoint(tmp_path, "Qwen3.8-27B-MTP-4bit", {
        "model_type": "qwen3_5_mtp",
        "vocab_size": 248320,
        "text_config": {"vocab_size": 248320, "hidden_size": 5120},
    })
    assert discover_sidecar_draft(str(tgt)) == str(mtp)
    # a same-vocab 4B is not a drafter
    _checkpoint(tmp_path, "Qwen3.8-27B-4B-4bit", {
        "vocab_size": 248320, "hidden_size": 2560, "model_type": "qwen3_5",
    })
    assert discover_sidecar_draft(str(tgt)) == str(mtp)


def test_discover_sidecar_dflash_when_no_mtp(tmp_path):
    tgt = _checkpoint(tmp_path, "Qwen3.8-27B-4bit", {
        "vocab_size": 248320, "hidden_size": 5120,
    })
    dflash = _checkpoint(tmp_path, "Qwen3.8-27B-DFlash2-4bit", {
        "model_type": "dflash2",
    })
    assert discover_sidecar_draft(str(tgt)) == str(dflash)
    assert discover_sidecar_draft(str(tmp_path / "missing")) is None


def test_render_provider_is_current_omp_schema():
    text = render_provider(
        model_id="Qwen3.8-27B-4bit", thinking=True, vision=True,
    )
    assert "type: anthropic" not in text
    assert "base_url:" not in text
    assert "baseUrl: http://127.0.0.1:8080/v1" in text
    assert "api: anthropic-messages" in text
    assert "auth: none" in text
    assert "id: Qwen3.8-27B-4bit" in text
    assert "id: Qwen3.8-27B-4bit-agent" in text
    assert "supportsTools: false" in text
    assert "supportsTools: true" in text
    assert "reasoning: true" in text
    assert "reasoning: false" in text
    assert "input: [text, image]" in text
    assert "streamIdleTimeoutMs: 600000" in text
    assert selector("Qwen3.8-27B-4bit") == "llama.cpp/Qwen3.8-27B-4bit"
    llama = render_provider(
        model_id="Qwen3.8-27B-4bit", thinking=True, vision=True,
        provider_id="llama.cpp", api="openai-completions", discovery=None,
    )
    assert "  llama.cpp:" in llama
    assert "api: openai-completions" in llama
    assert "thinkingFormat: qwen-chat-template" in llama
    assert "streamIdleTimeoutMs: 600000" in llama
    assert "discovery:" not in llama


def test_compose_replaces_legacy_and_keeps_valid_sibling():
    existing = """\
providers:
  slotbank-qwen38-27b:
    type: anthropic
    base_url: http://127.0.0.1:8080
    model: Qwen3.8-27B-4bit
  other-lab:
    baseUrl: http://127.0.0.1:9000/v1
    api: openai-completions
    auth: none
"""
    text = compose_models_yml(
        model_id="Qwen3.8-27B-4bit", thinking=True, existing=existing,
    )
    assert "slotbank-qwen38-27b" not in text
    assert "type: anthropic" not in text
    assert "base_url:" not in text
    assert "other-lab:" in text
    assert "baseUrl: http://127.0.0.1:9000/v1" in text
    assert BEGIN in text
    assert text.count("  slotbank:") == 1
    assert text.count("  llama.cpp:") == 1
    assert text.count("  lm-studio:") == 1
    assert "api: anthropic-messages" in text
    assert "api: openai-completions" in text
    assert "timeoutMs: 30000" in text
    assert "thinkingFormat: qwen-chat-template" in text
    assert "discovery:" not in _block(text, "llama.cpp")
    assert "discovery:" not in _block(text, "lm-studio")
    assert "type: openai-models-list" in _block(text, "slotbank")
    assert "contextWindow: 32768" in text
    assert "Qwen3.8-27B-4bit-agent" in text
    assert "cwd dump" in text
    assert "two-stage" in text.lower()
    assert selector("Qwen3.8-27B-4bit") == "llama.cpp/Qwen3.8-27B-4bit"


def test_upsert_roundtrip(tmp_path, monkeypatch):
    dest = tmp_path / "agent" / "models.yml"
    monkeypatch.setenv("SLOTBANK_OMP_YML", str(dest))
    from slotbank.omp import models_yml_path

    assert models_yml_path() == dest
    path = upsert(model_id="Qwen3.8-27B-4bit", thinking=True, vision=True, path=dest)
    assert path == dest
    body = dest.read_text()
    assert "Qwen3.8-27B-4bit" in body
    upsert(model_id="Qwen3.8-27B-4bit", thinking=True, path=dest)
    body = dest.read_text()
    assert body.count("  slotbank:") == 1
    assert body.count("  llama.cpp:") == 1
    assert body.count("  lm-studio:") == 1
    assert "discovery:" not in _block(body, "llama.cpp")


def test_example_yml_uses_current_omp_schema():
    root = Path(__file__).resolve().parents[1] / "examples"
    for name in (
        "omp-qwen38.yml",
        "omp-qwen38-mtp.yml",
        "omp-qwen35-4b.yml",
        "omp-iq3-sidecar.yml",
    ):
        text = (root / name).read_text()
        assert "type: anthropic" not in text, name
        assert "\n    base_url:" not in text and not text.startswith("base_url:"), name
        assert "baseUrl:" in text, name
        assert "auth: none" in text, name
        if name != "omp-iq3-sidecar.yml":
            assert "discovery:" not in _block(text, "llama.cpp"), name


def test_cli_omp_writes_yml(tmp_path, monkeypatch, capsys):
    dest = tmp_path / "models.yml"
    monkeypatch.setenv("SLOTBANK_OMP_YML", str(dest))
    from slotbank.cli import main

    rc = main([
        "omp", "--model", "Qwen3.8-27B-4bit", "--thinking", "--vision",
        "--port", "8080",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "llama.cpp/Qwen3.8-27B-4bit" in out
    assert "llama.cpp/Qwen3.8-27B-4bit-agent" in out
    assert "no tools" in out
    assert "--no-tools" in out
    assert "sb-hi" in out
    assert "two-stage" in out
    assert "condense" in out
    assert dest.is_file()
    text = dest.read_text()
    assert "  llama.cpp:" in text
    assert "  lm-studio:" in text
    assert "api: openai-completions" in text
    assert "api: anthropic-messages" in text
    assert "baseUrl: http://127.0.0.1:8080/v1" in text
    assert "discovery:" not in _block(text, "llama.cpp")
    assert "discovery:" not in _block(text, "lm-studio")


def test_apply_tuning_auto_attaches_sibling_mtp(tmp_path, monkeypatch):
    import argparse
    import os

    from slotbank.cli import _apply_tuning

    tgt = _checkpoint(tmp_path, "Qwen3.8-27B-4bit", {
        "vocab_size": 248320, "hidden_size": 5120,
    })
    mtp = _checkpoint(tmp_path, "Qwen3.8-27B-MTP-4bit", {
        "model_type": "qwen3_5_mtp",
        "vocab_size": 248320,
        "text_config": {"vocab_size": 248320, "hidden_size": 5120},
    })
    monkeypatch.delenv("SLOTBANK_DRAFT", raising=False)
    ns = argparse.Namespace(
        model=str(tgt), draft=None, no_draft=False, draft_kind=None,
        draft_block_size=None,
    )
    _apply_tuning(ns)
    assert os.environ["SLOTBANK_DRAFT"] == str(mtp)
    monkeypatch.delenv("SLOTBANK_DRAFT", raising=False)
    ns2 = argparse.Namespace(
        model=str(tgt), draft=None, no_draft=True, draft_kind=None,
        draft_block_size=None,
    )
    os.environ["SLOTBANK_DRAFT"] = str(mtp)
    _apply_tuning(ns2)
    assert "SLOTBANK_DRAFT" not in os.environ
