from __future__ import annotations

import json


def test_append_is_append_only(tmp_path):
    from slotbank.context_os import append, iter_log

    append(tmp_path, "user", "one")
    append(tmp_path, "assistant", "two", pointers=["file:src/a.py:1-2"])
    recs = list(iter_log(tmp_path))
    assert [r["seq"] for r in recs] == [1, 2]
    assert recs[0]["content"] == "one"
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == 2
    json.loads(lines[0])


def test_compile_is_verbatim_newest_first(tmp_path):
    from slotbank.context_os import append, compile_working_set

    append(tmp_path, "user", "old " * 200)
    append(tmp_path, "user", "KEEP-ME")
    text = compile_working_set(tmp_path, budget=40)
    assert "KEEP-ME" in text
    assert "[log:2 user]" in text
    assert "old " not in text
    assert "paraphrase" not in text.lower()


def test_expand_cap_cites_oversized_file_without_inlining(tmp_path, monkeypatch):
    """Local implement: keep the file on disk; only inline spans that fit."""
    from slotbank.context_os import append, compile_working_set

    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    (src / "big.py").write_text("LINE\n" * 80)
    append(tmp_path, "user", "fix this", pointers=["file:src/big.py:1-80"])
    monkeypatch.setenv("SLOTBANK_CONTEXT_EXPAND", "20")
    text = compile_working_set(tmp_path, repo=repo, budget=200)
    assert "file:src/big.py:1-80" in text
    assert text.count("LINE") < 10


def test_expand_zero_is_pointers_only(tmp_path, monkeypatch):
    from slotbank.context_os import append, compile_working_set

    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    (src / "foo.py").write_text("alpha\nbeta\n")
    append(tmp_path, "user", "look", pointers=["file:src/foo.py:1-2"])
    monkeypatch.setenv("SLOTBANK_CONTEXT_EXPAND", "0")
    text = compile_working_set(tmp_path, repo=repo)
    assert "file:src/foo.py:1-2" in text
    assert "alpha" not in text and "beta" not in text


def test_file_pointer_expands_verbatim_and_rejects_escape(tmp_path):
    from slotbank.context_os import append, compile_working_set

    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    (src / "foo.py").write_text("alpha\nbeta\ngamma\n")
    append(tmp_path, "user", "look", pointers=["file:src/foo.py:2-3", "file:../secret:1"])
    text = compile_working_set(tmp_path, repo=repo)
    assert "beta" in text and "gamma" in text
    assert "alpha" not in text
    assert "[file:src/foo.py:2-3]" in text
    assert "../secret" in text and "secret contents" not in text


def test_cloud_compiler_then_local_fallback(tmp_path, monkeypatch):
    from slotbank import context_os

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"working_set": "CLOUD-EXCERPT file:a.py:1"}).encode()

    monkeypatch.setattr(context_os.urllib.request, "urlopen", lambda *a, **k: _Resp())
    context_os.append(tmp_path, "user", "ignored-locally")
    text = context_os.compile_working_set(
        tmp_path, compiler_url="http://compiler.test/compile"
    )
    assert text == "CLOUD-EXCERPT file:a.py:1"

    def boom(*a, **k):
        raise context_os.urllib.error.URLError("offline")

    monkeypatch.setattr(context_os.urllib.request, "urlopen", boom)
    text = context_os.compile_working_set(
        tmp_path, compiler_url="http://compiler.test/compile"
    )
    assert "ignored-locally" in text


def test_prompt_injects_compiled_prefix(tmp_path, monkeypatch):
    from slotbank.context_os import append
    from slotbank.prompt import with_context_os

    append(tmp_path, "user", "verbatim-span")
    monkeypatch.delenv("SLOTBANK_ENVELOPE", raising=False)
    monkeypatch.delenv("SLOTBANK_CONTEXT_INJECT", raising=False)
    monkeypatch.setenv("SLOTBANK_CONTEXT_DIR", str(tmp_path))
    msgs = with_context_os([{"role": "user", "content": "now"}])
    assert msgs[0]["role"] == "system"
    assert "verbatim-span" in msgs[0]["content"]
    assert "now" == msgs[1]["content"]
    # Sidecar contract: same list is valid OpenAI chat JSON. No tokenizer, no MLX.
    payload = json.dumps({"messages": msgs, "temperature": 0})
    assert "verbatim-span" in payload and "now" in payload


def test_envelope_does_not_inject_logged_dump(tmp_path, monkeypatch):
    """Serve CONTEXT_DIR is a log. Compiling it back busts PrefixCache."""
    from slotbank.context_os import append
    from slotbank.prompt import with_context_os

    append(tmp_path, "user", "verbatim-span")
    monkeypatch.setenv("SLOTBANK_CONTEXT_DIR", str(tmp_path))
    monkeypatch.setenv("SLOTBANK_ENVELOPE", "1")
    monkeypatch.delenv("SLOTBANK_CONTEXT_INJECT", raising=False)
    msgs = with_context_os([{"role": "user", "content": "now"}])
    assert msgs == [{"role": "user", "content": "now"}]
    monkeypatch.setenv("SLOTBANK_CONTEXT_INJECT", "1")
    injected = with_context_os([{"role": "user", "content": "now"}])
    assert injected[0]["role"] == "system"
    assert "verbatim-span" in injected[0]["content"]


def test_encode_chat_drops_thinking_kwarg(monkeypatch):
    from slotbank.prompt import encode_chat

    monkeypatch.delenv("SLOTBANK_CONTEXT_DIR", raising=False)
    monkeypatch.delenv("SLOTBANK_THINKING", raising=False)
    monkeypatch.delenv("SLOTBANK_DIRECT", raising=False)

    class Tok:
        def apply_chat_template(self, msgs, **k):
            if "enable_thinking" in k:
                raise TypeError("no thinking")
            return [9]

        def encode(self, text):
            return [0]

    assert encode_chat(Tok(), [{"role": "user", "content": "hi"}], None) == [9]


def test_context_cli(tmp_path):
    from slotbank.cli import main

    assert main(["context", "init", "--dir", str(tmp_path)]) == 0
    assert main([
        "context", "append", "--dir", str(tmp_path),
        "--role", "user", "--content", "hello",
        "--pointer", "file:x.py:1",
    ]) == 0
    rc = main(["context", "compile", "--dir", str(tmp_path), "--budget", "200"])
    assert rc == 0
    assert "hello" in (tmp_path / "working_set.txt").read_text()
