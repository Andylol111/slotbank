import pytest


def test_check_capacity_matches_loader_policy():
    """The checker must predict the C the loader actually picks.

    Two earlier versions used capacity_for_budget, which maximises C to fill
    the budget -- the full-residency configuration that thrashes. The loader's
    default path is slot_capacity, and a verdict computed from anything else
    is a verdict about a run that will not happen.
    """
    from slotbank.layout import MIN_KV_BYTES, slot_capacity

    stored, e, k, ws = 19 * (1 << 30), 256, 8, 16 * (1 << 30)
    c = slot_capacity(e, k, stored_bytes=stored, working_set_bytes=ws,
                      kv_bytes=MIN_KV_BYTES, expert_param_frac=0.889)
    assert k <= c <= 64, f"C={c} is outside the measured-sane band"


def test_encode_chat_falls_back_without_a_template(monkeypatch):
    """A base model has apply_chat_template but no template, and transformers
    raises ValueError rather than returning None. Every chat endpoint 500s
    without this fallback."""
    from slotbank.prompt import encode_chat

    monkeypatch.delenv("SLOTBANK_CONTEXT_DIR", raising=False)
    monkeypatch.delenv("SLOTBANK_DIRECT", raising=False)

    class Tok:
        def apply_chat_template(self, *a, **k):
            raise ValueError("chat_template is not set")

        def encode(self, text):
            return [len(text)]

    assert encode_chat(Tok(), [{"role": "user", "content": "hi"}], None) == [8]   # "user: hi"


def test_encode_chat_thinking_defaults_off(monkeypatch):
    from slotbank.prompt import encode_chat

    monkeypatch.delenv("SLOTBANK_CONTEXT_DIR", raising=False)
    monkeypatch.delenv("SLOTBANK_THINKING", raising=False)
    monkeypatch.delenv("SLOTBANK_DIRECT", raising=False)
    seen = {}

    class Tok:
        def apply_chat_template(self, msgs, **k):
            seen.update(k)
            return [1]

        def encode(self, text):
            return [0]

    encode_chat(Tok(), [{"role": "user", "content": "hi"}], None)
    assert seen.get("enable_thinking") is False


def test_encode_chat_direct_injects_system(monkeypatch):
    from slotbank.prompt import encode_chat

    monkeypatch.delenv("SLOTBANK_CONTEXT_DIR", raising=False)
    monkeypatch.setenv("SLOTBANK_DIRECT", "1")
    got = {}

    class Tok:
        def apply_chat_template(self, msgs, **k):
            got["msgs"] = msgs
            return [1]

        def encode(self, text):
            return [0]

    encode_chat(Tok(), [{"role": "user", "content": "hi"}], None)
    assert got["msgs"][0]["role"] == "system"
    assert "Do not refuse ordinary" in got["msgs"][0]["content"]
    assert got["msgs"][1]["content"] == "hi"
    monkeypatch.delenv("SLOTBANK_DIRECT", raising=False)


def test_encode_chat_direct_prepends_existing_system(monkeypatch):
    from slotbank.prompt import encode_chat, direct_system_text

    monkeypatch.delenv("SLOTBANK_CONTEXT_DIR", raising=False)
    monkeypatch.setenv("SLOTBANK_DIRECT", "1")
    got = {}

    class Tok:
        def apply_chat_template(self, msgs, **k):
            got["msgs"] = msgs
            return [1]

        def encode(self, text):
            return [0]

    encode_chat(Tok(), [
        {"role": "system", "content": "custom"},
        {"role": "user", "content": "hi"},
    ], None)
    assert got["msgs"][0]["content"].startswith(direct_system_text())
    assert "custom" in got["msgs"][0]["content"]
    monkeypatch.delenv("SLOTBANK_DIRECT", raising=False)


def test_encode_chat_unwraps_vlm_batch_encoding(monkeypatch):
    from slotbank.prompt import encode_chat

    monkeypatch.delenv("SLOTBANK_CONTEXT_DIR", raising=False)
    monkeypatch.delenv("SLOTBANK_DIRECT", raising=False)

    class Enc(dict):
        pass

    class Map:
        def get(self, key, default=None):
            return [7, 8, 9] if key == "input_ids" else default

        def __iter__(self):
            yield "input_ids"

    class Tok:
        def __init__(self, payload):
            self.payload = payload

        def apply_chat_template(self, msgs, **k):
            return self.payload

        def encode(self, text):
            return [0]

    assert encode_chat(Tok(Enc(input_ids=[7, 8, 9])), [{"role": "user", "content": "hi"}], None) == [7, 8, 9]
    assert encode_chat(Tok(Map()), [{"role": "user", "content": "hi"}], None) == [7, 8, 9]


def test_keep_token_ids_is_sink_pyramid_tail():
    """Paper analogue on token ids, not hybrid KV: sink + dense-early middle + tail.

    Order stays sequential (TriAttention consolidate). Tail is the current turn.
    Middle is denser near the sink (PyramidKV). One cut when over cap (BUZZ).
    """
    from slotbank.prompt import keep_token_ids

    ids = list(range(100))
    got = keep_token_ids(ids, 20)
    assert got == sorted(got)
    assert len(got) == 20
    assert got[0] == 0
    assert got[-1] == 99
    # tail is a contiguous suffix; head is a contiguous prefix
    assert got[:2] == [0, 1]
    assert 99 in got and 98 in got
    assert keep_token_ids(ids, 100) == ids
    assert keep_token_ids(ids, 0) == ids
    assert keep_token_ids([], 8) == []


def test_condense_keeps_ask_and_cites_the_dump():
    """Local stage: OMP may send the full harness; 27B gets the ask + citations."""
    from slotbank.prompt import condense_harness_messages

    dump = "file:src/foo.py:1-80\n" + ("LINE\n" * 4000)
    msgs = [
        {"role": "system", "content": "You are OMP. " + ("tools " * 2000)},
        {"role": "user", "content": dump + "\n\nhi"},
    ]
    got = condense_harness_messages(msgs, budget=400)
    blob = "\n".join(str(m.get("content") or "") for m in got)
    assert "hi" in blob
    assert "foo.py" in blob
    assert blob.count("LINE") < 40
    assert "tools " * 50 not in blob


def test_encode_chat_condenses_when_asked(monkeypatch):
    from slotbank.prompt import encode_chat

    monkeypatch.delenv("SLOTBANK_CONTEXT_DIR", raising=False)
    monkeypatch.delenv("SLOTBANK_DIRECT", raising=False)
    monkeypatch.delenv("SLOTBANK_PROMPT_PACK", raising=False)
    monkeypatch.setenv("SLOTBANK_CONDENSE", "1")
    monkeypatch.setenv("SLOTBANK_CONDENSE_BUDGET", "200")
    monkeypatch.setenv("SLOTBANK_MAX_PROMPT", "0")

    class Tok:
        def apply_chat_template(self, msgs, **k):
            text = "\n".join(str(m.get("content") or "") for m in msgs)
            return [ord(c) % 97 for c in text[:80]]

        def encode(self, text):
            return [1, 2, 3]

    dump = "file:src/bar.py:2-3\n" + ("DUMP\n" * 2000) + "\n\nwhat is 2+2"
    ids = encode_chat(Tok(), [{"role": "user", "content": dump}], None)
    assert ids, "condensed prompt must still tokenize"
    # Tok encodes the first 80 chars of the condensed messages; the ask survives
    # in the kept tail, so a raw 2000-line dump cannot be what was templated.
    assert len(ids) <= 80


def test_encode_chat_packs_overlong_when_asked(monkeypatch):
    """Opt-in: pack to the cap instead of 400. Off by default so dumps still refuse."""
    from slotbank.prompt import encode_chat

    monkeypatch.delenv("SLOTBANK_CONTEXT_DIR", raising=False)
    monkeypatch.delenv("SLOTBANK_DIRECT", raising=False)
    monkeypatch.delenv("SLOTBANK_CONDENSE", raising=False)
    monkeypatch.setenv("SLOTBANK_MAX_PROMPT", "8")
    monkeypatch.setenv("SLOTBANK_PROMPT_PACK", "1")

    class Tok:
        def apply_chat_template(self, msgs, **k):
            return list(range(32))

        def encode(self, text):
            return list(range(32))

    got = encode_chat(Tok(), [{"role": "user", "content": "hi"}], None)
    assert len(got) == 8
    assert got[0] == 0 and got[-1] == 31


def test_encode_chat_refuses_overlong_prompt(monkeypatch):
    """A 26k OMP cwd dump must 400 before 27B prefills (and jetsams) the Air."""
    from slotbank.prompt import encode_chat, encode_text

    monkeypatch.delenv("SLOTBANK_CONTEXT_DIR", raising=False)
    monkeypatch.delenv("SLOTBANK_DIRECT", raising=False)
    monkeypatch.delenv("SLOTBANK_PROMPT_PACK", raising=False)
    monkeypatch.delenv("SLOTBANK_CONDENSE", raising=False)
    monkeypatch.setenv("SLOTBANK_MAX_PROMPT", "8")

    class Tok:
        def apply_chat_template(self, msgs, **k):
            return list(range(32))

        def encode(self, text):
            return list(range(32))

    with pytest.raises(ValueError, match="prompt is 32 tokens \\(cap 8\\)"):
        encode_chat(Tok(), [{"role": "user", "content": "hi"}], None)
    with pytest.raises(ValueError, match="omp --no-tools"):
        encode_text(Tok(), "hi")


def test_encode_chat_cap_zero_disables(monkeypatch):
    from slotbank.prompt import encode_chat

    monkeypatch.delenv("SLOTBANK_CONTEXT_DIR", raising=False)
    monkeypatch.delenv("SLOTBANK_DIRECT", raising=False)
    monkeypatch.delenv("SLOTBANK_PROMPT_PACK", raising=False)
    monkeypatch.setenv("SLOTBANK_MAX_PROMPT", "0")

    class Tok:
        def apply_chat_template(self, msgs, **k):
            return list(range(20000))

        def encode(self, text):
            return list(range(20000))

    assert len(encode_chat(Tok(), [{"role": "user", "content": "hi"}], None)) == 20000


def test_encode_chat_default_cap_is_8k(monkeypatch):
    from slotbank.prompt import DEFAULT_MAX_PROMPT_TOKENS, encode_chat

    monkeypatch.delenv("SLOTBANK_CONTEXT_DIR", raising=False)
    monkeypatch.delenv("SLOTBANK_DIRECT", raising=False)
    monkeypatch.delenv("SLOTBANK_MAX_PROMPT", raising=False)
    monkeypatch.delenv("SLOTBANK_PROMPT_PACK", raising=False)

    class Tok:
        def __init__(self, n):
            self.n = n

        def apply_chat_template(self, msgs, **k):
            return list(range(self.n))

        def encode(self, text):
            return list(range(self.n))

    assert DEFAULT_MAX_PROMPT_TOKENS == 8192
    assert len(encode_chat(
        Tok(DEFAULT_MAX_PROMPT_TOKENS), [{"role": "user", "content": "hi"}], None,
    )) == DEFAULT_MAX_PROMPT_TOKENS
    with pytest.raises(ValueError, match="8193 tokens"):
        encode_chat(
            Tok(DEFAULT_MAX_PROMPT_TOKENS + 1),
            [{"role": "user", "content": "hi"}], None,
        )


def test_encode_chat_caps_plain_fallback(monkeypatch):
    from slotbank.prompt import encode_chat

    monkeypatch.delenv("SLOTBANK_CONTEXT_DIR", raising=False)
    monkeypatch.delenv("SLOTBANK_DIRECT", raising=False)
    monkeypatch.delenv("SLOTBANK_PROMPT_PACK", raising=False)
    monkeypatch.setenv("SLOTBANK_MAX_PROMPT", "4")

    class Tok:
        def apply_chat_template(self, *a, **k):
            raise ValueError("chat_template is not set")

        def encode(self, text):
            return list(range(10))

    with pytest.raises(ValueError, match="prompt is 10 tokens"):
        encode_chat(Tok(), [{"role": "user", "content": "hi"}], None)


def test_resolve_passes_through_explicit_ids(tmp_path):
    from slotbank.registry import resolve

    assert resolve("owner/repo") == "owner/repo"
    assert resolve(str(tmp_path)) == str(tmp_path)
    assert "/" in resolve("Some-Unknown-Model-4bit")


def test_resolve_never_picks_gguf(monkeypatch):
    """GGUF is llama.cpp's format. Resolving to one sends the user to a
    checkpoint this runtime cannot load."""
    import slotbank.registry as reg

    fake = [reg.LocalModel("unsloth/Qwen3-GGUF", 20 << 30, "/a", 1),
            reg.LocalModel("mlx-community/Qwen3-4bit", 19 << 30, "/b", 1)]
    monkeypatch.setattr(reg, "local_models", lambda mlx_only=True: fake)
    assert reg.resolve("Qwen3") == "mlx-community/Qwen3-4bit"


def test_runtime_close_drops_refs_and_skips_clear_cache():
    """Drop refs first. mx.clear_cache after a Metal-thread generate GIL-aborts
    on Python 3.13 (PyThreadState_Get), so close must not call it."""
    import inspect

    from slotbank.runtime import Runtime

    src = inspect.getsource(Runtime.close)
    assert "self._model = None" in src
    assert "gc.collect()" in src
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("clear_cache" in ln for ln in code)


def test_engine_close_tears_down_on_the_worker_thread():
    """Caller-thread runtime.close() after join was the SIGTRAP after `run hi`."""
    import inspect

    from slotbank.engine import Engine

    close = inspect.getsource(Engine.close)
    loop = inspect.getsource(Engine._loop)
    assert "self.runtime.close()" not in close
    assert "self.runtime.close()" in loop


def test_serve_help_lists_draft(capsys):
    from slotbank.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["serve", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--draft" in out and "DFlash" in out
    assert "--thinking" in out and "--vision" in out
    assert "--direct" in out
    assert "--no-omp" in out
    assert "--no-draft" in out


def test_tps_help_and_catalog(capsys, tmp_path, monkeypatch):
    from slotbank.cli import main

    monkeypatch.setenv("SLOTBANK_TPS_LOG", str(tmp_path / "tps.jsonl"))
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    help_out = capsys.readouterr().out
    assert "tps" in help_out
    assert "omp" in help_out
    rc = main(["tps"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sidecar-mtp-k3" in out
    assert "adopted" in out
    assert "mtp-plus-dflash" in out
    rc = main(["tps", "--log"])
    assert rc == 0
    log_out = capsys.readouterr().out
    assert "sidecar-mtp-k3" in log_out
    assert "13.47" in log_out


def test_apply_tuning_sets_draft_env(monkeypatch, tmp_path):
    import argparse
    import os

    from slotbank.cli import _apply_tuning

    monkeypatch.delenv("SLOTBANK_DRAFT", raising=False)
    monkeypatch.delenv("SLOTBANK_DRAFT_KIND", raising=False)
    monkeypatch.delenv("SLOTBANK_DRAFT_BLOCK", raising=False)
    _apply_tuning(argparse.Namespace(
        draft=str(tmp_path), draft_kind="dflash", draft_block_size=None,
    ))
    assert os.environ["SLOTBANK_DRAFT"] == str(tmp_path)
    assert os.environ["SLOTBANK_DRAFT_KIND"] == "dflash"
    assert "SLOTBANK_DRAFT_BLOCK" not in os.environ
    monkeypatch.delenv("SLOTBANK_DRAFT", raising=False)
    monkeypatch.delenv("SLOTBANK_DRAFT_KIND", raising=False)
    monkeypatch.delenv("SLOTBANK_DRAFT_BLOCK", raising=False)
    _apply_tuning(argparse.Namespace(
        draft=str(tmp_path), draft_kind=None, draft_block_size=8,
    ))
    assert os.environ["SLOTBANK_DRAFT_BLOCK"] == "8"
    monkeypatch.delenv("SLOTBANK_DRAFT", raising=False)
    monkeypatch.delenv("SLOTBANK_DRAFT_BLOCK", raising=False)


def test_apply_tuning_sets_thinking_and_vision(monkeypatch):
    import argparse
    import os

    from slotbank.cli import _apply_tuning

    monkeypatch.delenv("SLOTBANK_THINKING", raising=False)
    monkeypatch.delenv("SLOTBANK_VISION", raising=False)
    monkeypatch.delenv("SLOTBANK_DIRECT", raising=False)
    monkeypatch.delenv("SLOTBANK_CONDENSE", raising=False)
    _apply_tuning(argparse.Namespace(
        thinking=True, vision=True, direct=True, condense=True, draft=None,
    ))
    assert os.environ["SLOTBANK_THINKING"] == "1"
    assert os.environ["SLOTBANK_VISION"] == "1"
    assert os.environ["SLOTBANK_DIRECT"] == "1"
    assert os.environ["SLOTBANK_CONDENSE"] == "1"
    monkeypatch.delenv("SLOTBANK_THINKING", raising=False)
    monkeypatch.delenv("SLOTBANK_VISION", raising=False)
    monkeypatch.delenv("SLOTBANK_DIRECT", raising=False)
    monkeypatch.delenv("SLOTBANK_CONDENSE", raising=False)


def test_check_accepts_short_names_like_every_other_command():
    """check took a bare repo id while run/pull/serve resolved short names,
    so `slotbank check Qwen3.5-35B-A3B-4bit` 401'd on a guessed URL."""
    import inspect

    from slotbank.cli import _check

    assert "resolve(args.repo)" in inspect.getsource(_check)


def test_check_and_pull_share_the_suggestion_path(monkeypatch, capsys):
    """A short name resolves by guessing a namespace, so a miss is routine.
    Both entry points must offer real repos rather than a bare 404."""
    import slotbank.cli as cli

    monkeypatch.setattr("slotbank.registry.search",
                        lambda q, limit=8: ["mlx-community/Real-4bit", "mlx-community/guess"])
    cli._suggest("kimi", "mlx-community/guess")
    err = capsys.readouterr().err
    assert "Real-4bit" in err and "mlx-community/guess" not in err


# --------------------------------------------------------------------------
# download progress (docs/cli-design.md §6)
# --------------------------------------------------------------------------

def test_hub_progress_receives_the_name_kwarg():
    """The one real external-contract check, and it runs offline.

    `snapshot_download` discriminates its three bars only by the `name` kwarg,
    and `utils/tqdm.py::_create_progress_bar` passes `name` **only** to classes
    that subclass huggingface_hub's own tqdm -- anything else is called as
    `cls(**kwargs)`. Calling that dispatcher directly is what catches a hub
    upgrade changing the contract, which is the single external dependency in
    this design.
    """
    import logging

    from huggingface_hub.utils.tqdm import _create_progress_bar

    from slotbank.ui import hub_progress

    seen = []
    cls = hub_progress(lambda name, n, total: seen.append((name, n, total)))
    bar = _create_progress_bar(cls=cls, log_level=logging.INFO,
                               name="huggingface_hub.snapshot_download",
                               desc="x", total=100, initial=0, unit="B")
    bar.update(40)
    bar.close()
    assert seen, "no repaint reached the sink"
    assert seen[-1][0] == "huggingface_hub.snapshot_download"
    assert seen[-1][1] == 40 and seen[-1][2] == 100


def test_hub_progress_forces_disable_off():
    """Off a TTY the hub resolves `disable` to None, tqdm reads that as "off",
    and a disabled tqdm's update() returns *before* touching self.n.

    A shim that does not force disable=False therefore reports zero forever,
    silently -- a worse bug than the silent wait it was meant to fix.
    """
    from slotbank.ui import hub_progress

    seen = []
    cls = hub_progress(lambda name, n, total: seen.append(n))
    bar = cls(desc="x", total=100, disable=None, name=None)   # what HF passes
    assert bar.disable is False
    bar.update(10)
    bar.close()
    assert seen and max(seen) == 10


def test_plain_progress_has_no_escape_codes_and_is_throttled():
    """Rule `cli._status` already encodes: off a TTY, same numbers, no cursor
    motion. And one line per 10 s *and* 5% -- forty lines for a three-hour
    pull, not forty thousand."""
    import io

    from rich.console import Console

    from slotbank import ui

    def run(seconds_per_tick):
        buf = io.StringIO()
        err = Console(file=buf, force_terminal=False, width=100)
        plan = ui.Plan([_F("a.safetensors", 1 << 30)])
        state = ui._State(plan)
        view = ui._PlainView(state, err)
        for i in range(1, 1001):                   # 1000 repaints, 0.1% apart
            state.written = (1 << 30) * i / 1000
            state.t0 -= seconds_per_tick
            view.tick()
        return buf.getvalue()

    quick = run(0.02)                              # a 20 s pull: 10 s is rarer
    assert "\x1b" not in quick, "escape codes off a TTY"
    assert 1 <= quick.count("pull:") <= 4, quick.count("pull:")

    slow = run(20.0)                               # a 5.5 h pull: 5% is rarer
    assert 15 <= slow.count("pull:") <= 22, slow.count("pull:")


def test_download_view_never_paints_over_a_pipe():
    import io

    from rich.console import Console

    from slotbank import ui

    buf = io.StringIO()
    err = Console(file=buf, force_terminal=False, width=100)
    plan = ui.Plan([_F("a.safetensors", 1 << 30)])
    with ui.download_view(plan, err=err) as view:
        view.on_bar(ui.BAR_WRITE, 1 << 29, 1 << 30)
    assert "\x1b" not in buf.getvalue()


def test_pull_keeps_the_check_exit_code():
    """`pull` refusing a model must return what `check` returned, not 2. The
    refusal is the feature; collapsing it into a generic error loses which
    number was binding."""
    import argparse
    import inspect

    import slotbank.cli as cli

    src = inspect.getsource(cli._pull)
    assert "return rc" in src
    # _check reads args.ram; the Namespace _pull builds must carry it or the
    # refusal path raises AttributeError instead of refusing.
    ns = argparse.Namespace(repo="x", revision="main", leave_free=None, ram=None)
    assert "ram=None" in src and hasattr(ns, "ram")


def test_pull_total_comes_from_the_manifest_not_the_hub_bar():
    """The hub's reconstruct total starts at 0 and grows as each file's bar is
    built ("Reconstructing (incomplete total...)"), so an early percentage read
    from it is wrong. The dry-run manifest knows the real total."""
    from slotbank import ui

    plan = ui.Plan([_F("a", 4 << 30), _F("b", 2 << 30, cached=True)])
    state = ui._State(plan)
    assert state.total == (4 << 30)                # cached bytes are not fetched
    state.on_bar(ui.BAR_WRITE, 1 << 30, 0)         # hub reports total=0 early
    assert state.total == (4 << 30)


class _F:
    """Stand-in for huggingface_hub's DryRunFileInfo."""

    def __init__(self, filename, file_size, cached=False):
        self.filename, self.file_size = filename, file_size
        self.is_cached, self.will_download = cached, not cached


def _card(**kw):
    from slotbank.probe import RemoteCard

    base = dict(repo="o/M-4bit", total_bytes=16 << 30, expert_bytes=int(0.949 * (16 << 30)),
                resident_bytes=int(0.051 * (16 << 30)), num_experts=128, top_k=8,
                layers=48, row_bytes=2 << 20, shards=4, scanned_layers=48,
                model_type="qwen3_moe")
    base.update(kw)
    return RemoteCard(**base)


def test_scale_reproduces_the_probe_it_came_from():
    """The known-answer case for the whole compare screen. The derived row for
    the quant that WAS probed must reproduce the probe field for field -- if it
    does not, every other row is wrong by the same factor."""
    from slotbank.probe import scale

    c = _card()
    d = scale(c, c.repo, c.total_bytes, 4.0)
    for f in ("total_bytes", "num_experts", "top_k", "layers", "row_bytes"):
        assert getattr(d, f) == getattr(c, f), f
    assert abs(d.expert_frac - c.expert_frac) < 1e-9
    assert d.touched_bytes == c.touched_bytes


def test_scale_moves_row_bytes_with_bits_not_expert_share():
    """Bits per weight scale byte totals; they do not change the architecture."""
    from slotbank.probe import scale

    c = _card()
    eight = scale(c, "o/M-8bit", c.total_bytes * 2, 8.0)
    assert eight.row_bytes == c.row_bytes * 2
    assert eight.num_experts == c.num_experts and eight.top_k == c.top_k
    assert abs(eight.expert_frac - c.expert_frac) < 1e-9


def test_roles_roundtrip_and_resolve(tmp_path, monkeypatch):
    import slotbank.registry as reg

    monkeypatch.setattr(reg, "ROLES_PATH", str(tmp_path / "roles.json"))
    reg.save_role("chat", "owner/Model-4bit", "medium")
    assert reg.load_roles()["chat"] == {"model": "owner/Model-4bit", "effort": "medium"}
    assert reg.resolve("@chat") == "owner/Model-4bit"
    # effort alone must not wipe the model
    reg.save_role("chat", None, "high")
    assert reg.load_roles()["chat"]["model"] == "owner/Model-4bit"
    with pytest.raises(ValueError):
        reg.resolve("@missing")


def test_save_role_rejects_a_role_with_no_model(tmp_path, monkeypatch):
    import slotbank.registry as reg

    monkeypatch.setattr(reg, "ROLES_PATH", str(tmp_path / "roles.json"))
    with pytest.raises(ValueError):
        reg.save_role("chat", None, "high")
