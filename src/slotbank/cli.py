from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

from slotbank.engine import Engine, leave_free_arg
from slotbank.types import SamplingParams


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="slotbank",
        description="Run MoE models whose expert bank does not fit in memory.",
        epilog=(
            "start here:\n"
            "  slotbank list                     what is already downloaded\n"
            "  slotbank check <model>            will it run here (no download)\n"
            "  slotbank pull <model>             download it\n"
            "  slotbank run <model>              chat with it\n"
            "  slotbank serve --model <model>    serve it to Claude Code / Codex / OMP\n"
            "  slotbank omp --model <model>      list that server in Oh My Pi\n"
            "  slotbank context                  session log + compiled working set\n"
            "  slotbank tps                      what was tried for 27B tok/s\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Not required: a bare `slotbank` should show what to do next, the way
    # every other model runner does, not an argparse error.
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("generate", help="one-shot completion")
    _model_args(g)
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=128)
    g.add_argument("--temp", type=float, default=0.0)
    g.add_argument("--top-p", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=-1)
    g.add_argument("--quiet", action="store_true",
                   help="suppress progress and the stats line on stderr")
    _tuning_args(g)

    s = sub.add_parser("serve", help="Chat / Claude / Codex HTTP")
    s.add_argument("--model", required=True,
                   help="short name, repo id, or local folder")
    s.add_argument("--leave-free", default=None, help="RAM to keep for macOS, e.g. 8g")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--api-key", default=None)
    s.add_argument(
        "--no-omp",
        action="store_true",
        help="do not write ~/.omp/agent/models.yml",
    )
    _tuning_args(s)

    om = sub.add_parser(
        "omp",
        help="write ~/.omp/agent/models.yml so Oh My Pi lists this server",
    )
    om.add_argument("--model", required=True, help="short name, repo id, or local folder")
    om.add_argument("--host", default="127.0.0.1")
    om.add_argument("--port", type=int, default=8080)
    om.add_argument("--thinking", action="store_true")
    om.add_argument("--vision", action="store_true")

    r = sub.add_parser("run", help="chat with a model, keeping it loaded")
    r.add_argument("model", help="short name, repo id, or local folder")
    r.add_argument("prompt", nargs="*", help="one-shot prompt; omit for a chat session")
    r.add_argument("--leave-free", default=None, help="RAM to keep for macOS, e.g. 8g")
    r.add_argument("--max-tokens", type=int, default=1024)
    r.add_argument("--temp", type=float, default=0.0)
    r.add_argument("--top-p", type=float, default=1.0)
    r.add_argument("--top-k", type=int, default=0)
    r.add_argument("--quiet", action="store_true")
    _tuning_args(r)

    u = sub.add_parser("use", help="set which model answers as a role")
    u.add_argument("role", nargs="?", help="e.g. chat, reasoning")
    u.add_argument("model", nargs="?", help="short name or repo id")
    u.add_argument("--effort", choices=sorted(EFFORT), default=None)

    cm = sub.add_parser("compare", help="compare every quant of a model, one probe")
    cm.add_argument("model")
    cm.add_argument("--revision", default="main")
    cm.add_argument("--leave-free", default=None)
    cm.add_argument("--ram", default=None, help="model a machine you do not own")

    ch = sub.add_parser("chat", help="interactive session (same as bare slotbank)")
    ch.add_argument("model", nargs="?")
    ch.add_argument("--leave-free", default=None)

    sr = sub.add_parser("search", help="find models on the Hub by name")
    sr.add_argument("query", nargs="+")
    sr.add_argument("--limit", type=int, default=12)

    ls = sub.add_parser("list", help="cached models and what each would cost here")
    ls.add_argument("--all", action="store_true", help="include non-MLX repos")

    pl = sub.add_parser("pull", help="download a model, refusing one that cannot run")
    pl.add_argument("model")
    pl.add_argument("--revision", default="main")
    pl.add_argument("--force", action="store_true", help="download even if it will not fit")
    pl.add_argument("--quiet", action="store_true",
                    help="suppress the progress region on stderr")

    rm = sub.add_parser("rm", help="delete a cached model")
    rm.add_argument("model")
    rm.add_argument("-y", "--yes", action="store_true")

    c = sub.add_parser("check", help="inspect a remote model without downloading it")
    c.add_argument("repo", help="Hugging Face repo id, e.g. mlx-community/Qwen3.5-35B-A3B-4bit")
    c.add_argument("--revision", default="main")
    c.add_argument("--leave-free", default=None, help="RAM to keep for macOS, e.g. 8g")
    c.add_argument("--ram", default=None,
                   help="model a machine you do not own, e.g. 128g")

    a = sub.add_parser("admit", help="print the memory card and refuse if it does not fit")
    _model_args(a)
    _draft_args(a)

    cx = sub.add_parser("context", help="session log + compiled working set")
    cx_sub = cx.add_subparsers(dest="context_cmd")
    cxi = cx_sub.add_parser("init", help="create an append-only log directory")
    cxi.add_argument("--dir", required=True)
    cxa = cx_sub.add_parser("append", help="append one message; never rewrites history")
    cxa.add_argument("--dir", required=True)
    cxa.add_argument("--role", required=True)
    cxa.add_argument("--content", required=True)
    cxa.add_argument("--pointer", action="append", default=[],
                     help="verbatim pointer, e.g. file:src/foo.py:10-40")
    cxc = cx_sub.add_parser("compile", help="print a verbatim working set from the log")
    cxc.add_argument("--dir", required=True)
    cxc.add_argument("--repo", default=None, help="repo root for file: pointers")
    cxc.add_argument("--budget", type=int, default=None, help="token budget (default 4096)")

    tps = sub.add_parser(
        "tps",
        help="speculative-decode attempt catalog (what was tried, what is daily)",
    )
    tps.add_argument(
        "--log",
        action="store_true",
        help="print the machine-local JSONL instead of the catalog",
    )

    args = p.parse_args(argv)
    if not args.cmd:
        # A bare `slotbank` enters the interactive shell when there is a
        # terminal to drive it; piping still gets the plain command listing.
        if sys.stdin.isatty() and sys.stdout.isatty():
            from slotbank.shell import Shell

            return Shell().run()
        return _home(p)
    _apply_tuning(args)
    try:
        if args.cmd == "run":
            return _run(args)
        if args.cmd == "chat":
            from slotbank.shell import Shell

            return Shell(args.model, leave_free_arg(args.leave_free)).run()
        if args.cmd == "use":
            return _use(args)
        if args.cmd == "compare":
            return _compare(args)
        if args.cmd == "search":
            return _search(args)
        if args.cmd == "list":
            return _list(args)
        if args.cmd == "pull":
            return _pull(args)
        if args.cmd == "rm":
            return _rm(args)
        if args.cmd == "check":
            return _check(args)
        if args.cmd == "admit":
            return _admit(args)
        if args.cmd == "context":
            return _context(args)
        if args.cmd == "tps":
            return _tps(args)
        if args.cmd == "omp":
            return _omp(args)
        if args.cmd == "generate":
            return _generate(args)
        return _serve(args)
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        return 130
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        # refusing to load, a missing model or an unreadable checkpoint are
        # ordinary outcomes, not crashes -- a traceback here is noise
        sys.stderr.write(f"slotbank: {exc}\n")
        return 2


# Effort presets. Each value is the one that measured best for that intent on
# a machine where the expert bank exceeds the page cache; see docs/DECISIONS.md.
# Note that "high" does NOT raise the slot count: past the policy's choice, more
# slots measured *slower* here, because the pack is taken from the page cache
# that serves its own misses (C=32 -> 8.6 tok/s, C=64 -> 6.5).
EFFORT = {
    "low": {
        # leave the machine alone: smallest footprint, no warm pass, small
        # prefill chunks so the transient peak stays low
        "SLOTBANK_BUDGET_GIB": "2",
        "SLOTBANK_WARM": "0",
        "SLOTBANK_PREFILL_STEP": "512",
    },
    "medium": {
        # the measured optimum; the capacity policy picks C for the machine
        "SLOTBANK_WARM_MIN_TOKENS": "128",
    },
    "high": {
        # dedicated machine: warm eagerly and take the full prefill chunk
        "SLOTBANK_WARM": "1",
        "SLOTBANK_WARM_MIN_TOKENS": "0",
        "SLOTBANK_PREFILL_STEP": "4096",
    },
}
EFFORT_HELP = {
    "low": "smallest footprint, no warm pass - for running alongside other work",
    "medium": "default; the capacity policy picks the measured optimum",
    "high": "warm eagerly, larger prefill chunks - for a dedicated machine",
}


def _tuning_args(p: argparse.ArgumentParser) -> None:
    """Knobs that were reachable only through the environment.

    Each maps to the SLOTBANK_* variable of the same name; the flag wins when
    both are set, so scripts that already export the variables keep working.
    """
    p.add_argument("--effort", choices=("low", "medium", "high"), default=None,
                   help="; ".join(f"{k}: {v}" for k, v in EFFORT_HELP.items()))
    p.add_argument("--budget-gib", type=float, default=None,
                   help="cap resident expert memory; capacity is solved from it")
    p.add_argument("--slots", type=int, default=None,
                   help="expert slots per layer (overrides the capacity policy)")
    p.add_argument("--read-threads", type=int, default=None,
                   help="threads for miss reads (default 8; 0 uses the mmap path)")
    p.add_argument("--prefill-step", type=int, default=None,
                   help="tokens per prefill chunk; lower caps the memory peak")
    p.add_argument("--warm-min-tokens", type=int, default=None,
                   help="tokens the hot-expert warm pass must pay back (default 128)")
    p.add_argument("--no-warm", action="store_true",
                   help="skip the hot-expert warm pass entirely")
    p.add_argument("--vision", action="store_true",
                   help="load the vision tower (off by default; saves ~0.4 GiB)")
    p.add_argument("--thinking", action="store_true",
                   help="enable the chat template thinking block (off by default)")
    p.add_argument("--direct", action="store_true",
                   help="inject a short no-lecture system prefix (does not change weights)")
    _draft_args(p)


def _draft_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--draft",
        default=None,
        help="drafter path (DFlash or MTP; mlx-vlm verify; trained block_size)",
    )
    p.add_argument(
        "--draft-kind",
        default=None,
        choices=("dflash", "mtp", "eagle3"),
        help="drafter family (default: auto from the drafter config)",
    )
    p.add_argument(
        "--draft-block-size",
        type=int,
        default=None,
        help="verify block size (default: the drafter's trained block_size)",
    )
    p.add_argument(
        "--no-draft",
        action="store_true",
        help="do not auto-attach a sibling MTP/DFlash folder",
    )


_ENV_FOR = {
    "budget_gib": "SLOTBANK_BUDGET_GIB",
    "read_threads": "SLOTBANK_READ_THREADS",
    "prefill_step": "SLOTBANK_PREFILL_STEP",
    "warm_min_tokens": "SLOTBANK_WARM_MIN_TOKENS",
}


def _apply_tuning(args) -> None:
    """Precedence: explicit flag > effort preset > environment > default.

    A preset is a starting point, not a straitjacket, so a flag set alongside
    it still wins. An environment variable the caller exported themselves also
    survives unless the preset is asked for explicitly.
    """
    import os

    effort = getattr(args, "effort", None)
    if effort:
        for env, val in EFFORT.get(effort, {}).items():
            os.environ[env] = val

    for attr, env in _ENV_FOR.items():
        val = getattr(args, attr, None)
        if val is not None:
            os.environ[env] = str(val)
    if getattr(args, "no_warm", False):
        os.environ["SLOTBANK_WARM"] = "0"
    if getattr(args, "slots", None) is not None:
        os.environ["SLOTBANK_SLOTS_OVERRIDE"] = str(args.slots)
    if getattr(args, "vision", False):
        os.environ["SLOTBANK_VISION"] = "1"
    if getattr(args, "thinking", False):
        os.environ["SLOTBANK_THINKING"] = "1"
    if getattr(args, "direct", False):
        os.environ["SLOTBANK_DIRECT"] = "1"
    draft = getattr(args, "draft", None)
    no_draft = getattr(args, "no_draft", False) and not draft
    if no_draft:
        os.environ.pop("SLOTBANK_DRAFT", None)
        os.environ.pop("SLOTBANK_DRAFT_KIND", None)
        os.environ.pop("SLOTBANK_DRAFT_BLOCK", None)
    elif draft:
        os.environ["SLOTBANK_DRAFT"] = os.path.expanduser(draft)
        kind = getattr(args, "draft_kind", None)
        if kind:
            os.environ["SLOTBANK_DRAFT_KIND"] = kind
        block = getattr(args, "draft_block_size", None)
        if block is not None:
            os.environ["SLOTBANK_DRAFT_BLOCK"] = str(block)
    elif not os.environ.get("SLOTBANK_DRAFT", "").strip():
        found = _auto_draft_path(getattr(args, "model", None))
        if found:
            os.environ["SLOTBANK_DRAFT"] = found
            args.draft = found


def _auto_draft_path(model: str | None) -> str | None:
    """Sibling MTP/DFlash next to a local checkpoint, or None."""
    if not model:
        return None
    try:
        from slotbank.admit import discover_sidecar_draft

        expanded = os.path.expanduser(str(model))
        if os.path.isdir(expanded):
            return discover_sidecar_draft(expanded)
        from slotbank.registry import local_path, resolve

        target = resolve(str(model))
        path = local_path(target)
        if not path:
            return None
        return discover_sidecar_draft(path)
    except (ValueError, OSError, ImportError):
        return None


def _model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="local mlx-lm folder or Hugging Face id")
    p.add_argument("--leave-free", default=None, help="RAM to keep for macOS, e.g. 8g")


def _status(quiet: bool):
    """Progress and stats go to stderr so stdout stays a clean completion.

    Loading and prefill take ~8 s before the first token on a large model. With
    no output in that window the tool reads as hung rather than busy.
    """
    if quiet:
        # same signature as the live emitter, or --quiet raises TypeError
        return lambda msg="", end=False: None

    tty = sys.stderr.isatty()

    def emit(msg: str = "", end: bool = False) -> None:
        if tty:
            # rewrite one line in place; without a terminal these become
            # literal "[K" in the output, so plain lines are used instead
            sys.stderr.write("\r\033[K" + msg + ("\n" if end else ""))
        elif msg:
            sys.stderr.write(msg + "\n")
        sys.stderr.flush()

    return emit




def _memory_note(engine) -> str:
    """Report the signal that actually predicts throughput.

    Page-cache residency of the expert bank correlates with decode speed at
    r = -0.866 and swings it 2.4x -- more than any flag here. Free memory is
    deliberately not reported: macOS drives it to ~0 by design, so it carries
    no information about whether the machine is tight.
    """
    um = getattr(engine, "um", None)
    if um is None:
        return ""
    try:
        snap = um.snapshot()
    except (OSError, ValueError):
        return ""
    G = float(1 << 30)
    note = (f" - cache {snap.file_backed_bytes / G:.1f} GiB"
            f", reclaimable {snap.reclaimable_bytes / G:.1f} GiB")
    if snap.pressure >= 2:
        note += f" - MEMORY PRESSURE {snap.pressure}"
    return note

def _generate(args) -> int:
    say = _status(args.quiet)
    t0 = time.perf_counter()
    say("loading model...")

    engine = Engine(args.model, leave_free=leave_free_arg(args.leave_free))
    try:
        ids = engine.tokenize_text(args.prompt)
        t_load = time.perf_counter()
        say(f"loaded in {t_load - t0:.1f}s - prefilling {len(ids)} tokens...")
        sampling = SamplingParams(
            temperature=args.temp,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
        )
        seen = {"n": 0, "first": None}

        def on_token(_tid: int, piece: str) -> None:
            if seen["first"] is None:
                seen["first"] = time.perf_counter()
                say()                      # clear the status line
            seen["n"] += 1
            sys.stdout.write(piece)
            sys.stdout.flush()

        result = engine.generate(ids, sampling, on_token=on_token)
        if not result.content.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
        end = time.perf_counter()
        if seen["first"] is not None and seen["n"] > 1:
            decode = (seen["n"] - 1) / max(end - seen["first"], 1e-9)
            say(f"{seen['n']} tokens - {decode:.2f} tok/s decode - "
                f"first token {seen['first'] - t0:.1f}s - "
                f"{end - t0:.1f}s total{_memory_note(engine)}", end=True)
        return 0
    finally:
        engine.close()


def _omp(args) -> int:
    from slotbank.admit import public_model_id
    from slotbank.omp import selector, upsert

    path = None
    try:
        from slotbank.registry import local_path, resolve

        path = local_path(resolve(args.model))
    except (ValueError, ImportError, OSError):
        path = None
    mid = public_model_id(path or args.model)
    written = upsert(
        model_id=mid,
        host=getattr(args, "host", "127.0.0.1"),
        port=int(getattr(args, "port", 8080)),
        thinking=bool(getattr(args, "thinking", False)),
        vision=bool(getattr(args, "vision", False)),
    )
    print(f"wrote {written}")
    print(f"picker: {selector(mid)}")
    print("refresh: omp models slotbank")
    return 0


def _serve(args) -> int:
    import uvicorn

    from slotbank.admit import public_model_id
    from slotbank.api.app import create_app
    from slotbank.omp import selector, upsert
    from slotbank.registry import local_path, resolve

    repo = resolve(args.model)
    path = local_path(repo)
    if path is None:
        sys.stderr.write(
            f"slotbank: {repo} is not downloaded. Run: slotbank pull {args.model}\n")
        return 2
    _apply_tuning(args)
    mid = public_model_id(path)
    omp_path = None
    if not getattr(args, "no_omp", False):
        omp_path = upsert(
            model_id=mid,
            host=args.host,
            port=args.port,
            thinking=bool(getattr(args, "thinking", False)),
            vision=bool(getattr(args, "vision", False)),
        )
    draft = os.environ.get("SLOTBANK_DRAFT", "").strip()
    extra = f"\n  draft {draft} (mlx-vlm verify; 27B tokens unchanged)" if draft else ""
    if not draft:
        extra += (
            "\n  warning: no MTP/DFlash sidecar; this is greedy ~5.7 tok/s on 27B. "
            "Put Qwen3.8-27B-MTP-4bit next to the model, or pass --draft"
        )
    omp_lines = ""
    if omp_path is not None:
        omp_lines = (
            f"\n  OMP picker                 {selector(mid)}\n"
            f"  OMP models.yml             {omp_path}\n"
            f"  refresh                    omp models slotbank"
        )
    print(f"slotbank serving {mid} on http://{args.host}:{args.port}\n"
          f"  OpenAI / Codex / OpenCode  "
          f"OPENAI_BASE_URL=http://{args.host}:{args.port}/v1\n"
          f"  Claude Code / OMP          "
          f"ANTHROPIC_BASE_URL=http://{args.host}:{args.port}"
          f"{omp_lines}"
          f"{extra}", flush=True)
    engine = Engine(path, leave_free=leave_free_arg(args.leave_free), model_id=mid)
    app = create_app(engine, api_key=args.api_key)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    engine.close()
    return 0


def _run(args) -> int:
    """Chat with a model, loading it once.

    The point of holding the session open is that load plus first prefill costs
    ~19 s on a 19 GiB model. Paying that per message, as a one-shot CLI does,
    dominates everything the runtime saves.
    """
    from slotbank.registry import local_path, resolve

    repo = resolve(args.model)
    # Engine reads the safetensors to build its memory card, so it needs the
    # snapshot directory -- a bare repo id has no files to measure.
    path = local_path(repo)
    if path is None:
        sys.stderr.write(
            f"slotbank: {repo} is not downloaded. Run: slotbank pull {args.model}\n")
        return 2

    _apply_tuning(args)
    say = _status(args.quiet)
    t0 = time.perf_counter()
    say("loading model...")
    from slotbank.admit import public_model_id

    engine = Engine(
        path, leave_free=leave_free_arg(args.leave_free),
        model_id=public_model_id(path),
    )
    sampling = SamplingParams(temperature=args.temp, top_p=args.top_p,
                              top_k=args.top_k, max_tokens=args.max_tokens)
    try:
        say(f"loaded in {time.perf_counter() - t0:.1f}s{_memory_note(engine)}", end=True)

        def answer(messages) -> str:
            ids = engine.tokenize_chat(messages, None)
            first = {"t": None, "n": 0}

            def on_token(_tid, piece):
                if first["t"] is None:
                    first["t"] = time.perf_counter()
                first["n"] += 1
                sys.stdout.write(piece)
                sys.stdout.flush()

            start = time.perf_counter()
            out = engine.generate(ids, sampling, on_token=on_token)
            sys.stdout.write("\n")
            if first["t"] and first["n"] > 1:
                rate = (first["n"] - 1) / max(time.perf_counter() - first["t"], 1e-9)
                say(f"{first['n']} tokens - {rate:.2f} tok/s decode - "
                    f"first token {first['t'] - start:.1f}s", end=True)
            return out.content

        if args.prompt:
            answer([{"role": "user", "content": " ".join(args.prompt)}])
            return 0

        history: list[dict] = []
        say("chat session - /bye to exit, /clear to reset the context", end=True)
        while True:
            try:
                line = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                sys.stdout.write("\n")
                return 0
            if not line:
                continue
            if line in ("/bye", "/exit", "/quit"):
                return 0
            if line == "/clear":
                history = []
                say("context cleared", end=True)
                continue
            history.append({"role": "user", "content": line})
            history.append({"role": "assistant", "content": answer(history)})
    finally:
        engine.close()


def _list(args) -> int:
    from slotbank.admit import expert_frac_from_files
    from slotbank.registry import local_models

    G = float(1 << 30)
    models = local_models(mlx_only=not args.all)
    if not models:
        print("no models cached. Try: slotbank pull Qwen3.5-35B-A3B-4bit")
        return 0
    width = max([len(m.repo_id) for m in models] + [20]) + 2
    from slotbank.layout import detect_device_profile

    budget = detect_device_profile(leave_free_bytes=None).max_working_set_bytes
    rows = []
    for m in models:
        try:
            frac = expert_frac_from_files(m.path)
        except Exception:
            frac = None
        state, wired = _role_state(m.repo_id, budget)
        rows.append((m.repo_id.split("/")[-1], f"{m.size_bytes / G:.1f}G",
                     f"{frac:.0%}" if frac else "-",
                     f"{wired:.1f}G" if wired else "-", state))
    try:
        from slotbank import ui

        if ui.bars_disabled():
            raise ImportError
        ui.model_table(rows)
    except Exception:
        print(f"{'MODEL':<{width}}{'SIZE':>10}{'EXPERTS':>10}")
        for name, on_disk, tag, _w, _s in rows:
            print(f"{name:<{width}}{on_disk:>10}{tag:>10}")
    total = sum(m.size_bytes for m in models)
    print(f"\n{len(models)} models, {total / G:.1f} GiB on disk")
    return 0


def _pull(args) -> int:
    """Download, but check first, and never go silent while it runs.

    Pulling tens of GiB and only then discovering the resident floor does not
    fit is the exact waste `check` exists to prevent, so pull runs it.

    The download itself then reports continuously. At 39.9 MiB/s a 132 GiB
    checkpoint is 56 minutes and an 86 GiB one is 37; a CLI that prints one
    line and goes quiet for that long is indistinguishable from a dead one, and
    the only recovery the user has is Ctrl-C and start over.
    """
    from slotbank import ui
    from slotbank.registry import disk_free, local_path, pull, pull_plan, resolve

    repo = resolve(args.model)
    if local_path(repo):
        print(f"{repo} is already downloaded")
        return 0
    if not args.force:
        rc = _check(argparse.Namespace(
            repo=repo, revision=args.revision, leave_free=None, ram=None))
        if rc != 0:
            sys.stderr.write("\nslotbank: refusing to download. "
                             "Use --force to download anyway.\n")
            return rc
    quiet = getattr(args, "quiet", False)
    try:
        # The manifest costs ~0.3 s and buys the byte total, the file count and
        # which files are already cached -- all three before anything moves.
        plan = ui.Plan(pull_plan(repo, revision=args.revision,
                                 tqdm_class=ui.hub_progress(lambda *_: None)))
        if not quiet:
            ui.pull_header(repo, plan, disk_free())
        with ui.download_view(plan, quiet=quiet) as view:
            path = pull(repo, revision=args.revision,
                        tqdm_class=ui.hub_progress(view.on_bar))
    except Exception as exc:
        # A short name resolves by guessing a namespace, so a miss is common
        # and "404" alone is a dead end. Offer the real repos instead.
        sys.stderr.write(f"slotbank: cannot download {repo}: {exc}\n")
        _suggest(args.model, repo)
        return 2
    if quiet:
        print(f"done: {path}\nRun it with: slotbank run {args.model}")
    else:
        ui.download_done(path, plan, view, args.model)
    return 0


def _rm(args) -> int:
    from slotbank.registry import remove, resolve

    repo = resolve(args.model)
    if not args.yes:
        if input(f"delete {repo}? [y/N] ").strip().lower() not in ("y", "yes"):
            print("cancelled")
            return 0
    freed = remove(repo)
    print(f"deleted {repo}, freed {freed / (1 << 30):.2f} GiB")
    return 0


LEAVE_FOR_OS = 8 << 30      # what macOS and your apps need; see leave-free table


def _suggest(query: str, tried: str) -> None:
    """A short name resolves by guessing a namespace, so a miss is routine.
    '404' alone is a dead end; print the real repos instead."""
    from slotbank.registry import search

    hits = [h for h in search(query) if h != tried]
    if not hits:
        return
    sys.stderr.write("\ndid you mean:\n")
    for h in hits[:6]:
        sys.stderr.write(f"    {h}\n")


def _home(parser) -> int:
    """A bare `slotbank`. Falls back to argparse off a TTY or if Rich is absent,
    so piping into a file or a script still yields plain parseable text."""
    from slotbank.registry import local_models

    try:
        from slotbank import ui
        from slotbank.layout import detect_device_profile

        if ui.bars_disabled():
            raise ImportError                    # honour the no-colour path
        models = local_models()
        prof = detect_device_profile(leave_free_bytes=None)
        ui.home(prof.total_bytes, prof.max_working_set_bytes, len(models),
                sum(m.size_bytes for m in models) / float(1 << 30))
        return 0
    except Exception:
        parser.print_help()
        return 0


def _compare(args) -> int:
    """Every quant of a model, from one probe.

    Quants of a model are one architecture: layers, expert count, top-k and the
    expert share do not change with bits per weight, and the Hub file listing
    carries the byte totals for free. So the expensive half of `check` runs once
    and every other row is derived -- and marked `est`, because a repo that
    keeps embeddings at higher precision tilts that scaling.
    """
    from slotbank.layout import (MIN_KV_BYTES, detect_device_profile,
                                 slot_capacity, slot_floor)
    from slotbank.probe import family, probe, scale
    from slotbank.registry import resolve

    G = float(1 << 30)
    target = resolve(args.model)
    sibs = family(target)
    if not sibs:
        sys.stderr.write(f"slotbank: found no quantised siblings for {target}\n")
        return 2

    probed_repo = next((r for r, _, _ in sibs if r == target), sibs[len(sibs) // 2][0])
    try:
        card = probe(probed_repo, revision=args.revision)
    except Exception as exc:
        sys.stderr.write(f"slotbank: cannot inspect {probed_repo}: {exc}\n")
        return 2

    if args.ram:
        total = leave_free_arg(args.ram)
        budget = int(total * 0.74)
    else:
        p = detect_device_profile(leave_free_bytes=leave_free_arg(args.leave_free))
        total, budget = p.total_bytes, p.max_working_set_bytes

    print(f"\n  {family_stem_of(target)}  {len(sibs)} quants on the Hub, "
          f"one architecture")
    print(f"  probed {probed_repo.split('/')[-1]}: {card.layers} layers, "
          f"{card.num_experts} experts, top-k {card.top_k}, "
          f"{card.expert_frac:.1%} expert bytes")
    print("  those four are the architecture. Every row below shares them; "
          "only bytes per weight change.\n")
    if card.num_experts <= 1 or card.expert_frac < 0.5:
        # Nothing to stream, so the slot machinery is inert and every row would
        # just restate the file size. Say so instead of drawing a table that
        # implies a decision.
        print(f"  this is a dense model ({card.num_experts} expert"
              f"{'' if card.num_experts == 1 else 's'}, "
              f"{card.expert_frac:.0%} expert bytes).")
        print("  slotbank gives no benefit here -- there is no expert bank to "
              "stream. Sizes:")
        for repo, tot, _bits in sibs:
            print(f"    {repo.split('/')[-1]:<{max(len(r.split('/')[-1]) for r,_,_ in sibs)+2}}"
                  f"{tot / G:>7.1f} GiB")
        return 0

    name_w = max(len(r.split("/")[-1]) for r, _, _ in sibs) + 6
    print(f"  {'QUANT':<{name_w}}{'ON DISK':>9}{'FLOOR':>8}{'WIRED':>8}{'B/TOK':>8}")
    any_fits, margins = False, []
    for repo, tot, bits in sibs:
        c = card if repo == probed_repo else scale(card, repo, tot, bits)
        slots = slot_capacity(c.num_experts, c.top_k, stored_bytes=c.total_bytes,
                              working_set_bytes=budget, kv_bytes=MIN_KV_BYTES,
                              expert_param_frac=c.expert_frac)
        wired = c.resident_bytes + slots * c.layers * c.row_bytes
        fits = wired <= budget
        any_fits = any_fits or fits
        mark = "" if repo == probed_repo else " est"
        label = repo.split("/")[-1] + mark
        tail = "" if fits else f"  x  over by {(wired - budget) / G:.1f}G"
        if not fits:
            margins.append((wired - budget, label))
        print(f"  {label:<{name_w}}{c.total_bytes / G:>8.0f}G{c.resident_bytes / G:>7.1f}G"
              f"{wired / G:>7.1f}G{c.touched_bytes / G:>7.1f}G{tail}")

    print()
    if not any_fits:
        floor = slot_floor(card.num_experts, card.top_k)
        best = min(margins)
        print(f"  no quant runs here, and every row fails the same way: resident "
              f"floor plus\n  the minimum C={floor} pack exceeds the "
              f"{budget / G:.1f} GiB working set. C cannot go\n  below top-k. "
              f"Closest is {best[1].strip()}, still {best[0] / G:.1f}G over.")
    print("  one probe covered all of them. 'est' rows are scaled from it; a repo that")
    print("  keeps embeddings or lm_head at higher precision tilts that scaling, so the")
    print("  quant you pick gets its own check before anything downloads.")
    print("  quantisation buys memory, not quality -- this tool cannot measure quality.")
    return 0 if any_fits else 1


def family_stem_of(repo: str) -> str:
    from slotbank.probe import family_stem

    return family_stem(repo)


def _role_state(repo: str, budget: int) -> tuple[str, float]:
    """What this role costs and whether it still works, computed offline.

    Recomputed on every draw rather than stored: a model that fitted when it was
    pulled does not fit after more apps are installed, and this is where the
    user finds that out. Goes through admit.estimate_card rather than reading
    config.json directly -- expert counts live under half a dozen key spellings
    and, on Qwen3.5, inside a nested text_config.
    """
    from types import SimpleNamespace

    from slotbank.admit import estimate_card
    from slotbank.layout import MIN_KV_BYTES, slot_capacity, slot_floor
    from slotbank.registry import local_path

    path = local_path(repo)
    if path is None:
        return "not downloaded", 0.0
    from slotbank.admit import stored_bytes_from_files

    try:
        card = estimate_card(SimpleNamespace(model_path=path))
    except (OSError, ValueError, TypeError, KeyError):
        # estimate_card refuses when it cannot derive bits or parameter count --
        # correct for admission, wrong for a listing. An unquantised checkpoint
        # is still a real model; report its size and that it is dense.
        try:
            b = stored_bytes_from_files(path)
        except (OSError, ValueError):
            return "unreadable", 0.0
        return "dense, resident", b / float(1 << 30)
    stored = int(getattr(card, "stored_bytes", 0) or 0)
    e = int(getattr(card, "n_routed_experts", 0) or 0)
    k = int(getattr(card, "top_k", 0) or 0)
    if not e or not k:
        return "dense, resident", stored / float(1 << 30)
    frac = float(getattr(card, "expert_param_frac", 0) or 0.8)
    c = slot_capacity(e, k, stored_bytes=stored, working_set_bytes=budget,
                      kv_bytes=MIN_KV_BYTES, expert_param_frac=frac)
    wired = stored * (1.0 - frac) + stored * frac * (c / float(e))
    floor = slot_floor(e, k)
    if wired > budget:
        return "will not fit", wired / float(1 << 30)
    state = "ready" if c > floor else f"ready, C={floor} only"
    return state, wired / float(1 << 30)


def _use(args) -> int:
    """Roles as a settings page: which model answers as what. Nothing else.

    Deliberately not a scheduler or a router -- one model is resident at a time,
    and a screen implying otherwise would lie about the hardware.
    """
    from slotbank.layout import detect_device_profile
    from slotbank.registry import load_roles, resolve, save_role

    if args.role and (args.model or args.effort):
        model = resolve(args.model) if args.model else None
        try:
            save_role(args.role, model, args.effort)
        except ValueError as exc:
            sys.stderr.write(f"slotbank: {exc}\n")
            return 2

    roles = load_roles()
    if not roles:
        print("no roles set yet.\n  slotbank use chat Qwen3.5-35B-A3B-4bit")
        return 0
    budget = detect_device_profile(
        leave_free_bytes=None).max_working_set_bytes
    print("  slotbank use -- which model answers as what")
    w = max([len(r) for r in roles] + [6]) + 2
    m = max([len(v["model"].split("/")[-1]) for v in roles.values()] + [10]) + 2
    print(f"  {'ROLE':<{w}}{'MODEL':<{m}}{'WIRED':>8}  {'EFFORT':<8}STATE")
    for role in sorted(roles):
        v = roles[role]
        state, wired = _role_state(v["model"], budget)
        name = v["model"].split("/")[-1]
        print(f"  {role:<{w}}{name:<{m}}{wired:>7.1f}G  "
              f"{v.get('effort', '-'):<8}{state}")
    print("\n  a role is an alias, so every command takes one:")
    print("     slotbank run @chat          slotbank serve --model @chat")
    print("  one model is resident at a time. Switching unloads and reloads, and")
    print("  the page cache the last model warmed does not survive it.")
    return 0


def _search(args) -> int:
    from slotbank.registry import local_path, search

    q = " ".join(args.query)
    hits = search(q, limit=args.limit)
    if not hits:
        print(f"nothing found for {q!r}")
        return 1
    for h in hits:
        mark = "  [downloaded]" if local_path(h) else ""
        print(f"  {h}{mark}")
    print(f"\nInspect one before downloading:  slotbank check <id>")
    return 0


def _check(args) -> int:
    """Report whether a remote model can run here, before downloading it.

    Reads only the safetensors headers over HTTP range requests -- a few MB
    against a checkpoint of any size. The two figures that decide viability are
    the resident floor (everything that is not a routed expert must fit in RAM)
    and the bytes touched per token (which sets throughput).
    """
    import shutil

    from slotbank.layout import (MIN_KV_BYTES, detect_device_profile,
                                 slot_capacity, slot_floor)
    from slotbank.probe import probe

    G = float(1 << 30)
    from slotbank.registry import resolve

    try:
        card = probe(resolve(args.repo), revision=args.revision)
    except Exception as exc:                      # network, 404, odd layout
        target = resolve(args.repo)
        sys.stderr.write(f"slotbank: cannot inspect {target}: {exc}\n")
        _suggest(args.repo, target)
        return 2

    profile = detect_device_profile(leave_free_bytes=leave_free_arg(args.leave_free))
    disk_free = shutil.disk_usage(os.path.expanduser("~")).free
    if args.ram:
        # Metal reports ~74% of installed RAM as the recommended working set
        # (17.76 of 24.00 measured on this M4); assume the same ratio.
        total = leave_free_arg(args.ram)
        budget = int(total * 0.74)
    else:
        total = profile.total_bytes
        budget = profile.max_working_set_bytes
    # C is chosen at load, not fixed at 32. Ask the same function the loader
    # asks (expert_slots._capacity_from_model) so the verdict matches reality.
    slots = slot_capacity(
        card.num_experts, card.top_k,
        stored_bytes=card.total_bytes,
        working_set_bytes=budget,
        kv_bytes=MIN_KV_BYTES,
        expert_param_frac=card.expert_frac,
    )
    floor = slot_floor(card.num_experts, card.top_k)
    pack_est = slots * card.layers * card.row_bytes

    print(f"{card.repo}  ({card.model_type}, {card.shards} shards)")
    print(f"  layers {card.layers}  experts {card.num_experts}  top-k {card.top_k}")
    print(f"  total on disk    {card.total_bytes / G:8.2f} GiB")
    print(f"    routed experts {card.expert_bytes / G:8.2f} GiB  ({card.expert_frac:.1%}, streamed)")
    print(f"    resident floor {card.resident_bytes / G:8.2f} GiB  (must fit in RAM)")
    print(f"  per expert       {card.row_bytes / (1 << 20):8.2f} MiB")
    print(f"  touched / token  {card.touched_bytes / G:8.2f} GiB  (sets throughput)")
    print()
    wired = card.resident_bytes + pack_est
    who = "this machine" if args.ram is None else f"{total / G:.0f} GiB machine (hypothetical)"
    print(f"  {who}: working set {budget / G:.1f} GiB"
          + ("" if args.ram else f", {disk_free / G:.0f} GiB disk free"))

    if slots >= card.num_experts:
        # The checkpoint fits under the working set, so slot_capacity returns
        # C=E and nothing is streamed. Worth saying outright: on such a machine
        # this runtime is stock mlx-lm with extra bookkeeping.
        print(f"  the whole checkpoint fits -- slotbank picks C=E={card.num_experts} "
              f"and streams nothing.\n  On this machine slotbank is not doing "
              f"anything; stock mlx-lm is equally fast.")
        return 0

    print()
    print(f"  MEMORY LEDGER at C={slots}")
    print(f"    wired (unreclaimable, competes with your apps)")
    print(f"      resident floor        {card.resident_bytes / G:8.2f} GiB")
    print(f"      slot pack             {pack_est / G:8.2f} GiB   "
          f"({slots} x {card.layers} rows)")
    print(f"      {'':<22}{'-' * 12}")
    print(f"      total wired           {wired / G:8.2f} GiB   "
          f"({wired / budget:.0%} of working set)")
    print(f"    evictable")
    cache = max(0.0, total - wired - LEAVE_FOR_OS)
    streamed = card.expert_bytes
    # An UPPER BOUND, not a prediction. Measured on this machine with the 35B:
    # 8.4 GiB reclaimable but only 4.71 GiB of the bank actually resident
    # (mincore, 2026-08-25) -- about half of what free RAM implies. See
    # ROADMAP.md section 1; closing that gap is the open work.
    print(f"      free for page cache   {cache / G:8.2f} GiB   "
          f"= at most {min(1.0, cache / streamed):.0%} of the "
          f"{streamed / G:.0f} GiB bank")
    print(f"        (upper bound: measured residency runs ~half of free RAM)")
    print(f"      the rest reads from SSD each time it is routed to")
    print(f"    the checkpoint is {card.total_bytes / wired:.0f}x the wired footprint")

    # Fitting and being usable are different questions. Bytes that must come
    # off SSD each token is what decides the second, so print it rather than
    # letting "runs here" imply a usable rate.
    cached_frac = min(1.0, cache / streamed)
    from_ssd = card.touched_bytes * (1.0 - cached_frac)
    print()
    print(f"    of the {card.touched_bytes / G:.2f} GiB touched per token, "
          f"at least ~{from_ssd / G:.2f} GiB comes off SSD")
    print(f"    (this is the figure that sets throughput, not the parameter "
          f"count.\n     Two effects push it either way and neither is "
          f"modelled here: routing\n     skew keeps hot experts cached, "
          f"while measured residency runs below\n     what free RAM implies.)")

    blockers, notes = [], []
    if card.expert_frac < 0.5:
        notes.append("not expert-dominated -- little to stream, little to gain")
    if card.total_bytes > disk_free and not args.ram:
        blockers.append(
            f"needs {card.total_bytes / G:.0f} GiB of disk, {disk_free / G:.0f} GiB free")
    if slots <= floor and card.resident_bytes + pack_est > budget:
        # Even the minimum pack (top-k per layer) does not fit alongside the
        # non-expert weights. This is the only hard no: C cannot go lower.
        blockers.append(
            f"resident floor {card.resident_bytes / G:.1f} GiB plus the minimum "
            f"C={floor} pack exceeds the {budget / G:.1f} GiB budget")
    elif slots <= floor:
        notes.append(f"only the minimum pack fits (C={floor}) -- expect a low hit rate")

    if blockers:
        print("\n  will not run as-is:")
        for b in blockers:
            print(f"    - {b}")
        for n in notes:
            print(f"    note: {n}")
        return 1
    print(f"\n  runs here. {card.total_bytes / G:.0f} GiB of weights, "
          f"{wired / G:.1f} GiB wired.")
    for n in notes:
        print(f"    note: {n}")
    return 0


def _admit(args) -> int:
    from slotbank.admit import admit_or_raise, estimate_card
    from slotbank.layout import (MIN_KV_BYTES, detect_device_profile,
                                 slot_capacity, slot_floor)

    ns = SimpleNamespace(model_path=args.model, leave_free=leave_free_arg(args.leave_free))
    profile = detect_device_profile(leave_free_bytes=ns.leave_free)
    card = estimate_card(ns)
    result = admit_or_raise(ns, profile=profile, card=card)
    G = 1 << 30
    print(
        f"ok={result.ok} kind={card.kind} stored={card.stored_bytes / G:.2f}GiB "
        f"active={card.active_bytes / G:.2f}GiB leave_free={result.leave_free_bytes / G:.0f}GiB "
        f"working_set={result.max_working_set_bytes / G:.0f}GiB"
    )
    print(result.reason)
    if card.kind == "moe":
        from slotbank.layout import slot_capacity

        c = slot_capacity(
            card.n_routed_experts, card.top_k,
            stored_bytes=card.stored_bytes,
            working_set_bytes=result.max_working_set_bytes,
            expert_param_frac=card.expert_param_frac,
        )
        fits = card.stored_bytes <= result.max_working_set_bytes
        print(f"moe: E={card.n_routed_experts} top_k={card.top_k} -> C={c} "
              f"({c / max(1, card.n_routed_experts):.1%} of bank)")
        if fits:
            print("note: bank fits the working set; stock mlx-lm will be faster "
                  "(slotbank is for models that do not fit)")
    from slotbank.admit import (
        draft_viable,
        hybrid_from_config,
        kv_bytes_per_token,
        load_hf_config,
        max_context_tokens,
    )

    cfg = load_hf_config(args.model)
    hybrid = hybrid_from_config(cfg)
    if card.kind == "dense" and result.ok:
        ctx = max_context_tokens(result, card, cfg)
        per = kv_bytes_per_token(cfg)
        print(
            f"resident: weights fit; no layer stream. "
            f"context ~{ctx} tokens at {per} B/tok after 1 GiB slop "
            f"(8k-16k comfortable on 24 GB; longer history is the disk log)"
        )
        if cfg.get("vision_config") and os.environ.get("SLOTBANK_VISION", "0").strip() not in {
            "1", "true", "yes", "on",
        }:
            print(
                "text-only: vision tower not loaded (SLOTBANK_VISION=0 / no --vision); "
                "mlx-lm language path, KV 8-bit if the pack is tight on the working set"
            )
    if hybrid:
        print(f"speculative decoding (trim): UNSAFE -- {hybrid}; a rejected draft cannot "
              f"be rewound and output would be silently wrong")
    draft = os.environ.get("SLOTBANK_DRAFT", "").strip()
    draft_bytes = 1 << 30
    if draft:
        from slotbank.admit import stored_bytes_from_files

        draft_bytes = stored_bytes_from_files(os.path.expanduser(draft)) or (1 << 30)
    why = draft_viable(result, card, hybrid, draft_bytes=draft_bytes, verify="dflash")
    if why:
        print(f"DFlash (mlx-vlm verify): refuse -- {why}")
        if "headroom" in why:
            print("hint: 4-bit 27B + DFlash on 24 GB needs --leave-free 6g")
    elif draft:
        print(f"DFlash (mlx-vlm verify): ok -- {draft}")
    else:
        print(
            "MTP/DFlash (mlx-vlm verify): headroom ok; a sibling "
            "Qwen3.8-27B-MTP-4bit is auto-attached on serve, or pass --draft"
        )
    if not result.ok and card.stored_bytes > (14 << 30):
        print("hint: 4-bit 27B on 24 GB needs --leave-free 6g (18 GiB working set)")
    return 0 if result.ok else 1


def _tps(args) -> int:
    from slotbank.tps import STRATEGIES, seed_local_log, read_attempts, daily_draft

    path = seed_local_log()
    if getattr(args, "log", False):
        rows = read_attempts()
        if not rows:
            print(f"no attempts in {path}")
            return 0
        for rec in rows:
            toks = rec.get("toks")
            extra = f" {toks:.2f} tok/s" if isinstance(toks, (int, float)) else ""
            print(f"{rec['id']:28} {rec['outcome']:10}{extra}")
        print(f"\n{len(rows)} rows in {path}")
        return 0
    print(f"daily --draft route: {daily_draft()}")
    print(f"{'ID':28} {'STATUS':10} SUMMARY")
    for s in STRATEGIES:
        print(f"{s.id:28} {s.status:10} {s.summary}")
    print(f"\nlocal log: {path}")
    return 0


def _context(args) -> int:
    from pathlib import Path

    from slotbank.context_os import append, compile_working_set, init_session

    cmd = getattr(args, "context_cmd", None)
    if cmd == "init":
        root = init_session(args.dir)
        print(root / "log.jsonl")
        return 0
    if cmd == "append":
        rec = append(args.dir, args.role, args.content, pointers=args.pointer)
        print(rec["seq"])
        return 0
    if cmd == "compile":
        text = compile_working_set(
            args.dir, budget=args.budget, repo=args.repo
        )
        out = Path(args.dir) / "working_set.txt"
        out.write_text(text, encoding="utf-8")
        sys.stdout.write(text)
        if text and not text.endswith("\n"):
            sys.stdout.write("\n")
        return 0
    sys.stderr.write("slotbank context: init | append | compile\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
