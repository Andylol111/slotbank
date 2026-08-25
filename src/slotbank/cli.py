from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

from slotbank.engine import Engine, leave_free_arg
from slotbank.types import SamplingParams


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="slotbank")
    sub = p.add_subparsers(dest="cmd", required=True)

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
    _model_args(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--api-key", default=None)
    _tuning_args(s)

    c = sub.add_parser("check", help="inspect a remote model without downloading it")
    c.add_argument("repo", help="Hugging Face repo id, e.g. mlx-community/Qwen3.5-35B-A3B-4bit")
    c.add_argument("--revision", default="main")
    c.add_argument("--leave-free", default=None, help="RAM to keep for macOS, e.g. 8g")

    a = sub.add_parser("admit", help="print the memory card and refuse if it does not fit")
    _model_args(a)

    args = p.parse_args(argv)
    _apply_tuning(args)
    try:
        if args.cmd == "check":
            return _check(args)
        if args.cmd == "admit":
            return _admit(args)
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


def _serve(args) -> int:
    import uvicorn

    from slotbank.api.app import create_app

    engine = Engine(args.model, leave_free=leave_free_arg(args.leave_free))
    app = create_app(engine, api_key=args.api_key)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    engine.close()
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
    try:
        card = probe(args.repo, revision=args.revision)
    except Exception as exc:                      # network, 404, odd layout
        sys.stderr.write(f"slotbank: cannot inspect {args.repo}: {exc}\n")
        return 2

    profile = detect_device_profile(leave_free_bytes=leave_free_arg(args.leave_free))
    disk_free = shutil.disk_usage(os.path.expanduser("~")).free
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
    print(f"  this machine: {profile.total_bytes / G:.0f} GiB RAM, "
          f"{disk_free / G:.0f} GiB disk free")
    print(f"  slotbank would pick C={slots}: {wired / G:.2f} GiB wired "
          f"({card.total_bytes / wired:.0f}x smaller than the checkpoint)")

    blockers, notes = [], []
    if card.expert_frac < 0.5:
        notes.append("not expert-dominated -- little to stream, little to gain")
    if card.total_bytes > disk_free:
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
    from slotbank.admit import hybrid_from_config, load_hf_config

    hybrid = hybrid_from_config(load_hf_config(args.model))
    if hybrid:
        print(f"speculative decoding: UNSAFE -- {hybrid}; a rejected draft cannot "
              f"be rewound and output would be silently wrong")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
