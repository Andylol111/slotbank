from __future__ import annotations

import argparse
import sys
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

    s = sub.add_parser("serve", help="Chat / Claude / Codex HTTP")
    _model_args(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--api-key", default=None)

    a = sub.add_parser("admit", help="print the memory card and refuse if it does not fit")
    _model_args(a)

    args = p.parse_args(argv)
    if args.cmd == "admit":
        return _admit(args)
    if args.cmd == "generate":
        return _generate(args)
    return _serve(args)


def _model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="local mlx-lm folder or Hugging Face id")
    p.add_argument("--leave-free", default=None, help="RAM to keep for macOS, e.g. 8g")


def _generate(args) -> int:
    engine = Engine(args.model, leave_free=leave_free_arg(args.leave_free))
    try:
        ids = engine.tokenize_text(args.prompt)
        sampling = SamplingParams(
            temperature=args.temp,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
        )
        def on_token(_tid: int, piece: str) -> None:
            sys.stdout.write(piece)
            sys.stdout.flush()
        result = engine.generate(ids, sampling, on_token=on_token)
        if not result.content.endswith("\n"):
            sys.stdout.write("\n")
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


def _admit(args) -> int:
    from slotbank.admit import admit_or_raise, estimate_card
    from slotbank.layout import detect_device_profile

    ns = SimpleNamespace(model_path=args.model, leave_free=leave_free_arg(args.leave_free))
    profile = detect_device_profile(leave_free_bytes=ns.leave_free)
    card = estimate_card(ns)
    result = admit_or_raise(ns, profile=profile, card=card)
    print(
        f"ok={result.ok} kind={card.kind} stored={card.stored_bytes} "
        f"active={card.active_bytes} leave_free={result.leave_free_bytes} "
        f"working_set={result.max_working_set_bytes}"
    )
    print(result.reason)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
