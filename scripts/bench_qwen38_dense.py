"""Measure warm decode tok/s and peak GiB on the slotbank serve path."""
from __future__ import annotations

import os
import time
from pathlib import Path

MODEL = os.environ.get("SLOTBANK_BENCH_MODEL") or str(
    Path.home() / "Models/Qwen3.8-27B-3.5bpw"
)
PROMPT = os.environ.get("SLOTBANK_BENCH_PROMPT") or (
    "Count from 1 to 40, integers only."
)
# Harness (OMP / Anthropic) defaults to 1.0 so thinking samples. Greedy 13.47
# was SLOTBANK_BENCH_TEMP=0. Pass 1 to see the door the harness actually uses.
TEMP = float(os.environ.get("SLOTBANK_BENCH_TEMP", "1"))


def main() -> None:
    import mlx.core as mx

    from slotbank.engine import Engine, leave_free_arg
    from slotbank.types import SamplingParams

    os.environ.setdefault("SLOTBANK_THINKING", "0")
    os.environ.setdefault("SLOTBANK_VISION", "0")

    t0 = time.perf_counter()
    leave = leave_free_arg(os.environ.get("SLOTBANK_LEAVE_FREE") or None)
    engine = Engine(MODEL, leave_free=leave, model_id=Path(MODEL).name)
    load_s = time.perf_counter() - t0
    peak_load = mx.get_peak_memory() / (1 << 30)
    active_load = mx.get_active_memory() / (1 << 30)
    draft = os.environ.get("SLOTBANK_DRAFT", "").strip()
    print(f"model={MODEL}")
    print(f"draft={draft or '-'}")
    print(f"draft_kind={engine.runtime._draft_kind}")
    print(f"draft_block={engine.runtime._draft_block}")
    print(f"temperature={TEMP}")

    ids = engine.tokenize_chat(
        [{"role": "user", "content": PROMPT}],
        None,
    )
    # Discard compile / first-graph tokens, then measure a warm window.
    mx.reset_peak_memory()
    sampling = SamplingParams(temperature=TEMP, max_tokens=8)
    engine.generate(ids, sampling)
    mx.reset_peak_memory()
    t1 = time.perf_counter()
    out = engine.generate(ids, SamplingParams(temperature=TEMP, max_tokens=64))
    wall = time.perf_counter() - t1
    n = max(1, out.completion_tokens)
    peak_gen = mx.get_peak_memory() / (1 << 30)
    active_gen = mx.get_active_memory() / (1 << 30)
    print(f"load_s={load_s:.1f}")
    print(f"warm_gen_s={wall:.1f}")
    print(f"warm_tokens={n}")
    print(f"warm_tok/s={n / wall:.2f}")
    print(f"peak_load_GiB={peak_load:.2f}")
    print(f"active_load_GiB={active_load:.2f}")
    print(f"peak_gen_GiB={peak_gen:.2f}")
    print(f"active_gen_GiB={active_gen:.2f}")
    print(f"draft_kind={out.draft_kind}")
    print(f"draft_block={out.draft_block}")
    print(f"draft_accept_rate={out.draft_accept_rate}")
    print(f"text={out.content[:160]!r}")
    # engine.close() GIL-aborts after a successful DFlash run; skip it.
    os._exit(0)


if __name__ == "__main__":
    main()
