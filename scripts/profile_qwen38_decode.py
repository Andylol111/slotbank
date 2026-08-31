"""Reference decode speed: stock mlx-lm stream_generate, not slotbank."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

MODEL = os.environ.get("SLOTBANK_BENCH_MODEL") or str(
    Path.home() / "Models/Qwen3.8-27B-3.5bpw"
)


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    log(f"mlx={mx.__version__}")
    t0 = time.perf_counter()
    model, tok = load(MODEL)
    mx.eval(model.parameters())
    log(f"loaded_s={time.perf_counter() - t0:.1f} active_GiB={mx.get_active_memory() / (1 << 30):.2f}")

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Count from 1 to 80, integers only, one line."}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    last = None
    n = 0
    t_first = None
    t_start = time.perf_counter()
    for resp in stream_generate(model, tok, prompt, max_tokens=80):
        n += 1
        if t_first is None:
            t_first = time.perf_counter()
            log(f"first_token_s={t_first - t_start:.2f} peak_GiB={mx.get_peak_memory() / (1 << 30):.2f}")
        last = resp
        if n == 16:
            log(f"mid_n=16 gen_tps={resp.generation_tps:.2f}")
    log(
        f"n={n} gen_tps={getattr(last, 'generation_tps', None)} "
        f"prompt_tps={getattr(last, 'prompt_tps', None)} "
        f"peak_GiB={getattr(last, 'peak_memory', None)} "
        f"text={getattr(last, 'text', '')[:80]!r}"
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
