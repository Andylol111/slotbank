"""Context OS → llama-server. Does not load MLX or the 4-bit 27B.

Prove the wrap: slotbank compiles the working set; IQ3 GGUF generates.
Default target is :8765 so :8080 can stay 4-bit + DFlash.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = os.environ.get("LLAMA_URL") or "http://127.0.0.1:8765/v1/chat/completions"


def _post(messages: list[dict], n: int) -> dict:
    body = json.dumps({
        "messages": messages,
        "temperature": 0,
        "max_tokens": n,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        URL, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    from slotbank.prompt import with_context_os

    os.environ.setdefault("SLOTBANK_THINKING", "0")
    user = os.environ.get("SLOTBANK_SIDECAR_PROMPT") or (
        "Reply with the single word pong."
    )
    msgs = with_context_os([{"role": "user", "content": user}])
    try:
        t0 = time.perf_counter()
        data = _post(msgs, int(os.environ.get("SLOTBANK_SIDECAR_TOKENS") or 32))
        wall = time.perf_counter() - t0
    except urllib.error.URLError as exc:
        print(f"llama-server not reachable at {URL}: {exc}", file=sys.stderr)
        print("start it on :8765; keep 4-bit+DFlash on :8080", file=sys.stderr)
        return 2
    choice = (data.get("choices") or [{}])[0]
    text = ((choice.get("message") or {}).get("content") or "").strip()
    usage = data.get("usage") or {}
    timings = data.get("timings") or {}
    gen = int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)
    print(text[:500])
    print(
        f"completion_tokens={gen} wall_s={wall:.2f} "
        f"prompt_tok_s={timings.get('prompt_per_second')} "
        f"decode_tok_s={timings.get('predicted_per_second')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
