"""Load-state helpers. MLX-free: used by the HTTP layer before Engine exists."""
from __future__ import annotations

import time


class EngineNotReady(Exception):
    """Weights are still on disk / Metal. Not a bad prompt."""


def load_error(engine) -> str | None:
    err = getattr(engine, "load_error", None)
    return str(err) if err else None


def is_loading(engine) -> bool:
    if load_error(engine):
        return False
    return bool(getattr(engine, "loading", False))


def busy_message(engine) -> str:
    err = load_error(engine)
    if err:
        return err
    mid = getattr(engine, "model_id", "model")
    return (
        f"slotbank is still loading {mid}. "
        "Wait until the serve terminal prints 'ready', then resend."
    )


def poll_until_ready(engine, *, timeout: float, ping_s: float):
    """Yield once per ping interval until the proxy is swapped in.

    Raises ``EngineNotReady`` on load failure or timeout.
    """
    deadline = time.monotonic() + float(timeout)
    interval = max(0.05, float(ping_s))
    while True:
        err = load_error(engine)
        if err:
            raise EngineNotReady(err)
        if not is_loading(engine):
            return
        if time.monotonic() >= deadline:
            raise EngineNotReady(busy_message(engine))
        yield
        end = min(time.monotonic() + interval, deadline)
        while time.monotonic() < end:
            if load_error(engine) or not is_loading(engine):
                break
            time.sleep(0.05)
