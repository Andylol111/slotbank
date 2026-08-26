"""Terminal rendering for the CLI. Rich, the filesystem and network progress.

No mlx import (see ``tests/test_fence.py``), so importing this to draw a
download bar never drags the GPU stack in.

The one non-obvious thing in here is :func:`hub_progress`. ``huggingface_hub``
lets you substitute the tqdm class ``snapshot_download`` uses, which is the only
supported way to learn how far along a download is. Three of its contract
details are load-bearing and were checked against the installed
``huggingface_hub 1.28.0`` rather than recalled -- they are written down at the
function itself.

Design rationale and every screen this implements: ``docs/cli-design.md``.
Working prototype of the same screens on fake data: ``scratch/rich_demo.py``.
"""

from __future__ import annotations

import io
import json
import os
import time
from contextlib import contextmanager

GIB = float(1 << 30)
MIB = float(1 << 20)

# One palette, shared with the prototype. Verdicts are the only loud thing.
STYLES = {
    "sb.repo": "bold cyan",
    "sb.key": "dim",
    "sb.num": "bold white",
    "sb.ok": "bold green",
    "sb.warn": "bold yellow",
    "sb.bad": "bold red",
    "sb.hint": "dim italic",
}

# The three bars `snapshot_download` builds with the caller's tqdm class.
BAR_TRANSFER = "huggingface_hub.snapshot_download.transfer"   # bytes off the wire
BAR_WRITE = "huggingface_hub.snapshot_download"               # bytes onto disk
# The file counter arrives as name=None (hf_thread_map calls the class directly).


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def gib(n: float, places: int = 2) -> str:
    return f"{n / GIB:,.{places}f} GiB"


def size(n: float) -> str:
    """GiB for a checkpoint, MiB below that.

    The models this tool exists for are 16 to 538 GiB, but a tokenizer-only
    repo is a few hundred KiB, and printing that as "0.00 GiB" is not a
    rounding choice, it is a wrong number on the screen.
    """
    if n >= GIB:
        return f"{n / GIB:,.2f} GiB"
    if n >= MIB:
        return f"{n / MIB:,.1f} MiB"
    return f"{n / 1024:,.0f} KiB"


def secs(n: float) -> str:
    n = int(max(0, n))
    if n < 90:
        return f"{n}s"
    if n < 5400:
        return f"{n // 60}m {n % 60:02d}s"
    return f"{n // 3600}h {(n % 3600) // 60:02d}m"


def _consoles():
    """stdout carries the payload, stderr carries the motion.

    Same split ``cli._status`` already uses, so ``slotbank check x > file``
    keeps producing a clean report with the spinner going to the terminal.
    """
    from rich.console import Console
    from rich.theme import Theme

    theme = Theme(STYLES)
    return (Console(theme=theme, highlight=False),
            Console(theme=theme, highlight=False, stderr=True))


# --------------------------------------------------------------------------
# the huggingface_hub tqdm contract
# --------------------------------------------------------------------------

def hub_progress(sink):
    """Build a ``tqdm_class`` for ``snapshot_download`` that feeds ``sink``.

    ``sink(name, n, total)`` is called on every repaint. ``name`` is
    :data:`BAR_TRANSFER`, :data:`BAR_WRITE`, or None for the file counter.

    Three details of the contract, each verified against the installed
    ``huggingface_hub 1.28.0`` source, not recalled:

    1. ``name=`` only reaches classes that subclass
       ``huggingface_hub.utils.tqdm.tqdm``. ``_create_progress_bar`` calls
       anything else as ``cls(**kwargs)`` with neither ``name`` nor ``disable``
       (utils/tqdm.py:338). So this subclasses the hub's class, not
       ``tqdm.auto.tqdm``.
    2. ``disable`` must be forced False. The hub resolves it to None
       (``is_tqdm_disabled``), tqdm reads None as "disable unless the file is a
       TTY", and our file is a StringIO -- and a disabled ``tqdm.update()``
       returns *before* touching ``self.n``. Without this every counter reads
       zero forever, silently, which is worse than the bug being fixed.
    3. ``file`` must be redirected. tqdm must not write to the stream Rich is
       repainting.

    A fresh subclass per call rather than a module-level one, so two pulls in
    one process cannot cross wires through class state.
    """
    from huggingface_hub.utils.tqdm import tqdm as _hf_tqdm

    class _Shim(_hf_tqdm):
        def __init__(self, *a, **kw):
            self._sb_name = kw.pop("name", None)
            kw["disable"] = False
            kw["file"] = io.StringIO()
            super().__init__(*a, **kw)

        def display(self, *a, **kw):
            try:
                sink(self._sb_name, self.n, self.total)
            except Exception:            # noqa: BLE001 - a bar must never
                pass                     # take the download down with it
            return True

    return _Shim


def bars_disabled() -> bool:
    """Whether the caller has switched hub progress off through the environment.

    ``hub_progress`` forces tqdm's own ``disable`` off, so this signal has to be
    read here instead -- otherwise ``HF_HUB_DISABLE_PROGRESS_BARS=1`` would be
    honoured everywhere except the one tool that overrode it.
    """
    return os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS", "").strip().lower() in (
        "1", "true", "yes", "on")


# --------------------------------------------------------------------------
# the measured bandwidth of the last completed pull
# --------------------------------------------------------------------------

_RATE_FILE = os.path.expanduser("~/.cache/slotbank/pulls.json")


def last_rate_mib_s() -> float | None:
    """Average MiB/s of the last completed pull, or None if there has not been one.

    An ETA before the first byte moves needs a bandwidth number, and the only
    honest source of one is a pull this machine actually finished. Inventing a
    default would be exactly the projected figure this project retracts
    everywhere else, so the header simply carries no ETA until a pull completes.
    """
    try:
        with open(_RATE_FILE) as fh:
            rate = float(json.load(fh)["mib_s"])
        return rate if rate > 0 else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def record_rate(nbytes: float, elapsed: float) -> None:
    """Remember this pull's average rate. Best effort: never fails a download."""
    if nbytes < (64 << 20) or elapsed <= 0:
        return                                  # too small to mean anything
    try:
        os.makedirs(os.path.dirname(_RATE_FILE), exist_ok=True)
        with open(_RATE_FILE, "w") as fh:
            json.dump({"mib_s": nbytes / MIB / elapsed,
                       "bytes": int(nbytes), "at": int(time.time())}, fh)
    except OSError:
        pass


# --------------------------------------------------------------------------
# the download screen
# --------------------------------------------------------------------------

class Plan:
    """What a pull is about to do, from the dry-run manifest."""

    def __init__(self, files):
        self.files = list(files)
        self.n_files = len(self.files)
        self.total_bytes = sum(int(f.file_size or 0) for f in self.files)
        self.download_bytes = sum(int(f.file_size or 0) for f in self.files
                                  if getattr(f, "will_download", True))
        self.cached_bytes = self.total_bytes - self.download_bytes


def pull_header(repo: str, plan: Plan, disk_free: int, out=None) -> None:
    """The commitment, printed before a byte moves.

    A download of 86 GiB is a decision, not an action. Someone who is told the
    size, the free space left afterwards and roughly how long it takes can
    abort in the first second instead of the fortieth minute.
    """
    out = out or _consoles()[0]
    out.print()
    out.print(f"  [sb.key]pull[/] [sb.repo]{repo}[/]", soft_wrap=True)
    after = disk_free - plan.download_bytes
    line = (f"  plan: {plan.n_files} files, {size(plan.total_bytes)}, "
            f"{gib(disk_free, 0)} free -> {gib(after, 0)} after")
    if plan.cached_bytes:
        line += f"  ({size(plan.cached_bytes)} already cached)"
    out.print(f"[sb.key]{line}[/]", soft_wrap=True)
    if plan.download_bytes <= 0:
        out.print("[sb.key]  everything is already in the cache; nothing to "
                  "fetch.[/]", soft_wrap=True)
        out.print()
        return
    rate = last_rate_mib_s()
    if rate:
        out.print(f"[sb.key]  about {secs(plan.download_bytes / MIB / rate)} at "
                  f"your measured {rate:.1f} MiB/s. Ctrl-C is safe; finished "
                  f"files stay cached.[/]", soft_wrap=True)
    else:
        # No completed pull to average, and a vendor bandwidth figure would be
        # a projection. Say why the estimate is missing instead of inventing it.
        out.print("[sb.key]  no ETA yet -- it is averaged from your last "
                  "completed pull. Ctrl-C is safe; finished files stay "
                  "cached.[/]", soft_wrap=True)
    out.print()


class _State:
    """What the three hub bars have reported so far. Written from hub threads."""

    def __init__(self, plan: Plan):
        # Finding 3: the hub's own reconstruct total starts at 0 and grows as
        # each file's bar is built, so an early percentage taken from it is
        # wrong. The dry-run manifest knows the real total, so use that.
        self.total = plan.download_bytes
        self.n_files = plan.n_files
        self.written = 0.0
        self.wire = 0.0
        self.files = 0
        self.t0 = time.monotonic()
        self.saw_bytes = False

    def on_bar(self, name, n, total) -> None:
        if name == BAR_WRITE:
            self.written = float(n or 0)
            self.saw_bytes = self.saw_bytes or self.written > 0
        elif name == BAR_TRANSFER:
            self.wire = float(n or 0)
        elif name is None:
            self.files = int(n or 0)
            if total:
                self.n_files = int(total)

    @property
    def elapsed(self) -> float:
        return max(1e-9, time.monotonic() - self.t0)

    @property
    def rate_mib_s(self) -> float:
        return self.written / MIB / self.elapsed

    @property
    def eta(self) -> float | None:
        left = self.total - self.written
        rate = self.rate_mib_s
        if left <= 0 or rate < 0.05:
            return None
        return left / MIB / rate


class _RichView:
    """Live region: how much is left, how fast, how many files, when it ends."""

    def __init__(self, state: _State, err):
        from rich.progress import (BarColumn, DownloadColumn, Progress,
                                   TaskProgressColumn, TextColumn)

        self.state = state
        self.err = err
        self.progress = Progress(
            TextColumn("[sb.key]  {task.description}"),
            BarColumn(bar_width=None, complete_style="cyan",
                      finished_style="green"),
            TaskProgressColumn(),
            DownloadColumn(binary_units=True),
            console=err, expand=True, auto_refresh=False,
        )
        self.task = self.progress.add_task("downloading", total=state.total or None)

    def __rich_console__(self, console, options):
        from rich.text import Text

        s = self.state
        self.progress.update(self.task, completed=s.written)
        if not s.saw_bytes and s.elapsed > 5.0:
            # Risk 1 in docs/cli-design.md: if the hub's bar contract ever
            # changes, degrade to an elapsed-time line. A 0% bar that is a lie
            # is the one outcome worse than no bar.
            yield Text(f"  downloading -- {secs(s.elapsed)} elapsed, no progress "
                       f"reported by huggingface_hub", style="sb.warn")
            return
        yield self.progress
        foot = Text("  ")
        foot.append(f"{s.files}/{s.n_files} files", style="sb.num")
        foot.append(f"   {s.rate_mib_s:.1f} MiB/s", style="sb.key")
        eta = s.eta
        if eta is not None:
            foot.append(f"   {secs(eta)} left", style="sb.key")
        if s.wire and s.written:
            # Only slotbank can show this: hf_xet dedupes chunks against a
            # local content-addressed store, so bytes written and bytes fetched
            # differ, and the gap is the user's bandwidth back.
            saved = 1.0 - min(1.0, s.wire / s.written)
            foot.append(f"   over the wire {size(s.wire)}", style="sb.key")
            if saved > 0.01:
                foot.append(f" ({saved:.0%} deduped by xet)", style="sb.ok")
        yield foot


class _PlainView:
    """No TTY: same numbers, same order, one line at a time, no cursor motion.

    Throttled to one line per 10 s *and* 5% of progress -- whichever of the two
    fires less often is what you get, so a three-hour pull writes about twenty
    lines to a CI log rather than forty thousand.
    """

    MIN_SECONDS = 10.0
    MIN_FRACTION = 0.05

    def __init__(self, state: _State, err):
        self.state = state
        self.err = err
        self.last_t = 0.0
        self.last_frac = 0.0

    def tick(self, final: bool = False) -> None:
        s = self.state
        if s.total <= 0:
            return
        frac = s.written / s.total
        if not final and not (s.elapsed - self.last_t >= self.MIN_SECONDS
                              and frac - self.last_frac >= self.MIN_FRACTION):
            return
        self.last_t, self.last_frac = s.elapsed, frac
        eta = s.eta
        self.err.print(f"pull: {frac:.0%} {size(s.written)}/{size(s.total)} "
                       f"{s.rate_mib_s:.1f} MiB/s"
                       + (f" eta {secs(eta)}" if eta is not None else ""),
                       highlight=False, markup=False, soft_wrap=True)


@contextmanager
def download_view(plan: Plan, quiet: bool = False, err=None):
    """Live download progress, degrading to plain lines and then to silence.

    Yields an object with ``on_bar``, which is what :func:`hub_progress` feeds.
    Three independent conditions turn the live rendering off and any one is
    enough: no TTY on stderr, ``--quiet``, and ``HF_HUB_DISABLE_PROGRESS_BARS``.
    """
    err = err or _consoles()[1]
    state = _State(plan)
    live_ok = err.is_terminal and not quiet and not bars_disabled()

    if quiet:
        yield state
        return

    if not live_ok:
        plain = _PlainView(state, err)
        state_on_bar = state.on_bar

        def on_bar(name, n, total):          # emit as the numbers arrive
            state_on_bar(name, n, total)
            plain.tick()

        state.on_bar = on_bar                # type: ignore[method-assign]
        err.print(f"pull: {plan.n_files} files, {size(plan.total_bytes)}",
                  highlight=False, markup=False, soft_wrap=True)
        try:
            yield state
        finally:
            state.written = state.written or 0
            plain.tick(final=True)
        return

    from rich.live import Live

    view = _RichView(state, err)
    with Live(view, console=err, refresh_per_second=8, transient=True):
        yield state
    # `transient` clears the live region; the completion lines below replace it.


def download_done(path: str, plan: Plan, state, model_arg: str, out=None) -> None:
    """Where it landed and what to type next."""
    out = out or _consoles()[0]
    elapsed = state.elapsed if state is not None else 0.0
    moved = float(getattr(state, "written", 0.0) or 0.0)
    record_rate(moved, elapsed)
    if moved <= 0:
        out.print("  [sb.ok]done[/][sb.key] -- nothing to fetch, it was already "
                  "cached[/]", soft_wrap=True)
    else:
        out.print(f"  [sb.ok]done[/][sb.key] -- {size(moved)} in {secs(elapsed)}"
                  f" ({moved / MIB / max(elapsed, 1e-9):.1f} MiB/s)[/]",
                  soft_wrap=True)
    wire = float(getattr(state, "wire", 0.0) or 0.0)
    if wire and moved and wire < moved * 0.99:
        # Bytes written and bytes fetched differ under hf_xet chunk dedup, and
        # the gap is bandwidth the user did not spend.
        out.print(f"[sb.key]  {size(wire)} over the wire "
                  f"({1 - wire / moved:.0%} deduped by xet)[/]", soft_wrap=True)
    out.print(f"[sb.key]  {path}[/]", soft_wrap=True)
    out.print(f"[sb.hint]  slotbank run {model_arg}[/]", soft_wrap=True)


# --------------------------------------------------------------------------
# home screen and list, per docs/cli-design.md sections 4.6 and 5
# --------------------------------------------------------------------------

COMMANDS = [
    ("search", "<name>", "find models on the Hub"),
    ("compare", "<model>", "every quant of a model, from one probe"),
    ("check", "<model>", "will it run here -- without downloading it"),
    ("pull", "<model>", "download it, refusing one that cannot run"),
    ("run", "<model>", "chat with it, model stays loaded"),
    ("serve", "--model <m>", "serve it to Claude Code / Codex / OpenCode"),
    ("list", "", "what is downloaded, and what each costs here"),
    ("use", "<role> <model>", "which model answers as chat / reasoning"),
    ("rm", "<model>", "delete a cached model"),
]


def home(total_bytes: int, working_set: int, n_models: int, on_disk: float,
         out=None) -> None:
    """The screen a bare `slotbank` prints.

    argparse's usage block lists commands alphabetically and says nothing about
    the machine. What a first-time user needs is the order to do things in and
    what this particular Mac can hold.
    """
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    con = out or _consoles()[0]
    G = float(1 << 30)
    head = Text()
    head.append("slotbank", style="sb.repo")
    head.append("   run MoE models whose expert bank does not fit in memory\n",
                style="sb.key")
    head.append(f"{total_bytes / G:.0f} GiB unified", style="sb.num")
    head.append(" · ", style="sb.key")
    head.append(f"{working_set / G:.1f} GiB working set", style="sb.num")
    head.append(" · ", style="sb.key")
    head.append(f"{n_models} model{'' if n_models == 1 else 's'}", style="sb.num")
    head.append(" · ", style="sb.key")
    head.append(f"{on_disk:.0f} GiB on disk", style="sb.num")
    con.print(Panel(head, border_style="sb.key", padding=(0, 2)))

    t = Table.grid(padding=(0, 1))
    t.add_column(style="sb.ok", no_wrap=True)
    t.add_column(style="sb.num", no_wrap=True)
    t.add_column(style="sb.key")
    for name, arg, blurb in COMMANDS:
        t.add_row(f"  {name}", arg, blurb)
    con.print(t)
    con.print("\n  new here? ", style="sb.key", end="")
    con.print("slotbank search qwen3 moe", style="sb.ok", end="")
    con.print("  then  ", style="sb.key", end="")
    con.print("slotbank check <id>", style="sb.ok")
    con.print("  every command takes a short name, a repo id, or a "
              "@role.\n", style="sb.hint")


def model_table(rows, out=None) -> None:
    """`slotbank list`: what is downloaded and what it costs *here*.

    Size alone is what every other tool prints and it is the least useful
    column -- the question is whether it runs, which is wired bytes against
    this machine's working set.
    """
    from rich.table import Table

    con = out or _consoles()[0]
    t = Table(box=None, pad_edge=False, header_style="sb.key")
    t.add_column("MODEL", style="sb.repo", no_wrap=True)
    t.add_column("ON DISK", justify="right", style="sb.num")
    t.add_column("EXPERTS", justify="right", style="sb.num")
    t.add_column("WIRED", justify="right", style="sb.num")
    t.add_column("STATE")
    for name, on_disk, experts, wired, state in rows:
        style = ("sb.ok" if state.startswith("ready") and "only" not in state
                 else "sb.warn" if "only" in state or "dense" in state
                 else "sb.bad" if "not" in state else "sb.key")
        t.add_row(name, on_disk, experts, wired, f"[{style}]{state}[/{style}]")
    con.print(t)
