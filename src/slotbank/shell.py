"""Persistent interactive shell.

The interaction architecture is Claude Code's: one process that stays alive, a
`/` command surface, and plain text going to the model. It is not a widget TUI
-- there is no mouse, and a REPL with slash commands is what the reference
actually is.

Why persistent matters here more than for a hosted model: loading the 35B costs
~3.4 s and its first prefill several seconds more, and the page cache the model
warms does not survive the process. A one-shot CLI pays all of that per message.

Imports mlx only through Engine, and only when a chat message is actually sent,
so `/search` and `/check` stay fast in a session that never loads a model.
"""

from __future__ import annotations

import logging
import os
import sys
import time

from slotbank import ui

# transformers logs a BPE cleanup warning on some tokenizers, mid-stream, into
# the middle of the model's reply. It is advisory and there is nothing for the
# user to do about it.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

BANNER = "slotbank"
HELP = [
    ("/model [name]", "show or switch the loaded model"),
    ("/search <q>", "find models on the Hub"),
    ("/compare <m>", "every quant of a model, from one probe"),
    ("/check <m>", "will it run here -- no download"),
    ("/list", "what is downloaded"),
    ("/use <role> <m>", "bind a role: chat, reasoning"),
    ("/memory", "what this machine is holding right now"),
    ("/think on|off", "let the model reason before answering"),
    ("/temp <n>", "sampling temperature (0.6 default; 0 loops)"),
    ("/clear", "forget the conversation, keep the model loaded"),
    ("/help", "this list"),
    ("/exit", "leave (ctrl-d also works)"),
]


def _state_of(repo: str) -> tuple[str, float]:
    """Reuse the CLI's offline estimate rather than a second copy of it."""
    from slotbank.cli import _role_state
    from slotbank.layout import detect_device_profile

    budget = detect_device_profile(leave_free_bytes=None).max_working_set_bytes
    state, wired = _role_state(repo, budget)
    return ("dense" if "dense" in state else state), wired


class Shell:
    def __init__(self, model: str | None = None, leave_free=None):
        self.model_arg = model
        self.leave_free = leave_free
        self.engine = None
        self.repo = None
        self.history: list[dict] = []
        self.temp, self.top_p, self.top_k = 0.6, 0.95, 20
        self.thinking = True
        self.max_tokens = 2048
        self.con, self.err = ui._consoles()

    # ---------------------------------------------------------------- chrome
    def header(self, clear: bool = False) -> None:
        from rich.panel import Panel
        from rich.text import Text

        from slotbank.layout import detect_device_profile
        from slotbank.registry import local_models

        # Start at the top of the screen the way a full-screen tool does, so the
        # panel is not buried under whatever the shell printed before it. Only
        # on a real terminal -- clearing a pipe emits escape codes into a file.
        if clear and self.con.is_terminal:
            self.con.clear()
        G = float(1 << 30)
        prof = detect_device_profile(leave_free_bytes=None)
        models = local_models()
        t = Text()
        t.append(BANNER, style="sb.repo")
        t.append("   type to chat · ", style="sb.key")
        t.append("/help", style="sb.ok")
        t.append(" for commands\n", style="sb.key")
        loaded = self.repo.split("/")[-1] if self.repo else "no model loaded"
        t.append(loaded, style="sb.num" if self.repo else "sb.key")
        t.append(" · ", style="sb.key")
        t.append(f"{prof.total_bytes / G:.0f} GiB unified", style="sb.num")
        t.append(" · ", style="sb.key")
        t.append(f"{len(models)} local", style="sb.num")
        self.con.print(Panel(t, border_style="sb.key", padding=(0, 2)))

    def status(self, msg: str, style: str = "sb.key") -> None:
        self.err.print(f"  {msg}", style=style)

    # ---------------------------------------------------------------- model
    def default_model(self) -> str | None:
        """What to load when the user did not say.

        The @chat role first -- that is what `slotbank use` exists to set --
        then the largest cached MoE, because a dense model gets no benefit from
        this runtime and picking one would misrepresent the tool.
        """
        from slotbank.registry import load_roles, local_models

        chat = (load_roles().get("chat") or {}).get("model")
        if chat:
            return chat
        # Prefer a MoE (a dense model gets no benefit from this runtime) that
        # can actually hold a conversation. A base model with no chat template
        # answers by continuing a transcript, inventing both sides.
        chat_moe, moe, chat_dense, dense = [], [], [], []
        for m in local_models():
            is_dense = _state_of(m.repo_id)[0] == "dense"
            tmpl = m.has_chat_template
            (chat_dense if is_dense and tmpl else dense if is_dense
             else chat_moe if tmpl else moe).append(m)
        for group in (chat_moe, chat_dense, moe, dense):
            if group:
                return group[0].repo_id
        return None

    def _quiet(self) -> None:
        for name in ("transformers", "transformers.tokenization_utils_base"):
            logging.getLogger(name).setLevel(logging.ERROR)

    def load(self, name: str | None = None) -> bool:
        from slotbank.registry import local_path, resolve

        target = name or self.model_arg or self.default_model()
        if target is None:
            self.status("no models downloaded yet. /search <name>, then /pull",
                        "sb.warn")
            return False
        repo = resolve(target)
        path = local_path(repo)
        if path is None:
            self.status(f"{repo} is not downloaded -- /pull it first", "sb.bad")
            return False
        if self.engine is not None:
            self.close()
        from slotbank.engine import Engine

        self._quiet()
        t0 = time.perf_counter()
        with self.err.status(f"loading {repo.split('/')[-1]}...", spinner="dots"):
            self.engine = Engine(path, leave_free=self.leave_free, model_id=repo)
        self.repo = repo
        self.history = []
        self.status(f"loaded in {time.perf_counter() - t0:.1f}s", "sb.ok")
        return True

    def sampling(self):
        """Never greedy.

        temperature=0.0 makes a reasoning model degenerate: the argmax keeps
        landing on the same continuation, so it loops "Wait... Okay, final...
        Wait..." until it hits the token cap and never emits an answer.
        Observed on Qwen3.5-35B answering "hi" -- 1024 tokens, no reply.
        These are the values Qwen documents for thinking mode.
        """
        from slotbank.types import SamplingParams

        return SamplingParams(temperature=self.temp, top_p=self.top_p,
                              top_k=self.top_k, max_tokens=self.max_tokens)

    def prompt(self) -> str:
        """The panel is a snapshot; this is live. A model loaded lazily on the
        first message would otherwise leave the header saying 'no model loaded'
        for the rest of the session."""
        if not self.repo:
            return "\n\033[1m> \033[0m"
        name = self.repo.split("/")[-1]
        for tag in ("-4bit", "-8bit", "-bf16", "-Instruct"):
            name = name.replace(tag, "")
        return f"\n\033[2m{name}\033[0m \033[1m> \033[0m"

    def close(self) -> None:
        """Cancel first, then close. Engine.close joins its worker with a 5 s
        timeout, so closing during a generation hangs for those 5 seconds and
        reads as 'exit did not work'."""
        if self.engine is not None:
            try:
                self.engine.runtime.cancel()
            except Exception:
                pass
            try:
                self.engine.close()
            except Exception:
                pass
            self.engine = None

    # ----------------------------------------------------------------- chat
    def say(self, text: str) -> None:
        from slotbank.types import SamplingParams

        if self.engine is None and not self.load():
            return
        self.history.append({"role": "user", "content": text})
        ids = self.engine.tokenize_chat(self.history, None)
        n = {"c": 0, "t0": None}
        self.con.print()

        def on_token(_tid, piece):
            if n["t0"] is None:
                n["t0"] = time.perf_counter()
            n["c"] += 1
            sys.stdout.write(piece)
            sys.stdout.flush()

        start = time.perf_counter()
        try:
            out = self.engine.generate(ids, self.sampling(), on_token=on_token)
        except KeyboardInterrupt:
            sys.stdout.write("\n")
            self.status("interrupted", "sb.warn")
            self.history.pop()
            return
        sys.stdout.write("\n")
        self.history.append({"role": "assistant", "content": out.content})
        if n["t0"] and n["c"] > 1:
            rate = (n["c"] - 1) / max(time.perf_counter() - n["t0"], 1e-9)
            self.status(f"{n['c']} tokens · {rate:.2f} tok/s · "
                        f"first token {n['t0'] - start:.1f}s")

    # ------------------------------------------------------------- commands
    def command(self, line: str) -> bool:
        """Returns False to exit the shell."""
        import shlex

        try:
            parts = shlex.split(line[1:])
        except ValueError:
            parts = line[1:].split()
        if not parts:
            return True
        cmd, rest = parts[0], parts[1:]
        from slotbank import cli

        if cmd in ("exit", "quit", "bye"):
            return False
        if cmd == "help":
            from rich.table import Table

            t = Table.grid(padding=(0, 2))
            t.add_column(style="sb.ok", no_wrap=True)
            t.add_column(style="sb.key")
            for name, blurb in HELP:
                t.add_row(f"  {name}", blurb)
            self.con.print(t)
            self.con.print("  anything else is sent to the model.\n",
                           style="sb.hint")
        elif cmd == "clear":
            self.history = []
            self.header(clear=True)
            self.status("conversation cleared, model still loaded", "sb.ok")
        elif cmd == "model":
            if rest:
                self.load(rest[0])
            else:
                self._models()
        elif cmd == "think":
            on = not rest or rest[0].lower() not in ("off", "no", "0", "false")
            self.thinking = on
            self.status(f"thinking {'on' if on else 'off'}"
                        + ("" if on else " -- if the model's template supports it"),
                        "sb.ok")
        elif cmd == "temp":
            try:
                self.temp = float(rest[0])
                self.status(f"temperature {self.temp}", "sb.ok")
            except (IndexError, ValueError):
                self.status(f"temperature is {self.temp}; /temp 0.6")
        elif cmd == "memory":
            self._memory()
        elif cmd in ("search", "check", "compare", "list", "use", "pull", "rm"):
            self._delegate(cmd, rest)
        else:
            self.status(f"unknown command /{cmd} -- /help", "sb.warn")
        return True

    def _delegate(self, cmd: str, rest: list[str]) -> None:
        """Run a real subcommand in-process, so the shell and the CLI cannot
        drift apart. argparse exits on error, which must not kill the shell."""
        from slotbank.cli import main

        try:
            main([cmd, *rest])
        except SystemExit as exc:
            if exc.code not in (0, 1, None):
                self.status(f"/{cmd} needs an argument -- /help", "sb.warn")
        except (ValueError, RuntimeError, OSError) as exc:
            self.status(str(exc), "sb.bad")

    def _models(self) -> None:
        """`/model` with no argument. A bare 'no model loaded' is a dead end;
        what the user needs is the list and the command to switch."""
        from slotbank.registry import local_models

        models = local_models()
        if not models:
            self.status("nothing downloaded. /search <name>, then /pull",
                        "sb.warn")
            return
        for m in models:
            name = m.repo_id.split("/")[-1]
            state, wired = _state_of(m.repo_id)
            here = "* " if m.repo_id == self.repo else "  "
            note = ("no benefit, dense" if state == "dense"
                    else f"{wired:.1f}G wired" if wired else state)
            style = "sb.ok" if m.repo_id == self.repo else "sb.key"
            self.con.print(f"  {here}[{style}]{name:<34}[/{style}]"
                           f"[sb.key]{note}[/sb.key]")
        self.con.print("  /model <name> to switch\n", style="sb.hint")

    def _memory(self) -> None:
        import subprocess

        from slotbank.um import snapshot_from_vm_stat

        G = float(1 << 30)
        x = snapshot_from_vm_stat(
            subprocess.run(["vm_stat"], capture_output=True, text=True).stdout)
        level = {1: ("normal", "sb.ok"), 2: ("warn", "sb.warn")}.get(
            x.pressure, ("critical", "sb.bad"))
        self.con.print(
            f"  reclaimable [sb.num]{x.reclaimable_bytes / G:.1f} GiB[/sb.num]"
            f" · cached [sb.num]{x.file_backed_bytes / G:.1f} GiB[/sb.num]"
            f" · pressure [{level[1]}]{level[0]}[/{level[1]}]")
        if self.engine is not None:
            # Process RSS, not mx.get_active_memory: the import fence keeps mlx
            # out of this module, and RSS is what the OS actually charges us.
            rss = int(subprocess.run(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                capture_output=True, text=True).stdout.strip() or 0) * 1024
            self.con.print(f"  slotbank RSS [sb.num]{rss / G:.2f} GiB[/sb.num]")

    # ------------------------------------------------------------------ loop
    def run(self) -> int:
        if self.con.is_terminal:
            self.con.clear()
        # Load before the header so the panel is accurate, and load even when
        # no model was named: opening the tool and finding nothing ready is the
        # thing this shell exists to avoid.
        self.load()
        self.header()
        try:
            while True:
                try:
                    line = input(self.prompt()).strip()
                except EOFError:
                    self.con.print()
                    break
                except KeyboardInterrupt:
                    self.con.print()
                    continue
                if not line:
                    continue
                if line.startswith("!"):
                    os.system(line[1:])
                    continue
                if line.startswith("/"):
                    if not self.command(line):
                        break
                    continue
                self.say(line)
        finally:
            self.close()
        return 0
