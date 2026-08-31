from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Iterator

from slotbank.decode import finish_text
from slotbank.layout import parse_byte_size
from slotbank.prompt import encode_chat, encode_text
from slotbank.types import GenResult, SamplingParams


@dataclass
class Job:
    input_ids: list[int]
    sampling: SamplingParams
    out: queue.Queue


class Engine:
    def __init__(self, model_path: str, *, leave_free: int | None = None,
                 progress=None, model_id: str | None = None):
        from slotbank.runtime import Runtime
        from slotbank.um import UmManager

        args = SimpleNamespace(
            model_path=model_path,
            leave_free=leave_free,
            prefill_step_size=2048,
        )
        self.um = UmManager.from_args(args)
        self.runtime = Runtime(args, um=self.um)
        # An HF snapshot directory is named after the commit hash, so without
        # an explicit name every client would list "23511b94..." as the model.
        from slotbank.admit import public_model_id

        self.model_id = model_id or public_model_id(model_path)
        self.context_window = 16384
        self._jobs: queue.Queue[Job | None] = queue.Queue()
        # Load on the worker thread, not here. MLX arrays are bound to the
        # thread that created them, so loading on the main thread and
        # generating on this one fails with "no Stream(cpu, 0) in current
        # thread". This is also what "the Metal thread is exclusive" implies.
        self._progress = progress
        self._ready = threading.Event()
        self._load_error: str | None = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._load_error is not None:
            raise RuntimeError(self._load_error)

    def close(self) -> None:
        self._jobs.put(None)
        self._thread.join(timeout=5)
        self.runtime.close()

    def tokenize_chat(self, messages: list[dict], tools: list[dict] | None) -> list[int]:
        return encode_chat(self.runtime.tokenizer, messages, tools)

    def tokenize_text(self, text: str) -> list[int]:
        return encode_text(self.runtime.tokenizer, text)

    def generate(
        self,
        input_ids: list[int],
        sampling: SamplingParams,
        on_token: Callable[[int, str], None] | None = None,
    ) -> GenResult:
        job = Job(input_ids=input_ids, sampling=sampling, out=queue.Queue())
        self._jobs.put(job)
        pieces: list[str] = []
        ids: list[int] = []
        finish = "stop"
        matched = None
        while True:
            item = job.out.get()
            if item[0] == "tok":
                tid, piece = item[1], item[2]
                ids.append(tid)
                pieces.append(piece)
                if on_token is not None:
                    on_token(tid, piece)
            elif item[0] == "done":
                finish, matched = item[1], item[2]
                break
            elif item[0] == "err":
                raise RuntimeError(item[1])
        raw = "".join(pieces)
        content, reasoning, calls = finish_text(raw)
        if calls:
            finish = "tool_calls"
        kind, block, rate = self.runtime.draft_report()
        return GenResult(
            content=content,
            reasoning=reasoning,
            tool_calls=calls,
            finish_reason=finish,
            matched_stop=matched,
            prompt_tokens=len(input_ids),
            completion_tokens=len(ids),
            draft_kind=kind,
            draft_block=block,
            draft_accept_rate=rate,
        )

    def stream(self, input_ids: list[int], sampling: SamplingParams) -> Iterator[tuple[str, object]]:
        q: queue.Queue = queue.Queue()

        def on_token(_tid: int, piece: str) -> None:
            q.put(("delta", piece))

        def run() -> None:
            try:
                result = self.generate(input_ids, sampling, on_token=on_token)
                q.put(("result", result))
            except Exception as exc:
                q.put(("err", str(exc)))

        threading.Thread(target=run, daemon=True).start()
        while True:
            kind, payload = q.get()
            yield kind, payload
            if kind in {"result", "err"}:
                return

    def _loop(self) -> None:
        try:
            self.runtime.load(progress=self._progress)
        except Exception as exc:               # surface load failures to __init__
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._ready.set()
            return
        self._ready.set()
        while True:
            job = self._jobs.get()
            if job is None:
                # Save here, not in close(). The slot-id arrays belong to this
                # thread, so mx.eval on the caller's thread raises "no
                # Stream(gpu, 1)" -- and close() runs in a finally, so that
                # error replaces whatever really went wrong.
                try:
                    self.runtime.save_profile()
                except Exception:
                    pass
                return
            try:
                self._run_job(job)
            except Exception as exc:
                import traceback

                job.out.put(("err", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))

    def _run_job(self, job: Job) -> None:
        tok = self.runtime.tokenizer
        self.runtime.start_request(job.input_ids, job.sampling)
        prev = ""
        generated: list[int] = []
        finish = "stop"
        matched = None
        for step in self.runtime.iter_steps():
            generated.append(int(step.token_id))
            text = tok.decode(generated)
            piece = text[len(prev):]
            prev = text
            job.out.put(("tok", int(step.token_id), piece))
            if step.finished:
                finish = step.finish_reason or "stop"
                matched = step.matched_stop
                break
        self.runtime.shed_if_needed()
        job.out.put(("done", finish, matched))


def leave_free_arg(text: str | None) -> int | None:
    if text is None or text == "":
        return None
    return parse_byte_size(text)
