from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Iterator

from slotbank.decode import completion_cap, finish_text, merge_stops, SpecialHoldback
from slotbank.layout import parse_byte_size
from slotbank.prompt import apply_qwen_sampling, encode_chat, encode_text
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
        from slotbank.omp import DEFAULT_CONTEXT_WINDOW

        self.model_id = model_id or public_model_id(model_path)
        self.context_window = DEFAULT_CONTEXT_WINDOW
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
        # Never call runtime.close() here. MLX arrays and mx.clear_cache()
        # belong to the worker (same thread as load). On Python 3.13 that
        # path fatal-aborts: PyThreadState_Get / GIL released, then SIGTRAP.
        try:
            self._jobs.put(None)
        except Exception:
            return
        self._thread.join(timeout=30)

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
        sampling = SamplingParams(
            temperature=sampling.temperature,
            top_k=sampling.top_k,
            top_p=sampling.top_p,
            ignore_eos=sampling.ignore_eos,
            max_tokens=completion_cap(sampling.max_tokens),
            stop_strs=merge_stops(sampling.stop_strs),
            qwen_mode=sampling.qwen_mode,
        )
        sampling = apply_qwen_sampling(sampling, sampling.qwen_mode)
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
                # Teardown on this thread only. Caller-thread close() used to
                # mx.clear_cache after join and GIL-abort on 3.13.
                try:
                    self.runtime.close()
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
        scrub = SpecialHoldback()
        for step in self.runtime.iter_steps():
            generated.append(int(step.token_id))
            text = tok.decode(generated)
            piece = text[len(prev):]
            prev = text
            emit = scrub.push(piece)
            if step.finished:
                emit += scrub.flush()
            if emit:
                job.out.put(("tok", int(step.token_id), emit))
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
