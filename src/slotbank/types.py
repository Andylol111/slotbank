from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024
    stop_strs: list[str] = field(default_factory=list)
    # Qwen3.8 /think vs /no_think. None = not an OMP/Qwen turn.
    qwen_mode: str | None = None

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(frozen=True)
class GenerationStep:
    token_id: int
    finished: bool
    finish_reason: str | None = None
    matched_stop: str | None = None
    active_memory_bytes: int = 0


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: str
    call_id: str = ""


@dataclass
class GenResult:
    content: str
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    matched_stop: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    draft_kind: str | None = None
    draft_block: int | None = None
    draft_accept_rate: float | None = None


class RuntimeProtocol(Protocol):
    def start_request(self, input_ids: list[int], sampling_params: SamplingParams) -> None: ...
    def step(self) -> GenerationStep: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...
