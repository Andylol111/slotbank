from __future__ import annotations

from slotbank.decode import (
    completion_cap,
    finish_text,
    merge_stops,
    parse_tool_calls,
    split_think,
    SpecialHoldback,
    strip_special,
)
from slotbank.prompt import normalize_messages
from slotbank.runtime import reuse_prefill_start


def test_split_think():
    reasoning, body = split_think("<think>plan</think>\nhello")
    assert reasoning == "plan"
    assert body == "hello"


def test_split_think_unclosed():
    reasoning, body = split_think("<think>still planning")
    assert reasoning == "still planning"
    assert body == ""


def test_split_think_close_without_open():
    reasoning, body = split_think("scratch</think>\nHi. What do you need?")
    assert reasoning == "scratch"
    assert body == "Hi. What do you need?"


def test_parse_tool_call_json():
    calls = parse_tool_calls('<tool_call>{"name": "ls", "arguments": {"path": "."}}</tool_call>')
    assert len(calls) == 1
    assert calls[0].name == "ls"
    assert '"path"' in calls[0].arguments


def test_finish_text_strips_markup():
    body, reasoning, calls = finish_text(
        '<think>x</think>\n<tool_call>{"name": "ls", "arguments": {}}</tool_call>'
    )
    assert reasoning == "x"
    assert calls[0].name == "ls"
    assert "tool_call" not in body


def test_finish_text_drops_streamed_eos():
    body, reasoning, calls = finish_text(
        "<think>The user just said hi.</think>\nHi. What do you need?<|im_end|>"
    )
    assert reasoning == "The user just said hi."
    assert body == "Hi. What do you need?"
    assert "<|im_end|>" not in body
    assert not calls


def test_strip_special_and_merge_stops():
    assert strip_special("Hi.<|im_end|>") == "Hi."
    assert "<|im_end|>" in merge_stops(["\n"])
    assert merge_stops(["<|im_end|>"]).count("<|im_end|>") == 1


def test_special_holdback_split_eos():
    scrub = SpecialHoldback()
    assert scrub.push("<|im") == ""
    assert scrub.push("_end|>") == ""
    assert scrub.flush() == ""


def test_special_holdback_keeps_real_text():
    scrub = SpecialHoldback()
    assert scrub.push("Hi. What do you need?") == "Hi. What do you need?"
    assert scrub.push("<|im_end|>") == ""
    assert scrub.flush() == ""


def test_special_holdback_drops_think_then_answer():
    scrub = SpecialHoldback()
    blob = (
        'The user just said "hi". This is a simple greeting.</think>\n'
        "Hey! What are you working on in slotbank?<|im_end|>"
    )
    assert scrub.push(blob) == "Hey! What are you working on in slotbank?"
    assert scrub.flush() == ""


def test_special_holdback_holds_open_think():
    scrub = SpecialHoldback()
    assert scrub.push("<think>plan") == ""
    assert scrub.push("</think>\nHi there.<|im_end|>") == "Hi there."
    assert scrub.flush() == ""


def test_completion_cap(monkeypatch):
    monkeypatch.delenv("SLOTBANK_MAX_COMPLETION", raising=False)
    assert completion_cap(8192) == 2048
    assert completion_cap(128) == 128
    assert completion_cap(None) == 1024
    monkeypatch.setenv("SLOTBANK_MAX_COMPLETION", "512")
    assert completion_cap(8192) == 512
    monkeypatch.setenv("SLOTBANK_MAX_COMPLETION", "junk")
    assert completion_cap(9999) == 2048


def test_normalize_developer_to_system():
    msgs = normalize_messages([{"role": "developer", "content": "rules"}])
    assert msgs == [{"role": "system", "content": "rules"}]


def test_reuse_prefill_start():
    assert reuse_prefill_start([1, 2], [1, 2, 3]) == 2
    assert reuse_prefill_start([1, 9], [1, 2, 3]) == 0
    assert reuse_prefill_start([], [1, 2]) == 0
