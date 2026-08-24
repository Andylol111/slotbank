from __future__ import annotations

from slotbank.decode import finish_text, parse_tool_calls, split_think
from slotbank.prompt import normalize_messages
from slotbank.runtime import reuse_prefill_start


def test_split_think():
    reasoning, body = split_think("<think>plan</think>\nhello")
    assert reasoning == "plan"
    assert body == "hello"


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


def test_normalize_developer_to_system():
    msgs = normalize_messages([{"role": "developer", "content": "rules"}])
    assert msgs == [{"role": "system", "content": "rules"}]


def test_reuse_prefill_start():
    assert reuse_prefill_start([1, 2], [1, 2, 3]) == 2
    assert reuse_prefill_start([1, 9], [1, 2, 3]) == 0
    assert reuse_prefill_start([], [1, 2]) == 0
