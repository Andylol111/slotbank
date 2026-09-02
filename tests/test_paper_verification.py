"""Conference-paper verification: every claim id in paper_claims.py has a test.

These tests do not import MLX. Air tok/s are pins, not retimes.
When SLOTBANK_VERIF_OUT is set, the envelope suite writes suite_latest.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from paper_claims import CLAIMS, OPTIONAL_DEP_FAILURES, RESEARCH_QUESTIONS
from slotbank.layout import GIB, admit, device_profile, model_memory_card, slot_capacity
from slotbank.prompt import (
    DEFAULT_ENVELOPE_MAX_PROMPT,
    ENVELOPE_SYS_TOKENS,
    TOOL_SLIM_BUDGET,
    apply_qwen_sampling,
    condense_harness_messages,
    encode_chat,
    keep_token_ids,
    slim_tools,
)
from slotbank.tps import (
    ADOPTED,
    M4_AIR_24G,
    REJECTED,
    STRATEGIES,
    air_scale,
    catalog_sound,
    pack_read_ceiling_toks,
    review_mac_speedup,
)


def _approx_tok(text: str) -> int:
    return max(0, (len(text) + 3) // 4)


def _cwd_dump(n_files: int, ask: str, child: str = "llama.cpp-dflash2") -> str:
    tree = "\n".join(f"src/file_{i}.py" for i in range(n_files))
    return (
        "╭── π  │ ⬢ Qwen3.8-27B-4bit-agent\n"
        + tree
        + f"\n╰─  /tmp ↳ {child} ─╯\n\n{ask}"
    )


def _file_dump(n_lines: int, ask: str, path: str = "src/foo.py") -> str:
    return f"file:{path}:1-{n_lines}\n" + ("LINE\n" * n_lines) + f"\n\n{ask}"


class _Tok:
    """Deterministic stand-in for Qwen apply_chat_template."""

    def apply_chat_template(self, msgs, **k):
        ids = []
        for m in msgs:
            ids.append({"system": 1, "user": 2, "assistant": 3}.get(m["role"], 4))
            ids.extend((ord(c) % 40) + 10 for c in str(m.get("content") or ""))
            ids.append(9)
        if k.get("add_generation_prompt"):
            ids += [8, 7]
        return ids

    def encode(self, text):
        return [0]


def _dump_json(name: str, payload) -> None:
    dest = os.environ.get("SLOTBANK_VERIF_OUT", "").strip()
    if not dest:
        return
    path = Path(dest)
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(json.dumps(payload, indent=2, default=str) + "\n")


# --- claim registry ----------------------------------------------------------


def test_C_claim_registry_is_unique():
    ids = [c["id"] for c in CLAIMS]
    assert len(ids) == len(set(ids))
    kinds = {c["kind"] for c in CLAIMS}
    assert kinds <= {"verified_here", "air_pinned", "cannot_retime"}
    rqs = {q["id"] for q in RESEARCH_QUESTIONS}
    assert {c["rq"] for c in CLAIMS} <= rqs
    for c in CLAIMS:
        assert c["test"].startswith("tests/")
        assert "::" in c["test"]


def test_every_claim_has_a_named_test():
    import tests.test_paper_verification as here
    import tests.test_fence as fence

    have = {n for n in dir(here) if n.startswith("test_")} | {
        n for n in dir(fence) if n.startswith("test_")
    }
    for c in CLAIMS:
        fn = c["test"].rsplit("::", 1)[-1]
        assert fn in have, (c["id"], fn)


# --- C-fence -----------------------------------------------------------------


def test_C_fence():
    from tests.test_fence import (
        test_no_brand_or_torch_in_source,
        test_only_three_files_import_mlx,
        test_package_import_does_not_load_mlx,
    )

    test_only_three_files_import_mlx()
    test_no_brand_or_torch_in_source()
    test_package_import_does_not_load_mlx()


# --- C-catalog / C-clocks / C-pack-ceil / C-mac200 / C-scale -----------------


def test_C_catalog():
    catalog_sound()
    banned = {
        "ane-npu-prefill",
        "qwen-mac-200pct",
        "qwen35-4b-as-27b-drafter",
        "hybrid-kv-dynamic-page",
        "kv-quant-with-draft",
        "spec-prefill-sparse",
        "gdn-chunked-cuda-prefill",
        "distserve-pd",
    }
    for sid in banned:
        assert next(s for s in STRATEGIES if s.id == sid).status == REJECTED
    adopted = [s for s in STRATEGIES if s.status == ADOPTED]
    assert adopted
    for s in adopted:
        assert not s.needs_trim_cache
        assert not s.changes_target_weights
    statuses = {s.status for s in STRATEGIES}
    assert statuses <= {"adopted", "attempted", "in_tree", "deferred", "rejected"}
    _dump_json(
        "catalog_snapshot.json",
        [{"id": s.id, "status": s.status} for s in STRATEGIES],
    )


def test_C_clocks():
    a = M4_AIR_24G
    assert a["greedy_toks"] == pytest.approx(5.71)
    assert a["mtp_k3_count"] == pytest.approx(13.47)
    assert a["mtp_k3_code"] == pytest.approx(9.95)
    assert a["dflash_k8_count"] == pytest.approx(11.76)
    assert a["dflash_k8_code"] == pytest.approx(9.10)
    assert a["prefill_819_s"] == pytest.approx(17.0)
    assert a["prefill_819_reuse_s"] == pytest.approx(0.88)
    assert a["mtp_k3_count"] > a["dflash_k8_count"]
    assert a["prefill_819_reuse_s"] < a["prefill_819_s"] / 10
    assert a["mtp_k3_count"] / a["greedy_toks"] == pytest.approx(2.36, abs=0.02)
    assert a["bandwidth_bytes_s"] == 120 << 30
    assert a["weight_bytes_4bit"] == 15 << 30


def test_C_pack_ceil():
    ceil = pack_read_ceiling_toks(M4_AIR_24G["weight_bytes_4bit"])
    assert 7.0 <= ceil <= 9.0
    assert ceil == pytest.approx(8.0, abs=0.2)


def test_C_mac200():
    review = review_mac_speedup()
    assert review["official_qwen_mac_toks_rows"] == 0
    assert "omlx-ane-prefill-4k" in review["over_200pct_ids"]
    assert review["ane_fits_24g_air"] is False
    assert review["ane_prefill_increase_pct"] >= 200
    assert review["over_200pct_are_ane_or_cache"] is True
    _dump_json("mac_review.json", review)


def test_C_scale():
    assert air_scale(19.8, 400) == pytest.approx(5.94, rel=0.01)
    assert air_scale(32.6, 400) == pytest.approx(9.78, rel=0.01)
    assert abs(air_scale(19.8, 400) - M4_AIR_24G["greedy_toks"]) < 0.3
    assert abs(air_scale(32.6, 400) - M4_AIR_24G["mtp_k3_code"]) < 0.3


def test_C_hybrid_kv():
    from slotbank.admit import check_speculative_supported, kv_bytes_per_token

    cfg = {
        "text_config": {
            "layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
        }
    }
    assert kv_bytes_per_token(cfg) == 64 << 10

    class Recurrent:
        def is_trimmable(self):
            return False

    msg = check_speculative_supported([Recurrent()])
    assert msg and "not trimmable" in msg


def test_C_trained_k(tmp_path):
    from slotbank.admit import draft_block_from_config

    d = tmp_path / "d"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"dflash_config": {"block_size": 8}, "block_size": 4})
    )
    assert draft_block_from_config(str(d)) == 8
    m = tmp_path / "m"
    m.mkdir()
    (m / "config.json").write_text(json.dumps({"block_size": 3}))
    assert draft_block_from_config(str(m)) == 3
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "config.json").write_text(json.dumps({}))
    # Fallback 5 is only when the checkpoint has no trained block.
    assert draft_block_from_config(str(empty)) == 5


def test_C_leave_free():
    from slotbank.layout import recommended_leave_free

    assert recommended_leave_free(24 * GIB) == 8 * GIB
    sidecar = 239 * (1 << 20)
    vision = int(0.4 * GIB)
    kv_8k = 8192 * (64 << 10)
    extra = sidecar + vision + kv_8k
    pack = 15 * GIB
    tight = device_profile(24 * GIB, leave_free_bytes=8 * GIB)
    roomy = device_profile(24 * GIB, leave_free_bytes=6 * GIB)
    assert roomy.max_working_set_bytes == 18 * GIB
    assert tight.max_working_set_bytes == 16 * GIB
    assert roomy.fits(pack, extra)
    assert not tight.fits(pack, extra)
    math_card = model_memory_card(27_780_000_000, 4.0, kind="dense")
    assert 13 * GIB < math_card.stored_bytes < 16 * GIB
    assert admit(roomy, math_card, extra).ok
    # Affine-4 math on 27.78B is ~13.7 GiB; the advertised pack is 15 GiB.
    assert not tight.fits(pack, extra)


def test_C_slots_c():
    from slotbank.layout import slot_floor

    assert slot_floor(256, 8) == 16
    p = device_profile(24 * GIB)
    moe = model_memory_card(
        35_000_000_000, 4.0, kind="moe", n_routed_experts=256, top_k=8
    )
    c = slot_capacity(
        256, 8, stored_bytes=moe.stored_bytes, working_set_bytes=p.max_working_set_bytes
    )
    assert 16 < c < 256
    olmoe = model_memory_card(
        7_000_000_000, 4.0, kind="moe", n_routed_experts=64, top_k=8
    )
    c2 = slot_capacity(
        64, 8, stored_bytes=olmoe.stored_bytes, working_set_bytes=p.max_working_set_bytes
    )
    assert c2 == 16


def test_C_sys_cap(monkeypatch):
    monkeypatch.setenv("SLOTBANK_ENVELOPE", "1")
    fat = "You are OMP. " + ("tools " * 4000)
    dump = _file_dump(4000, "hi")
    first = condense_harness_messages(
        [{"role": "system", "content": fat}, {"role": "user", "content": dump}],
        budget=8192,
    )
    follow = condense_harness_messages(
        [
            {"role": "system", "content": fat},
            {"role": "user", "content": dump},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "again " * 40},
        ],
        budget=8192,
    )
    assert _approx_tok(first[0]["content"]) <= ENVELOPE_SYS_TOKENS + 8
    assert first[0]["content"] == follow[0]["content"]


def test_C_history():
    dump = _file_dump(4000, "please review the diff")
    sys_m = {"role": "system", "content": "You are OMP."}
    first = condense_harness_messages([sys_m, {"role": "user", "content": dump}], budget=400)
    follow = condense_harness_messages(
        [
            sys_m,
            {"role": "user", "content": dump},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "and the tests"},
        ],
        budget=400,
    )
    assert first[1]["content"] == follow[1]["content"]
    assert "please review the diff" in first[1]["content"]


def test_C_prefix(monkeypatch):
    monkeypatch.delenv("SLOTBANK_CONTEXT_DIR", raising=False)
    monkeypatch.delenv("SLOTBANK_DIRECT", raising=False)
    monkeypatch.setenv("SLOTBANK_ENVELOPE", "1")
    monkeypatch.setenv("SLOTBANK_MAX_PROMPT", "0")
    sys_m = {"role": "system", "content": "You are a local assistant on this machine."}
    first = encode_chat(_Tok(), [sys_m, {"role": "user", "content": "hi"}], None)
    follow = encode_chat(
        _Tok(),
        [
            sys_m,
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "again"},
        ],
        None,
    )
    third = encode_chat(
        _Tok(),
        [
            sys_m,
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "still ok"},
            {"role": "user", "content": "once more"},
        ],
        None,
    )
    n = getattr(first, "stable_prefix_n", 0)
    assert n >= 32
    assert list(follow)[:n] == list(first)[:n]
    assert list(third)[:n] == list(first)[:n]


def test_C_pack_8k(monkeypatch):
    from slotbank.prompt import enforce_prompt_cap

    monkeypatch.setenv("SLOTBANK_ENVELOPE", "1")
    monkeypatch.delenv("SLOTBANK_MAX_PROMPT", raising=False)
    assert DEFAULT_ENVELOPE_MAX_PROMPT == 8192
    fat = "You are OMP. " + ("tools " * 4000)
    for dump in (_file_dump(6000, "hi"), _cwd_dump(900, "hi")):
        got = condense_harness_messages(
            [{"role": "system", "content": fat}, {"role": "user", "content": dump}],
            budget=8192,
        )
        blob = "\n".join(str(m.get("content") or "") for m in got)
        assert _approx_tok(blob) < 4000
        assert "hi" in blob
    packed = enforce_prompt_cap(list(range(20000)))
    assert len(packed) == 8192


def test_C_pyramid():
    ids = list(range(20000))
    got = keep_token_ids(ids, 8192)
    assert len(got) == 8192
    assert got[:2048] == ids[:2048]
    assert got[-1] == 19999
    assert got == sorted(got)
    # Mid is a proper subset of the original middle, order preserved.
    mid = got[2048:-4096] if len(got) > 6144 else got[2048:]
    assert mid == sorted(set(mid))
    assert all(2048 <= x < 20000 - 1 for x in mid) or not mid


def test_C_slim():
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": "does a thing " * 40,
                "parameters": {
                    "type": "object",
                    "properties": {f"p{j}": {"type": "string"} for j in range(20)},
                },
            },
        }
        for i in range(80)
    ]
    slim = slim_tools(tools)
    assert slim
    budget = sum(
        _approx_tok(t["function"]["name"] + " " + t["function"]["description"]) + 8
        for t in slim
    )
    assert budget <= TOOL_SLIM_BUDGET + 16
    assert all(t["function"]["parameters"]["properties"] == {} for t in slim)


def test_C_inject(tmp_path, monkeypatch):
    from slotbank.context_os import append, init_session
    from slotbank.prompt import with_context_os

    root = init_session(tmp_path / "job")
    append(root, "user", "secret dump " * 200)
    monkeypatch.setenv("SLOTBANK_CONTEXT_DIR", str(root))
    monkeypatch.setenv("SLOTBANK_ENVELOPE", "1")
    monkeypatch.delenv("SLOTBANK_CONTEXT_INJECT", raising=False)
    msgs = [{"role": "user", "content": "hi"}]
    assert with_context_os(msgs) == msgs
    monkeypatch.setenv("SLOTBANK_CONTEXT_INJECT", "1")
    injected = with_context_os(msgs)
    assert injected[0]["role"] == "system"
    assert "secret dump" in injected[0]["content"]


def test_C_context(tmp_path):
    from slotbank.context_os import append, compile_working_set, init_session

    root = init_session(tmp_path / "job")
    append(root, "user", "alpha")
    append(root, "assistant", "beta")
    append(root, "user", "gamma")
    log = (root / "log.jsonl").read_text()
    assert log.count("\n") == 3
    text = compile_working_set(root, budget=4096)
    assert "gamma" in text
    assert "paraphrase" not in text.lower()
    later = compile_working_set(root, budget=4096)
    assert later == text


def test_C_session():
    from slotbank.runtime import dflash_session_ok

    class Layer:
        def __init__(self, offset):
            self.offset = offset

    assert dflash_session_ok([Layer(3), Layer(3)], 3)
    assert not dflash_session_ok([Layer(3), Layer(5)], 3)
    assert dflash_session_ok([], 0)


def test_C_sampling():
    from slotbank.types import SamplingParams

    monkey = pytest.MonkeyPatch()
    monkey.setenv("SLOTBANK_ENVELOPE", "1")
    try:
        mapped = apply_qwen_sampling(SamplingParams(temperature=1.0, top_p=1.0, top_k=0))
        assert mapped.temperature == pytest.approx(0.7)
        assert mapped.top_p == pytest.approx(0.8)
        edge = apply_qwen_sampling(SamplingParams(temperature=0.99, top_p=1.0, top_k=0))
        assert edge.temperature == pytest.approx(0.7)
        kept = apply_qwen_sampling(SamplingParams(temperature=0.2, top_p=0.9, top_k=20))
        assert kept.temperature == pytest.approx(0.2)
        just_under = apply_qwen_sampling(
            SamplingParams(temperature=0.989, top_p=0.95, top_k=40)
        )
        assert just_under.temperature == pytest.approx(0.989)
    finally:
        monkey.undo()


# --- envelope suite (C-suite): many OMP-shaped prompts -----------------------

def _suite_cases() -> list[dict]:
    asks = [
        "hi",
        "what is in src/foo.py",
        "fix the failing test",
        "summarise the last commit",
        "explain the slot pack",
        "review the diff and say whether tests cover it",
        "写一段关于这个仓库的说明",
        "list files under src/slotbank",
    ]
    cases = []
    for i, ask in enumerate(asks):
        cases.append({
            "id": f"short-{i}",
            "class": "short",
            "user": ask,
            "system": "You are OMP. " + ("Be concise. " * 12),
        })
        cases.append({
            "id": f"file-{i}",
            "class": "file",
            "user": _file_dump(800 + i * 200, ask),
            "system": "You are OMP. " + ("tool schema " * 80),
        })
        cases.append({
            "id": f"cwd-{i}",
            "class": "cwd",
            "user": _cwd_dump(200 + i * 80, ask, child=f"repo-{i}"),
            "system": "You are OMP. " + ("catalog " * 60),
        })
    cases.append({
        "id": "cwd-26k",
        "class": "omp-26k",
        "user": _file_dump(6500, "hi"),
        "system": "You are OMP. " + ("tools " * 2000),
    })
    cases.append({
        "id": "tmp-39k",
        "class": "omp-39k",
        "user": _cwd_dump(2800, "hi", child="llama.cpp-dflash2"),
        "system": "You are OMP. " + ("tools " * 2500),
    })
    cases.append({
        "id": "empty-ask",
        "class": "edge",
        "user": "",
        "system": "You are OMP. " + ("Be concise. " * 12),
    })
    cases.append({
        "id": "follow-is-dump",
        "class": "edge",
        "user": _file_dump(1200, "first ask"),
        "system": "You are OMP. " + ("Be concise. " * 12),
        "follow": _file_dump(400, "second dump ask"),
    })
    cases.append({
        "id": "think-tags",
        "class": "edge",
        "user": "hi",
        "system": "You are OMP. " + ("Be concise. " * 12),
        "assistant": "<think>\nscratch\n</think>\n\nok",
    })
    return cases


def test_C_suite(monkeypatch):
    monkeypatch.setenv("SLOTBANK_ENVELOPE", "1")
    monkeypatch.delenv("SLOTBANK_CONTEXT_DIR", raising=False)
    monkeypatch.delenv("SLOTBANK_DIRECT", raising=False)
    monkeypatch.setenv("SLOTBANK_MAX_PROMPT", "0")
    tok = _Tok()
    rows = []
    for case in _suite_cases():
        sys_m = {"role": "system", "content": case["system"]}
        user_m = {"role": "user", "content": case["user"]}
        assistant = case.get("assistant", "ok")
        follow_user = case.get("follow", "continue")
        first_msgs = [sys_m, user_m]
        follow_msgs = [
            sys_m,
            user_m,
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": follow_user},
        ]
        third_msgs = follow_msgs + [
            {"role": "assistant", "content": "still ok"},
            {"role": "user", "content": "once more"},
        ]
        c1 = condense_harness_messages(first_msgs, budget=8192)
        c2 = condense_harness_messages(follow_msgs, budget=8192)
        c3 = condense_harness_messages(third_msgs, budget=8192)
        assert c1[0]["content"] == c2[0]["content"] == c3[0]["content"], case["id"]
        assert _approx_tok(c1[0]["content"]) <= ENVELOPE_SYS_TOKENS + 8, case["id"]
        assert c1[1]["content"] == c2[1]["content"] == c3[1]["content"], case["id"]
        e1 = encode_chat(tok, first_msgs, None)
        e2 = encode_chat(tok, follow_msgs, None)
        e3 = encode_chat(tok, third_msgs, None)
        n = getattr(e1, "stable_prefix_n", 0)
        assert n >= 16, case["id"]
        assert list(e2)[:n] == list(e1)[:n], case["id"]
        assert list(e3)[:n] == list(e1)[:n], case["id"]
        packed = keep_token_ids(list(e1) + list(range(9000)), 8192)
        assert len(packed) <= 8192
        rows.append({
            "id": case["id"],
            "class": case["class"],
            "raw_user_tok": _approx_tok(case["user"]),
            "raw_sys_tok": _approx_tok(case["system"]),
            "sys_tok": _approx_tok(c1[0]["content"]),
            "stable_prefix_n": n,
            "first_ids": len(e1),
            "follow_ids": len(e2),
            "third_ids": len(e3),
            "packed": len(packed),
        })
    assert len(rows) >= 27
    assert all(r["sys_tok"] <= ENVELOPE_SYS_TOKENS + 8 for r in rows)
    assert all(r["packed"] <= 8192 for r in rows)
    big = [r for r in rows if r["id"] in {"cwd-26k", "tmp-39k"}]
    assert len(big) == 2 and all(r["raw_user_tok"] > 4000 for r in big)
    classes = {r["class"] for r in rows}
    assert {"short", "file", "cwd", "omp-26k", "omp-39k", "edge"} <= classes
    _dump_json("suite_latest.json", {"n": len(rows), "rows": rows})


def test_C_cannot_retime_are_documented():
    leftover = [c["id"] for c in CLAIMS if c["kind"] == "cannot_retime"]
    assert set(leftover) == {"C-quality", "C-soak"}


def test_C_air_pinned_are_documented():
    pinned = [c["id"] for c in CLAIMS if c["kind"] == "air_pinned"]
    assert "C-clocks" in pinned
    assert "C-metal-toks" in pinned


def test_optional_dep_failures_are_named():
    assert len(OPTIONAL_DEP_FAILURES) == 7
    assert all("::" in n for n in OPTIONAL_DEP_FAILURES)
