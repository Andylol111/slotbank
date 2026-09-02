"""Conference protocol: well-formed suite + pilot stats. No Metal numbers invented."""

from __future__ import annotations

import ast
from pathlib import Path

from eval.conference import (
    alternating_summary,
    bandwidth_scale,
    c_sweep_summary,
    interior_c,
    load_pilot,
    metal_available,
    validate_prompts,
)


def test_prompt_suite_covers_five_classes():
    info = validate_prompts()
    assert info["n"] >= 40
    assert info["n"] == 48
    for n in info["by_class"].values():
        assert n >= 6


def test_pilot_is_labeled_pilot():
    p = load_pilot()
    assert p["kind"] == "pilot"
    assert p["n_machines"] == 1
    assert p["dense_27b"]["n_prompts"] == 1
    assert "48-prompt" in " ".join(p["gaps"])


def test_reference_execution_is_quantized_not_bf16():
    ref = load_pilot()["reference_execution"]
    assert "quantized" in ref["definition"].lower()
    assert "bf16" in ref["not_equivalent_to"]


def test_c_sweep_interior_is_32():
    rows = c_sweep_summary()
    assert interior_c(rows) == 32
    by_c = {r["C"]: r for r in rows}
    assert by_c[32]["cold"]["mean"] > by_c[64]["cold"]["mean"]
    assert by_c[32]["warm"]["mean"] > by_c[64]["warm"]["mean"]


def test_alternating_larger_pack_reads_more_disk():
    alt = alternating_summary()
    assert alt["c32_toks"]["mean"] > alt["c64_toks"]["mean"]
    assert alt["disk_ratio"] > 1.5
    assert 25 <= alt["miss_drop_pct"] <= 40


def test_bandwidth_scale_matches_air_pins():
    greedy = bandwidth_scale(19.8, 400)
    mtp = bandwidth_scale(32.6, 400)
    assert abs(greedy - 5.94) < 0.02
    assert abs(mtp - 9.78) < 0.02
    clocks = load_pilot()["dense_27b"]["clocks"]
    assert abs(greedy - clocks["greedy_toks"]) < 0.3
    assert abs(mtp - clocks["mtp_k3_code"]) < 0.3


def test_conference_eval_does_not_import_monte_carlo():
    root = Path(__file__).resolve().parents[1]
    for rel in ("eval/conference.py", "scripts/conference_eval.py"):
        tree = ast.parse((root / rel).read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names.extend(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
        assert "stress_mac_speedup" not in names


def test_this_host_has_no_metal_27b():
    assert metal_available() is False
