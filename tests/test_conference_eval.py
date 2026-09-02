"""Conference protocol: well-formed suite + pilot stats. No Metal numbers invented."""

from __future__ import annotations

import ast
from pathlib import Path

from eval.conference import (
    HEADROOM_SENTINELS,
    alternating_summary,
    bandwidth_scale,
    c_sweep_summary,
    decode_tier_rows,
    decode_tier_summary,
    emit_decode_tier_tikz,
    interior_c,
    load_pilot,
    metal_available,
    project_air,
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


def test_projections_are_air_times_bandwidth_and_unverified():
    clocks = load_pilot()["dense_27b"]["clocks"]
    expected = {
        273: (13.0, 22.6),
        400: (19.0, 33.2),
        546: (26.0, 45.3),
    }
    for bw, (greedy, mtp) in expected.items():
        proj = project_air(clocks, bw)
        assert proj["verified"] is False
        assert abs(proj["greedy_toks"] - clocks["greedy_toks"] * bw / 120.0) < 1e-12
        assert abs(proj["mtp_k3_code"] - clocks["mtp_k3_code"] * bw / 120.0) < 1e-12
        assert proj["greedy_toks_display"] == greedy
        assert proj["mtp_k3_code_display"] == mtp
        assert "mtp_k3_count" not in proj


def test_tier_rows_keep_three_kinds_disjoint():
    rows = decode_tier_rows()
    kinds = {r["kind"] for r in rows}
    assert kinds == {"measured", "projected", "published"}
    assert all(r["verified"] is True for r in rows if r["kind"] == "measured")
    assert all(r["verified"] is False for r in rows if r["kind"] != "measured")
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))
    summary = decode_tier_summary()
    assert summary["n_measured"] == 5
    assert summary["n_projected"] == 6
    assert summary["n_published"] == 8


def test_published_pins_match_cited_cases():
    by_id = {r["id"]: r for r in decode_tier_rows() if r["kind"] == "published"}
    assert by_id["omlx-2874-metal"]["toks"] == 19.8
    assert by_id["omlx-2874-mtp"]["toks"] == 32.6
    assert by_id["terminalbytes-ollama-38"]["toks"] == 14.0
    assert by_id["terminalbytes-ollama-36"]["toks"] == 28.6
    assert by_id["mlxlm-990-mtp"]["toks"] == 24.0
    assert by_id["nathanmaine-mlxvlm-mtp"]["toks"] == 47.0
    assert by_id["weschera-ane-mtp"]["toks_lo"] == 53.0
    assert by_id["weschera-ane-mtp"]["toks_hi"] == 72.0
    assert by_id["weschera-ane-mtp"]["fits_24g"] is False
    assert by_id["orcarouter-m4-mini-greedy"]["toks_lo"] == 5.0
    assert by_id["orcarouter-m4-mini-greedy"]["toks_hi"] == 6.0


def test_figure_omits_headroom_and_prefill_headlines():
    omits = " ".join(decode_tier_summary()["omits"]).lower()
    assert "15.1" in omits
    assert "prefill" in omits
    assert "35b" in omits
    tikz = emit_decode_tier_tikz()
    assert "Measured on this Air" in tikz
    assert "Projected, not verified" in tikz
    assert "Published other cases" in tikz
    assert "5.71" in tikz
    assert "13.47" in tikz
    assert "9.95" in tikz
    assert "19.8" in tikz
    assert "32.6" in tikz
    assert "north east lines" in tikz
    assert "leftover DRAM is not on this axis" in tikz
    assert "headroom" not in tikz.lower()
    assert "9.10" in tikz
    assert "13.0" in tikz
    committed = Path(__file__).resolve().parents[1] / "eval/figures/qwen27b_three_category_body.tex"
    assert committed.read_text() == tikz
    paper_copy = Path(__file__).resolve().parents[1] / "paper/figures/qwen27b_three_category_body.tex"
    assert paper_copy.is_file(), "paper/figures copy required for standalone Overleaf compiles"
    assert paper_copy.read_text() == tikz
    main_tex = Path(__file__).resolve().parents[1] / "paper/main.tex"
    main = main_tex.read_text()
    assert "\\begin{filecontents*}[overwrite]{refs.bib}" in main
    assert "\\begin{tikzpicture}" in main
    assert "\\IfFileExists" not in main
    assert "../eval/figures/" not in main
    toks_values = []
    for row in decode_tier_rows():
        if row.get("toks") is not None:
            toks_values.append(round(float(row["toks"]), 1))
        if row.get("toks_display") is not None:
            toks_values.append(float(row["toks_display"]))
    for sentinel in HEADROOM_SENTINELS:
        assert sentinel not in toks_values
