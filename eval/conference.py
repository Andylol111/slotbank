"""Conference-eval helpers. Pilot stats only. No Metal fabrication."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REQUIRED_CLASSES = ("code", "prose", "reasoning", "repetitive", "structured")


def load_pilot(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or ROOT / "pilot.json").read_text())


def load_prompts(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or ROOT / "prompts.json").read_text())


def _mean_range(xs: list[float]) -> dict[str, float]:
    if not xs:
        raise ValueError("empty reps")
    return {
        "n": float(len(xs)),
        "mean": statistics.fmean(xs),
        "min": min(xs),
        "max": max(xs),
        "range": max(xs) - min(xs),
        "stdev": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
    }


def c_sweep_summary(pilot: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    p = pilot or load_pilot()
    cold = {r["C"]: r for r in p["moe_35b"]["c_sweep_cold_e2e"]["rows"]}
    warm = {r["C"]: r for r in p["moe_35b"]["c_sweep_warm_decode"]["rows"]}
    out = []
    for c, row in cold.items():
        item = {"C": c, "active_gib": row["active_gib"], "cold": _mean_range(row["reps"])}
        if c in warm:
            item["warm"] = _mean_range(warm[c]["reps"])
        out.append(item)
    return out


def alternating_summary(pilot: dict[str, Any] | None = None) -> dict[str, Any]:
    a = (pilot or load_pilot())["moe_35b"]["alternating_warm_c32_c64"]
    return {
        "c32_toks": _mean_range(a["c32_toks"]),
        "c64_toks": _mean_range(a["c64_toks"]),
        "c32_disk": _mean_range(a["c32_disk_mib_per_tok"]),
        "c64_disk": _mean_range(a["c64_disk_mib_per_tok"]),
        "miss_drop_pct": 100.0 * (1.0 - a["c64_miss_per_call"] / a["c32_miss_per_call"]),
        "disk_ratio": statistics.fmean(a["c64_disk_mib_per_tok"])
        / statistics.fmean(a["c32_disk_mib_per_tok"]),
    }


def bandwidth_scale(toks: float, src_gbs: float, dst_gbs: float = 120.0) -> float:
    return toks * (dst_gbs / src_gbs)


def interior_c(summary: list[dict[str, Any]] | None = None) -> int:
    rows = summary or c_sweep_summary()
    return max(rows, key=lambda r: r["cold"]["mean"])["C"]


def validate_prompts(blob: dict[str, Any] | None = None) -> dict[str, Any]:
    data = blob or load_prompts()
    prompts = data["prompts"]
    ids = [p["id"] for p in prompts]
    classes = {p["class"] for p in prompts}
    missing = [c for c in REQUIRED_CLASSES if c not in classes]
    if missing:
        raise ValueError(f"prompt classes missing: {missing}")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prompt id")
    if len(prompts) < 40:
        raise ValueError(f"need >= 40 prompts, have {len(prompts)}")
    for p in prompts:
        if len(p["text"].strip()) < 8:
            raise ValueError(f"short prompt {p['id']}")
    return {
        "n": len(prompts),
        "by_class": {c: sum(1 for p in prompts if p["class"] == c) for c in REQUIRED_CLASSES},
    }


def metal_available() -> bool:
    return False


def refuse_fabricated_metal() -> None:
    if not metal_available():
        return
    raise RuntimeError("Metal path is not wired on this host")


AIR_BW_GBS = 120.0
# Project greedy 5.71 and MTP-code 9.95 only. The 13.47 count pin is a
# measured ceiling on this Air, not a second cartoon to scale up.
PROJECT_CLOCKS = ("greedy_toks", "mtp_k3_code")
HEADROOM_SENTINELS = (8.9, 32.9, 112.9)


def _round1(value: float) -> float:
    return round(value + 1e-12, 1)


def project_air(
    clocks: dict[str, float],
    bw_gbs: float,
    src_gbs: float = AIR_BW_GBS,
) -> dict[str, float]:
    """Bandwidth cartoon of this Air. Not a measurement on the target host."""
    if src_gbs <= 0:
        raise ValueError("src_gbs must be positive")
    scale = bw_gbs / src_gbs
    out = {"scale": scale, "verified": False}
    for key in PROJECT_CLOCKS:
        out[key] = clocks[key] * scale
        out[f"{key}_display"] = _round1(out[key])
    return out


def decode_tier_rows(pilot: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """27B decode rows in three disjoint kinds: measured, projected, published."""
    p = pilot or load_pilot()
    d27 = p["dense_27b"]
    clocks = d27["clocks"]
    rows: list[dict[str, Any]] = []

    measured = (
        ("greedy", "Air greedy", clocks["greedy_toks"], None, None),
        ("dflash_code", "Air DFlash $K{=}8$ code", clocks["dflash_k8_code"], None, None),
        ("mtp_code", "Air MTP $K{=}3$ code", clocks["mtp_k3_code"], None, None),
        ("dflash_count", "Air DFlash $K{=}8$ count", clocks["dflash_k8_count"], None, None),
        ("mtp_count", "Air MTP $K{=}3$ count", clocks["mtp_k3_count"], None, None),
    )
    for key, label, toks, lo, hi in measured:
        rows.append(
            {
                "kind": "measured",
                "id": f"air-{key}",
                "label": label,
                "toks": float(toks),
                "toks_lo": lo,
                "toks_hi": hi,
                "verified": True,
                "fits_24g": True,
                "machine": "M4 Air 24GB",
                "note": "one prefix, 64 tok, cool, 2026-08-31",
            }
        )

    for host in d27["projection_hosts"]:
        proj = project_air(clocks, host["bw_gbs"])
        for key, suffix in (("greedy_toks", "greedy"), ("mtp_k3_code", "MTP-code")):
            rows.append(
                {
                    "kind": "projected",
                    "id": f"{host['id']}-{key}",
                    "label": f"{host['label']} {int(host['bw_gbs'])} {suffix}",
                    "toks": proj[key],
                    "toks_display": proj[f"{key}_display"],
                    "toks_lo": None,
                    "toks_hi": None,
                    "verified": False,
                    "fits_24g": None,
                    "machine": host["label"],
                    "bw_gbs": host["bw_gbs"],
                    "note": host["note"],
                }
            )

    for pub in d27["published_other_cases"]:
        rows.append(
            {
                "kind": "published",
                "id": pub["id"],
                "label": pub["label"],
                "toks": pub.get("toks"),
                "toks_lo": pub.get("toks_lo"),
                "toks_hi": pub.get("toks_hi"),
                "verified": False,
                "fits_24g": pub.get("fits_24g"),
                "machine": pub["machine"],
                "model": pub["model"],
                "cite": pub["cite"],
                "loader": pub["loader"],
                "note": pub.get("note", ""),
            }
        )
    return rows


def decode_tier_summary(pilot: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = decode_tier_rows(pilot)
    by_kind = {kind: [r for r in rows if r["kind"] == kind] for kind in ("measured", "projected", "published")}
    return {
        "n_measured": len(by_kind["measured"]),
        "n_projected": len(by_kind["projected"]),
        "n_published": len(by_kind["published"]),
        "kinds": ("measured", "projected", "published"),
        "omits": list((pilot or load_pilot())["dense_27b"]["figure_omits"]),
        "rows": rows,
    }


def emit_decode_tier_tikz(pilot: dict[str, Any] | None = None) -> str:
    """TikZ body. Three visual kinds; leftover RAM is not a bar."""
    p = pilot or load_pilot()
    rows = decode_tier_rows(p)
    measured = [r for r in rows if r["kind"] == "measured"]
    projected = [r for r in rows if r["kind"] == "projected"]
    published = [r for r in rows if r["kind"] == "published"]

    x_max = 80.0
    x_unit = 0.112  # cm per tok/s
    row_h = 0.46
    bar_h = 0.15
    label_x = -0.25

    def xv(toks: float) -> float:
        return toks * x_unit

    def bar_value(row: dict[str, Any]) -> float | None:
        if row.get("toks") is not None:
            return float(row["toks"])
        return None

    def display_toks(row: dict[str, Any]) -> str:
        if row.get("toks_lo") is not None and row.get("toks_hi") is not None:
            lo, hi = row["toks_lo"], row["toks_hi"]
            if float(lo).is_integer() and float(hi).is_integer():
                return f"{int(lo)}--{int(hi)}"
            return f"{lo:g}--{hi:g}"
        value = row.get("toks_display", row.get("toks"))
        if value is None:
            return ""
        value = float(value)
        if row["kind"] == "measured":
            return f"{value:.2f}"
        if row["kind"] == "projected":
            return f"{value:.1f}"
        return f"{value:.1f}"

    # y grows up from the axis. Stack published, projected, measured.
    y = 0.85
    placed: list[tuple[float, dict[str, Any]]] = []
    band_spans: list[tuple[str, float, float, str]] = []

    def place_band(title: str, fill: str, band_rows: list[dict[str, Any]]) -> None:
        nonlocal y
        y0 = y
        for row in reversed(band_rows):
            placed.append((y, row))
            y += row_h
        y_top = y
        y += 0.42
        band_spans.append((title, y0 - 0.24, y_top + 0.08, fill))
        y += 0.12

    place_band(
        r"{\textbf{3. Published other cases}} --- other loaders and machines; not this codebase",
        "pubband",
        published,
    )
    place_band(
        r"{\textbf{2. Projected, not verified}} --- Air greedy 5.71 and MTP-code 9.95 $\times B/120$",
        "projband",
        projected,
    )
    place_band(
        r"{\textbf{1. Measured on this Air}} --- $n{=}1$ prefix, 64 tok, cool chassis, 2026-08-31",
        "measband",
        measured,
    )

    y_top = round(y + 0.15, 3)
    x_right = round(xv(x_max), 3)
    lines = [
        r"% Auto-generated by eval.conference.emit_decode_tier_tikz. Do not hand-edit.",
        r"\begin{tikzpicture}[font=\scriptsize, >=Stealth]",
        r"\definecolor{meas}{HTML}{1F4E79}",
        r"\definecolor{measband}{HTML}{E4EEF6}",
        r"\definecolor{proj}{HTML}{5E5E5E}",
        r"\definecolor{projband}{HTML}{F1F1F1}",
        r"\definecolor{pub}{HTML}{B45309}",
        r"\definecolor{pubane}{HTML}{9B1D4A}",
        r"\definecolor{pubband}{HTML}{F8EDE3}",
        f"\\fill[white] ({-6.55},{-0.85}) rectangle ({x_right + 2.35:.3f},{y_top + 0.55:.3f});",
    ]

    for title, y0, y1, fill in band_spans:
        lines.append(
            f"\\fill[{fill}] ({-6.45},{y0:.3f}) rectangle ({x_right + 2.25:.3f},{y1:.3f});"
        )
        lines.append(
            f"\\node[anchor=west, font=\\scriptsize] at ({-6.35},{y1 - 0.20:.3f}) {{{title}}};"
        )

    # Light vertical guides.
    for tick in (10, 20, 30, 40, 50, 60, 70, 80):
        lines.append(
            f"\\draw[black!12] ({xv(tick):.3f},0.55) -- ({xv(tick):.3f},{y_top - 0.35:.3f});"
        )

    for y_row, row in placed:
        kind = row["kind"]
        lo = row.get("toks_lo")
        hi = row.get("toks_hi")
        toks = bar_value(row)
        ane = row.get("loader") == "omlx-ane"
        color = "pubane" if ane else ("meas" if kind == "measured" else "proj" if kind == "projected" else "pub")
        y0b, y1b = y_row - bar_h, y_row + bar_h

        if kind == "measured" and toks is not None:
            lines.append(
                f"\\fill[{color}] (0,{y0b:.3f}) rectangle ({xv(toks):.3f},{y1b:.3f});"
            )
        elif kind == "projected" and toks is not None:
            lines.append(
                f"\\fill[pattern=north east lines, pattern color={color}] "
                f"(0,{y0b:.3f}) rectangle ({xv(toks):.3f},{y1b:.3f});"
            )
            lines.append(
                f"\\draw[{color}, thin] (0,{y0b:.3f}) rectangle ({xv(toks):.3f},{y1b:.3f});"
            )
        elif lo is not None and hi is not None:
            lines.append(
                f"\\draw[{color}, line width=1.35pt] ({xv(float(lo)):.3f},{y_row:.3f}) "
                f"-- ({xv(float(hi)):.3f},{y_row:.3f});"
            )
            lines.append(f"\\fill[{color}] ({xv(float(lo)):.3f},{y_row:.3f}) circle (1.7pt);")
            lines.append(f"\\fill[{color}] ({xv(float(hi)):.3f},{y_row:.3f}) circle (1.7pt);")
        elif toks is not None:
            lines.append(
                f"\\draw[{color}, line width=0.9pt] (0,{y_row:.3f}) -- ({xv(toks):.3f},{y_row:.3f});"
            )
            lines.append(f"\\fill[{color}] ({xv(toks):.3f},{y_row:.3f}) circle (2.05pt);")

        label = row["label"]
        lines.append(
            f"\\node[anchor=east, align=right] at ({label_x},{y_row:.3f}) {{{label}}};"
        )
        value = display_toks(row)
        x_text = xv(float(hi if hi is not None else toks or 0.0)) + 0.12
        extra = ""
        if row["id"].startswith("weschera"):
            extra = r" {\tiny no 24\,GB}"
        elif row["id"].startswith("omlx-2874"):
            extra = r" {\tiny 16K host}"
        elif row["id"] == "m3-max-128g-mtp_k3_code":
            extra = r" {\tiny $\approx$ oMLX 32.6}"
        elif row["id"] == "m3-max-128g-greedy_toks":
            extra = r" {\tiny $\approx$ oMLX 19.8}"
        elif row["id"] == "nathanmaine-mlxvlm-mtp":
            extra = r" {\tiny 15.5\,GB peak}"
        elif row["id"] == "mlxlm-990-mtp":
            extra = r" {\tiny from 15.3}"
        lines.append(
            f"\\node[anchor=west] at ({x_text:.3f},{y_row:.3f}) {{{value}{extra}}};"
        )

    # Axis
    lines.append(f"\\draw[->] (0,0.45) -- ({x_right + 0.25:.3f},0.45);")
    for tick in (0, 10, 20, 30, 40, 50, 60, 70, 80):
        lines.append(
            f"\\draw ({xv(tick):.3f},0.45) -- ++(0,-0.08) "
            f"node[below, font=\\scriptsize] {{{tick}}};"
        )
    lines.append(
        rf"\node[anchor=north] at ({x_right / 2:.3f},-0.22) "
        r"{decode tok/s (not prefill; leftover DRAM is not on this axis)};"
    )

    # Legend
    ly = y_top + 0.22
    lines.append(rf"\fill[meas] ({-6.35},{ly - 0.08:.3f}) rectangle ({-6.05},{ly + 0.08:.3f});")
    lines.append(rf"\node[anchor=west] at ({-5.95},{ly:.3f}) {{measured (this Air)}};")
    lines.append(
        rf"\fill[pattern=north east lines, pattern color=proj] "
        rf"({-3.15},{ly - 0.08:.3f}) rectangle ({-2.85},{ly + 0.08:.3f});"
    )
    lines.append(
        rf"\draw[proj] ({-3.15},{ly - 0.08:.3f}) rectangle ({-2.85},{ly + 0.08:.3f});"
    )
    lines.append(rf"\node[anchor=west] at ({-2.75},{ly:.3f}) {{projected (not timed)}};")
    lines.append(rf"\draw[pub, line width=0.9pt] ({0.55},{ly:.3f}) -- ({0.95},{ly:.3f});")
    lines.append(rf"\fill[pub] ({0.95},{ly:.3f}) circle (2pt);")
    lines.append(rf"\node[anchor=west] at ({1.08},{ly:.3f}) {{published Metal / MLX / Ollama}};")
    lines.append(rf"\draw[pubane, line width=1.2pt] ({4.85},{ly:.3f}) -- ({5.25},{ly:.3f});")
    lines.append(rf"\fill[pubane] ({5.25},{ly:.3f}) circle (2pt);")
    lines.append(rf"\node[anchor=west] at ({5.38},{ly:.3f}) {{published ANE (no 24\,GB)}};")

    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines) + "\n"
