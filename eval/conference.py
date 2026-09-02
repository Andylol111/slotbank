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
