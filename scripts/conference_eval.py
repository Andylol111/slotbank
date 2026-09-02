#!/usr/bin/env python3
"""Print the conference evaluation protocol and pilot tables.

Does not invent Metal 27B tok/s. This host has no weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.conference import (  # noqa: E402
    alternating_summary,
    bandwidth_scale,
    c_sweep_summary,
    interior_c,
    load_pilot,
    metal_available,
    validate_prompts,
)


def main() -> int:
    pilot = load_pilot()
    prompts = validate_prompts()
    sweep = c_sweep_summary(pilot)
    alt = alternating_summary(pilot)
    d27 = pilot["dense_27b"]
    omlx = d27["omlx_m3_max"]
    print("conference evaluation protocol")
    print(f"prompts: {prompts['n']} {prompts['by_class']}")
    print(f"metal_available: {metal_available()}")
    print(f"pilot machines: {pilot['n_machines']} ({pilot['machine']['id']})")
    print(f"gaps: {', '.join(pilot['gaps'])}")
    print()
    print("C sweep (cold e2e mean / range, n<=2):")
    for row in sweep:
        c = row["cold"]
        print(
            f"  C={row['C']:>3}  active={row['active_gib']:.2f} GiB  "
            f"cold={c['mean']:.3f}  range={c['range']:.2f}  n={int(c['n'])}"
        )
    print(f"interior C* by cold mean: {interior_c(sweep)}")
    print()
    print("alternating warm C=32 vs C=64 (n=3):")
    print(
        f"  C=32 toks {alt['c32_toks']['mean']:.3f} "
        f"(range {alt['c32_toks']['range']:.2f})"
    )
    print(
        f"  C=64 toks {alt['c64_toks']['mean']:.3f} "
        f"(range {alt['c64_toks']['range']:.2f})"
    )
    print(
        f"  miss drop {alt['miss_drop_pct']:.1f}%  "
        f"disk ratio {alt['disk_ratio']:.2f}x"
    )
    print()
    print("bandwidth scale 400 -> 120 GB/s:")
    print(
        f"  {omlx['metal_decode']} -> "
        f"{bandwidth_scale(omlx['metal_decode'], omlx['bw_gbs']):.2f}  "
        f"measured greedy {d27['clocks']['greedy_toks']}"
    )
    print(
        f"  {omlx['mtp_decode']} -> "
        f"{bandwidth_scale(omlx['mtp_decode'], omlx['bw_gbs']):.2f}  "
        f"measured MTP-code {d27['clocks']['mtp_k3_code']}"
    )
    print()
    print("27B clocks are a 1-prompt, 64-token pilot. Not the 48-prompt suite.")
    out = {
        "prompts": prompts,
        "interior_C": interior_c(sweep),
        "alternating": alt,
        "metal_available": metal_available(),
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
