#!/usr/bin/env python3
"""Emit the three-category 27B decode figure from eval/pilot.json.

Does not invent Metal rows. Projections are labeled unverified.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.conference import emit_decode_tier_tikz, load_pilot  # noqa: E402

STANDALONE = r"""\documentclass[border=8pt]{standalone}
\usepackage{times}
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,patterns,calc}
\begin{document}
\input{qwen27b_three_category_body.tex}
\end{document}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "eval" / "figures",
        help="Directory for the TikZ body (and optional PNG/PDF copy)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile a standalone PDF and rasterize a PNG",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="Optional PNG destination (implies --compile)",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    body = emit_decode_tier_tikz(load_pilot())
    body_path = out_dir / "qwen27b_three_category_body.tex"
    body_path.write_text(body)
    print(f"wrote {body_path}")
    if not args.compile and args.png is None:
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "qwen27b_three_category_body.tex").write_text(body)
        (tmp_path / "figure.tex").write_text(STANDALONE)
        tex = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "figure.tex"],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
        )
        pdf = tmp_path / "figure.pdf"
        if tex.returncode != 0 or not pdf.exists():
            sys.stderr.write(tex.stdout)
            sys.stderr.write(tex.stderr)
            return tex.returncode or 1
        dest_pdf = out_dir / "qwen27b_three_category.pdf"
        shutil.copy2(pdf, dest_pdf)
        print(f"wrote {dest_pdf}")
        png_dest = args.png or (out_dir / "qwen27b_three_category.png")
        png_dest.parent.mkdir(parents=True, exist_ok=True)
        raster = subprocess.run(
            [
                "pdftocairo",
                "-png",
                "-r",
                "220",
                "-singlefile",
                str(pdf),
                str(png_dest.with_suffix("")),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if raster.returncode != 0:
            sys.stderr.write(raster.stderr)
            return raster.returncode
        print(f"wrote {png_dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
