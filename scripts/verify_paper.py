#!/usr/bin/env python3
"""Run the paper verification campaign and write a dated, traceable log.

Air tok/s are pins (M4_AIR_24G). This host has no Metal 27B weights.

Writes the same campaign to:
  verification/          (committed methodology trail)
  paper/verification/    (next to the uncommitted manuscript)
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_OUT = ROOT / "verification"
PAPER_OUT = ROOT / "paper" / "verification"
SRC = ROOT / "src"


def _run(cmd: list[str], extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(SRC), **(extra_env or {})}
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)


def _write_both(name: str, text: str) -> None:
    for out in (REPO_OUT, PAPER_OUT):
        out.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(text)


def _copy_both(src: Path, name: str) -> None:
    if not src.exists():
        return
    for out in (REPO_OUT, PAPER_OUT):
        out.mkdir(parents=True, exist_ok=True)
        dest = out / name
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)


def _parse_counts(text: str) -> dict[str, int]:
    # e.g. "218 passed, 23 skipped, 7 deselected, 1 warning in 1.37s"
    keys = ("passed", "failed", "skipped", "deselected", "error", "warning")
    found = {k: 0 for k in keys}
    tail = text.strip().splitlines()[-3:]
    blob = " ".join(tail)
    for k in keys:
        m = re.search(rf"(\d+)\s+{k}", blob)
        if m:
            found[k] = int(m.group(1))
    return found


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def main() -> int:
    REPO_OUT.mkdir(parents=True, exist_ok=True)
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite_dir = REPO_OUT / f"suite_{stamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(SRC))
    from paper_claims import (
        CLAIMS,
        OPTIONAL_DEP_FAILURES,
        OPTIONAL_DEP_FILTER,
        PROTOCOL,
        RESEARCH_QUESTIONS,
    )
    from slotbank.tps import M4_AIR_24G, review_mac_speedup

    filtered_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "--tb=line",
        "-o",
        "cache_dir=/tmp/pytest-slotbank",
        "-k",
        OPTIONAL_DEP_FILTER,
    ]
    full_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "--tb=line",
        "-o",
        "cache_dir=/tmp/pytest-slotbank",
    ]
    paper_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_paper_verification.py",
        "tests/test_fence.py",
        "-q",
        "--tb=short",
        "-o",
        "cache_dir=/tmp/pytest-slotbank",
    ]

    filtered = _run(filtered_cmd)
    full = _run(full_cmd)
    paper = _run(paper_cmd, extra_env={"SLOTBANK_VERIF_OUT": str(suite_dir)})

    _write_both(f"pytest_filtered_{stamp}.txt", (filtered.stdout or "") + (filtered.stderr or ""))
    _write_both("pytest_filtered_latest.txt", (filtered.stdout or "") + (filtered.stderr or ""))
    _write_both(f"pytest_full_{stamp}.txt", (full.stdout or "") + (full.stderr or ""))
    _write_both("pytest_full_latest.txt", (full.stdout or "") + (full.stderr or ""))
    _write_both(f"pytest_paper_{stamp}.txt", (paper.stdout or "") + (paper.stderr or ""))
    _write_both("pytest_paper_latest.txt", (paper.stdout or "") + (paper.stderr or ""))

    suite_src = suite_dir / "suite_latest.json"
    catalog_src = suite_dir / "catalog_snapshot.json"
    review_src = suite_dir / "mac_review.json"
    _copy_both(suite_src, "suite_latest.json")
    _copy_both(catalog_src, "catalog_snapshot.json")
    _copy_both(review_src, "mac_review.json")

    suite = json.loads(suite_src.read_text()) if suite_src.exists() else {"n": 0, "rows": []}
    filtered_counts = _parse_counts((filtered.stdout or "") + (filtered.stderr or ""))
    full_counts = _parse_counts((full.stdout or "") + (full.stderr or ""))
    paper_counts = _parse_counts((paper.stdout or "") + (paper.stderr or ""))

    claim_rows = []
    for c in CLAIMS:
        if c["kind"] == "cannot_retime":
            status = "documented_gap"
        elif c["kind"] == "air_pinned":
            status = "pin_checked" if paper.returncode == 0 else "pin_failed"
        else:
            status = "verified_here" if paper.returncode == 0 else "failed"
        claim_rows.append({**c, "campaign_status": status})

    summary = {
        "campaign_id": stamp,
        "utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "research_questions": list(RESEARCH_QUESTIONS),
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "metal_27b": False,
            "note": "Linux cloud workspace; Air clocks are pins, not retimes.",
        },
        "pytest": {
            "filtered": {
                "cmd": filtered_cmd,
                "returncode": filtered.returncode,
                "counts": filtered_counts,
                "tail": ((filtered.stdout or "") + (filtered.stderr or "")).strip().splitlines()[-8:],
            },
            "full": {
                "cmd": full_cmd,
                "returncode": full.returncode,
                "counts": full_counts,
                "expected_optional_failures": list(OPTIONAL_DEP_FAILURES),
                "tail": ((full.stdout or "") + (full.stderr or "")).strip().splitlines()[-12:],
            },
            "paper": {
                "cmd": paper_cmd,
                "returncode": paper.returncode,
                "counts": paper_counts,
                "tail": ((paper.stdout or "") + (paper.stderr or "")).strip().splitlines()[-8:],
            },
        },
        "air_pins": dict(M4_AIR_24G),
        "mac_review": review_mac_speedup(),
        "claims": claim_rows,
        "suite": {
            "n": suite.get("n", 0),
            "classes": sorted({r.get("class", "") for r in suite.get("rows", [])}),
            "rows": suite.get("rows", []),
        },
        "ok": paper.returncode == 0 and filtered.returncode == 0,
    }
    blob = json.dumps(summary, indent=2, default=str) + "\n"
    _write_both(f"campaign_{stamp}.json", blob)
    _write_both("campaign_latest.json", blob)

    lines = [
        f"# Verification campaign {stamp}",
        "",
        "Traceability log for the slotbank systems manuscript.",
        "Air tok/s are pinned constants from `M4_AIR_24G`, measured 2026-08-31.",
        "This host has no Metal 27B weights; live decode is not retimed here.",
        "",
        "## Host",
        "",
        f"- OS: {platform.system()} {platform.machine()}",
        f"- Python: {sys.version.split()[0]}",
        "- Metal 27B: absent",
        f"- UTC: {summary['utc']}",
        "",
        "## Method",
        "",
        f"- Independent variables: {PROTOCOL['independent']}",
        f"- Dependent variables: {PROTOCOL['dependent']}",
        f"- Not measured: {PROTOCOL['not_measured']}",
        f"- Oracle: {PROTOCOL['oracle']}",
        "",
        "Three pytest invocations:",
        "",
        "1. **Paper + fence** — every claim id; writes suite JSON.",
        "2. **Filtered tree** — project CI filter (optional deps dropped).",
        "3. **Full tree** — records the seven optional-dep failures so they are not silent.",
        "",
        "## Research questions",
        "",
    ]
    for q in RESEARCH_QUESTIONS:
        lines.append(f"- **{q['id']} ({q['title']}).** {q['question']}")
    lines += [
        "",
        "## pytest",
        "",
        f"| Run | passed | failed | skipped | deselected | returncode |",
        f"|---|---:|---:|---:|---:|---:|",
        f"| paper+fence | {paper_counts['passed']} | {paper_counts['failed']} | {paper_counts['skipped']} | {paper_counts['deselected']} | {paper.returncode} |",
        f"| filtered CI | {filtered_counts['passed']} | {filtered_counts['failed']} | {filtered_counts['skipped']} | {filtered_counts['deselected']} | {filtered.returncode} |",
        f"| full tree | {full_counts['passed']} | {full_counts['failed']} | {full_counts['skipped']} | {full_counts['deselected']} | {full.returncode} |",
        "",
        "Full-tree failures are the named optional-dep tests:",
        "",
    ]
    for name in OPTIONAL_DEP_FAILURES:
        lines.append(f"- `{name}`")
    lines += [
        "",
        "```",
        *summary["pytest"]["filtered"]["tail"],
        "```",
        "",
        "## Claim matrix",
        "",
        "| Id | RQ | Kind | Campaign | Test | Statement |",
        "|---|---|---|---|---|---|",
    ]
    for c in claim_rows:
        lines.append(
            f"| `{c['id']}` | {c['rq']} | {c['kind']} | {c['campaign_status']} | "
            f"`{c['test']}` | {_md_escape(c['statement'])} |"
        )
    lines += [
        "",
        "## How to read kind / campaign status",
        "",
        "- `verified_here` — reproduced on this host by the paper tests.",
        "- `air_pinned` / `pin_checked` — measured 2026-08-31 on the author's M4 Air; tests fail if the pin drifts.",
        "- `cannot_retime` / `documented_gap` — needs the 27B process or a soak; listed so the paper does not pretend otherwise.",
        "",
        "## Envelope suite (structural, not task quality)",
        "",
        f"n = {suite.get('n', 0)}. Classes: {', '.join(summary['suite']['classes']) or '(none — suite dump missing)'}.",
        "Every case asserts system ≤ 256 tokens, packed ids ≤ 8192, and a history-stable prefix across first / follow-up / third turn.",
        "This is not SWE-bench. `C-quality` remains a documented gap.",
        "",
        "| Id | class | raw user tok | sys tok | prefix n | first ids | packed |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in suite.get("rows", []):
        lines.append(
            f"| `{r['id']}` | {r.get('class','')} | {r.get('raw_user_tok',0)} | "
            f"{r.get('sys_tok',0)} | {r.get('stable_prefix_n',0)} | "
            f"{r.get('first_ids',0)} | {r.get('packed',0)} |"
        )
    lines += [
        "",
        "## Reproduction",
        "",
        "```",
        "PYTHONPATH=src python3 scripts/verify_paper.py",
        "```",
        "",
        "Outputs: `verification/campaign_latest.md`, `verification/campaign_latest.json`,",
        "`verification/suite_latest.json`, and timestamped copies of each.",
        "",
    ]
    md = "\n".join(lines) + "\n"
    _write_both(f"campaign_{stamp}.md", md)
    _write_both("campaign_latest.md", md)

    method = "\n".join(
        [
            "# Verification methodology",
            "",
            "This file is the standing protocol. Dated runs live next to it as",
            "`campaign_<UTC>.md` / `.json`. The manuscript cites `campaign_latest.md`.",
            "",
            "## What a run is allowed to claim",
            "",
            "| Kind | Allowed claim in the paper |",
            "|---|---|",
            "| `verified_here` | Reproduced on the campaign host. |",
            "| `air_pinned` | Measured on the author's M4 Air; pins must not drift. |",
            "| `cannot_retime` | Named gap. Do not upgrade to a result. |",
            "",
            "## What a run is not",
            "",
            "- Not a Metal retime of 27B tok/s.",
            "- Not a task-quality study of the envelope versus the full OMP harness.",
            "- Not a fanless soak.",
            "- Not permission to mark a rejected catalog id as adopted.",
            "",
            "## Research questions",
            "",
        ]
        + [f"### {q['id']}. {q['title']}\n\n{q['question']}\n" for q in RESEARCH_QUESTIONS]
        + [
            "## Independent / dependent variables",
            "",
            f"- IV: {PROTOCOL['independent']}",
            f"- DV: {PROTOCOL['dependent']}",
            f"- Held out: {PROTOCOL['not_measured']}",
            "",
            "## Command",
            "",
            "```",
            "PYTHONPATH=src python3 scripts/verify_paper.py",
            "```",
            "",
        ]
    )
    _write_both("METHODOLOGY.md", method)

    print((paper.stdout or "") + (paper.stderr or ""))
    print((filtered.stdout or "") + (filtered.stderr or ""))
    print(f"wrote {REPO_OUT / ('campaign_' + stamp + '.md')}")
    print(f"ok={summary['ok']} paper={paper.returncode} filtered={filtered.returncode} full={full.returncode}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
