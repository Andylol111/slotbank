from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "slotbank"
MLX_FILES = {"runtime.py", "expert_slots.py", "offload_cache.py"}


def test_no_brand_or_torch_in_source():
    brand = "free" + "token"
    for path in ROOT.rglob("*.py"):
        text = path.read_text().lower()
        assert brand not in text, path
        assert "torch" not in text, path


def test_only_three_files_import_mlx():
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            if path.name not in MLX_FILES:
                assert "mlx" not in names and "mlx_lm" not in names, path.name


def test_package_import_does_not_load_mlx():
    src = ROOT.parent
    code = (
        "import slotbank\n"
        "import sys\n"
        "assert slotbank.UmManager is not None\n"
        "assert 'mlx' not in sys.modules\n"
        "assert 'mlx_lm' not in sys.modules\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(src)},
    )
    assert proc.returncode == 0, proc.stderr
