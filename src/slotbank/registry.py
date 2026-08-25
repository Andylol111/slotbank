"""Local model management over the Hugging Face cache.

Ollama keeps its own blob store and manifest format. There is no reason to
here: the models are already safetensors in ``~/.cache/huggingface``, mlx-lm
resolves from it, and ``huggingface_hub`` ships the scan and delete APIs. So
this module is name resolution plus a thin wrapper -- no second copy of any
model, and a checkpoint pulled by any other MLX tool is already visible.

Filesystem and network only. Nothing here imports mlx (see the import fence
test), so ``list`` and ``rm`` stay fast and work without a GPU.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

# Short names resolve against these, in order, so `slotbank run qwen3.5-35b`
# works without a registry file to maintain.
NAMESPACES = ("mlx-community", "unsloth", "Qwen", "mlx-lm")


@dataclass(frozen=True)
class LocalModel:
    repo_id: str
    size_bytes: int
    path: str
    nb_files: int

    @property
    def is_mlx(self) -> bool:
        return "mlx" in self.repo_id.lower() or "-4bit" in self.repo_id


def _repos():
    from huggingface_hub import scan_cache_dir

    try:
        return scan_cache_dir().repos
    except Exception:                             # cache absent or unreadable
        return ()


def local_models(mlx_only: bool = True) -> list[LocalModel]:
    """Cached models, largest first. Weight-bearing repos only.

    A repo whose largest revision is under 64 MiB is a config-only or tokenizer
    pull -- ``scan_cache_dir`` reports those alongside real checkpoints, and
    listing them as runnable models would be a lie.
    """
    out = []
    for r in _repos():
        if r.repo_type != "model" or r.size_on_disk < (64 << 20):
            continue
        rev = max(r.revisions, key=lambda v: v.size_on_disk, default=None)
        m = LocalModel(r.repo_id, r.size_on_disk,
                       str(rev.snapshot_path) if rev else str(r.repo_path),
                       len(rev.files) if rev else 0)
        if mlx_only and not m.is_mlx:
            continue
        out.append(m)
    return sorted(out, key=lambda m: -m.size_bytes)


def resolve(name: str) -> str:
    """Turn a short name into a repo id.

    A path or an explicit ``owner/repo`` passes through untouched. Otherwise
    match the cache case-insensitively first -- so a name that already works
    keeps working -- then fall back to the known namespaces.
    """
    if os.path.isdir(name) or "/" in name:
        return name
    low = name.lower()
    cached = local_models(mlx_only=False)
    for m in cached:                              # exact repo name
        if m.repo_id.split("/")[-1].lower() == low:
            return m.repo_id
    for m in cached:                              # unique substring
        if low in m.repo_id.lower():
            return m.repo_id
    return f"{NAMESPACES[0]}/{name}"


def local_path(repo_id: str) -> str | None:
    """Snapshot directory of an already-pulled model, or None."""
    if os.path.isdir(repo_id):
        return repo_id
    for m in local_models(mlx_only=False):
        if m.repo_id == repo_id:
            return m.path
    return None


def pull(repo_id: str, revision: str = "main") -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id, revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
    )


def remove(repo_id: str) -> int:
    """Delete every revision of a repo. Returns bytes freed."""
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    hashes = [rev.commit_hash for r in cache.repos if r.repo_id == repo_id
              for rev in r.revisions]
    if not hashes:
        raise ValueError(f"{repo_id} is not in the cache")
    strategy = cache.delete_revisions(*hashes)
    freed = strategy.expected_freed_size
    strategy.execute()
    return freed


def disk_free() -> int:
    return shutil.disk_usage(os.path.expanduser("~")).free
