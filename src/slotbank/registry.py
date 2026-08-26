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
    def has_chat_template(self) -> bool:
        """Whether this checkpoint can hold a conversation.

        A base model without a template gets plain 'user: ...' text and
        responds by continuing the transcript -- inventing both sides of a
        dialogue. Fine as a completion model, wrong as a chat default.
        """
        import json
        import os.path as _p

        if _p.exists(_p.join(self.path, "chat_template.jinja")):
            return True
        cfg = _p.join(self.path, "tokenizer_config.json")
        try:
            with open(cfg) as fh:
                return bool(json.load(fh).get("chat_template"))
        except (OSError, ValueError):
            return False

    @property
    def is_mlx(self) -> bool:
        """Loadable by this runtime, decided from the files rather than the name.

        The old test pattern-matched "mlx" or "-4bit" in the repo id, so any
        -8bit/-6bit/bf16/mxfp4 checkpoint became invisible to `list` the moment
        it was pulled, and a valid repo whose name says neither was hidden too.
        The answer is on disk: safetensors plus a config, and no GGUF.
        """
        import glob

        if not os.path.isdir(self.path):
            return False
        if glob.glob(os.path.join(self.path, "*.gguf")):
            return False            # llama.cpp format, not readable here
        cfg = os.path.join(self.path, "config.json")
        if not glob.glob(os.path.join(self.path, "*.safetensors")) \
                or not os.path.exists(cfg):
            return False
        # An embedding or reranker model has safetensors and a config too, so
        # the file test alone lists bge/ModernBERT/MiniLM as runnable. Only a
        # causal LM generates text.
        import json

        try:
            with open(cfg) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            return False
        arch = " ".join(d.get("architectures") or [])
        if "ForCausalLM" in arch or "ForConditionalGeneration" in arch:
            return True
        text = d.get("text_config") or {}
        return bool(text.get("num_hidden_layers")) and "Bert" not in arch


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
    # A role is an alias, resolved in the one funnel every command already uses,
    # so `run @chat`, `serve --model @chat` and `check @chat` all inherit it
    # without any per-command special casing. See docs/cli-design.md section 8.
    if name.startswith("@"):
        entry = load_roles().get(name[1:])
        if entry is None:
            role = name[1:]
            raise ValueError(
                f"no role {role!r}. Set one: slotbank use {role} <model>")
        name = entry["model"]
    if os.path.isdir(name) or "/" in name:
        return name
    low = name.lower()
    # GGUF is llama.cpp's format; this runtime reads safetensors. Matching one
    # sends the user to a checkpoint that cannot load.
    cached = [m for m in local_models(mlx_only=False)
              if "gguf" not in m.repo_id.lower()]
    for m in cached:                              # exact repo name
        if m.repo_id.split("/")[-1].lower() == low:
            return m.repo_id
    # MLX repos first, so a substring shared by several prefers a runnable one.
    for m in sorted(cached, key=lambda m: not m.is_mlx):
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


ALLOW = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]


def pull_plan(repo_id: str, revision: str = "main", tqdm_class=None) -> list:
    """What a pull would fetch, without fetching it.

    Returns one ``DryRunFileInfo`` per file: filename, file_size, is_cached,
    will_download, commit_hash. Measured at 0.33 s for a 12-file repo. This is
    what makes an honest byte total possible before the first byte moves --
    ``snapshot_download``'s own progress total starts at 0 and grows as each
    file's bar is built, so a percentage taken from it is wrong early on.

    ``tqdm_class`` exists only to silence the hub's own "[dry-run] Fetching N
    files" bar, which would otherwise paint over the header this call feeds.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id, revision=revision, allow_patterns=ALLOW, dry_run=True,
        **({"tqdm_class": tqdm_class} if tqdm_class is not None else {}),
    )


def pull(repo_id: str, revision: str = "main", tqdm_class=None) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id, revision=revision, allow_patterns=ALLOW,
        **({"tqdm_class": tqdm_class} if tqdm_class is not None else {}),
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


def search(query: str, limit: int = 8) -> list[str]:
    """Repo ids matching a query, MLX ones first.

    Only called when a name fails to resolve, so `list` and `run` stay offline.
    """
    from huggingface_hub import HfApi

    try:
        ids = [m.id for m in HfApi().list_models(search=f"{query} mlx", limit=limit)]
    except Exception:
        return []
    return sorted(ids, key=lambda i: (not i.startswith("mlx-community/"), i))


# --- roles: a model alias with an effort, per docs/cli-design.md section 8 ----

ROLES_PATH = os.path.expanduser("~/.config/slotbank/roles.json")


def load_roles() -> dict:
    import json

    try:
        with open(ROLES_PATH) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    # Tolerate a hand-edited file: keep only well-formed entries rather than
    # failing every command because one role is malformed.
    return {k: v for k, v in data.items()
            if isinstance(v, dict) and isinstance(v.get("model"), str)}


def save_role(role: str, model: str | None = None, effort: str | None = None) -> dict:
    import json

    roles = load_roles()
    entry = dict(roles.get(role) or {})
    if model is not None:
        entry["model"] = model
    if effort is not None:
        entry["effort"] = effort
    if "model" not in entry:
        raise ValueError(f"role {role!r} has no model yet; give one: "
                         f"slotbank use {role} <model>")
    roles[role] = entry
    os.makedirs(os.path.dirname(ROLES_PATH), exist_ok=True)
    tmp = ROLES_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(roles, fh, indent=2, sort_keys=True)
    os.replace(tmp, ROLES_PATH)          # never leave a half-written config
    return roles[role]
