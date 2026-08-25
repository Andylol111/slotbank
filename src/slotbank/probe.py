"""Inspect a remote checkpoint without downloading it.

A safetensors shard begins with an 8-byte length and a JSON header describing
every tensor's dtype, shape and byte range. Range-requesting that header costs
a couple of MB against a checkpoint of any size, which is enough to compute the
two numbers that decide whether a model can run here at all: the resident floor
(everything that is not a routed expert) and the per-expert row size.

Network and JSON only -- this module must stay free of MLX (tests/test_fence).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

EXPERT_MARKERS = (".switch_mlp.", ".experts.", ".block_sparse_moe.experts.")
_UA = {"User-Agent": "slotbank-probe"}


@dataclass(frozen=True)
class RemoteCard:
    repo: str
    total_bytes: int
    expert_bytes: int
    resident_bytes: int
    num_experts: int
    top_k: int
    layers: int
    row_bytes: int          # one expert, all projections
    shards: int
    scanned_layers: int
    model_type: str

    @property
    def expert_frac(self) -> float:
        return self.expert_bytes / self.total_bytes if self.total_bytes else 0.0

    @property
    def touched_bytes(self) -> int:
        """Expert bytes read per token: top_k experts on every MoE layer."""
        return self.top_k * self.layers * self.row_bytes


def _get(url: str, rng: tuple[int, int] | None = None, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=dict(_UA))
    if rng is not None:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _base(repo: str, revision: str = "main") -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/"


def _header(url: str) -> dict:
    """Read one shard's tensor table via two range requests."""
    n = int.from_bytes(_get(url, (0, 7)), "little")
    if n <= 0 or n > (64 << 20):
        raise ValueError(f"implausible safetensors header length: {n}")
    return json.loads(_get(url, (8, 8 + n - 1)))


def probe(repo: str, revision: str = "main", max_shards: int | None = None) -> RemoteCard:
    """Measure a remote checkpoint's expert/resident split from its headers."""
    base = _base(repo, revision)
    cfg = json.loads(_get(base + "config.json"))
    text = cfg.get("text_config") or cfg

    def pick(*names, default=0):
        for n in names:
            v = cfg.get(n, text.get(n))
            if v is not None:
                return v
        return default

    try:
        shards = sorted(set(json.loads(
            _get(base + "model.safetensors.index.json"))["weight_map"].values()))
    except (urllib.error.HTTPError, KeyError, ValueError):
        shards = ["model.safetensors"]
    if max_shards:
        # partial scans extrapolate, and early shards are embedding-heavy, so
        # the resident figure skews high; prefer a full scan
        shards = shards[:max_shards]

    expert = resident = 0
    rows: dict[str, int] = {}
    layers: set[str] = set()
    # Headers are a few KB each; fetch them concurrently so a 42-shard model
    # costs about as long as a single request.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(min(8, max(1, len(shards)))) as ex:
        heads = list(ex.map(lambda sh: _header(base + sh), shards))
    for head in heads:
        for key, meta in head.items():
            if not isinstance(meta, dict) or "data_offsets" not in meta:
                continue
            lo, hi = meta["data_offsets"]
            size = int(hi) - int(lo)
            if ".layers." in key:
                layers.add(key.split(".layers.")[1].split(".")[0])
            if any(m in key for m in EXPERT_MARKERS):
                expert += size
                shape = meta.get("shape") or []
                if shape and shape[0]:
                    # one expert's slice of this tensor
                    rows[key] = size // int(shape[0])
            else:
                resident += size

    # Trust the config for layer count: a partial scan sees only some layers,
    # and undercounting them silently deflates touched-per-token.
    n_layers = int(pick("num_hidden_layers", default=0)) or len(layers) or 1
    n_experts = int(pick("num_experts", "n_routed_experts")) or 1
    scanned = len(layers) or n_layers
    # One expert across all projections, derived from bytes actually seen so a
    # partial scan still gives the right per-expert figure.
    row_bytes = int(expert / (n_experts * scanned)) if expert else 0
    if scanned < n_layers:
        # extrapolate the bank and resident weights to the full model
        expert = int(expert * n_layers / scanned)
        resident = int(resident * n_layers / scanned)
    return RemoteCard(
        repo=repo,
        total_bytes=expert + resident,
        expert_bytes=expert,
        resident_bytes=resident,
        num_experts=n_experts,
        top_k=int(pick("num_experts_per_tok", "moe_top_k")),
        layers=n_layers,
        row_bytes=row_bytes,
        shards=len(shards),
        scanned_layers=scanned,
        model_type=str(pick("model_type", default="?")),
    )
