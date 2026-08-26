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


# --- family comparison: one probe pays for every quant, see cli-design.md s7 --

import re as _re

# Ordered longest-first so "4bit-DWQ" is not shortened to "4bit" before the tag
# is stripped from the family stem.
_QUANT = _re.compile(
    r"[-_](?:"
    r"(?P<bits>\d+(?:\.\d+)?)\s*bit(?:s)?(?:[-_][A-Za-z0-9]+)?"
    r"|(?P<fp>mx?fp(?P<fpb>\d+))"
    r"|bf16|fp16|f16"
    r")$", _re.IGNORECASE)


def quant_bits(repo: str) -> float | None:
    """Bits per weight implied by a repo name, or None if it does not say."""
    m = _QUANT.search(repo.split("/")[-1])
    if not m:
        return None
    if m.group("bits"):
        return float(m.group("bits"))
    if m.group("fpb"):
        return float(m.group("fpb"))
    return 16.0


def family_stem(repo: str) -> str:
    """The repo name with its quant tag removed."""
    name = repo.split("/")[-1]
    m = _QUANT.search(name)
    return name[: m.start()] if m else name


def family(query: str, limit: int = 40) -> list[tuple[str, int, float]]:
    """Sibling quants of `query`: (repo_id, total_bytes, bits), smallest first.

    Sizes come from the Hub file listing, which costs one request per repo and
    no download. Repos whose name does not state a quantisation are skipped --
    the scaling in `scale` is defined by bits per weight, so a row without one
    could not be derived honestly.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    stem = family_stem(query).lower()
    out: dict[str, tuple[str, int, float]] = {}
    try:
        hits = list(api.list_models(search=f"{family_stem(query)} mlx", limit=limit))
    except Exception:
        return []
    for m in hits:
        if family_stem(m.id).lower() != stem:
            continue
        bits = quant_bits(m.id)
        if bits is None:
            continue
        try:
            info = api.model_info(m.id, files_metadata=True)
        except Exception:
            continue
        total = sum(int(f.size or 0) for f in (info.siblings or [])
                    if f.rfilename.endswith(".safetensors"))
        if total:
            out[m.id] = (m.id, total, bits)
    return sorted(out.values(), key=lambda r: r[1])


def scale(card: RemoteCard, repo: str, total_bytes: int, bits: float) -> RemoteCard:
    """A card for a sibling quant, derived from one probe plus its size.

    Layers, expert count, top-k and the expert share are architecture: they do
    not move when bits per weight do. Only byte totals scale. That assumption is
    the whole trick and it is why derived rows are marked `est` on screen -- a
    repo keeping embeddings or lm_head at higher precision tilts the expert
    share, which moves the floor.
    """
    probed = quant_bits(card.repo)
    ratio = (bits / probed) if probed else 1.0
    return RemoteCard(
        repo=repo,
        total_bytes=int(total_bytes),
        expert_bytes=int(total_bytes * card.expert_frac),
        resident_bytes=int(total_bytes * (1.0 - card.expert_frac)),
        num_experts=card.num_experts,
        top_k=card.top_k,
        layers=card.layers,
        row_bytes=int(card.row_bytes * ratio),
        shards=card.shards,
        scanned_layers=card.scanned_layers,
        model_type=card.model_type,
    )
