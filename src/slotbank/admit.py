from __future__ import annotations

import json
import os
import platform
import re
from pathlib import Path
from typing import Any

from slotbank.layout import MIN_KV_BYTES, admit, detect_device_profile, model_memory_card


def require_apple_silicon() -> None:
    if platform.system().lower() != "darwin":
        raise ValueError("slotbank requires macOS")
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        raise ValueError("slotbank requires Apple Silicon")


def load_hf_config(model_path: str) -> dict[str, Any]:
    path = Path(model_path)
    cfg_path = path / "config.json" if path.is_dir() else None
    if cfg_path is None or not cfg_path.is_file():
        return {}
    return json.loads(cfg_path.read_text())


def draft_block_from_config(model_path: str) -> int:
    """Trained verify length. DFlash2 is 8, native MTP is 3; else 5.

    Forcing one K for every drafter makes both routes the same wrong length.
    """
    cfg = load_hf_config(model_path)
    inner = cfg.get("dflash_config")
    n = _first_int(
        inner.get("block_size") if isinstance(inner, dict) else None,
        cfg.get("block_size"),
    )
    return n if n > 0 else 5


def _first_int(*values: Any) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


_HEXISH = re.compile(r"^[0-9a-f]{16,}$", re.I)


def _params_from_name(name: str) -> int | None:
    """Parameter count from a folder name, e.g. 'OLMoE-1B-7B' -> 7B.

    Two traps: a Hugging Face snapshot directory is a commit hash, and hex
    digits followed by 'b' parse as a parameter count ('23511b9407...' -> 23.5T).
    And a MoE name carries both active and total ('1B-7B', '35B-A3B'), where the
    total is the larger one.
    """
    if _HEXISH.match(name):
        return None
    hits = [float(m.group(1))
            for m in re.finditer(r"(?<![0-9a-fA-F])(\d+(?:\.\d+)?)\s*[Bb](?![0-9a-zA-Z])", name)]
    if not hits:
        return None
    return int(max(hits) * 1_000_000_000)


def stored_bytes_from_files(model_path: str) -> int:
    """Actual weight bytes on disk. Preferred over guessing from the name."""
    root = Path(model_path)
    if not root.is_dir():
        return 0
    total = 0
    for f in root.glob("*.safetensors"):
        try:
            total += os.path.getsize(os.path.realpath(f))
        except OSError:
            continue
    return total


_EXPERT_MARKERS = (".switch_mlp.", ".experts.", ".block_sparse_moe.experts.")


def expert_frac_from_files(model_path: str) -> float | None:
    """Share of weight bytes that live in routed experts, read from the index.

    The policy defaults to 0.8; this model measures 0.889, and the gap moves a
    budget calculation by ~1.7 GiB. Reading the safetensors headers costs one
    small read per shard and makes the memory card exact instead of assumed.
    """
    root = Path(model_path)
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        return None
    try:
        weight_map = json.loads(index.read_text())["weight_map"]
    except (OSError, ValueError, KeyError):
        return None
    headers: dict[str, dict] = {}
    expert = total = 0
    for key, shard in weight_map.items():
        path = os.path.realpath(root / shard)
        head = headers.get(path)
        if head is None:
            try:
                with open(path, "rb") as fh:
                    n = int.from_bytes(fh.read(8), "little")
                    head = headers[path] = json.loads(fh.read(n))
            except (OSError, ValueError):
                return None
        meta = head.get(key)
        if not isinstance(meta, dict) or "data_offsets" not in meta:
            continue
        lo, hi = meta["data_offsets"]
        size = int(hi) - int(lo)
        total += size
        if any(m in key for m in _EXPERT_MARKERS):
            expert += size
    if total <= 0:
        return None
    return expert / total


def _params_from_stored(stored: int, bits: float, group_size: int = 64) -> int:
    """Invert quantized_bytes: bytes-per-param is bits/8 plus scale overhead."""
    per = bits / 8.0 + (0.0 if bits >= 16 else 2.0 / group_size)
    return int(stored / per) if per > 0 else 0


def _bits_from_config(cfg: dict[str, Any]) -> float | None:
    quant = cfg.get("quantization") or cfg.get("quantization_config") or {}
    bits = quant.get("bits") or quant.get("n_bits")
    if bits is None:
        return None
    return float(bits)


def estimate_card(args: Any):
    n = getattr(args, "n_params", None)
    bits = getattr(args, "quant_bits", None)
    kind = getattr(args, "model_kind", None)
    n_e = int(getattr(args, "n_routed_experts", 0) or 0)
    top_k = int(getattr(args, "top_k_experts", 0) or 0)
    path = str(getattr(args, "model_path", "") or "")
    cfg = load_hf_config(path)
    text = cfg.get("text_config") or {}
    if not n_e:
        n_e = _first_int(
            cfg.get("num_experts"),
            cfg.get("num_local_experts"),
            cfg.get("n_routed_experts"),
            text.get("num_experts"),
            text.get("num_local_experts"),
            text.get("n_routed_experts"),
        )
    if not top_k:
        top_k = _first_int(
            cfg.get("num_experts_per_tok"),
            cfg.get("moe_top_k"),
            text.get("num_experts_per_tok"),
            text.get("moe_top_k"),
        )
    if bits is None:
        bits = _bits_from_config(cfg)
    if n is None:
        n = cfg.get("num_parameters")
    if n is None and bits is not None:
        # Real bytes beat a parsed name: the name may be a commit hash, and a
        # MoE name carries both active and total counts.
        stored = stored_bytes_from_files(path)
        if stored > 0:
            n = _params_from_stored(stored, float(bits))
    if n is None:
        n = _params_from_name(Path(path).name)
    if n is None or bits is None:
        raise ValueError("cannot estimate model memory card; refuse to load blind")
    if kind is None:
        kind = "moe" if n_e > 0 and top_k > 0 else "dense"
    frac = expert_frac_from_files(path) if kind == "moe" else None
    kwargs = {} if frac is None else {"expert_param_frac": float(frac)}
    return model_memory_card(
        int(n),
        float(bits),
        kind=kind,
        n_routed_experts=n_e,
        top_k=top_k,
        **kwargs,
    )


_RECURRENT_KEYS = (
    "linear_conv_kernel_dim", "linear_key_head_dim", "linear_num_key_heads",
    "mamba_d_conv", "ssm_state_size", "conv_kernel",
)


def kv_bytes_per_token(cfg: dict[str, Any]) -> int:
    """Full-attention KV bytes per token. Linear-attn state is O(1), not here."""
    t = cfg.get("text_config") or cfg
    types = t.get("layer_types")
    if isinstance(types, list):
        n_full = sum(1 for x in types if "full" in str(x) and "linear" not in str(x))
    else:
        n_layers = _first_int(t.get("num_hidden_layers"), t.get("num_layers"))
        interval = _first_int(t.get("full_attention_interval")) or 4
        n_full = max(1, n_layers // interval) if n_layers else 16
    heads = _first_int(t.get("num_key_value_heads")) or 4
    dim = _first_int(t.get("head_dim")) or 256
    return max(1, n_full) * heads * dim * 2 * 2


def max_context_tokens(profile, card, cfg: dict[str, Any], *, slop: int = 1 << 30) -> int:
    """How many tokens fit after weights + slop. Dense hybrid advisor."""
    per = kv_bytes_per_token(cfg)
    room = profile.max_working_set_bytes - card.stored_bytes - slop
    return max(0, room // per) if per else 0


def draft_viable(
    profile,
    card,
    hybrid: str | None,
    *,
    draft_bytes: int = 1 << 30,
    verify: str = "trim",
) -> str | None:
    """Why a draft model should stay off, or None if it could fit.

    ``verify="trim"`` is mlx-lm speculative_generate_step: a rejected draft
    is rewound with trim_prompt_cache. Hybrid linear-attn caches cannot do
    that, so this path is unsafe on Qwen3.8.

    ``verify="dflash"`` is mlx-vlm's exact DFlash/MTP target verification,
    which rolls Gated DeltaNet state back on reject. Hybrid is allowed;
    the draft still has to fit beside the target. No extra 1 GiB slop:
    4-bit 27B + a 1 GiB draft is already tight on 24 GB (leave-free 6).
    """
    if hybrid and verify != "dflash":
        return f"UNSAFE -- {hybrid}; a rejected draft cannot be rewound"
    room = profile.max_working_set_bytes - card.stored_bytes - MIN_KV_BYTES
    need = draft_bytes if verify == "dflash" else draft_bytes + (1 << 30)
    if room < need:
        return f"no headroom: {room} bytes free, need {need} for a {draft_bytes} byte draft"
    return None


def refuse_draft_if_needed(model_path: str, profile, card) -> None:
    """Admit a ``SLOTBANK_DRAFT`` pack using mlx-vlm verify, not trim."""
    path = os.path.expanduser(os.environ.get("SLOTBANK_DRAFT", "").strip())
    if not path:
        return
    if not os.path.isdir(path):
        raise ValueError(f"draft not found: {path}")
    stored = stored_bytes_from_files(path) or 0
    cfg = load_hf_config(path)
    quant = cfg.get("quantization") or cfg.get("quantization_config")
    if stored > (2 << 30) and not quant:
        raise ValueError(
            f"draft is {stored / (1 << 30):.1f} GiB unquantized; convert to 4-bit:\n"
            f"  python -m mlx_vlm.convert --hf-path {path} "
            f"--mlx-path {path}-4bit --quantize --q-bits 4 --q-group-size 64"
        )
    why = check_draft_compatible(model_path, path)
    if why:
        raise ValueError(f"draft: {why}")
    stored = stored or (1 << 30)
    why = draft_viable(
        profile,
        card,
        hybrid_from_config(load_hf_config(model_path)),
        draft_bytes=stored,
        verify="dflash",
    )
    if why:
        raise ValueError(
            f"draft: {why}; 4-bit 27B + DFlash on 24 GB needs --leave-free 6g"
        )


def hybrid_from_config(cfg: dict[str, Any]) -> str | None:
    """Detect recurrent/linear-attention layers from config alone.

    These models keep a running state that cannot be rolled back, which makes
    speculative decoding silently wrong. Detected without loading weights so
    ``admit`` stays cheap.
    """
    t = cfg.get("text_config") or cfg
    types = t.get("layer_types")
    if isinstance(types, list):
        kinds = {str(x) for x in types}
        odd = {k for k in kinds if "linear" in k or "mamba" in k or "recurrent" in k}
        if odd and len(kinds) > 1:
            return "mixed layer_types: " + ", ".join(sorted(kinds))
    if t.get("full_attention_interval"):
        return f"full_attention_interval={t['full_attention_interval']} (most layers recurrent)"
    hits = [k for k in _RECURRENT_KEYS if t.get(k)]
    if hits:
        return "recurrent config keys: " + ", ".join(hits)
    return None


def check_draft_compatible(model_path: str, draft_path: str) -> str | None:
    """Why a draft model cannot be used for speculative decoding, or None.

    Speculative decoding requires an identical vocabulary: the target verifies
    token ids the draft produced. Qwen3 (151936) and Qwen3.5 (248320) look
    interchangeable and are not, and a mismatch corrupts output silently.
    """
    tgt, dft = load_hf_config(model_path), load_hf_config(draft_path)
    if not tgt or not dft:
        return "cannot read config.json for target or draft"
    t = tgt.get("text_config") or tgt
    d = dft.get("text_config") or dft
    tv, dv = t.get("vocab_size"), d.get("vocab_size")
    dft_type = str(dft.get("model_type") or "").lower()
    dflash = "dflash" in dft_type or dft.get("dflash_config") is not None
    if tv is None or dv is None:
        # DFlash2 checkpoints often omit vocab_size; they emit target token ids.
        if dflash and tv is not None:
            return None
        return "vocab_size missing from a config; refuse to guess"
    if int(tv) != int(dv):
        return f"vocab mismatch: target {tv} vs draft {dv}"
    # Same vocab is not enough. A Qwen3.5-4B and Qwen3.8-27B both speak 248320
    # tokens; the 4B is a full LM, not an MTP/DFlash student. mlx-lm trim-spec
    # of an independent small model is silently wrong on Gated DeltaNet.
    # Matching MTP/DFlash/EAGLE drafters are allowed even when they are shallower.
    if not (dflash or "mtp" in dft_type or "eagle" in dft_type):
        th = _first_int(t.get("hidden_size"))
        dh = _first_int(d.get("hidden_size"))
        if th and dh and th != dh:
            return (
                f"hidden_size mismatch: target {th} vs draft {dh}; "
                "a smaller full model cannot draft this hybrid "
                "(use the matching MTP sidecar)"
            )
    return None


def check_speculative_supported(cache) -> str | None:
    """Why speculative decoding is unsafe for this model's cache, or None.

    On rejection the target cache must be rewound. Hybrid models keep recurrent
    linear-attention state that cannot be rolled back, and mlx-lm's
    speculative_generate_step does not check: trim_prompt_cache returns 0 and
    generation silently continues conditioned on rejected tokens.
    """
    if not cache:
        return "empty cache"
    # Duck-typed rather than importing mlx-lm: this module must stay MLX-free
    # (tests/test_fence.py). Mirrors can_trim_prompt_cache.
    trimmable = all(getattr(c, "is_trimmable", lambda: False)() for c in cache)
    if not trimmable:
        kinds = sorted({type(c).__name__ for c in cache})
        return ("cache is not trimmable (" + ", ".join(kinds) + "); rejected draft "
                "tokens could not be rewound and output would be silently wrong")
    return None


def admit_or_raise(args: Any, *, profile=None, card=None, kv_bytes=None):
    require_apple_silicon()
    profile = profile or detect_device_profile(
        leave_free_bytes=getattr(args, "leave_free", None)
    )
    card = card or estimate_card(args)
    kv = MIN_KV_BYTES if kv_bytes is None else kv_bytes
    result = admit(profile, card, kv, use_active=(card.kind == "moe"))
    if not result.ok:
        raise ValueError(result.reason)
    return result
