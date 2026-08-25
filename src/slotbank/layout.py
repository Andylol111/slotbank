from __future__ import annotations

import os
from dataclasses import dataclass

GIB = 1 << 30
M4_AIR_UNIFIED_BANDWIDTH = 120e9
MIN_KV_BYTES = 256 << 20
MLX_QUANT_BITS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
LARGE_MOE_STORED_BYTES = 8 * GIB


def parse_byte_size(text: str) -> int:
    s = text.strip().replace(" ", "").lower()
    if not s:
        raise ValueError("empty byte size")
    suffixes = (
        ("gib", GIB),
        ("gb", GIB),
        ("g", GIB),
        ("mib", 1 << 20),
        ("mb", 1 << 20),
        ("m", 1 << 20),
        ("kib", 1024),
        ("kb", 1024),
        ("k", 1024),
        ("b", 1),
    )
    mul = 1
    num = s
    for suf, m in suffixes:
        if s.endswith(suf):
            num, mul = s[: -len(suf)], m
            break
    try:
        value = float(num) if "." in num else int(num)
    except ValueError as exc:
        raise ValueError(f"invalid byte size: {text!r}") from exc
    if value < 0:
        raise ValueError("byte size must be >= 0")
    return int(value * mul)


def recommended_leave_free(total_bytes: int) -> int:
    gb = total_bytes / GIB
    if gb < 12:
        return min(total_bytes // 2, 4 * GIB)
    if gb < 20:
        return 6 * GIB
    if gb < 32:
        return 8 * GIB
    if gb < 40:
        return 10 * GIB
    if gb < 56:
        return 12 * GIB
    return 16 * GIB


@dataclass(frozen=True)
class DeviceProfile:
    total_bytes: int
    available_bytes: int
    leave_free_bytes: int
    max_working_set_bytes: int
    heap_kind: str

    def fits(self, stored_or_active_bytes: int, kv_bytes: int) -> bool:
        if stored_or_active_bytes < 0 or kv_bytes < 0:
            return False
        return stored_or_active_bytes + kv_bytes <= self.max_working_set_bytes


def device_profile(
    total_bytes: int,
    *,
    leave_free_bytes: int | None = None,
    available_bytes: int | None = None,
    heap_kind: str = "unified",
) -> DeviceProfile:
    if total_bytes <= 0:
        raise ValueError("total_bytes must be > 0")
    lf = recommended_leave_free(total_bytes) if leave_free_bytes is None else int(leave_free_bytes)
    if lf < 0:
        raise ValueError("leave_free_bytes must be >= 0")
    if lf >= total_bytes:
        raise ValueError("leave_free_bytes must be < total_bytes")
    avail = total_bytes if available_bytes is None else int(available_bytes)
    return DeviceProfile(
        total_bytes=total_bytes,
        available_bytes=avail,
        leave_free_bytes=lf,
        max_working_set_bytes=total_bytes - lf,
        heap_kind=heap_kind,
    )


def _available_bytes(total: int) -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return total


def detect_device_profile(*, leave_free_bytes: int | None = None) -> DeviceProfile:
    total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return device_profile(
        total,
        leave_free_bytes=leave_free_bytes,
        available_bytes=_available_bytes(total),
    )


def slot_floor(num_experts: int, top_k: int) -> int:
    e = max(0, int(num_experts))
    k = max(0, int(top_k))
    if e <= 0:
        return 0
    if k <= 0:
        return min(e, 16)
    return min(e, max(1, 2 * k))


def resident_expert_bytes(
    stored_bytes: int,
    capacity: int,
    num_experts: int,
    *,
    expert_param_frac: float = 0.8,
) -> int:
    if stored_bytes < 0 or capacity < 0 or num_experts <= 0:
        raise ValueError("stored_bytes, capacity >= 0 and num_experts > 0")
    frac = min(1.0, max(0.0, float(expert_param_frac)))
    shared = stored_bytes * (1.0 - frac)
    experts = stored_bytes * frac
    return int(shared + experts * (min(int(capacity), int(num_experts)) / int(num_experts)))


def capacity_for_budget(
    stored_bytes: int,
    num_experts: int,
    top_k: int,
    budget_bytes: int,
    *,
    expert_param_frac: float = 0.8,
) -> int:
    """Largest slot capacity whose resident bytes fit ``budget_bytes``.

    The inverse of :func:`resident_expert_bytes`. Non-expert weights are
    resident no matter what, so a budget below that floor cannot be honoured --
    the floor capacity is returned rather than a nonsensical zero, and the
    caller is over budget by construction.
    """
    e = max(0, int(num_experts))
    floor = slot_floor(e, top_k)
    if e <= 0 or stored_bytes <= 0 or budget_bytes <= 0:
        return floor
    frac = min(1.0, max(0.0, float(expert_param_frac)))
    shared = stored_bytes * (1.0 - frac)
    experts = stored_bytes * frac
    if experts <= 0:
        return floor
    room = int(budget_bytes) - shared
    if room <= 0:
        return floor
    return min(e, max(floor, int(e * room / experts)))


def slot_capacity(
    num_experts: int,
    top_k: int,
    *,
    stored_bytes: int | None = None,
    working_set_bytes: int | None = None,
    kv_bytes: int = MIN_KV_BYTES,
    expert_param_frac: float = 0.8,
    mode: str = "auto",
) -> int:
    floor = slot_floor(num_experts, top_k)
    e = int(num_experts)
    if floor <= 0 or e <= 0:
        return floor
    kind = (mode or "auto").strip().lower()
    if kind not in {"ram", "auto", "full"}:
        raise ValueError(f"unknown slot mode {mode!r}")
    if kind == "ram" or stored_bytes is None or working_set_bytes is None:
        return floor
    stored = int(stored_bytes)
    work = int(working_set_bytes)
    kv = max(0, int(kv_bytes))
    if stored < 0 or work <= 0:
        return floor

    def budget_c() -> int:
        max_res = work - kv
        if max_res <= 0:
            return floor
        frac = min(1.0, max(0.0, float(expert_param_frac)))
        shared = stored * (1.0 - frac)
        experts = stored * frac
        if experts <= 0:
            return floor
        take = min(1.0, max(0.0, (max_res - shared) / experts))
        return min(e, max(floor, int(take * e)))

    fits = stored + kv <= work
    large = stored >= LARGE_MOE_STORED_BYTES

    def slotted_c() -> int:
        # A pack that does not fit competes with the page cache holding the rest
        # of the bank, so filling the working set is slower than staying small.
        # Measured on Qwen3.5-35B-A3B (E=256) on 24 GB: C=32 matched C=64 for
        # speed at 2.1 GiB less, and beat the budget-filling C=227 by ~16x.
        return min(budget_c(), max(floor, e // 8))

    if kind == "full":
        return e if fits else slotted_c()
    if not fits:
        return slotted_c()
    if large:
        return e
    return floor


def decode_toks_ceiling(bandwidth_bytes_s: float, bytes_per_token: int) -> float:
    if bandwidth_bytes_s <= 0 or bytes_per_token <= 0:
        return 0.0
    return bandwidth_bytes_s / bytes_per_token


@dataclass(frozen=True)
class ModelMemoryCard:
    kind: str
    n_params: int
    bits: float
    group_size: int
    stored_bytes: int
    active_bytes: int
    active_frac: float
    n_routed_experts: int
    top_k: int
    expert_param_frac: float


def quantized_bytes(n_params: int, bits: float, group_size: int = 64) -> int:
    if n_params < 0 or bits <= 0 or group_size <= 0:
        raise ValueError("n_params, bits, group_size must be positive")
    weights = n_params * bits / 8.0
    scales = 0.0 if bits >= 16 else (n_params / group_size) * 2.0
    return int(weights + scales)


def model_memory_card(
    n_params: int,
    bits: float,
    *,
    kind: str = "dense",
    n_routed_experts: int = 0,
    top_k: int = 0,
    expert_param_frac: float = 0.8,
    group_size: int = 64,
) -> ModelMemoryCard:
    stored = quantized_bytes(n_params, bits, group_size)
    if kind == "dense":
        active, frac = stored, 1.0
    elif kind == "moe":
        if n_routed_experts <= 0 or top_k <= 0:
            raise ValueError("moe card needs n_routed_experts > 0 and top_k > 0")
        shared = n_params * (1.0 - expert_param_frac)
        experts = n_params * expert_param_frac
        active_params = shared + experts * (top_k / n_routed_experts)
        active = quantized_bytes(int(active_params), bits, group_size)
        frac = active_params / n_params
    else:
        raise ValueError(f"unknown card kind {kind!r}")
    return ModelMemoryCard(
        kind=kind,
        n_params=n_params,
        bits=float(bits),
        group_size=group_size,
        stored_bytes=stored,
        active_bytes=active,
        active_frac=frac,
        n_routed_experts=n_routed_experts,
        top_k=top_k,
        expert_param_frac=expert_param_frac,
    )


@dataclass(frozen=True)
class Admission:
    ok: bool
    weight_bytes: int
    kv_bytes: int
    max_working_set_bytes: int
    leave_free_bytes: int
    recommend_bits: float | None
    reason: str


def _card_at_bits(card: ModelMemoryCard, bits: float) -> ModelMemoryCard:
    return model_memory_card(
        card.n_params,
        bits,
        kind=card.kind,
        n_routed_experts=card.n_routed_experts,
        top_k=card.top_k,
        expert_param_frac=card.expert_param_frac,
        group_size=card.group_size,
    )


def recommend_lower_bits(
    profile: DeviceProfile,
    card: ModelMemoryCard,
    kv_bytes: int,
    *,
    use_active: bool,
) -> float | None:
    for bits in reversed(MLX_QUANT_BITS):
        if bits >= card.bits:
            continue
        smaller = _card_at_bits(card, bits)
        weights = smaller.active_bytes if use_active else smaller.stored_bytes
        if profile.fits(weights, kv_bytes):
            return bits
    return None


def admit(
    profile: DeviceProfile,
    card: ModelMemoryCard,
    kv_bytes: int,
    *,
    use_active: bool = False,
) -> Admission:
    weights = card.active_bytes if use_active else card.stored_bytes
    ok = profile.fits(weights, kv_bytes)
    rec = None if ok else recommend_lower_bits(
        profile, card, kv_bytes, use_active=use_active
    )
    if ok:
        reason = (
            f"fits: weights {weights} + kv {kv_bytes} "
            f"<= max working set {profile.max_working_set_bytes} "
            f"(leave-free {profile.leave_free_bytes})"
        )
    else:
        hint = (
            f"; try {rec:g}-bit"
            if rec is not None
            else "; no supported lower bit width fits"
        )
        reason = (
            f"working set does not fit: weights {weights} + kv {kv_bytes} "
            f"> max working set {profile.max_working_set_bytes} "
            f"(total {profile.total_bytes}, leave-free {profile.leave_free_bytes})"
            f"{hint}"
        )
    return Admission(
        ok=ok,
        weight_bytes=weights,
        kv_bytes=kv_bytes,
        max_working_set_bytes=profile.max_working_set_bytes,
        leave_free_bytes=profile.leave_free_bytes,
        recommend_bits=rec,
        reason=reason,
    )
