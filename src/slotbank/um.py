from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from slotbank.layout import Admission, DeviceProfile, detect_device_profile

PRESSURE_NORMAL = 1
PRESSURE_WARN = 2
PRESSURE_CRITICAL = 4


@dataclass(frozen=True)
class UmSnapshot:
    pressure: int
    page_size: int
    free_bytes: int
    wired_bytes: int
    compressor_occupied_bytes: int
    compressor_stored_bytes: int
    should_shed: bool


def parse_vm_stat(text: str) -> dict[str, int]:
    page_size = 16384
    m = re.search(r"page size of (\d+) bytes", text)
    if m:
        page_size = int(m.group(1))
    pages: dict[str, int] = {"page_size": page_size}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        digits = raw.strip().rstrip(".").replace(",", "")
        if digits.isdigit():
            pages[key.strip().strip('"')] = int(digits)
    return pages


def _pages(pages: dict[str, int], *names: str) -> int:
    for name in names:
        if name in pages:
            return pages[name]
    return 0


def snapshot_from_vm_stat(text: str, *, pressure: int = PRESSURE_NORMAL) -> UmSnapshot:
    pages = parse_vm_stat(text)
    ps = pages["page_size"]
    free = _pages(pages, "Pages free") * ps
    wired = _pages(pages, "Pages wired down") * ps
    occ = _pages(pages, "Pages occupied by compressor") * ps
    stored = _pages(pages, "Pages stored in compressor") * ps
    shed = pressure >= PRESSURE_WARN or (wired > 0 and free < 256 << 20)
    return UmSnapshot(
        pressure=int(pressure),
        page_size=ps,
        free_bytes=free,
        wired_bytes=wired,
        compressor_occupied_bytes=occ,
        compressor_stored_bytes=stored,
        should_shed=shed,
    )


def read_pressure_level() -> int:
    if sys.platform != "darwin":
        return PRESSURE_NORMAL
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            text=True,
            timeout=2,
        )
        return int(out.strip() or PRESSURE_NORMAL)
    except (OSError, ValueError, subprocess.SubprocessError):
        return PRESSURE_NORMAL


def read_vm_stat_text() -> str:
    if sys.platform != "darwin":
        return ""
    try:
        return subprocess.check_output(["vm_stat"], text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return ""


class UmManager:
    def __init__(
        self,
        profile: DeviceProfile,
        admission: Admission | None = None,
        card: Any = None,
    ):
        self.profile = profile
        self.admission = admission
        self.card = card

    @classmethod
    def from_args(cls, args: Any) -> UmManager:
        from slotbank.admit import admit_or_raise, estimate_card

        profile = detect_device_profile(leave_free_bytes=getattr(args, "leave_free", None))
        card = estimate_card(args)
        admission = admit_or_raise(args, profile=profile, card=card)
        return cls(profile, admission, card)

    def load_kwargs(self) -> dict[str, Any]:
        return {"lazy": True, "tokenizer_config": {"trust_remote_code": False}}

    def snapshot(self) -> UmSnapshot:
        text = read_vm_stat_text()
        if not text:
            return UmSnapshot(
                pressure=read_pressure_level(),
                page_size=0,
                free_bytes=0,
                wired_bytes=0,
                compressor_occupied_bytes=0,
                compressor_stored_bytes=0,
                should_shed=False,
            )
        return snapshot_from_vm_stat(text, pressure=read_pressure_level())

    def should_shed(self, snap: UmSnapshot | None = None) -> bool:
        return (snap or self.snapshot()).should_shed

    def blocking_pressure(self, snap: UmSnapshot | None = None) -> bool:
        return (snap or self.snapshot()).pressure >= PRESSURE_CRITICAL

    def note(self, snap: UmSnapshot | None = None) -> dict[str, int | str | bool]:
        s = snap or self.snapshot()
        return {
            "heap_kind": self.profile.heap_kind,
            "total_bytes": self.profile.total_bytes,
            "leave_free_bytes": self.profile.leave_free_bytes,
            "max_working_set_bytes": self.profile.max_working_set_bytes,
            "pressure": s.pressure,
            "free_bytes": s.free_bytes,
            "wired_bytes": s.wired_bytes,
            "compressor_occupied_bytes": s.compressor_occupied_bytes,
            "should_shed": s.should_shed,
        }
