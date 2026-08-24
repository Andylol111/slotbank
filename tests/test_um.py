from __future__ import annotations

from slotbank.um import PRESSURE_WARN, parse_vm_stat, snapshot_from_vm_stat


def test_parse_vm_stat():
    text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100.\nPages wired down: 200.\n"
    pages = parse_vm_stat(text)
    assert pages["page_size"] == 16384
    assert pages["Pages free"] == 100


def test_snapshot_sheds_on_warn():
    text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages wired down: 100.\n"
    snap = snapshot_from_vm_stat(text, pressure=PRESSURE_WARN)
    assert snap.should_shed is True
    assert snap.free_bytes == 10 * 16384


def test_warm_budget_scales_with_pressure():
    from types import SimpleNamespace

    from slotbank.runtime import Runtime

    args = SimpleNamespace(model_path="x", leave_free=None, prefill_step_size=2048)

    def um_at(pressure, shed=False):
        snap = SimpleNamespace(pressure=pressure, should_shed=shed)
        return SimpleNamespace(
            profile=SimpleNamespace(max_working_set_bytes=12 << 30),
            snapshot=lambda: snap,
        )

    ceiling = (12 << 30) // 3
    assert Runtime(args, um=um_at(1))._warm_budget() == ceiling
    # should_shed is keyed on free_bytes, which macOS drives to ~0 normally;
    # it must NOT reduce the warm budget or warm start never fires.
    assert Runtime(args, um=um_at(1, shed=True))._warm_budget() == ceiling
    assert Runtime(args, um=um_at(2))._warm_budget() == ceiling // 4
    assert Runtime(args, um=um_at(4))._warm_budget() == 0
    # no um: still bounded, never unbounded
    assert 0 < Runtime(args, um=None)._warm_budget() <= (4 << 30)
