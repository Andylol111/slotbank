

def test_check_capacity_matches_loader_policy():
    """The checker must predict the C the loader actually picks.

    Two earlier versions used capacity_for_budget, which maximises C to fill
    the budget -- the full-residency configuration that thrashes. The loader's
    default path is slot_capacity, and a verdict computed from anything else
    is a verdict about a run that will not happen.
    """
    from slotbank.layout import MIN_KV_BYTES, slot_capacity

    stored, e, k, ws = 19 * (1 << 30), 256, 8, 16 * (1 << 30)
    c = slot_capacity(e, k, stored_bytes=stored, working_set_bytes=ws,
                      kv_bytes=MIN_KV_BYTES, expert_param_frac=0.889)
    assert k <= c <= 64, f"C={c} is outside the measured-sane band"
