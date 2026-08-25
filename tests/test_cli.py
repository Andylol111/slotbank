

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


def test_encode_chat_falls_back_without_a_template():
    """A base model has apply_chat_template but no template, and transformers
    raises ValueError rather than returning None. Every chat endpoint 500s
    without this fallback."""
    from slotbank.prompt import encode_chat

    class Tok:
        def apply_chat_template(self, *a, **k):
            raise ValueError("chat_template is not set")

        def encode(self, text):
            return [len(text)]

    assert encode_chat(Tok(), [{"role": "user", "content": "hi"}], None) == [8]   # "user: hi"


def test_resolve_passes_through_explicit_ids(tmp_path):
    from slotbank.registry import resolve

    assert resolve("owner/repo") == "owner/repo"
    assert resolve(str(tmp_path)) == str(tmp_path)
    assert "/" in resolve("Some-Unknown-Model-4bit")
