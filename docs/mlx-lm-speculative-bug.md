# mlx-lm: `speculative_generate_step` silently corrupts output on untrimmable caches

**Affects:** `mlx-lm` 0.31.3 (checked), any model whose prompt cache is not trimmable —
Qwen3-Next / Qwen3.5 / Qwen3.6 MoE, and any hybrid architecture with recurrent
(linear-attention) layers.

**Severity:** silent wrong output. No exception, no warning.

## What happens

`speculative_generate_step` rewinds the target cache when draft tokens are rejected:

```python
def _rewind_cache(num_draft, num_accept):
    cache.trim_prompt_cache(model_cache, num_draft - num_accept)
    cache.trim_prompt_cache(draft_cache, max(num_draft - num_accept - 1, 0))
```

`trim_prompt_cache` guards itself and is a no-op when the cache cannot be trimmed:

```python
def trim_prompt_cache(cache, num_tokens):
    if not can_trim_prompt_cache(cache) or len(cache) == 0:
        return 0
    return [c.trim(num_tokens) for c in cache][0]
```

But `speculative_generate_step` never calls `can_trim_prompt_cache` and ignores the
return value. On a model with recurrent state the rewind silently does nothing, so the
cache retains the state of draft tokens that were **rejected and never emitted**. Every
subsequent token is generated conditioned on tokens that are not in the output.

## Why some models cannot be trimmed

`ArraysCache` (used for linear-attention / gated-delta layers) has neither `trim` nor
`is_trimmable`. Its state is a running summary of the whole prefix, not per-token
history, so it cannot be rolled back. `Qwen3.5-35B-A3B` builds 30 `ArraysCache` and 10
`KVCache` — `can_trim_prompt_cache` returns `False`.

## Reproduction

```python
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache

model, tok = load("mlx-community/Qwen3.5-35B-A3B-4bit", lazy=True)
cache = make_prompt_cache(model)
print(collections.Counter(type(c).__name__ for c in cache))
# Counter({'ArraysCache': 30, 'KVCache': 10})
print(can_trim_prompt_cache(cache))
# False   -> speculative_generate_step will produce wrong output, silently
```

## Suggested fix

Fail fast in `speculative_generate_step`:

```python
if not can_trim_prompt_cache(model_cache):
    raise ValueError(
        "speculative decoding requires a trimmable prompt cache; this model has "
        "non-trimmable (recurrent) layers"
    )
```

Falling back to non-speculative generation would also be reasonable. What is not
reasonable is the current behaviour, because the failure is invisible.

## Related

`quantized_scaled_dot_product_attention` and other paths are unaffected. The draft
model must also share the target's vocabulary — `Qwen3` (151936) and `Qwen3.5` (248320)
are not interchangeable, another silent-corruption path worth guarding at the same time.
