"""Custom collate function for multi-modal samples.

The default openpi collate uses `jax.tree.map(np.stack)`, which works only for
homogeneous numeric leaves. Multi-modal samples carry extra Python-typed leaves
(`task_name`, `prompt`) and bool masks that must be preserved. This collator:

- Stacks numeric leaves with `np.stack`.
- Stacks bool/int leaves with their native dtype.
- Collects string leaves into a Python list (keeps order, no `np.stack`).
- Preserves the nested dict structure verbatim.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

_STRING_KEYS = frozenset({"prompt", "task_name"})


def _stack_leaves(values: list):
    first = values[0]
    if isinstance(first, str):
        return list(values)
    arrs = [np.asarray(v) for v in values]
    return np.stack(arrs, axis=0)


def _collate_recursive(samples: Iterable[dict | object], key_chain: tuple[str, ...] = ()):
    samples = list(samples)
    first = samples[0]

    # Dict node: recurse on each child key.
    if isinstance(first, dict):
        out: dict = {}
        for k in first:
            children = [s[k] for s in samples]
            out[k] = _collate_recursive(children, key_chain=(*key_chain, k))
        return out

    # String leaves stay as a list.
    last_key = key_chain[-1] if key_chain else ""
    if last_key in _STRING_KEYS or isinstance(first, str):
        return [s if isinstance(s, str) else str(s) for s in samples]

    return _stack_leaves(samples)


def multimodal_collate_fn(items: list[dict]) -> dict:
    """Collate a list of multi-modal sample dicts into a batched dict."""
    if not items:
        raise ValueError("Cannot collate an empty list of items.")
    return _collate_recursive(items)
