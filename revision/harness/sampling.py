
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _as_arrays(p0: Mapping[str, float]) -> tuple[list[str], np.ndarray]:
    keys = list(p0.keys())
    probs = np.fromiter((p0[k] for k in keys), dtype=float, count=len(keys))
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("p0 does not normalise")
    return keys, probs / total


def draw_counts(p0: Mapping[str, float], m: int,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    keys, probs = _as_arrays(p0)
    draw = rng.multinomial(int(m), probs)
    nz = np.flatnonzero(draw)
    if nz.size == 0:
        d = len(keys[0]) if keys else 0
        return np.empty((0, d), dtype=np.int64), np.empty(0, dtype=np.int64)
    states = np.array([[int(c) for c in keys[i]] for i in nz], dtype=np.int64)
    counts = draw[nz].astype(np.int64)
    return states, counts


def X_from_counts(states: np.ndarray, counts: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    states = np.asarray(states, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if states.shape[0] != counts.shape[0]:
        raise ValueError(f"states {states.shape} and counts {counts.shape} disagree")
    X = np.repeat(states, counts, axis=0)
    rng.shuffle(X, axis=0)
    return X


def counts_from_X(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.int64)
    if X.size == 0:
        return X.reshape(0, X.shape[1] if X.ndim == 2 else 0), np.empty(0, dtype=np.int64)
    states, counts = np.unique(X, axis=0, return_counts=True)
    return states, counts.astype(np.int64)


def states_to_keys(states: np.ndarray) -> list[str]:
    return ["".join(str(int(b)) for b in row) for row in np.asarray(states)]
