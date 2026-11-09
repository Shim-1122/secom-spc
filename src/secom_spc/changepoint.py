"""Locating a sustained process change without using the pass/fail labels.

A fab re-baselines its control limits after a known event (preventive
maintenance, a recipe change, new instrumentation). SECOM records no such
events, so the most likely one is estimated from the sensors themselves.

Method
------
Each sensor is rank-transformed over the whole record (robust to heavy
tails), missing ranks are set to the median, and the result is
standardised. For every candidate boundary ``k`` at the first run of a
production day, the statistic

    S(k) = sum_j  k (n - k) / n * (mean_j[:k] - mean_j[k:])^2

is the sum over sensors of the squared two-sample mean difference scaled
as in the CUSUM / binary-segmentation change-point statistic. The
estimate is the boundary that maximises S(k). Applying the same search
inside each resulting segment gives secondary change points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ChangePoint:
    index: int                 # first run after the change (0-based)
    timestamp: pd.Timestamp
    statistic: float
    profile: pd.Series         # S(k) at each candidate boundary, indexed by run index
    secondary: list[int]       # change points found inside each segment


def _rank_standardise(X: pd.DataFrame) -> np.ndarray:
    R = X.rank(pct=True)
    R = R.fillna(R.median())
    sd = R.std().replace(0, 1.0)
    return ((R - R.mean()) / sd).to_numpy()


def _profile(Z: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    n = Z.shape[0]
    cs = np.cumsum(Z, axis=0)
    total = cs[-1]
    k = candidates
    left = cs[k - 1] / k[:, None]
    right = (total - cs[k - 1]) / (n - k)[:, None]
    return (k * (n - k) / n)[:, None].ravel() * ((left - right) ** 2).sum(axis=1)


def day_boundaries(dates: pd.Series) -> np.ndarray:
    """Index of the first run of each production day."""
    return np.flatnonzero(dates.ne(dates.shift()).to_numpy())


def detect(
    X: pd.DataFrame, dates: pd.Series, timestamps: pd.Series, min_frac: float = 0.05
) -> ChangePoint:
    """Single most likely mean-shift change point at a day boundary."""
    Z = _rank_standardise(X)
    n = Z.shape[0]
    cand = day_boundaries(dates)
    cand = cand[(cand >= min_frac * n) & (cand <= (1 - min_frac) * n)]
    prof = _profile(Z, cand)
    best = cand[int(np.argmax(prof))]

    secondary = []
    for a, b in ((0, best), (best, n)):
        seg = cand[(cand > a + 0.1 * (b - a)) & (cand < b - 0.1 * (b - a))]
        if seg.size:
            secondary.append(int(seg[int(np.argmax(_profile(Z[a:b], seg - a)))]))

    return ChangePoint(
        index=int(best),
        timestamp=timestamps.iloc[best],
        statistic=float(prof.max()),
        profile=pd.Series(prof, index=cand, name="S"),
        secondary=secondary,
    )
