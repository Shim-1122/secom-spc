"""Nelson (Western Electric) run rules.

Each rule takes standardised values ``z = (x - CL) / sigma`` and returns a
boolean array that is True at the point completing the pattern, which is
how most SPC software marks a violation.

=====  ========================================================
Rule   Pattern (Nelson, 1984)
=====  ========================================================
1      one point beyond 3 sigma
2      nine in a row on the same side of the centre line
3      six in a row steadily increasing or decreasing
4      fourteen in a row alternating up and down
5      two of three beyond 2 sigma, same side
6      four of five beyond 1 sigma, same side
7      fifteen in a row within 1 sigma (stratification)
8      eight in a row beyond 1 sigma, either side (mixture)
=====  ========================================================

Rules 1, 5, 6 and an eight-point version of rule 2 form the original
Western Electric set.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

RULE_NAMES: dict[int, str] = {
    1: "1 point > 3 sigma",
    2: "9 in a row same side",
    3: "6 in a row trending",
    4: "14 in a row alternating",
    5: "2 of 3 > 2 sigma same side",
    6: "4 of 5 > 1 sigma same side",
    7: "15 in a row within 1 sigma",
    8: "8 in a row > 1 sigma either side",
}

# Number of consecutive points each run-length rule inspects.
RUN_LENGTH: dict[int, int] = {2: 9, 3: 6, 4: 14, 7: 15, 8: 8}


def _count_in_window(cond: np.ndarray, window: int, need: int) -> np.ndarray:
    """True at i if at least ``need`` of ``cond[i - window + 1 : i + 1]`` are True."""
    out = np.zeros(cond.shape[0], dtype=bool)
    if cond.shape[0] >= window:
        counts = sliding_window_view(cond.astype(np.int16), window).sum(axis=-1)
        out[window - 1 :] = counts >= need
    return out


def _shift_right(flags: np.ndarray, n: int) -> np.ndarray:
    """Map flags indexed by differences back onto point indices."""
    out = np.zeros(n, dtype=bool)
    out[1:] = flags
    return out


def rule1(z: np.ndarray) -> np.ndarray:
    return np.abs(z) > 3


def rule2(z: np.ndarray, n: int = RUN_LENGTH[2]) -> np.ndarray:
    return _count_in_window(z > 0, n, n) | _count_in_window(z < 0, n, n)


def rule3(z: np.ndarray, n: int = RUN_LENGTH[3]) -> np.ndarray:
    d = np.diff(z)  # n points trending = n - 1 differences of one sign
    trend = _count_in_window(d > 0, n - 1, n - 1) | _count_in_window(d < 0, n - 1, n - 1)
    return _shift_right(trend, z.size)


def rule4(z: np.ndarray, n: int = RUN_LENGTH[4]) -> np.ndarray:
    d = np.sign(np.diff(z))
    flip = np.zeros(d.shape[0], dtype=bool)
    flip[1:] = d[1:] * d[:-1] < 0  # n alternating points = n - 2 consecutive flips
    return _shift_right(_count_in_window(flip, n - 2, n - 2), z.size)


def rule5(z: np.ndarray) -> np.ndarray:
    return _count_in_window(z > 2, 3, 2) | _count_in_window(z < -2, 3, 2)


def rule6(z: np.ndarray) -> np.ndarray:
    return _count_in_window(z > 1, 5, 4) | _count_in_window(z < -1, 5, 4)


def rule7(z: np.ndarray, n: int = RUN_LENGTH[7]) -> np.ndarray:
    return _count_in_window(np.abs(z) < 1, n, n)


def rule8(z: np.ndarray, n: int = RUN_LENGTH[8]) -> np.ndarray:
    return _count_in_window(np.abs(z) > 1, n, n)


_RULES: dict[int, Callable[[np.ndarray], np.ndarray]] = {
    1: rule1, 2: rule2, 3: rule3, 4: rule4, 5: rule5, 6: rule6, 7: rule7, 8: rule8,
}


def evaluate(z: np.ndarray, which: Iterable[int] = tuple(RULE_NAMES)) -> dict[int, np.ndarray]:
    """Apply the selected rules to a standardised series.

    A missing value never triggers a rule, and a run-length pattern cannot
    span a missing value.
    """
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(z)
    filled = np.where(finite, z, 0.0)
    out = {}
    for r in which:
        flags = _RULES[r](filled)
        if r in RUN_LENGTH:
            flags &= ~_count_in_window(~finite, RUN_LENGTH[r], 1)
        out[r] = flags & finite
    return out


def any_violation(z: np.ndarray, which: Iterable[int] = tuple(RULE_NAMES)) -> np.ndarray:
    """True wherever at least one of the selected rules fires."""
    return np.logical_or.reduce(list(evaluate(z, which).values()))
