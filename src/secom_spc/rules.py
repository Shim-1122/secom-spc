"""Nelson / Western Electric run rules.

Each rule returns a boolean array that is True at the point that
*completes* the offending pattern, which is how most SPC software marks
violations. Inputs are standardised values ``z = (x - CL) / sigma``.

Rule summary (Nelson, 1984):

1. One point beyond 3 sigma.
2. Nine points in a row on the same side of the centre line.
3. Six points in a row steadily increasing or decreasing.
4. Fourteen points in a row alternating up and down.
5. Two out of three points beyond 2 sigma on the same side.
6. Four out of five points beyond 1 sigma on the same side.
7. Fifteen points in a row within 1 sigma (stratification).
8. Eight points in a row beyond 1 sigma on either side (mixture).

Rules 1, 5 and 6 together with a 8-point version of rule 2 are the
classic Western Electric rules.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

RULE_NAMES = {
    1: "1 point > 3 sigma",
    2: "9 in a row same side",
    3: "6 in a row trending",
    4: "14 in a row alternating",
    5: "2 of 3 > 2 sigma same side",
    6: "4 of 5 > 1 sigma same side",
    7: "15 in a row within 1 sigma",
    8: "8 in a row > 1 sigma either side",
}


def _window_flag(cond: np.ndarray, w: int, need: int) -> np.ndarray:
    """True at index i if at least ``need`` of cond[i-w+1 : i+1] are True."""
    out = np.zeros(cond.shape[0], dtype=bool)
    if cond.shape[0] < w:
        return out
    counts = swv(cond.astype(np.int16), w).sum(axis=-1)
    out[w - 1 :] = counts >= need
    return out


def rule1(z):
    return np.abs(z) > 3


def rule2(z, n=9):
    return _window_flag(z > 0, n, n) | _window_flag(z < 0, n, n)


def rule3(z, n=6):
    d = np.diff(z)
    up = _window_flag(d > 0, n - 1, n - 1)
    dn = _window_flag(d < 0, n - 1, n - 1)
    out = np.zeros_like(z, dtype=bool)
    out[1:] = up | dn
    return out


def rule4(z, n=14):
    d = np.sign(np.diff(z))
    alt = np.zeros(d.shape[0], dtype=bool)
    alt[1:] = (d[1:] * d[:-1]) < 0  # direction flipped
    # n points alternating = n-1 differences = n-2 consecutive flips
    flags = _window_flag(alt, n - 2, n - 2)
    out = np.zeros_like(z, dtype=bool)
    out[1:] = flags
    return out


def rule5(z):
    return _window_flag(z > 2, 3, 2) | _window_flag(z < -2, 3, 2)


def rule6(z):
    return _window_flag(z > 1, 5, 4) | _window_flag(z < -1, 5, 4)


def rule7(z, n=15):
    return _window_flag(np.abs(z) < 1, n, n)


def rule8(z, n=8):
    return _window_flag(np.abs(z) > 1, n, n)


_RULES = {1: rule1, 2: rule2, 3: rule3, 4: rule4, 5: rule5, 6: rule6, 7: rule7, 8: rule8}


def evaluate(z: np.ndarray, rules=(1, 2, 3, 4, 5, 6, 7, 8)) -> dict[int, np.ndarray]:
    """Apply the selected rules to a standardised series.

    NaNs never trigger a rule and break run-length patterns.
    """
    z = np.asarray(z, dtype=float)
    finite = np.isfinite(z)
    zz = np.where(finite, z, 0.0)
    out = {}
    for r in rules:
        f = _RULES[r](zz)
        if r in (2, 3, 4, 7, 8):
            # a pattern that spans a NaN is not valid; approximate by
            # requiring the flagged point itself to be finite and the
            # window to contain no NaN.
            w = {2: 9, 3: 6, 4: 14, 7: 15, 8: 8}[r]
            nan_in_window = _window_flag(~finite, w, 1)
            f = f & ~nan_in_window
        out[r] = f & finite
    return out


def any_violation(z: np.ndarray, rules=(1, 2, 3, 4, 5, 6, 7, 8)) -> np.ndarray:
    res = evaluate(z, rules)
    return np.logical_or.reduce(list(res.values()))
