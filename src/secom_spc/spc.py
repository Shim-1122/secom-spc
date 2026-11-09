"""Univariate control charts: I-MR, X-bar/S, p, Laney p', EWMA and CUSUM.

Two masks recur throughout this module:

``baseline``
    Runs used to *estimate* control limits (here, the passing runs of
    Phase I).
``phase1``
    Runs that belong to Phase I in time. It only determines the
    ``phase`` label of each point; it does not affect the limits. When
    omitted it defaults to ``baseline``.

Limits estimated on the baseline are applied unchanged to every run.
Optionally, baseline points beyond the limits are removed and the limits
re-estimated, which is the usual Phase I clean-up step (Montgomery,
2019, ch. 6).

Control-chart constants are computed in closed form, except d2 and d3,
which are taken from the standard table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import gamma, sqrt

import numpy as np
import pandas as pd

from . import rules

# Mean and standard deviation of the relative range W = R / sigma, n = 2..10.
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
D3_TABLE = {2: 0.853, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848, 7: 0.833, 8: 0.820, 9: 0.808, 10: 0.797}


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
def c4(n: int) -> float:
    """Bias-correction constant for the sample standard deviation."""
    return sqrt(2.0 / (n - 1)) * gamma(n / 2.0) / gamma((n - 1) / 2.0)


def a3(n: int, k: float = 3.0) -> float:
    """X-bar limit factor when sigma is estimated by S-bar."""
    return k / (c4(n) * sqrt(n))


def b3_b4(n: int, k: float = 3.0) -> tuple[float, float]:
    """Lower and upper S-chart limit factors."""
    c = c4(n)
    w = k * sqrt(1 - c * c) / c
    return max(0.0, 1 - w), 1 + w


def d3_d4(n: int = 2, k: float = 3.0) -> tuple[float, float]:
    """Lower and upper range-chart limit factors."""
    d2, d3 = D2[n], D3_TABLE[n]
    return max(0.0, 1 - k * d3 / d2), 1 + k * d3 / d2


@dataclass(frozen=True)
class Limits:
    """Centre line, sigma and control limits of one chart."""

    center: float
    sigma: float
    lcl: float
    ucl: float
    n_baseline: int
    n_removed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _phase_labels(n: int, baseline: np.ndarray, phase1: np.ndarray | None) -> np.ndarray:
    mask = baseline if phase1 is None else phase1
    if len(mask) != n:
        raise ValueError("mask length does not match the series")
    return np.where(mask, "I", "II")


# ----------------------------------------------------------------------
# Individuals and moving range
# ----------------------------------------------------------------------
def imr_limits(
    x: np.ndarray, baseline: np.ndarray, k: float = 3.0, max_iter: int = 3
) -> tuple[Limits, Limits]:
    """Estimate I and MR chart limits from the baseline runs.

    The within-run sigma is MR-bar / d2(2), with moving ranges taken over
    consecutive baseline readings. Up to ``max_iter`` rounds of removing
    points beyond ``k`` sigma are applied before the final estimate.
    """
    xb = np.asarray(x, dtype=float)[baseline]
    xb = xb[np.isfinite(xb)]
    removed = 0
    for iteration in range(max_iter + 1):
        mr = np.abs(np.diff(xb))
        mrbar = float(mr.mean()) if mr.size else np.nan
        sigma = mrbar / D2[2]
        center = float(xb.mean())
        if not np.isfinite(sigma) or sigma == 0 or iteration == max_iter:
            break
        outside = np.abs(xb - center) > k * sigma
        if not outside.any():
            break
        removed += int(outside.sum())
        xb = xb[~outside]

    lim_i = Limits(center, sigma, center - k * sigma, center + k * sigma, xb.size, removed)
    d3l, d4u = d3_d4(2, k)
    lim_mr = Limits(mrbar, sigma, d3l * mrbar, d4u * mrbar, max(xb.size - 1, 0))
    return lim_i, lim_mr


def imr_chart(
    x: np.ndarray,
    baseline: np.ndarray,
    k: float = 3.0,
    max_iter: int = 3,
    phase1: np.ndarray | None = None,
) -> pd.DataFrame:
    """Individuals and moving-range chart for one sensor.

    Returns one row per run with the value, moving range, limits, the
    standardised value ``z``, a column per Nelson rule and ``any_rule``.
    Limits are stored in ``df.attrs["limits_i"]`` and ``df.attrs["limits_mr"]``.
    """
    x = np.asarray(x, dtype=float)
    li, lmr = imr_limits(x, baseline, k, max_iter)
    mr = np.r_[np.nan, np.abs(np.diff(x))]
    z = (x - li.center) / li.sigma if li.sigma > 0 else np.full_like(x, np.nan)
    flags = rules.evaluate(z)

    df = pd.DataFrame(
        {
            "value": x,
            "mr": mr,
            "cl": li.center,
            "lcl": li.lcl,
            "ucl": li.ucl,
            "mr_cl": lmr.center,
            "mr_ucl": lmr.ucl,
            "z": z,
            "phase": _phase_labels(len(x), baseline, phase1),
        }
    )
    for r, f in flags.items():
        df[f"rule{r}"] = f
    df["mr_ooc"] = mr > lmr.ucl
    df["any_rule"] = np.logical_or.reduce(list(flags.values()))
    df.attrs["limits_i"] = li.as_dict()
    df.attrs["limits_mr"] = lmr.as_dict()
    return df


# ----------------------------------------------------------------------
# X-bar / S on rational subgroups of consecutive runs
# ----------------------------------------------------------------------
def make_subgroups(n_runs: int, size: int) -> np.ndarray:
    """Subgroup index for each run; runs in an incomplete final block get -1."""
    gid = np.arange(n_runs) // size
    return np.where(gid < n_runs // size, gid, -1)


def xbar_s_chart(
    x: np.ndarray,
    baseline: np.ndarray,
    size: int = 5,
    k: float = 3.0,
    max_iter: int = 3,
    phase1: np.ndarray | None = None,
) -> pd.DataFrame:
    """X-bar and S chart on blocks of ``size`` consecutive runs.

    Limits are estimated from complete subgroups whose runs all belong to
    ``baseline``. A subgroup is labelled Phase I when all of its runs are
    in ``phase1``.
    """
    x = np.asarray(x, dtype=float)
    phase_mask = baseline if phase1 is None else phase1
    gid = make_subgroups(len(x), size)
    df = pd.DataFrame({"x": x, "g": gid, "base": baseline, "p1": phase_mask})
    df = df[df.g >= 0]
    grp = df.groupby("g")
    st = pd.DataFrame(
        {
            "xbar": grp.x.mean(),
            "s": grp.x.std(ddof=1),
            "n": grp.x.count(),
            "first_run": grp.x.apply(lambda s: s.index.min()),
            "baseline": grp.base.all(),
            "phase_I": grp.p1.all(),
        }
    )

    ref = st[st.baseline & (st.n == size)]
    A3 = a3(size, k)
    removed = 0
    for iteration in range(max_iter + 1):
        xbb, sbar = ref.xbar.mean(), ref.s.mean()
        lcl, ucl = xbb - A3 * sbar, xbb + A3 * sbar
        outside = (ref.xbar < lcl) | (ref.xbar > ucl)
        if iteration == max_iter or not outside.any():
            break
        removed += int(outside.sum())
        ref = ref[~outside]

    B3, B4 = b3_b4(size, k)
    sigma_xbar = sbar / (c4(size) * sqrt(size))
    st["cl"], st["lcl"], st["ucl"] = xbb, lcl, ucl
    st["s_cl"], st["s_lcl"], st["s_ucl"] = sbar, B3 * sbar, B4 * sbar
    st["z"] = (st.xbar - xbb) / sigma_xbar if sigma_xbar > 0 else np.nan
    flags = rules.evaluate(st.z.to_numpy())
    for r, f in flags.items():
        st[f"rule{r}"] = f
    st["s_ooc"] = (st.s > st.s_ucl) | (st.s < st.s_lcl)
    st["any_rule"] = np.logical_or.reduce(list(flags.values()))
    st.attrs["n_removed"] = removed
    st.attrs["sigma_within"] = sbar / c4(size)
    return st.reset_index()


# ----------------------------------------------------------------------
# Attribute charts for the yield KPI
# ----------------------------------------------------------------------
def p_chart(fails, totals, baseline: np.ndarray, k: float = 3.0) -> pd.DataFrame:
    """p chart with variable limits, one point per day or lot."""
    fails = np.asarray(fails, dtype=float)
    totals = np.asarray(totals, dtype=float)
    pbar = fails[baseline].sum() / totals[baseline].sum()
    p = fails / totals
    se = np.sqrt(pbar * (1 - pbar) / totals)
    lcl = np.clip(pbar - k * se, 0, None)
    ucl = np.clip(pbar + k * se, None, 1)
    return pd.DataFrame(
        {
            "p": p,
            "n": totals,
            "fails": fails,
            "pbar": pbar,
            "lcl": lcl,
            "ucl": ucl,
            "z": (p - pbar) / se,
            "ooc_high": p > ucl,
            "ooc_low": p < lcl,
            "phase": np.where(baseline, "I", "II"),
        }
    )


def laney_p_chart(fails, totals, baseline: np.ndarray, k: float = 3.0) -> pd.DataFrame:
    """Laney p' chart, which widens p-chart limits for over-dispersion.

    sigma_z is the moving-range sigma of the baseline z scores (Laney,
    2002). With sigma_z = 1 the chart reduces to the ordinary p chart.
    """
    df = p_chart(fails, totals, baseline, k)
    zb = df.z.to_numpy()[baseline]
    zb = zb[np.isfinite(zb)]
    sigma_z = float(np.abs(np.diff(zb)).mean() / D2[2]) if zb.size > 1 else 1.0
    se = np.sqrt(df.pbar * (1 - df.pbar) / df.n) * sigma_z
    df["lcl"] = np.clip(df.pbar - k * se, 0, None)
    df["ucl"] = np.clip(df.pbar + k * se, None, 1)
    df["ooc_high"] = df.p > df.ucl
    df["ooc_low"] = df.p < df.lcl
    df.attrs["sigma_z"] = sigma_z
    return df


# ----------------------------------------------------------------------
# Time-weighted charts
# ----------------------------------------------------------------------
def ewma_chart(x, center: float, sigma: float, lam: float = 0.2, L: float = 2.86) -> pd.DataFrame:
    """EWMA chart with exact, time-varying limits. Missing values hold the statistic."""
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, center)
    z = np.empty_like(x)
    prev = center
    for i, xi in enumerate(x):
        prev = lam * xi + (1 - lam) * prev
        z[i] = prev
    t = np.arange(1, len(x) + 1)
    width = L * sigma * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
    return pd.DataFrame(
        {"ewma": z, "cl": center, "lcl": center - width, "ucl": center + width, "ooc": np.abs(z - center) > width}
    )


def cusum_chart(x, center: float, sigma: float, k: float = 0.5, h: float = 5.0) -> pd.DataFrame:
    """Two-sided tabular CUSUM; ``k`` and ``h`` are in units of sigma."""
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, center)
    K, H = k * sigma, h * sigma
    c_plus = np.zeros_like(x)
    c_minus = np.zeros_like(x)
    sp = sm = 0.0
    for i, xi in enumerate(x):
        sp = max(0.0, xi - (center + K) + sp)
        sm = max(0.0, (center - K) - xi + sm)
        c_plus[i], c_minus[i] = sp, sm
    return pd.DataFrame({"c_plus": c_plus, "c_minus": c_minus, "h": H, "ooc": (c_plus > H) | (c_minus > H)})


# ----------------------------------------------------------------------
# All sensors at once
# ----------------------------------------------------------------------
def run_rule_matrix(
    X: pd.DataFrame,
    baseline: np.ndarray,
    k: float = 3.0,
    max_iter: int = 3,
    phase1: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """Apply an I-MR chart with Nelson rules to every column of ``X``.

    Returns
    -------
    dict with keys
        ``summary``: one row per sensor (limits and alarm rates by phase);
        ``rule1``: run x sensor booleans for points beyond ``k`` sigma;
        ``any_rule``: run x sensor booleans for any Nelson rule.
    """
    summary, rule1, any_rule = [], {}, {}
    for col in X.columns:
        ch = imr_chart(X[col].to_numpy(), baseline, k, max_iter, phase1)
        li = ch.attrs["limits_i"]
        in_p2 = (ch.phase == "II").to_numpy()
        row = {
            "sensor": col,
            "center": li["center"],
            "sigma_within": li["sigma"],
            "lcl": li["lcl"],
            "ucl": li["ucl"],
            "n_baseline": li["n_baseline"],
            "n_removed_phase1": li["n_removed"],
            "rule1_rate_phase1": ch.rule1[~in_p2].mean(),
            "rule1_rate_phase2": ch.rule1[in_p2].mean(),
            "any_rule_rate_phase2": ch.any_rule[in_p2].mean(),
        }
        for r in rules.RULE_NAMES:
            row[f"n_rule{r}_phase2"] = int(ch[f"rule{r}"][in_p2].sum())
        summary.append(row)
        rule1[col] = ch.rule1.to_numpy()
        any_rule[col] = ch.any_rule.to_numpy()
    return {
        "summary": pd.DataFrame(summary).set_index("sensor"),
        "rule1": pd.DataFrame(rule1, index=X.index),
        "any_rule": pd.DataFrame(any_rule, index=X.index),
    }
