"""Univariate control charts: I-MR, X-bar/S, p, EWMA and CUSUM.

Conventions
-----------
* Limits are estimated on a Phase I (baseline) mask and then applied to
  every run, so Phase II points are judged against a frozen reference.
* Phase I estimation optionally iterates: points beyond 3 sigma are
  removed and limits re-estimated (up to ``max_iter`` times), the usual
  textbook "clean the baseline" step (Montgomery, Ch. 6).
* Control chart constants are computed exactly rather than read from a
  printed table, except d2/d3 which come from the standard table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import gamma, sqrt

import numpy as np
import pandas as pd

from . import rules as R

# d2 and d3 for n = 2..10 (range-based sigma estimators)
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
D3_TABLE = {2: 0.853, 3: 0.888, 4: 0.880, 5: 0.864, 6: 0.848, 7: 0.833, 8: 0.820, 9: 0.808, 10: 0.797}


def c4(n: int) -> float:
    """Bias correction for the sample standard deviation."""
    return sqrt(2.0 / (n - 1)) * gamma(n / 2.0) / gamma((n - 1) / 2.0)


def b3_b4(n: int, k: float = 3.0) -> tuple[float, float]:
    c = c4(n)
    w = k * sqrt(1 - c * c) / c
    return max(0.0, 1 - w), 1 + w


def a3(n: int, k: float = 3.0) -> float:
    return k / (c4(n) * sqrt(n))


def d3_d4(n: int = 2, k: float = 3.0) -> tuple[float, float]:
    d2, d3 = D2[n], D3_TABLE[n]
    return max(0.0, 1 - k * d3 / d2), 1 + k * d3 / d2


@dataclass
class Limits:
    center: float
    sigma: float
    lcl: float
    ucl: float
    n_baseline: int
    n_removed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------
# Individuals and moving range
# ----------------------------------------------------------------------
def imr_limits(x: np.ndarray, baseline: np.ndarray, k: float = 3.0, max_iter: int = 3) -> tuple[Limits, Limits]:
    """Estimate I and MR chart limits from the baseline runs.

    sigma_within = MR-bar / d2(2). Moving ranges are computed only over
    consecutive baseline runs.
    """
    x = np.asarray(x, dtype=float)
    xb = x[baseline]
    xb = xb[np.isfinite(xb)]
    removed = 0
    for _ in range(max_iter + 1):
        mr = np.abs(np.diff(xb))
        mrbar = float(np.mean(mr)) if mr.size else np.nan
        sigma = mrbar / D2[2]
        center = float(np.mean(xb))
        if not np.isfinite(sigma) or sigma == 0:
            break
        out = np.abs(xb - center) > k * sigma
        if not out.any() or _ == max_iter:
            break
        removed += int(out.sum())
        xb = xb[~out]
    lim_i = Limits(center, sigma, center - k * sigma, center + k * sigma, xb.size, removed)
    d3l, d4u = d3_d4(2, k)
    lim_mr = Limits(mrbar, sigma, d3l * mrbar, d4u * mrbar, max(xb.size - 1, 0), 0)
    return lim_i, lim_mr


def imr_chart(x: np.ndarray, baseline: np.ndarray, k: float = 3.0, max_iter: int = 3) -> pd.DataFrame:
    """Return a per-run frame with value, MR, limits, z and rule flags."""
    x = np.asarray(x, dtype=float)
    li, lmr = imr_limits(x, baseline, k, max_iter)
    mr = np.r_[np.nan, np.abs(np.diff(x))]
    z = (x - li.center) / li.sigma if li.sigma > 0 else np.full_like(x, np.nan)
    flags = R.evaluate(z)
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
            "phase": np.where(baseline, "I", "II"),
        }
    )
    for r, f in flags.items():
        df[f"rule{r}"] = f
    df["mr_ooc"] = mr > lmr.ucl
    df["any_rule"] = np.logical_or.reduce([flags[r] for r in flags])
    df.attrs["limits_i"] = li.as_dict()
    df.attrs["limits_mr"] = lmr.as_dict()
    return df


# ----------------------------------------------------------------------
# X-bar / S with rational subgroups of consecutive runs
# ----------------------------------------------------------------------
def make_subgroups(n_runs: int, size: int) -> np.ndarray:
    """Subgroup id for each run (consecutive blocks; ragged tail dropped)."""
    gid = np.arange(n_runs) // size
    n_full = n_runs // size
    gid = np.where(gid < n_full, gid, -1)
    return gid


def xbar_s_chart(
    x: np.ndarray, baseline: np.ndarray, size: int = 5, k: float = 3.0, max_iter: int = 3
) -> pd.DataFrame:
    """X-bar and S chart on consecutive-run subgroups.

    A subgroup belongs to Phase I only if all its runs are baseline runs.
    """
    x = np.asarray(x, dtype=float)
    gid = make_subgroups(len(x), size)
    df = pd.DataFrame({"x": x, "g": gid, "base": baseline})
    df = df[df.g >= 0]
    grp = df.groupby("g")
    stats = pd.DataFrame(
        {
            "xbar": grp.x.mean(),
            "s": grp.x.std(ddof=1),
            "n": grp.x.count(),
            "first_run": grp.x.apply(lambda s: s.index.min()),
            "phase_I": grp.base.all(),
        }
    )
    b = stats[stats.phase_I & (stats.n == size)].copy()
    removed = 0
    for it in range(max_iter + 1):
        xbb = b.xbar.mean()
        sbar = b.s.mean()
        A3 = a3(size, k)
        lcl, ucl = xbb - A3 * sbar, xbb + A3 * sbar
        out = (b.xbar < lcl) | (b.xbar > ucl)
        if not out.any() or it == max_iter:
            break
        removed += int(out.sum())
        b = b[~out]
    B3, B4 = b3_b4(size, k)
    sigma_x = sbar / (c4(size) * sqrt(size))
    stats["cl"] = xbb
    stats["lcl"] = lcl
    stats["ucl"] = ucl
    stats["s_cl"] = sbar
    stats["s_lcl"] = B3 * sbar
    stats["s_ucl"] = B4 * sbar
    stats["z"] = (stats.xbar - xbb) / sigma_x if sigma_x > 0 else np.nan
    flags = R.evaluate(stats.z.values)
    for r, f in flags.items():
        stats[f"rule{r}"] = f
    stats["s_ooc"] = (stats.s > stats.s_ucl) | (stats.s < stats.s_lcl)
    stats["any_rule"] = np.logical_or.reduce([flags[r] for r in flags])
    stats.attrs["n_removed"] = removed
    stats.attrs["sigma_within"] = sbar / c4(size)
    return stats.reset_index()


# ----------------------------------------------------------------------
# Attribute chart for the yield KPI
# ----------------------------------------------------------------------
def p_chart(fails: np.ndarray, totals: np.ndarray, baseline: np.ndarray, k: float = 3.0) -> pd.DataFrame:
    """p chart with variable limits (one point per day / lot)."""
    fails = np.asarray(fails, dtype=float)
    totals = np.asarray(totals, dtype=float)
    pbar = fails[baseline].sum() / totals[baseline].sum()
    p = fails / totals
    se = np.sqrt(pbar * (1 - pbar) / totals)
    lcl = np.clip(pbar - k * se, 0, None)
    ucl = np.clip(pbar + k * se, None, 1)
    z = (p - pbar) / se
    return pd.DataFrame(
        {
            "p": p,
            "n": totals,
            "fails": fails,
            "pbar": pbar,
            "lcl": lcl,
            "ucl": ucl,
            "z": z,
            "ooc_high": p > ucl,
            "ooc_low": p < lcl,
            "phase": np.where(baseline, "I", "II"),
        }
    )


def laney_p_chart(fails, totals, baseline, k: float = 3.0) -> pd.DataFrame:
    """Laney p' chart: corrects p-chart limits for over-dispersion.

    sigma_z is estimated from the moving range of the baseline z scores.
    When sigma_z ~ 1 this reduces to the ordinary p chart.
    """
    df = p_chart(fails, totals, baseline, k)
    zb = df.z.values[baseline]
    zb = zb[np.isfinite(zb)]
    sigma_z = float(np.mean(np.abs(np.diff(zb))) / D2[2]) if zb.size > 1 else 1.0
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
    """Tabular CUSUM (k and h in sigma units)."""
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, center)
    K, H = k * sigma, h * sigma
    cp = np.zeros_like(x)
    cm = np.zeros_like(x)
    sp = sm = 0.0
    for i, xi in enumerate(x):
        sp = max(0.0, xi - (center + K) + sp)
        sm = max(0.0, (center - K) - xi + sm)
        cp[i], cm[i] = sp, sm
    return pd.DataFrame({"c_plus": cp, "c_minus": cm, "h": H, "ooc": (cp > H) | (cm > H)})


def run_rule_matrix(Xs: pd.DataFrame, baseline: np.ndarray, k: float = 3.0, max_iter: int = 3) -> dict:
    """Run I-MR + Nelson rules on every column.

    Returns a dict with a per-sensor summary frame, a run x sensor
    boolean frame for rule 1 (beyond 3 sigma) and one for any rule.
    """
    summary = []
    r1 = {}
    anyr = {}
    for c in Xs.columns:
        ch = imr_chart(Xs[c].values, baseline, k, max_iter)
        li = ch.attrs["limits_i"]
        p2 = ch.phase == "II"
        p1 = ~p2
        row = {
            "sensor": c,
            "center": li["center"],
            "sigma_within": li["sigma"],
            "lcl": li["lcl"],
            "ucl": li["ucl"],
            "n_baseline": li["n_baseline"],
            "n_removed_phase1": li["n_removed"],
            "rule1_rate_phase1": ch.rule1[p1].mean(),
            "rule1_rate_phase2": ch.rule1[p2].mean(),
            "any_rule_rate_phase2": ch.any_rule[p2].mean(),
        }
        for r in range(1, 9):
            row[f"n_rule{r}_phase2"] = int(ch[f"rule{r}"][p2].sum())
        summary.append(row)
        r1[c] = ch.rule1.values
        anyr[c] = ch.any_rule.values
    return {
        "summary": pd.DataFrame(summary).set_index("sensor"),
        "rule1": pd.DataFrame(r1, index=Xs.index),
        "any_rule": pd.DataFrame(anyr, index=Xs.index),
    }
