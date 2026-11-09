"""Operating characteristics of a multivariate monitor.

An engineer judges a monitor by two numbers: how many alarms someone has
to triage per production day, and how many real problems it catches.
This module compares three ways of setting the limits of the PCA
T^2 / SPE monitor:

frozen
    Fitted once on the Phase I passing runs and never updated.
re-baselined
    Refitted on the first ``n`` passing runs after the detected process
    change, then frozen. This is the standard response to a known event.
rolling
    Refitted at the start of every production day on the most recent
    ``window`` passing runs. No knowledge of the change point is needed.

A run is in alarm when T^2 or SPE exceeds its empirical 99 % limit. All
limits come from out-of-fold reference statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .multivariate import fit_mspc


def alarms_from(X: pd.DataFrame, ref_mask: np.ndarray, var_explained: float, alpha: float, seed: int = 0) -> np.ndarray:
    """Alarm flags for every run from a monitor fitted on ``ref_mask``."""
    ms = fit_mspc(X, ref_mask, var_explained, alpha, seed=seed)
    t2, spe = ms.alarms(empirical=True)
    return t2 | spe


def rolling_alarms(
    X: pd.DataFrame, y: np.ndarray, dates: pd.Series, window: int, var_explained: float, alpha: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Refit daily on the trailing ``window`` passing runs.

    Returns alarm flags and a mask of runs that could be monitored (days
    after enough passing runs had accumulated).
    """
    n = len(y)
    alarm = np.zeros(n, dtype=bool)
    covered = np.zeros(n, dtype=bool)
    passing = np.flatnonzero(np.asarray(y) == 0)
    day_vals = dates.to_numpy()
    for day in pd.unique(day_vals):
        idx = np.flatnonzero(day_vals == day)
        ref = passing[passing < idx[0]][-window:]
        if ref.size < window:
            continue
        rows = np.r_[ref, idx]
        mask = np.zeros(rows.size, dtype=bool)
        mask[: ref.size] = True
        sub = X.iloc[rows].reset_index(drop=True)
        flags = alarms_from(sub, mask, var_explained, alpha, seed)
        alarm[idx] = flags[ref.size :]
        covered[idx] = True
    return alarm, covered


def burden(alarm: np.ndarray, y: np.ndarray, dates: pd.Series, mask: np.ndarray) -> dict:
    """Alarm rate, alarms per day and failure capture over ``mask``."""
    a, yy = alarm[mask], np.asarray(y)[mask]
    days = dates[mask].nunique()
    n_alarm = int(a.sum())
    return {
        "runs": int(mask.sum()),
        "fails": int(yy.sum()),
        "days": int(days),
        "alarm_rate_pct": 100 * float(a.mean()),
        "alarms_per_day": n_alarm / days,
        "false_alarms_per_day": int((a & (yy == 0)).sum()) / days,
        "fails_flagged_pct": 100 * float(a[yy == 1].mean()) if yy.sum() else np.nan,
        "fail_rate_given_alarm_pct": 100 * float(yy[a].mean()) if n_alarm else np.nan,
        "fail_rate_given_no_alarm_pct": 100 * float(yy[~a].mean()) if (~a).any() else np.nan,
    }


def daily_alarm_table(alarms: dict[str, np.ndarray], y: np.ndarray, dates: pd.Series, covered: np.ndarray) -> pd.DataFrame:
    """Per-day runs, fails and alarm counts for each monitor."""
    df = pd.DataFrame({"date": dates.to_numpy(), "fail": y, "covered": covered})
    for name, a in alarms.items():
        df[f"alarms_{name}"] = a
    agg = {"runs": ("fail", "size"), "fails": ("fail", "sum"), "covered": ("covered", "all")}
    agg.update({f"alarms_{name}": (f"alarms_{name}", "sum") for name in alarms})
    out = df.groupby("date").agg(**agg).reset_index()
    out["fail_rate"] = out.fails / out.runs
    return out


def high_fail_days(daily: pd.DataFrame, min_fails: int, min_rate: float) -> pd.Series:
    """Days with at least ``min_fails`` failures and a fail rate of at least ``min_rate``."""
    return (daily.fails >= min_fails) & (daily.fail_rate >= min_rate)
