"""Sensor screening and imputation, fit on Phase I only.

All statistics (which sensors to keep, medians for imputation,
duplicate groups) are learned from the baseline period and then applied
unchanged to later runs. This mirrors how a fab would freeze a
monitoring plan and avoids leaking Phase II information backwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ScreenReport:
    n_input: int = 0
    dropped_missing: list[str] = field(default_factory=list)
    dropped_constant: list[str] = field(default_factory=list)
    dropped_duplicate: dict[str, str] = field(default_factory=dict)  # dropped -> kept
    kept: list[str] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stage": [
                    "input sensors",
                    "dropped: > max missing fraction",
                    "dropped: constant / near-constant",
                    "dropped: near-duplicate (|r| above threshold)",
                    "retained",
                ],
                "count": [
                    self.n_input,
                    len(self.dropped_missing),
                    len(self.dropped_constant),
                    len(self.dropped_duplicate),
                    len(self.kept),
                ],
            }
        )


class SensorScreen:
    """Drop uninformative sensors and median-impute the rest."""

    def __init__(
        self,
        max_missing_frac: float = 0.5,
        min_unique: int = 3,
        min_rel_std: float = 1e-8,
        corr_dedup_threshold: float | None = 0.98,
    ):
        self.max_missing_frac = max_missing_frac
        self.min_unique = min_unique
        self.min_rel_std = min_rel_std
        self.corr_dedup_threshold = corr_dedup_threshold
        self.report_ = ScreenReport()
        self.medians_: pd.Series | None = None

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> SensorScreen:
        rep = ScreenReport(n_input=X.shape[1])

        miss = X.isna().mean()
        rep.dropped_missing = miss.index[miss > self.max_missing_frac].tolist()
        cols = [c for c in X.columns if c not in set(rep.dropped_missing)]

        sub = X[cols]
        nun = sub.nunique(dropna=True)
        std = sub.std(skipna=True)
        scale = sub.abs().mean(skipna=True).replace(0, 1.0)
        rel_std = (std / scale).fillna(0)
        const_mask = (nun < self.min_unique) | (rel_std < self.min_rel_std) | std.isna()
        rep.dropped_constant = const_mask.index[const_mask].tolist()
        cols = [c for c in cols if c not in set(rep.dropped_constant)]

        if self.corr_dedup_threshold is not None and len(cols) > 1:
            corr = X[cols].corr(min_periods=30).abs()
            keep: list[str] = []
            for c in cols:  # greedy: keep first sensor of each near-duplicate group
                dup_of = next(
                    (k for k in keep if pd.notna(corr.at[c, k]) and corr.at[c, k] > self.corr_dedup_threshold),
                    None,
                )
                if dup_of is None:
                    keep.append(c)
                else:
                    rep.dropped_duplicate[c] = dup_of
            cols = keep

        rep.kept = cols
        self.medians_ = X[cols].median(skipna=True)
        self.report_ = rep
        return self

    def transform(self, X: pd.DataFrame, impute: bool = True) -> pd.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("call fit() first")
        out = X[self.report_.kept].copy()
        if impute:
            out = out.fillna(self.medians_)
        return out

    def fit_transform(self, X: pd.DataFrame, impute: bool = True) -> pd.DataFrame:
        return self.fit(X).transform(X, impute=impute)


def missingness_profile(X: pd.DataFrame) -> pd.DataFrame:
    """Per-sensor missing fraction, sorted descending."""
    m = X.isna().mean().rename("missing_frac").to_frame()
    m["n_missing"] = X.isna().sum()
    return m.sort_values("missing_frac", ascending=False)


def robust_center_scale(x: np.ndarray) -> tuple[float, float]:
    """Median and MAD-based sigma (1.4826 * MAD), with an IQR fallback."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    if mad <= 0:
        q75, q25 = np.percentile(x, [75, 25])
        mad = float(q75 - q25) / 1.349
    if mad <= 0:
        mad = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    return med, mad
