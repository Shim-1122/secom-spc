"""PCA-based multivariate SPC: Hotelling T^2 and SPE (Q) charts.

With hundreds of correlated sensors, charting each one separately
gives a flood of false alarms (at 3 sigma and 400 sensors, roughly 1 in
3 perfectly normal runs trips at least one chart). Projecting onto a
few principal components and monitoring two statistics is the standard
fix:

* T^2 measures distance *within* the model plane (unusual but
  correlated-consistent behaviour).
* SPE / Q measures distance *off* the plane (the correlation structure
  itself broke, e.g. a single sensor drifted).

References: Jackson & Mudholkar (1979); MacGregor & Kourti (1995);
Montgomery, *Introduction to Statistical Quality Control*, Ch. 11.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class MSPCResult:
    scores: pd.DataFrame
    t2: np.ndarray
    spe: np.ndarray
    t2_ucl: float
    spe_ucl: float
    n_components: int
    explained: np.ndarray
    loadings: pd.DataFrame
    t2_contrib: pd.DataFrame
    spe_contrib: pd.DataFrame
    t2_ucl_emp: float = np.nan
    spe_ucl_emp: float = np.nan


def t2_ucl_phase2(p: int, m: int, alpha: float) -> float:
    """UCL for future (Phase II) individual observations."""
    return p * (m + 1) * (m - 1) / (m * (m - p)) * stats.f.ppf(1 - alpha, p, m - p)


def spe_ucl_box(spe_ref: np.ndarray, alpha: float) -> float:
    """Box (1954) scaled chi-square approximation: g * chi2_h."""
    mu, var = float(np.mean(spe_ref)), float(np.var(spe_ref, ddof=1))
    g = var / (2 * mu)
    h = 2 * mu * mu / var
    return g * stats.chi2.ppf(1 - alpha, h)


def fit_mspc(
    X: pd.DataFrame,
    ref_mask: np.ndarray,
    var_explained: float = 0.8,
    alpha: float = 0.01,
    clip_z: float | None = 10.0,
) -> MSPCResult:
    """Fit scaler + PCA on reference runs, score every run.

    ``clip_z`` winsorises standardised values so that a single extreme
    reading does not dominate the PCA fit. Scores for monitoring are
    computed on the same clipped scale.
    """
    scaler = StandardScaler().fit(X.values[ref_mask])
    Z = scaler.transform(X.values)
    sd = scaler.scale_
    Z[:, sd == 0] = 0.0
    if clip_z is not None:
        Z = np.clip(Z, -clip_z, clip_z)

    pca_full = PCA(svd_solver="full").fit(Z[ref_mask])
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    a = int(np.searchsorted(cum, var_explained) + 1)
    pca = PCA(n_components=a, svd_solver="full").fit(Z[ref_mask])

    T = pca.transform(Z)
    lam = pca.explained_variance_
    t2 = np.sum(T**2 / lam, axis=1)
    Zhat = pca.inverse_transform(T)
    E = Z - Zhat
    spe = np.sum(E**2, axis=1)

    m = int(ref_mask.sum())
    t2_ucl = t2_ucl_phase2(a, m, alpha)
    spe_ucl = spe_ucl_box(spe[ref_mask], alpha)

    P = pca.components_.T  # (n_vars, a)
    # T^2 contribution of variable j: sum_a (t_a / lam_a) * p_ja * z_j
    Zc = Z - pca.mean_
    t2c = (T / lam) @ P.T * Zc
    cols = list(X.columns)
    return MSPCResult(
        scores=pd.DataFrame(T[:, : min(a, 5)], columns=[f"PC{i+1}" for i in range(min(a, 5))], index=X.index),
        t2=t2,
        spe=spe,
        t2_ucl=float(t2_ucl),
        spe_ucl=float(spe_ucl),
        n_components=a,
        explained=pca_full.explained_variance_ratio_,
        loadings=pd.DataFrame(P, index=cols, columns=[f"PC{i+1}" for i in range(a)]),
        t2_contrib=pd.DataFrame(t2c, columns=cols, index=X.index),
        spe_contrib=pd.DataFrame(E**2, columns=cols, index=X.index),
        # Distribution-free alternative: the (1 - alpha) quantile of the
        # reference statistics. Preferable when sensors are heavy tailed,
        # because the F / chi-square limits assume multivariate normality.
        t2_ucl_emp=float(np.quantile(t2[ref_mask], 1 - alpha)),
        spe_ucl_emp=float(np.quantile(spe[ref_mask], 1 - alpha)),
    )


def top_contributors(contrib: pd.DataFrame, run_idx, k: int = 5) -> list[tuple[str, float]]:
    row = contrib.loc[run_idx]
    share = row.abs() / row.abs().sum()
    top = share.sort_values(ascending=False).head(k)
    return list(zip(top.index, top.values))
