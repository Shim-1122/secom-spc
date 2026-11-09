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
from sklearn.model_selection import KFold
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
    t2_ucl_emp: float
    spe_ucl_emp: float

    def alarms(self, empirical: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Boolean T^2 and SPE alarm arrays under the chosen limits."""
        t2_lim, spe_lim = (self.t2_ucl_emp, self.spe_ucl_emp) if empirical else (self.t2_ucl, self.spe_ucl)
        return self.t2 > t2_lim, self.spe > spe_lim


def t2_ucl_phase2(p: int, m: int, alpha: float) -> float:
    """UCL for future (Phase II) individual observations."""
    return p * (m + 1) * (m - 1) / (m * (m - p)) * stats.f.ppf(1 - alpha, p, m - p)


def spe_ucl_box(spe_ref: np.ndarray, alpha: float) -> float:
    """Box (1954) scaled chi-square approximation: g * chi2_h."""
    mu, var = float(np.mean(spe_ref)), float(np.var(spe_ref, ddof=1))
    g = var / (2 * mu)
    h = 2 * mu * mu / var
    return g * stats.chi2.ppf(1 - alpha, h)


def _fit_model(Zref: np.ndarray, var_explained: float) -> PCA:
    full = PCA(svd_solver="full").fit(Zref)
    a = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), var_explained) + 1)
    pca = PCA(n_components=a, svd_solver="full").fit(Zref)
    pca.full_ratio_ = full.explained_variance_ratio_
    return pca


def _standardise(scaler: StandardScaler, A: np.ndarray, clip_z: float | None) -> np.ndarray:
    Z = scaler.transform(A)
    Z[:, scaler.scale_ == 0] = 0.0
    return np.clip(Z, -clip_z, clip_z) if clip_z is not None else Z


def _t2_spe(pca: PCA, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = pca.transform(Z)
    E = Z - pca.inverse_transform(T)
    return T, np.sum(T**2 / pca.explained_variance_, axis=1), E, np.sum(E**2, axis=1)


def fit_mspc(
    X: pd.DataFrame,
    ref_mask: np.ndarray,
    var_explained: float = 0.8,
    alpha: float = 0.01,
    clip_z: float | None = 10.0,
    cv_folds: int | None = 5,
    seed: int = 0,
) -> MSPCResult:
    """Fit scaler and PCA on the reference runs and score every run.

    ``clip_z`` winsorises standardised values so that a single extreme
    reading does not dominate the fit.

    With ``cv_folds`` set, the reference runs are scored out-of-fold: each
    is scored by a model fitted without it. Scoring reference runs with the
    model fitted on them understates their T^2 and SPE, which biases both
    the empirical limits and any comparison between reference runs and
    other runs (for example, failed runs that were excluded from the
    reference). Non-reference runs are always scored by the full model.
    """
    ref_idx = np.flatnonzero(ref_mask)
    scaler = StandardScaler().fit(X.values[ref_mask])
    Z = _standardise(scaler, X.values, clip_z)
    pca = _fit_model(Z[ref_mask], var_explained)
    a = pca.n_components_
    T, t2, E, spe = _t2_spe(pca, Z)

    if cv_folds:
        A = X.values[ref_mask]
        for tr, te in KFold(cv_folds, shuffle=True, random_state=seed).split(A):
            sc_k = StandardScaler().fit(A[tr])
            Ztr = _standardise(sc_k, A[tr], clip_z)
            pca_k = PCA(n_components=a, svd_solver="full").fit(Ztr)
            _, t2_k, _, spe_k = _t2_spe(pca_k, _standardise(sc_k, A[te], clip_z))
            t2[ref_idx[te]], spe[ref_idx[te]] = t2_k, spe_k

    m = int(ref_mask.sum())
    P = pca.components_.T  # (n_vars, a)
    # T^2 contribution of variable j: sum_a (t_a / lam_a) * p_ja * z_j
    t2c = (T / pca.explained_variance_) @ P.T * (Z - pca.mean_)
    cols = list(X.columns)
    k = min(a, 5)
    return MSPCResult(
        scores=pd.DataFrame(T[:, :k], columns=[f"PC{i + 1}" for i in range(k)], index=X.index),
        t2=t2,
        spe=spe,
        t2_ucl=float(t2_ucl_phase2(a, m, alpha)),
        spe_ucl=float(spe_ucl_box(spe[ref_mask], alpha)),
        n_components=a,
        explained=pca.full_ratio_,
        loadings=pd.DataFrame(P, index=cols, columns=[f"PC{i + 1}" for i in range(a)]),
        t2_contrib=pd.DataFrame(t2c, columns=cols, index=X.index),
        spe_contrib=pd.DataFrame(E**2, columns=cols, index=X.index),
        # Distribution-free limits: the (1 - alpha) quantile of the
        # (out-of-fold) reference statistics. Preferable when sensors are
        # heavy tailed, because the F and chi-square limits assume
        # multivariate normality.
        t2_ucl_emp=float(np.quantile(t2[ref_mask], 1 - alpha)),
        spe_ucl_emp=float(np.quantile(spe[ref_mask], 1 - alpha)),
    )


def top_contributors(contrib: pd.DataFrame, run_idx, k: int = 5) -> list[tuple[str, float]]:
    row = contrib.loc[run_idx]
    share = row.abs() / row.abs().sum()
    top = share.sort_values(ascending=False).head(k)
    return list(zip(top.index, top.values))
