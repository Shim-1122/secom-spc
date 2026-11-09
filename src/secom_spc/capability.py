"""Process capability (Cp, Cpk, Pp, Ppk and a non-normal Cpk).

Important caveat
----------------
SECOM sensors are anonymised and ship with **no specification limits**.
Real capability needs engineering specs. Here proxy limits are derived
from the Phase I baseline:

    LSL, USL = baseline median -/+ spec_k * baseline robust sigma

With ``spec_k = 4`` a sensor that keeps behaving like its baseline has
Cpk ~ 1.33, the common "capable" threshold. The useful signal is
therefore the *change* in Cpk between Phase I and Phase II, i.e. which
sensors drifted or widened relative to their own reference. This is a
"capability drift" KPI, not a claim about true product capability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .preprocess import robust_center_scale
from .spc import D2


def _sigma_within(x: np.ndarray) -> float:
    mr = np.abs(np.diff(x))
    return float(np.mean(mr) / D2[2]) if mr.size else np.nan


def capability_indices(x: np.ndarray, lsl: float, usl: float) -> dict:
    """Short-term (within) and long-term (overall) capability.

    * Cp, Cpk use sigma_within = MR-bar / d2 (consecutive runs).
    * Pp, Ppk use the overall sample standard deviation.
    * Cpk_pct is the Clements percentile method: it replaces mu +/- 3
      sigma with the 0.135th / 99.865th percentiles and the median, so
      it does not assume normality.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out = {k: np.nan for k in ("n", "mean", "sigma_within", "sigma_overall", "Cp", "Cpk", "Pp", "Ppk", "Cpk_pct", "ppm_out")}
    if x.size < 10:
        return out
    mu = float(np.mean(x))
    sw = _sigma_within(x)
    so = float(np.std(x, ddof=1))
    tol = usl - lsl
    out.update(n=int(x.size), mean=mu, sigma_within=sw, sigma_overall=so)
    if sw > 0:
        out["Cp"] = tol / (6 * sw)
        out["Cpk"] = min(usl - mu, mu - lsl) / (3 * sw)
    if so > 0:
        out["Pp"] = tol / (6 * so)
        out["Ppk"] = min(usl - mu, mu - lsl) / (3 * so)
    lo, med, hi = np.percentile(x, [0.135, 50, 99.865])
    if hi > med > lo:
        out["Cpk_pct"] = min((usl - med) / (hi - med), (med - lsl) / (med - lo))
    out["ppm_out"] = 1e6 * float(np.mean((x < lsl) | (x > usl)))
    return out


def capability_table(
    X: pd.DataFrame, phase1: np.ndarray, spec_k: float = 4.0, sensors: list[str] | None = None
) -> pd.DataFrame:
    """Capability per sensor in Phase I and Phase II against proxy specs."""
    rows = []
    for c in sensors or X.columns:
        x = X[c].values.astype(float)
        base = x[phase1]
        center, rsig = robust_center_scale(base)
        if not np.isfinite(rsig) or rsig <= 0:
            continue
        lsl, usl = center - spec_k * rsig, center + spec_k * rsig
        c1 = capability_indices(base, lsl, usl)
        c2 = capability_indices(x[~phase1], lsl, usl)
        xf = x[np.isfinite(x)]
        sample = xf[: min(xf.size, 5000)]
        nt_p = float(stats.normaltest(sample).pvalue) if sample.size > 20 and np.ptp(sample) > 0 else np.nan
        rows.append(
            {
                "sensor": c,
                "LSL_proxy": lsl,
                "USL_proxy": usl,
                "Cpk_I": c1["Cpk"],
                "Ppk_I": c1["Ppk"],
                "Cpk_II": c2["Cpk"],
                "Ppk_II": c2["Ppk"],
                "Cpk_pct_II": c2["Cpk_pct"],
                "ppm_out_II": c2["ppm_out"],
                "mean_shift_sigma": (c2["mean"] - c1["mean"]) / c1["sigma_overall"] if c1["sigma_overall"] else np.nan,
                "sd_ratio_II_I": c2["sigma_overall"] / c1["sigma_overall"] if c1["sigma_overall"] else np.nan,
                "normaltest_p": nt_p,  # D'Agostino-Pearson K^2
                "normal_at_5pct": bool(nt_p >= 0.05) if np.isfinite(nt_p) else False,
            }
        )
    df = pd.DataFrame(rows).set_index("sensor")
    df["delta_Ppk"] = df["Ppk_II"] - df["Ppk_I"]
    df["capable_II"] = df["Ppk_II"] >= 1.33
    return df.sort_values("Ppk_II")
