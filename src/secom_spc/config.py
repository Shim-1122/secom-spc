"""Central configuration for the SECOM SPC pipeline.

Every tunable number in the analysis lives here so that results are
reproducible and the choices are visible in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
FIG_DIR = REPO_ROOT / "reports" / "figures"
TABLE_DIR = REPO_ROOT / "reports" / "tables"
EXPORT_DIR = REPO_ROOT / "reports" / "exports"

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
RAW_FILES = ("secom.data", "secom_labels.data", "secom.names")


@dataclass(frozen=True)
class Config:
    # ---- Phase I / Phase II split (time ordered, no shuffling) ----
    # Fraction of runs (in time order) used as the Phase I baseline that
    # control limits, scalers, imputers and feature rankings are fit on.
    phase1_frac: float = 0.5
    # Phase I limits are estimated from PASSING runs only, mimicking a
    # "known good" reference period.
    phase1_pass_only: bool = True

    # ---- Preprocessing ----
    max_missing_frac: float = 0.50   # drop sensors missing in > 50 % of Phase I runs
    min_unique: int = 3              # drop sensors with < 3 distinct values in Phase I
    min_rel_std: float = 1e-8        # drop near-constant sensors
    corr_dedup_threshold: float = 0.98  # collapse near-duplicate sensors

    # ---- Univariate SPC ----
    subgroup_size: int = 5           # consecutive runs per X-bar/S subgroup
    sigma_limits: float = 3.0
    ewma_lambda: float = 0.2
    ewma_L: float = 2.86             # ARL0 ~ 370 for lambda = 0.2
    cusum_k: float = 0.5             # in sigma units
    cusum_h: float = 5.0             # in sigma units

    # ---- Capability ----
    # SECOM is anonymised: there are no engineering spec limits. Proxy
    # limits are set at baseline centre +/- spec_k robust sigma, so a
    # perfectly stable sensor has Cpk = spec_k / 3 (1.33 for spec_k = 4).
    spec_k: float = 4.0

    # ---- Multivariate SPC ----
    pca_var_explained: float = 0.80
    mspc_alpha: float = 0.01

    # ---- Feature ranking / modelling ----
    random_state: int = 42
    n_stability_boot: int = 200
    stability_C: float = 0.05
    top_k_sensors: int = 20
    test_frac: float = 0.30          # final 30 % of runs (in time) held out
    cv_folds: int = 5

    # Sensors charted individually in the report
    n_charted_sensors: int = 6

    extra: dict = field(default_factory=dict)


CFG = Config()


def ensure_dirs() -> None:
    for d in (FIG_DIR, TABLE_DIR, EXPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
