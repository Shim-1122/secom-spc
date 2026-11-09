"""Central configuration for the SECOM SPC pipeline.

Every tunable number in the analysis lives here so that results are
reproducible and the choices are visible in one place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    # ---- Phase I / Phase II boundary (time ordered, no shuffling) ----
    # "changepoint": Phase I ends at the change point detected in the
    # sensors (see changepoint.py). "fraction": Phase I is the first
    # ``phase1_frac`` of runs.
    phase_boundary: str = "changepoint"
    phase1_frac: float = 0.5
    # Limits are estimated from PASSING Phase I runs only, mimicking a
    # "known good" reference period.
    phase1_pass_only: bool = True

    # ---- Re-baselining and monitoring ----
    rebaseline_runs: int = 250       # passing runs after the change used to refit limits
    rolling_window: int = 150        # passing runs in the daily-refit rolling monitor
    high_fail_min_count: int = 3     # a "high-fail day" has at least this many failures ...
    high_fail_rate_multiple: float = 2.0  # ... and a fail rate of at least this multiple of Phase I

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
    walk_forward_block: int = 100    # runs scored per retraining step
    walk_forward_window: int = 500   # training-window length for the sliding variant

    # Sensors charted individually in the report
    n_charted_sensors: int = 6

    def as_dict(self) -> dict:
        return asdict(self)


CFG = Config()


def ensure_dirs() -> None:
    """Create the output directories if they do not exist."""
    for d in (FIG_DIR, TABLE_DIR, EXPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)
