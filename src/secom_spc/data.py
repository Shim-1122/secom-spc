"""Loading the raw UCI SECOM files into tidy pandas objects."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DIR, RAW_FILES, UCI_BASE


@dataclass
class SecomData:
    """Sensor matrix plus run metadata.

    X : (n_runs, 590) float sensor readings, columns ``S000`` .. ``S589``
    meta : run_id, timestamp, label (-1 pass / 1 fail), fail (0/1), date
    """

    X: pd.DataFrame
    meta: pd.DataFrame

    @property
    def y(self) -> pd.Series:
        return self.meta["fail"]

    def __len__(self) -> int:
        return len(self.X)


def download(dest: Path = RAW_DIR, overwrite: bool = False) -> None:
    """Fetch the three raw files from the UCI repository."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        target = dest / name
        if target.exists() and not overwrite:
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(UCI_BASE + name, target)


def load(raw_dir: Path = RAW_DIR) -> SecomData:
    """Load SECOM, parse timestamps and attach a stable run id.

    The labels file encodes pass as -1 and fail as 1, followed by a
    quoted ``dd/mm/yyyy HH:MM:SS`` timestamp.
    """
    data_path = raw_dir / "secom.data"
    label_path = raw_dir / "secom_labels.data"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run `python scripts/download_data.py` first."
        )

    X = pd.read_csv(data_path, sep=r"\s+", header=None, dtype=float)
    X.columns = [f"S{i:03d}" for i in range(X.shape[1])]

    labels = pd.read_csv(
        label_path, sep=r"\s+", header=None, names=["label", "timestamp"], quotechar='"'
    )
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S")

    if len(labels) != len(X):
        raise ValueError("secom.data and secom_labels.data have different lengths")

    meta = pd.DataFrame(
        {
            "run_id": np.arange(1, len(X) + 1),
            "timestamp": labels["timestamp"],
            "label": labels["label"].astype(int),
        }
    )
    meta["fail"] = (meta["label"] == 1).astype(int)
    meta["date"] = meta["timestamp"].dt.normalize()

    # The file is already chronological; enforce it so every downstream
    # time-ordered split is valid even if the source changes.
    order = np.argsort(meta["timestamp"].values, kind="stable")
    X = X.iloc[order].reset_index(drop=True)
    meta = meta.iloc[order].reset_index(drop=True)
    return SecomData(X=X, meta=meta)


def phase_split(n: int, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for Phase I (first ``frac`` of runs) and Phase II."""
    cut = int(round(n * frac))
    p1 = np.zeros(n, dtype=bool)
    p1[:cut] = True
    return p1, ~p1
