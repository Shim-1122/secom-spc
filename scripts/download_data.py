"""Download the UCI SECOM files into data/raw/ (skips files already present)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from secom_spc.data import download

if __name__ == "__main__":
    download(overwrite="--force" in sys.argv)
    print("done")
