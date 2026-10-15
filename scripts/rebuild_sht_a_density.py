"""Rebuild adaptive ShanghaiTech-A density targets and verify every transform."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rebuild_sht_b_density import main


if __name__ == "__main__":
    main(dataset="sht_a")
