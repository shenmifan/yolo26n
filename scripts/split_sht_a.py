"""Freeze ShanghaiTech-A partitions, excluding train content repeated in test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.split_sht_b import exclude_test_duplicates, main as split_main, stratified_split as split_records


def stratified_split(records, seed=42, excluded_count=0):
    return split_records(records, seed=seed, dataset="sht_a", excluded_count=excluded_count)


if __name__ == "__main__":
    split_main(dataset="sht_a")
