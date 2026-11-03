from __future__ import annotations

from pathlib import Path
from typing import Any


def _year_from_path(path: str) -> int:
    p = Path(path)
    for part in p.parts:
        if part.isdigit() and len(part) == 4:
            return int(part)
    raise ValueError(f"Could not infer year from path: {path}")


def build_year_splits(
    index: dict[str, Any],
    train_years: tuple[int, ...] = (2017, 2018),
    val_years: tuple[int, ...] = (2019,),
    test_years: tuple[int, ...] = (),
    val_ratio_within_year: float = 0.5,
) -> dict[str, list[int]]:


    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for fm in index["files"]:
        year = _year_from_path(fm["path"])
        start = int(fm["global_start"])
        end = int(fm["global_end"])
        idxs = list(range(start, end))

        if year in train_years:
            train_indices.extend(idxs)
        elif year in val_years:
            cut = int(len(idxs) * val_ratio_within_year)
            val_indices.extend(idxs[:cut])
            test_indices.extend(idxs[cut:])
        elif year in test_years:
            test_indices.extend(idxs)

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def summarize_splits(splits: dict[str, list[int]]) -> dict[str, int]:
    return {k: len(v) for k, v in splits.items()}
