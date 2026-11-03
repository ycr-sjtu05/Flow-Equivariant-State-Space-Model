from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from sfm.utils.paths import get_vil_h5_files, ensure_dir


@dataclass
class FileLayout:
    path: str
    video_key: str
    shape: tuple[int, ...]
    dtype: str
    sample_axis: int
    time_axis: int
    height_axis: int
    width_axis: int
    num_samples: int
    num_frames: int
    height: int
    width: int
    global_start: int
    global_end: int


def _is_numeric_dataset(ds: h5py.Dataset) -> bool:
    return np.issubdtype(ds.dtype, np.number)


def _collect_all_datasets(f: h5py.File) -> list[tuple[str, h5py.Dataset]]:
    out: list[tuple[str, h5py.Dataset]] = []

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            out.append((name, obj))

    f.visititems(visitor)
    return out


def _collect_1d_lengths(f: h5py.File) -> set[int]:
    lengths: set[int] = set()
    for _, ds in _collect_all_datasets(f):
        if ds.ndim == 1:
            lengths.add(int(ds.shape[0]))
    return lengths


def _infer_sample_axis(shape: tuple[int, ...], one_d_lengths: set[int]) -> int:
    candidates = [ax for ax, dim in enumerate(shape) if dim in one_d_lengths]
    if 0 in candidates:
        return 0
    if candidates:
        return candidates[0]
    return 0


def _infer_thw_axes(shape: tuple[int, ...], sample_axis: int) -> tuple[int, int, int]:


    remaining = [(ax, shape[ax]) for ax in range(len(shape)) if ax != sample_axis]
    remaining_sorted = sorted(remaining, key=lambda x: x[1], reverse=True)


    (h_ax, _), (w_ax, _), (t_ax, _) = remaining_sorted[0], remaining_sorted[1], remaining_sorted[2]


    hw = sorted([h_ax, w_ax])
    h_ax, w_ax = hw[0], hw[1]

    return t_ax, h_ax, w_ax


def _score_video_candidate(
    name: str,
    ds: h5py.Dataset,
    one_d_lengths: set[int],
) -> tuple[int, dict[str, Any]] | None:
    if ds.ndim != 4:
        return None
    if not _is_numeric_dataset(ds):
        return None

    shape = tuple(int(x) for x in ds.shape)
    sample_axis = _infer_sample_axis(shape, one_d_lengths)
    time_axis, height_axis, width_axis = _infer_thw_axes(shape, sample_axis)

    num_samples = shape[sample_axis]
    num_frames = shape[time_axis]
    height = shape[height_axis]
    width = shape[width_axis]

    score = 0


    lower_name = name.lower()
    if "vil" in lower_name:
        score += 100


    if height >= 64 and width >= 64:
        score += 20
    if 4 <= num_frames <= 128:
        score += 20
    if num_samples in one_d_lengths:
        score += 10


    score += min(height, 512) // 16
    score += min(width, 512) // 16

    info = {
        "video_key": name,
        "shape": shape,
        "dtype": str(ds.dtype),
        "sample_axis": sample_axis,
        "time_axis": time_axis,
        "height_axis": height_axis,
        "width_axis": width_axis,
        "num_samples": num_samples,
        "num_frames": num_frames,
        "height": height,
        "width": width,
    }
    return score, info


def infer_file_layout(h5_path: str | Path) -> dict[str, Any]:
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as f:
        one_d_lengths = _collect_1d_lengths(f)
        candidates: list[tuple[int, dict[str, Any]]] = []

        for name, ds in _collect_all_datasets(f):
            result = _score_video_candidate(name, ds, one_d_lengths)
            if result is not None:
                candidates.append(result)

        if not candidates:
            raise RuntimeError(f"Could not infer primary video dataset from {h5_path}")

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best = candidates[0]

        out = {
            "path": str(h5_path.resolve()),
            "score": best_score,
            **best,
        }
        return out


def build_vil_index(
    h5_files: list[str | Path] | None = None,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    if h5_files is None:
        h5_files = get_vil_h5_files()

    h5_paths = [Path(p) for p in h5_files]
    h5_paths = sorted(h5_paths)

    files_meta: list[dict[str, Any]] = []
    global_start = 0

    for p in h5_paths:
        meta = infer_file_layout(p)
        num_samples = int(meta["num_samples"])

        layout = FileLayout(
            path=meta["path"],
            video_key=meta["video_key"],
            shape=tuple(meta["shape"]),
            dtype=meta["dtype"],
            sample_axis=int(meta["sample_axis"]),
            time_axis=int(meta["time_axis"]),
            height_axis=int(meta["height_axis"]),
            width_axis=int(meta["width_axis"]),
            num_samples=num_samples,
            num_frames=int(meta["num_frames"]),
            height=int(meta["height"]),
            width=int(meta["width"]),
            global_start=global_start,
            global_end=global_start + num_samples,
        )
        files_meta.append(asdict(layout))
        global_start += num_samples

    index = {
        "version": 1,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "num_files": len(files_meta),
        "total_samples": global_start,
        "files": files_meta,
    }

    if save_path is not None:
        save_path = Path(save_path)
        ensure_dir(save_path.parent)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    return index


def load_vil_index(index_path: str | Path) -> dict[str, Any]:
    index_path = Path(index_path)
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_global_index(index: dict[str, Any], global_idx: int) -> tuple[dict[str, Any], int]:
    total = int(index["total_samples"])
    if not (0 <= global_idx < total):
        raise IndexError(f"global_idx={global_idx} out of range [0, {total})")

    for file_meta in index["files"]:
        start = int(file_meta["global_start"])
        end = int(file_meta["global_end"])
        if start <= global_idx < end:
            local_idx = global_idx - start
            return file_meta, local_idx

    raise RuntimeError(f"Failed to resolve global_idx={global_idx}")
