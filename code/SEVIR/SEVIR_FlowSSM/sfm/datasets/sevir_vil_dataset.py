from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from sfm.datasets.vil_index import load_vil_index, resolve_global_index


def _reduced_axis_index(original_axis: int, removed_axis: int) -> int:


    if original_axis < removed_axis:
        return original_axis
    if original_axis > removed_axis:
        return original_axis - 1
    raise ValueError("original_axis cannot equal removed_axis here")


class SEVIRVILDataset(Dataset):


    def __init__(
        self,
        index_path: str | Path,
        input_len: int,
        pred_len: int,
        frame_start: int | None = None,
        window_mode: str = "first",
        normalize: str = "none",
        dtype: torch.dtype = torch.float32,
        return_metadata: bool = True,
    ) -> None:
        super().__init__()
        self.index_path = Path(index_path)
        self.index = load_vil_index(self.index_path)
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.total_len = self.input_len + self.pred_len
        self.frame_start = frame_start
        self.window_mode = window_mode
        self.normalize = normalize
        self.dtype = dtype
        self.return_metadata = return_metadata

        self._h5_cache: dict[str, h5py.File] = {}

        self._validate_index()

    def _validate_index(self) -> None:
        if self.window_mode not in {"first", "center"}:
            raise ValueError(f"Unsupported window_mode={self.window_mode}")

        if self.normalize not in {"none", "zero_one_255"}:
            raise ValueError(f"Unsupported normalize={self.normalize}")

        for file_meta in self.index["files"]:
            num_frames = int(file_meta["num_frames"])
            if num_frames < self.total_len:
                raise ValueError(
                    f"File {file_meta['path']} only has {num_frames} frames, "
                    f"but total_len={self.total_len}"
                )

    def __len__(self) -> int:
        return int(self.index["total_samples"])

    def _get_h5(self, path: str) -> h5py.File:
        if path not in self._h5_cache:
            self._h5_cache[path] = h5py.File(path, "r")
        return self._h5_cache[path]

    def _select_window(self, num_frames: int) -> tuple[int, int]:
        if self.frame_start is not None:
            start = int(self.frame_start)
        else:
            if self.window_mode == "first":
                start = 0
            elif self.window_mode == "center":
                start = (num_frames - self.total_len) // 2
            else:
                raise ValueError(f"Unsupported window_mode={self.window_mode}")

        end = start + self.total_len
        if start < 0 or end > num_frames:
            raise ValueError(
                f"Invalid window [{start}, {end}) for num_frames={num_frames}, "
                f"total_len={self.total_len}"
            )
        return start, end

    def _normalize_frames(self, frames: np.ndarray) -> np.ndarray:
        if self.normalize == "none":
            return frames
        if self.normalize == "zero_one_255":
            return frames / 255.0
        raise ValueError(f"Unsupported normalize={self.normalize}")

    def _load_sample_frames(self, file_meta: dict[str, Any], local_idx: int) -> np.ndarray:
        h5_path = str(file_meta["path"])
        video_key = str(file_meta["video_key"])

        sample_axis = int(file_meta["sample_axis"])
        time_axis = int(file_meta["time_axis"])
        height_axis = int(file_meta["height_axis"])
        width_axis = int(file_meta["width_axis"])

        f = self._get_h5(h5_path)
        ds = f[video_key]

        slicer: list[Any] = [slice(None)] * ds.ndim
        slicer[sample_axis] = int(local_idx)
        arr = ds[tuple(slicer)]

        red_time_axis = _reduced_axis_index(time_axis, sample_axis)
        red_height_axis = _reduced_axis_index(height_axis, sample_axis)
        red_width_axis = _reduced_axis_index(width_axis, sample_axis)

        arr = np.asarray(arr)
        arr = np.transpose(arr, (red_time_axis, red_height_axis, red_width_axis))

        num_frames = arr.shape[0]
        start, end = self._select_window(num_frames)
        arr = arr[start:end]

        arr = arr.astype(np.float32, copy=False)
        arr = self._normalize_frames(arr)

        return arr

    def __getitem__(self, global_idx: int) -> dict[str, Any] | tuple[torch.Tensor, torch.Tensor]:
        file_meta, local_idx = resolve_global_index(self.index, int(global_idx))
        frames = self._load_sample_frames(file_meta, local_idx)

        x_np = frames[:self.input_len]
        y_np = frames[self.input_len:self.input_len + self.pred_len]

        x = torch.from_numpy(x_np).unsqueeze(1).to(self.dtype)
        y = torch.from_numpy(y_np).unsqueeze(1).to(self.dtype)

        if not self.return_metadata:
            return x, y

        return {
            "x": x,
            "y": y,
            "global_idx": int(global_idx),
            "local_idx": int(local_idx),
            "file_path": str(file_meta["path"]),
            "video_key": str(file_meta["video_key"]),
            "num_frames_in_file": int(file_meta["num_frames"]),
            "height": int(file_meta["height"]),
            "width": int(file_meta["width"]),
        }

    def close(self) -> None:
        for f in self._h5_cache.values():
            try:
                f.close()
            except Exception:
                pass
        self._h5_cache.clear()

    def __del__(self) -> None:
        self.close()
