from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Literal, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


Subset = Literal["train", "val", "train_full", "test"]


@dataclass(frozen=True)
class VideoRecord:
    relative_path: str
    label: int


def read_class_index(
    annotation_root: str | Path,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    annotation_root = Path(annotation_root)
    path = annotation_root / "classInd.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing UCF101 class index: {path}"
        )

    name_to_label: Dict[str, int] = {}
    label_to_name: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw_index, name = line.split(maxsplit=1)
            label = int(raw_index) - 1
            name_to_label[name] = label
            label_to_name[label] = name
    return name_to_label, label_to_name


def _load_official_records(
    annotation_root: str | Path,
    split_id: int,
    train: bool,
) -> List[VideoRecord]:
    if split_id not in {1, 2, 3}:
        raise ValueError("split_id must be 1, 2, or 3.")

    annotation_root = Path(annotation_root)
    name_to_label, _ = read_class_index(annotation_root)

    filename = (
        f"trainlist{split_id:02d}.txt"
        if train
        else f"testlist{split_id:02d}.txt"
    )
    path = annotation_root / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing UCF101 split file: {path}"
        )

    records: List[VideoRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            relative_path = line.split()[0]
            class_name = relative_path.split("/")[0]
            if class_name not in name_to_label:
                raise KeyError(
                    f"Class {class_name} absent from classInd.txt."
                )
            records.append(
                VideoRecord(
                    relative_path=relative_path,
                    label=name_to_label[class_name],
                )
            )
    return records


def stratified_train_val_split(
    records: Sequence[VideoRecord],
    val_fraction: float,
    seed: int,
) -> Tuple[List[VideoRecord], List[VideoRecord]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0,1).")

    by_class: Dict[int, List[VideoRecord]] = {}
    for record in records:
        by_class.setdefault(record.label, []).append(record)

    train_records: List[VideoRecord] = []
    val_records: List[VideoRecord] = []

    for label, class_records in sorted(by_class.items()):
        class_records = sorted(
            class_records,
            key=lambda item: item.relative_path,
        )
        rng = random.Random(seed + label * 1009)
        rng.shuffle(class_records)

        val_count = max(
            1,
            int(round(len(class_records) * val_fraction)),
        )
        val_count = min(val_count, len(class_records) - 1)

        val_records.extend(class_records[:val_count])
        train_records.extend(class_records[val_count:])

    return train_records, val_records


def build_records(
    annotation_root: str | Path,
    subset: Subset,
    split_id: int = 1,
    val_fraction: float = 0.1,
    split_seed: int = 2026,
) -> List[VideoRecord]:
    if subset == "test":
        return _load_official_records(
            annotation_root,
            split_id,
            train=False,
        )

    full_train = _load_official_records(
        annotation_root,
        split_id,
        train=True,
    )
    if subset == "train_full":
        return full_train

    train_records, val_records = stratified_train_val_split(
        full_train,
        val_fraction=val_fraction,
        seed=split_seed,
    )
    if subset == "train":
        return train_records
    if subset == "val":
        return val_records
    raise ValueError(f"Unknown subset: {subset}.")


def _decode_selected_frames(
    video_path: Path,
    clip_len: int,
    temporal_stride: int,
    random_clip: bool,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(
            f"Video reports no frames: {video_path}"
        )

    if total_frames >= clip_len:
        max_feasible_stride = max(
            1,
            (total_frames - 1) // max(clip_len - 1, 1),
        )
        stride = min(
            max(1, temporal_stride),
            max_feasible_stride,
        )
    else:
        stride = 1

    span = stride * (clip_len - 1) + 1
    max_start = max(total_frames - span, 0)

    if random_clip and max_start > 0:
        start = int(np.random.randint(0, max_start + 1))
    else:
        start = max_start // 2

    indices = [
        min(start + i * stride, total_frames - 1)
        for i in range(clip_len)
    ]

    frames = []
    last_frame = None
    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        if not success:
            if last_frame is None:
                cap.release()
                raise RuntimeError(
                    f"Failed to decode frame {frame_index} "
                    f"from {video_path}"
                )
            frame = last_frame.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_frame = frame
        frames.append(frame)

    cap.release()
    return np.stack(frames, axis=0)


def _resize_shorter_side(
    frames: torch.Tensor,
    resize_size: int,
) -> torch.Tensor:
    _, _, height, width = frames.shape
    if height <= width:
        new_height = resize_size
        new_width = int(round(width * resize_size / height))
    else:
        new_width = resize_size
        new_height = int(round(height * resize_size / width))

    return F.interpolate(
        frames,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def _consistent_spatial_transform(
    frames: torch.Tensor,
    resize_size: int,
    crop_size: int,
    train: bool,
    horizontal_flip: bool,
) -> torch.Tensor:
    frames = _resize_shorter_side(frames, resize_size)
    _, _, height, width = frames.shape

    if height < crop_size or width < crop_size:
        raise ValueError(
            "resize_size/crop_size combination is invalid."
        )

    if train:
        top = int(
            np.random.randint(0, height - crop_size + 1)
        )
        left = int(
            np.random.randint(0, width - crop_size + 1)
        )
    else:
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2

    frames = frames[
        :,
        :,
        top:top + crop_size,
        left:left + crop_size,
    ]

    if (
        train
        and horizontal_flip
        and np.random.rand() < 0.5
    ):
        frames = torch.flip(frames, dims=(-1,))

    return frames.contiguous()


class UCF101ClipDataset(Dataset):
    """
    One clip per UCF101 video.

    Training clips use a random temporal start and consistent random crop.
    Validation/test clips use a centered temporal clip and center crop.
    """

    def __init__(
        self,
        video_root: str | Path,
        annotation_root: str | Path,
        subset: Subset,
        *,
        split_id: int = 1,
        clip_len: int = 12,
        temporal_stride: int = 2,
        resize_size: int = 72,
        crop_size: int = 64,
        val_fraction: float = 0.1,
        split_seed: int = 2026,
        horizontal_flip: bool = True,
        max_videos: int | None = None,
    ):
        super().__init__()

        if clip_len <= 1:
            raise ValueError("clip_len must exceed one.")
        if temporal_stride <= 0:
            raise ValueError("temporal_stride must be positive.")
        if resize_size < crop_size:
            raise ValueError(
                "resize_size must be at least crop_size."
            )

        self.video_root = Path(video_root)
        self.annotation_root = Path(annotation_root)
        self.subset = subset
        self.clip_len = clip_len
        self.temporal_stride = temporal_stride
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.horizontal_flip = horizontal_flip
        self.train_mode = subset in {"train", "train_full"}

        self.records = build_records(
            annotation_root=annotation_root,
            subset=subset,
            split_id=split_id,
            val_fraction=val_fraction,
            split_seed=split_seed,
        )
        if max_videos is not None:
            self.records = self.records[:max_videos]

        _, self.label_to_name = read_class_index(
            annotation_root
        )

        if not self.video_root.exists():
            raise FileNotFoundError(
                f"Video root does not exist: {self.video_root}"
            )

    @property
    def num_classes(self) -> int:
        return len(self.label_to_name)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        video_path = self.video_root / record.relative_path
        if not video_path.exists():
            raise FileNotFoundError(
                f"Missing video file: {video_path}"
            )

        frames_np = _decode_selected_frames(
            video_path,
            clip_len=self.clip_len,
            temporal_stride=self.temporal_stride,
            random_clip=self.train_mode,
        )

        frames = torch.from_numpy(frames_np).permute(
            0, 3, 1, 2
        ).float().div_(255.0)

        frames = _consistent_spatial_transform(
            frames,
            resize_size=self.resize_size,
            crop_size=self.crop_size,
            train=self.train_mode,
            horizontal_flip=self.horizontal_flip,
        )

        return {
            "video": frames,
            "label": torch.tensor(
                record.label,
                dtype=torch.long,
            ),
            "path": record.relative_path,
        }


def motion_score(context: torch.Tensor) -> torch.Tensor:
    """
    Cheap clip-level motion proxy from mean grayscale frame differences.

    Args:
        context: (B,T,3,H,W)

    Returns:
        scores: (B,)
    """
    if context.ndim != 5 or context.shape[2] != 3:
        raise ValueError(
            "Expected RGB context with shape (B,T,3,H,W)."
        )

    weights = context.new_tensor(
        [0.299, 0.587, 0.114]
    ).view(1, 1, 3, 1, 1)
    grayscale = (context * weights).sum(dim=2)
    difference = (
        grayscale[:, 1:] - grayscale[:, :-1]
    ).abs()
    return difference.mean(dim=(1, 2, 3))
