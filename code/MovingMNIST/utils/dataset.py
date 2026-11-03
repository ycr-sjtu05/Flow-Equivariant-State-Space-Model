"""Generate Moving MNIST sequences with periodic integer translations."""

from collections.abc import Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class MovingMNISTDataset(Dataset):
    """Generate synthetic trajectories of MNIST digits on a square torus.

    Args:
        root: MNIST download/cache directory.
        train: Use torchvision MNIST train or test split.
        seq_len: Total frames in a sampled trajectory.
        image_size: Output height/width (>=28; the source digit is 28x28).
        velocity_range_x/y: Inclusive integer velocity bounds in pixels/frame.
        num_digits: Number of independently translating digits.
        random: True for dynamically resampled examples; False for fixed examples.
        seed: Dataset seed used only when ``random=False``.
        digit_indices: Optional MNIST source-index pool, useful for disjoint
            train/validation source sets. The dataset length follows this pool.
    """

    def __init__(self, root: str, train: bool = True, seq_len: int = 20,
                 image_size: int = 28, velocity_range_x=(-2, 2),
                 velocity_range_y=(-2, 2), num_digits: int = 2,
                 transform=None, download: bool = True, random: bool = True,
                 seed: int = 42, digit_indices: Sequence[int] | None = None):
        super().__init__()
        if seq_len <= 0 or image_size < 28 or num_digits <= 0:
            raise ValueError("seq_len and num_digits must be positive; image_size must be >=28.")
        if velocity_range_x[0] > velocity_range_x[1] or velocity_range_y[0] > velocity_range_y[1]:
            raise ValueError("Velocity bounds must be ordered (min, max).")

        self.mnist = MNIST(root=root, train=train, download=download)
        self.seq_len = seq_len
        self.image_size = image_size
        self.vx_min, self.vx_max = velocity_range_x
        self.vy_min, self.vy_max = velocity_range_y
        self.num_digits = num_digits
        self.transform = transform
        self.to_tensor = ToTensor()
        self.random = random
        self.seed = int(seed)
        self.digit_indices = (np.arange(len(self.mnist), dtype=np.int64)
                              if digit_indices is None
                              else np.asarray(digit_indices, dtype=np.int64))
        if not len(self.digit_indices):
            raise ValueError("digit_indices cannot be empty.")
        if self.digit_indices.min() < 0 or self.digit_indices.max() >= len(self.mnist):
            raise ValueError("digit_indices contains an invalid MNIST index.")

    def __len__(self) -> int:
        return len(self.digit_indices)

    def __getitem__(self, index: int):
        if not 0 <= index < len(self):
            raise IndexError(index)
        # Fresh trajectories during training; index-specific trajectories for evaluation.
        rng = np.random if self.random else np.random.RandomState(
            np.random.SeedSequence([self.seed, int(index)]).generate_state(1)[0]
        )
        digit_images, labels = [], []
        for _ in range(self.num_digits):
            mnist_index = int(self.digit_indices[rng.randint(0, len(self.digit_indices))])
            image, label = self.mnist[mnist_index]
            digit_images.append(self._pad_digit(self.to_tensor(image)))
            labels.append(label)

        # Sample (vx, vy, x0, y0) for each digit.
        velocities_x, velocities_y, initial_x, initial_y = [], [], [], []
        for _ in range(self.num_digits):
            velocities_x.append(int(rng.randint(self.vx_min, self.vx_max + 1)))
            velocities_y.append(int(rng.randint(self.vy_min, self.vy_max + 1)))
            initial_x.append(int(rng.randint(0, self.image_size)))
            initial_y.append(int(rng.randint(0, self.image_size)))

        frames = []
        for t in range(self.seq_len):
            frame = torch.zeros(1, self.image_size, self.image_size, dtype=digit_images[0].dtype)
            for i, image in enumerate(digit_images):
                dx = int(initial_x[i] + t * velocities_x[i])
                dy = int(initial_y[i] + t * velocities_y[i])
                frame += torch.roll(image, shifts=(dy, dx), dims=(1, 2))
            frames.append(frame.clamp(0.0, 1.0))
        sequence = torch.stack(frames, dim=0)
        if self.transform is not None:
            sequence = self.transform(sequence)
        labels = labels[0] if self.num_digits == 1 else torch.tensor(labels, dtype=torch.long)
        return sequence, labels

    def _pad_digit(self, image: torch.Tensor) -> torch.Tensor:
        """Center a 28×28 digit in the output canvas."""
        pad = (self.image_size - image.shape[-1]) // 2
        return F.pad(image, (pad, self.image_size - image.shape[-1] - pad,
                             pad, self.image_size - image.shape[-2] - pad))


class FixedVelocityMovingMNIST(MovingMNISTDataset):
    """Generate digits with the same fixed translation velocity (vx, vy)."""

    def __init__(self, vx: int, vy: int, **kwargs):
        super().__init__(velocity_range_x=(vx, vx), velocity_range_y=(vy, vy), **kwargs)
