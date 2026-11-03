import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class RotatingMNISTDataset(Dataset):
    def __init__(
        self,
        root="data",
        train=True,
        seq_len=20,
        image_size=34,
        angular_velocities=(-40, -20, 0, 20, 40),
        num_digits=2,
        transform=None,
        download=True,
        random=True,
        seed=42,
    ):
        super().__init__()
        if seq_len < 1 or image_size < 28 or num_digits < 1 or not angular_velocities:
            raise ValueError("Invalid sequence, image, digit, or velocity configuration")
        self.mnist = MNIST(root=root, train=train, download=download)
        self.seq_len = seq_len
        self.image_size = image_size
        self.angular_velocities = tuple(angular_velocities)
        self.num_digits = num_digits
        self.transform = transform
        self.to_tensor = ToTensor()
        self.random = random
        self.seed = int(seed)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        rng = np.random if self.random else np.random.RandomState(
            np.random.SeedSequence([self.seed, int(index)]).generate_state(1)[0]
        )
        images = []
        labels = []
        for _ in range(self.num_digits):
            digit_index = rng.randint(0, len(self.mnist))
            image, label = self.mnist[int(digit_index)]
            images.append(self.to_tensor(image))
            labels.append(label)

        velocities = []
        initial_angles = []
        for _ in range(self.num_digits):
            velocities.append(rng.choice(self.angular_velocities))
            initial_angles.append(rng.uniform(0, 360))

        sequence = torch.zeros(self.seq_len, 1, self.image_size, self.image_size, dtype=images[0].dtype)
        pad_size = int(np.ceil(28 * np.sqrt(2) / 2) - 14)
        for t in range(self.seq_len):
            frame = torch.zeros(1, self.image_size, self.image_size)
            for i, image in enumerate(images):
                padded = F.pad(image, (pad_size, pad_size, pad_size, pad_size), "constant", 0)
                angle = initial_angles[i] + velocities[i] * t
                c = np.cos(np.radians(angle))
                s = np.sin(np.radians(angle))
                matrix = torch.tensor([[c, -s, 0], [s, c, 0]], dtype=torch.float32)
                grid = F.affine_grid(matrix.unsqueeze(0), padded.unsqueeze(0).shape, align_corners=True)
                rotated = F.grid_sample(
                    padded.unsqueeze(0), grid, align_corners=True, mode="bilinear", padding_mode="zeros"
                ).squeeze(0)
                offset = (self.image_size - rotated.shape[1]) // 2
                if offset > 0:
                    rotated = F.pad(rotated, (offset, offset, offset, offset), "constant", 0)
                if rotated.shape[1] > self.image_size:
                    start = (rotated.shape[1] - self.image_size) // 2
                    rotated = rotated[:, start : start + self.image_size, start : start + self.image_size]
                frame += rotated
            sequence[t] = frame.clamp(0.0, 1.0)

        if self.transform is not None:
            sequence = self.transform(sequence)
        if self.num_digits == 1:
            return sequence, labels[0]
        return sequence, torch.tensor(labels, dtype=torch.long)
