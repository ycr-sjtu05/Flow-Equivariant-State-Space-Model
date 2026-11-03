"""Offline test data: no network access or MNIST download required."""

import numpy as np
import pytest
import torch
from PIL import Image


class FakeMNIST:
    def __init__(self, root, train=True, download=False):
        del root, download
        self.train = train
        self.length = 50 if train else 12

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = np.zeros((28, 28), dtype=np.uint8)
        row = (index * 3) % 24
        col = (index * 7) % 24
        img[row:row + 4, col:col + 4] = 255
        return Image.fromarray(img, mode="L"), index % 10


@pytest.fixture(autouse=True)
def avoid_cpu_thread_overhead():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def fake_mnist(monkeypatch):
    import utils.dataset
    monkeypatch.setattr(utils.dataset, "MNIST", FakeMNIST)
    return FakeMNIST
