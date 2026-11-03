import numpy as np
import pytest
from PIL import Image


class DummyMNIST:
    def __init__(self, root, train=True, download=True):
        self.length = 30 if train else 10

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        arr = np.zeros((28, 28), dtype=np.uint8)
        arr[5:23, 6:21] = (index * 31 + 17) % 255
        arr[8:17, 14:19] = 255
        return Image.fromarray(arr), index % 10


@pytest.fixture
def mock_mnist(monkeypatch):
    import utils.dataset as dataset
    monkeypatch.setattr(dataset, "MNIST", DummyMNIST)
    return DummyMNIST
