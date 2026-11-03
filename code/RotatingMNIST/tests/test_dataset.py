import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.dataset import RotatingMNISTDataset


def test_fixed_samples(mock_mnist):
    data = RotatingMNISTDataset(train=False, random=False, seed=123, seq_len=4)
    a, labels_a = data[3]
    _ = data[5]
    b, labels_b = data[3]
    assert a.shape == (4, 1, 34, 34)
    assert torch.equal(a, b)
    assert torch.equal(labels_a, labels_b)
    assert 0 <= a.min() <= a.max() <= 1


def test_dataloader_worker_independence(mock_mnist):
    data = RotatingMNISTDataset(train=False, random=False, seed=12, seq_len=3)
    x0 = torch.cat([seq for seq, _ in DataLoader(data, batch_size=2, num_workers=0)], dim=0)
    x2 = torch.cat([seq for seq, _ in DataLoader(data, batch_size=2, num_workers=2)], dim=0)
    assert torch.equal(x0, x2)


def test_dynamic_sampling(mock_mnist):
    np.random.seed(7)
    data = RotatingMNISTDataset(train=True, random=True, seq_len=3)
    a, _ = data[0]
    b, _ = data[0]
    assert not torch.equal(a, b)


def test_one_digit(mock_mnist):
    data = RotatingMNISTDataset(train=False, random=False, num_digits=1, seq_len=2)
    x, label = data[2]
    assert x.shape == (2, 1, 34, 34)
    assert isinstance(label, int)
