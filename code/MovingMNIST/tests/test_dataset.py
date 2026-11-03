import torch
from torch.utils.data import DataLoader

from utils.dataset import FixedVelocityMovingMNIST, MovingMNISTDataset


def test_fixed_evaluation_is_stable_per_index(fake_mnist):
    ds = MovingMNISTDataset("data", train=False, random=False, seed=17, seq_len=4)
    first, labels = ds[3]
    _ = ds[1]
    repeated, repeated_labels = ds[3]
    assert torch.equal(first, repeated)
    assert torch.equal(labels, repeated_labels)
    assert first.shape == (4, 1, 28, 28)
    assert first.min() >= 0 and first.max() <= 1


def test_fixed_evaluation_is_worker_invariant(fake_mnist):
    ds = MovingMNISTDataset("data", train=False, random=False, seed=17, seq_len=3)
    a = [x for x, _ in DataLoader(ds, batch_size=2, num_workers=0)]
    b = [x for x, _ in DataLoader(ds, batch_size=2, num_workers=2)]
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_fixed_velocity_is_roll(fake_mnist):
    ds = FixedVelocityMovingMNIST(vx=1, vy=-2, root="data", train=False,
                                   num_digits=1, seq_len=3, random=False)
    seq, _ = ds[0]
    assert torch.equal(seq[1], torch.roll(seq[0], shifts=(-2, 1), dims=(1, 2)))
    assert torch.equal(seq[2], torch.roll(seq[1], shifts=(-2, 1), dims=(1, 2)))


def test_disjoint_digit_pools(fake_mnist, tmp_path):
    from train import make_dataloaders, parse_args
    args = parse_args(["--root", str(tmp_path), "--num-workers", "0"])
    train_loader, val_loader, test_loader = make_dataloaders(args, torch.device("cpu"))
    train_indices = set(train_loader.dataset.digit_indices.tolist())
    val_indices = set(val_loader.dataset.digit_indices.tolist())
    assert len(train_indices) == 45
    assert len(val_indices) == 5
    assert not train_indices.intersection(val_indices)
    assert len(test_loader.dataset) == 12
