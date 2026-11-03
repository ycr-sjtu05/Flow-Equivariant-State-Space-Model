"""Training and evaluation utilities for sequence prediction."""

import numpy as np
import torch
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, device, input_frames: int,
                teacher_forcing_ratio: float = 0.0, grad_clip: float = 1.0) -> float:
    model.train()
    error_sum, sample_count = 0.0, 0
    for sequence, _ in tqdm(dataloader, desc="Train", leave=False):
        sequence = sequence.to(device, non_blocking=True)
        inputs, targets = sequence[:, :input_frames], sequence[:, input_frames:]
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs, pred_len=targets.size(1),
                            teacher_forcing_ratio=teacher_forcing_ratio, target_seq=targets)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch_size = sequence.size(0)
        error_sum += float(loss.item()) * batch_size
        sample_count += batch_size
    if not sample_count:
        raise ValueError("Empty training dataloader.")
    return error_sum / sample_count


@torch.no_grad()
def evaluate_mse(model, dataloader, device, input_frames: int) -> float:
    """Average per-frame per-pixel MSE over all sequences and predicted steps."""
    model.eval()
    error_sum, count = 0.0, 0
    for sequence, _ in tqdm(dataloader, desc="Evaluate", leave=False):
        sequence = sequence.to(device, non_blocking=True)
        inputs, targets = sequence[:, :input_frames], sequence[:, input_frames:]
        prediction = model(inputs, pred_len=targets.size(1))
        batch_errors = ((prediction - targets) ** 2).mean(dim=(1, 2, 3, 4))
        error_sum += float(batch_errors.sum().item())
        count += int(batch_errors.numel())
    if not count:
        raise ValueError("Empty evaluation dataloader.")
    return error_sum / count


@torch.no_grad()
def evaluate_per_horizon(model, dataloader, device, input_frames: int):
    """Return (overall_mse, horizon_mean, horizon_std) across fixed examples."""
    model.eval()
    error_sum, squared_sum, count = None, None, 0
    for sequence, _ in tqdm(dataloader, desc="Evaluate", leave=False):
        sequence = sequence.to(device, non_blocking=True)
        inputs, targets = sequence[:, :input_frames], sequence[:, input_frames:]
        prediction = model(inputs, pred_len=targets.size(1))
        per_sequence_step = ((prediction - targets) ** 2).mean(dim=(2, 3, 4))
        this_sum = per_sequence_step.double().sum(dim=0).cpu()
        this_squared = per_sequence_step.double().square().sum(dim=0).cpu()
        error_sum = this_sum if error_sum is None else error_sum + this_sum
        squared_sum = this_squared if squared_sum is None else squared_sum + this_squared
        count += per_sequence_step.size(0)
    if count == 0:
        raise ValueError("Empty evaluation dataloader.")
    mean = (error_sum / count).numpy()
    std = np.sqrt(np.maximum((squared_sum / count).numpy() - mean**2, 0.0))
    return float(mean.mean()), mean, std
