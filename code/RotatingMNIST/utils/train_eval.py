import torch
from tqdm import tqdm


def evaluate_mse(model, loader, device, input_frames=10):
    model.eval()
    total_error = 0.0
    total_elements = 0
    with torch.inference_mode():
        for seq, _ in tqdm(loader, desc="Evaluation", leave=False):
            seq = seq.to(device, non_blocking=True)
            inp, target = seq[:, :input_frames], seq[:, input_frames:]
            prediction = model(inp, pred_len=target.size(1), teacher_forcing_ratio=0.0)
            total_error += (prediction - target).square().sum().item()
            total_elements += target.numel()
    if not total_elements:
        raise ValueError("Evaluation dataset is empty")
    return total_error / total_elements


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    input_frames=10,
    teacher_forcing_ratio=0.0,
    val_loader=None,
    batches_per_eval=160,
):
    model.train()
    total_error = 0.0
    total_elements = 0
    for batch_index, (seq, _) in enumerate(tqdm(loader, desc="Training", leave=False), 1):
        seq = seq.to(device, non_blocking=True)
        inp, target = seq[:, :input_frames], seq[:, input_frames:]
        optimizer.zero_grad()
        prediction = model(
            inp,
            pred_len=target.size(1),
            target_seq=target,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        loss = (prediction - target).square().mean()
        loss.backward()
        optimizer.step()
        total_error += loss.item() * target.numel()
        total_elements += target.numel()
        if val_loader is not None and batches_per_eval and batch_index % batches_per_eval == 0:
            val_mse = evaluate_mse(model, val_loader, device, input_frames)
            print(f"Batch {batch_index}: val MSE {val_mse:.7f}")
            model.train()
    if not total_elements:
        raise ValueError("Training dataset is empty")
    return total_error / total_elements
