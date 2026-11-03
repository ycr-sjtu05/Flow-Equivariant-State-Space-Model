from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch


def _squeeze_channel_if_needed(x: torch.Tensor) -> torch.Tensor:


    if x.ndim == 5:
        if x.shape[2] != 1:
            raise ValueError(f"Expected channel dimension 1, got shape={tuple(x.shape)}")
        x = x[:, :, 0]
    if x.ndim != 4:
        raise ValueError(f"Expected 4D or 5D tensor, got shape={tuple(x.shape)}")
    return x


def thresholds_raw_to_eval_space(
    thresholds_raw: Iterable[float],
    normalize: str,
) -> list[float]:


    out: list[float] = []
    for thr in thresholds_raw:
        thr = float(thr)
        if normalize == "none":
            out.append(thr)
        elif normalize == "zero_one_255":
            out.append(thr / 255.0)
        else:
            raise ValueError(f"Unsupported normalize={normalize}")
    return out


def _safe_div(numer: torch.Tensor, denom: torch.Tensor, eps: float) -> torch.Tensor:
    return numer / (denom + eps)


@dataclass
class _RunningCounts:
    hits: torch.Tensor
    miss: torch.Tensor
    fa: torch.Tensor


class SEVIRMetricAccumulator:


    def __init__(
        self,
        thresholds_raw: Iterable[float] = (74.0, 133.0),
        normalize: str = "zero_one_255",
        clamp_pred_for_event: bool = True,
        eps: float = 1e-8,
    ) -> None:
        self.thresholds_raw = [float(x) for x in thresholds_raw]
        self.thresholds_eval = thresholds_raw_to_eval_space(self.thresholds_raw, normalize=normalize)
        self.normalize = normalize
        self.clamp_pred_for_event = bool(clamp_pred_for_event)
        self.eps = float(eps)

        self._mse_sse_norm: torch.Tensor | None = None
        self._mse_count: torch.Tensor | None = None
        self._event_counts: dict[str, _RunningCounts] = {}
        self.num_batches = 0
        self.num_samples = 0

    def _ensure_storage(self, T: int, device: torch.device) -> None:
        if self._mse_sse_norm is None:
            self._mse_sse_norm = torch.zeros(T, dtype=torch.float64, device=device)
            self._mse_count = torch.zeros(T, dtype=torch.float64, device=device)
            for thr_raw in self.thresholds_raw:
                key = str(int(thr_raw))
                self._event_counts[key] = _RunningCounts(
                    hits=torch.zeros(T, dtype=torch.float64, device=device),
                    miss=torch.zeros(T, dtype=torch.float64, device=device),
                    fa=torch.zeros(T, dtype=torch.float64, device=device),
                )

    @torch.no_grad()
    def update(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> None:
        y_hat = _squeeze_channel_if_needed(y_hat).detach()
        y_true = _squeeze_channel_if_needed(y_true).detach()

        if y_hat.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_hat={tuple(y_hat.shape)} y_true={tuple(y_true.shape)}")

        B, T, H, W = y_hat.shape
        self._ensure_storage(T=T, device=y_hat.device)


        diff = y_hat - y_true
        sse_norm = diff.pow(2).sum(dim=(0, 2, 3), dtype=torch.float64)
        count = torch.full((T,), float(B * H * W), dtype=torch.float64, device=y_hat.device)

        self._mse_sse_norm += sse_norm
        self._mse_count += count


        y_hat_event = y_hat.clamp(0.0, 1.0) if self.clamp_pred_for_event else y_hat
        y_true_event = y_true

        for thr_raw, thr_eval in zip(self.thresholds_raw, self.thresholds_eval):
            key = str(int(thr_raw))

            pred_event = (y_hat_event >= thr_eval)
            true_event = (y_true_event >= thr_eval)

            hits = (pred_event & true_event).sum(dim=(0, 2, 3), dtype=torch.float64)
            miss = ((~pred_event) & true_event).sum(dim=(0, 2, 3), dtype=torch.float64)
            fa = (pred_event & (~true_event)).sum(dim=(0, 2, 3), dtype=torch.float64)

            self._event_counts[key].hits += hits
            self._event_counts[key].miss += miss
            self._event_counts[key].fa += fa

        self.num_batches += 1
        self.num_samples += B

    @torch.no_grad()
    def compute(self) -> Dict[str, torch.Tensor]:
        if self._mse_sse_norm is None or self._mse_count is None:
            raise RuntimeError("No data has been accumulated. Call update() first.")

        out: Dict[str, torch.Tensor] = {}

        mse_norm_per_t = self._mse_sse_norm / self._mse_count
        out["mse_norm_per_t"] = mse_norm_per_t
        out["mse_norm_avg"] = mse_norm_per_t.mean()

        if self.normalize == "zero_one_255":
            mse_raw_per_t = mse_norm_per_t * (255.0 ** 2)
        else:
            mse_raw_per_t = mse_norm_per_t.clone()

        out["mse_raw_per_t"] = mse_raw_per_t
        out["mse_raw_avg"] = mse_raw_per_t.mean()

        for thr_raw in self.thresholds_raw:
            key = str(int(thr_raw))
            counts = self._event_counts[key]
            hits = counts.hits
            miss = counts.miss
            fa = counts.fa

            pod = _safe_div(hits, hits + miss, self.eps)
            sucr = _safe_div(hits, hits + fa, self.eps)
            csi = _safe_div(hits, hits + miss + fa, self.eps)
            bias = _safe_div(hits + fa, hits + miss, self.eps)

            out[f"hits_{key}_per_t"] = hits
            out[f"miss_{key}_per_t"] = miss
            out[f"fa_{key}_per_t"] = fa

            out[f"pod_{key}_per_t"] = pod
            out[f"pod_{key}_avg"] = pod.mean()

            out[f"sucr_{key}_per_t"] = sucr
            out[f"sucr_{key}_avg"] = sucr.mean()

            out[f"csi_{key}_per_t"] = csi
            out[f"csi_{key}_avg"] = csi.mean()

            out[f"bias_{key}_per_t"] = bias
            out[f"bias_{key}_avg"] = bias.mean()

        out["num_batches"] = torch.tensor(float(self.num_batches), dtype=torch.float64)
        out["num_samples"] = torch.tensor(float(self.num_samples), dtype=torch.float64)
        return out


def metrics_to_python(metrics: Dict[str, torch.Tensor]) -> Dict[str, float | list[float]]:


    out: Dict[str, float | list[float]] = {}
    for k, v in metrics.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue

        if v.ndim == 0:
            out[k] = float(v.item())
        elif v.ndim == 1:
            out[k] = [float(x) for x in v.detach().cpu().tolist()]
        else:
            raise ValueError(f"Unexpected tensor rank for key={k}: shape={tuple(v.shape)}")
    return out
