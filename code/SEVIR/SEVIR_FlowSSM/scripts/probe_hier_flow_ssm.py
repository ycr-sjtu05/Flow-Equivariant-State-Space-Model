from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from sfm.datasets.sevir_vil_dataset import SEVIRVILDataset
from sfm.datasets.vil_index import load_vil_index
from sfm.datasets.splits import build_year_splits, summarize_splits
from sfm.utils.paths import CACHE_ROOT
from sfm.utils.seed import set_seed

from sfm.models.build import build_model


def get_state_dict_from_ckpt(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            return ckpt["model_state"]
        if "model" in ckpt:
            return ckpt["model"]
    return ckpt


def fill_default_cfg(args_dict: dict[str, Any]) -> Namespace:
    d = dict(args_dict)

    defaults = {
        "index_path": str(CACHE_ROOT / "vil_index.json"),
        "input_len": 8,
        "pred_len": 8,
        "normalize": "zero_one_255",

        "latent_channels": 64,
        "encoder_mid_channels": 32,
        "guide_channels": 8,
        "decoder_channels": 64,
        "core_feat_dim": 64,
        "no_guides": False,

        "plain_v_axis_size": 3,
        "plain_state_dim0": 64,
        "plain_state_dim1": 80,
        "plain_state_dim2": 96,
        "no_residual_latent": False,
        "residual_scale": 0.5,

        "hier_adapter_channels": 16,
        "hier_gamma_max": 0.10,
        "hier_gamma_init": -2.0,
        "hier_norm_groups": 8,
        "no_zero_init_adapter": False,

        "seed": 42,
        "amp": True,
    }

    for k, v in defaults.items():
        if k not in d or d[k] is None:
            d[k] = v

    return Namespace(**d)


def to_vil_units(x: torch.Tensor, normalize: str) -> torch.Tensor:
    if normalize == "zero_one_255":
        return x.float() * 255.0
    return x.float()


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else 0.0


def cosine_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    num = torch.dot(a, b)
    den = a.norm() * b.norm() + eps
    return float((num / den).detach().cpu())


def pearson_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm() + eps
    return float((torch.dot(a, b) / den).detach().cpu())


def mean_dict_values(acc: dict[str, list[float]]) -> dict[str, float]:
    return {
        k: float(sum(v) / max(len(v), 1))
        for k, v in acc.items()
        if len(v) > 0
    }


def add_metric(acc: dict[str, list[float]], key: str, value: float) -> None:
    if value == value:
        acc.setdefault(key, []).append(float(value))


def flatten_bt(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int, int] | None]:
    if x.dim() == 4:
        return x, None
    if x.dim() == 5:
        b, t, c, h, w = x.shape
        return x.reshape(b * t, c, h, w), (b, t, c, h, w)
    raise ValueError(f"Expected 4D or 5D tensor, got {tuple(x.shape)}")


def unflatten_bt(x: torch.Tensor, info: tuple[int, int, int, int, int] | None) -> torch.Tensor:
    if info is None:
        return x
    b, t, c, h, w = info
    return x.reshape(b, t, c, h, w)


def resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def get_guide_context(model, guides: dict[int, torch.Tensor]) -> dict[int, torch.Tensor] | None:
    if not getattr(model, "use_last_context_guides", True):
        return None
    if len(guides) == 0:
        return None
    return {res: guides[res][:, -1] for res in guides.keys()}


@torch.no_grad()
def rollout_plain_latents(model, x: torch.Tensor, pred_steps: int):
    enc = model.encoder(x)
    latents = enc.latents
    guides = enc.guides

    z_pred_pyramid, _ = model.core.rollout_pyramid(
        x0_ctx=latents[0],
        x1_ctx=latents[1],
        x2_ctx=latents[2],
        pred_steps=pred_steps,
        teacher_forcing_pyramid=None,
        state0=None,
        return_probe=False,
    )

    guide_context = get_guide_context(model, guides)
    return z_pred_pyramid, guide_context


@torch.no_grad()
def encode_target_latents(model, y: torch.Tensor) -> dict[int, torch.Tensor]:
    enc_y = model.encoder(y)
    return enc_y.latents


@torch.no_grad()
def decompose_adapter(model, z_pred_pyramid: dict[int, torch.Tensor]):


    adapter = model.decoder.adapter
    scales = [s for s in adapter.scales if s in z_pred_pyramid]

    flat: dict[int, torch.Tensor] = {}
    info: dict[int, tuple[int, int, int, int, int] | None] = {}

    for s in scales:
        x4, inf = flatten_bt(z_pred_pyramid[s])
        flat[s] = x4
        info[s] = inf

    local_msg: dict[int, torch.Tensor] = {}
    for s in scales:
        local_msg[s] = adapter.blocks[str(s)](flat[s])

    gamma = adapter.gamma

    edge_corr_4d: dict[str, torch.Tensor] = {}
    edge_target: dict[str, int] = {}

    for dst in scales:
        self_gate = torch.sigmoid(adapter.raw_self_gates[str(dst)])
        key = f"self{dst}"
        edge_corr_4d[key] = gamma * self_gate * local_msg[dst]
        edge_target[key] = dst

        for src in scales:
            if src <= dst:
                continue
            edge_name = f"{src}_to_{dst}"
            if edge_name not in adapter.raw_td_gates:
                continue

            td_gate = torch.sigmoid(adapter.raw_td_gates[edge_name])
            msg = resize_like(local_msg[src], flat[dst])
            key = edge_name
            edge_corr_4d[key] = gamma * td_gate * msg
            edge_target[key] = dst

    total_corr_4d: dict[int, torch.Tensor] = {
        s: torch.zeros_like(flat[s]) for s in scales
    }

    for key, corr in edge_corr_4d.items():
        dst = edge_target[key]
        total_corr_4d[dst] = total_corr_4d[dst] + corr

    adapted: dict[int, torch.Tensor] = dict(z_pred_pyramid)
    edge_corr: dict[str, torch.Tensor] = {}
    total_corr: dict[int, torch.Tensor] = {}

    for s in scales:
        adapted[s] = unflatten_bt(flat[s] + total_corr_4d[s], info[s])
        total_corr[s] = unflatten_bt(total_corr_4d[s], info[s])

    for key, corr4 in edge_corr_4d.items():
        dst = edge_target[key]
        edge_corr[key] = unflatten_bt(corr4, info[dst])

    return adapted, edge_corr, edge_target, total_corr


def make_adapted_with_omitted_edges(
    z_pred_pyramid: dict[int, torch.Tensor],
    edge_corr: dict[str, torch.Tensor],
    edge_target: dict[str, int],
    omit_edges: set[str],
) -> dict[int, torch.Tensor]:
    out = {s: z.clone() for s, z in z_pred_pyramid.items()}

    for key, corr in edge_corr.items():
        if key in omit_edges:
            continue
        dst = edge_target[key]
        out[dst] = out[dst] + corr

    return out


def get_omit_edges(mode: str, all_edges: list[str]) -> set[str]:
    if mode == "normal":
        return set()

    if mode == "zero_all":
        return set(all_edges)

    if mode == "zero_self":
        return {e for e in all_edges if e.startswith("self")}

    if mode == "zero_top_down":
        return {e for e in all_edges if "_to_" in e}

    if mode.startswith("zero_"):
        edge = mode[len("zero_"):]
        return {edge}

    raise ValueError(f"Unknown ablation mode: {mode}")


def categorical_counts(pred_vil: torch.Tensor, target_vil: torch.Tensor, thr: float):
    pred_bin = pred_vil >= thr
    target_bin = target_vil >= thr

    hit = torch.logical_and(pred_bin, target_bin).sum().item()
    miss = torch.logical_and(~pred_bin, target_bin).sum().item()
    fa = torch.logical_and(pred_bin, ~target_bin).sum().item()
    cn = torch.logical_and(~pred_bin, ~target_bin).sum().item()

    return float(hit), float(miss), float(fa), float(cn)


def csi_from_counts(hit: float, miss: float, fa: float) -> float:
    return safe_div(hit, hit + miss + fa)


@torch.no_grad()
def run_decoder(
    model,
    z_pyramid: dict[int, torch.Tensor],
    guide_context: dict[int, torch.Tensor] | None,
):
    return model.decoder.inner(
        z_pred_pyramid=z_pyramid,
        guide_context=guide_context,
    )


@torch.no_grad()
def run_decoder_with_acts(
    model,
    z_pyramid: dict[int, torch.Tensor],
    guide_context: dict[int, torch.Tensor] | None,
):
    inner = model.decoder.inner
    module_names = ["dec2", "dec1", "dec0", "refine96", "refine192", "refine384", "head"]

    acts: dict[str, list[torch.Tensor]] = {name: [] for name in module_names}
    handles = []

    def make_hook(name: str):
        def hook(module, inp, out):
            if torch.is_tensor(out):
                acts[name].append(out.detach().float())
        return hook

    for name in module_names:
        if hasattr(inner, name):
            handles.append(getattr(inner, name).register_forward_hook(make_hook(name)))

    y = inner(
        z_pred_pyramid=z_pyramid,
        guide_context=guide_context,
    )

    for h in handles:
        h.remove()

    return y, acts


def activation_delta(base_acts: dict[str, list[torch.Tensor]], new_acts: dict[str, list[torch.Tensor]]):
    out: dict[str, float] = {}

    for name, base_list in base_acts.items():
        new_list = new_acts.get(name, [])
        n = min(len(base_list), len(new_list))
        if n == 0:
            continue

        num = 0.0
        den = 0.0
        for i in range(n):
            b = base_list[i]
            a = new_list[i]
            num += float((b - a).norm().detach().cpu())
            den += float(b.norm().detach().cpu())

        out[name] = safe_div(num, den)

    return out


def spatial_probe_for_edge(
    edge_tensor: torch.Tensor,
    pred_vil: torch.Tensor,
    target_vil: torch.Tensor,
    thresholds: list[float],
):


    b, t, c, h, w = edge_tensor.shape

    energy = edge_tensor.float().pow(2).sum(dim=2).sqrt()

    error = (pred_vil - target_vil).abs()
    error_ds = F.interpolate(
        error.reshape(b * t, 1, error.shape[-2], error.shape[-1]),
        size=(h, w),
        mode="area",
    ).reshape(b, t, h, w)

    result: dict[str, float] = {}

    result["corr_with_error"] = pearson_flat(energy, error_ds)

    q = torch.quantile(error_ds.reshape(-1), 0.90)
    high_err_mask = (error_ds >= q).float()
    result["energy_frac_top10_error"] = safe_div(
        float((energy * high_err_mask).sum().detach().cpu()),
        float(energy.sum().detach().cpu()) + 1e-8,
    )

    for thr in thresholds:
        rain = (target_vil >= thr).float()
        rain_ds = F.interpolate(
            rain.reshape(b * t, 1, rain.shape[-2], rain.shape[-1]),
            size=(h, w),
            mode="area",
        ).reshape(b, t, h, w)

        rain_mask = (rain_ds > 0.05).float()
        result[f"energy_frac_rain_{thr:g}"] = safe_div(
            float((energy * rain_mask).sum().detach().cpu()),
            float(energy.sum().detach().cpu()) + 1e-8,
        )

    return result


@torch.no_grad()
def probe(
    model,
    loader,
    device,
    cfg,
    thresholds: list[float],
    amp: bool,
    clamp_pred: bool,
    probe_decoder_acts: bool,
    print_every: int,
):
    model.eval()

    acc: dict[str, list[float]] = {}


    counts: dict[str, dict[float, dict[str, float]]] = {}

    ablation_modes = [
        "normal",
        "zero_all",
        "zero_self",
        "zero_top_down",
        "zero_1_to_0",
        "zero_2_to_0",
        "zero_2_to_1",
    ]

    for mode in ablation_modes:
        counts[mode] = {
            thr: {"hit": 0.0, "miss": 0.0, "fa": 0.0, "cn": 0.0}
            for thr in thresholds
        }

    for step, batch in enumerate(loader):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(amp and device.type == "cuda")):
            z_pred, guide_context = rollout_plain_latents(model, x, cfg.pred_len)
            z_gt = encode_target_latents(model, y)
            z_adapted, edge_corr, edge_target, total_corr = decompose_adapter(model, z_pred)

            if probe_decoder_acts:
                y_normal, acts_normal = run_decoder_with_acts(model, z_adapted, guide_context)
            else:
                y_normal = run_decoder(model, z_adapted, guide_context)
                acts_normal = None

        if clamp_pred and cfg.normalize == "zero_one_255":
            y_normal = y_normal.clamp(0.0, 1.0)

        pred_vil_normal = to_vil_units(y_normal, cfg.normalize)
        target_vil = to_vil_units(y, cfg.normalize)


        all_edges = sorted(edge_corr.keys())

        for edge in all_edges:
            dst = edge_target[edge]
            corr = edge_corr[edge]
            base = z_pred[dst]
            residual = z_gt[dst] - z_pred[dst]

            norm_ratio = safe_div(
                float(corr.float().norm().detach().cpu()),
                float(base.float().norm().detach().cpu()) + 1e-8,
            )
            add_metric(acc, f"edge_norm_ratio/{edge}", norm_ratio)

            align = cosine_flat(corr, residual)
            add_metric(acc, f"target_alignment/{edge}", align)


            for tt in range(corr.shape[1]):
                align_t = cosine_flat(corr[:, tt], residual[:, tt])
                add_metric(acc, f"target_alignment_h{tt+1}/{edge}", align_t)

        for s, corr in total_corr.items():
            base = z_pred[s]
            residual = z_gt[s] - z_pred[s]

            add_metric(
                acc,
                f"total_corr_norm_ratio/s{s}",
                safe_div(
                    float(corr.float().norm().detach().cpu()),
                    float(base.float().norm().detach().cpu()) + 1e-8,
                ),
            )
            add_metric(acc, f"total_target_alignment/s{s}", cosine_flat(corr, residual))


        for mode in ablation_modes:
            omit_edges = get_omit_edges(mode, all_edges)
            z_mode = make_adapted_with_omitted_edges(
                z_pred_pyramid=z_pred,
                edge_corr=edge_corr,
                edge_target=edge_target,
                omit_edges=omit_edges,
            )

            if mode == "normal":
                y_mode = y_normal
                acts_mode = acts_normal
            else:
                if probe_decoder_acts:
                    y_mode, acts_mode = run_decoder_with_acts(model, z_mode, guide_context)
                else:
                    y_mode = run_decoder(model, z_mode, guide_context)
                    acts_mode = None

            if clamp_pred and cfg.normalize == "zero_one_255":
                y_mode = y_mode.clamp(0.0, 1.0)

            pred_vil = to_vil_units(y_mode, cfg.normalize)

            l1 = float((pred_vil - target_vil).abs().mean().detach().cpu())
            mse = float(((pred_vil - target_vil) ** 2).mean().detach().cpu())
            add_metric(acc, f"ablation_l1/{mode}", l1)
            add_metric(acc, f"ablation_mse/{mode}", mse)

            if mode != "normal":
                out_delta = safe_div(
                    float((y_normal - y_mode).float().norm().detach().cpu()),
                    float(y_normal.float().norm().detach().cpu()) + 1e-8,
                )
                add_metric(acc, f"output_delta_ratio/{mode}", out_delta)

                if probe_decoder_acts and acts_normal is not None and acts_mode is not None:
                    act_delta = activation_delta(acts_normal, acts_mode)
                    for layer_name, val in act_delta.items():
                        add_metric(acc, f"decoder_act_delta/{mode}/{layer_name}", val)

            for thr in thresholds:
                hit, miss, fa, cn = categorical_counts(pred_vil, target_vil, thr)
                counts[mode][thr]["hit"] += hit
                counts[mode][thr]["miss"] += miss
                counts[mode][thr]["fa"] += fa
                counts[mode][thr]["cn"] += cn


        spatial_edges = [e for e in all_edges if e in {"self0", "1_to_0", "2_to_0", "2_to_1"}]
        for edge in spatial_edges:
            result = spatial_probe_for_edge(
                edge_tensor=edge_corr[edge],
                pred_vil=pred_vil_normal,
                target_vil=target_vil,
                thresholds=thresholds,
            )
            for k, v in result.items():
                add_metric(acc, f"spatial/{edge}/{k}", v)


        if 0 in total_corr:
            result = spatial_probe_for_edge(
                edge_tensor=total_corr[0],
                pred_vil=pred_vil_normal,
                target_vil=target_vil,
                thresholds=thresholds,
            )
            for k, v in result.items():
                add_metric(acc, f"spatial/total_s0/{k}", v)

        if print_every > 0 and step % print_every == 0:
            print(f"[probe] step={step}")

    summary = mean_dict_values(acc)


    threshold_summary: dict[str, Any] = {}
    for mode in ablation_modes:
        threshold_summary[mode] = {}
        for thr in thresholds:
            c = counts[mode][thr]
            hit, miss, fa, cn = c["hit"], c["miss"], c["fa"], c["cn"]

            csi = csi_from_counts(hit, miss, fa)
            pod = safe_div(hit, hit + miss)
            far = safe_div(fa, hit + fa)
            bias = safe_div(hit + fa, hit + miss)

            threshold_summary[mode][str(thr)] = {
                "CSI": csi,
                "POD": pod,
                "FAR": far,
                "Bias": bias,
                "hit": hit,
                "miss": miss,
                "false_alarm": fa,
                "correct_negative": cn,
            }

    return {
        "summary": summary,
        "threshold_summary": threshold_summary,
    }


def print_key_results(result: dict[str, Any], thresholds: list[float]) -> None:
    s = result["summary"]
    th = result["threshold_summary"]

    print("=" * 100)
    print("EDGE NORM RATIOS")
    for k in sorted(s.keys()):
        if k.startswith("edge_norm_ratio/"):
            print(f"{k:40s} {s[k]:.6g}")

    print("=" * 100)
    print("TARGET ALIGNMENT")
    for k in sorted(s.keys()):
        if k.startswith("target_alignment/"):
            print(f"{k:40s} {s[k]:.6g}")

    print("=" * 100)
    print("TOTAL CORRECTION")
    for k in sorted(s.keys()):
        if k.startswith("total_corr_norm_ratio/") or k.startswith("total_target_alignment/"):
            print(f"{k:40s} {s[k]:.6g}")

    print("=" * 100)
    print("ABLATION L1 / OUTPUT DELTA")
    modes = ["normal", "zero_all", "zero_self", "zero_top_down", "zero_1_to_0", "zero_2_to_0", "zero_2_to_1"]
    for mode in modes:
        l1 = s.get(f"ablation_l1/{mode}", float("nan"))
        mse = s.get(f"ablation_mse/{mode}", float("nan"))
        out_delta = s.get(f"output_delta_ratio/{mode}", 0.0)
        print(f"{mode:16s} L1={l1:.6f} MSE={mse:.6f} output_delta={out_delta:.6g}")

    print("=" * 100)
    print("THRESHOLD CSI")
    for mode in modes:
        parts = []
        for thr0 in thresholds:
            m = th[mode][str(thr0)]
            parts.append(f"CSI@{thr0:g}={m['CSI']:.6f}")
        print(f"{mode:16s} " + " ".join(parts))

    print("=" * 100)
    print("SPATIAL LOCALIZATION")
    for k in sorted(s.keys()):
        if k.startswith("spatial/"):
            print(f"{k:60s} {s[k]:.6g}")

    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--take", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--thresholds", type=float, nargs="+", default=[74.0, 133.0])
    parser.add_argument("--output_json", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--clamp_pred", action="store_true")
    parser.add_argument("--probe_decoder_acts", action="store_true")
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    cfg = fill_default_cfg(ckpt_args)

    amp = bool(getattr(cfg, "amp", True))
    if args.amp:
        amp = True
    if args.no_amp:
        amp = False

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("[config]")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  split/take: {args.split}/{args.take}")
    print(f"  input_len/pred_len: {cfg.input_len}->{cfg.pred_len}")
    print(f"  normalize: {cfg.normalize}")
    print(f"  thresholds: {args.thresholds}")
    print(f"  device: {device}")
    print(f"  amp: {amp}")
    print(f"  probe_decoder_acts: {args.probe_decoder_acts}")

    dataset = SEVIRVILDataset(
        index_path=cfg.index_path,
        input_len=cfg.input_len,
        pred_len=cfg.pred_len,
        normalize=cfg.normalize,
        return_metadata=True,
    )

    index = load_vil_index(cfg.index_path)
    splits = build_year_splits(index)
    print("split sizes:", summarize_splits(splits))

    ids = splits[args.split]
    if args.take is not None and args.take > 0:
        ids = ids[:args.take]

    loader = DataLoader(
        Subset(dataset, ids),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model("hier_flow_ssm", vars(cfg))
    state = get_state_dict_from_ckpt(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    model = model.to(device)
    model.eval()

    result = probe(
        model=model,
        loader=loader,
        device=device,
        cfg=cfg,
        thresholds=args.thresholds,
        amp=amp,
        clamp_pred=args.clamp_pred,
        probe_decoder_acts=args.probe_decoder_acts,
        print_every=args.print_every,
    )

    output = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "take": args.take,
        "num_samples": len(ids),
        "input_len": cfg.input_len,
        "pred_len": cfg.pred_len,
        "thresholds": args.thresholds,
        "clamp_pred": args.clamp_pred,
        "probe_decoder_acts": args.probe_decoder_acts,
        "result": result,
    }

    print_key_results(result, args.thresholds)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    dataset.close()
    print(f"[save] {output_json}")


if __name__ == "__main__":
    main()
