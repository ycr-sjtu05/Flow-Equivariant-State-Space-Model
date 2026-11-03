from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from sfm.datasets.sevir_vil_dataset import SEVIRVILDataset
from sfm.datasets.splits import build_year_splits
from sfm.datasets.vil_index import load_vil_index
from sfm.models.build import build_model
from sfm.training.sevir_metrics import SEVIRMetricAccumulator, metrics_to_python
from sfm.utils.paths import CACHE_ROOT


def parse_task(task):
    sep = ':' if ':' in task else ','
    parts = task.split(sep)
    if len(parts) != 2:
        raise ValueError(f'Invalid task: {task}')
    inp, pred = map(int, parts)
    if inp < 1 or pred < 1:
        raise ValueError(f'Invalid task: {task}')
    return inp, pred


@torch.no_grad()
def evaluate_task(model, args, device, index_path, normalize, input_len, pred_len):
    dataset = SEVIRVILDataset(index_path, input_len, pred_len, normalize=normalize, return_metadata=True)
    try:
        ids = build_year_splits(load_vil_index(index_path))[args.split]
        if args.take > 0:
            ids = ids[:args.take]
        if not ids:
            raise ValueError(f'Empty {args.split} split')
        loader = DataLoader(Subset(dataset, ids), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        model.eval()
        acc = SEVIRMetricAccumulator(
            thresholds_raw=args.thresholds, normalize=normalize, clamp_pred_for_event=True, eps=1e-8
        )
        skipped = 0
        t0 = time.perf_counter()
        for step, batch in enumerate(loader, start=1):
            x = batch['x'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=bool(args.amp and device.type == 'cuda')):
                y_hat, _ = model(x, pred_steps=pred_len, return_probe=False)
            if not torch.isfinite(y_hat).all():
                skipped += 1
                continue
            acc.update(y_hat=y_hat, y_true=y)
            if args.progress_every > 0 and step % args.progress_every == 0:
                print(f'{input_len}->{pred_len} {step}/{len(loader)} {time.perf_counter()-t0:.0f}s')
        metrics = metrics_to_python(acc.compute())
        metrics['skipped_nonfinite_pred'] = skipped
        print(f'{args.split} {input_len}->{pred_len}: ', end='')
        print(' '.join(f'CSI{int(thr)}={metrics[f"csi_{int(thr)}_avg"]:.4f}' for thr in args.thresholds),
              f'MSE={metrics["mse_raw_avg"]:.4f}')
        return {
            'input_len': input_len, 'pred_len': pred_len, 'split': args.split,
            'take': args.take, 'batch_size': args.batch_size,
            'thresholds_raw': args.thresholds, 'normalize': normalize, 'metrics': metrics,
        }
    finally:
        dataset.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model', choices=['mamba', 'fernn', 'flow_ssm', 'hier_flow_ssm'], default=None)
    p.add_argument('--index_path', default=None)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    p.add_argument('--tasks', nargs='*', default=None)
    p.add_argument('--input_len', type=int, default=None)
    p.add_argument('--pred_len', type=int, default=None)
    p.add_argument('--normalize', default=None)
    p.add_argument('--thresholds', type=float, nargs='+', default=[74.0, 133.0])
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--take', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--progress_every', type=int, default=10)
    p.add_argument('--save_json', default=None)
    args = p.parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_args = dict(ckpt.get('args', {}))
    model_name = args.model or ckpt.get('model_name') or ckpt_args.get('model')
    if model_name is None:
        raise ValueError('--model is required for checkpoints without model metadata')
    model = build_model(model_name, ckpt_args)
    model.load_state_dict(ckpt['model_state'], strict=True)
    device = torch.device(args.device if args.device != 'cpu' and torch.cuda.is_available() else 'cpu')
    model.to(device)
    normalize = args.normalize or ckpt_args.get('normalize', 'zero_one_255')
    index_path = args.index_path or ckpt_args.get('index_path') or str(CACHE_ROOT / 'vil_index.json')
    tasks = ([parse_task(t) for t in args.tasks]
             if args.tasks else [(args.input_len or ckpt_args.get('input_len', 8),
                                   args.pred_len or ckpt_args.get('pred_len', 8))])
    results = {'model': model_name, 'checkpoint': args.checkpoint, 'split': args.split,
               'normalize': normalize, 'thresholds_raw': args.thresholds, 'tasks': {}}
    for inp, pred in tasks:
        results['tasks'][f'{inp}to{pred}'] = evaluate_task(
            model, args, device, index_path, normalize, inp, pred
        )
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
