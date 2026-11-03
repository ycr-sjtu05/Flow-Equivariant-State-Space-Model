from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from sfm.datasets.sevir_vil_dataset import SEVIRVILDataset
from sfm.datasets.splits import build_year_splits, summarize_splits
from sfm.datasets.vil_index import load_vil_index
from sfm.models.build import build_model, freeze_except_hierarchy, initialize_hierarchy
from sfm.utils.paths import CACHE_ROOT, CHECKPOINT_ROOT, LOG_ROOT, ensure_dir
from sfm.utils.seed import set_seed

MODELS = ('mamba', 'fernn', 'flow_ssm', 'hier_flow_ssm')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--run_name', required=True)
    p.add_argument('--index_path', default=None)
    p.add_argument('--flow_checkpoint', default=None)
    p.add_argument('--resume', default=None)
    p.add_argument('--input_len', type=int, default=None)
    p.add_argument('--pred_len', type=int, default=None)
    p.add_argument('--normalize', default=None)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--train_take', type=int, default=-1)
    p.add_argument('--val_take', type=int, default=None)
    p.add_argument('--latent_channels', type=int, default=None)
    p.add_argument('--encoder_mid_channels', type=int, default=None)
    p.add_argument('--guide_channels', type=int, default=None)
    p.add_argument('--decoder_channels', type=int, default=None)
    p.add_argument('--core_feat_dim', type=int, default=None)
    p.add_argument('--no_guides', action='store_true', default=None)
    p.add_argument('--mamba_d_state', type=int, default=None)
    p.add_argument('--mamba_d_conv', type=int, default=None)
    p.add_argument('--mamba_expand', type=int, default=None)
    p.add_argument('--mamba_num_blocks', type=int, default=None)
    p.add_argument('--fernn_hidden_channels', type=int, default=None)
    p.add_argument('--fernn_v_axis_size', type=int, default=None)
    p.add_argument('--fernn_kernel_size', type=int, default=None)
    p.add_argument('--plain_v_axis_size', type=int, default=None)
    p.add_argument('--plain_state_dim0', type=int, default=None)
    p.add_argument('--plain_state_dim1', type=int, default=None)
    p.add_argument('--plain_state_dim2', type=int, default=None)
    p.add_argument('--no_residual_latent', action='store_true', default=None)
    p.add_argument('--residual_scale', type=float, default=None)
    p.add_argument('--hier_adapter_channels', type=int, default=16)
    p.add_argument('--hier_gamma_max', type=float, default=0.10)
    p.add_argument('--hier_gamma_init', type=float, default=-2.0)
    p.add_argument('--hier_norm_groups', type=int, default=8)
    p.add_argument('--no_zero_init_adapter', action='store_true')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--grad_clip_norm', type=float, default=1.0)
    p.add_argument('--amp', action='store_true', default=None)
    p.add_argument('--no_amp', action='store_false', dest='amp')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def resolve_config(args, base_args):
    cfg = vars(args).copy()
    defaults = {
        'input_len': 8,
        'pred_len': 8,
        'normalize': 'zero_one_255',
        'index_path': str(CACHE_ROOT / 'vil_index.json'),
        'latent_channels': 64,
        'encoder_mid_channels': 32,
        'guide_channels': 8,
        'decoder_channels': 64,
        'core_feat_dim': 64,
        'no_guides': False,
        'mamba_d_state': 16,
        'mamba_d_conv': 4,
        'mamba_expand': 2,
        'mamba_num_blocks': 1,
        'fernn_hidden_channels': None,
        'fernn_v_axis_size': 3,
        'fernn_kernel_size': 3,
        'plain_v_axis_size': 3,
        'plain_state_dim0': 64,
        'plain_state_dim1': 80,
        'plain_state_dim2': 96,
        'no_residual_latent': False,
        'residual_scale': 0.5,
        'amp': False,
        'seed': 42,
    }
    for key, default in defaults.items():
        if cfg[key] is None:
            cfg[key] = base_args.get(key, default) if base_args.get(key) is not None else default
    if cfg['val_take'] is None:
        cfg['val_take'] = (base_args.get('val_take', 1024) if base_args else -1)
    if cfg['train_take'] == -1 and base_args and args.model == 'hier_flow_ssm':
        cfg['train_take'] = base_args.get('train_take', -1)
    if args.model == 'hier_flow_ssm' and args.amp is None:
        cfg['amp'] = bool(base_args.get('amp', True))
    return cfg


def make_loaders(cfg, device):
    dataset = SEVIRVILDataset(
        index_path=cfg['index_path'],
        input_len=cfg['input_len'],
        pred_len=cfg['pred_len'],
        normalize=cfg['normalize'],
        return_metadata=True,
    )
    splits = build_year_splits(load_vil_index(cfg['index_path']))
    print('split sizes:', summarize_splits(splits))
    def loader(split, take, shuffle):
        ids = splits[split]
        if take is not None and take > 0:
            ids = ids[:take]
        if not ids:
            raise ValueError(f'Empty {split} split')
        return DataLoader(
            Subset(dataset, ids),
            batch_size=cfg['batch_size'],
            shuffle=shuffle,
            num_workers=cfg['num_workers'],
            pin_memory=device.type == 'cuda',
        )
    return dataset, loader('train', cfg['train_take'], True), loader('val', cfg['val_take'], False)


def run_epoch(model, loader, device, pred_len, amp, optimizer=None, scaler=None, grad_clip_norm=None):
    training = optimizer is not None
    model.train(training)
    total_l1, total_mse, count, skipped = 0.0, 0.0, 0, 0
    for batch in loader:
        x = batch['x'].to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast('cuda', enabled=bool(amp and device.type == 'cuda')):
                y_hat, _ = model(x, pred_steps=pred_len, return_probe=False)
                if not torch.isfinite(y_hat).all():
                    skipped += 1
                    continue
                l1 = F.l1_loss(y_hat, y)
                if not torch.isfinite(l1):
                    skipped += 1
                    continue
                mse = F.mse_loss(y_hat, y) if not training else None
            if training:
                scaler.scale(l1).backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad), grad_clip_norm
                    )
                scaler.step(optimizer)
                scaler.update()
        bs = x.size(0)
        total_l1 += l1.item() * bs
        if mse is not None:
            total_mse += mse.item() * bs
        count += bs
    if count == 0:
        raise RuntimeError('No valid batches in epoch')
    return {'loss': total_l1 / count, 'mse': total_mse / count, 'skipped': skipped}


def save_checkpoint(path, model, optimizer, scaler, epoch, best_val, history, cfg, base_args):
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'best_val': best_val,
        'history': history,
        'args': cfg,
        'model_name': cfg['model'],
    }
    if base_args:
        state['base_args'] = base_args
    torch.save(state, path)


def main():
    args = parse_args()
    base_ckpt = None
    base_args = {}
    resume_ckpt = (torch.load(args.resume, map_location='cpu', weights_only=False)
                   if args.resume else None)
    if args.model == 'hier_flow_ssm':
        if not args.flow_checkpoint and not args.resume:
            raise ValueError('--flow_checkpoint is required for hier_flow_ssm training')
        if args.flow_checkpoint:
            base_ckpt = torch.load(args.flow_checkpoint, map_location='cpu', weights_only=False)
            base_args = dict(base_ckpt.get('args', {}))
    elif resume_ckpt is not None:
        base_args = dict(resume_ckpt.get('args', {}))
    cfg = resolve_config(args, base_args)
    if resume_ckpt is not None:
        retained = resume_ckpt.get('args', {})
        for key in ('model', 'input_len', 'pred_len', 'normalize', 'latent_channels',
                    'encoder_mid_channels', 'guide_channels', 'decoder_channels',
                    'core_feat_dim', 'no_guides', 'mamba_d_state', 'mamba_d_conv',
                    'mamba_expand', 'mamba_num_blocks', 'fernn_hidden_channels',
                    'fernn_v_axis_size', 'fernn_kernel_size', 'plain_v_axis_size',
                    'plain_state_dim0', 'plain_state_dim1', 'plain_state_dim2',
                    'no_residual_latent', 'residual_scale', 'hier_adapter_channels',
                    'hier_gamma_max', 'hier_gamma_init', 'hier_norm_groups',
                    'no_zero_init_adapter'):
            if key in retained:
                cfg[key] = retained[key]
    set_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(cfg['device'] if torch.cuda.is_available() and cfg['device'] != 'cpu' else 'cpu')
    dataset, train_loader, val_loader = make_loaders(cfg, device)
    try:
        model = build_model(cfg['model'], cfg).to(device)
        if base_ckpt is not None:
            initialize_hierarchy(model, base_ckpt)
        if resume_ckpt is not None:
            model.load_state_dict(resume_ckpt['model_state'], strict=True)
        if cfg['model'] == 'hier_flow_ssm':
            parameters = freeze_except_hierarchy(model)
        else:
            parameters = list(model.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg['amp'] and device.type == 'cuda'))
        ckpt_dir = ensure_dir(CHECKPOINT_ROOT / cfg['run_name'])
        log_dir = ensure_dir(LOG_ROOT / cfg['run_name'])
        (log_dir / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        history = []
        first_epoch, best_val = 1, float('inf')
        if args.resume:
            resumed = resume_ckpt
            optimizer.load_state_dict(resumed['optimizer_state'])
            if 'scaler_state' in resumed:
                scaler.load_state_dict(resumed['scaler_state'])
            history = resumed.get('history', [])
            first_epoch = int(resumed.get('epoch', 0)) + 1
            best_val = float(resumed.get('best_val', float('inf')))
        print('model:', cfg['model'], 'device:', device)
        for epoch in range(first_epoch, cfg['epochs'] + 1):
            tr = run_epoch(model, train_loader, device, cfg['pred_len'], cfg['amp'],
                           optimizer, scaler, cfg['grad_clip_norm'])
            val = run_epoch(model, val_loader, device, cfg['pred_len'], cfg['amp'])
            improved = val['loss'] < best_val
            if improved:
                best_val = val['loss']
            row = {'epoch': epoch, 'train_loss': tr['loss'], 'val_loss': val['loss'],
                   'val_mse': val['mse'], 'train_skipped': tr['skipped'], 'val_skipped': val['skipped']}
            history.append(row)
            print(f"epoch={epoch} train_l1={tr['loss']:.6f} val_l1={val['loss']:.6f} val_mse={val['mse']:.6f}")
            save_checkpoint(ckpt_dir / 'latest.pt', model, optimizer, scaler, epoch,
                            best_val, history, cfg, base_args)
            if improved:
                save_checkpoint(ckpt_dir / 'best.pt', model, optimizer, scaler, epoch,
                                best_val, history, cfg, base_args)
            (log_dir / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
