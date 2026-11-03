
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_BASE = "http://www.csc.kth.se/cvap/actions/"
_ACTIONS = [
    ("walking", 242),
    ("jogging", 168),
    ("running", 149),
    ("boxing", 194),
    ("handwaving", 218),
    ("handclapping", 176),
]
_ACTION2IDX = {name: idx for idx, (name, _) in enumerate(_ACTIONS)}
_SEQ_TXT = "00sequences.txt"

def _human(mib: int) -> str:
    return f'{mib}\xa0MB' if mib < 1024 else f'{mib / 1024:.1f}\xa0GB'

def _safe_dl(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    with urllib.request.urlopen(url) as resp, open(dst, 'wb') as fh:
        total = int(resp.getheader('Content-Length', '0'))
        pbar = tqdm(total=total, unit='B', unit_scale=True, desc=f'⬇ {dst.name}')
        while (buf := resp.read(16 << 10)):
            fh.write(buf)
            pbar.update(len(buf))
        pbar.close()
    return dst

def _download_kth(root: Path) -> Path:
    root = root.expanduser()
    video_dir = root / 'kth_actions'
    video_dir.mkdir(parents=True, exist_ok=True)
    if (root / _SEQ_TXT).exists() and any(video_dir.glob("*.avi")):
        return video_dir
    _safe_dl(_BASE + _SEQ_TXT, root / _SEQ_TXT)
    for act, mib in _ACTIONS:
        zip_path = root / f'{act}.zip'
        if not zip_path.exists():
            print(f'Downloading {act} ({_human(mib)}) …')
            _safe_dl(_BASE + f'{act}.zip', zip_path)
        marker = video_dir / f'.{act}_unzipped'
        if not marker.exists():
            print(f'🗜️  Extracting {zip_path.name} …')
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(video_dir)
            marker.touch()
    return video_dir

def preprocess_videos(video_dir: Path, hdf5_path: Path, height: int=64, width: int=64) -> None:
    if hdf5_path.exists():
        return

    def process_video(video_path: Path) -> Tuple[str, np.ndarray]:
        video_name = video_path.stem
        if video_name.endswith('_uncomp'):
            video_name = video_name[:-7]
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(gray)
        cap.release()
        return (video_name, np.stack(frames))
    video_paths = list(video_dir.glob('*.avi'))
    print(f'Pre-extracting frames from {len(video_paths)} videos...')
    with h5py.File(hdf5_path, 'w') as f:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_video, vp) for vp in video_paths]
            for future in tqdm(futures, total=len(video_paths)):
                video_name, frames = future.result()
                f.create_dataset(video_name, data=frames, compression='gzip', compression_opts=1)

class KTHVideoClips(Dataset):
    _LINE_RE = re.compile('^(?P<fname>\\S+)\\s+frames\\s+(?P<ranges>[\\d,\\s\\-]+)$')

    def __init__(self, root: Path, seq_len: int=16, height: int=64, width: int | None=None, step: int=1, split: str='train', vx_data_range: int=0, vy_data_range: int=0, split_method: str='person', train_repeat: int=1, train_velocity_resample: bool=False, return_velocity: bool=False, eval_repeat: int=1):
        self.video_root = _download_kth(root)
        self.hdf5_path = root / f'kth_frames_{height}x{width or height}.h5'
        self.seq_len = seq_len
        self.step = step
        self.h = height
        self.w = width or height
        self.split = split
        self.vx_data_range = vx_data_range
        self.vy_data_range = vy_data_range
        self.split_method = split_method
        self.train_repeat = max(1, train_repeat) if split == 'train' else 1
        self.eval_repeat = max(1, eval_repeat) if split != 'train' else 1
        self.train_velocity_resample = bool(train_velocity_resample and split == 'train')
        self.return_velocity = return_velocity
        if not self.hdf5_path.exists():
            print('Pre-extracting frames for faster loading...')
            preprocess_videos(self.video_root, self.hdf5_path, height, self.w)
        print('Loading dataset into memory...')
        self.frames_dict = {}
        with h5py.File(self.hdf5_path, 'r') as f:
            for video_name in tqdm(f.keys(), desc='Loading videos'):
                self.frames_dict[video_name] = f[video_name][:]
        self.sequences: List[Tuple[str, int, int]] = []
        self._parse_sequences(root / _SEQ_TXT)
        if vx_data_range > 0 or vy_data_range > 0:
            if self.train_velocity_resample:
                self.velocities = None
                print('Training split will RESAMPLE velocities on every __getitem__ call.')
            else:
                print('Generating deterministic velocities for data augmentation...')
                rng = np.random.RandomState(1314)
                self.velocities = []
                for _ in range(len(self.sequences)):
                    vx = rng.randint(-vx_data_range, vx_data_range + 1)
                    vy = rng.randint(-vy_data_range, vy_data_range + 1)
                    self.velocities.append((vy, vx))
                print(f'Generated {len(self.velocities)} velocity vectors')
        else:
            self.velocities = None
        total_gb = sum((frames.nbytes for frames in self.frames_dict.values())) / 1024 ** 3
        print(f'Dataset loaded into memory. Total size: {total_gb:.2f} GB')
        repeat_factor = self.train_repeat if self.split == 'train' else self.eval_repeat
        print(f'{self.split} dataset length = {len(self)} (base subsequences: {len(self.sequences)}, repeat factor: {repeat_factor})')

    def _parse_sequences(self, txt_path: Path):
        all_sequences = []
        with open(txt_path) as fh:
            for line in fh:
                m = self._LINE_RE.match(line.strip())
                if not m:
                    continue
                vname = m.group('fname')
                rng_blob = m.group('ranges')
                vpath = self.video_root / f'{vname}_uncomp.avi'
                if not vpath.exists():
                    continue
                if vname not in self.frames_dict:
                    continue
                total_frames = len(self.frames_dict[vname])
                for rng in rng_blob.split(','):
                    rng = rng.strip()
                    if not rng:
                        continue
                    if '-' in rng:
                        s, e = map(int, rng.split('-'))
                    else:
                        s = e = int(rng)
                    e = min(e, total_frames - 1)
                    if e - s + 1 >= self.seq_len * self.step:
                        all_sequences.append((vname, s, e))
        if self.split_method == 'person':
            train_sequences = []
            val_sequences = []
            test_sequences = []
            for vname, s, e in all_sequences:
                person_id = int(vname.split('_')[0][6:8])
                if person_id <= 16:
                    train_sequences.append((vname, s, e))
                elif person_id <= 20:
                    val_sequences.append((vname, s, e))
                else:
                    test_sequences.append((vname, s, e))
        else:
            rng = np.random.RandomState(1314)
            rng.shuffle(all_sequences)
            n = len(all_sequences)
            train_size = int(0.7 * n)
            val_size = int(0.15 * n)
            train_sequences = all_sequences[:train_size]
            val_sequences = all_sequences[train_size:train_size + val_size]
            test_sequences = all_sequences[train_size + val_size:]
        if self.split == 'train':
            self.sequences = train_sequences
        elif self.split == 'val':
            self.sequences = val_sequences
        else:
            self.sequences = test_sequences
        print(f'Loaded {len(self.sequences):,} valid subsequences for {self.split} split using {self.split_method} method.')

    def __len__(self):
        if self.split == 'train':
            return len(self.sequences) * self.train_repeat
        return len(self.sequences) * self.eval_repeat

    def _apply_velocity_shift(self, frame, vy, vx):
        return torch.roll(frame, shifts=(int(vy), int(vx)), dims=(-2, -1))

    def _sample_velocity(self, base_idx: int):
        if self.vx_data_range <= 0 and self.vy_data_range <= 0:
            return (0, 0)
        if self.train_velocity_resample:
            vx = random.randint(-self.vx_data_range, self.vx_data_range)
            vy = random.randint(-self.vy_data_range, self.vy_data_range)
            return (vy, vx)
        vy, vx = self.velocities[base_idx]
        return (vy, vx)

    def __getitem__(self, idx):
        base_count = len(self.sequences)
        base_idx = idx % base_count
        repeat_slot = idx // base_count
        vname, s, e = self.sequences[base_idx]
        max_start = e - self.seq_len * self.step + 1
        if self.split == 'train':
            start = random.randint(s, max_start)
        else:
            num_positions = max_start - s + 1
            if num_positions <= 1:
                start = s
            elif self.eval_repeat == 1:
                start = s + (num_positions - 1) // 2
            else:
                frac = repeat_slot / max(1, self.eval_repeat - 1)
                offset = int(round(frac * (num_positions - 1)))
                start = s + offset
        frames = []
        for i in range(self.seq_len):
            frame_idx = start + i * self.step
            frame = self.frames_dict[vname][frame_idx]
            frame = torch.from_numpy(frame).float().unsqueeze(0) / 255.0
            frames.append(frame)
        clip = torch.stack(frames, dim=0)
        if self.split == 'train' and random.random() < 0.5:
            clip = torch.flip(clip, dims=[3])
        vy, vx = self._sample_velocity(base_idx)
        if vy != 0 or vx != 0:
            clip = torch.stack([
                self._apply_velocity_shift(frame, t * vy, t * vx)
                for t, frame in enumerate(clip)
            ], dim=0)
        action = vname.split('_')[1]
        label = _ACTION2IDX[action]
        if self.return_velocity:
            return (clip, label, (vy, vx))
        return (clip, label)

class FlowMambaCell_KTH(nn.Module):

    def __init__(self, d_model, d_state, v_list):
        super().__init__()
        self.D = d_model
        self.N = d_state
        self.v_list = v_list
        self.num_v = len(self.v_list)
        self.norm = nn.GroupNorm(4, d_model)
        self.conv_delta = nn.Conv2d(d_model, d_state, kernel_size=3, padding=1, padding_mode='circular')
        self.dt_inv_softplus = nn.Parameter(torch.tensor(math.log(math.exp(1) - 1)))
        self.conv_B = nn.Conv2d(d_model, d_state * d_model, kernel_size=3, padding=1, padding_mode='circular', bias=False)
        self.conv_C = nn.Conv2d(d_model, d_model * d_state, kernel_size=3, padding=1, padding_mode='circular', bias=False)
        self.state_mix = nn.Conv2d(d_state, d_state, kernel_size=3, padding=1, groups=d_state, padding_mode='circular', bias=False)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_state)))
        self.D_param = nn.Parameter(torch.ones(d_model))
        self.skip_logit = nn.Parameter(torch.tensor(-2.2))
        self.v_skip = nn.Parameter(torch.zeros(self.num_v, d_model))

    def forward(self, u_t, s_prev):
        B_batch, _, H, W = u_t.shape
        u_t_norm = self.norm(u_t)
        delta = F.softplus(self.conv_delta(u_t_norm) + self.dt_inv_softplus)
        delta = torch.clamp(delta, min=0.0001, max=5.0)
        B_val = self.conv_B(u_t_norm).view(B_batch, self.N, self.D, H, W)
        C_val = self.conv_C(u_t_norm).view(B_batch, self.D, self.N, H, W)
        A = -torch.exp(self.log_A_real)
        A_bar = torch.exp(delta * A.view(1, self.N, 1, 1))
        Bu = torch.einsum('bndhw, bdhw -> bnhw', B_val, u_t)
        B_bar_u = delta * Bu
        A_bar_lifted = A_bar.unsqueeze(1)
        B_bar_u_lifted = B_bar_u.unsqueeze(1)
        s_shift = []
        for i, (vy, vx) in enumerate(self.v_list):
            s_shift.append(torch.roll(s_prev[:, i], shifts=(vy, vx), dims=(2, 3)))
        s_shift = torch.stack(s_shift, dim=1)
        s_shift_flat = s_shift.view(B_batch * self.num_v, self.N, H, W)
        s_shift_mixed = self.state_mix(s_shift_flat).view(B_batch, self.num_v, self.N, H, W)
        s_new = A_bar_lifted * s_shift_mixed + B_bar_u_lifted
        y_state = torch.einsum('bdnhw, bvnhw -> bvdhw', C_val, s_new)
        skip_scale = torch.sigmoid(self.skip_logit)
        branch_skip = 1.0 + 0.05 * torch.tanh(self.v_skip).view(1, self.num_v, self.D, 1, 1)
        y_skip = skip_scale * u_t.unsqueeze(1) * self.D_param.view(1, 1, self.D, 1, 1) * branch_skip
        y_new = y_state + y_skip
        return (y_new, s_new)

class FlowMamba_Classifier(nn.Module):

    def __init__(self, input_channels=1, d_model=128, d_state=8, num_patterns=6, v_list=[(0, 0)], num_layers=3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.v_list = v_list
        self.num_v = len(v_list)
        self.num_layers = num_layers
        self.encoder = nn.Sequential(nn.Conv2d(input_channels, d_model, kernel_size=3, padding=1, padding_mode='circular'), nn.GroupNorm(4, d_model), nn.ReLU())
        self.cells = nn.ModuleList([FlowMambaCell_KTH(d_model=d_model, d_state=d_state, v_list=v_list) for _ in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(d_model, num_patterns)
        self.shallow_fuse_logit = nn.Parameter(torch.tensor(-1.1))
        self._n_params = sum((p.numel() for p in self.parameters()))

    def count_parameters(self, verbose=True):
        if verbose:
            print(f'Total trainable parameters: {self._n_params / 1000000.0:.2f} M')
        return self._n_params

    def _normalize_branch_idx(self, branch_idx, batch_size, device):
        if branch_idx is None:
            return None
        if isinstance(branch_idx, int):
            idx = torch.full((batch_size,), branch_idx, dtype=torch.long, device=device)
        elif isinstance(branch_idx, list):
            idx = torch.tensor(branch_idx, dtype=torch.long, device=device)
        elif torch.is_tensor(branch_idx):
            idx = branch_idx.to(device=device, dtype=torch.long)
            if idx.ndim == 0:
                idx = idx.repeat(batch_size)
        else:
            raise TypeError(f'Unsupported branch_idx type: {type(branch_idx)}')
        if idx.shape[0] != batch_size:
            raise ValueError(f'branch_idx must have shape [B], got {idx.shape}')
        return idx.clamp(min=0, max=self.num_v - 1)

    def _apply_branch_control(self, branches, drop_branch_idx=None, keep_only_branch_idx=None):
        B, V, D, H, W = branches.shape
        out = branches
        keep_idx = self._normalize_branch_idx(keep_only_branch_idx, B, branches.device)
        if keep_idx is not None:
            keep_mask = torch.zeros(B, V, device=branches.device, dtype=branches.dtype)
            keep_mask[torch.arange(B, device=branches.device), keep_idx] = 1.0
            out = out * keep_mask[:, :, None, None, None]
        drop_idx = self._normalize_branch_idx(drop_branch_idx, B, branches.device)
        if drop_idx is not None:
            drop_mask = torch.ones(B, V, device=branches.device, dtype=branches.dtype)
            drop_mask[torch.arange(B, device=branches.device), drop_idx] = 0.0
            out = out * drop_mask[:, :, None, None, None]
        return out

    def forward(self, x, return_aux=False, full_aux=False, drop_branch_idx=None, keep_only_branch_idx=None):
        B, T, C, H, W = x.size()
        s_t = [torch.zeros(B, self.num_v, self.d_state, H, W, device=x.device) for _ in range(self.num_layers)]
        full_branch_outputs = [] if full_aux else None
        full_branch_states = [] if full_aux else None
        last_first_branches = None
        last_final_branches = None
        for t in range(T):
            u_t = self.encoder(x[:, t])
            y_first = None
            for l in range(self.num_layers):
                y_t, s_t[l] = self.cells[l](u_t, s_t[l])
                if l == 0:
                    y_first = y_t
                if l < self.num_layers - 1:
                    u_t = y_t.max(dim=1)[0]
            last_first_branches = y_first
            last_final_branches = y_t
            if full_aux:
                full_branch_outputs.append(y_t)
                full_branch_states.append(s_t[-1].clone())
        fuse_scale = torch.sigmoid(self.shallow_fuse_logit)
        last_branches = last_final_branches + fuse_scale * last_first_branches
        branches_for_cls = self._apply_branch_control(last_branches, drop_branch_idx=drop_branch_idx, keep_only_branch_idx=keep_only_branch_idx)
        out = branches_for_cls.max(dim=1)[0]
        out = self.pool(out)
        out = out.reshape(B, -1)
        logits = self.fc(out)
        if not return_aux:
            return logits
        aux = {'branch_outputs_last': last_branches, 'v_list': list(self.v_list)}
        if full_aux:
            aux['branch_outputs'] = torch.stack(full_branch_outputs, dim=1)
            aux['branch_states'] = torch.stack(full_branch_states, dim=1)
        return (logits, aux)

class PlainMambaCell_KTH(nn.Module):

    def __init__(self, d_model, d_state):
        super().__init__()
        self.D = d_model
        self.N = d_state
        self.norm = nn.GroupNorm(4, d_model)
        self.conv_delta = nn.Conv2d(d_model, d_state, kernel_size=3, padding=1, padding_mode='circular')
        self.dt_inv_softplus = nn.Parameter(torch.tensor(math.log(math.exp(1) - 1)))
        self.conv_B = nn.Conv2d(d_model, d_state * d_model, kernel_size=3, padding=1, padding_mode='circular', bias=False)
        self.conv_C = nn.Conv2d(d_model, d_model * d_state, kernel_size=3, padding=1, padding_mode='circular', bias=False)
        self.state_mix = nn.Conv2d(d_state, d_state, kernel_size=3, padding=1, groups=d_state, padding_mode='circular', bias=False)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_state)))
        self.D_param = nn.Parameter(torch.ones(d_model))
        self.skip_logit = nn.Parameter(torch.tensor(-2.2))

    def forward(self, u_t, s_prev):
        B_batch, _, H, W = u_t.shape
        u_t_norm = self.norm(u_t)
        delta = F.softplus(self.conv_delta(u_t_norm) + self.dt_inv_softplus)
        delta = torch.clamp(delta, min=0.0001, max=5.0)
        B_val = self.conv_B(u_t_norm).view(B_batch, self.N, self.D, H, W)
        C_val = self.conv_C(u_t_norm).view(B_batch, self.D, self.N, H, W)
        A = -torch.exp(self.log_A_real)
        A_bar = torch.exp(delta * A.view(1, self.N, 1, 1))
        Bu = torch.einsum('bndhw, bdhw -> bnhw', B_val, u_t)
        B_bar_u = delta * Bu
        s_prev_mixed = self.state_mix(s_prev)
        s_new = A_bar * s_prev_mixed + B_bar_u
        y_state = torch.einsum('bdnhw, bnhw -> bdhw', C_val, s_new)
        skip_scale = torch.sigmoid(self.skip_logit)
        y_skip = skip_scale * u_t * self.D_param.view(1, self.D, 1, 1)
        y_new = y_state + y_skip
        return (y_new, s_new)

class PlainMamba_Classifier(nn.Module):

    def __init__(self, input_channels=1, d_model=128, d_state=8, num_patterns=6, num_layers=3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        self.encoder = nn.Sequential(nn.Conv2d(input_channels, d_model, kernel_size=3, padding=1, padding_mode='circular'), nn.GroupNorm(4, d_model), nn.ReLU())
        self.cells = nn.ModuleList([PlainMambaCell_KTH(d_model=d_model, d_state=d_state) for _ in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(d_model, num_patterns)
        self._n_params = sum((p.numel() for p in self.parameters()))

    def forward(self, x):
        B, T, C, H, W = x.size()
        s_t = [torch.zeros(B, self.d_state, H, W, device=x.device) for _ in range(self.num_layers)]
        out_t = None
        for t in range(T):
            u_t = self.encoder(x[:, t])
            for l in range(self.num_layers):
                u_t, s_t[l] = self.cells[l](u_t, s_t[l])
            out_t = u_t
        out = self.pool(out_t)
        out = out.reshape(B, -1)
        logits = self.fc(out)
        return logits

    def count_parameters(self, verbose=True):
        if verbose:
            print(f'Total trainable parameters: {self._n_params / 1000000.0:.2f} M')
        return self._n_params

class FERNN(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers=1, v_list=[(0, 0)]):
        super(FERNN, self).__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.v_list = v_list
        self.num_v = len(v_list)
        self.num_layers = num_layers
        self.register_buffer('shift_kernels', self._create_shift_kernels(kernel_size))
        self.conv_u_first = nn.Sequential(nn.Conv2d(input_channels, 32, 5, 1, 2, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, hidden_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True))
        self.conv_u_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(hidden_channels), nn.ReLU(inplace=True)) for _ in range(num_layers - 1)])
        self.conv_h_layers = nn.ModuleList([nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False, padding_mode='circular')) for _ in range(num_layers)])
        self.activation = nn.Tanh()

    def _create_shift_kernels(self, kernel_size):
        max_shift = max(max((abs(vx) for _, vx in self.v_list)), max((abs(vy) for vy, _ in self.v_list)))
        kernel_size = max_shift * 2 + 1
        shift_kernels = torch.zeros(self.num_v, 1, kernel_size, kernel_size)
        for i, (vy, vx) in enumerate(self.v_list):
            shift_kernels[i, 0, kernel_size // 2 - vy, kernel_size // 2 - vx] = 1.0
        return shift_kernels

    def _process_layer(self, h, u_t, layer_idx):
        batch_size = h.size(0)
        height, width = (h.size(-2), h.size(-1))
        h_reshaped = h.view(batch_size, self.num_v, self.hidden_channels, height, width)
        h_reshaped = h_reshaped.transpose(1, 2)
        h_reshaped = h_reshaped.reshape(-1, self.num_v, height, width)
        pad_size = self.shift_kernels.size(-1) // 2
        h_padded = F.pad(h_reshaped, (pad_size, pad_size, pad_size, pad_size), mode='circular')
        h_shifted = F.conv2d(h_padded, self.shift_kernels, padding=0, groups=self.num_v)
        h_shifted = h_shifted.view(batch_size, self.hidden_channels, self.num_v, height, width)
        h_shifted = h_shifted.transpose(1, 2)
        h_shifted = h_shifted.reshape(batch_size * self.num_v, self.hidden_channels, height, width)
        if layer_idx == 0:
            u_processed = self.conv_u_first(u_t).view(batch_size, 1, self.hidden_channels, height, width).repeat(1, self.num_v, 1, 1, 1)
        else:
            u_t_reshaped = u_t.view(batch_size * self.num_v, self.hidden_channels, height, width)
            u_processed = self.conv_u_layers[layer_idx - 1](u_t_reshaped)
            u_processed = u_processed.view(batch_size, self.num_v, self.hidden_channels, height, width)
        h_new = self.activation(u_processed + self.conv_h_layers[layer_idx](h_shifted).view(batch_size, self.num_v, self.hidden_channels, height, width))
        return h_new

    def forward(self, u):
        batch_size, time_steps, channels, in_height, in_width = u.size()
        height, width = (in_height, in_width)
        h_layers = [torch.zeros(batch_size, self.num_v, self.hidden_channels, height, width, device=u.device) for _ in range(self.num_layers)]
        outputs = []
        for t in range(time_steps):
            u_t = u[:, t]
            for layer_idx in range(self.num_layers):
                h_layers[layer_idx] = self._process_layer(h_layers[layer_idx], u_t if layer_idx == 0 else h_layers[layer_idx - 1], layer_idx)
            out = h_layers[-1].view(batch_size, self.num_v, self.hidden_channels, height, width)
            out = out.permute(1, 0, 2, 3, 4)
            outputs.append(out.unsqueeze(2))
        outputs = torch.cat(outputs, dim=2)
        return outputs

class FERNN_Classifier(nn.Module):

    def __init__(self, input_size, hidden_size, hidden_channels, num_patterns, v_list=[(0, 0)], num_layers=1):
        super(FERNN_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_channels = hidden_channels
        self.v_list = v_list
        self.num_v = len(v_list)
        self.lstm = FERNN(input_channels=1, hidden_channels=self.hidden_channels, kernel_size=3, v_list=v_list, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.hidden_channels, num_patterns)
        self._n_params = sum((p.numel() for p in self.parameters()))

    def count_parameters(self, verbose=True):
        if verbose:
            print(f'Total trainable parameters: {self._n_params / 1000000.0:.2f} M')
        return self._n_params

    def _normalize_branch_idx(self, branch_idx, batch_size, device):
        if branch_idx is None:
            return None
        if isinstance(branch_idx, int):
            idx = torch.full((batch_size,), branch_idx, dtype=torch.long, device=device)
        elif isinstance(branch_idx, list):
            idx = torch.tensor(branch_idx, dtype=torch.long, device=device)
        elif torch.is_tensor(branch_idx):
            idx = branch_idx.to(device=device, dtype=torch.long)
            if idx.ndim == 0:
                idx = idx.repeat(batch_size)
        else:
            raise TypeError(f'Unsupported branch_idx type: {type(branch_idx)}')
        if idx.shape[0] != batch_size:
            raise ValueError(f'branch_idx must have shape [B], got {idx.shape}')
        return idx.clamp(min=0, max=self.num_v - 1)

    def _apply_branch_control(self, branches, drop_branch_idx=None, keep_only_branch_idx=None):
        B, T, V, D, H, W = branches.shape
        out = branches
        keep_idx = self._normalize_branch_idx(keep_only_branch_idx, B, branches.device)
        if keep_idx is not None:
            keep_mask = torch.zeros(B, V, device=branches.device, dtype=branches.dtype)
            keep_mask[torch.arange(B, device=branches.device), keep_idx] = 1.0
            out = out * keep_mask[:, None, :, None, None, None]
        drop_idx = self._normalize_branch_idx(drop_branch_idx, B, branches.device)
        if drop_idx is not None:
            drop_mask = torch.ones(B, V, device=branches.device, dtype=branches.dtype)
            drop_mask[torch.arange(B, device=branches.device), drop_idx] = 0.0
            out = out * drop_mask[:, None, :, None, None, None]
        return out

    def forward(self, x, return_aux=False, drop_branch_idx=None, keep_only_branch_idx=None):
        rnn_out = self.lstm(x)
        branches = rnn_out.permute(1, 2, 0, 3, 4, 5).contiguous()
        branches_for_cls = self._apply_branch_control(branches, drop_branch_idx=drop_branch_idx, keep_only_branch_idx=keep_only_branch_idx)
        out = branches_for_cls.max(dim=2)[0]
        out = out[:, -1]
        out = self.pool(out)
        out = out.reshape(out.shape[0], -1)
        logits = self.fc(out)
        if not return_aux:
            return logits
        aux = {'branch_outputs': branches, 'branch_states': branches, 'v_list': list(self.v_list)}
        return (logits, aux)

def unpack_velocity_batch(vel):
    if isinstance(vel, (list, tuple)) and len(vel) == 2 and torch.is_tensor(vel[0]):
        vy = vel[0].cpu().numpy()
        vx = vel[1].cpu().numpy()
        return list(zip(vy.tolist(), vx.tolist()))
    if torch.is_tensor(vel):
        arr = vel.cpu().numpy()
        if arr.ndim == 2 and arr.shape[1] == 2:
            return [tuple(map(int, x)) for x in arr]
    raise TypeError(f'Unsupported velocity batch type: {type(vel)}')

def flow_alignment_aux_loss(branch_outputs_last, raw_velocities, v_list, radius=1, margin=0.05, exact_ce_weight=0.5, temp=0.7):
    device = branch_outputs_last.device
    B, V, D, H, W = branch_outputs_last.shape
    energy = branch_outputs_last.abs().mean(dim=(2, 3, 4))
    gt_idx = []
    exact_mask = []
    dist = torch.empty(B, V, device=device)
    for b, (tvy, tvx) in enumerate(raw_velocities):
        tvy, tvx = (int(tvy), int(tvx))
        if (tvy, tvx) in v_list:
            gt_idx.append(v_list.index((tvy, tvx)))
            exact_mask.append(True)
        else:
            best_i, best_d = (None, None)
            for i, (vy, vx) in enumerate(v_list):
                d = max(abs(vy - tvy), abs(vx - tvx))
                if best_d is None or d < best_d:
                    best_d = d
                    best_i = i
            gt_idx.append(best_i)
            exact_mask.append(False)
        for i, (vy, vx) in enumerate(v_list):
            dist[b, i] = max(abs(vy - tvy), abs(vx - tvx))
    gt_idx = torch.tensor(gt_idx, dtype=torch.long, device=device)
    exact_mask = torch.tensor(exact_mask, dtype=torch.bool, device=device)
    near_mask = dist <= radius
    far_mask = dist > radius
    near_count = near_mask.sum(dim=1).clamp_min(1)
    far_count = far_mask.sum(dim=1).clamp_min(1)
    near_mean = (energy * near_mask).sum(dim=1) / near_count
    far_mean = (energy * far_mask).sum(dim=1) / far_count
    loss_near_far = F.relu(margin - (near_mean - far_mean)).mean()
    if exact_mask.any():
        logits_e = energy[exact_mask] / temp
        loss_exact = F.cross_entropy(logits_e, gt_idx[exact_mask])
    else:
        loss_exact = energy.new_tensor(0.0)
    return loss_near_far + exact_ce_weight * loss_exact


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_velocities(vx_range, vy_range):
    return [(vy, vx) for vx in range(-vx_range, vx_range + 1)
                     for vy in range(-vy_range, vy_range + 1)]


def build_model(args):
    velocities = build_velocities(args.vx_range, args.vy_range)
    if args.model == "flow_mamba":
        return FlowMamba_Classifier(
            input_channels=1, d_model=args.d_model, d_state=args.d_state,
            num_patterns=6, v_list=velocities, num_layers=3,
        )
    if args.model == "plain_mamba":
        return PlainMamba_Classifier(
            input_channels=1, d_model=args.d_model, d_state=args.d_state,
            num_patterns=6, num_layers=3,
        )
    if args.model == "fernn":
        return FERNN_Classifier(
            input_size=args.height, hidden_size=args.height * args.width,
            hidden_channels=args.fernn_channels, num_patterns=6,
            v_list=velocities, num_layers=args.fernn_layers,
        )
    raise ValueError(args.model)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(args, split, data_range=None, return_velocity=False):
    if data_range is None:
        data_range = args.vx_data_range, args.vy_data_range
    return KTHVideoClips(
        Path(args.data_root), seq_len=16, height=args.height, width=args.width,
        step=2, split=split, vx_data_range=data_range[0],
        vy_data_range=data_range[1], split_method="person",
        train_repeat=args.train_repeat if split == "train" else 1,
        train_velocity_resample=args.train_velocity_resample if split == "train" else False,
        return_velocity=return_velocity,
        eval_repeat=1 if split == "train" else args.eval_repeat,
    )


def make_loader(dataset, args, shuffle=False):
    kwargs = dict(
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    if args.num_workers:
        kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    return DataLoader(dataset, **kwargs)


def train_epoch(model, loader, optimizer, device, args):
    model.train()
    n, correct, total_loss = 0, 0, 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        clips, labels = batch[:2]
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if args.model == "flow_mamba" and args.align_aux_weight > 0:
            velocities = unpack_velocity_batch(batch[2])
            logits, aux = model(clips, return_aux=True, full_aux=False)
            cls_loss = F.cross_entropy(logits, labels)
            aux_loss = flow_alignment_aux_loss(
                aux["branch_outputs_last"], velocities, aux["v_list"],
                radius=args.align_radius, margin=args.align_margin,
                exact_ce_weight=args.align_exact_ce_weight,
                temp=args.align_temp,
            )
            loss = cls_loss + args.align_aux_weight * aux_loss
        else:
            logits = model(clips)
            loss = F.cross_entropy(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n += len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total_loss += loss.item() * len(labels)
    return {"loss": total_loss / n, "accuracy": correct / n}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n, correct, total_loss = 0, 0, 0.0
    for batch in tqdm(loader, desc="eval", leave=False):
        clips, labels = batch[:2]
        clips, labels = clips.to(device), labels.to(device)
        logits = model(clips)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
    return {"loss": total_loss / n, "accuracy": correct / n}


def save_checkpoint(path, model, optimizer, epoch, args, best_val_acc, metrics):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "best_val_acc": best_val_acc,
        "metrics": metrics,
    }, path)


def train_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/kth")
    parser.add_argument("--checkpoint_root", type=str, default="./checkpoints")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--model", choices=["flow_mamba", "plain_mamba", "fernn"], default="flow_mamba")
    parser.add_argument("--d_model", type=int, default=40)
    parser.add_argument("--d_state", type=int, default=6)
    parser.add_argument("--fernn_channels", type=int, default=128)
    parser.add_argument("--fernn_layers", type=int, default=2)
    parser.add_argument("--vx_range", type=int, default=1)
    parser.add_argument("--vy_range", type=int, default=1)
    parser.add_argument("--vx_data_range", type=int, default=2)
    parser.add_argument("--vy_data_range", type=int, default=2)
    parser.add_argument("--test_velocity_range", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--train_repeat", type=int, default=4)
    parser.add_argument("--eval_repeat", type=int, default=4)
    parser.add_argument("--train_velocity_resample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--align_aux_weight", type=float, default=None)
    parser.add_argument("--align_radius", type=int, default=1)
    parser.add_argument("--align_margin", type=float, default=0.05)
    parser.add_argument("--align_exact_ce_weight", type=float, default=0.5)
    parser.add_argument("--align_temp", type=float, default=0.7)
    args = parser.parse_args()
    if args.align_aux_weight is None:
        args.align_aux_weight = 0.08 if args.model == "flow_mamba" else 0.0
    if args.model != "flow_mamba" and args.align_aux_weight:
        parser.error("Alignment loss is available only for flow_mamba")
    set_seed(args.seed)
    model = build_model(args).to(args.device)
    print(f"Model: {args.model}; trainable parameters: {count_parameters(model):,}")
    run_name = args.run_name or f"{args.model}_v{args.vx_data_range}_dm{args.d_model}_ds{args.d_state}_seed{args.seed}"
    output_dir = Path(args.checkpoint_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch, best_val_acc = 0, float("-inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_acc = float(checkpoint.get("best_val_acc", best_val_acc))
    train_loader = make_loader(
        make_dataset(args, "train", return_velocity=args.model == "flow_mamba" and args.align_aux_weight > 0),
        args, shuffle=True,
    )
    val_loader = make_loader(make_dataset(args, "val"), args)
    test_loader = make_loader(
        make_dataset(args, "test", data_range=(args.test_velocity_range, args.test_velocity_range)), args
    )
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, args)
        val_metrics = evaluate(model, val_loader, args.device)
        metrics = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        print(json.dumps(metrics))
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        improved = val_metrics["accuracy"] > best_val_acc
        if improved:
            best_val_acc = val_metrics["accuracy"]
        save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, args, best_val_acc, metrics)
        if improved:
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, args, best_val_acc, metrics)
    best_path = output_dir / "best.pt"
    if best_path.exists():
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        model.load_state_dict(best["model"], strict=True)
        print(json.dumps({"best_epoch": best["epoch"] + 1, "test": evaluate(model, test_loader, args.device)}))


def nearest_branch_indices(velocities, branches, device):
    target = torch.tensor(velocities, dtype=torch.float32, device=device)
    candidates = torch.tensor(branches, dtype=torch.float32, device=device)
    distances = ((target[:, None, :] - candidates[None, :, :]) ** 2).sum(dim=-1)
    return distances.argmin(dim=1)


def restore_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    saved = checkpoint["args"]
    args = SimpleNamespace(
        model=saved["model"],
        d_model=int(saved.get("d_model", 40)) * int(saved.get("channel_multiplier", 1)),
        d_state=int(saved.get("d_state", 6)),
        height=int(saved.get("height", 32)),
        width=int(saved.get("width", 32)),
        vx_range=int(saved.get("vx_range", 1)),
        vy_range=int(saved.get("vy_range", 1)),
        fernn_channels=128 * int(saved.get("channel_multiplier", 1)),
        fernn_layers=int(saved.get("fernn_layers", saved.get("num_rnn_layers", 2))),
        data_root=saved.get("data_root", "./data/kth"),
        vx_data_range=2,
        vy_data_range=2,
        train_repeat=1,
        train_velocity_resample=False,
        eval_repeat=int(saved.get("eval_repeat", 4)),
        num_workers=0,
        batch_size=16,
        device=str(device),
        prefetch_factor=2,
    )
    if args.model not in {"flow_mamba", "fernn"}:
        raise ValueError("Branch diagnostics require flow_mamba or fernn")
    model = build_model(args).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, args


@torch.no_grad()
def analyze(model, loader, branches, device):
    total = correct = hit1 = top5 = drop_correct = 0
    for clips, labels, velocity in tqdm(loader, desc="analyze"):
        clips, labels = clips.to(device), labels.to(device)
        values = unpack_velocity_batch(velocity)
        target = nearest_branch_indices(values, branches, device)
        logits, aux = model(clips, return_aux=True)
        if "branch_outputs_last" in aux:
            outputs = aux["branch_outputs_last"]
        else:
            outputs = aux["branch_outputs"][:, -1]
        energy = outputs.abs().mean(dim=(2, 3, 4))
        hit1 += (energy.argmax(dim=1) == target).sum().item()
        top5 += (energy.topk(k=min(5, energy.shape[1]), dim=1).indices == target[:, None]).any(dim=1).sum().item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        drop_logits = model(clips, drop_branch_idx=target)
        drop_correct += (drop_logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return {
        "samples": total,
        "accuracy": correct / total,
        "hit_at_1": hit1 / total,
        "top_5": top5 / total,
        "drop_gt_accuracy": drop_correct / total,
    }


def analyze_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--test_velocity_range", type=int, default=2)
    parser.add_argument("--eval_repeat", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    options = parser.parse_args()
    model, args = restore_model(options.checkpoint, options.device)
    if options.data_root is not None:
        args.data_root = options.data_root
    args.vx_data_range = options.test_velocity_range
    args.vy_data_range = options.test_velocity_range
    args.eval_repeat = options.eval_repeat
    args.batch_size = options.batch_size
    args.num_workers = options.num_workers
    dataset = make_dataset(args, options.split, return_velocity=True)
    loader = make_loader(dataset, args)
    branches = build_velocities(args.vx_range, args.vy_range)
    result = analyze(model, loader, branches, options.device)
    print(json.dumps(result, indent=2))
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


def visualize_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data/kth")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("kth_motion.png"))
    args = parser.parse_args()
    dataset = KTHVideoClips(
        root=Path(args.data_root), seq_len=16, height=32, width=32, step=2,
        split=args.split, vx_data_range=0, vy_data_range=0,
        split_method="person", eval_repeat=1,
    )
    clip, label = dataset[args.index]
    times = (0, 3, 6, 9, 12, 15)
    fig, axes = plt.subplots(3, len(times), figsize=(12, 5))
    for row, vx in enumerate((0, 1, 2)):
        for col, t in enumerate(times):
            frame = dataset._apply_velocity_shift(clip[t], 0, vx * t)
            ax = axes[row, col]
            ax.imshow(frame.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t={t}")
            if col == 0:
                ax.set_ylabel(f"vx={vx}")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        sys.argv.pop(1)
        analyze_main()
    elif len(sys.argv) > 1 and sys.argv[1] == "visualize":
        sys.argv.pop(1)
        visualize_main()
    else:
        train_main()
