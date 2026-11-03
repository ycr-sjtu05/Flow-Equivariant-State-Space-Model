from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv('SEVIR_DATA_ROOT', PROJECT_ROOT / 'vil')).resolve()
OUTPUT_ROOT = PROJECT_ROOT / 'outputs'
CHECKPOINT_ROOT = OUTPUT_ROOT / 'checkpoints'
LOG_ROOT = OUTPUT_ROOT / 'logs'
CACHE_ROOT = OUTPUT_ROOT / 'cache'


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_project_dirs():
    for path in (CHECKPOINT_ROOT, LOG_ROOT, CACHE_ROOT):
        ensure_dir(path)


def get_vil_h5_files(years=None) -> list[Path]:
    if not DATA_ROOT.exists():
        return []
    year_dirs = (DATA_ROOT / str(y) for y in years) if years is not None else DATA_ROOT.iterdir()
    files = []
    for year_dir in year_dirs:
        if year_dir.is_dir():
            files.extend(year_dir.glob('*.h5'))
    return sorted(files)
