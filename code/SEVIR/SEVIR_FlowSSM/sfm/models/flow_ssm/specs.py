from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PairSpec:
    name: str
    m: int
    ell: int
    v_size: int
    h: int
    w: int
    state_dim: int
    strict: bool
    parent_name: Optional[str] = None
