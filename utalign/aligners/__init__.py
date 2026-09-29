"""強制アライメントのバックエンド (hf_ctc — Apache-2.0 の日本語 CTC モデルのみ)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import Aligner, UnitSpan


def make_aligner(model_id: Optional[str] = None, models_dir: Optional[Path] = None,
                 device: str = "auto", star: bool = False) -> Aligner:
    from .hf_ctc import HFCTCAligner
    if not model_id:
        from ..config import DEFAULT_MODEL
        model_id = DEFAULT_MODEL
    return HFCTCAligner(model_id, models_dir=models_dir, device=device, star=star)


__all__ = ["Aligner", "UnitSpan", "make_aligner"]
