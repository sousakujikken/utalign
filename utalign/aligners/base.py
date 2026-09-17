"""バックエンド共通インターフェース.

流れ: モーラ列 --(mora_tokens)--> CTC ユニット列 (lyrics.build_ctc_units) ; 音声 --(compute_emissions)--> [T, V]
      --(align)--> ユニット毎の (start, end, score)。
16 秒窓 + 前後 2 秒オーバーラップの emission 計算とキャッシュは共通実装 (chunked_emissions)。
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..lyrics import CtcUnit, Mora

SR = 16000
HOP = 320            # wav2vec2 / HuBERT のフレーム間隔 (20ms)
# 1 回にモデルへ入れる窓。macOS 15.1 未満の MPS では畳み込み出力の各次元が 65,536 以下に制限され
# (config.MPS_CONV_MAX_OUTPUT)、HuBERT の最初の Conv1d (kernel 10, stride 5) の出力フレーム数がこれを超えると
# "Output channels > 65536 not supported at the MPS device" で落ちる。セグメント (窓 + 前後オーバーラップ) は
# MPS_MAX_SEG_SAMPLES (約 20.48 秒) 以下に保つこと。16 + 2×2 = 20 秒 = 320,000 サンプル → 63,999 フレームで収まる。
CHUNK_SEC = 16.0
OVERLAP_SEC = 2.0
MPS_MAX_SEG_SAMPLES = (65536 - 1) * 5 + 10   # = 327,685


@dataclass
class UnitSpan:
    unit: int
    start: float
    end: float
    score: float


def load_audio_16k(path: Path) -> np.ndarray:
    """互換用 — 実体は utalign.audio (ffmpeg 不使用: libsndfile / macOS afconvert)."""
    from ..audio import load_audio_16k as _load
    return _load(path)


def num_frames(n_samples: int) -> int:
    return max(0, (n_samples - 400) // HOP + 1)


def chunked_emissions(wav: np.ndarray, run_chunk: Callable[[np.ndarray], np.ndarray],
                      cache_key: str, cache_dir: Optional[Path],
                      chunk_sec: float = CHUNK_SEC, overlap_sec: float = OVERLAP_SEC) -> np.ndarray:
    """run_chunk(セグメント波形) -> log-prob [t, V] を chunk_sec の窓 (前後 overlap_sec 付き) で呼び、
    全曲の [T, V] に貼り合わせる。窓幅を変えると emission が僅かに変わるのでキャッシュキーに窓設定を含める."""
    chunk = int(chunk_sec * SR) // HOP * HOP
    ov = int(overlap_sec * SR) // HOP * HOP
    full_key = f"{cache_key}|chunk={chunk}|ov={ov}"
    key = hashlib.sha1(wav.tobytes() + full_key.encode()).hexdigest()[:16]
    cache = (cache_dir / f"emission_{key}.npy") if cache_dir else None
    if cache and cache.exists():
        return np.load(cache)
    n = len(wav)
    T = num_frames(n)
    out = None
    pos = 0
    while pos < n:
        core_end = min(n, pos + chunk)
        a = max(0, pos - ov)
        b = min(n, core_end + ov)
        em = run_chunk(wav[a:b])
        if out is None:
            out = np.full((T, em.shape[1]), -1e4, dtype=np.float32)
        f0 = a // HOP
        c0 = pos // HOP
        c1 = min(T, core_end // HOP if core_end < n else T)
        for g in range(c0, c1):
            k = g - f0
            if 0 <= k < em.shape[0]:
                out[g] = em[k]
        pos = core_end
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return out


class Aligner(ABC):
    name: str = ""
    model_id: str = ""
    vocab_kind: str = ""        # romaji | phoneme | hira | kata
    supports_star: bool = False
    device: str = "cpu"
    device_fallback: Optional[str] = None   # 実行中に CPU へ退避したときの理由 (pipeline が警告に載せる)
    log: Callable[[str], None] = staticmethod(print)

    @abstractmethod
    def mora_tokens(self, mora: "Mora", next_mora: Optional["Mora"]) -> list[str]:
        """モーラ -> このバックエンドのトークン列. 空リストは「直前ユニットに畳む (トークンを出さない)」."""

    @abstractmethod
    def compute_emissions(self, wav: np.ndarray, cache_dir: Optional[Path] = None) -> np.ndarray:
        ...

    @abstractmethod
    def align(self, emission: np.ndarray, units: list["CtcUnit"]) -> list[Optional[UnitSpan]]:
        """units と同じ長さのリスト. star ユニットや空ユニットは None."""

    def describe(self) -> str:
        return f"{self.name}:{self.model_id} (vocab={self.vocab_kind})"
