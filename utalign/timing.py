"""オフセット推定と最終時刻の確定."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .lyrics import Mora
from .matcher import MoraAssign, Params, match
from .midi import Note


def xcorr_offset(wav16k: np.ndarray, notes: list[Note], max_lag: float = 0.5) -> float:
    """オンセット強度 x MIDI ノートオン列の相互相関 (音声 - MIDI, 秒)."""
    import librosa
    sr = 16000; hop = 160
    oenv = librosa.onset.onset_strength(y=wav16k, sr=sr, hop_length=hop)
    T = len(oenv)
    imp = np.zeros(T)
    for n in notes:
        k = int(round(n.start * sr / hop))
        if 0 <= k < T:
            imp[k] = 1
    oc = (oenv - oenv.mean()) / (oenv.std() + 1e-9)
    L = int(max_lag * sr / hop)
    best = (-1e9, 0.0)
    for lag in range(-L, L + 1):
        if lag >= 0:
            c = float(np.dot(oc[lag:], imp[:T - lag]))
        else:
            c = float(np.dot(oc[:T + lag], imp[-lag:]))
        if c > best[0]:
            best = (c, lag * hop / sr)
    return best[1]


@dataclass
class MatchResult:
    assigns: list[MoraAssign]
    offset: float
    residual_median: float
    residual_p90: float
    residual_over150: float
    drift_slope: float
    n_mora_skipped: int
    n_note_skipped: int
    cost: float
    n_note_unvoiced: int = 0


def note_voiced_flags(wav16k: np.ndarray, notes: list[Note], offset: float, frac: float = 0.4) -> np.ndarray:
    """各ノート区間に音声エネルギーがあるか (RMS が全体の有声レベルの一定比率を超えるフレームが frac 以上)."""
    sr = 16000; hop = 160
    T = len(wav16k) // hop
    rms = np.sqrt(np.mean(wav16k[:T * hop].reshape(T, hop) ** 2, axis=1))
    thr = np.percentile(rms[rms > 0], 90) * 0.08
    out = np.zeros(len(notes), dtype=bool)
    for k, n in enumerate(notes):
        a = int((n.start + offset) * sr / hop); b = max(a + 1, int((n.end + offset) * sr / hop))
        seg = rms[max(0, a):min(T, b)]
        out[k] = len(seg) > 0 and np.mean(seg > thr) >= frac
    return out


def run_match(moras: list[Mora], ctc_start: np.ndarray, ctc_score: np.ndarray,
              notes: list[Note], offset0: float, wav16k: Optional[np.ndarray] = None,
              params: Params = Params(), iters: int = 2) -> MatchResult:
    special = np.array([m.kind in ('choon', 'sokuon', 'hatsuon') for m in moras])
    w = 0.5 + 0.5 * np.nan_to_num(ctc_score, nan=0.5)
    on0 = np.array([n.start for n in notes]); off0 = np.array([n.end for n in notes])
    ph = np.array([n.phrase for n in notes])
    voiced = note_voiced_flags(wav16k, notes, offset0) if wav16k is not None else np.ones(len(notes), dtype=bool)
    offset = offset0
    assigns = None
    for _ in range(iters):
        assigns, cost = match(ctc_start, w, special, on0 + offset, off0 + offset, ph, voiced, params)
        # 1:1 かつ CTC ありのペアで残差を評価
        pairs = [(ctc_start[i], on0[a.note_ids[0]]) for i, a in enumerate(assigns)
                 if a.note_ids and a.share_index == 0 and not np.isnan(ctc_start[i])]
        d = np.array([c - o for c, o in pairs])
        new_offset = float(np.median(d))
        if abs(new_offset - offset) < 0.005:
            offset = new_offset
            break
        offset = new_offset
    assigns, cost = match(ctc_start, w, special, on0 + offset, off0 + offset, ph, voiced, params)
    pairs = [(ctc_start[i], on0[a.note_ids[0]]) for i, a in enumerate(assigns)
             if a.note_ids and a.share_index == 0 and not np.isnan(ctc_start[i])]
    c = np.array([x for x, _ in pairs]); o = np.array([y for _, y in pairs])
    resid = np.abs(c - (o + offset))
    slope = float(np.polyfit(o, c, 1)[0]) if len(o) > 10 else 1.0
    used = set(n for a in assigns for n in a.note_ids)
    return MatchResult(assigns=assigns, offset=offset,
                       residual_median=float(np.median(resid)), residual_p90=float(np.percentile(resid, 90)),
                       residual_over150=float(np.mean(resid > 0.15)), drift_slope=slope,
                       n_mora_skipped=sum(a.skipped for a in assigns), n_note_skipped=len(notes) - len(used),
                       n_note_unvoiced=int((~voiced).sum()),
                       cost=cost)


@dataclass
class MoraTime:
    start: Optional[float]
    end: Optional[float]
    source: str          # midi | ctc_clamped | ctc_shared | split | none
    end_kind: str        # next_onset | note_off | none
    midi_start: Optional[float]
    ctc_start: Optional[float]
    phrase: int


def resolve_times(moras: list[Mora], assigns: list[MoraAssign], notes: list[Note],
                  ctc_start: np.ndarray, offset: float, clamp: float = 0.10, min_gap: float = 0.04) -> list[MoraTime]:
    M = len(moras)
    out: list[Optional[MoraTime]] = [None] * M
    on = [n.start + offset for n in notes]; off = [n.end + offset for n in notes]
    # start を決める
    starts: list[Optional[float]] = [None] * M
    for i, a in enumerate(assigns):
        if a.skipped or not a.note_ids:
            continue
        j = a.note_ids[0]
        c = ctc_start[i]
        if a.share_index == 0:
            first_of_phrase = (j == 0) or (notes[j].phrase != notes[j - 1].phrase)
            if first_of_phrase or np.isnan(c):
                starts[i] = on[j]; src = 'midi'
            else:
                starts[i] = float(np.clip(c, on[j] - clamp, on[j] + clamp)); src = 'ctc_clamped'
        else:
            prev = starts[i - 1] if i > 0 and starts[i - 1] is not None else on[j]
            # 共有ノートの終端は量子化されているので、次ノートの開始まで (休符区間も) 許容する
            limit = off[j]
            if j + 1 < len(notes) and on[j + 1] - off[j] < 0.5:
                limit = max(off[j], on[j + 1])
            lo = prev + min_gap; hi = limit - min_gap * (a.share_count - a.share_index)
            if hi < lo:
                hi = lo
            if np.isnan(c):
                remain = a.share_count - a.share_index + 1
                starts[i] = prev + (limit - prev) / remain; src = 'split'
            else:
                starts[i] = float(np.clip(c, lo, hi)); src = 'ctc_shared'
        out[i] = MoraTime(start=starts[i], end=None, source=src, end_kind='none',
                          midi_start=on[j], ctc_start=None if np.isnan(c) else float(c), phrase=notes[j].phrase)
    # end を決める
    sung = [i for i in range(M) if out[i] is not None]
    for idx, i in enumerate(sung):
        a = assigns[i]; last = a.note_ids[-1]
        nxt = sung[idx + 1] if idx + 1 < len(sung) else None
        if nxt is not None:
            nj = assigns[nxt].note_ids[0]
            same_phrase = notes[nj].phrase == notes[last].phrase
            if same_phrase and out[nxt].start is not None and out[nxt].start > out[i].start:
                out[i].end = out[nxt].start; out[i].end_kind = 'next_onset'
                continue
        out[i].end = max(off[last], out[i].start + 0.03); out[i].end_kind = 'note_off'
    for i in range(M):
        if out[i] is None:
            out[i] = MoraTime(None, None, 'none', 'none', None,
                              None if np.isnan(ctc_start[i]) else float(ctc_start[i]), -1)
    return out  # type: ignore
