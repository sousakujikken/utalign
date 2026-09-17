"""モーラ列と MIDI ノート列の単調 DP 対応付け.

遷移:
  1. 1:1            mora i <-> note j
  2. 1 ノート k モーラ (k<=MAX_K)   note j <- moras i..i+k-1
  3. 1 モーラ l ノート (l<=MAX_L)   mora i <- notes j..j+l-1 (メリスマ)
  4. モーラ skip (歌われない)
  5. ノート skip (歌詞にない)
主コストは CTC 残差 |ctc_onset(i) - note_onset(j)|.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Params:
    max_k: int = 3
    max_l: int = 8
    resid_cap: float = 1.5
    share_pen_normal: float = 0.12
    share_pen_special: float = 0.02   # ー/っ/ん
    min_share_dur: float = 0.06
    short_dur_pen: float = 1.0
    melisma_pen: float = 0.10
    melisma_early_tol: float = 0.06
    mora_skip: float = 0.6
    mora_skip_special: float = 0.25
    note_skip: float = 0.4            # 有声ノートを skip
    note_skip_silent: float = 0.1     # 無声 (音声にエネルギーなし) ノートを skip
    melisma_silent_pen: float = 0.6   # 無声ノートをメリスマに使う
    no_ctc_resid: float = 0.10
    phrase_cross_pen: float = 0.3   # メリスマがフレーズ境界 (休符) をまたぐ


@dataclass
class MoraAssign:
    note_ids: list[int] = field(default_factory=list)
    share_index: int = 0        # 共有ノート内で何番目のモーラか (0 = ノートの先頭)
    share_count: int = 1        # そのノートを共有するモーラ数
    skipped: bool = False


def match(ctc_start: np.ndarray, ctc_w: np.ndarray, special: np.ndarray,
          note_on: np.ndarray, note_off: np.ndarray, note_phrase: np.ndarray,
          note_voiced: np.ndarray, p: Params = Params()) -> tuple[list[MoraAssign], float]:
    """ctc_start: [M] (nan = 情報なし), ctc_w: [M] 重み, special: [M] bool (ー/っ/ん),
    note_on/off: [N] (オフセット適用済み秒). 返り値: (モーラ毎の割当, 総コスト)."""
    M = len(ctc_start); N = len(note_on)
    INF = float('inf')
    cost = np.full((M + 1, N + 1), INF)
    back = np.zeros((M + 1, N + 1, 3), dtype=np.int32)  # (type, di, dj)
    cost[0, 0] = 0.0
    has = ~np.isnan(ctc_start)
    dur = note_off - note_on

    def resid(i: int, j: int) -> float:
        if not has[i]:
            return p.no_ctc_resid
        return ctc_w[i] * min(p.resid_cap, abs(ctc_start[i] - note_on[j]))

    def inside(i: int, j: int) -> float:
        if not has[i]:
            return 0.0
        d = max(0.0, note_on[j] - ctc_start[i]) + max(0.0, ctc_start[i] - note_off[j])
        return ctc_w[i] * min(p.resid_cap, d)

    for i in range(M + 1):
        for j in range(N + 1):
            c = cost[i, j]
            if c == INF:
                continue
            # 4. mora skip
            if i < M:
                sk = p.mora_skip_special if special[i] else p.mora_skip
                if c + sk < cost[i + 1, j]:
                    cost[i + 1, j] = c + sk; back[i + 1, j] = (4, 1, 0)
            # 5. note skip
            if j < N:
                sk = p.note_skip if note_voiced[j] else p.note_skip_silent
                if c + sk < cost[i, j + 1]:
                    cost[i, j + 1] = c + sk; back[i, j + 1] = (5, 0, 1)
            if i < M and j < N:
                r0 = resid(i, j)
                # 1. 1:1
                if c + r0 < cost[i + 1, j + 1]:
                    cost[i + 1, j + 1] = c + r0; back[i + 1, j + 1] = (1, 1, 1)
                # 2. share: note j <- moras i..i+k-1
                acc = r0
                n_normal = 0 if special[i] else 1
                for k in range(2, p.max_k + 1):
                    if i + k > M:
                        break
                    m = i + k - 1
                    acc += inside(m, j) + (p.share_pen_special if special[m] else p.share_pen_normal)
                    n_normal += 0 if special[m] else 1
                    # 音価の物理制約: 特殊モーラ (ー/っ/ん) は数に入れない
                    cc = acc + (p.short_dur_pen if n_normal > 1 and dur[j] / n_normal < p.min_share_dur else 0.0)
                    if c + cc < cost[i + k, j + 1]:
                        cost[i + k, j + 1] = c + cc; back[i + k, j + 1] = (2, k, 1)
                # 3. melisma: mora i <- notes j..j+l-1
                acc = r0
                for l in range(2, p.max_l + 1):
                    if j + l > N:
                        break
                    n = j + l - 1
                    early = 0.0
                    if i + 1 < M and has[i + 1]:
                        early = ctc_w[i + 1] * min(p.resid_cap, max(0.0, note_on[n] - p.melisma_early_tol - ctc_start[i + 1]))
                    acc += p.melisma_pen + early
                    if note_phrase[n] != note_phrase[n - 1]:
                        # 休符が長いほどメリスマとして不自然 (0.8s 以上で最大ペナルティ)
                        gap = note_on[n] - note_off[n - 1]
                        acc += p.phrase_cross_pen * min(1.0, max(0.0, gap) / 0.8)
                    if not note_voiced[n]:
                        acc += p.melisma_silent_pen
                    if c + acc < cost[i + 1, j + l]:
                        cost[i + 1, j + l] = c + acc; back[i + 1, j + l] = (3, 1, l)
    # backtrack
    assigns = [MoraAssign() for _ in range(M)]
    i, j = M, N
    while i > 0 or j > 0:
        t, di, dj = (int(v) for v in back[i, j])
        pi, pj = i - di, j - dj
        if t == 1:
            assigns[pi].note_ids = [pj]
        elif t == 2:
            for q in range(di):
                assigns[pi + q].note_ids = [pj]
                assigns[pi + q].share_index = q
                assigns[pi + q].share_count = di
        elif t == 3:
            assigns[pi].note_ids = list(range(pj, pj + dj))
        elif t == 4:
            assigns[pi].skipped = True
        i, j = pi, pj
    return assigns, float(cost[M, N])
