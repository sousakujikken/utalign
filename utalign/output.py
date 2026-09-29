"""結果 JSON, 波形ピーク, 再生用 WAV, クリック音 WAV の出力."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .lyrics import LyricsResult
from .matcher import MoraAssign
from .midi import Note
from .timing import MatchResult, MoraTime


def build_result(lyr: LyricsResult, notes: list[Note], mr: MatchResult, times: list[MoraTime],
                 meta: dict[str, Any]) -> dict[str, Any]:
    offset = mr.offset
    moras_out = []
    for m, a, t in zip(lyr.moras, mr.assigns, times):
        moras_out.append({
            "id": m.id, "kana": m.kana, "romaji": m.romaji, "kind": m.kind,
            "text": "".join(lyr.chars[i] for i in m.char_idx), "char_idx": m.char_idx,
            "line": m.line, "seg": m.seg,
            "sung": not a.skipped, "start": t.start, "end": t.end,
            "source": t.source, "end_kind": t.end_kind,
            "midi_start": t.midi_start, "ctc_start": t.ctc_start,
            "note_ids": a.note_ids, "share_index": a.share_index, "share_count": a.share_count,
            "phrase": t.phrase, "split_estimated": m.split_estimated,
        })
    # 文字
    char_moras: dict[int, list[int]] = defaultdict(list)
    for m in lyr.moras:
        for ci in m.char_idx:
            char_moras[ci].append(m.id)
    chars_out = []
    line_of = {}
    for li, (a, b) in enumerate(lyr.lines):
        for ci in range(a, b):
            line_of[ci] = li
    for ci, ch in enumerate(lyr.chars):
        mids = char_moras.get(ci, [])
        sung = [moras_out[k] for k in mids if moras_out[k]["sung"]]
        entry = {
            "index": ci, "char": ch, "line": line_of.get(ci, -1),
            "pronounceable": bool(mids), "sung": bool(sung),
            "start": min(x["start"] for x in sung) if sung else None,
            "end": max(x["end"] for x in sung) if sung else None,
            "mora_ids": mids,
            "split_estimated": any(lyr.moras[k].split_estimated for k in mids),
        }
        if sung:
            entry["confidence"] = float(np.mean([
                1.0 if x["source"] == "midi" else max(0.0, 1.0 - abs((x["ctc_start"] or x["midi_start"]) - x["midi_start"]) / 0.3)
                for x in sung]))
        chars_out.append(entry)
    notes_out = []
    note_moras: dict[int, list[int]] = defaultdict(list)
    for m, a in zip(lyr.moras, mr.assigns):
        for n in a.note_ids:
            note_moras[n].append(m.id)
    for n in notes:
        notes_out.append({"id": n.id, "start": n.start + offset, "end": n.end + offset, "pitch": n.pitch,
                          "phrase": n.phrase, "midi_start": n.start, "mora_ids": note_moras.get(n.id, [])})
    # 行・フレーズ集計
    lines_out = []
    for li, (a, b) in enumerate(lyr.lines):
        sung = [x for x in moras_out if x["line"] == li and x["sung"]]
        lines_out.append({"line": li, "text": lyr.text[a:b], "char_start": a, "char_end": b,
                          "start": min(x["start"] for x in sung) if sung else None,
                          "end": max(x["end"] for x in sung) if sung else None,
                          "n_moras": sum(1 for x in moras_out if x["line"] == li), "n_sung": len(sung)})
    phrases_out = []
    for ph in sorted(set(n.phrase for n in notes)):
        ns = [x for x in notes_out if x["phrase"] == ph]
        ms = [x for x in moras_out if x["sung"] and x["phrase"] == ph]
        phrases_out.append({"phrase": ph, "start": ns[0]["start"], "end": ns[-1]["end"], "n_notes": len(ns),
                            "text": "".join(x["text"] for x in ms), "mora_ids": [x["id"] for x in ms]})
    return {
        "meta": {**meta, "offset": offset, "residual_median": mr.residual_median, "residual_p90": mr.residual_p90,
                 "residual_over150": mr.residual_over150, "drift_slope": mr.drift_slope,
                 "n_mora_skipped": mr.n_mora_skipped, "n_note_skipped": mr.n_note_skipped, "dp_cost": mr.cost},
        "text": lyr.text, "chars": chars_out, "moras": moras_out, "notes": notes_out,
        "lines": lines_out, "phrases": phrases_out,
    }


def write_peaks(wav16k: np.ndarray, path: Path, px_per_sec: int = 100) -> None:
    n = 16000 // px_per_sec
    T = len(wav16k) // n
    x = wav16k[:T * n].reshape(T, n)
    peaks = {"px_per_sec": px_per_sec, "min": np.round(x.min(axis=1), 3).tolist(),
             "max": np.round(x.max(axis=1), 3).tolist()}
    path.write_text(json.dumps(peaks))


def decode_audio(src: Path, dst: Path) -> None:
    """再生用 WAV (44.1kHz stereo) を生成. 解析と同じデコーダ (utalign.audio) なので時刻が一致する."""
    from .audio import write_playback_wav
    write_playback_wav(src, dst)


def write_click(wav16k: np.ndarray, starts: list[float], path: Path, freq: float = 1500.0, dur: float = 0.02) -> None:
    import soundfile as sf
    sr = 16000
    y = wav16k.copy() * 0.6
    L = int(dur * sr)
    t = np.arange(L) / sr
    click = (np.sin(2 * np.pi * freq * t) * np.exp(-t * 200)).astype(np.float32)
    for s in starts:
        k = int(round(s * sr))
        if 0 <= k < len(y) - L:
            y[k:k + L] += click * 0.8
    sf.write(str(path), np.clip(y, -1, 1), sr)
