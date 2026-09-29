"""MIDI からボーカルノート列を抽出する."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mido


@dataclass
class Note:
    id: int
    start: float      # 秒 (MIDI 時刻, オフセット未適用)
    end: float
    pitch: int
    track: int
    phrase: int = -1  # 休符で区切ったフレーズ番号


DRUM_NAME_RE = __import__("re").compile(r"drum|perc|kick|snare|hihat|hat", __import__("re").I)


def load_notes(path: Path, min_rest: float = 0.2, tracks: list[int] | None = None,
               exclude_drums: bool = True, log: Callable[[str], None] = print) -> list[Note]:
    """tracks: 使うトラック番号 (None = 全部). exclude_drums: 名前が drum 系 / チャンネル 10 のトラックを除外."""
    try:
        m = mido.MidiFile(str(path))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"MIDI ファイルを読めません: {path.name} ({e})") from e
    skip: set[int] = set()
    for ti, tr in enumerate(m.tracks):
        if tracks is not None and ti not in tracks:
            skip.add(ti); continue
        if exclude_drums:
            chans = {msg.channel for msg in tr if hasattr(msg, "channel")}
            if DRUM_NAME_RE.search(tr.name or "") or (chans and chans <= {9}):
                skip.add(ti)
    if skip:
        log(f"[midi]   skipping tracks {sorted(skip)}: " + ", ".join(f"{ti}:'{m.tracks[ti].name}'" for ti in sorted(skip)))
    # テンポマップ (全トラック統合)
    events = []
    for ti, tr in enumerate(m.tracks):
        tick = 0
        for msg in tr:
            tick += msg.time
            events.append((tick, ti, msg))
    events.sort(key=lambda e: (e[0], 0 if e[2].type == 'set_tempo' else 1))
    tempo = 500000
    last_tick = 0
    sec = 0.0
    raw = []
    on: dict[tuple[int, int], list[float]] = {}
    for tick, ti, msg in events:
        sec += mido.tick2second(tick - last_tick, m.ticks_per_beat, tempo)
        last_tick = tick
        if msg.type == 'set_tempo':
            tempo = msg.tempo
        elif ti in skip:
            continue
        elif msg.type == 'note_on' and msg.velocity > 0:
            on.setdefault((ti, msg.note), []).append(sec)
        elif msg.type in ('note_off', 'note_on'):
            key = (ti, msg.note)
            if on.get(key):
                s = on[key].pop(0)
                if sec > s:
                    raw.append((s, sec, msg.note, ti))
    raw.sort()
    # 単旋律化: 次のノート onset で前のノートを切る
    notes: list[Note] = []
    for s, e, p, ti in raw:
        if notes and s < notes[-1].end - 1e-4:
            if s - notes[-1].start < 0.03:
                # ほぼ同時 (和音): 高い方を残す
                if p > notes[-1].pitch:
                    notes[-1].pitch = p
                notes[-1].end = max(notes[-1].end, e)
                continue
            notes[-1].end = s
        notes.append(Note(id=len(notes), start=s, end=e, pitch=p, track=ti))
    if not notes:
        raise ValueError("MIDI にノートがありません (ドラム系トラックを除外した結果なら --keep-drums / --midi-tracks を確認してください)")
    # フレーズ分割
    ph = 0
    for i, n in enumerate(notes):
        if i > 0 and n.start - notes[i - 1].end >= min_rest:
            ph += 1
        n.phrase = ph
    return notes
