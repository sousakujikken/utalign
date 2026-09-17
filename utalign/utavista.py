"""UTAVISTA の AI 歌詞タイミング中間 JSON (lyrics-timing/1.0) へのエクスポート.

正本: utavista2-refactor/src/lyricsTiming/{types,validate,normalize}.ts
  - phrase(行) → word(空白区切りセグメント) → char(コードポイント 1 文字)
  - 各階層 endMs >= startMs + 10、子は親区間に包含、兄弟は startMs 非降順
  - chars 連結 = word.text、words 連結 = phrase.text (空白は無視して比較)
歌われない文字 (記号・括弧内・未割当) は隣接文字の端 10ms を借りて区間を作る
(発音時刻としては意味を持たないので `sung: false` を付加フィールドで示す)。
"""
from __future__ import annotations

import re
from typing import Any

SCHEMA_VERSION = "lyrics-timing/1.0"
MIN_MS = 10
WS_RE = re.compile(r"[\s　]")


def _ms(x: float) -> int:
    return int(round(x * 1000))


KANA_RE = re.compile(r"[\u3041-\u309F\u30A0-\u30FF]")


def _kata_to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in s)


def result_to_lyrics_timing(result: dict[str, Any], keep_spaces_in_phrase: bool = True, ruby: bool = False) -> dict[str, Any]:
    chars = result["chars"]
    moras = result["moras"]
    phrases: list[dict[str, Any]] = []
    for ln in result["lines"]:
        a, b = ln["char_start"], ln["char_end"]
        line_chars = [c for c in chars[a:b] if c["char"] != "\n"]
        text_raw = "".join(c["char"] for c in line_chars)
        if not text_raw.strip() and not text_raw.replace("　", "").strip():
            continue
        # 単語 = 空白区切り
        words_raw: list[list[dict[str, Any]]] = []
        cur: list[dict[str, Any]] = []
        for c in line_chars:
            if WS_RE.match(c["char"]):
                if cur:
                    words_raw.append(cur); cur = []
            else:
                cur.append(c)
        if cur:
            words_raw.append(cur)
        # 行全体で歌われた文字が無ければスキップ (タグ行など)
        if not any(c["sung"] for w in words_raw for c in w):
            continue
        # 文字の時刻: 同一モーラ集合を共有する連続文字は均等按分 (拗音「きゃ」等)
        timed: dict[int, tuple[int, int]] = {}
        for w in words_raw:
            i = 0
            while i < len(w):
                c = w[i]
                if not c["sung"]:
                    i += 1; continue
                j = i
                while j + 1 < len(w) and w[j + 1]["sung"] and w[j + 1]["mora_ids"] == c["mora_ids"]:
                    j += 1
                s, e = _ms(c["start"]), _ms(c["end"])
                n = j - i + 1
                if e - s < MIN_MS * n:
                    e = s + MIN_MS * n
                for k in range(n):
                    cs = s + (e - s) * k // n
                    ce = s + (e - s) * (k + 1) // n
                    timed[w[i + k]["index"]] = (cs, ce)
                i = j + 1
        # 歌われない文字: 直前の歌唱文字の末尾 10ms (行頭なら直後の先頭 10ms) を借りる
        all_line = [c for w in words_raw for c in w]
        for k, c in enumerate(all_line):
            if c["index"] in timed:
                continue
            prev = next((timed[x["index"]] for x in reversed(all_line[:k]) if x["index"] in timed), None)
            nxt = next((timed[x["index"]] for x in all_line[k + 1:] if x["index"] in timed), None)
            if prev is not None:
                timed[c["index"]] = (max(prev[0], prev[1] - MIN_MS), prev[1])
            elif nxt is not None:
                timed[c["index"]] = (nxt[0], nxt[0] + MIN_MS)
        # 非降順の保証 (借用文字が直前より早く始まることはないが、丸めの防御)
        words_out = []
        last_start = -10**9
        for w in words_raw:
            cs_out = []
            for c in w:
                s, e = timed[c["index"]]
                if s < last_start:
                    s = last_start
                if e < s + MIN_MS:
                    e = s + MIN_MS
                last_start = s
                entry: dict[str, Any] = {"char": c["char"], "startMs": s, "endMs": e}
                if not c["sung"]:
                    entry["sung"] = False
                elif ruby and not KANA_RE.match(c["char"]) and c["mora_ids"]:
                    # 漢字にはモーラの読みをルビとして付与 (importer の char.ruby)
                    entry["ruby"] = _kata_to_hira("".join(moras[k]["kana"] for k in c["mora_ids"]))
                cs_out.append(entry)
            words_out.append({
                "text": "".join(c["char"] for c in w),
                "startMs": min(c["startMs"] for c in cs_out),
                "endMs": max(c["endMs"] for c in cs_out),
                "chars": cs_out,
            })
        ptext = text_raw.strip("　 \t") if keep_spaces_in_phrase else "".join(w["text"] for w in words_out)
        phrases.append({
            "text": ptext,
            "startMs": min(w["startMs"] for w in words_out),
            "endMs": max(w["endMs"] for w in words_out),
            "words": words_out,
        })
    return {"schemaVersion": SCHEMA_VERSION, "phrases": phrases}


def self_check(doc: dict[str, Any]) -> list[str]:
    """validate.ts と同じ制約を Python 側でも機械検査する (ゼロ件なら OK)."""
    issues = []
    strip = lambda s: WS_RE.sub("", s)
    prev_p = None
    for pi, p in enumerate(doc["phrases"]):
        if p["endMs"] < p["startMs"] + MIN_MS: issues.append(f"phrases[{pi}] duration")
        if prev_p is not None and p["startMs"] < prev_p: issues.append(f"phrases[{pi}] unsorted")
        prev_p = p["startMs"]
        if strip("".join(w["text"] for w in p["words"])) != strip(p["text"]): issues.append(f"phrases[{pi}] concat")
        prev_w = None
        for wi, w in enumerate(p["words"]):
            if w["endMs"] < w["startMs"] + MIN_MS: issues.append(f"phrases[{pi}].words[{wi}] duration")
            if w["startMs"] < p["startMs"] or w["endMs"] > p["endMs"]: issues.append(f"phrases[{pi}].words[{wi}] containment")
            if prev_w is not None and w["startMs"] < prev_w: issues.append(f"phrases[{pi}].words[{wi}] unsorted")
            prev_w = w["startMs"]
            if strip("".join(c["char"] for c in w["chars"])) != strip(w["text"]): issues.append(f"phrases[{pi}].words[{wi}] concat")
            prev_c = None
            for ci, c in enumerate(w["chars"]):
                if len(c["char"]) != 1: issues.append(f"phrases[{pi}].words[{wi}].chars[{ci}] char")
                if c["endMs"] < c["startMs"] + MIN_MS: issues.append(f"phrases[{pi}].words[{wi}].chars[{ci}] duration")
                if c["startMs"] < w["startMs"] or c["endMs"] > w["endMs"]: issues.append(f"phrases[{pi}].words[{wi}].chars[{ci}] containment")
                if prev_c is not None and c["startMs"] < prev_c: issues.append(f"phrases[{pi}].words[{wi}].chars[{ci}] unsorted")
                prev_c = c["startMs"]
                if c["endMs"] - c["startMs"] > 10000: issues.append(f"phrases[{pi}].words[{wi}].chars[{ci}] outlier")
    return issues
