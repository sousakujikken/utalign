"""解析パイプライン (CLI / GUI サーバ / UTAVISTA 連携の共通実装).

run_align():    歌詞・音声・MIDI -> out/result.json (+ 任意で audio.wav, click.wav, peaks.json, ビューア)
run_export():   result.json -> UTAVISTA lyrics-timing/1.0 JSON
run_utavista(): 上記 2 つを一括で行い、UTAVISTA が読む JSON と要約を返す (再生用 WAV 等は作らない)

進捗は progress(event) に dict で通知する:
  {"event": "stage", "stage": <STAGES のいずれか>, "elapsed": 秒}
ログ行は log(str) に渡す (既定は print)。
"""
from __future__ import annotations

import json
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from . import config as _cfg

WEB_DIR = Path(__file__).resolve().parent / "web"
VIEWER_FILES = ("viewer.html", "viewer.js", "style.css")

Progress = Callable[[dict[str, Any]], None]

# 解析ステージ (UTAVISTA 側の進捗表示と対応させる)
STAGES = ("model", "lyrics", "midi", "audio", "emission", "align", "match", "write", "export")


@dataclass
class AlignOptions:
    model: str = _cfg.DEFAULT_MODEL
    device: str = "auto"
    star: bool = False
    min_rest: float = 0.2
    midi_tracks: Optional[list[int]] = None
    keep_drums: bool = False
    readings: Optional[Path] = None
    models_dir: Optional[Path] = None
    # 再生用 WAV / クリック音 / 波形ピーク / ビューアを out/ に書くか (GUI 用。UTAVISTA 連携では不要)
    write_playback: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def _noop(_: dict[str, Any]) -> None:
    pass


def run_align(audio: Path, lyrics: Path, midi: Path, out: Path, work: Path,
              opt: AlignOptions = AlignOptions(), log: Callable[[str], None] = print,
              progress: Progress = _noop) -> dict[str, Any]:
    from .aligners import make_aligner
    from .audio import load_audio_16k
    from .lyrics import build_ctc_units, parse_lyrics
    from .midi import load_notes
    from .output import build_result, decode_audio, write_click, write_peaks
    from .timing import resolve_times, run_match, xcorr_offset

    t0 = time.time()

    def stage(name: str) -> None:
        progress({"event": "stage", "stage": name, "elapsed": round(time.time() - t0, 2)})

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    stage("model")
    aligner = make_aligner(opt.model, models_dir=opt.models_dir, device=opt.device, star=opt.star)
    log(f"[aligner] {aligner.describe()}")
    stage("lyrics")
    text = Path(lyrics).read_text(encoding="utf-8")
    lyr = parse_lyrics(text, opt.readings if opt.readings and Path(opt.readings).exists() else None)
    lyr.units = build_ctc_units(lyr.moras, aligner.mora_tokens)
    log(f"[lyrics] {len(lyr.moras)} moras, {len(lyr.units)} ctc units, {len(lyr.unsung_chars)} unsung chars")
    if not lyr.moras:
        raise ValueError("歌詞に発音できる文字がありません")
    stage("midi")
    notes = load_notes(Path(midi), min_rest=opt.min_rest, tracks=opt.midi_tracks, exclude_drums=not opt.keep_drums, log=log)
    log(f"[midi]   {len(notes)} notes, {notes[-1].phrase + 1} phrases, {notes[0].start:.2f}-{notes[-1].end:.2f}s")
    stage("audio")
    wav = load_audio_16k(Path(audio))
    log(f"[audio]  {len(wav) / 16000:.2f}s")
    stage("emission")
    em = aligner.compute_emissions(wav, work)
    log(f"[ctc]    emission {em.shape} ({time.time() - t0:.1f}s)")
    stage("align")
    spans = aligner.align(em, lyr.units)
    M = len(lyr.moras)
    ctc_start = np.full(M, np.nan); ctc_score = np.full(M, np.nan)
    for u, sp in zip(lyr.units, spans):
        if sp is None or u.is_star or not u.mora_ids:
            continue
        ctc_start[u.mora_ids[0]] = sp.start; ctc_score[u.mora_ids[0]] = sp.score
    log(f"[ctc]    aligned {int((~np.isnan(ctc_start)).sum())} moras with ctc onsets ({time.time() - t0:.1f}s)")
    stage("match")
    off0 = xcorr_offset(wav, notes)
    mr = run_match(lyr.moras, ctc_start, ctc_score, notes, off0, wav)
    log(f"[match]  xcorr offset={off0:+.3f}s -> refined offset={mr.offset:+.3f}s "
        f"(diff {abs(off0 - mr.offset) * 1000:.0f}ms), drift slope={mr.drift_slope:.5f}")
    log(f"[match]  residual median={mr.residual_median * 1000:.0f}ms p90={mr.residual_p90 * 1000:.0f}ms "
        f">150ms={mr.residual_over150 * 100:.1f}%  mora_skipped={mr.n_mora_skipped} note_skipped={mr.n_note_skipped} "
        f"(unvoiced notes={mr.n_note_unvoiced})")
    warnings = []
    if abs(off0 - mr.offset) > 0.08:
        warnings.append("xcorr offset と CTC 残差オフセットが 80ms 以上ずれています。GUI で確認してください")
    if abs(mr.drift_slope - 1.0) > 0.001:
        warnings.append("MIDI と音声にドリフトがあります (slope != 1)。定数オフセットでは不十分かもしれません")
    for w in warnings:
        log("[warn]   " + w)
    times = resolve_times(lyr.moras, mr.assigns, notes, ctc_start, mr.offset)
    meta = {"audio": str(audio), "lyrics": str(lyrics), "midi": str(midi), "xcorr_offset": off0,
            "model": aligner.model_id, "vocab": aligner.vocab_kind,
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "warnings": warnings}
    result = build_result(lyr, notes, mr, times, meta)
    # 繰り返し行の一致率
    pat = defaultdict(list)
    for ln in result["lines"]:
        key = ln["text"].strip()
        if not key:
            continue
        ms = [x for x in result["moras"] if x["line"] == ln["line"]]
        pat[key].append("".join("x" if not x["sung"] else f"{len(x['note_ids'])}{x['share_index']}" for x in ms))
    tot = sum(len(v) - 1 for v in pat.values() if len(v) > 1)
    mism = sum(1 for v in pat.values() if len(v) > 1 for p in v[1:] if p != v[0])
    result["meta"]["repeated_lines"] = tot; result["meta"]["repeat_mismatches"] = mism
    log(f"[metric] repeated lines: {tot}, assignment pattern mismatches: {mism}")
    stage("write")
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    if opt.write_playback:
        write_peaks(wav, out / "peaks.json")
        decode_audio(Path(audio), out / "audio.wav")
        write_click(wav, [t.start for t in times if t.start is not None], out / "click.wav")
        for name in VIEWER_FILES:
            f = WEB_DIR / name
            if f.exists():
                shutil.copy(f, out / ("index.html" if name == "viewer.html" else name))
    result["meta"]["elapsed"] = time.time() - t0
    log(f"[done]   {out / 'result.json'}  ({time.time() - t0:.1f}s)")
    return result["meta"]


def run_export(result_path: Path, out_path: Path, ruby: bool = True, strip_spaces: bool = False,
               log: Callable[[str], None] = print) -> dict[str, Any]:
    from .utavista import result_to_lyrics_timing, self_check
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    doc = result_to_lyrics_timing(result, keep_spaces_in_phrase=not strip_spaces, ruby=ruby)
    issues = self_check(doc)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    n_w = sum(len(p["words"]) for p in doc["phrases"])
    n_c = sum(len(w["chars"]) for p in doc["phrases"] for w in p["words"])
    log(f"[export] {out_path}: {len(doc['phrases'])} phrases, {n_w} words, {n_c} chars; self-check issues: {len(issues)}")
    for i in issues[:20]:
        log("   " + i)
    return {"phrases": len(doc["phrases"]), "words": n_w, "chars": n_c, "issues": issues, "path": str(out_path)}


UTAVISTA_META_KEYS = ("generated", "model", "vocab", "offset", "residual_median", "residual_p90", "residual_over150",
                      "n_mora_skipped", "n_note_skipped", "repeated_lines", "repeat_mismatches", "warnings", "elapsed")


def run_utavista(audio: Path, lyrics: Path, midi: Path, out_json: Path, work: Path,
                 opt: AlignOptions = AlignOptions(), ruby: bool = True, strip_spaces: bool = False,
                 out_dir: Optional[Path] = None, log: Callable[[str], None] = print,
                 progress: Progress = _noop) -> dict[str, Any]:
    """UTAVISTA 連携の一括処理: align → export-utavista。result.json は out_dir (既定 work/out) に置く."""
    out_dir = Path(out_dir) if out_dir else Path(work) / "out"
    meta = run_align(audio, lyrics, midi, out_dir, work, opt, log=log, progress=progress)
    progress({"event": "stage", "stage": "export", "elapsed": round(meta.get("elapsed", 0.0), 2)})
    info = run_export(out_dir / "result.json", Path(out_json), ruby=ruby, strip_spaces=strip_spaces, log=log)
    return {"outJson": str(out_json), "resultJson": str(out_dir / "result.json"),
            "phrases": info["phrases"], "words": info["words"], "chars": info["chars"], "selfCheckIssues": info["issues"],
            "meta": {k: meta.get(k) for k in UTAVISTA_META_KEYS}}


def run_utavista_validator(utavista_dir: Path, timing_json: Path, lyrics_txt: Path, duration_ms: int,
                           log: Callable[[str], None] = print) -> dict[str, Any]:
    """utavista2 リポジトリの検証器 (tools/validate_with_utavista.ts) を tsx で実行する."""
    import subprocess
    script = Path(__file__).resolve().parent / "data" / "validate_with_utavista.ts"
    tsx = Path(utavista_dir) / "node_modules" / ".bin" / "tsx"
    if not tsx.exists():
        return {"ok": False, "output": f"tsx が見つかりません: {tsx}"}
    if not script.exists():
        return {"ok": False, "output": f"検証スクリプトが見つかりません: {script}"}
    cmd = [str(tsx), str(script), str(timing_json), str(lyrics_txt), str(duration_ms)]
    log("[validate] " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(utavista_dir), capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    log(out)
    return {"ok": r.returncode == 0, "output": out}
