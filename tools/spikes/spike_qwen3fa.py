"""Issue #1 検証スパイク: Qwen3-ForcedAligner (mlx-audio) を任意ユニットで走らせ MMS_FA と比較する."""
import json, sys, time, argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import mlx.core as mx

"""
使い方 (Issue #1 の検証スパイク。torch 不要の別 venv で実行する):
  python3.13 -m venv qvenv && qvenv/bin/pip install mlx-audio librosa soundfile "fugashi[unidic-lite]" mido
  PYTHONPATH=. qvenv/bin/python tools/spike_qwen3fa.py --root . --gran unit_hira --scope section
前提: `align` 実行済みで out/result.json (MMS_FA の ctc_start と MIDI 割当) と work/vocal16k.wav があること。
"""
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".", help="リポジトリのルート (歌詞 txt, out/result.json, work/vocal16k.wav がある場所)")
ap.add_argument("--lyrics", default="無限☆おもちゃ箱歌詞.txt")
ap.add_argument("--gran", default="unit", choices=["unit", "mora", "char", "line", "unit_hira", "seg"])
ap.add_argument("--scope", default="song", choices=["song", "section", "window"])
ap.add_argument("--win", type=float, default=20.0)
ap.add_argument("--model", default="mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
ap.add_argument("--out", default=None)
a = ap.parse_args()
M = Path(a.root).resolve(); WT = M
sys.path.insert(0, str(WT))
from utalign.lyrics import parse_lyrics

text = (M / a.lyrics).read_text(encoding="utf-8")
lyr = parse_lyrics(text, WT / "readings.override.txt")
res = json.loads((M / "out/result.json").read_text(encoding="utf-8"))
rmoras = res["moras"]; assert len(rmoras) == len(lyr.moras)
wav, sr = sf.read(str(M / "work/vocal16k.wav"), dtype="float32"); assert sr == 16000
if wav.ndim > 1: wav = wav.mean(axis=1)

def kata2hira(s): return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in s)

# ---- ユニット列を作る: [(text, first_mora_id)]
units = []
if a.gran in ("unit", "unit_hira"):          # MMS の CTC ユニットと同じ区切り (ー/っ は前に畳む)
    for u in lyr.units:
        if u.is_star or not u.mora_ids: continue
        t = "".join(lyr.moras[i].kana for i in u.mora_ids)
        units.append((kata2hira(t) if a.gran == "unit_hira" else t, u.mora_ids[0]))
elif a.gran == "mora":                        # 全モーラ (ー/っ/ん も独立)
    for m in lyr.moras: units.append((m.kana, m.id))
elif a.gran == "char":                        # 元テキストの発音文字 1 つずつ (漢字はそのまま)
    cur = None
    for m in lyr.moras:
        key = tuple(m.char_idx)
        if key != cur:
            units.append(("".join(lyr.chars[c] for c in m.char_idx), m.id)); cur = key
elif a.gran == "seg":                         # 行内の空白区切り
    cur = None
    for m in lyr.moras:
        key = (m.line, m.seg)
        if key != cur:
            units.append(("".join(lyr.chars[c] for c in m.char_idx), m.id)); cur = key
        else:
            t, f = units[-1]; units[-1] = (t + "".join(lyr.chars[c] for c in m.char_idx if c not in set(x for mm in lyr.moras[f:m.id] for x in mm.char_idx)), f)
elif a.gran == "line":
    cur = None
    for m in lyr.moras:
        if m.line != cur:
            s, e = lyr.lines[m.line]
            units.append(("".join(c for i, c in enumerate(lyr.chars[s:e], s) if i not in lyr.unsung_chars and not c.isspace()), m.id)); cur = m.line
print(f"[units] gran={a.gran} n={len(units)} e.g. {[t for t,_ in units[:12]]}")

from mlx_audio.stt import load
t0 = time.time()
model = load(a.model)
print(f"[model] loaded in {time.time()-t0:.1f}s")

def align(audio_np, word_list):
    """mlx-audio generate() と同じ処理を、ユニット列を直接与えて実行. (start_ms, end_ms, raw, conf) を返す."""
    input_features, feature_attention_mask, n_audio = model._preprocess_audio(audio_np)
    txt = "<timestamp><timestamp>".join(word_list) + "<timestamp><timestamp>"
    txt = "<|audio_start|>" + "<|audio_pad|>" * n_audio + "<|audio_end|>" + txt
    ids = mx.array(model._tokenizer.encode(txt, return_tensors="np", add_special_tokens=False))
    logits = model(ids, input_features=input_features, feature_attention_mask=feature_attention_mask)
    mx.eval(logits)
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    out = np.array(mx.argmax(logits, axis=-1))[0]; pmax = np.array(mx.max(probs, axis=-1))[0]
    mask = np.array(ids)[0] == model.config.timestamp_token_id
    raw = out[mask] * model.config.timestamp_segment_time
    conf = pmax[mask]
    fixed = np.array(model.aligner_processor.fix_timestamp(raw))
    mx.clear_cache()
    return fixed, raw, conf, n_audio

M_ = len(lyr.moras)
q_start = np.full(M_, np.nan); q_end = np.full(M_, np.nan); q_conf = np.full(M_, np.nan); n_fix = 0
t1 = time.time()
if a.scope == "song":
    fixed, raw, conf, n_audio = align(wav, [t for t, _ in units])
    n_fix = int((fixed != raw).sum())
    for k, (_, mid) in enumerate(units):
        q_start[mid] = fixed[2*k] / 1000; q_end[mid] = fixed[2*k+1] / 1000; q_conf[mid] = conf[2*k]
    print(f"[align] whole song {len(wav)/16000:.1f}s, audio tokens={n_audio}, units={len(units)}, fixed={n_fix}/{len(raw)} ts, {time.time()-t1:.1f}s, peak mem {mx.get_peak_memory()/1e9:.2f}GB")
elif a.scope == "window":
    # 固定長の窓 (win 秒) に、MIDI 割当時刻がその窓に入るユニットを渡す (窓境界は最寄りの 0.3s 以上の休符に寄せる)
    mids = [mid for _, mid in units if rmoras[mid]["midi_start"] is not None]
    cur = 0; nwin = 0
    while cur < len(mids):
        t0w = rmoras[mids[cur]]["midi_start"]; k = cur
        while k + 1 < len(mids) and rmoras[mids[k + 1]]["midi_start"] - t0w < a.win: k += 1
        # 窓末尾を休符に寄せる: k から戻って直前モーラとの間隔が 0.3s 以上の点
        j = k
        while j > cur + 3 and rmoras[mids[j]]["midi_start"] - rmoras[mids[j - 1]]["end"] < 0.3: j -= 1
        if j > cur + 3: k = j - 1
        sel = [(t, mid) for t, mid in units if mid in set(mids[cur:k + 1])]
        t1w = rmoras[mids[k]]["end"]
        a0 = max(0, int((t0w - 0.5) * 16000)); a1 = min(len(wav), int((t1w + 0.5) * 16000))
        fixed, raw, conf, n_audio = align(wav[a0:a1], [t for t, _ in sel])
        n_fix += int((fixed != raw).sum()); nwin += 1
        for kk, (_, mid) in enumerate(sel):
            q_start[mid] = fixed[2*kk] / 1000 + a0 / 16000; q_end[mid] = fixed[2*kk+1] / 1000 + a0 / 16000; q_conf[mid] = conf[2*kk]
        cur = k + 1
    print(f"[align] {nwin} windows of <= {a.win}s, fixed={n_fix} ts, {time.time()-t1:.1f}s, peak mem {mx.get_peak_memory()/1e9:.2f}GB")
else:
    # MIDI セクション (result.json のノートのフレーズ間隔 > 5s を境界とする) ごとに、そこに割当済みのモーラを渡す
    notes = res["notes"]; bounds = []; s = notes[0]["start"]
    for p, n in zip(notes, notes[1:]):
        if n["start"] - p["end"] > 5.0: bounds.append((s, p["end"])); s = n["start"]
    bounds.append((s, notes[-1]["end"]))
    for (bs, be) in bounds:
        ms = [(t, mid) for t, mid in units if rmoras[mid]["midi_start"] is not None and bs - 0.5 <= rmoras[mid]["midi_start"] <= be + 0.5]
        a0 = max(0, int((bs - 1.0) * 16000)); a1 = min(len(wav), int((be + 1.0) * 16000))
        fixed, raw, conf, n_audio = align(wav[a0:a1], [t for t, _ in ms])
        n_fix += int((fixed != raw).sum())
        for k, (_, mid) in enumerate(ms):
            q_start[mid] = fixed[2*k] / 1000 + a0 / 16000; q_end[mid] = fixed[2*k+1] / 1000 + a0 / 16000; q_conf[mid] = conf[2*k]
        print(f"[align] section {bs:.1f}-{be:.1f}s units={len(ms)} fixed={int((fixed != raw).sum())}/{len(raw)} tokens={n_audio}")
    print(f"[align] {len(bounds)} sections, {time.time()-t1:.1f}s, peak mem {mx.get_peak_memory()/1e9:.2f}GB")

# ---- 指標: 1:1 先頭モーラ (share_index==0, note あり) で |onset - midi| を中央値オフセット除去後に評価
def metrics(start_arr, label):
    idx = [i for i, m in enumerate(rmoras) if m["note_ids"] and m["share_index"] == 0 and not np.isnan(start_arr[i])]
    c = np.array([start_arr[i] for i in idx]); o = np.array([rmoras[i]["midi_start"] for i in idx])
    off = float(np.median(c - o)); r = np.abs(c - o - off)
    print(f"[{label:9s}] n={len(idx)} offset={off*1000:+.0f}ms median={np.median(r)*1000:.0f}ms p90={np.percentile(r,90)*1000:.0f}ms "
          f">150ms={np.mean(r>0.15)*100:.1f}% >300ms={np.mean(r>0.3)*100:.1f}% max={r.max()*1000:.0f}ms")
    return idx, r
mms = np.array([m["ctc_start"] if m["ctc_start"] is not None else np.nan for m in rmoras])
metrics(mms, "MMS_FA")
idx, r = metrics(q_start, "Qwen3-FA")
both = [i for i in idx if not np.isnan(mms[i])]
d = np.array([q_start[i] - mms[i] for i in both]); d -= np.median(d)
print(f"[Qwen-MMS ] n={len(both)} |diff| median={np.median(np.abs(d))*1000:.0f}ms p90={np.percentile(np.abs(d),90)*1000:.0f}ms >150ms={np.mean(np.abs(d)>0.15)*100:.1f}%")
mono = np.array([q_start[i] for i in range(M_) if not np.isnan(q_start[i])])
print(f"[mono     ] non-monotonic starts: {int((np.diff(mono) < 0).sum())}, zero-length units: {int((q_end[~np.isnan(q_end)] <= q_start[~np.isnan(q_start)]).sum())}, conf median={np.nanmedian(q_conf):.2f}")
# 行ごとの粗さ: 行頭モーラの残差
worst = sorted(zip(r, idx), reverse=True)[:8]
print("[worst   ]", [(rmoras[i]["kana"], rmoras[i]["line"], f"{rr*1000:.0f}ms", f"q={q_start[i]:.2f}", f"midi={rmoras[i]['midi_start']:.2f}") for rr, i in worst])
if a.out:
    json.dump({"gran": a.gran, "scope": a.scope, "start": q_start.tolist(), "end": q_end.tolist(), "conf": q_conf.tolist()}, open(a.out, "w"))
