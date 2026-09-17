"""Apache-2.0 の日本語 CTC モデル (HF) で MMS_FA と同じ強制アライメントを行い、残差指標を比較する (Issue #1 代替案の検証).

使い方 (torch / torchaudio / transformers / fugashi が入った venv で):
  PYTHONPATH=. python tools/spike_hf_ctc.py --model prj-beatrice/japanese-hubert-base-phoneme-ctc-v4
  PYTHONPATH=. python tools/spike_hf_ctc.py --model TKU410410103/hubert-large-japanese-asr --star max
前提: `align` 実行済みで out/result.json (MMS_FA の ctc_start と MIDI 割当) と work/vocal16k.wav があること。
語彙が音素 (cl/N あり) ならローマ字→音素、ひらがな/カタカナならモーラのかなをそのままトークンにする。
"""
import argparse, json, sys, time, hashlib
from pathlib import Path
import numpy as np, soundfile as sf, torch, torchaudio.functional as F
from transformers import AutoModelForCTC, AutoFeatureExtractor, AutoConfig

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".", help="リポジトリのルート (歌詞, out/result.json, work/vocal16k.wav)")
ap.add_argument("--code", default=".", help="utalign パッケージのある場所")
ap.add_argument("--lyrics", default="無限☆おもちゃ箱歌詞.txt")
ap.add_argument("--audio", default="work/vocal16k.wav", help="16kHz mono wav (root 相対)")
ap.add_argument("--result", default="out/result.json", help="MMS_FA で align した result.json (root 相対)")
ap.add_argument("--model", required=True)
ap.add_argument("--star", default="none", choices=["none", "max", "lse"])
ap.add_argument("--star-penalty", type=float, default=1.0)
ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
ap.add_argument("--cache", default="emcache")
ap.add_argument("--out", default=None)
a = ap.parse_args()
M = Path(a.root); sys.path.insert(0, a.code)
from utalign.lyrics import parse_lyrics
lyr = parse_lyrics((M / a.lyrics).read_text(encoding="utf-8"), Path(a.code) / "readings.override.txt")
rm = json.loads((M / a.result).read_text())["moras"]
wav, sr = sf.read(str(M / a.audio), dtype="float32"); assert sr == 16000
if wav.ndim > 1: wav = wav.mean(axis=1).astype(np.float32)

# ---------------- emission (20s 窓 + 2s オーバーラップ, ctc.py と同じ)
SR, HOP = 16000, 320
def num_frames(n): return max(0, (n - 400) // HOP + 1)
cfg = AutoConfig.from_pretrained(a.model)
key = hashlib.sha1((a.model + str(len(wav))).encode()).hexdigest()[:12]
cache = Path(a.cache) / f"{key}.npy"
t0 = time.time()
if cache.exists():
    em = np.load(cache); print(f"[em] cached {em.shape}")
else:
    try:
        fe = AutoFeatureExtractor.from_pretrained(a.model)
    except Exception as e:
        from transformers import Wav2Vec2FeatureExtractor
        fe = Wav2Vec2FeatureExtractor(sampling_rate=16000, do_normalize=True, return_attention_mask=False); print("[em] default feature extractor:", str(e)[:80])
    model = AutoModelForCTC.from_pretrained(a.model).to(a.device).eval()
    n = len(wav); T = num_frames(n); chunk = int(20 * SR) // HOP * HOP; ov = int(2 * SR) // HOP * HOP
    em = None; pos = 0
    with torch.inference_mode():
        while pos < n:
            core_end = min(n, pos + chunk); s0 = max(0, pos - ov); s1 = min(n, core_end + ov)
            x = fe(wav[s0:s1], sampling_rate=16000, return_tensors="pt").input_values.to(a.device)
            lp = torch.log_softmax(model(x).logits[0].float(), dim=-1).cpu().numpy()
            if em is None: em = np.full((T, lp.shape[1]), -1e4, dtype=np.float32)
            f0 = s0 // HOP; c0 = pos // HOP; c1 = min(T, core_end // HOP if core_end < n else T)
            for g in range(c0, c1):
                k = g - f0
                if 0 <= k < lp.shape[0]: em[g] = lp[k]
            pos = core_end
    cache.parent.mkdir(exist_ok=True); np.save(cache, em)
    print(f"[em] {em.shape} computed on {a.device} in {time.time()-t0:.1f}s")

# ---------------- 語彙とユニットのトークン化
try:
    vocab = json.load(open(__import__("huggingface_hub").hf_hub_download(a.model, "vocab.json")))
except Exception:
    from transformers import AutoTokenizer
    vocab = AutoTokenizer.from_pretrained(a.model).get_vocab()
inv = {v: k for k, v in vocab.items()}
blank = cfg.pad_token_id if cfg.pad_token_id is not None else vocab.get("<pad>", vocab.get("[PAD]", vocab.get("PAD", vocab.get("<blk>", 0))))
def kata2hira(s): return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in s)
has_hira = "あ" in vocab; has_kata = "ア" in vocab; phon = ("cl" in vocab and "N" in vocab)
print(f"[vocab] {a.model}: V={len(vocab)} blank={blank}({inv[blank]!r}) hira={has_hira} kata={has_kata} phoneme={phon} frames={em.shape[0]}")
VOW = "aiueo"
def mora_tokens(m):
    if phon:
        if m.kind == "hatsuon": return ["N"]
        if m.kind == "sokuon": return ["cl"] if a.star != "none" or True else []
        if m.kind == "choon": return []           # 前モーラの母音に畳む (MMS と同じ扱い)
        r = m.romaji
        if r[-1] not in VOW: return [c for c in r if c in vocab]
        cons, v = r[:-1], r[-1]
        if not cons: return [v]
        if cons in vocab: return [cons, v]
        # ヘボン式 -> 音素表記の差 (例: 'shi'->'sh','i' は同じ; 'tsu'->'ts'; 'ji'->'j'); 見つからなければ 1 文字ずつ
        return [c for c in cons if c in vocab] + [v]
    s = m.kana
    if m.kind == "choon" and "ー" not in vocab: return []
    if has_hira and not has_kata: s = kata2hira(s)
    toks = []
    for ch in s:
        if ch in vocab: toks.append(ch)
        elif kata2hira(ch) in vocab: toks.append(kata2hira(ch))
        else: print("  [warn] no vocab for", ch, "in", m.kana)
    return toks
units = []   # (first_mora_id, [token ids], is_star)
for u in lyr.units:
    if u.is_star:
        if a.star != "none": units.append((None, ["*"], True))
        continue
    toks = []
    for mid in u.mora_ids: toks += mora_tokens(lyr.moras[mid])
    if not toks: continue
    units.append((u.mora_ids[0], toks, False))
# star 列を追加
V = em.shape[1]
if a.star != "none":
    nb = np.delete(em, blank, axis=1)
    star = (nb.max(axis=1) if a.star == "max" else np.logaddexp.reduce(nb, axis=1)) - a.star_penalty
    em = np.concatenate([em, star[:, None]], axis=1); star_id = V
targets = []; unit_tok_ranges = []
for mid, toks, is_star in units:
    ids = [star_id] if is_star else [vocab[t] for t in toks]
    unit_tok_ranges.append((len(targets), len(targets) + len(ids))); targets += ids
print(f"[units] {len(units)} units, {len(targets)} tokens, star={a.star}, e.g. {[ (lyr.moras[m].kana if m is not None else '*', t) for m, t, _ in units[:8]]}")
t1 = time.time()
lp = torch.from_numpy(em)[None]; tg = torch.tensor(targets)[None]
ali, scores = F.forced_align(lp, tg, blank=blank)
spans = F.merge_tokens(ali[0], scores[0].exp(), blank=blank)
assert len(spans) == len(targets), (len(spans), len(targets))
print(f"[align] forced_align {time.time()-t1:.1f}s")
M_ = len(lyr.moras); qs = np.full(M_, np.nan); qsc = np.full(M_, np.nan)
for (mid, toks, is_star), (i0, i1) in zip(units, unit_tok_ranges):
    if is_star: continue
    sp = spans[i0:i1]
    qs[mid] = sp[0].start * HOP / SR
    qsc[mid] = float(np.mean([s.score for s in sp]))
mms = np.array([m["ctc_start"] if m["ctc_start"] is not None else np.nan for m in rm])
def metrics(arr, label):
    idx = [i for i, m in enumerate(rm) if m["note_ids"] and m["share_index"] == 0 and not np.isnan(arr[i]) and not np.isnan(mms[i])]
    c = np.array([arr[i] for i in idx]); o = np.array([rm[i]["midi_start"] for i in idx]); off = float(np.median(c - o)); r = np.abs(c - o - off)
    print(f"[{label:8s}] n={len(idx)} offset={off*1000:+.0f}ms median={np.median(r)*1000:.0f}ms p90={np.percentile(r,90)*1000:.0f}ms >150ms={np.mean(r>0.15)*100:.1f}% >300ms={np.mean(r>0.3)*100:.1f}% max={r.max()*1000:.0f}ms")
    return idx, r
metrics(mms, "MMS_FA"); idx, r = metrics(qs, "HF-CTC")
d = np.array([qs[i] - mms[i] for i in idx]); d -= np.median(d)
print(f"[vs MMS ] |diff| median={np.median(np.abs(d))*1000:.0f}ms p90={np.percentile(np.abs(d),90)*1000:.0f}ms >150ms={np.mean(np.abs(d)>0.15)*100:.1f}%  score median={np.nanmedian(qsc):.2f}")
per = {}
for rr, i in zip(r, idx): per.setdefault(rm[i]["line"], []).append(rr)
print("[lines  ] per-line median ms:", {k: int(np.median(v)*1000) for k, v in sorted(per.items())})
print("[worst  ]", [(rm[i]["kana"], rm[i]["line"], f"{rr*1000:.0f}ms") for rr, i in sorted(zip(r, idx), reverse=True)[:6]])
if a.out: json.dump({"model": a.model, "star": a.star, "start": qs.tolist(), "score": qsc.tolist()}, open(a.out, "w"))
