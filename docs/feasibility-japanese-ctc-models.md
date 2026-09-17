# Issue #1 代替案の実現性検証: Apache-2.0 の日本語 CTC モデルで MMS_FA を置き換える (2026-09-16)

## 結論

**実現可能。第一候補は `prj-beatrice/japanese-hubert-base-phoneme-ctc-v4` (Apache-2.0, 音素 CTC, 378MB)。**
同一素材で MMS_FA と同等以上の精度を出し、2 曲目でも再現した。emission 計算は MMS_FA より速く (全曲 3.5〜5 秒)、
既存の `ctc.py` の枠組み (20 秒窓 emission + torchaudio `forced_align`、尤度スコア) をそのまま流用できる。

Qwen3-ForcedAligner のような別方式ではなく「同じ wav2vec2/HuBERT-CTC 方式で重みだけ差し替える」ため、
DP・時刻確定・GUI は無変更で済む。変わるのはユニットのトークン化 (ローマ字 → 音素 or かな) だけ。

## 選定

HF 上の日本語 CTC 系モデルを 170 件超列挙し、(1) ライセンスが Apache-2.0 / CC-BY 等の商用可、(2) CTC ヘッド付きで
emission が取れる (`Wav2Vec2ForCTC` / `HubertForCTC`)、(3) 語彙が音素またはかな (漢字語彙は歌詞側の読み付与と衝突する)、
(4) ダウンロード可、で絞った 7 モデルを実測した。ESPnet の OWSM-CTC (CC-BY-4.0) と NeMo / k2 の ReazonSpeech
Transducer 系は、フレームワーク依存が重く CTC 直アクセスに手間がかかるため今回は見送った。

## 実測 (無限☆おもちゃ箱、CTC ユニット先頭モーラの MIDI onset 残差、MMS_FA と同じ定義)

| モデル | ライセンス | 語彙 | サイズ | emission (M5 MPS) | 中央値 | p90 | >150ms | >300ms |
|---|---|---|---|---|---|---|---|---|
| MMS_FA (現行) | CC BY-NC 4.0 | ローマ字 | 1.2GB | (CPU 約 15s) | 25 ms | 68 ms | 1.6 % | 0.0 % |
| **prj-beatrice/japanese-hubert-base-phoneme-ctc-v4** | Apache-2.0 | 音素 (OpenJTalk 系 48) | 378MB | **3.5 s** (CPU 4.9 s) | **21 ms** | **63 ms** | 1.8 % | 0.4 % |
| prj-beatrice/japanese-hubert-base-phoneme-ctc-v3 | Apache-2.0 | 音素 | 378MB | 27 s (初回) | 22 ms | 58 ms | 1.5 % | 0.2 % |
| TKU410410103/hubert-large-japanese-asr | Apache-2.0 | ひらがな | 1.26GB | 87 s | 22 ms | 58 ms | 1.8 % | 0.2 % |
| TKU410410103/wav2vec2-base-japanese-asr | Apache-2.0 | ひらがな | 378MB | 30 s (初回) | 22 ms | 68 ms | 3.8 % | 1.5 % |
| vumichien/wav2vec2-large-xlsr-japanese-hiragana | Apache-2.0 | ひらがな | 2.5GB | 82 s | 21 ms | 61 ms | 2.9 % | 1.1 % |
| reazon-research/japanese-wav2vec2-large-rs35kh | Apache-2.0 | 文字 (漢字含む 3000) | 1.27GB | 79 s | 24 ms | 64 ms | 2.7 % | 0.9 % |
| AndrewMcDowell/wav2vec2-xls-r-1b-japanese-hiragana-katakana | Apache-2.0 | ひらがな+カタカナ | 3.85GB | 240 s | 28 ms | 81 ms | 4.4 % | 1.3 % |

- 2 曲目 (オーバーヒートな季節、439 モーラ) での beatrice v4: 中央値 14 ms / p90 44 ms / >150ms 2.5 %
  (MMS_FA: 18 ms / 71 ms / 2.5 %)。emission 3.7 秒。
- beatrice v4 と MMS_FA のユニット開始時刻の差は中央値 0 ms / p90 20 ms (1 フレーム) で、ほぼ同じ場所を指している。
  残差 150ms 超の位置も両者で共通 (行 24 の「ア」、行 33 の「ト」など MIDI 側の採譜差と思われる箇所)。
- star トークンの有無 (max / logsumexp、ペナルティ 0〜3) は結果に影響しなかった (この曲には歌詞外発声が無い)。
  歌詞外発声を吸収する必要が出たら、emission に「非 blank の最大値 − ペナルティ」列を足す方式で代用できる。
- 尤度スコア (フレーム確率の平均) は中央値 0.96 で得られるので、`run_match` の重み `w` は現状のまま使える。

## 第一候補の詳細: prj-beatrice/japanese-hubert-base-phoneme-ctc-v4

- rinna/japanese-hubert-base を ReazonSpeech で音素 CTC にファインチューンしたもの (reazon-research 関係者の prj-beatrice)。
  読みは pyopenjtalk-plus で付与。語彙は `a i u e o N cl ky sh ch ts j f w y ... pau sil` の 48 トークン。
- 音素語彙なので、既存のモーラ → ローマ字 (`lyrics.py` の `mora_romaji`) から機械的に変換できる:
  子音部 (ローマ字から末尾母音を除いた部分) が語彙にあればそのまま (`ky`,`sh`,`ch`,`ts`,`j`,`f` 等)、
  「ん」→ `N`、「っ」→ `cl`、「ー」→ トークン無し (現行 MMS と同じ折り畳み)。
- `preprocessor_config.json` は `do_normalize: false` (HF の `AutoFeatureExtractor` で読めば自動で反映される)。
- フレーム間隔は 20 ms (HOP 320) で MMS_FA と同一。`ctc.py` の窓処理・キャッシュはそのまま。
- 注意点:
  - ベースモデル `rinna/japanese-hubert-base` の HF リポジトリは現在 401 (非公開化または削除) で参照できない。
    beatrice の重みは自己完結しており Apache-2.0 で公開されているが、系譜のライセンス確認と、
    重みのローカル保全 (HF から消えた場合に備える) を推奨する。ミラー `yky-h/japanese-hubert-base` は Apache-2.0 表記。
  - v5 はライセンス未記載のため採用しない。v3 も同等精度で予備になる。
  - 学習データは話し言葉のみ (ReazonSpeech)。今回の 2 曲では歌唱でも問題なかったが、
    曲が増えたら残差指標で継続確認する。

## 第二候補

- `TKU410410103/hubert-large-japanese-asr` (ひらがな語彙、1.26GB): 精度は同等、emission は 25 倍遅い (87 秒)。
  音素変換が不要で、かな語彙の方が扱いやすい場合の代替。
- ひらがな語彙モデルでは「ー」を語彙にそのまま持つので、長音を独立トークンにする実験が可能。

## 実装方針 (Issue #1 の作業 2〜5 の読み替え)

1. `ctc.py` を「backend = mms_fa | hf_ctc(model_id)」に分岐。hf_ctc は `AutoModelForCTC` + `AutoFeatureExtractor` で
   emission を計算し (窓処理は共通化)、`torchaudio.functional.forced_align` + `merge_tokens(blank=pad_id)` で整列。
2. `lyrics.py` の `build_ctc_units` に語彙種別 (romaji / phoneme / kana) を渡し、ユニット text を語彙に合わせて生成。
   音素変換テーブルは `tools/spike_hf_ctc.py` の `mora_tokens` を移植。
3. DP・時刻確定は無変更 (スコアあり、star 不要)。
4. 依存: torch / torchaudio / transformers。MMS_FA 用の torchaudio pipelines は不要になるが torch 自体は残る
   (torch は BSD なのでライセンス上の問題はない)。`--aligner mms_fa` は互換のため残す。
5. README にライセンス比較 (MMS_FA: CC BY-NC 4.0 / beatrice v4: Apache-2.0) と選択指針を記載。

## 再現手順

```bash
python3.13 -m venv tvenv.nosync
./tvenv.nosync/bin/pip install torch torchaudio transformers soundfile numpy "fugashi[unidic-lite]" mido
# MMS_FA の結果 (out/result.json, work/vocal16k.wav) が align 済みであること
PYTHONPATH=. ./tvenv.nosync/bin/python tools/spike_hf_ctc.py --model prj-beatrice/japanese-hubert-base-phoneme-ctc-v4
PYTHONPATH=. ./tvenv.nosync/bin/python tools/spike_hf_ctc.py --model TKU410410103/hubert-large-japanese-asr
```

`--star max --star-penalty 1.0` で star 列の追加、`--device cpu` で CPU 実行、`--audio/--lyrics/--result` で別の曲を指定できる。
評価は result.json の 1:1 割当先頭モーラ (`share_index == 0`) について、中央値オフセット除去後の
`|開始時刻 − MIDI onset|` を MMS_FA と同じ定義で計算している。

関連: [feasibility-qwen3-forced-aligner.md](feasibility-qwen3-forced-aligner.md) (Qwen3-ForcedAligner は歌唱で不採用)。
