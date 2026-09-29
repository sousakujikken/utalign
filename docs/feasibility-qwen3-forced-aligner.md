# Issue #1 実現性検証: Qwen3-ForcedAligner バックエンド (2026-09-16)

## 結論

**現状のモデルでは MMS_FA の代替にならない (歌唱で精度不足)。** 作業 2〜5 (バックエンド抽象化・DP 調整・依存分割・README) は着手しない方がよい。

- 同一素材 (無限☆おもちゃ箱) で、CTC ユニット単位の開始時刻と MIDI onset の残差を比較した結果:

| 条件 | 中央値 | p90 | >150ms |
|---|---|---|---|
| MMS_FA (現行) | 25 ms | 68 ms | 1.6 % |
| Qwen3-FA 全曲一括 / ユニット単位 | 76 ms | 1099 ms | 38.4 % |
| Qwen3-FA セクション分割 / ユニット単位 (ひらがな) | 57 ms | 667 ms | 32.0 % |
| Qwen3-FA セクション分割 / モーラ単位 | 60 ms | 778 ms | 31.8 % |
| Qwen3-FA セクション分割 / 文字単位 (漢字そのまま) | 63 ms | 855 ms | 33.0 % |
| Qwen3-FA セクション分割 / nagisa 単語単位 (公式 Japanese 経路、単語先頭で評価) | 58 ms | 903 ms | 33.1 % |
| Qwen3-FA セクション分割 / 空白区切り文節単位 (文節先頭で評価) | 373 ms | 1620 ms | 77.5 % |
| Qwen3-FA セクション分割 / 行単位 (行頭で評価) | 640 ms | 3154 ms | 90.3 % |
| Qwen3-FA 固定窓 30 / 15 / 8 秒 / ユニット単位 | 64 / 63 / 68 ms | 839 / 1127 / 890 ms | 34〜35 % |
| Qwen3-FA bf16 (非量子化) セクション分割 / ユニット単位 | 58 ms | 667 ms | 31.6 % |
| Qwen3-FA 公式 PyTorch 実装 (qwen-asr, MPS, bf16) セクション 1 のみ | 52 ms | 400 ms | 24.3 % |

- 受け入れ条件「中央値 ≤ 40ms 目安」を、どの粒度・分割・精度でも満たさない。
  さらに深刻なのは裾で、**約 3 割のモーラが 150ms 超**、2 割強が 300ms 超ずれる。
  DP は CTC 残差を主コストにしているため、この分布では MIDI±100ms クランプ内に収まらず割当自体が崩れる。
- 誤りは特定の行に集中する (セクション 3〜4、行 25 / 31 / 33 / 36〜38 で中央値 200ms〜7s)。
  複数モーラが同一時刻に潰れる (長さ 0 のユニットが 150〜200 個) 「崩壊」型で、
  ロングトーンや休符後の再開で位置を見失っている。粒度を粗くする (文節・行) と むしろ悪化する。
- **話し言葉では問題ない**: 同じ歌詞を macOS TTS (Kyoko) で読ませた音声では Qwen3-FA と MMS_FA のユニット開始時刻が
  中央値 20ms / p90 54ms で一致した。つまり日本語・モーラ粒度・MLX 移植の問題ではなく、**歌唱音声に対する
  モデル側の弱さ**である (公式も Speech のみ評価)。

## 確認できた良い点 (将来モデルが改善した場合に有効)

- **任意ユニットを渡せる**: mlx-audio の `language` はテキスト分割にしか使われず、モデル入力には言語トークンが無い。
  `"<timestamp><timestamp>".join(units)` を直接組めば、モーラ / CTC ユニット / 文字など好きな粒度で整列できる
  (`tools/spike_qwen3fa.py` の `align()`; 公式 `generate()` と出力が完全一致することを確認済み)。
  nagisa は不要。ひらがな / カタカナの差はほぼ無い (ひらがながわずかに良い)。
- **速い・軽い**: Apple M5 で全曲 215 秒を 1 パス 3.3 秒、セクション分割で 1.2〜1.9 秒、ピークメモリ 2.5〜3.7GB
  (MMS_FA は emission 計算に数十秒)。全曲 1 パス (5 分上限内) も動作する。
- **torch 非依存**: mlx-audio 0.5.4 + mlx 0.32 のみで動作 (transformers はトークナイザ用に入る)。
- **擬似スコアは取れる**: 各タイムスタンプは 5000 クラス分類なので argmax 確率 (中央値 0.3〜0.4) が得られ、
  尤度の代わりに DP 重みへ使える余地はある。
- PyTorch MPS 経路 (`qwen-asr` 0.0.6, transformers 4.57.6 固定, `device_map="mps"`) も動作した。
  モデル読み込み 111 秒 (初回 DL 込み)、セクション 1 の整列 2.6 秒。精度傾向は MLX 版と同じ。

## 設計上の制約 (モデル固有)

- タイムスタンプは **80 ms 刻みの分類** (`timestamp_segment_time = 80`)。量子化誤差だけで一様 ±40ms
  (期待絶対誤差 20ms) が乗り、MMS_FA の 20ms フレームより粗い。話し言葉で中央値 20ms 一致したのはこの下限。
- 非単調な予測を LIS で補正しており (`fix_timestamp`)、歌唱では 1 セクションあたり 10〜30 % のタイムスタンプが補正対象になった。
  補正後は単調だが「同一時刻に潰れる」形で歪みが残る。
- 5 分上限 (5000 クラス × 80ms = 400s)。

## 代替案

ライセンス目的 (CC BY-NC 回避) を満たす別経路の候補:

1. **wav2vec2 系の Apache/MIT モデルで CTC アライナを差し替える** — 既存の `ctc.py` / DP をほぼそのまま使える。
   例: `facebook/wav2vec2-xlsr-53` 系の日本語ファインチューン (ライセンス要確認、多くは Apache-2.0)、
   `reazon-research/reazonspeech-*` (Apache-2.0、日本語 CTC)。ローマ字ではなく かな/音素 の語彙になるので
   ユニット生成の変更が必要。歌唱での精度は要検証だが、CTC の枠組み (star トークン相当・尤度スコア) は維持できる。
2. **WhisperX 型** (Whisper + wav2vec2 CTC) — Whisper 自体は MIT だが、アライメント部分は結局 wav2vec2 の日本語モデルに依存する。
3. Qwen3-ForcedAligner は **歌唱データでのファインチューン版**が出るまで保留。

## 再現手順

```bash
python3.13 -m venv qvenv.nosync
./qvenv.nosync/bin/pip install mlx-audio librosa soundfile "fugashi[unidic-lite]" mido
# MMS_FA の結果 (out/result.json, work/vocal16k.wav) が align 済みであること
PYTHONPATH=. ./qvenv.nosync/bin/python tools/spike_qwen3fa.py --root . --gran unit_hira --scope section
```

`--gran {unit,unit_hira,mora,char,seg,line}`、`--scope {song,section,window --win N}`、
`--model mlx-community/Qwen3-ForcedAligner-0.6B-bf16` で上表の各条件を再現できる。
評価は result.json の 1:1 割当先頭モーラ (`share_index == 0`) について、中央値オフセット除去後の
`|開始時刻 − MIDI onset|` を MMS_FA と同じ定義で計算している。
