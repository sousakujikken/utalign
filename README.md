# utalign (うたらいん) — 歌詞の文字単位タイミング抽出ツール

ボーカルトラック・歌詞テキスト・ボーカル MIDI の 3 つから、歌詞の **各文字の発音開始・終了時刻** を推定し、
UTAVISTA の歌詞タイミング JSON (`lyrics-timing/1.0`) を出力します。
UTAVISTA から子プロセスとして呼ばれる CLI、ブラウザで動くアプリ (UTAVISTA v2 と同じデザイントークン)、単体の CLI があります。

> **位置づけ**: utalign は歌詞アニメーションツール UTAVISTA の歌詞タイミング解析エンジンです。UTAVISTA は
> 「utalign をインストール」で本リポジトリの Release にある wheel を自動的に導入します (下記「UTAVISTA からの利用」)。
> 単体でも使えますが、単体利用のサポートは限定的です。処理はすべてローカル (macOS、Apple Silicon 推奨) で行われ、
> 音声・歌詞・MIDI をネットワークへ送信しません。ライセンスは MIT、既定モデルは Apache-2.0 です (「ライセンス」参照)。

## 仕組み (概要)

```
歌詞 txt ─→ 正規化・読み付与 (fugashi/unidic) ─→ モーラ列 (各モーラ → 元の文字インデックス)
音声     ─→ CTC 強制アライメント (日本語 HuBERT 音素モデル) ─→ モーラ毎の音声上の開始時刻 (20ms 精度) + スコア
MIDI     ─→ ノート列・フレーズ分割・オフセット推定
            └→ モーラ↔ノート の単調 DP 突合 (1:1 / 1ノート複数モーラ / メリスマ / skip)
            └→ 時刻確定: フレーズ先頭=MIDI, フレーズ内=CTC を MIDI±100ms でクランプ, end=次モーラ開始 or ノート終端
            └→ result.json (文字 / モーラ / ノート / 行 / フレーズ) → UTAVISTA JSON
```

- 括弧・タグ内 (`[...]`, `(...)`, `（...）`, `<...>`, `【...】` 等) と記号 (`☆ ・ ！ ？` 等) は非発音として扱います。
- 漢字の読みは `fugashi` + `unidic-lite` で付与し、送り仮名を後方一致で切り出して漢字部分にモーラを割り当てます。
  誤りはプロジェクトの「読みの上書き」(または `--readings` に渡すファイル、例: `言葉 こと|ば`) で修正できます。
- MIDI は「1 ノート = 1 モーラ」とは限らないため、DP では複数モーラのノート共有とメリスマの両方を許し、
  ノート内の分割位置は CTC の検出時刻で決めます。歌詞にないノートは音声エネルギーを見て skip します。
- MIDI は必須入力です (ノートが 1 つも無い MIDI はエラーになります)。

設計の経緯は [docs/PLAN.md](docs/PLAN.md) (歴史的文書)、アライナ選定の検証は [docs/](docs/) を参照してください。

## アライナ (モデル) とライセンス

| モデル | ライセンス | 商用 | 備考 |
|---|---|---|---|
| `prj-beatrice/japanese-hubert-base-phoneme-ctc-v4` (既定) | Apache-2.0 | ○ | 音素 CTC、378MB。残差中央値 14〜21ms、emission は M5 で 3.5〜6 秒 |
| `TKU410410103/hubert-large-japanese-asr` ほか (かな語彙) | Apache-2.0 | ○ | 同等精度だが約 25 倍遅い。「設定」から取得可能 |

非商用ライセンスのモデル (torchaudio MMS_FA など) はコードから削除済みで、選択肢に現れません。
選定の根拠と実測値は [docs/feasibility-japanese-ctc-models.md](docs/feasibility-japanese-ctc-models.md)、
Qwen3-ForcedAligner を採用しなかった理由は [docs/feasibility-qwen3-forced-aligner.md](docs/feasibility-qwen3-forced-aligner.md)。

モデルは **モデル保管場所** (既定 `~/Library/Application Support/utalign/models/`) に HF スナップショットを丸ごと保存し、
以後は Hub に接続せずそこから読みます。「設定」→「バックアップ」で別フォルダにコピー、「フォルダから取り込み」で復元できます。

## 音声デコード (ffmpeg 不要)

`utalign/audio.py` が次の順でデコードします。解析用 16kHz モノラルと再生用 WAV は同じデコーダを通るため時刻が一致します
(サンプル MP3 で ffmpeg デコードとの差は 0 サンプルでした)。

1. `soundfile` (libsndfile 1.2 以降): WAV / AIFF / CAF / FLAC / OGG (Vorbis) / MP3
2. macOS の `afconvert` (CoreAudio): M4A / AAC / ALAC など libsndfile が読めない形式 (macOS 以外では非対応)

プロジェクト方針として ffmpeg は使いません (ライセンス条件が平易でないため。用途も macOS に限定)。

afconvert の AAC / ALAC デコーダは CoreAudio の XPC サービス (AudioComponentRegistrar) への mach-lookup を必要とします。
これを遮断する実行サンドボックス内 (`sandbox-exec` の deny default など) では `ExtAudioFileSetProperty ('cfmt') failed ('fmt?')`
で失敗しますが、これは CoreAudio XPC の制約で、サンドボックス外の macOS では正常にデコードできます (エラーメッセージでも案内)。
実 M4A / AAC / ALAC を使ったデコードの回帰テストは、サンドボックス外の macOS で `python -m pytest tests` を実行してください
(`pip install -e '.[dev]'`)。

## セットアップ (開発者向け)

[uv](https://docs.astral.sh/uv/) を使う場合 (`uv.lock` で依存を固定しています):

```bash
uv sync --extra dev                  # venv/ に Python 3.13 + 依存を入れる (UV_PROJECT_ENVIRONMENT=venv)
./venv/bin/utalign models download   # 既定モデル (378MB) を取得
./venv/bin/utalign doctor            # 環境確認
```

pip を使う場合:

```bash
python3.13 -m venv venv
./venv/bin/pip install -e '.[dev]'   # torch / torchaudio / transformers / soundfile / fugashi 等が入る
./venv/bin/utalign models download
./venv/bin/utalign doctor
```

リポジトリを iCloud Drive の中に置く場合は、venv や `work/` が同期されないよう `venv.nosync` のように `.nosync` を付けてください。

## UTAVISTA からの利用

UTAVISTA (AI 歌詞分析タブ) は utalign を子プロセスとして起動します。利用する入口は次の 2 つです。

```bash
utalign doctor --json
# {"name":"utalign","version":"0.3.1","home":...,"model":...,"modelDownloaded":true,"device":"mps",
#  "macos":"14.6.1","mpsConvLimited":true,"torch":"2.14.0",...}

utalign utavista-align --audio vocal.mp3 --lyrics lyrics.txt --midi vocal.mid \
  --out-json <出力 JSON> --work <キャッシュ置き場> [--strip-spaces] [--no-ruby] [--progress-json]
```

`--progress-json` のとき stdout は 1 行 1 JSON のイベントのみになります (人間向けログは `{"event":"log"}` に包み、
ライブラリの標準出力は stderr に逃がします)。

| event | 内容 |
|---|---|
| `start` | `version`, `model`, `device` |
| `stage` | `stage` = `model` / `lyrics` / `midi` / `audio` / `emission` / `align` / `match` / `write` / `export`、`elapsed` (秒) |
| `log` | `text` (人間向けログ 1 行) |
| `done` | `result` = `{outJson, resultJson, phrases, words, chars, selfCheckIssues, meta}` |
| `error` | `message`, `traceback` (終了コード 1) |

UTAVISTA 側は `utalign` 実行ファイルを「設定の上書きパス → 環境変数 → UTAVISTA 管理のインストール → ログインシェルの
`command -v utalign` → 既知パス」の順に探します。venv 内の `bin/utalign` を指定すれば venv の有効化は不要です。

`doctor --json` の `macos` は macOS のバージョン (macOS 以外は null)、`mpsConvLimited` は「その macOS の MPS に
畳み込み出力 65,536 上限があるか (macOS 15.1 未満で true)」です。utalign 自体はこの上限に収まる窓幅で推論するので
true でも MPS で動きますが、UTAVISTA 側の案内表示に使えます。

### Apple Silicon / macOS 15.1 未満での注意

macOS 15.1 未満の PyTorch MPS は畳み込み出力の各次元が 65,536 を超えると
`NotImplementedError: Output channels > 65536 not supported at the MPS device.` を投げます。utalign は音声を
16 秒窓 + 前後 2 秒 (合計 20 秒 = 320,000 サンプル → HuBERT 初段で 63,999 フレーム) ずつモデルへ入れるので、
この上限内で全 macOS の MPS が使えます。それでも GPU/MPS で推論できなかった場合は自動的に CPU へ切り替えて続行し、
ログと結果の `meta.warnings` / `meta.device_fallback` に理由を残します。最初から CPU で動かすには:

```bash
utalign config --set device=cpu
```

### UTAVISTA からの自動インストール

UTAVISTA の「utalign をインストール」は、同梱の uv で `~/Library/Application Support/utalign/venv` に Python 3.13 を用意し、
本リポジトリの GitHub Release に添付された wheel (`utalign-<version>-py3-none-any.whl`) を `uv pip install` で導入したうえで
既定モデルを取得します。Release には再現性の記録として `requirements.lock` (依存の固定版とハッシュ) も添付します。
Release 資産は `tools/build_release.sh` で生成します。

## アプリ (GUI)

```bash
./venv/bin/utalign serve          # http://127.0.0.1:8791/ をブラウザで開く
```

- **プロジェクト**: 左の一覧。「＋ 新規」で作成。入力ファイルはローカルパスを「参照…」で選ぶか、
  「アップロード」/ドラッグ&ドロップでプロジェクトフォルダにコピーします。
- **入力**: 音声 / 歌詞 / MIDI、モデル・デバイス・フレーズ分割・MIDI トラック、読みの上書き、解析の実行とログ。
- **結果**: 波形 / MIDI ピアノロール (フレーズごとに濃淡、ノート上に割当モーラ) / 文字ボックス、緑の縦線 = MIDI onset、
  赤 ▲ = CTC 検出位置。再生中は歌詞がカラオケ風にハイライト。「音源」でクリック音入りに切り替えると耳で同期を確認できます。
- **書き出し**: UTAVISTA `lyrics-timing/1.0` JSON (ルビ・空白除去オプション)。自己検査に加え、「設定」で utavista2 の
  リポジトリを指定すると本体の `validateLyricsTiming` / インポータでも自動検査します。
- **設定**: モデル保管場所・プロジェクト置き場・既定値、モデルの取得 / バックアップ / 取り込み / 削除。

設定とデータの場所は環境変数 `UTALIGN_HOME` で変えられます (既定: macOS は `~/Library/Application Support/utalign`)。
旧名 `utaalign` のフォルダが残っていて新しいフォルダが無い場合は初回起動時に自動で移動します。

```
$UTALIGN_HOME/config.json
$UTALIGN_HOME/models/<repo--id>/           モデル保管場所 (config の models_dir で変更可)
$UTALIGN_HOME/projects/<プロジェクト>/     project.json, inputs/, out/, work/
```

## CLI

```bash
./venv/bin/utalign align --audio vocal.mp3 --lyrics lyrics.txt --midi vocal.mid --out out --work work
./venv/bin/utalign export-utavista --result out/result.json --out out/utavista-lyrics-timing.json --ruby
./venv/bin/utalign models list | download [id] | delete <id> | import --path <dir> | export --path <dir>
./venv/bin/utalign config [--set key=value]
./venv/bin/utalign serve --dir out                       # 旧形式の out/ をビューアで開く
```

`out/` には `result.json`, `audio.wav`, `click.wav`, `peaks.json` とビューア (`index.html`) が出ます。
`work/` には 16kHz 音声と emission のキャッシュ (パラメータ調整時の再実行が数秒で済む)。

UTAVISTA 本体での検証 (utavista2-refactor のリポジトリで実行):

```bash
./node_modules/.bin/tsx <utalign>/tools/validate_with_utavista.ts <lyrics-timing.json> <歌詞.txt> <audioDurationMs>
```

読み込みは UTAVISTA の Lyrics タブ →「歌詞読込」でこの JSON を指定します。

## result.json のスキーマ

- `meta`: 入力ファイル名、`model` / `vocab`、`offset` (MIDI → 音声の秒オフセット)、残差統計 (`residual_median` 等)、未割当数、警告
- `chars[]`: 元テキストの全文字 (改行・空白・記号を含む)
  - `index`, `char`, `line`, `pronounceable`, `sung`, `start`, `end`, `mora_ids`, `split_estimated`, `confidence`
- `moras[]`: `kana`, `romaji`, `kind` (normal/sokuon/choon/hatsuon), `text`, `char_idx`, `start`, `end`,
  `source` (midi / ctc_clamped / ctc_shared / split), `end_kind`, `midi_start`, `ctc_start`, `note_ids`, `share_index`, `share_count`, `phrase`
- `notes[]`: `start`, `end` (オフセット適用済み), `pitch`, `phrase`, `mora_ids` (空 = 歌われていないノート)
- `lines[]`, `phrases[]`: 行・フレーズ単位の集計

## 品質指標 (実測)

| 曲 | 残差 中央値 / p90 | 150 ms 超 | 未割当モーラ / 歌われないノート | 全体所要時間 (M5, mps) |
|---|---|---|---|---|
| 無限☆おもちゃ箱 (3:35) | 21 ms / 55 ms | 0.9 % | 0 / 9 | 15 秒 |
| オーバーヒートな季節 | 14 ms / 44 ms | 2.5 % | – | – |

## ファイル構成

- `utalign/lyrics.py` 正規化・読み・モーラ化・文字対応・CTC ユニット生成 (バックエンドのトークン化関数を受け取る)
- `utalign/audio.py` 音声デコード (libsndfile / afconvert) とリサンプル
- `utalign/aligners/` 強制アライメントのバックエンド (`base.py` 共通 emission 窓処理、`hf_ctc.py`)
- `utalign/models.py` モデル登録簿とローカル保管 (取得 / 削除 / 取り込み / バックアップ)
- `utalign/config.py` 設定ファイルと場所
- `utalign/midi.py` ノート抽出・フレーズ分割
- `utalign/matcher.py` モーラ↔ノート DP (コストは `Params` で調整)
- `utalign/timing.py` オフセット推定・時刻確定
- `utalign/output.py` JSON / peaks / WAV / クリック音
- `utalign/utavista.py` UTAVISTA lyrics-timing/1.0 へのエクスポート
- `utalign/pipeline.py` 解析・書き出しの一括処理と進捗イベント (CLI / GUI / UTAVISTA 共通)
- `utalign/project.py` プロジェクト管理とジョブ実行
- `utalign/server.py` GUI サーバ (標準ライブラリのみ、JSON API + Range 対応の静的配信)
- `utalign/web/` アプリ (`index.html`, `app.js`, `style.css`, ビューア `viewer.js`)
- `utalign/cli.py` `align` / `export-utavista` / `utavista-align` / `doctor` / `serve` / `models` / `config`
- `tools/validate_with_utavista.ts` UTAVISTA 本体の検証器で出力を検査するスクリプト
- `tools/spikes/` アライナ選定の検証スパイク (履歴、製品コードでは未使用)
- `tools/dump_assignments.py` 行別のモーラ割当ダンプ (デバッグ用)
- `tools/build_release.sh` Release 資産 (wheel / requirements.lock / THIRD_PARTY_NOTICES.md) の生成
- `tools/generate_third_party_notices.py` 依存パッケージのライセンス一覧の生成

## ライセンス

- utalign 本体: [MIT](LICENSE)
- 既定モデル `prj-beatrice/japanese-hubert-base-phoneme-ctc-v4`: Apache-2.0 (初回に Hugging Face Hub から取得、再配布はしない)
- 依存パッケージ (torch、transformers、librosa、fugashi ほか): [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) に `uv.lock` の固定版から生成した一覧を置いています。
  wheel 自体は依存を同梱せず、利用者の環境で PyPI から取得されます。
- テスト用の音声 fixture (`tests/fixtures/`) は合成トーンで、ファイル名はエンコードに使ったツールを示します。
  utalign の実行時に ffmpeg は使いません (「音声デコード」参照)。
