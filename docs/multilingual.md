# 多言語の音節解析（v0.4.1）

ボーカル音声・旋律MIDI・原文歌詞を使い、元の単語の何番目の音節が発音されているかを20ms刻みで推定する。
日本語・英語・ドイツ語・フランス語・中国語（普通話）・韓国語・スペイン語・イタリア語に対応し、同じ行に混在できる。
コード・再生画面・依存関係は正式な `utalign` パッケージに含まれ、試作フォルダやサンプル固有辞書は不要。

## 利用方法

既存のコマンドをそのまま使える。既定の `auto` は、注釈を除いた歌詞にLatin文字・Hangulがあれば音節解析へ切り替える。
漢字だけの歌詞は日中の言語判定を使う。言語指定・読み指定を与えた場合も音節解析へ進む。
日本語だけの歌詞は従来のモーラ解析を使い、`--alignment-mode multilingual` で音節解析を明示選択できる。

```sh
utalign align --audio vocal.wav --lyrics lyrics.txt --midi vocal.mid \
  --languages ja,en,de --out out --work work

utalign utavista-align --audio vocal.wav --lyrics lyrics.txt --midi vocal.mid \
  --languages ja,en,de --out-json timing.json --work work --progress-json
```

- `--languages`: 曲で使う言語コードをカンマ区切りで指定。省略時は8言語。
- `--alignment-mode auto|multilingual|japanese`: 自動切替・多言語音節・従来の日本語モーラ解析。
- `--midi-tracks`: 歌唱旋律トラックの番号。伴奏を含むMIDIでは指定を推奨。
- `--midi-offset`: 音声時刻−MIDI時刻（秒）。省略時は既存のオンセット相関で推定。
- `--search-band`: 最初のCTC時刻を中心とするMIDI対応探索の範囲（既定10秒）。
- `--readings`: 日本語の読み上書き。音節解析でも日本語の箇所へ適用する。

音節解析は日本語**音素**モデルを使う。既定の `japanese-hubert-base-phoneme-ctc-v4` が対応する。
かな文字モデル・starオプションとの組み合わせは明示的なエラーにする。

Webではプロジェクトの「歌詞の言語と音節」で方式・言語・MIDIオフセットを設定する。
解析後は単語ごとの音節ボタン、MIDI対応、弱い音声支持の表示、解析JSONの保存が使える。

## 言語・近似読みの補助指定

短語や漢字のみの区間は言語判定が曖昧な場合がある。Webの言語ヒントは1行ずつ `fr: Bonjour` の形式。
CLIの `--language-hints hints.json` は次の形式（`line` は1始まり、`char_start` は0始まりのUnicodeコードポイント位置）：

```json
{"spans":[{"text":"Bonjour","language":"fr"},{"text":"你好","language":"zh","line":3}]}
```

カタカナの正確な転写ではなく、音声との照合に使いやすい近似音を使う。
原語の単語・音節IDは別に保持するので、`out → キャウ` でも `out` の1音節として扱う。
Webの歌唱の近似読みは `knock out = ノッ / キャウ`。単語間を `/`、単語内の音節を `|` で区切る。
CLIの `--proxy-readings readings.json` は次の形式：

```json
{"spans":[{"text":"knock out","syllables":[["ノッ"],["キャウ"]]}]}
```

位置指定を省略すると同じ原文の全出現に適用する。元の単語数・音節数を変える指定や重複は拒否する。
補助指定なしでも各言語のG2Pから近似音を生成する。サンプルの読みを本体へハードコードしていない。

## 時刻とJSONの意味

多言語の `result.json` は `schema: utalign-result/2`、`timing_unit: syllable`。
`words`、`syllables`、原文位置、原語の発音仮説、近似読み、音素区間、母音核時刻、MIDI音符ID、支持度を保存する。
既存の文字エクスポータ用に `chars` / `moras` / `lines` も保持する。多言語時の `moras` は音節の互換ビューであり、モーラ数ではない。

MIDI割当は音声尤度を参照した動的計画法で求め、そのMIDI区間の前140ms〜後100msに全音素を制約してCTC整列する。
1音符に最大3音節、1音節に最大8音符のメリスマと余分なMIDI音符のスキップを扱う。歌詞の音節を削除しない。
成立する経路がなければエラーにし、制約を外した結果へ暗黙に切り替えない。音高照合はまだ使わない。

`start` は先頭音素の開始、`nucleus_start` は母音核の開始、`acoustic_end` は音素列の末尾。
表示用の `end` はMIDIの保持時間も使い次の音節開始までに制限するため、独立に検出した発声終了ではない。
`acoustic_support` は正解確率ではなく、0.02未満を `low_support` と表示する。

UTAVISTAの `lyrics-timing/1.0` は従来の構造を維持し、各wordへ `syllables` の追加情報を付ける。
外国語の綴りの各文字は `timingBasis: word_envelope` として同じ単語区間を共有する。音節の時刻を文字数で等分しない。
音節拡張を読まないUTAVISTAクライアントは外国語を単語単位で表示する。音節自体の詳細は `result.json` と追加フィールドに残る。

## 検証と制限

v0.4.1はeSpeak NGをアンインストールした環境で確認した。既存の日英独サンプルは387単語・751音節を出力し、
v0.4.0との原文位置・単語内音節の対応を維持した。751音節中748音節の開始時刻が一致し、差の中央値・90パーセンタイルは0ms。
最大差は `creators` 内の220msであり、旧版・新版どちらが正解かは手動の正解ラベルなしには判断できない。
114音節は支持が弱い（旧版113）。全音節を出力したことは全時刻の正しさを保証しない。
UTAVISTA形式への変換は105行・177空白区切り語・1136文字で、自己検査エラー0件。

保存済みの8言語の合成読み上げを新しいG2Pで再解析し、18音節すべてを出力した。
新たな音声合成やeSpeak呼び出しは行っていない。合成器の音素イベントとの差は中央値28.2ms、最大84.6msだった。
同じ合成器の母音核からMIDIを作っており、正確なMIDIを与えた結合動作試験である。実歌唱8言語の精度検証とは区別する。
歌唱による連結・省略・リエゾン、短語の言語判定、音節化の曖昧さは残る。
数値表記は実際に歌う読みへ書き換える必要がある。未対応の文字体系・数値を含む場合は、黙って歌詞から除かずエラーにする。
英語の同綴異音語は辞書の先頭候補を使う。未知語はモデル推定、未知の大文字略語は字名読みとする。
欧州言語・韓国語は規則ベースなので外来語・固有名詞・形態素境界をまたぐ発音などに限界がある。
中国語はpypinyinの語句読みを使い、ピンイン1音節を複数の母音文字に分割しない。
97件のテストと8件の言語別サブテストを実施。音素変換中の外部プロセス・ネットワーク・eSpeak importを禁止するテストも含む。

## 依存関係

G2Pは言語ごとのローカル変換器を利用する。`LocalG2P` の出力を共通の音節・近似音素列へ接続する。

| 言語 | 変換器 | ライセンス |
|---|---|---|
| 日本語 | 既存のfugashi / UniDic読み変換 | 従来どおり |
| 英語 | 同梱CMUdict＋g2pEのNumPy推論 | BSD-style / Apache-2.0 |
| 中国語（普通話） | pypinyin 0.55.0＋Epitranのピンイン変換規則 | MIT |
| 独・仏・西・伊・韓 | Epitran 1.35.2＋音節化の補助規則 | MIT |

eSpeak NG / espeakng-loader / phonemizer / g2pK / gruut は本番依存に含めない。
Epitranはパッケージ内の変換表と前後処理だけを使用し、Fliteや外部辞書のダウンロードには進まない。
libmarisaはBSD-2-ClauseとLGPLの選択ライセンスで、BSD-2-Clause側を利用する。
英語辞書・モデルはwheelに同梱し、初期化時にSHA-256を確認する。
JSONは各音節の `pronunciation_source` / `pronunciation_variants` と、`meta` のバージョン・変換器版を記録する。
ライセンス全文・固定revisionは [同梱リソース](../utalign/multilingual/resources/README.md) を参照。

言語判定はLinguaを利用する。Python 3.11はLingua 2.1.1、3.12以降は2.2.0を依存ロックへ記録している。
音響モデル導入後、解析中に音声・歌詞を外部送信しない。
依存パッケージと配布元は [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) を参照。
