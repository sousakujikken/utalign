"""歌詞テキストの正規化・読み付与・モーラ分割・文字対応.

出力の中心は Mora のリスト。各 Mora は元テキストの文字インデックス集合を持つ。
CTC 用のローマ字ユニット (CtcUnit) はモーラとは多対一 (ー/っ は独立トークンを持たない)。

原文 (LyricsResult.text / chars) は変更しない。読みは解析用テキスト
(lyric_input.analysis_text: NFC・半角カナ等の限定的な幅変換) で求め、
Mora.char_idx は常に原文のコードポイント位置を指す。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import fugashi

from .lyric_input import (BRACKET_RE, AnalysisText, LyricsInputError, analysis_text, annotation_indices,
                          coverage_issues, line_spans, make_issue)

# ---------------------------------------------------------------- 文字クラス
HIRAGANA = re.compile(r"[ぁ-ゖ]")
KATAKANA = re.compile(r"[ァ-ヺ]")
KANA_ANY = re.compile(r"[ぁ-ゖァ-ヺー]")  # ー を含む
# 々〆〇・互換漢字・拡張漢字 (𠮷 など) も漢字として読みを求める (読めなければ診断する)
KANJI = re.compile("[一-鿿㐀-䶿々〆〇豈-﫿\U00020000-\U0003ffff]")
# 括弧・タグ (BRACKET_RE) は lyric_input で定義し、両経路で共有する

SMALL_KATA = set("ャュョァィゥェォヮ")
SMALL_HIRA = set("ゃゅょぁぃぅぇぉゎ")

BASE = {
    'ア':'a','イ':'i','ウ':'u','エ':'e','オ':'o',
    'カ':'ka','キ':'ki','ク':'ku','ケ':'ke','コ':'ko',
    'サ':'sa','シ':'shi','ス':'su','セ':'se','ソ':'so',
    'タ':'ta','チ':'chi','ツ':'tsu','テ':'te','ト':'to',
    'ナ':'na','ニ':'ni','ヌ':'nu','ネ':'ne','ノ':'no',
    'ハ':'ha','ヒ':'hi','フ':'fu','ヘ':'he','ホ':'ho',
    'マ':'ma','ミ':'mi','ム':'mu','メ':'me','モ':'mo',
    'ヤ':'ya','ユ':'yu','ヨ':'yo',
    'ラ':'ra','リ':'ri','ル':'ru','レ':'re','ロ':'ro',
    'ワ':'wa','ヰ':'i','ヱ':'e','ヲ':'o','ン':'n',
    'ガ':'ga','ギ':'gi','グ':'gu','ゲ':'ge','ゴ':'go',
    'ザ':'za','ジ':'ji','ズ':'zu','ゼ':'ze','ゾ':'zo',
    'ダ':'da','ヂ':'ji','ヅ':'zu','デ':'de','ド':'do',
    'バ':'ba','ビ':'bi','ブ':'bu','ベ':'be','ボ':'bo',
    'パ':'pa','ピ':'pi','プ':'pu','ペ':'pe','ポ':'po',
    'ヴ':'vu',
    # 単独の小書き (フォールバック)
    'ャ':'ya','ュ':'yu','ョ':'yo','ァ':'a','ィ':'i','ゥ':'u','ェ':'e','ォ':'o','ヮ':'wa',
}
SMALL_V = {'ァ':'a','ィ':'i','ゥ':'u','ェ':'e','ォ':'o','ャ':'ya','ュ':'yu','ョ':'yo','ヮ':'wa'}


def hira_to_kata(s: str) -> str:
    return "".join(chr(ord(c) + 0x60) if 0x3041 <= ord(c) <= 0x3096 else c for c in s)


def is_kana_only(s: str) -> bool:
    return bool(s) and all(KANA_ANY.match(c) for c in s)


# ---------------------------------------------------------------- データ構造
@dataclass
class Mora:
    id: int
    kana: str                 # カタカナ表記 (例: 'キャ', 'ッ', 'ー', 'ン')
    romaji: str               # 'kya', 'Q'(促音), 'H'(長音), 'n'
    char_idx: list[int]       # 元テキストの文字インデックス
    line: int
    seg: int                  # 行内のスペース区切りセグメント番号
    kind: str = "normal"      # normal | sokuon | choon | hatsuon
    split_estimated: bool = False
    ctc_unit: Optional[int] = None  # 対応する CtcUnit の index
    run_start: bool = False   # 連続した発音文字のまとまりの先頭モーラ (直前の音が無い)
    tokens: list[str] = field(default_factory=list)  # build_ctc_units が付けたトークン
    analysis_idx: list[int] = field(default_factory=list)  # 解析用テキスト上の位置 (被覆判定用)
    # 読み上書きの表記全体に属するモーラ (文字ごとの読みは不明: '100 ひゃく', 'LOVE らぶ')。
    # char_idx は表記の全文字。文字ごとの発音時刻やルビを作らず、表記全体の区間 (word_envelope) で扱う。
    envelope: bool = False


@dataclass
class CtcUnit:
    id: int
    text: str                 # 表示用 (tokens の連結)
    mora_ids: list[int]
    is_star: bool = False
    line: int = -1
    tokens: list[str] = field(default_factory=list)   # バックエンド語彙のトークン列 (star は ['*'])


@dataclass
class LyricsResult:
    text: str
    chars: list[str]
    lines: list[tuple[int, int]]        # 行ごとの (start_char, end_char)。改行 (CRLF/LF/CR/U+2028 等) は含まない
    moras: list[Mora]
    units: list[CtcUnit]
    unsung_chars: set[int] = field(default_factory=set)  # 非発音文字
    analysis: Optional[AnalysisText] = None             # 解析用テキストと原文位置の対応
    annotated_chars: set[int] = field(default_factory=set)  # 注釈 (括弧・© 行) の原文位置
    unread_tokens: dict[int, str] = field(default_factory=dict)  # 読めなかった文字の原文位置 -> 形態素表層
    # 安全に対応付けられない読み上書き (解析用テキスト上の [analysis_start, analysis_end), key, message, suggestion)
    override_problems: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------- モーラ分割
def split_moras_kata(k: str) -> list[str]:
    """カタカナ文字列をモーラに分割. 'キャ','ッ','ー','ン' はそれぞれ 1 モーラ."""
    res = []
    i = 0
    while i < len(k):
        c = k[i]
        if i + 1 < len(k) and k[i + 1] in SMALL_KATA and c not in SMALL_KATA and c not in "ッーン":
            res.append(c + k[i + 1]); i += 2
        else:
            res.append(c); i += 1
    return res


def mora_romaji(m: str) -> tuple[str, str]:
    """モーラ -> (romaji, kind)."""
    if m == 'ッ':
        return 'Q', 'sokuon'
    if m == 'ー':
        return 'H', 'choon'
    if m == 'ン':
        return 'n', 'hatsuon'
    if len(m) == 2:
        base = BASE.get(m[0], '')
        s = SMALL_V.get(m[1], '')
        if s.startswith('y'):
            stem = base[:-1]
            if stem in ('sh', 'ch', 'j'):
                return stem + s[1:], 'normal'   # シャ -> sha, チュ -> chu, ジョ -> jo
            return stem + s, 'normal'           # キャ -> kya
        # ファ, ティ, ウィ など
        stem = base[:-1] if base else ''
        if base == 'u':      # ウィ -> wi
            return 'w' + s, 'normal'
        if base == 'vu':
            return 'v' + s, 'normal'
        if base in ('te', 'de') and s == 'i':   # ティ/ディ
            return base[0] + 'i', 'normal'
        if base in ('to', 'do') and s == 'u':   # トゥ/ドゥ
            return base[0] + 'u', 'normal'
        return stem + s, 'normal'                # ファ -> fa, シェ -> she
    return BASE.get(m, m.lower()), 'normal'


# ---------------------------------------------------------------- 読み付与
class Reader:
    def __init__(self, override_path: Optional[Path] = None):
        self.tagger = fugashi.Tagger()
        self.override: dict[str, list[str]] = {}
        if override_path and Path(override_path).exists():
            for ln in Path(override_path).read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln or ln.startswith('#'):
                    continue
                surf, _, read = ln.partition(' ')
                if surf and read:
                    # 歌詞と同じ解析用表記で照合する (半角カナ・結合濁点で書いても一致させる)
                    self.override[analysis_text(surf).text] = [
                        hira_to_kata(analysis_text(x).text) for x in read.strip().split('|')]

    def token_reading(self, surface: str, feature) -> str:
        """形態素の読み (カタカナ). かな のみなら表層ベース."""
        if surface in self.override:
            return "".join(self.override[surface])
        if is_kana_only(surface):
            surf_k = hira_to_kata(surface)
            pron = _known(getattr(feature, 'pron', None))
            if pron and len(split_moras_kata(pron)) == len(split_moras_kata(surf_k)):
                # 小書き仮名を潰された pron (ゅ->ユ) は採用しない
                if any(c in SMALL_KATA for c in surf_k) and not any(c in SMALL_KATA for c in pron):
                    return surf_k
                return pron
            return surf_k
        pron = _known(getattr(feature, 'pron', None)) or _known(getattr(feature, 'kana', None))
        if pron:
            return pron
        # 未知語: 読めない漢字は空 (その文字は後段で「読み未解決」として位置付きで報告する)
        return hira_to_kata("".join(c for c in surface if KANA_ANY.match(c)))


def _known(value):
    """UniDic の未定義値 ('*') は読みとして扱わない."""
    return value if value and value != '*' else None


def assign_moras_to_chars(surface: str, reading: str, base_idx: int, override: Optional[list[str]]) -> list[tuple[str, list[int], bool]]:
    """形態素の読み (カタカナ) を表層文字に割り当てる.

    返り値: [(mora_kana, [char indices], split_estimated)]
    """
    moras = split_moras_kata(reading)
    n = len(surface)
    if override:
        # override は文字ごとの読み ('景色 け|しき')
        out = []
        if len(override) == n:
            for ci, r in enumerate(override):
                for m in split_moras_kata(r):
                    out.append((m, [base_idx + ci], False))
            return out
    if is_kana_only(surface):
        surf_moras = split_moras_kata(hira_to_kata(surface))
        if len(surf_moras) == len(moras):
            out = []
            pos = 0
            for sm, m in zip(surf_moras, moras):
                out.append((m, [base_idx + pos + k for k in range(len(sm))], False))
                pos += len(sm)
            return out
        # 長さ不一致: 表層をそのまま使う
        out = []
        pos = 0
        for sm in surf_moras:
            out.append((sm, [base_idx + pos + k for k in range(len(sm))], False))
            pos += len(sm)
        return out
    # 漢字を含む: 送り仮名 (末尾・先頭のかな) を後方/前方一致で剥がす
    surf_k = hira_to_kata(surface)
    # 末尾
    tail = 0
    while tail < n and KANA_ANY.match(surf_k[n - 1 - tail]):
        tail += 1
    head = 0
    while head < n - tail and KANA_ANY.match(surf_k[head]):
        head += 1
    tail_moras = split_moras_kata(surf_k[n - tail:]) if tail else []
    head_moras = split_moras_kata(surf_k[:head]) if head else []
    mid = moras[:]
    # 末尾一致 (pron は長音化されている可能性 -> モーラ数で剥がし、表層側のかなを使う)
    tail_out = []
    if tail_moras and len(mid) >= len(tail_moras):
        tm = mid[len(mid) - len(tail_moras):]
        mid = mid[:len(mid) - len(tail_moras)]
        pos = n - tail
        for sm, m in zip(tail_moras, tm):
            tail_out.append((m, [base_idx + pos + k for k in range(len(sm))], False))
            pos += len(sm)
    head_out = []
    if head_moras and len(mid) >= len(head_moras):
        hm = mid[:len(head_moras)]
        mid = mid[len(head_moras):]
        pos = 0
        for sm, m in zip(head_moras, hm):
            head_out.append((m, [base_idx + pos + k for k in range(len(sm))], False))
            pos += len(sm)
    kanji_chars = list(range(head, n - tail))
    mid_out = []
    if not kanji_chars:
        # 読みが余った: 末尾に付ける
        for m in mid:
            mid_out.append((m, [base_idx + n - 1], True))
    elif len(kanji_chars) == 1:
        for m in mid:
            mid_out.append((m, [base_idx + kanji_chars[0]], False))
    elif len(mid) == len(kanji_chars):
        for m, ci in zip(mid, kanji_chars):
            mid_out.append((m, [base_idx + ci], False))
    else:
        # 各漢字に 1〜3 モーラを割り当てる組合せのうち、音読みの典型
        # (2 モーラ目が ン/ッ/ー/イ/ウ/キ/ク/チ/ツ) に合うものを選ぶ (推定フラグ)
        for m, ci in split_kanji_moras(mid, kanji_chars):
            mid_out.append((m, [base_idx + ci], True))
    return head_out + mid_out + tail_out


SECOND_MORA = set("ンッーイウキクチツ")


def split_kanji_moras(moras: list[str], kanji_chars: list[int]) -> list[tuple[str, int]]:
    import itertools
    k = len(kanji_chars); n = len(moras)
    best = None
    if k == 0:
        return []
    for comp in itertools.product([1, 2, 3], repeat=k):
        if sum(comp) != n:
            continue
        score = 0.0
        pos = 0
        for c in comp:
            grp = moras[pos:pos + c]
            if c == 1:
                score += 0.0
            elif c == 2:
                score += 0.0 if grp[1] in SECOND_MORA else 1.0
            else:
                score += 2.0 + (0.0 if grp[1] in SECOND_MORA else 1.0)
            pos += c
        if best is None or score < best[0]:
            best = (score, comp)
    if best is None:  # 3 モーラ超が必要: 均等按分
        out = []
        for mi, m in enumerate(moras):
            out.append((m, kanji_chars[min(k - 1, mi * k // max(1, n))]))
        return out
    out = []; pos = 0
    for ci, c in zip(kanji_chars, best[1]):
        for m in moras[pos:pos + c]:
            out.append((m, ci))
        pos += c
    return out


ENVELOPE_RUBY_OMITTED = "whole_spelling_reading_without_per_character_mapping"


def envelope_ruby_warning(groups) -> Optional[str]:
    """Warning for whole-spelling overrides ((source spelling, kana reading) pairs) whose ruby is omitted.

    lyrics-timing/1.0 importers attach ruby to one character each, so a reading of the whole spelling
    ('100' = ひゃく) cannot be exported as ruby without misattributing it to one character.
    """
    groups = list(dict.fromkeys(groups))
    if not groups:
        return None
    examples = ', '.join(f"{s!r}={r}" for s, r in groups[:5])
    return (f"{len(groups)} whole-spelling readings overrides have no per-character mapping ({examples}): "
            "their characters share one timing interval and no per-character ruby is exported "
            "(lyrics-timing/1.0 ruby is per character); the full reading stays in the result moras/syllables. "
            "For ruby, give one reading per character separated by '|' (e.g. '2人 ふた|り')")


def envelope_groups(lyr: "LyricsResult") -> list[tuple[str, str]]:
    """(source spelling, reading) of each whole-spelling override in a mora analysis."""
    groups: dict[tuple[int, ...], list[str]] = {}
    for m in lyr.moras:
        if m.envelope:
            groups.setdefault(tuple(m.char_idx), []).append(m.kana)
    return [(lyr.text[k[0]:k[-1] + 1], ''.join(v)) for k, v in groups.items()]


OVERRIDE_READING_RE = re.compile(r"[ァ-ヺー]+")
OVERRIDE_SUGGESTION = ("write the whole kana reading without '|' (e.g. '100 ひゃく'), or exactly one kana reading "
                       "per character separated by '|' (e.g. '言葉 こと|ば')")


def override_moras(surface: str, pieces: list[str], base_idx: int):
    """明示的な読み上書き -> ([(mora_kana, [char indices], split_estimated, envelope)], problem).

    上書きは決して表記どおりの読みへ黙って戻さない。対応付けの規則:
    - '|' 区切りで文字数と同じ個数: 各文字にその読み (景色 け|しき)。
    - '|' 無しの読み 1 つで 1 文字: その文字に全モーラ。
    - '|' 無しで複数文字:
      かなだけの表記でモーラ数が同じなら 1 モーラずつ対応 (表記と同じ並び)。
      漢字を含む表記は従来の送り仮名・漢字按分 (split_estimated) で全文字が読まれる場合だけそれを使う。
      それ以外 (数字・英字・記号を含む、モーラ数が違う、按分で読まれない文字が残る) は文字ごとの読みが
      分からないので、全モーラを表記全体に属させる (envelope)。文字ごとの発音開始やルビは作らない。
    - 安全に対応付けられない指定 ('|' の個数が文字数と違う、空の読み、かな以外の読み) は problem を返し、
      呼び出し側が原文位置付きの診断にする (表記どおりの読みで黙って成功しない)。
    """
    n = len(surface)
    if any(not p for p in pieces):
        return [], (f"the readings override for {surface!r} has an empty reading "
                    f"(per-character readings separated by '|' must each be nonempty kana)")
    bad = next((p for p in pieces if not OVERRIDE_READING_RE.fullmatch(p)), None)
    if bad is not None:
        return [], f"the readings override for {surface!r} must be written in kana (got {bad!r})"
    if len(pieces) == n:
        return [(m, [base_idx + ci], False, False) for ci, r in enumerate(pieces) for m in split_moras_kata(r)], None
    if len(pieces) > 1:
        return [], (f"the readings override for {surface!r} gives {len(pieces)} per-character readings "
                    f"separated by '|', but the spelling has {n} characters")
    reading = pieces[0]
    moras = split_moras_kata(reading)
    every = [base_idx + ci for ci in range(n)]
    if is_kana_only(surface):
        surf_moras = split_moras_kata(hira_to_kata(surface))
        if len(surf_moras) == len(moras):
            out, pos = [], 0
            for sm, m in zip(surf_moras, moras):
                out.append((m, [base_idx + pos + k for k in range(len(sm))], False, False))
                pos += len(sm)
            return out, None
    elif all(KANA_ANY.match(c) or KANJI.match(c) for c in surface):
        split = assign_moras_to_chars(surface, reading, base_idx, None)
        if [m for m, _, _ in split] == moras and {i for _, idx, _ in split for i in idx} == set(every):
            return [(m, idx, est, False) for m, idx, est in split], None
    return [(m, list(every), False, True) for m in moras], None


def _token_groups(ctext: str, reader: "Reader") -> list[tuple[str, Optional[str], Optional[list[str]]]]:
    """形態素を (表層, 読み, 上書き) のまとまりにする.

    かな・漢字だけの上書き表記は、連続する形態素の表層を連結したものと一致したときに適用する
    (かな の 1 形態素でも か|な の 2 形態素でも同じ)。形態素の途中では一致させない (部分一致で既存の読みを
    変えない)。連続する かな のみ形態素は 1 つにまとめる (モニゅっと 等の拗音分断対策)。上書きはまとめない。
    """
    tokens = [(w.surface, w.feature) for w in reader.tagger(ctext)]
    longest = max((len(k) for k in reader.override), default=0)
    groups: list[tuple[str, Optional[str], Optional[list[str]]]] = []
    t = 0
    while t < len(tokens):
        match, acc = None, ''
        for u in range(t, len(tokens)):
            acc += tokens[u][0]
            if len(acc) > longest:
                break
            if acc in reader.override:
                match = (u + 1, acc)
        if match:
            t, key = match
            groups.append((key, None, reader.override[key]))
            continue
        surf, feature = tokens[t]
        reading = reader.token_reading(surf, feature)
        if groups and groups[-1][2] is None and is_kana_only(surf) and is_kana_only(groups[-1][0]):
            groups[-1] = (groups[-1][0] + surf, groups[-1][1] + reading, None)
        else:
            groups.append((surf, reading, None))
        t += 1
    return groups


# ---------------------------------------------------------------- メイン
def _lexical_class(ch: str) -> Optional[str]:
    if unicodedata.category(ch) == 'Nd':
        return 'digit'
    if ch.isalpha():
        name = unicodedata.name(ch, '')
        if name.startswith('LATIN') or name.startswith('FULLWIDTH LATIN'):
            return 'latin'
        if name.startswith('HANGUL'):
            return 'hangul'
    return None


def _lexically_bounded(text: str, i: int, j: int) -> bool:
    """解析用テキストの [i, j) が、英字・ハングル・数字の語/数の途中で始まらず途中で終わらないか."""
    if j < len(text) and unicodedata.category(text[j])[0] == 'M':
        return False   # 書記素 (基底文字 + 結合文字) を切らない
    first, last = _lexical_class(text[i]), _lexical_class(text[j - 1])
    if first is not None and i > 0:
        p = i - 1
        while p > 0 and unicodedata.category(text[p])[0] == 'M':
            p -= 1
        if _lexical_class(text[p]) == first:
            return False
        if first == 'latin' and text[p] in "'’" and p > 0 and _lexical_class(text[p - 1]) == 'latin':
            return False   # don't の 't
    if last is not None and j < len(text):
        if _lexical_class(text[j]) == last:
            return False
        if last == 'latin' and text[j] in "'’" and j + 1 < len(text) and _lexical_class(text[j + 1]) == 'latin':
            return False   # don't の don
    return True


def override_spans(text: str, override: dict[str, list[str]], blocked=()) -> list[tuple[int, int, str]]:
    """読み上書きの表層が かな・漢字以外 (数字・記号・ヽ など) を含む場合の出現位置.

    これらの文字は形態素解析器の分割に依存せず上書き読みで読む ('2 ふた', 'カヽ か|か')。
    ただし英字・ハングル・数字で始まる/終わる表記は、同じ種類の文字が隣に続く位置 (より大きな語や数の一部:
    'ME' に対する 'HOME'、'I' に対する 'In'、'2' に対する '12'、"don't" の "don") では一致させない。
    結合文字の直前でも切らない。大文字・小文字は区別する (表記どおり)。
    かな・漢字だけの表層は形態素の境界にそろった位置だけで適用する (_token_groups。部分一致で既存の読みを
    変えないため)。左から最長一致・重複なし。
    """
    keys = sorted((k for k in override if any(not (KANA_ANY.match(c) or KANJI.match(c)) for c in k)),
                  key=len, reverse=True)
    if not keys:
        return []
    blocked = set(blocked)
    spans = []
    i = 0
    while i < len(text):
        for k in keys:
            if (text.startswith(k, i) and not any(j in blocked for j in range(i, i + len(k)))
                    and _lexically_bounded(text, i, i + len(k))):
                spans.append((i, i + len(k), k))
                i += len(k)
                break
        else:
            i += 1
    return spans


def parse_lyrics(text: str, override_path: Optional[Path] = None, tokenizer: Optional[Callable] = None,
                 reader: Optional[Reader] = None) -> LyricsResult:
    """tokenizer(mora, next_mora) -> list[str] でユニットのトークン列を決める (省略時はローマ字).

    読みは解析用テキスト上で求め、Mora.char_idx などの位置はすべて原文のコードポイント位置で返す。
    読めなかった文字は黙って捨てず、lyrics_issues() で位置付きの診断になる。
    """
    reader = reader or Reader(override_path)
    norm = analysis_text(text)
    a = norm.text
    annotated = annotation_indices(a)  # 括弧・タグ内、© 行 (多言語経路と共通)
    forced = {s: (e, k) for s, e, k in override_spans(a, reader.override, annotated)}
    unsung: set[int] = set(annotated)
    unread: dict[int, str] = {}
    problems: list[dict] = []
    moras: list[Mora] = []
    # 行区切りは lyric_input.line_spans (CRLF/LF/CR/U+2028 など) を全経路で共有する。
    # 解析用テキストは改行文字を変えないので、解析用テキストの行番号 = 原文の行番号。
    for li, (pos, line_end, _) in enumerate(line_spans(a)):
        line = a[pos:line_end]
        # セグメント: 空白 (全角含む) 区切り
        seg_no = 0
        for seg_m in re.finditer(r"[^\s　]+", line):
            seg_start = pos + seg_m.start()
            seg_end = pos + seg_m.end()
            # 非発音文字を除いた発音対象部分を、連続する発音文字のまとまりごとに処理
            # (記号で分断された「タ・タ・タ」は各まとまりが 1 文字)
            chunk = ""; chunk_idx: list[int] = []
            chunks: list[tuple[str, list[int], Optional[str]]] = []
            gi = seg_start
            while gi < seg_end:
                c = a[gi]
                if gi in forced and forced[gi][0] <= seg_end:
                    if chunk:
                        chunks.append((chunk, chunk_idx, None)); chunk = ""; chunk_idx = []
                    end, key = forced[gi]
                    chunks.append((key, list(range(gi, end)), key))
                    gi = end
                    continue
                pronounceable = (gi not in annotated) and (KANA_ANY.match(c) or KANJI.match(c) or c in 'ヽヾゝゞ')
                if pronounceable:
                    chunk += c; chunk_idx.append(gi)
                else:
                    unsung.add(gi)
                    if chunk:
                        chunks.append((chunk, chunk_idx, None)); chunk = ""; chunk_idx = []
                gi += 1
            if chunk:
                chunks.append((chunk, chunk_idx, None))
            # 上書き表記で区切っただけのまとまりは同じ連続した発音文字の続き (OK|ー, 2|っ)。
            # 空白・句読点・記号・注釈で区切られたまとまりだけを新しい run の先頭にする。
            prev_end, prev_sounded = None, False
            for ctext, cidx, key in chunks:
                off = 0
                first = len(moras)
                if key is not None:
                    groups = [(key, None, reader.override[key])]
                else:
                    groups = _token_groups(ctext, reader)
                for surf, reading, ov in groups:
                    if ov is not None:
                        assigned, problem = override_moras(surf, ov, off)
                        if problem:
                            problems.append(dict(analysis_start=cidx[off], analysis_end=cidx[off + len(surf) - 1] + 1,
                                                 key=surf, message=problem, suggestion=OVERRIDE_SUGGESTION))
                    else:
                        assigned = [(mk, rel, est, False) for mk, rel, est in assign_moras_to_chars(surf, reading, off, None)]
                    covered = set()
                    for mk, rel_idx, est, env in assigned:
                        rom, kind = mora_romaji(mk)
                        covered.update(rel_idx)
                        moras.append(Mora(id=len(moras), kana=mk, romaji=rom,
                                          char_idx=norm.source_indices(cidx[i] for i in rel_idx), line=li, seg=seg_no,
                                          kind=kind, split_estimated=est, envelope=env,
                                          analysis_idx=sorted(cidx[i] for i in rel_idx)))
                    missing = [cidx[r] for r in range(off, off + len(surf)) if r not in covered]
                    if missing:
                        # 診断で「どの表層に読みを与えればよいか」を示すため、形態素の原文表記を残す
                        s0, s1 = norm.source_span(cidx[off], cidx[off + len(surf) - 1] + 1)
                        for si in norm.source_indices(missing):
                            unread[si] = text[s0:s1]
                    off += len(surf)
                contiguous = prev_sounded and cidx[0] == prev_end
                if len(moras) > first and not contiguous:
                    moras[first].run_start = True
                prev_end, prev_sounded = cidx[-1] + 1, len(moras) > first
            seg_no += 1
    lines = [(start, end) for start, end, _ in line_spans(text)]
    units = build_ctc_units(moras, tokenizer)
    return LyricsResult(text=text, chars=list(text), lines=lines, moras=moras, units=units,
                        unsung_chars=set(norm.source_indices(unsung)), analysis=norm,
                        annotated_chars=set(norm.source_indices(annotated)), unread_tokens=unread,
                        override_problems=problems)


def lyrics_issues(lyr: LyricsResult) -> list[dict]:
    """モーラ解析で表現されなかった発音対象と、音を持たないモーラを位置付きで返す.

    - 発音対象 (文字・数字・その結合記号) がどのモーラにも属さない
    - ー (と音を持たない ッ) に直前の音が無い: 連続した発音文字の先頭や行頭
      (空白・句読点・改行をまたいで前の語へ黙って結合しない)
    - 通常モーラなのにモデルの音素が無い (読めないかな・'*' など)。長音のように直前へ畳まない
    """
    src = lyr.text
    issues: list[dict] = []
    reported: set[int] = set()
    for p in lyr.override_problems:
        # 読み上書きを安全に対応付けられない: 表記どおりの読みで黙って成功させず、上書きした表記の位置で報告する
        if lyr.analysis is not None:
            s, e = lyr.analysis.source_span(p['analysis_start'], p['analysis_end'])
        else:
            s, e = p['analysis_start'], p['analysis_end']
        issues.append(make_issue(src, s, e, 'reading_override_unmappable', p['message'], p['suggestion']))
        reported.update(range(s, e))
    for m in lyr.moras:
        a, b = min(m.char_idx), max(m.char_idx) + 1
        if m.kind == 'choon' and (m.run_start or m.ctc_unit is None):
            issues.append(make_issue(src, a, b, 'no_preceding_sound',
                                     'long-vowel mark has no preceding sound in the same word',
                                     'write the vowel to be sung or attach the mark to the preceding word'))
        elif m.kind in ('sokuon', 'hatsuon') and m.ctc_unit is None:
            # トークンを持たない ッ/ン は同じまとまりの直前の音にだけ畳む。空白・句読点・改行の向こうの音は借りない
            issues.append(make_issue(src, a, b, 'no_preceding_sound',
                                     f'{m.kana} has no phone of its own in this model and no preceding sound '
                                     'in the same word to attach to',
                                     'attach it to the preceding word, or use a phoneme model'))
        elif m.kind == 'normal' and not m.tokens:
            issues.append(make_issue(src, a, b, 'no_model_phones',
                                     f'reading {m.kana!r} cannot be represented completely by the acoustic model '
                                     'inventory (no phone sequence covering the whole mora)',
                                     f"add a readings override line such as '{src[a:b]} <kana reading>'"))
        else:
            continue
        reported.update(range(a, b))
    # 原文の 1 文字が複数の解析文字に属する場合 (合成できない結合文字・正準分解で 2 文字になる記号)、
    # その全部が読まれたときだけ被覆とみなす (あ だけ読めても あ+U+3099 の濁点は被覆にならない)
    if lyr.analysis is not None and all(m.analysis_idx for m in lyr.moras):
        covered = lyr.analysis.covered_source(i for m in lyr.moras for i in m.analysis_idx)
    else:
        covered = {i for m in lyr.moras for i in m.char_idx}
    issues += coverage_issues(src, covered, lyr.annotated_chars, 'japanese', lyr.unread_tokens, reported)
    return sorted(issues, key=lambda x: (x['char_start'], x['char_end']))


def prepare_mora_lyrics(text: str, override_path: Optional[Path] = None,
                        tokenizer: Optional[Callable] = None) -> LyricsResult:
    """従来のモーラ解析の入口. 未解決の入力があれば位置付きの LyricsInputError にする."""
    lyr = parse_lyrics(text, override_path, tokenizer)
    issues = lyrics_issues(lyr)
    if issues:
        raise LyricsInputError(issues)
    return lyr


def romaji_tokens(m: Mora, nxt: Optional[Mora]) -> list[str]:
    """ローマ字の既定トークン化 (hf_ctc は aligner.mora_tokens を使う): ー/っ はトークン無し (直前に畳む), ん は n (母音の前は n')."""
    if m.kind in ('choon', 'sokuon'):
        return []
    if m.kind == 'hatsuon':
        if nxt is not None and nxt.line == m.line and nxt.romaji and nxt.romaji[0] in 'aiueoy':
            return ["n'"]
        return ['n']
    return [m.romaji] if m.romaji else []


def build_ctc_units(moras: list[Mora], tokenizer: Optional[Callable] = None) -> list[CtcUnit]:
    """モーラ列 -> CTC ユニット列. 行の境界に <star> ユニットを入れる.

    tokenizer(mora, next_mora) が空リストを返した特殊モーラ (ー/っ/ん) は、同じ連続した発音文字の
    まとまり (run_start で区切る) の直前ユニットにだけ畳む (モーラ層では独立モーラのまま、多対一で対応付ける)。
    まとまりの先頭 (空白・句読点・改行の後) では畳まず ctc_unit=None のまま残し、lyrics_issues が
    位置付きで報告する。通常モーラ (音素の無い未知の読み) は決して畳まない."""
    tok = tokenizer or romaji_tokens
    units: list[CtcUnit] = []
    prev_line = None
    for i, m in enumerate(moras):
        m.ctc_unit = None
        if m.line != prev_line:
            units.append(CtcUnit(id=len(units), text='*', mora_ids=[], is_star=True, line=m.line, tokens=['*']))
            prev_line = m.line
        tokens = tok(m, moras[i + 1] if i + 1 < len(moras) else None)
        m.tokens = list(tokens)
        if not tokens:
            prev = moras[i - 1] if i > 0 else None
            if (m.kind in ('choon', 'sokuon', 'hatsuon') and not m.run_start and prev is not None
                    and units and not units[-1].is_star and prev.ctc_unit == units[-1].id):
                units[-1].mora_ids.append(m.id)
                m.ctc_unit = units[-1].id
            continue
        units.append(CtcUnit(id=len(units), text="".join(tokens), mora_ids=[m.id], line=m.line, tokens=list(tokens)))
        m.ctc_unit = units[-1].id
    units.append(CtcUnit(id=len(units), text='*', mora_ids=[], is_star=True, line=-1, tokens=['*']))
    return units
