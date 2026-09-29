"""歌詞テキストの正規化・読み付与・モーラ分割・文字対応.

出力の中心は Mora のリスト。各 Mora は元テキストの文字インデックス集合を持つ。
CTC 用のローマ字ユニット (CtcUnit) はモーラとは多対一 (ー/っ は独立トークンを持たない)。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import fugashi

# ---------------------------------------------------------------- 文字クラス
HIRAGANA = re.compile(r"[ぁ-ゖ]")
KATAKANA = re.compile(r"[ァ-ヺ]")
KANA_ANY = re.compile(r"[ぁ-ゖァ-ヺー]")  # ー を含む
KANJI = re.compile(r"[一-鿿㐀-䶿々]")  # 々 含む
# 括弧・タグ: 中身ごと非発音
BRACKET_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）|<[^>]*>|【[^】]*】|《[^》]*》|〈[^〉]*〉|〔[^〕]*〕|｛[^｝]*｝|\{[^}]*\}")

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
    lines: list[tuple[int, int]]        # 行ごとの (start_char, end_char)
    moras: list[Mora]
    units: list[CtcUnit]
    unsung_chars: set[int] = field(default_factory=set)  # 非発音文字


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
                    self.override[surf] = [hira_to_kata(x) for x in read.strip().split('|')]

    def token_reading(self, surface: str, feature) -> str:
        """形態素の読み (カタカナ). かな のみなら表層ベース."""
        if surface in self.override:
            return "".join(self.override[surface])
        if is_kana_only(surface):
            surf_k = hira_to_kata(surface)
            pron = getattr(feature, 'pron', None)
            if pron and len(split_moras_kata(pron)) == len(split_moras_kata(surf_k)):
                # 小書き仮名を潰された pron (ゅ->ユ) は採用しない
                if any(c in SMALL_KATA for c in surf_k) and not any(c in SMALL_KATA for c in pron):
                    return surf_k
                return pron
            return surf_k
        pron = getattr(feature, 'pron', None) or getattr(feature, 'kana', None)
        if pron:
            return pron
        # 未知語: 読めない漢字は空
        return hira_to_kata("".join(c for c in surface if KANA_ANY.match(c)))


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


# ---------------------------------------------------------------- メイン
def parse_lyrics(text: str, override_path: Optional[Path] = None, tokenizer: Optional[Callable] = None) -> LyricsResult:
    """tokenizer(mora, next_mora) -> list[str] でユニットのトークン列を決める (省略時はローマ字)."""
    reader = Reader(override_path)
    chars = list(text)
    unsung: set[int] = set()
    # 括弧・タグ内を非発音に
    for m in BRACKET_RE.finditer(text):
        unsung.update(range(m.start(), m.end()))
    moras: list[Mora] = []
    lines: list[tuple[int, int]] = []
    pos = 0
    for li, line in enumerate(text.split("\n")):
        lines.append((pos, pos + len(line)))
        # セグメント: 空白 (全角含む) 区切り
        seg_no = 0
        for seg_m in re.finditer(r"[^\s　]+", line):
            seg_start = pos + seg_m.start()
            seg_text = seg_m.group()
            # 非発音文字を除いた発音対象部分を、連続する発音文字のまとまりごとに処理
            # (記号で分断された「タ・タ・タ」は各まとまりが 1 文字)
            chunk = ""; chunk_idx: list[int] = []
            chunks: list[tuple[str, list[int]]] = []
            for k, c in enumerate(seg_text):
                gi = seg_start + k
                pronounceable = (gi not in unsung) and (KANA_ANY.match(c) or KANJI.match(c) or c in 'ヽヾゝゞ')
                if pronounceable:
                    chunk += c; chunk_idx.append(gi)
                else:
                    unsung.add(gi)
                    if chunk:
                        chunks.append((chunk, chunk_idx)); chunk = ""; chunk_idx = []
            if chunk:
                chunks.append((chunk, chunk_idx))
            for ctext, cidx in chunks:
                off = 0
                # 連続する かな のみ形態素は 1 つにまとめる (モニゅっと 等の拗音分断対策)
                groups: list[tuple[str, str]] = []
                for w in reader.tagger(ctext):
                    surf = w.surface
                    reading = reader.token_reading(surf, w.feature)
                    if groups and is_kana_only(surf) and is_kana_only(groups[-1][0]) and surf not in reader.override:
                        groups[-1] = (groups[-1][0] + surf, groups[-1][1] + reading)
                    else:
                        groups.append((surf, reading))
                for surf, reading in groups:
                    ov = reader.override.get(surf)
                    for mk, rel_idx, est in assign_moras_to_chars(surf, reading, off, ov):
                        rom, kind = mora_romaji(mk)
                        moras.append(Mora(id=len(moras), kana=mk, romaji=rom,
                                          char_idx=[cidx[i] for i in rel_idx], line=li, seg=seg_no,
                                          kind=kind, split_estimated=est))
                    off += len(surf)
            seg_no += 1
        pos += len(line) + 1
    units = build_ctc_units(moras, tokenizer)
    return LyricsResult(text=text, chars=chars, lines=lines, moras=moras, units=units, unsung_chars=unsung)


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

    tokenizer(mora, next_mora) が空リストを返したモーラ (ー/っ など) は直前ユニットに畳む
    (モーラ層では独立モーラのまま、多対一でユニットに対応付ける)."""
    tok = tokenizer or romaji_tokens
    units: list[CtcUnit] = []
    prev_line = None
    for i, m in enumerate(moras):
        m.ctc_unit = None
        if m.line != prev_line:
            units.append(CtcUnit(id=len(units), text='*', mora_ids=[], is_star=True, line=m.line, tokens=['*']))
            prev_line = m.line
        tokens = tok(m, moras[i + 1] if i + 1 < len(moras) else None)
        if not tokens:
            if units and not units[-1].is_star:
                units[-1].mora_ids.append(m.id)
                m.ctc_unit = units[-1].id
            continue
        units.append(CtcUnit(id=len(units), text="".join(tokens), mora_ids=[m.id], line=m.line, tokens=list(tokens)))
        m.ctc_unit = units[-1].id
    units.append(CtcUnit(id=len(units), text='*', mora_ids=[], is_star=True, line=-1, tokens=['*']))
    return units
