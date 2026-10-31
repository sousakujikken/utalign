"""Source-preserving lyric analysis text, annotations and coverage diagnostics.

The original lyric string is never modified or replaced. Readers work on a
separate *analysis text* that differs only by:

- Unicode canonical composition (NFC), e.g. か + U+3099 -> が, e + U+0301 -> é;
- halfwidth katakana/punctuation (U+FF61-U+FF9F) -> fullwidth, with halfwidth
  voiced marks composed into the preceding kana (ｶﾞ -> ガ, ﾊﾟ -> パ);
- spacing voiced marks ゛゜ after kana composed like their combining forms;
- fullwidth ASCII letters/digits -> ASCII (ＡＢＣ -> ABC);
- variation selectors folded into the preceding character.

NFC is computed as the real canonical algorithm (full canonical decomposition,
canonical reordering of combining marks, canonical composition), so canonically
equivalent spellings give the same analysis text (a+U+0301+U+0323 and
a+U+0323+U+0301 both give ạ+U+0301). Every analysis character carries the exact
set of source code points it was made from (``members``):

- composition unions the members (ạ <- {a, U+0323});
- reordering can make that set non-contiguous (a, U+0301, U+0323 -> ạ = {0, 2},
  U+0301 = {1});
- a canonical expansion (U+0344 -> U+0308 U+0301) gives several analysis
  characters that share one source code point.

So the map is many-to-many: every source code point belongs to at least one
analysis character (a variation selector belongs to the character it follows),
not to exactly one. A source code point counts as represented only when *every*
analysis character containing it is represented (``covered_source``), so reading
あ never makes an uncomposable U+3099 after it look covered. ``origin`` keeps the
contiguous hull [min, max+1) of the members for display.

No other compatibility mapping (NFKC) is applied: symbols keep their identity.
Which symbols must be sung (or explicitly annotated / given a readings override)
is decided by ``requires_reading``. Only these classes are diagnosed when nothing
reads them: enclosed/squared CJK symbols (U+3200-U+33FF, ㍑), currency signs
(category Sc: ¥ $ € ＄), letterlike symbols U+2100-U+214F (℃ ™ № ℓ) and the
lexical symbols in ``READING_SYMBOLS`` (% ％ & ＆ @ ＠ ° ∞ ...), plus enclosed
numbers such as ① (category N). Their sung reading is ambiguous, so none is
invented. Ordinary punctuation, emoji, arrows and other symbols are deliberately
treated as not sung and are not diagnosed; not every Unicode symbol is covered.

Lines are delimited the same way everywhere (parsing, hints, diagnostics and
results): CRLF, LF, CR, U+2028, U+2029 and the other ``str.splitlines``
separators. The delimiters stay in the source text; only positions use them.

Both the Japanese mora path and the multilingual syllable path use the same
annotation mask and the same "must be represented" rule, so unsupported or
unreadable characters fail with a line/column instead of disappearing.
"""
from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass, field

import regex

# Brackets/tags: their content is a deliberate annotation and is not sung.
BRACKET_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）|<[^>]*>|【[^】]*】|《[^》]*》|〈[^〉]*〉|〔[^〕]*〕|｛[^｝]*｝|\{[^}]*\}")

_SPACING_VOICING = {'゛': '゙', '゜': '゚'}
_HALFWIDTH_VOICING = 'ﾞﾟ'

# The same separators as str.splitlines (CRLF is one delimiter).
LINE_BREAK_RE = re.compile('\r\n|[\n\r\x0b\x0c\x1c\x1d\x1e\x85  ]')
LINE_BREAK_CHARS = frozenset('\n\r\x0b\x0c\x1c\x1d\x1e\x85  ')


def line_spans(text: str) -> list[tuple[int, int, int]]:
    """Physical lines: (content_start, content_end, next_line_start); the delimiter is [content_end, next)."""
    spans = []
    pos = 0
    for m in LINE_BREAK_RE.finditer(text):
        spans.append((pos, m.start(), m.end()))
        pos = m.end()
    spans.append((pos, len(text), len(text)))
    return spans


def line_column(text: str, index: int) -> tuple[int, int]:
    """1-based (line, column) of a code-point index; a delimiter belongs to the line it ends."""
    starts = [s for s, _, _ in line_spans(text)]
    li = bisect.bisect_right(starts, index) - 1
    return li + 1, index - starts[li] + 1


def line_index(text: str) -> list[int]:
    """0-based line number of every code point (delimiters belong to the line they end)."""
    out = [0] * len(text)
    for li, (s, _, n) in enumerate(line_spans(text)):
        for i in range(s, n):
            out[i] = li
    return out


def _is_variation_selector(ch: str) -> bool:
    o = ord(ch)
    return 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF or 0x180B <= o <= 0x180D


def _width(ch: str) -> str:
    o = ord(ch)
    if 0xFF61 <= o <= 0xFF9F:  # halfwidth katakana, voiced marks and CJK punctuation only
        return unicodedata.normalize('NFKC', ch)
    if 0xFF10 <= o <= 0xFF19 or 0xFF21 <= o <= 0xFF3A or 0xFF41 <= o <= 0xFF5A:
        return chr(o - 0xFEE0)
    return ch


@dataclass
class AnalysisText:
    source: str
    text: str
    members: list[tuple[int, ...]]  # analysis index -> exact sorted source code points (many-to-many)
    origin: list[tuple[int, int]] = field(default_factory=list)  # hull [min, max+1) of members

    def __post_init__(self):
        if not self.origin:
            self.origin = [(m[0], m[-1] + 1) for m in self.members]
        owners: dict[int, list[int]] = {}
        for i, m in enumerate(self.members):
            for k in m:
                owners.setdefault(k, []).append(i)
        self.owners = owners  # source code point -> analysis indices containing it
        # prefix max / suffix min of source points, for O(1) boundary safety checks
        self._prefix_max, self._suffix_min = [], [0] * len(self.members)
        for m in self.members:
            self._prefix_max.append(max(m[-1], self._prefix_max[-1] if self._prefix_max else -1))
        low = len(self.source)
        for i in range(len(self.members) - 1, -1, -1):
            low = min(low, self.members[i][0])
            self._suffix_min[i] = low

    def source_span(self, a: int, b: int) -> tuple[int, int]:
        """Analysis [a, b) -> hull of their source code points [start, end)."""
        if a >= b:
            s = self.origin[a][0] if a < len(self.origin) else len(self.source)
            return s, s
        return (min(self.origin[i][0] for i in range(a, b)), max(self.origin[i][1] for i in range(a, b)))

    def source_indices(self, indices) -> list[int]:
        return sorted({k for i in indices for k in self.members[i]})

    def covered_source(self, analysis_indices) -> set[int]:
        """Source code points all of whose analysis characters are in ``analysis_indices``.

        A source point shared with an unrepresented analysis character (an uncomposable
        mark, the second half of a canonical expansion) is not covered.
        """
        chosen = set(analysis_indices)
        return {k for k in self.source_indices(chosen) if all(i in chosen for i in self.owners[k])}

    def is_boundary(self, k: int) -> bool:
        """Whether analysis position k can delimit a span (hint, word) without splitting source text.

        False inside a grapheme cluster (before a combining mark, inside CRLF) or where reordered /
        expanded source code points would be shared by both sides.
        """
        if k <= 0 or k >= len(self.text):
            return True
        # A shared source point (canonical expansion) or a reordered one (a+U+0301+U+0323) straddles k.
        if self._prefix_max[k - 1] >= self._suffix_min[k]:
            return False
        nxt = self.text[k]
        if unicodedata.category(nxt)[0] == 'M' or _is_variation_selector(nxt) or nxt in 'ﾞﾟ':
            return False
        return not (self.text[k - 1] == '\r' and nxt == '\n')

    def changes(self) -> list[dict]:
        """Analysis characters whose spelling differs from the source."""
        out = []
        for i, m in enumerate(self.members):
            spelled = ''.join(self.source[k] for k in m)
            if spelled != self.text[i] or m != tuple(range(m[0], m[-1] + 1)):
                s, e = self.origin[i]
                out.append(dict(char_start=s, char_end=e, source=self.source[s:e], analysis=self.text[i],
                                source_chars=list(m)))
        return out


def _ccc(ch: str) -> int:
    return unicodedata.combining(ch)


def _compose_pair(starter: str, ch: str) -> str | None:
    """Canonical (primary) composite of starter+ch, or None."""
    composed = unicodedata.normalize('NFC', starter + ch)
    return composed if len(composed) == 1 else None


def _canonical_compose(items: list[tuple[str, frozenset]]) -> list[tuple[str, frozenset]]:
    """NFC of a tagged code-point sequence; every output char carries the union of its sources."""
    # 1. full canonical decomposition, each decomposed code point keeps its source tags
    seq = [(d, tags) for ch, tags in items for d in unicodedata.normalize('NFD', ch)]
    # 2. canonical ordering: stable sort of each run of non-starters by combining class
    i = 0
    while i < len(seq):
        if _ccc(seq[i][0]) == 0:
            i += 1
            continue
        j = i
        while j < len(seq) and _ccc(seq[j][0]) != 0:
            j += 1
        seq[i:j] = sorted(seq[i:j], key=lambda x: _ccc(x[0]))
        i = j
    # 3. canonical composition (UAX #15): a mark composes with the last starter unless blocked
    out: list[list] = []
    starter = None      # index in out of the last starter
    last_class = -1     # -1: nothing between the starter and the current char
    for ch, tags in seq:
        cc = _ccc(ch)
        if starter is not None and not (last_class != -1 and (last_class == 0 or last_class >= cc)):
            composite = _compose_pair(out[starter][0], ch)
            if composite is not None:
                out[starter][0] = composite
                out[starter][1] = out[starter][1] | tags
                continue
        if cc == 0:
            starter, last_class = len(out), -1
        else:
            last_class = cc
        out.append([ch, tags])
    return [(ch, tags) for ch, tags in out]


def _fold_spacing_voicing(items: list[tuple[str, frozenset]]) -> list[tuple[str, frozenset]]:
    """Spacing ゛゜ compose with a preceding kana exactly like U+3099/U+309A (か゛ -> が)."""
    out: list[tuple[str, frozenset]] = []
    for ch, tags in items:
        if ch in _SPACING_VOICING and out and out[-1][0] not in LINE_BREAK_CHARS:
            composite = _compose_pair(out[-1][0], _SPACING_VOICING[ch])
            if composite is not None:
                out[-1] = (composite, out[-1][1] | tags)
                continue
        out.append((ch, tags))
    return out


def analysis_text(source: str) -> AnalysisText:
    items: list[tuple[str, frozenset]] = []
    for i, c in enumerate(source):
        if items and items[-1][0] not in LINE_BREAK_CHARS and _is_variation_selector(c):
            # folded into the preceding code point (and whatever it composes into)
            items[-1] = (items[-1][0], items[-1][1] | {i})
            continue
        items.extend((p, frozenset((i,))) for p in _width(c))
    plain = ''.join(ch for ch, _ in items)
    items = _canonical_compose(items)
    if ''.join(ch for ch, _ in items) != unicodedata.normalize('NFC', plain):
        raise RuntimeError(f'internal error: canonical composition disagrees with NFC for {source!r}')
    folded = _fold_spacing_voicing(items)
    if len(folded) != len(items):
        items = folded
        text = ''.join(ch for ch, _ in items)
        if unicodedata.normalize('NFC', text) != text:
            items = _canonical_compose(items)
    text = ''.join(ch for ch, _ in items)
    return AnalysisText(source, text, [tuple(sorted(tags)) for _, tags in items])


def annotation_indices(text: str) -> set[int]:
    """Bracketed annotations and copyright lines (same rule for every path)."""
    excluded = {i for m in BRACKET_RE.finditer(text) for i in range(m.start(), m.end())}
    for start, end, _ in line_spans(text):
        if text[start:end].lstrip().startswith('©'):
            excluded.update(range(start, end))
    return excluded


# Lexical symbols that are normally pronounced as a word (percent, and, at, degree, infinity)
# but whose sung reading depends on the language and context: never silently unsung, never guessed.
READING_SYMBOLS = frozenset('%％﹪‰‱&＆﹠@＠﹫°∞')


def is_reading_symbol(ch: str) -> bool:
    """A symbol that stands for a word (currency, letterlike unit, READING_SYMBOLS, squared CJK)."""
    o = ord(ch)
    return (unicodedata.category(ch) == 'Sc' or 0x2100 <= o <= 0x214F or ch in READING_SYMBOLS
            or (unicodedata.category(ch) == 'So' and 0x3200 <= o <= 0x33FF))


def _base_requires_reading(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat[0] in 'LN' or is_reading_symbol(ch)


def mark_base(text: str, i: int) -> int | None:
    """Index of the character a combining mark at i modifies (skipping other marks/selectors), or None."""
    j = i - 1
    while j >= 0 and (unicodedata.category(text[j])[0] == 'M' or _is_variation_selector(text[j])):
        j -= 1
    if j < 0 or text[j] in LINE_BREAK_CHARS:
        return None
    return j


def is_orphan_mark(text: str, i: int) -> bool:
    """A sound-modifying combining mark (Mn/Mc, e.g. U+3099, U+0301) with no letter/number base.

    At the start of a line, after a space, punctuation or an emoji it modifies nothing
    pronounceable; it is an input error, not a silent no-op. Variation selectors and
    enclosing marks (Me, e.g. the emoji keycap U+20E3) are presentation, not sound.
    Halfwidth ﾞﾟ count as marks here: the analysis text maps them to U+3099/U+309A.
    """
    ch = text[i]
    if _is_variation_selector(ch) or (unicodedata.category(ch) not in ('Mn', 'Mc') and ch not in _HALFWIDTH_VOICING):
        return False
    base = mark_base(text, i)
    return base is None or not _base_requires_reading(text[base])


def requires_reading(text: str, i: int) -> bool:
    """Characters that must be sung (or explicitly annotated), never silently dropped.

    Letters, numbers, and combining marks on such characters; symbols whose reading
    is a word (is_reading_symbol: squared CJK ㍑, currency ¥ $ €, letterlike ℃ ™,
    % & @ ° ∞); and orphan sound-modifying marks (see is_orphan_mark).
    Punctuation, other symbols, emoji, variation selectors, spaces and format
    characters are deliberately treated as not pronounced.
    """
    ch = text[i]
    if _is_variation_selector(ch):
        return False
    cat = unicodedata.category(ch)
    if cat[0] == 'M':
        base = mark_base(text, i)
        if base is not None and _base_requires_reading(text[base]):
            return True
        return is_orphan_mark(text, i)
    return _base_requires_reading(ch)


_JA_KANA_RE = regex.compile(r'[\p{Hiragana}\p{Katakana}ーヽヾゝゞ]')


def _reason(source: str, i: int, mode: str) -> tuple[str, str]:
    ch = source[i]
    cat = unicodedata.category(ch)
    if cat[0] == 'N':
        return 'number_needs_reading', 'numbers are not given an automatic sung reading'
    if regex.match(r'\p{Han}', ch) or ch in '々〆〇':
        return 'no_dictionary_reading', 'the Japanese dictionary has no reading for this character here'
    if is_reading_symbol(ch) or cat == 'So':
        return 'symbol_needs_reading', ('this symbol stands for a word whose sung reading is ambiguous; '
                                        'no reading is invented for it')
    if cat[0] == 'M' or ch in _HALFWIDTH_VOICING:
        if is_orphan_mark(source, i):
            return 'orphan_mark', ('combining mark has no letter to modify (at the start of a line, '
                                   'or after a space, punctuation or symbol)')
        return 'unattached_mark', 'combining mark cannot be composed with the preceding character'
    if _JA_KANA_RE.match(ch):
        return 'kana_without_reading', 'this kana or mark has no reading in this position'
    if mode == 'japanese' and regex.match(r'[\p{Latin}\p{Hangul}]', ch):
        return 'script_needs_multilingual', 'Japanese mora mode cannot pronounce this script'
    return 'unsupported_script', 'no enabled language can pronounce this script'


def _suggestion(reason: str, text: str, token: str | None) -> str | None:
    key = token or text
    if reason == 'symbol_needs_reading':
        return (f"add a readings override line such as '{key} <kana reading>' if it is sung "
                f"(the lyric text itself is kept), or put it in brackets as an annotation if it is not sung")
    if reason in ('number_needs_reading', 'no_dictionary_reading', 'kana_without_reading'):
        return (f"add a readings override line such as '{key} <kana reading>' "
                f"(the lyric text itself is kept), or write the sung words")
    if reason == 'script_needs_multilingual':
        return 'use alignment mode auto or multilingual for non-Japanese words'
    if reason == 'orphan_mark':
        return 'write the mark directly after the character it modifies (e.g. か゛), or remove it'
    return None


def make_issue(source: str, start: int, end: int, reason: str, message: str, suggestion: str | None = None) -> dict:
    line, column = line_column(source, start)
    return dict(line=line, column=column, char_start=start, char_end=end, text=source[start:end],
                reason=reason, message=message, suggestion=suggestion)


def coverage_issues(source: str, covered, annotated, mode: str, token_at: dict | None = None,
                    already_reported=()) -> list[dict]:
    """Group source characters that need a reading but are represented by nothing."""
    token_at = token_at or {}
    skip = set(covered) | set(annotated) | set(already_reported)
    issues: list[dict] = []
    current = None
    for i, ch in enumerate(source):
        if i in skip or not requires_reading(source, i):
            current = None
            continue
        if current and current['char_end'] == i and unicodedata.category(ch)[0] == 'M' and not is_orphan_mark(source, i):
            # A combining mark on an unrepresented base belongs to the base's span (é = e + U+0301).
            current['char_end'] = i + 1
            current['text'] = source[current['char_start']:i + 1]
            continue
        reason, message = _reason(source, i, mode)
        if current and current['char_end'] == i and current['reason'] == reason:
            current['char_end'] = i + 1
            current['text'] = source[current['char_start']:i + 1]
            continue
        current = make_issue(source, i, i + 1, reason, message)
        issues.append(current)
    for issue in issues:
        token = next((token_at[i] for i in range(issue['char_start'], issue['char_end']) if i in token_at), None)
        issue['suggestion'] = _suggestion(issue['reason'], issue['text'], token)
    return issues


class LyricsInputError(ValueError):
    """Unresolved lyric input with original positions. Nothing was silently dropped."""

    def __init__(self, issues: list[dict]):
        self.issues = sorted(issues, key=lambda x: (x['char_start'], x['char_end']))
        parts = []
        for x in self.issues[:8]:
            part = f"line {x['line']} col {x['column']} {x['text']!r}: {x['message']} [{x['reason']}]"
            if x.get('suggestion'):
                part += f" ({x['suggestion']})"
            parts.append(part)
        more = f'; and {len(self.issues) - 8} more' if len(self.issues) > 8 else ''
        super().__init__('Unsupported lyric characters / unresolved lyric input: ' + '; '.join(parts) + more)
