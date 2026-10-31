"""Lyric input integrity: source spelling, readings and source mapping.

Generic synthetic lyric inputs run over four paths:

- japanese:     legacy mora analysis (prepare_mora_lyrics, as run_align uses it)
- multilingual: the syllable frontend with automatic language selection
- auto:         the production route chosen by use_multilingual
- mixed:        auto route after prefixing an English word ('hello ')

Expectations are invariants and intended outcomes, not dumps of current output:
valid ordinary spellings keep their phones and full source coverage; unsupported
or ambiguous input fails with LyricsInputError carrying original positions and a
reason; nothing pronounceable is silently dropped. Legacy mora counts are never
compared with multilingual syllable counts; only phone sequences are compared.
Synthetic emissions test the alignment graph, not real singing accuracy.
"""
import functools
import re
import unicodedata
from types import SimpleNamespace

import numpy as np
import pytest
import regex

from utalign.lyric_input import LyricsInputError, analysis_text, annotation_indices, line_column, requires_reading
from utalign.lyrics import envelope_groups, envelope_ruby_warning, parse_lyrics, prepare_mora_lyrics, romaji_tokens
from utalign.multilingual.frontend import INFERRED_SOURCES, Frontend, input_warnings
from utalign.multilingual.joint_alignment import align_syllables
from utalign.multilingual.pipeline import build_result, use_multilingual
from utalign.multilingual.proxy_readings import apply_proxy_readings
from utalign.pipeline import AlignOptions
from utalign.utavista import result_to_lyrics_timing, self_check
from test_multilingual import FakeAligner, log_emissions

MODEL = FakeAligner()
PREFIX = 'hello '
PATHS = ('japanese', 'multilingual', 'auto', 'mixed')
REASONS = {'number_needs_reading', 'no_dictionary_reading', 'symbol_needs_reading', 'kana_without_reading',
           'unattached_mark', 'script_needs_multilingual', 'unsupported_script', 'no_preceding_sound',
           'no_model_phones', 'japanese_reader_no_syllables', 'pronunciation_failed', 'hint_splits_word',
           'language_undetermined', 'model_phones_unsupported', 'orphan_mark', 'language_not_enabled',
           'proxy_reading_invalid', 'reading_override_unmappable', 'reading_override_language_ambiguous'}
# Independent reference for physical lines (same separators as str.splitlines; CRLF is one break).
REFERENCE_BREAK = re.compile('\r\n|[\n\r\x0b\x0c\x1c\x1d\x1e\x85  ]')


def reference_line_column(text, a):
    before = REFERENCE_BREAK.split(text[:a])
    return len(before), len(before[-1]) + 1
JA_RUN_CHAR = regex.compile(r'[\p{Han}\p{Hiragana}\p{Katakana}ー]')


@functools.lru_cache(maxsize=None)
def frontend(readings=None, model=None):
    return Frontend(model or MODEL, readings=readings)


# ---------------------------------------------------------------- expectations
class Ok:
    def __init__(self, phones=None, syllables=None):
        self.phones = phones.split() if isinstance(phones, str) else phones
        self.syllables = syllables


class Bad:
    def __init__(self, at=(), reason=None):
        self.at = set(at)
        self.reason = reason


EMPTY, ANY = 'empty', 'any'
LATIN_JA = 'script_needs_multilingual'

CASES = [
    # long vowel
    ('long', 'あー', dict(all=Ok('a', ['アー']))),
    ('long', 'すーん', dict(all=Ok('s u N', ['スーン']))),
    ('long', 'うぃー', dict(all=Ok('w i', ['ウィー']))),
    ('long', 'あーー', dict(all=Ok('a', ['アーー']))),
    ('long', 'ー', dict(all=Bad({0}, 'no_preceding_sound'))),
    ('long', 'ーあ', dict(all=Bad({0}, 'no_preceding_sound'))),
    ('long', 'あ ー', dict(all=Bad({2}, 'no_preceding_sound'))),
    ('long', 'あ、ー', dict(all=Bad({2}, 'no_preceding_sound'))),
    ('long', 'あ\nー', dict(all=Bad({2}, 'no_preceding_sound'))),
    ('long', 'んー', dict(all=Ok('N', ['ンー']))),
    ('long', 'っー', dict(all=Ok('cl', ['ッー']))),
    # sokuon
    ('sokuon', 'きっと', dict(all=Ok('k i cl t o', ['キッ', 'ト']))),
    ('sokuon', 'モニゅっと', dict(all=Ok('m o ny u cl t o', ['モ', 'ニュッ', 'ト']))),
    ('sokuon', 'あっ', dict(all=Ok('a cl', ['アッ']))),
    ('sokuon', 'っ', dict(all=Ok('cl', ['ッ']))),
    ('sokuon', 'っあ', dict(all=Ok('cl a', ['ッ', 'ア']))),
    ('sokuon', 'かっか', dict(all=Ok('k a cl k a', ['カッ', 'カ']))),
    ('sokuon', 'かッと', dict(all=Ok('k a cl t o', ['カッ', 'ト']))),
    # small kana
    ('small', 'きゃ', dict(all=Ok('ky a', ['キャ']))),
    ('small', 'うぃ', dict(all=Ok('w i', ['ウィ']))),
    ('small', 'き ゃ', dict(all=Ok('k i y a', ['キ', 'ャ']))),
    ('small', 'ゃ', dict(all=Ok('y a', ['ャ']))),
    ('small', 'ティ', dict(all=Ok('t i', ['ティ']))),
    ('small', 'ヴュ', dict(all=Ok())),
    # hatsuon
    ('nasal', 'ん', dict(all=Ok('N', ['ン']))),
    ('nasal', 'あん', dict(all=Ok('a N', ['アン']))),
    ('nasal', 'んあ', dict(all=Ok('N a', ['ン', 'ア']))),
    ('nasal', 'カンパイ', dict(all=Ok('k a N p a i', ['カン', 'パ', 'イ']))),
    ('nasal', 'さん、あい', dict(all=Ok('s a N a i', ['サン', 'ア', 'イ']))),
    # canonical equivalence and width
    ('unicode', 'が', dict(all=Ok('g a', ['ガ']))),
    ('unicode', 'が', dict(all=Ok('g a', ['ガ']))),
    ('unicode', 'ぱ', dict(all=Ok('p a', ['パ']))),
    ('unicode', 'ぱ', dict(all=Ok('p a', ['パ']))),
    ('unicode', 'パーティー', dict(all=Ok('p a t i', ['パー', 'ティー']))),
    ('unicode', 'ﾊﾟｰﾃｨｰ', dict(all=Ok('p a t i', ['パー', 'ティー']))),
    ('unicode', 'ｶﾞ', dict(all=Ok('g a', ['ガ']))),
    ('unicode', 'ＡＢＣ', dict(japanese=Bad({0, 1, 2}, LATIN_JA), rest=ANY)),
    ('unicode', '①', dict(all=Bad({0}, 'number_needs_reading'))),
    ('unicode', '㍑', dict(all=Bad({0}, 'symbol_needs_reading'))),
    # iteration marks
    ('iteration', '時々', dict(all=Ok('t o k i d o k i'))),
    ('iteration', '人々', dict(all=Ok('h i t o b i t o'))),
    ('iteration', 'すゞしい', dict(all=Ok())),
    ('iteration', 'くゝ', dict(all=Ok())),
    ('iteration', 'いすゞ', dict(all=Ok())),
    ('iteration', 'カヽ', dict(all=Bad({1}))),
    ('iteration', 'カヾ', dict(all=Bad({1}))),
    ('iteration', '々', dict(all=Bad({0}))),
    # punctuation, symbols, line separators
    ('punct', 'あ・い', dict(all=Ok('a i', ['ア', 'イ']))),
    ('punct', 'あ〜', dict(all=Ok('a', ['ア']))),
    ('punct', 'あ～', dict(all=Ok('a', ['ア']))),
    ('punct', 'あ!', dict(all=Ok('a', ['ア']))),
    ('punct', 'あ😊い', dict(all=Ok('a i', ['ア', 'イ']))),
    ('punct', 'か‍な', dict(all=Ok('k a n a', ['カ', 'ナ']))),
    ('punct', 'あ\r\nい', dict(all=Ok('a i', ['ア', 'イ']))),
    ('punct', 'あ\rい', dict(all=Ok('a i', ['ア', 'イ']))),
    ('punct', 'あ い', dict(all=Ok('a i', ['ア', 'イ']))),
    # numbers: no automatic reading, never silently dropped
    ('numbers', '123', dict(all=Bad({0, 1, 2}, 'number_needs_reading'))),
    ('numbers', '１２３', dict(all=Bad({0, 1, 2}, 'number_needs_reading'))),
    ('numbers', '一二三', dict(all=Ok())),
    ('numbers', '第1章', dict(all=Bad({1}, 'number_needs_reading'))),
    ('numbers', '夜に2人', dict(all=Bad({2}, 'number_needs_reading'))),
    ('numbers', '3.14', dict(all=Bad({0, 2, 3}, 'number_needs_reading'))),
    # Latin: the Japanese mora mode must say it cannot pronounce it
    ('latin', 'LINE', dict(japanese=Bad({0, 1, 2, 3}, LATIN_JA), rest=Ok())),
    ('latin', 'SNS', dict(japanese=Bad({0, 1, 2}, LATIN_JA), rest=Bad({0, 1, 2}, 'pronunciation_failed'))),
    ('latin', 'I love you', dict(japanese=Bad({0, 2, 3, 4, 5, 7, 8, 9}, LATIN_JA), rest=Ok())),
    ('latin', "don't", dict(japanese=Bad({0, 1, 2, 4}, LATIN_JA), rest=Ok())),
    ('latin', 'rock-n-roll', dict(japanese=Bad({0, 5, 10}, LATIN_JA), rest=Bad(reason='pronunciation_failed'))),
    ('latin', 'café', dict(japanese=Bad({0, 1, 2, 3}, LATIN_JA), rest=Ok())),
    ('latin', 'café', dict(japanese=Bad({0, 1, 2, 3, 4}, LATIN_JA), rest=Ok())),
    # annotations: deliberate unsung text, same rule on every path
    ('annotation', '[Chorus]\nあ', dict(all=Ok('a', ['ア']))),
    ('annotation', '（ああ）\nい', dict(all=Ok('i', ['イ']))),
    ('annotation', '(oh) あ', dict(all=Ok('a', ['ア']))),
    ('annotation', '© AUTHOR\nあ', dict(mixed=Ok(), rest=Ok('a', ['ア']))),
    ('annotation', 'hello (oh) あ', dict(japanese=Bad({0, 1, 2, 3, 4}, LATIN_JA), rest=Ok())),
    # unknown characters
    ('unknown', '𠮷野家', dict(all=ANY)),
    ('unknown', '髙い', dict(all=Bad({0}, 'no_dictionary_reading'))),
    ('unknown', 'αあ', dict(all=Bad({0}, 'unsupported_script'))),
    ('unknown', '😀', dict(mixed=Ok(), rest=EMPTY)),
    ('unknown', '[あ]', dict(mixed=Ok(), rest=EMPTY)),
    # ordinary Japanese
    ('japanese', 'こんにちは', dict(all=Ok('k o N n i ch i w a', ['コン', 'ニ', 'チ', 'ワ']))),
    ('japanese', '私は君が好き', dict(all=Ok())),
    ('japanese', '日本', dict(all=Ok())),
    ('japanese', '世界', dict(all=Ok())),
    ('japanese', '東京', dict(all=Ok())),
]
assert len(CASES) == 84


def expectation(spec, path):
    return spec.get(path, spec.get('rest', spec.get('all')))


# ---------------------------------------------------------------- running a path
def run(text, path, hints=(), readings=None):
    if path == 'mixed':
        text = PREFIX + text
    if path == 'japanese':
        route = 'mora'
    elif path == 'multilingual':
        route = 'syllable'
    else:
        route = 'syllable' if use_multilingual(text, AlignOptions()) else 'mora'
    out = SimpleNamespace(text=text, route=route, path=path, offset=len(PREFIX) if path == 'mixed' else 0)
    try:
        if route == 'mora':
            lyr = prepare_mora_lyrics(text, readings, MODEL.mora_tokens)
            if not lyr.moras:
                out.status = EMPTY
                return out
            out.lyr = lyr
            out.phones = [t for u in lyr.units if not u.is_star for t in u.tokens]
            out.covered = {i for m in lyr.moras for i in m.char_idx}
        else:
            doc = frontend(readings).parse(text, hints)
            out.doc = doc
            mine = [s for s in doc['syllables'] if doc['words'][s['word_id']]['char_start'] >= out.offset]
            out.phones = [p for s in mine for p in s['model_phones']]
            out.readings = [s['reading'] for s in mine]
            out.covered = {i for s in doc['syllables'] for i in s['source_chars']}
        out.status = 'ok'
    except LyricsInputError as exc:
        out.status = 'unresolved'
        out.issues = exc.issues
        out.message = str(exc)
    except ValueError as exc:
        if 'No pronounceable' not in str(exc):
            raise
        out.status = EMPTY
    return out


def required_positions(text):
    norm = analysis_text(text)
    annotated = set(norm.source_indices(annotation_indices(norm.text)))
    return {i for i in range(len(text)) if requires_reading(text, i) and i not in annotated}


def check_ok_invariants(out):
    text = out.text
    missing = required_positions(text) - out.covered
    assert not missing, f'silently dropped {[(i, text[i]) for i in sorted(missing)]}'
    assert all(0 <= i < len(text) for i in out.covered)
    if out.route == 'mora':
        for m in out.lyr.moras:
            assert m.char_idx and all(0 <= i < len(text) for i in m.char_idx)
            assert not (m.kind == 'normal' and not m.tokens), f'phone-less mora {m.kana} folded away'
            assert not (m.kind == 'choon' and (m.run_start or m.ctc_unit is None))
        return
    doc = out.doc
    assert doc['text'] == text
    for w in doc['words']:
        assert text[w['char_start']:w['char_end']] == w['text']
        for sid in w['syllable_ids']:
            s = doc['syllables'][sid]
            assert s['model_phones'] and set(s['model_phones']) <= MODEL.vocab.keys()
            assert all(w['char_start'] <= i < w['char_end'] for i in s['source_chars'])
            if s['language'] != 'ja':
                continue
            moras = s['moras']
            assert [p for m in moras for p in m['model_phones']] == s['model_phones']
            assert ''.join(m['kana'] for m in moras) == s['reading']
            assert sorted({i for m in moras for i in m['source_chars']}) == s['source_chars']
            if not set(s['model_phones']) & set('aiueo'):
                # Only a genuinely standalone ッ/ン (nothing Japanese directly before it) may lack a vowel.
                assert moras[0]['kind'] in {'sokuon', 'hatsuon'}
                first = s['source_chars'][0]
                assert first == 0 or not JA_RUN_CHAR.match(text[first - 1]), s
    for d in doc['language_diagnostics']:
        assert text[d['char_start']:d['char_end']] == d['text'] and d['line'] >= 1 and d['column'] >= 1
    inferred = {(w['char_start'], w['char_end']) for w in doc['words'] if w['language_source'] in INFERRED_SOURCES}
    diagnosed = [(d['char_start'], d['char_end']) for d in doc['language_diagnostics']]
    for a, b in inferred:
        assert any(x <= a and b <= y for x, y in diagnosed), f'inferred language without diagnostic at {a}:{b}'


def check_unresolved_invariants(out):
    text = out.text
    assert out.issues
    for x in out.issues:
        a, b = x['char_start'], x['char_end']
        assert 0 <= a < b <= len(text)
        assert x['text'] == text[a:b]
        # CR-only and U+2028 are line breaks too (identical to the old LF count for LF-only text).
        assert (x['line'], x['column']) == reference_line_column(text, a)
        assert x['reason'] in REASONS and x['message']
        assert f"line {x['line']} col {x['column']}" in out.message or len(out.issues) > 8


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('group,text,spec', CASES, ids=[f'{g}-{i}' for i, (g, _, _) in enumerate(CASES)])
def test_audit_matrix(group, text, spec, path):
    expected = expectation(spec, path)
    out = run(text, path)
    if out.status == 'ok':
        check_ok_invariants(out)
    elif out.status == 'unresolved':
        check_unresolved_invariants(out)
    if expected == ANY:
        assert out.status in {'ok', 'unresolved'}
        return
    if expected == EMPTY:
        assert out.status == EMPTY
        return
    if isinstance(expected, Bad):
        assert out.status == 'unresolved', f'{text!r} on {path} should ask for correction, got {out.status}'
        reported = {i for x in out.issues for i in range(x['char_start'], x['char_end'])}
        assert {i + out.offset for i in expected.at} <= reported
        if expected.reason:
            assert any(x['reason'] == expected.reason for x in out.issues), out.issues
        return
    assert out.status == 'ok', f'{text!r} on {path}: {getattr(out, "message", out.status)}'
    if expected.phones is not None:
        assert out.phones == expected.phones
    if expected.syllables is not None and out.route == 'syllable':
        assert out.readings == expected.syllables


@pytest.mark.parametrize('group,text,spec', CASES, ids=[f'{g}-{i}' for i, (g, _, _) in enumerate(CASES)])
def test_japanese_paths_preserve_the_same_phones(group, text, spec):
    """Mora and syllable views regroup the same reader moras; phones never change."""
    legacy, multi = run(text, 'japanese'), run(text, 'multilingual')
    if legacy.status != 'ok' or multi.status != 'ok':
        return
    if any(w['language'] != 'ja' for w in multi.doc['words']):
        return
    assert multi.phones == legacy.phones
    assert multi.covered == legacy.covered


# ---------------------------------------------------------------- equivalent spellings
EQUIVALENT = [('かっと', 'かッと'), ('しんじる', 'しンじる'), ('うんこ', 'うンこ'), ('みんな', 'みンな'),
              ('あっ', 'あッ'), ('あん', 'あン'), ('がっこう', 'ガッコウ'), ('が', 'が'), ('ぱ', 'ぱ'),
              ('パーティー', 'ﾊﾟｰﾃｨｰ'), ('ガ', 'ｶﾞ'), ('すーん', 'すｰん')]


@pytest.mark.parametrize('path', ['multilingual', 'mixed'])
@pytest.mark.parametrize('first,second', EQUIVALENT)
def test_equivalent_spellings_give_identical_syllables(first, second, path):
    a, b = run(first, path), run(second, path)
    assert a.status == b.status == 'ok'
    assert a.readings == b.readings
    assert a.phones == b.phones
    check_ok_invariants(a)
    check_ok_invariants(b)


@pytest.mark.parametrize('first,second', EQUIVALENT)
def test_equivalent_spellings_give_identical_moras(first, second):
    a, b = run(first, 'japanese'), run(second, 'japanese')
    assert a.status == b.status == 'ok'
    assert [m.kana for m in a.lyr.moras] == [m.kana for m in b.lyr.moras]
    assert a.phones == b.phones


@pytest.mark.parametrize('text', ['かッと', 'しンじる', 'うンこ', 'モニゅっと', 'みンな', 'あッ'])
def test_tokenization_never_creates_closure_or_nasal_syllables(text):
    doc = frontend().parse(text)
    assert not [s for s in doc['syllables'] if not set(s['model_phones']) & set('aiueo')]


def test_genuinely_standalone_nasal_and_closure_are_kept():
    doc = frontend().parse('ん、さん、ん っ')
    assert [s['reading'] for s in doc['syllables']] == ['ン', 'サン', 'ン', 'ッ']


def test_per_mora_metadata_is_retained_under_syllables():
    doc = frontend().parse('すーん が')
    first, second = doc['syllables']
    assert [(m['kana'], m['kind'], m['source_chars']) for m in first['moras']] == [
        ('ス', 'normal', [0]), ('ー', 'choon', [1]), ('ン', 'hatsuon', [2])]
    assert [(m['kana'], m['source_chars']) for m in second['moras']] == [('ガ', [4, 5])]
    assert doc['words'][1]['text'] == 'が'


class GapAligner(FakeAligner):
    """A model without phones for ナ: such a mora must not be folded away like a long vowel."""

    def mora_tokens(self, mora, next_mora):
        return [] if mora.kana == 'ナ' else FakeAligner.mora_tokens(self, mora, next_mora)


def test_phone_less_readings_are_reported_not_attached():
    with pytest.raises(LyricsInputError) as caught:
        prepare_mora_lyrics('あな', None, GapAligner().mora_tokens)
    assert [(x['char_start'], x['reason']) for x in caught.value.issues] == [(1, 'no_model_phones')]
    with pytest.raises(LyricsInputError) as caught:
        Frontend(GapAligner(), languages=('ja',)).parse('あな')
    assert [(x['char_start'], x['reason']) for x in caught.value.issues] == [(1, 'no_model_phones')]


# ---------------------------------------------------------------- normalization and positions
def test_analysis_text_maps_back_to_every_source_code_point():
    cases = {'ﾊﾟｰﾃｨｰ': ('パーティー', [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6)]),
             'が': ('が', [(0, 2)]), 'か゛': ('が', [(0, 2)]), 'ｶﾞ': ('ガ', [(0, 2)]),
             'ＡＢＣ１': ('ABC1', [(0, 1), (1, 2), (2, 3), (3, 4)]), 'é': ('é', [(0, 2)]),
             '葛\U000e0100': ('葛', [(0, 2)]), 'あ゙': ('あ゙', [(0, 1), (1, 2)])}
    for source, (analysis, origin) in cases.items():
        norm = analysis_text(source)
        assert (norm.text, norm.origin) == (analysis, origin), source
        assert analysis_text(norm.text).origin == [(i, i + 1) for i in range(len(norm.text))]
    # No arbitrary compatibility mapping: symbols keep their identity.
    for symbol in ['①', '㍑', '™', '½', '～', '〜']:
        assert analysis_text(symbol).text == symbol


def test_source_lyrics_are_never_rewritten():
    text = 'ﾊﾟｰﾃｨｰ が\nＡＢＣ'
    # The hint is written in the source (fullwidth) spelling and matched through the same normalization.
    doc = frontend().parse(text, [dict(text='ＡＢＣ', language='en', line=2, char_start=10)])
    assert doc['text'] == text
    assert ''.join(text[w['char_start']:w['char_end']] for w in doc['words']) == 'ﾊﾟｰﾃｨｰがＡＢＣ'
    assert {c['source'] for c in doc['normalization']} == {'ﾊﾟ', 'ﾃ', 'ｨ', 'ｰ', 'が', 'Ａ', 'Ｂ', 'Ｃ'}
    lyr = parse_lyrics('ﾊﾟｰﾃｨｰ')
    assert lyr.text == 'ﾊﾟｰﾃｨｰ' and lyr.chars == list('ﾊﾟｰﾃｨｰ')


def test_uncomposable_mark_is_reported():
    out = run('あ゙', 'japanese')
    assert out.status == 'unresolved'
    assert [x['char_start'] for x in out.issues] == [1]


def test_error_message_gives_line_column_reason_and_suggestion():
    with pytest.raises(LyricsInputError) as caught:
        prepare_mora_lyrics('あ\n夜に2人', None, MODEL.mora_tokens)
    issue, = caught.value.issues
    assert (issue['line'], issue['column'], issue['text'], issue['reason']) == (2, 3, '2', 'number_needs_reading')
    assert 'readings override' in issue['suggestion']
    assert 'line 2 col 3' in str(caught.value)
    with pytest.raises(LyricsInputError) as caught:
        prepare_mora_lyrics('あ LINE', None, MODEL.mora_tokens)
    assert 'multilingual' in caught.value.issues[0]['suggestion']


def test_explicit_reading_resolves_numbers_and_iteration_marks_without_rewriting(tmp_path):
    readings = tmp_path / 'readings.txt'
    # The counter is part of the explicit phrase: 2人 -> ふた|り (a '2' override alone does not guess 人).
    readings.write_text('2人 ふた|り\nカヽ か|か\n', encoding='utf-8')
    for path in ('japanese', 'multilingual'):
        out = run('夜に2人 カヽ', path, readings=readings)
        assert out.status == 'ok', getattr(out, 'message', '')
        check_ok_invariants(out)
    lyr = run('夜に2人', 'japanese', readings=readings).lyr
    assert ''.join(m.kana for m in lyr.moras) == 'ヨルニフタリ'
    assert [m.kana for m in lyr.moras if m.char_idx == [2]] == ['フ', 'タ']
    assert [m.kana for m in lyr.moras if m.char_idx == [3]] == ['リ']
    doc = run('夜に2人', 'multilingual', readings=readings).doc
    assert ''.join(s['reading'] for s in doc['syllables']) == 'ヨルニフタリ'
    assert [m['kana'] for s in doc['syllables'] for m in s['moras'] if m['source_chars'] == [2]] == ['フ', 'タ']
    export = synthetic_export(doc)
    chars = {c['char']: c for p in export['phrases'] for w in p['words'] for c in w['chars']}
    assert chars['2']['ruby'] == 'ふた' and chars['人']['ruby'] == 'り'
    assert 'rubySpan' not in chars['2']


# ---------------------------------------------------------------- annotations
@pytest.mark.parametrize('text', ['©山田太郎\nあ', '© 作詞 山田太郎\nあ', '[Chorus]\nあ'])
def test_annotation_handling_agrees_between_paths(text):
    legacy, multi = run(text, 'japanese'), run(text, 'multilingual')
    assert legacy.status == multi.status == 'ok'
    assert legacy.phones == multi.phones == ['a']
    assert legacy.covered == multi.covered == {len(text) - 1}


# ---------------------------------------------------------------- language hints
def test_same_language_hint_on_a_substring_keeps_the_reading():
    plain = frontend().parse('今日は')
    hinted = frontend().parse('今日は', [dict(text='日', language='ja')])
    assert [s['reading'] for s in hinted['syllables']] == [s['reading'] for s in plain['syllables']]
    assert [p for s in hinted['syllables'] for p in s['model_phones']] == \
           [p for s in plain['syllables'] for p in s['model_phones']]
    assert ''.join(w['text'] for w in hinted['words']) == '今日は'


def test_hint_that_splits_a_word_into_another_language_is_rejected_with_position():
    with pytest.raises(LyricsInputError) as caught:
        frontend().parse('今日は', [dict(text='日', language='zh')])
    issue, = caught.value.issues
    assert issue['reason'] == 'hint_splits_word' and issue['char_start'] <= 1 < issue['char_end']


def test_partial_latin_hint_never_splits_the_word():
    try:
        doc = frontend().parse('hello', [dict(text='hell', language='en')])
    except LyricsInputError as exc:
        assert [x['reason'] for x in exc.issues] == ['hint_splits_word']
    else:
        assert [w['text'] for w in doc['words']] == ['hello']


def test_explicit_hints_take_precedence_and_chinese_stays_supported():
    doc = frontend().parse('東京', [dict(text='東京', language='ja')])
    assert [w['language'] for w in doc['words']] == ['ja'] and not doc['language_diagnostics']
    doc = frontend().parse('你好 世界', [dict(text='你好', language='zh'), dict(text='世界', language='ja')])
    assert [w['language'] for w in doc['words']] == ['zh', 'ja']
    doc = frontend().parse('你好世界', [dict(text='你好世界', language='zh')])
    assert len(doc['syllables']) == 4 and {s['language'] for s in doc['syllables']} == {'zh'}


def test_japanese_spelling_cues_route_iteration_marks_to_japanese():
    assert use_multilingual('時々', AlignOptions()) is False
    assert use_multilingual('你好世界', AlignOptions()) is True
    doc = frontend().parse('時々 hello')
    assert doc['words'][0]['language'] == 'ja' and doc['words'][0]['language_source'] == 'script'


# ---------------------------------------------------------------- diagnostics propagation
def test_inferred_language_is_diagnosed_with_position_and_surfaced():
    doc = frontend().parse('こんにちは\n東京')
    diagnostic, = doc['language_diagnostics']
    assert (diagnostic['line'], diagnostic['column'], diagnostic['text']) == (2, 1, '東京')
    assert diagnostic['status'] == 'language_inferred_not_verified' and diagnostic['suggestion']
    assert any('inferred' in w for w in input_warnings(doc))
    assert not frontend().parse('こんにちは')['language_diagnostics']


def test_estimated_kanji_split_reaches_output_and_warnings():
    doc = frontend().parse('大人', [dict(text='大人', language='ja')])
    assert any(s['split_estimated'] for s in doc['syllables'])
    result = synthetic_result(doc)
    assert any(c['split_estimated'] for c in result['chars'][:2])
    assert any(m['split_estimated'] for m in result['moras'])
    assert any('estimated' in w for w in input_warnings(doc))


# ---------------------------------------------------------------- export: ruby and grapheme timing
def synthetic_result(doc):
    apply_proxy_readings(doc, MODEL)
    syllables = [dict(s, start=.2 + i * .5, end=.6 + i * .5, note_ids=[i], timing_basis='synthetic_for_mapping_test',
                      end_basis='synthetic') for i, s in enumerate(doc['syllables'])]
    notes = [SimpleNamespace(id=i, start=.2 + i * .5, end=.6 + i * .5, pitch=60, track=0, phrase=0)
             for i in range(len(syllables))]
    return build_result(doc, syllables, notes, 0, {}, {}, len(syllables) * .5 + 1)


def synthetic_export(doc):
    export = result_to_lyrics_timing(synthetic_result(doc), ruby=True)
    assert self_check(export) == []
    return export


def export_chars(export):
    return [c for p in export['phrases'] for w in p['words'] for c in w['chars']]


def test_ruby_uses_only_the_moras_of_each_kanji():
    export = synthetic_export(frontend().parse('待って 叫んで'))
    ruby = {c['char']: c.get('ruby') for c in export_chars(export)}
    assert ruby['待'] == 'ま' and ruby['叫'] == 'さけ'
    assert all(ruby[k] is None for k in 'ってんで')
    syllable = export['phrases'][0]['words'][0]['syllables'][0]
    assert [m['reading'] for m in syllable['moras']] == ['マ', 'ッ']


@pytest.mark.parametrize('text,groups', [('が', [[0, 1]]), ('ｶﾞ', [[0, 1]]),
                                         ('ﾊﾟｰﾃｨｰ', [[0, 1], [2], [3], [4], [5]]),
                                         # spacing mark; NFD + selector (three source points, one grapheme)
                                         ('か゛', [[0, 1]]), ('ぱ︀', [[0, 1, 2]]),
                                         ('ぱ︀い', [[0, 1, 2], [3]])])
def test_combining_and_halfwidth_marks_share_their_base_interval(text, groups):
    chars = export_chars(synthetic_export(frontend().parse(text)))
    assert ''.join(c['char'] for c in chars) == text
    intervals = [{(chars[k]['startMs'], chars[k]['endMs']) for k in group} for group in groups]
    assert all(len(x) == 1 for x in intervals)
    starts = [min(x)[0] for x in intervals]
    assert starts == sorted(starts) and len(set(starts[:3])) == min(3, len(starts))
    assert not any('ruby' in c for c in chars)


def test_foreign_word_envelope_is_unchanged():
    doc = frontend().parse('hello', [dict(text='hello', language='en')])
    chars = export_chars(synthetic_export(doc))
    assert len({(c['startMs'], c['endMs']) for c in chars}) == 1
    assert all(c['timingBasis'] == 'word_envelope' for c in chars)


# ---------------------------------------------------------------- synthetic acoustics / MIDI
def align(text):
    doc = frontend().parse(text)
    notes = [SimpleNamespace(start=.2, end=.68), SimpleNamespace(start=.8, end=1.28)]
    events = []
    for i, s in enumerate(doc['syllables']):
        frame = 10 + i * 30
        events += [(frame + 2 * q, frame + 2 * q + 1, MODEL.vocab[p]) for q, p in enumerate(s['model_phones'])]
    em = log_emissions(100, len(MODEL.vocab), events)
    return align_syllables(em, doc, notes, 0., MODEL)


def test_sokuon_spelling_does_not_change_midi_assignment():
    (hira, hira_summary), (kata, kata_summary) = align('かっと'), align('かッと')
    for out in (hira, kata):
        assert [s['reading'] for s in out] == ['カッ', 'ト']
        assert [s['note_ids'] for s in out] == [[0], [1]]
        assert [s['share_count'] for s in out] == [1, 1]
    assert [s['start'] for s in hira] == [s['start'] for s in kata]
    assert hira_summary['midi_assignment_cost'] == pytest.approx(kata_summary['midi_assignment_cost'])


# ---------------------------------------------------------------- canonical normalization (many-to-many map)
def assert_total_map(norm):
    """Every source point is in some analysis char; re-normalizing is the identity."""
    assert unicodedata.normalize('NFC', norm.text) == norm.text
    assert {k for m in norm.members for k in m} == set(range(len(norm.source)))
    again = analysis_text(norm.text)
    assert again.text == norm.text and again.members == [(i,) for i in range(len(norm.text))]


def test_reordered_combining_marks_give_true_nfc():
    acute_first, dot_first = analysis_text('ạ́'), analysis_text('ạ́')
    assert acute_first.text == dot_first.text == unicodedata.normalize('NFC', 'ạ́') == 'ạ́'
    # ạ is made of a + dot below wherever the dot was written; the acute keeps its own source point.
    assert acute_first.members == [(0, 2), (1,)]
    assert dot_first.members == [(0, 1), (2,)]
    for norm in (acute_first, dot_first):
        assert_total_map(norm)
        assert not norm.is_boundary(1)   # a span may not separate ạ from its acute
        assert norm.source_span(0, 2) == (0, 3)
    # Reordering is reported as a normalization even when no character changes spelling.
    assert any(c['source_chars'] == [0, 2] for c in acute_first.changes())


def test_canonical_expansion_maps_two_analysis_marks_to_one_source_point():
    alone = analysis_text('̈́')
    assert alone.text == '̈́' and alone.members == [(0,), (0,)]
    assert alone.covered_source({0}) == set()           # half of an expansion is not coverage
    assert alone.covered_source({0, 1}) == {0}
    on_a = analysis_text('ä́')
    assert on_a.text == unicodedata.normalize('NFC', 'ä́') == 'ä́'
    assert on_a.members == [(0, 1), (1,)]
    assert on_a.covered_source({0}) == {0}
    assert not on_a.is_boundary(1)
    assert_total_map(alone)
    assert_total_map(on_a)


def test_reading_the_base_never_covers_an_uncomposable_mark():
    norm = analysis_text('あ゙')
    assert norm.members == [(0,), (1,)] and norm.covered_source({0}) == {0}
    for text, mark in [('あ゙', 1), ('か̈́', 1), ('か̈́い', 1)]:
        for path in ('japanese', 'multilingual'):
            out = run(text, path)
            assert out.status == 'unresolved', (text, path)
            check_unresolved_invariants(out)
            issue = next(x for x in out.issues if x['char_start'] <= mark < x['char_end'])
            assert issue['reason'] == 'unattached_mark', (text, path, out.issues)


@pytest.mark.parametrize('path', ['japanese', 'multilingual'])
def test_canonically_equivalent_mark_orders_behave_identically(path):
    first, second = run('ạ́', path), run('ạ́', path)
    assert first.status == second.status
    if first.status == 'unresolved':
        for out in (first, second):
            check_unresolved_invariants(out)
            assert {i for x in out.issues for i in range(x['char_start'], x['char_end'])} >= {0, 1, 2}
        assert [x['reason'] for x in first.issues] == [x['reason'] for x in second.issues]
    else:
        assert first.phones == second.phones
        assert first.covered == second.covered == {0, 1, 2}


def test_hint_cannot_end_inside_a_reordered_cluster():
    for text in ['ạ́', 'ạ́']:
        with pytest.raises(ValueError, match='does not match'):
            frontend().parse(text, [dict(text='ạ', language='en')])


# ---------------------------------------------------------------- orphan marks vs presentation selectors
@pytest.mark.parametrize('path', ['japanese', 'multilingual'])
@pytest.mark.parametrize('text,mark', [('゙あ', 0), ('あ ゙い', 2), ('あ\n゙い', 2), ('、́あ', 1),
                                       ('̈́あ', 0)])
def test_orphan_sound_marks_are_diagnosed_with_position(text, mark, path):
    assert requires_reading(text, mark)
    out = run(text, path)
    assert out.status == 'unresolved', f'{text!r} silently accepted on {path}'
    check_unresolved_invariants(out)
    issue = next(x for x in out.issues if x['char_start'] <= mark < x['char_end'])
    assert issue['reason'] == 'orphan_mark' and issue['suggestion']


@pytest.mark.parametrize('path', ['japanese', 'multilingual'])
def test_orphan_halfwidth_voiced_mark_is_diagnosed(path):
    out = run('ﾞあ', path)
    assert out.status == 'unresolved'
    check_unresolved_invariants(out)
    assert [x['reason'] for x in out.issues if x['char_start'] == 0] == ['orphan_mark']


@pytest.mark.parametrize('path', ['japanese', 'multilingual'])
@pytest.mark.parametrize('text', ['❤️あ', '️あ', '#️⃣あ', '😊\U0001f3fbあ'])
def test_emoji_presentation_selectors_are_not_sound(text, path):
    out = run(text, path)
    assert out.status == 'ok', getattr(out, 'message', out.status)
    assert out.phones == ['a']
    check_ok_invariants(out)


# ---------------------------------------------------------------- legacy empty-token specials
def test_tokenless_specials_never_borrow_another_runs_sound():
    for text in ['あ っ', 'あ、っ', 'あ\nっ', 'あ\rっ']:
        with pytest.raises(LyricsInputError) as caught:
            prepare_mora_lyrics(text)                       # romaji: っ has no token of its own
        assert [(x['char_start'], x['reason']) for x in caught.value.issues] == [(len(text) - 1, 'no_preceding_sound')]
        # A phoneme model represents a standalone closure; it stays its own unit.
        lyr = prepare_mora_lyrics(text, None, MODEL.mora_tokens)
        assert [u.tokens for u in lyr.units if not u.is_star] == [['a'], ['cl']]
    for text, units in [('あっ', [['a']]), ('かッと', [['ka'], ['to']]), ('あーっ', [['a']])]:
        lyr = prepare_mora_lyrics(text)
        assert [u.tokens for u in lyr.units if not u.is_star] == units
        assert all(m.ctc_unit is not None for m in lyr.moras)


def test_tokenless_nasal_and_unknown_readings_never_attach_across_runs():
    def no_nasal(mora, nxt):
        return [] if mora.kind == 'hatsuon' else romaji_tokens(mora, nxt)
    assert [u.tokens for u in prepare_mora_lyrics('あん', None, no_nasal).units if not u.is_star] == [['a']]
    with pytest.raises(LyricsInputError) as caught:
        prepare_mora_lyrics('あ ん', None, no_nasal)
    assert [(x['char_start'], x['reason']) for x in caught.value.issues] == [(2, 'no_preceding_sound')]
    lyr = parse_lyrics('あな', tokenizer=GapAligner().mora_tokens)
    assert lyr.moras[1].ctc_unit is None and lyr.units[-2].mora_ids == [0]


# ---------------------------------------------------------------- physical lines: CRLF, LF, CR, U+2028
SEPARATORS = ['\n', '\r\n', '\r', ' ']


@pytest.mark.parametrize('sep', SEPARATORS)
def test_line_breaks_give_the_same_positions_on_every_path(sep):
    text = 'あ' + sep + '2'
    for parse in (lambda: prepare_mora_lyrics(text, None, MODEL.mora_tokens), lambda: frontend().parse(text)):
        with pytest.raises(LyricsInputError) as caught:
            parse()
        issue, = caught.value.issues
        assert (issue['line'], issue['column'], issue['char_start']) == (2, 1, len(text) - 1)
        assert 'line 2 col 1' in str(caught.value)
    assert line_column(text, 1) == (1, 2)     # the delimiter belongs to the line it ends


@pytest.mark.parametrize('sep', SEPARATORS)
def test_line_breaks_split_result_and_export_without_rewriting(sep):
    text = 'あ' + sep + 'い'
    second = 1 + len(sep)
    lyr = prepare_mora_lyrics(text, None, MODEL.mora_tokens)
    assert lyr.text == text and lyr.lines == [(0, 1), (second, second + 1)]
    assert [m.line for m in lyr.moras] == [0, 1]
    assert [u.is_star for u in lyr.units] == [True, False, True, False, True]
    doc = frontend().parse(text)
    assert doc['text'] == text and [w['line'] for w in doc['words']] == [0, 1]
    result = synthetic_result(doc)
    assert [(x['text'], x['char_start'], x['char_end']) for x in result['lines']] == [('あ', 0, 1), ('い', second, second + 1)]
    assert [c['line'] for c in result['chars']] == [0] * second + [1]
    assert ''.join(c['char'] for c in result['chars']) == text
    export = synthetic_export(doc)
    assert [p['text'] for p in export['phrases']] == ['あ', 'い']


@pytest.mark.parametrize('sep', SEPARATORS)
def test_hint_and_proxy_line_selectors_use_physical_lines(sep):
    text = 'hello' + sep + 'hello'
    doc = frontend().parse(text, [dict(text='hello', language='en', line=2)])
    assert [w['line'] for w in doc['words']] == [0, 1]
    assert [w['language_source'] for w in doc['words']][1] == 'explicit_hint'
    assert doc['words'][0]['language_source'] != 'explicit_hint'
    doc = frontend().parse(text, [dict(text='hello', language='en')])
    n = len(doc['words'][1]['syllable_ids'])
    apply_proxy_readings(doc, MODEL, [dict(text='hello', line=2, syllables=[['ハ'] * n])])
    assert [s['proxy_source'] for s in doc['syllables']] == \
           ['automatic_phonetic_approximation'] * n + ['explicit_sung_approximation'] * n


# ---------------------------------------------------------------- strict proxy readings
def test_proxy_readings_use_the_strict_contextual_mora_rules():
    doc = frontend().parse('hello', [dict(text='hello', language='en')])
    n = len(doc['words'][0]['syllable_ids'])
    with pytest.raises(LyricsInputError) as caught:
        apply_proxy_readings(doc, MODEL, [dict(text='hello', syllables=[['ーあ'] + ['ロ'] * (n - 1)])])
    issue, = caught.value.issues
    assert issue['reason'] == 'proxy_reading_invalid' and issue['proxy_reading'] == 'ーあ'
    assert (issue['char_start'], issue['char_end'], issue['line'], issue['column']) == (0, 5, 1, 1)
    assert issue['proxy_issue']['reason'] == 'no_preceding_sound' and issue['proxy_issue']['char_start'] == 0
    assert 'line 1 col 1' in str(caught.value)
    doc = frontend().parse('hello', [dict(text='hello', language='en')])
    with pytest.raises(LyricsInputError) as caught:
        apply_proxy_readings(doc, MODEL, [dict(text='hello', syllables=[['ha'] + ['ロ'] * (n - 1)])])
    assert caught.value.issues[0]['reason'] == 'proxy_reading_invalid'


def test_multi_mora_proxy_stays_one_syllable():
    doc = frontend().parse('knock out', [dict(text='knock out', language='en')])
    apply_proxy_readings(doc, MODEL, [dict(text='knock out', syllables=[['ノッ'], ['キャウ']])])
    assert [s['model_phones'] for s in doc['syllables']] == [['n', 'o', 'cl'], ['ky', 'a', 'u']]
    assert [w['syllable_ids'] for w in doc['words']] == [[0], [1]]
    assert [s['source_chars'] for s in doc['syllables']] == [list(range(0, 5)), list(range(6, 9))]


# ---------------------------------------------------------------- configured languages vs content
def test_content_in_a_disabled_language_is_a_positioned_input_issue():
    front = Frontend(MODEL, languages=('en',))
    for text, start in [('あ', 0), ('hello あ', 6)]:
        with pytest.raises(LyricsInputError) as caught:
            front.parse(text, [dict(text='hello', language='en')] if 'hello' in text else ())
        issue = next(x for x in caught.value.issues if x['char_start'] == start)
        assert issue['reason'] == 'language_not_enabled' and 'ja' in issue['message'] and issue['suggestion']
    with pytest.raises(LyricsInputError) as caught:
        Frontend(MODEL, languages=('ja', 'en')).parse('사랑')
    assert [(x['char_start'], x['reason']) for x in caught.value.issues] == [(0, 'language_not_enabled')]
    # Invalid option configuration is still a configuration error.
    with pytest.raises(ValueError) as caught:
        Frontend(MODEL, languages=('xx',))
    assert not isinstance(caught.value, LyricsInputError)


# ---------------------------------------------------------------- explicit readings overrides
JA_EN = ('ja', 'en')


def readings_file(tmp_path, body):
    path = tmp_path / 'readings.txt'
    path.write_text(body, encoding='utf-8')
    return path


def legacy_phones(lyr):
    return [t for u in lyr.units if not u.is_star for t in u.tokens]


def legacy_export(lyr):
    """Legacy mora result (output.build_result) -> UTAVISTA, with synthetic per-mora times."""
    export = result_to_lyrics_timing(legacy_result(lyr), ruby=True)
    assert self_check(export) == []
    return export


def legacy_result(lyr):
    from utalign.matcher import MoraAssign
    from utalign.output import build_result as legacy_build_result
    from utalign.timing import MatchResult, MoraTime
    n = len(lyr.moras)
    notes = [SimpleNamespace(id=i, start=.2 + i * .5, end=.6 + i * .5, pitch=60, phrase=0) for i in range(n)]
    times = [MoraTime(start=.2 + i * .5, end=.6 + i * .5, source='midi', end_kind='note_off', midi_start=.2 + i * .5,
                      ctc_start=None, phrase=0) for i in range(n)]
    match = MatchResult(assigns=[MoraAssign(note_ids=[i]) for i in range(n)], offset=0., residual_median=0.,
                        residual_p90=0., residual_over150=0., drift_slope=0., n_mora_skipped=0, n_note_skipped=0,
                        cost=0.)
    return legacy_build_result(lyr, notes, match, times, {})


@pytest.mark.parametrize('text,reading,expected', [('かな', 'き', 'キ'), ('あー', 'い', 'イ'), ('すーん', 'い', 'イ'),
                                                   ('う', 'かき', 'カキ'), ('髙', 'たか', 'タカ')])
def test_count_changing_override_is_never_replaced_by_the_spelling(tmp_path, text, reading, expected):
    readings = readings_file(tmp_path, f'{text} {reading}\n')
    lyr = prepare_mora_lyrics(text, readings, MODEL.mora_tokens)
    assert ''.join(m.kana for m in lyr.moras) == expected
    assert {i for m in lyr.moras for i in m.char_idx} == set(range(len(text)))
    # Several characters with no per-character reading: one envelope over the whole spelling.
    assert [m.envelope for m in lyr.moras] == [len(text) > 1] * len(lyr.moras)
    for hints in ((), [dict(text=text, language='ja')]):
        doc = Frontend(MODEL, languages=JA_EN, readings=readings).parse(text, hints)
        assert ''.join(s['reading'] for s in doc['syllables']) == expected
        assert {i for s in doc['syllables'] for i in s['source_chars']} == set(range(len(text)))
        assert [p for s in doc['syllables'] for p in s['model_phones']] == legacy_phones(lyr)


def test_valid_per_character_and_token_group_overrides(tmp_path):
    readings = readings_file(tmp_path, '言葉 か|き\n')
    for text in ('言葉', '言葉が'):
        lyr = prepare_mora_lyrics(text, readings, MODEL.mora_tokens)
        assert [(m.kana, m.char_idx, m.envelope) for m in lyr.moras][:2] == [('カ', [0], False), ('キ', [1], False)]
        doc = Frontend(MODEL, languages=JA_EN, readings=readings).parse(text, [dict(text=text, language='ja')])
        moras = [(m['kana'], m['source_chars']) for s in doc['syllables'] for m in s['moras']]
        assert moras[:2] == [('カ', [0]), ('キ', [1])]
        assert [p for s in doc['syllables'] for p in s['model_phones']] == legacy_phones(lyr)


@pytest.mark.parametrize('key,reading,phones', [('100', 'ひゃく', 'hy a k u'), ('LOVE', 'らぶ', 'r a b u'),
                                                ('LINE', 'らいん', 'r a i N'), ('1000', 'せん', 's e N')])
def test_whole_spelling_override_reads_every_character(tmp_path, key, reading, phones):
    readings = readings_file(tmp_path, f'{key} {reading}\n')
    lyr = prepare_mora_lyrics(key, readings, MODEL.mora_tokens)
    assert legacy_phones(lyr) == phones.split()
    assert all(m.envelope and m.char_idx == list(range(len(key))) and not m.split_estimated for m in lyr.moras)
    doc = Frontend(MODEL, languages=JA_EN, readings=readings).parse(key)
    assert [p for s in doc['syllables'] for p in s['model_phones']] == phones.split()
    word, = doc['words']
    assert (word['text'], word['language'], word['language_source']) == (key, 'ja', 'readings_override')
    assert all(s['source_chars'] == list(range(len(key))) for s in doc['syllables'])
    assert all(m.get('envelope') for s in doc['syllables'] for m in s['moras'])
    assert not doc['language_diagnostics']


@pytest.mark.parametrize('path', ['japanese', 'multilingual'])
def test_whole_spelling_override_shares_one_interval_and_exports_no_per_character_ruby(tmp_path, path):
    """lyrics-timing/1.0 ruby is per character (consumers read only char.ruby): a reading of the whole spelling
    is never put on one character, duplicated or redistributed; it stays in the result moras/syllables."""
    readings = readings_file(tmp_path, '100 ひゃく\n')
    if path == 'japanese':
        lyr = prepare_mora_lyrics('100', readings, MODEL.mora_tokens)
        result, export = legacy_result(lyr), legacy_export(lyr)
        assert [(m['kana'], m['char_idx']) for m in result['moras']] == [('ヒャ', [0, 1, 2]), ('ク', [0, 1, 2])]
        warning = envelope_ruby_warning(envelope_groups(lyr))
    else:
        doc = Frontend(MODEL, languages=JA_EN, readings=readings).parse('100')
        result, export = synthetic_result(doc), synthetic_export(doc)
        assert [(m['kana'], m['source_chars']) for s in result['syllables'] for m in s['moras']] == [
            ('ヒャ', [0, 1, 2]), ('ク', [0, 1, 2])]
        warning = next(w for w in input_warnings(doc) if 'whole-spelling' in w)
    assert [c['char'] for c in result['chars']] == list('100')
    assert all(c['ruby_reading'] == '' and c['ruby_omitted'] for c in result['chars'])
    assert "'100'=ヒャク" in warning and "'|'" in warning
    chars = export_chars(export)
    assert ''.join(c['char'] for c in chars) == '100'
    assert len({(c['startMs'], c['endMs']) for c in chars}) == 1       # no invented character onsets
    assert all(c['timingBasis'] == 'word_envelope' for c in chars)
    assert not any('ruby' in c or 'rubySpan' in c for c in chars)


@pytest.mark.parametrize('path', ['japanese', 'multilingual'])
def test_exact_per_character_and_estimated_ruby_is_unchanged(tmp_path, path):
    readings = readings_file(tmp_path, '2人 ふた|り\n')
    for text, expected in [('2人', {'2': 'ふた', '人': 'り'}), ('大人', None)]:
        if path == 'japanese':
            export = legacy_export(prepare_mora_lyrics(text, readings, MODEL.mora_tokens))
        else:
            export = synthetic_export(Frontend(MODEL, languages=JA_EN, readings=readings).parse(
                text, [dict(text=text, language='ja')]))
        ruby = {c['char']: c.get('ruby') for c in export_chars(export)}
        if expected:
            assert ruby == expected
        else:   # estimated kanji split: each kanji keeps its own (estimated) ruby, together the whole reading
            assert all(ruby[k] for k in text) and ''.join(ruby[k] for k in text) == 'おとな'


@pytest.mark.parametrize('line,text,span', [('言葉 か|き|く', 'あ 言葉', (2, 4)), ('LOVE ら|ぶ', 'あ LOVE', (2, 6)),
                                           ('2 two', 'あ2', (1, 2)), ('言葉 か|', '言葉', (0, 2))])
def test_unmappable_override_is_a_positioned_issue(tmp_path, line, text, span):
    readings = readings_file(tmp_path, line + '\n')
    for parse in (lambda: prepare_mora_lyrics(text, readings, MODEL.mora_tokens),
                  lambda: Frontend(MODEL, languages=JA_EN, readings=readings).parse(text)):
        with pytest.raises(LyricsInputError) as caught:
            parse()
        assert [(x['char_start'], x['char_end'], x['reason']) for x in caught.value.issues] == [
            (*span, 'reading_override_unmappable')]
        assert caught.value.issues[0]['suggestion']


def test_override_keys_do_not_capture_parts_of_larger_words(tmp_path):
    readings = readings_file(tmp_path, 'ME みー\nI あい\n2 ふた\n')
    front = Frontend(MODEL, languages=JA_EN, readings=readings)
    doc = front.parse('HOME In')
    assert [(w['text'], w['language']) for w in doc['words']] == [('HOME', 'en'), ('In', 'en')]
    doc = front.parse('TIME ME I')
    assert [(w['text'], w['language']) for w in doc['words']] == [('TIME', 'en'), ('ME', 'ja'), ('I', 'ja')]
    assert [w['language_source'] for w in doc['words']][1:] == ['readings_override'] * 2
    # Case-sensitive keys: 'I' never matches the lowercase i of 'Hi' (unchanged behaviour, not a boundary rule).
    assert [w['language'] for w in front.parse('Hi')['words']] == ['en']
    with pytest.raises(LyricsInputError) as caught:
        prepare_mora_lyrics('HOME', readings, MODEL.mora_tokens)
    assert {(x['char_start'], x['char_end'], x['reason']) for x in caught.value.issues} == {(0, 4, LATIN_JA)}
    # A digit key never reads part of a longer number ('2' in '12').
    for parse in (lambda: prepare_mora_lyrics('12', readings, MODEL.mora_tokens), lambda: front.parse('12')):
        with pytest.raises(LyricsInputError) as caught:
            parse()
        assert [(x['char_start'], x['char_end'], x['reason']) for x in caught.value.issues] == [
            (0, 2, 'number_needs_reading')]


@pytest.mark.parametrize('path', ['japanese', 'multilingual'])
@pytest.mark.parametrize('text,at', [('三十七℃', 3), ('あなた＆わたし', 3), ('百％', 1), ('五¥', 1), ('あ™', 1),
                                     ('あ$', 1), ('あ€', 1), ('あ@', 1)])
def test_reading_bearing_symbols_are_never_silently_unsung(text, at, path):
    assert requires_reading(text, at)
    out = run(text, path)
    assert out.status == 'unresolved', f'{text!r} silently accepted on {path}'
    check_unresolved_invariants(out)
    issue = next(x for x in out.issues if x['char_start'] <= at < x['char_end'])
    assert issue['reason'] == 'symbol_needs_reading' and 'readings override' in issue['suggestion']


def test_symbol_reading_is_given_explicitly_or_by_annotation(tmp_path):
    with pytest.raises(LyricsInputError) as caught:
        Frontend(MODEL, languages=JA_EN).parse('you & me')
    assert [(x['char_start'], x['reason']) for x in caught.value.issues] == [(4, 'symbol_needs_reading')]
    readings = readings_file(tmp_path, '℃ ど\n')
    for path in ('japanese', 'multilingual'):
        out = run('三十七℃', path, readings=readings)
        assert out.status == 'ok', getattr(out, 'message', '')
        check_ok_invariants(out)
        assert 3 in out.covered
        out = run('百(％)', path)
        assert out.status == 'ok', getattr(out, 'message', '')
        check_ok_invariants(out)
    # Ordinary punctuation and symbols stay deliberately unsung.
    for ch in '!?・♪→～':
        assert not requires_reading('あ' + ch, 1)


@pytest.mark.parametrize('text,units,reading', [('OKー', [['o'], ['ke']], 'オッケー'), ('2ー', [['ni']], 'ニー'),
                                                ('2っ', [['ni']], 'ニッ')])
def test_override_chunk_boundary_is_not_a_sound_boundary(tmp_path, text, units, reading):
    readings = readings_file(tmp_path, 'OK おっけ\n2 に\n')
    lyr = prepare_mora_lyrics(text, readings)        # romaji: ー/ッ fold into the preceding unit
    assert [u.tokens for u in lyr.units if not u.is_star] == units
    legacy, multi = run(text, 'japanese', readings=readings), run(text, 'multilingual', readings=readings)
    assert legacy.status == multi.status == 'ok'
    check_ok_invariants(legacy)
    check_ok_invariants(multi)
    assert legacy.phones == multi.phones
    assert ''.join(multi.readings) == reading


def test_real_boundaries_after_an_override_are_kept(tmp_path):
    readings = readings_file(tmp_path, 'OK おっけ\n2 に\n')
    for text in ['2 ー', '2、ー', 'OK ー']:
        for path in ('japanese', 'multilingual'):
            out = run(text, path, readings=readings)
            assert out.status == 'unresolved', (text, path)
            assert [x['reason'] for x in out.issues if x['text'] == 'ー'] == ['no_preceding_sound']
    multi = run('2 っ', 'multilingual', readings=readings)
    assert multi.readings == ['ニ', 'ッ']
    assert [u.tokens for u in run('2 っ', 'japanese', readings=readings).lyr.units if not u.is_star] == [
        ['n', 'i'], ['cl']]


# ---------------------------------------------------------------- overrides vs enabled languages and hints
def test_override_never_enables_a_disabled_language(tmp_path):
    readings = readings_file(tmp_path, '2 ふた\nLINE らいん\n')
    with pytest.raises(LyricsInputError) as caught:
        Frontend(MODEL, languages=('en',), readings=readings).parse('2')
    issue, = caught.value.issues
    assert (issue['char_start'], issue['char_end'], issue['reason']) == (0, 1, 'language_not_enabled')
    assert "'2'" in issue['message'] and 'ja' in issue['suggestion']
    doc = Frontend(MODEL, languages=JA_EN, readings=readings).parse('2')
    assert [(w['language'], w['language_source']) for w in doc['words']] == [('ja', 'readings_override')]
    assert [s['reading'] for s in doc['syllables']] == ['フ', 'タ']
    # An explicit language hint takes precedence over the Japanese reading for the span it covers.
    for languages in (('en',), JA_EN):
        doc = Frontend(MODEL, languages=languages, readings=readings).parse('LINE', [dict(text='LINE', language='en')])
        assert [(w['language'], w['language_source']) for w in doc['words']] == [('en', 'explicit_hint')]


def test_override_neighbours_are_an_unverified_inference(tmp_path):
    readings = readings_file(tmp_path, '1 いっ\n')
    front = Frontend(MODEL, languages=('ja', 'zh'), readings=readings)
    doc = front.parse('第1章')
    assert {w['language'] for w in doc['words']} == {'ja'}
    assert [(d['char_start'], d['char_end'], d['basis'], d['confidence']) for d in doc['language_diagnostics']] == [
        (0, 1, 'readings_override_context', None), (2, 3, 'readings_override_context', None)]
    for w in doc['words']:
        if (w['char_start'], w['char_end']) == (1, 2):
            assert (w['language_source'], w['language_confidence']) == ('readings_override', 1.0)
        else:
            assert (w['language_source'], w['language_confidence']) == ('readings_override_context', None)
    assert any('2 text spans' in x for x in input_warnings(doc))
    hinted = front.parse('第1章', [dict(text='第1章', language='ja')])
    assert not hinted['language_diagnostics']
    assert {w['language_source'] for w in hinted['words']} == {'explicit_hint'}
    assert [s['reading'] for s in hinted['syllables']] == [s['reading'] for s in doc['syllables']]


# ---------------------------------------------------------------- override keys inside Han runs
HAN_READINGS = '世界 せ|かい\n日本 に|っぽん\n'


@pytest.mark.parametrize('text,key', [('中国世界', (2, 4)), ('世界人民', (0, 2)), ('你好世界', (2, 4)),
                                      ('世界你好', (0, 2))])
def test_override_key_inside_a_han_run_is_never_replaced_by_chinese(tmp_path, text, key):
    """Either the Japanese reading is used for the key, or a positioned ambiguity asks for a whole-word hint.
    Which one depends on the detector (not pinned); a zh pronunciation of the key is never accepted."""
    front = Frontend(MODEL, languages=('ja', 'zh'), readings=readings_file(tmp_path, HAN_READINGS))
    try:
        doc = front.parse(text)
    except LyricsInputError as exc:
        issue, = exc.issues
        assert issue['reason'] == 'reading_override_language_ambiguous'
        assert (issue['char_start'], issue['char_end']) == (0, len(text))
        assert f"'ja: {text}'" in issue['suggestion'] and f"'zh: {text}'" in issue['suggestion']
        assert 'col' in issue['message'] and "'世界'" in issue['message']
    else:
        own = [s for s in doc['syllables'] if set(s['source_chars']) <= set(range(*key))]
        assert own and {s['language'] for s in own} == {'ja'}
        assert ''.join(m['kana'] for s in own for m in s['moras']) == 'セカイ'


def test_han_run_spelled_wholly_by_override_keys_is_explicit_japanese(tmp_path):
    front = Frontend(MODEL, languages=('ja', 'zh'), readings=readings_file(tmp_path, HAN_READINGS))
    doc = front.parse('日本世界')
    assert {(w['language'], w['language_source']) for w in doc['words']} == {('ja', 'readings_override')}
    assert ''.join(s['reading'] for s in doc['syllables']) == 'ニッポンセカイ'
    assert not doc['language_diagnostics']


def test_han_override_respects_hints_and_enabled_languages(tmp_path):
    readings = readings_file(tmp_path, HAN_READINGS)
    plain = Frontend(MODEL, languages=('ja', 'zh')).parse('中国世界', [dict(text='中国世界', language='zh')])
    # An explicit zh hint for the whole word wins over the Japanese reading override.
    doc = Frontend(MODEL, languages=('ja', 'zh'), readings=readings).parse('中国世界', [dict(text='中国世界', language='zh')])
    assert [(w['language'], w['language_source']) for w in doc['words']] == [('zh', 'explicit_hint')]
    assert [s['reading'] for s in doc['syllables']] == [s['reading'] for s in plain['syllables']]
    # An explicit ja hint uses the override for its whole tokens and the dictionary for the rest.
    doc = Frontend(MODEL, languages=('ja', 'zh'), readings=readings).parse('中国世界', [dict(text='中国世界', language='ja')])
    assert {w['language'] for w in doc['words']} == {'ja'}
    assert [m['kana'] for s in doc['syllables'] for m in s['moras'] if m['source_chars'][0] >= 2] == ['セ', 'カ', 'イ']
    # Japanese disabled: the override contradicts the configuration -> positioned, not silently zh.
    for text in ('中国世界', '日本世界'):
        with pytest.raises(LyricsInputError) as caught:
            Frontend(MODEL, languages=('zh', 'en'), readings=readings).parse(text)
        issue, = caught.value.issues
        assert (issue['char_start'], issue['char_end'], issue['reason']) == (0, 4, 'language_not_enabled')
        assert "'ja'" in issue['suggestion']


OVERLAP_READINGS = '日本世界 にっ|ぽん|せ|かい\n世界人民 せ|かい|じん|みん\n'


def test_reader_tiles_overlapping_keys_longest_first_left_to_right(tmp_path):
    """The keys the reader applies (independent of any language detector): 日本世界, then dictionary 人民."""
    from utalign.lyrics import Reader, _token_groups
    reader = Reader(readings_file(tmp_path, OVERLAP_READINGS))
    groups = _token_groups('日本世界人民', reader)
    assert [(surf, ov is not None) for surf, _, ov in groups][0] == ('日本世界', True)
    assert ''.join(surf for surf, _, _ in groups) == '日本世界人民'
    assert not any(ov is not None for surf, _, ov in groups[1:])     # 世界人民 overlaps: never applied


def test_overlapping_keys_never_give_explicit_certainty_to_the_unused_tail(tmp_path):
    readings = readings_file(tmp_path, OVERLAP_READINGS)
    front = Frontend(MODEL, languages=('ja', 'zh'), readings=readings)
    try:
        doc = front.parse('日本世界人民')
    except LyricsInputError as exc:
        # Detector chose non-ja for the partly explicit run: positioned ambiguity over the whole word.
        issue, = exc.issues
        assert (issue['char_start'], issue['char_end'], issue['reason']) == (0, 6, 'reading_override_language_ambiguous')
        assert "'日本世界'" in issue['message'] and "'世界人民'" not in issue['message']
    else:
        tail = [w for w in doc['words'] if w['char_end'] > 4]
        assert tail and not any(w['language_source'] == 'readings_override' for w in tail)
        assert any(d['char_start'] <= 4 and 6 <= d['char_end'] for d in doc['language_diagnostics'])
    # With a whole-word ja hint: 日本世界 is read from its key, 人民 from the dictionary (not 世界人民's じんみん split).
    doc = front.parse('日本世界人民', [dict(text='日本世界人民', language='ja')])
    moras = [(m['kana'], m['source_chars']) for s in doc['syllables'] for m in s['moras']]
    assert [k for k, chars in moras if chars[0] < 4] == ['ニ', 'ッ', 'ポ', 'ン', 'セ', 'カ', 'イ']
    assert {w['language_source'] for w in doc['words']} == {'explicit_hint'}


def test_nonoverlapping_tiling_of_the_whole_han_run_stays_explicit(tmp_path):
    for body in ('日本 に|っぽん\n世界 せ|かい\n人民 じん|みん\n',
                 # The reader's tiling 日本世界|人民 covers the run even though 世界人民 also exists.
                 OVERLAP_READINGS + '人民 じん|みん\n'):
        doc = Frontend(MODEL, languages=('ja', 'zh'), readings=readings_file(tmp_path, body)).parse('日本世界人民')
        assert {(w['language'], w['language_source']) for w in doc['words']} == {('ja', 'readings_override')}
        assert ''.join(s['reading'] for s in doc['syllables']) == 'ニッポンセカイジンミン'
        assert not doc['language_diagnostics']


def test_key_that_is_not_a_whole_token_does_not_change_the_han_run(tmp_path):
    # '本世' would only match across the token boundary of 日本|世界: never applied, nothing changes.
    readings = readings_file(tmp_path, '本世 ほん|せ\n')
    for languages in (('ja', 'zh'), ('zh', 'en')):
        try:
            expected = Frontend(MODEL, languages=languages).parse('日本世界')
        except LyricsInputError as exc:
            expected = [(x['char_start'], x['reason']) for x in exc.issues]
        try:
            got = Frontend(MODEL, languages=languages, readings=readings).parse('日本世界')
        except LyricsInputError as exc:
            got = [(x['char_start'], x['reason']) for x in exc.issues]
        if isinstance(expected, dict):
            assert [(w['text'], w['language']) for w in got['words']] == [(w['text'], w['language']) for w in expected['words']]
            assert [s['reading'] for s in got['syllables']] == [s['reading'] for s in expected['syllables']]
        else:
            assert got == expected


# ---------------------------------------------------------------- proxy hints: same matching as language hints
def test_proxy_hints_match_through_the_analysis_text_and_grapheme_boundaries():
    for spelled in ('LOVE', 'ＬＯＶＥ'):
        doc = frontend().parse('ＬＯＶＥ', [dict(text='LOVE', language='en')])
        n = len(doc['words'][0]['syllable_ids'])
        apply_proxy_readings(doc, MODEL, [dict(text=spelled, line=1, char_start=0, syllables=[['ラ'] * n])])
        assert [s['proxy_source'] for s in doc['syllables']] == ['explicit_sung_approximation'] * n
    nfd = 'café'
    doc = frontend().parse(nfd, [dict(text=nfd, language='fr')])
    with pytest.raises(ValueError, match='does not match'):
        apply_proxy_readings(doc, MODEL, [dict(text='cafe', syllables=[['カ', 'フェ']])])


# ---------------------------------------------------------------- legacy result -> UTAVISTA for NFD / halfwidth
@pytest.mark.parametrize('text,group', [('が', [0, 1]), ('ﾊﾟｰﾃｨｰ', [0, 1]), ('か゛い', [0, 1]),
                                        ('ぱ︀い', [0, 1]), ('ﾊ゚い', [0, 1])])
def test_legacy_result_export_keeps_marks_with_their_base(text, group):
    chars = export_chars(legacy_export(prepare_mora_lyrics(text, None, MODEL.mora_tokens)))
    assert ''.join(c['char'] for c in chars) == text
    assert len({(chars[k]['startMs'], chars[k]['endMs']) for k in group}) == 1
    if group[-1] + 1 < len(chars):
        # The following independent character (e.g. い) keeps its own interval, after the base's.
        nxt = chars[group[-1] + 1]
        assert (nxt['startMs'], nxt['endMs']) != (chars[group[0]]['startMs'], chars[group[0]]['endMs'])
        assert nxt['startMs'] >= chars[group[0]]['endMs']
    assert not any('ruby' in c for c in chars)


# ---------------------------------------------------------------- model inventory covers the whole mora
class KanaVocabAligner:
    """A hiragana-vocabulary model without ゔ, ッ and ー (but with ぁ)."""
    from utalign.aligners.hf_ctc import HFCTCAligner
    mora_tokens = HFCTCAligner.mora_tokens
    vocab = {k: i for i, k in enumerate(['<pad>', 'あ', 'い', 'う', 'ぁ', 'か', 'き'])}
    is_phoneme, has_hira, has_kata = False, True, False


class NoVPhonemeAligner(FakeAligner):
    vocab = {p: i for p, i in FakeAligner.vocab.items() if p != 'v'}


class NoTsPhonemeAligner(FakeAligner):
    vocab = {p: i for p, i in FakeAligner.vocab.items() if p != 'ts'}


def single_mora(kana):
    from utalign.lyrics import Mora, mora_romaji
    romaji, kind = mora_romaji(kana)
    return Mora(id=0, kana=kana, romaji=romaji, char_idx=[0], line=0, seg=0, kind=kind)


def test_model_inventory_must_cover_the_whole_mora():
    kana, no_v, no_ts = KanaVocabAligner(), NoVPhonemeAligner(), NoTsPhonemeAligner()
    assert kana.mora_tokens(single_mora('ヴァ'), None) == []          # not ['ぁ']
    assert kana.mora_tokens(single_mora('ァ'), None) == ['ぁ']
    assert no_v.mora_tokens(single_mora('ヴァ'), None) == []          # not ['a']
    assert MODEL.mora_tokens(single_mora('ヴァ'), None) == ['v', 'a']
    # Deliberate decomposition into phones that all exist is kept.
    assert no_ts.mora_tokens(single_mora('ツ'), None) == ['t', 's', 'u']
    assert no_ts.mora_tokens(single_mora('ツァ'), None) == ['t', 's', 'a']
    for aligner in (kana, no_v):
        with pytest.raises(LyricsInputError) as caught:
            prepare_mora_lyrics('あヴァ', None, aligner.mora_tokens)
        assert [(x['char_start'], x['char_end'], x['reason']) for x in caught.value.issues] == [(1, 3, 'no_model_phones')]
        with pytest.raises(LyricsInputError) as caught:
            Frontend(aligner, languages=('ja',)).parse('あヴァ')
        assert [(x['char_start'], x['char_end'], x['reason']) for x in caught.value.issues] == [(1, 3, 'no_model_phones')]
    # Legitimate phone-less specials still fold into the preceding sound of the same run.
    lyr = prepare_mora_lyrics('あーっ', None, kana.mora_tokens)
    assert [u.tokens for u in lyr.units if not u.is_star] == [['あ']]
    assert all(m.ctc_unit is not None for m in lyr.moras)
