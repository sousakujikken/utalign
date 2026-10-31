"""Acoustic surrogate readings with stable original word/syllable ownership.

Kana is a pronunciation proxy, not a transcription or a new syllabification.
For example, ノッ | キャウ represents knock (1 syllable) | out (1 syllable).
The three kana in キャウ never become three source-language timing units.

Each proxy is checked by the same strict contextual mora rules as lyrics
(utalign.lyrics.lyrics_issues): a long-vowel mark or a phone-less ッ/ン with no
preceding sound inside the proxy, or a reading with no model phones, is an error
reported at the original lyric word/syllable with the proxy input as the cause.
Nothing in a proxy is silently dropped. Proxy phones stay one syllable's phones.
"""
import re

from .kana import kana_hint


def _proxy_issue(document, syllable, word, index, kana, reason, cause, inner=None):
    from utalign.lyric_input import make_issue
    chars = syllable['source_chars'] or list(range(word['char_start'], word['char_end']))
    issue = make_issue(document['text'], min(chars), max(chars) + 1, reason,
                       f"proxy reading {kana!r} for syllable {index + 1} of {word['text']!r}: {cause}",
                       'write the proxy as sung kana for this one syllable, e.g. ノッ or キャウ')
    issue['proxy_reading'] = kana
    issue['syllable_id'] = syllable['id']
    if inner is not None:
        # Position inside the proxy text itself (the cause), next to the original lyric position.
        issue['proxy_issue'] = dict(char_start=inner['char_start'], char_end=inner['char_end'],
                                    text=inner['text'], reason=inner['reason'], message=inner['message'])
    return issue


def apply_proxy_readings(document, aligner, hints=()):
    from utalign.lyric_input import LyricsInputError, analysis_text, line_column
    from utalign.lyrics import Reader, lyrics_issues, parse_lyrics
    norm = analysis_text(document['text'])
    for s in document['syllables']:
        s['proxy_reading'] = s['reading'] if s['language']=='ja' else kana_hint([s['model_phones']])
        s['proxy_source'] = 'japanese_reader' if s['language']=='ja' else 'automatic_phonetic_approximation'
        s['original_model_phones'] = list(s['model_phones'])
    changed = set()
    issues = []
    reader = None
    for hint in hints:
        if not hint.get('text'):
            raise ValueError('Proxy reading text must be nonempty')
        # Same matching as language hints: the needle and the lyric are compared in the analysis text
        # (NFC, narrow width), a match may not start or end inside a grapheme or a reordered/expanded
        # cluster, and positions/line numbers (shared delimiters CRLF/LF/CR/U+2028...) are source positions.
        needle = analysis_text(hint['text']).text
        spans = [norm.source_span(m.start(), m.end()) for m in re.finditer(re.escape(needle), norm.text)
                 if norm.is_boundary(m.start()) and norm.is_boundary(m.end())]
        matches = [(s, e) for s, e in spans
                   if ('line' not in hint or line_column(document['text'], s)[0]==hint['line'])
                   and ('char_start' not in hint or s==hint['char_start'])]
        if not matches:
            raise ValueError('Proxy reading does not match source: '+hint['text'])
        for start, end in matches:
            words = [w for w in document['words'] if start<=w['char_start'] and w['char_end']<=end]
            if not words or words[0]['char_start']!=start or words[-1]['char_end']!=end or len(words)!=len(hint['syllables']):
                raise ValueError('Proxy reading must cover complete source words, with one syllable list per word')
            for word, readings in zip(words, hint['syllables']):
                if len(readings)!=len(word['syllable_ids']):
                    raise ValueError('Proxy reading must retain every original syllable: '+word['text'])
                for index, (si, kana) in enumerate(zip(word['syllable_ids'],readings)):
                    if si in changed:
                        raise ValueError('Overlapping proxy readings')
                    syllable=document['syllables'][si]
                    if not isinstance(kana,str) or not re.fullmatch(r'[ァ-ヺーぁ-ゖ]+',kana):
                        issues.append(_proxy_issue(document, syllable, word, index, kana, 'proxy_reading_invalid',
                                                   'Proxy reading must be nonempty kana'))
                        continue
                    reader = reader or Reader()
                    # Strict contextual path (same as lyrics): no leading ー, no borrowed sound, no phone-less mora.
                    parsed=parse_lyrics(kana, tokenizer=aligner.mora_tokens, reader=reader)
                    problems=lyrics_issues(parsed)
                    if problems:
                        for inner in problems:
                            issues.append(_proxy_issue(document, syllable, word, index, kana, 'proxy_reading_invalid',
                                                       f"{inner['message']} (at {inner['text']!r} in the proxy)", inner))
                        continue
                    phones=[p for mora in parsed.moras for p in mora.tokens]
                    if not phones or not any(p in {'a','i','u','e','o'} for p in phones) or set(phones)-aligner.vocab.keys():
                        issues.append(_proxy_issue(document, syllable, word, index, kana, 'proxy_reading_invalid',
                                                   f'Proxy reading requires supported phones and a vowel nucleus (got {phones})'))
                        continue
                    syllable.update(model_phones=phones,proxy_reading=kana,proxy_source='explicit_sung_approximation',
                                    nucleus_token=next(i for i,p in enumerate(phones) if p in {'a','i','u','e','o'}),
                                    phone_mapping='sung_kana_proxy_preserving_original_syllable')
                    changed.add(si)
    if issues:
        raise LyricsInputError(issues)
