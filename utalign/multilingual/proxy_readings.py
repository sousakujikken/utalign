"""Acoustic surrogate readings with stable original word/syllable ownership.

Kana is a pronunciation proxy, not a transcription or a new syllabification.
For example, ノッ | キャウ represents knock (1 syllable) | out (1 syllable).
The three kana in キャウ never become three source-language timing units.
"""
import re

from .kana import kana_hint


def apply_proxy_readings(document, aligner, hints=()):
    from utalign.lyrics import parse_lyrics
    for s in document['syllables']:
        s['proxy_reading'] = s['reading'] if s['language']=='ja' else kana_hint([s['model_phones']])
        s['proxy_source'] = 'japanese_reader' if s['language']=='ja' else 'automatic_phonetic_approximation'
        s['original_model_phones'] = list(s['model_phones'])
    changed = set()
    for hint in hints:
        if not hint.get('text'):
            raise ValueError('Proxy reading text must be nonempty')
        matches = [m for m in re.finditer(re.escape(hint['text']), document['text'])
                   if ('line' not in hint or document['text'].count('\n',0,m.start())+1==hint['line'])
                   and ('char_start' not in hint or m.start()==hint['char_start'])]
        if not matches:
            raise ValueError('Proxy reading does not match source: '+hint['text'])
        for match in matches:
            words = [w for w in document['words'] if match.start()<=w['char_start'] and w['char_end']<=match.end()]
            if not words or words[0]['char_start']!=match.start() or words[-1]['char_end']!=match.end() or len(words)!=len(hint['syllables']):
                raise ValueError('Proxy reading must cover complete source words, with one syllable list per word')
            for word, readings in zip(words, hint['syllables']):
                if len(readings)!=len(word['syllable_ids']):
                    raise ValueError('Proxy reading must retain every original syllable: '+word['text'])
                for si, kana in zip(word['syllable_ids'],readings):
                    if si in changed:
                        raise ValueError('Overlapping proxy readings')
                    if not isinstance(kana,str) or not re.fullmatch(r'[ァ-ヺーぁ-ゖ]+',kana):
                        raise ValueError('Proxy reading must be nonempty kana')
                    moras=parse_lyrics(kana).moras
                    phones=[p for mora in moras for p in aligner.mora_tokens(mora,None)]
                    if not phones or not any(p in {'a','i','u','e','o'} for p in phones) or set(phones)-aligner.vocab.keys():
                        raise ValueError('Proxy reading requires supported phones and a vowel nucleus')
                    syllable=document['syllables'][si]
                    syllable.update(model_phones=phones,proxy_reading=kana,proxy_source='explicit_sung_approximation',
                                    nucleus_token=next(i for i,p in enumerate(phones) if p in {'a','i','u','e','o'}),
                                    phone_mapping='sung_kana_proxy_preserving_original_syllable')
                    changed.add(si)
