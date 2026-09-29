"""Replaceable local pronunciation backends without eSpeak or subprocesses.

IPA is a pronunciation hypothesis, independent of the acoustic proxy. Chinese
keeps pinyin syllable boundaries instead of splitting diphthongs into vowels.
"""
from dataclasses import dataclass
import re
import unicodedata


@dataclass
class Pronunciation:
    phones: list[str]
    source: str
    parts: list[list[str]] | None = None
    variants: int = 1


def ipa_segments(text):
    """Keep IPA combining marks, affricate ties and nonsyllabic offglides."""
    segments = []
    tied = False
    for char in unicodedata.normalize('NFD', text):
        if unicodedata.combining(char) or char in 'ːˑʰʲʷ':
            if not segments:
                raise ValueError(f'IPA modifier without a phone: {text!r}')
            segments[-1] += char
            tied = tied or char in '\u0361\u035c'
        elif tied:
            segments[-1] += char
            tied = False
        elif char in 'ˈˌ':
            continue
        else:
            segments.append(char)
    if tied:
        raise ValueError(f'Unfinished IPA affricate: {text!r}')
    # An offglide belongs to the preceding vowel's nucleus (German ei/au/eu).
    vowels = set('aeiouɑɐɒæɔəɚɛɜɝɞɤɨɪɯɵøœʉʊʌʏyɶᵻ')
    merged = []
    for segment in segments:
        if '\u032f' in segment and merged and merged[-1][0] in vowels and '\u032f' not in merged[-1]:
            merged[-1] += segment
        else:
            merged.append(segment)
    return merged


def romance_glides(phones, spelling, language):
    """Resolve adjacent vowel nuclei that the rule tables leave unmarked.

    Spanish accented í/ú remain hiatus; unaccented weak vowels can be glides.
    Italian rising diphthongs use j/w, with common stressed hiatus words kept.
    These are phonological hypotheses, not acoustic measurements.
    """
    vowels = set('aeiouɛɔ')
    protected = {v for c, v in [('í', 'i'), ('ú', 'u')] if c in spelling}
    if language == 'it':
        protected |= {v for c, v in [('ì', 'i'), ('ù', 'u')] if c in spelling}
        if spelling in {'io', 'dio', 'dei', 'mio', 'mia', 'miei', 'mie', 'tuo', 'tua', 'tuoi', 'tue',
                        'suo', 'sua', 'suoi', 'sue', 'zio', 'zia', 'zii', 'zie', 'via', 'vie', 'paura'}:
            return phones
    out = list(phones)
    for i in range(len(out) - 1):
        a, b = out[i], out[i + 1]
        if a not in vowels or b not in vowels:
            continue
        if a in {'i', 'u'} - protected:
            out[i] = 'j' if a == 'i' else 'w'
        elif b in {'i', 'u'} - protected:
            out[i] += b + '̯'
            out[i + 1] = ''
    return [p for p in out if p]


class LocalG2P:
    CODES = {'de': 'deu-Latn', 'fr': 'fra-Latn', 'es': 'spa-Latn',
             'it': 'ita-Latn', 'ko': 'kor-Hang', 'zh': 'cmn-Latn'}

    def __init__(self):
        self.backends = {}
        self.english = None

    def _transcribe(self, text, language):
        # SimpleEpitran selects only packaged rules; no Flite/CEDICT backend.
        from epitran.simple import SimpleEpitran
        if language not in self.backends:
            self.backends[language] = SimpleEpitran(self.CODES[language])
        backend = self.backends[language]
        text = unicodedata.normalize('NFC', text).replace('’', "'").replace("'", '')
        before = dict(backend.nils)
        ipa = backend.transliterate(text)
        unknown = [c for c, count in backend.nils.items() if count > before.get(c, 0)]
        if unknown:
            raise ValueError(f'Unsupported spelling for {language}: {text!r} ({unknown}); check language hints')
        return ipa_segments(ipa)

    def pronounce(self, text, language):
        spelling = unicodedata.normalize('NFC', text).lower()
        if language == 'en':
            from .english import EnglishG2P, arpabet_to_ipa
            if self.english is None:
                self.english = EnglishG2P()
            phones, source, variants = self.english.lookup(text)
            return Pronunciation(arpabet_to_ipa(phones), source, variants=variants)
        if language == 'zh':
            from pypinyin import lazy_pinyin, Style

            def unknown(chars):
                raise ValueError(f'No Mandarin reading for {chars!r}; check the sung spelling')

            readings = lazy_pinyin(unicodedata.normalize('NFC', text), style=Style.NORMAL,
                                   v_to_u=True, errors=unknown)
            parts = [self._transcribe(reading, 'zh') for reading in readings]
            return Pronunciation([p for part in parts for p in part], 'pypinyin_epitran', parts)
        if language in self.CODES:
            if language == 'de':
                # Productive directional compounds retain the second morpheme's
                # h and stressed vowel; generic intervocalic-h deletion loses it.
                directional = re.fullmatch(r'(wo|da|hier|dort)?(her|hin)', spelling)
                if directional:
                    prefix = self._transcribe(directional[1], language) if directional[1] else []
                    tail = ['h', 'eː', 'ʁ'] if directional[2] == 'her' else ['h', 'ɪ', 'n']
                    return Pronunciation(prefix + tail, 'epitran_directional_morphemes')
            if language == 'fr':
                # Protect eau as one grapheme before the upstream schwa rules.
                spelling = spelling.replace('eau', 'ô')
            if language == 'it' and spelling == 'ciao':
                return Pronunciation(['tʃ', 'ao̯'], 'italian_pronunciation_rule')
            phones = self._transcribe(spelling, language)
            if language in {'es', 'it'}:
                phones = romance_glides(phones, spelling, language)
            return Pronunciation(phones, 'epitran_rules')
        raise ValueError(f'Unsupported pronunciation language: {language}')
