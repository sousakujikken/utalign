"""Language-aware words and phonetic syllables with lossless source positions.

Japanese uses the existing reader; foreign languages use local non-eSpeak G2P.
IPA is retained; Japanese-model approximations are a separate representation.
"""
import re
import unicodedata

import regex

LANGUAGES = ('ja', 'en', 'de', 'fr', 'zh', 'ko', 'es', 'it')
FOREIGN_LANGUAGES = set(LANGUAGES) - {'ja'}
WORD_RE = regex.compile(r"\p{Latin}[\p{Latin}\p{M}]*(?:['’]\p{Latin}[\p{Latin}\p{M}]*)*|[\p{Han}\p{Hiragana}\p{Katakana}ー]+|\p{Hangul}+")
VOWELS = set('aeiouɑɐɒæɔəɚɛɜɝɞɤɨɪɯɵøœʉʊʌʏyɶᵻ')
IPA_MAP = dict(zip('aeiou', 'aeiou')) | {
    'ɑ':'a','ɐ':'a','ɒ':'o','æ':'a','ɔ':'o','ə':'a','ɛ':'e','ɜ':'a','ɞ':'o',
    'ɤ':'o','ɨ':'u','ɪ':'i','ɯ':'u','ɵ':'o','ø':'e','œ':'e','ʉ':'u','ʊ':'u','ʌ':'a','ʏ':'u','y':'u','ɶ':'a',
    'ɚ':'a r','ɝ':'a r','ɻ':'r','ʝ':'y','ɿ':'i','ʅ':'i','ᵻ':'i',
    'b':'b','p':'p','m':'m','f':'f','v':'v','t':'t','d':'d','n':'n','s':'s','z':'z',
    'k':'k','ɡ':'g','g':'g','h':'h','w':'w','j':'y','r':'r','l':'r','ɹ':'r','ɾ':'r','ɽ':'r','ʁ':'r','ʀ':'r',
    'θ':'s','ð':'z','ʃ':'sh','ʒ':'j','ɕ':'sh','ʑ':'j','ŋ':'N','ɲ':'ny','ɳ':'n','ɴ':'N',
    'x':'h','χ':'h','ç':'hy','ɣ':'g','ɦ':'h','ʋ':'v','β':'b','ʔ':'cl','ɸ':'f','ɰ':'w','ɥ':'y',
    'ʎ':'ry','ʈ':'t','ɖ':'d','ʂ':'sh','ʐ':'j','ɭ':'r','ʙ':'b',
}
AFFRICATES = {'tʃ':'ch', 'dʒ':'j', 'tɕ':'ch', 'dʑ':'j', 'ts':'ts', 'dz':'z'}


def bare(phone):
    return ''.join(c for c in unicodedata.normalize('NFD', phone)
                   if not unicodedata.combining(c) and c not in 'ˈˌːˑ.0123456789ʰʲʷ')


def nucleus(phone):
    units = regex.findall(r'\P{M}\p{M}*', unicodedata.normalize('NFD', phone))
    return any(unit[0] in VOWELS and '̯' not in unit for unit in units) or '̩' in phone or 'ɿ' in phone or 'ʅ' in phone


def model_tokens(phone):
    normalized = bare(phone)
    if normalized in AFFRICATES:
        result = [AFFRICATES[normalized]]
    else:
        result = []
        for char in normalized:
            if char not in IPA_MAP:
                raise ValueError(f'Unmapped IPA phone {phone!r}, symbol {char!r}')
            result.extend(IPA_MAP[char].split())
    if '̃' in unicodedata.normalize('NFD', phone):
        result.append('N')
    return result


def syllabify(phones, language):
    """Vowel nuclei, diphthongs, and syllabic consonants; never kana character counts."""
    # Backends mark diphthongs explicitly; adjacent full vowels are hiatus.
    # In particular Paolo/paese must not be collapsed like ciao.
    nuclei = [i for i, p in enumerate(phones) if nucleus(p)]
    if not nuclei:
        raise ValueError('Pronunciation has no syllabic nucleus: ' + ' '.join(phones))
    cuts = [0]
    for a, b in zip(nuclei, nuclei[1:]):
        between = [bare(p) for p in phones[a+1:b]]
        # A conservative onset: final consonant; stop/fricative + liquid or glide.
        onset = 0 if not between else 1
        if between and between[-1] in {'ŋ','ɴ'}:
            onset = 0
        if len(between) >= 2 and between[-1] in {'r','ɹ','ɾ','l','j','w'} and between[-2] not in {'n','m','ŋ'}:
            onset = 2
        if language == 'ko' and len(between) >= 2 and between[-1] in {'j','w'}:
            onset = 2
        cuts.append(b-onset)
    cuts.append(len(phones))
    return [phones[a:b] for a,b in zip(cuts,cuts[1:])]


class Frontend:
    def __init__(self, aligner, languages=LANGUAGES, readings=None):
        from lingua import Language, LanguageDetectorBuilder
        names = dict(ja='JAPANESE', en='ENGLISH', de='GERMAN', fr='FRENCH', zh='CHINESE', ko='KOREAN', es='SPANISH', it='ITALIAN')
        if not languages or set(languages)-set(LANGUAGES):
            raise ValueError('Unsupported language configuration')
        self.languages = tuple(languages)
        self.detector = LanguageDetectorBuilder.from_languages(*(getattr(Language,names[x]) for x in languages)).build()
        self.aligner = aligner
        self.readings = readings
        from .g2p import LocalG2P
        self.g2p = LocalG2P()

    def parse(self, text, hints=()):
        from utalign.lyrics import BRACKET_RE, parse_lyrics, Reader
        masked = list(text)
        excluded = set()
        for m in BRACKET_RE.finditer(text):
            excluded.update(range(m.start(),m.end()))
        pos = 0
        for line in text.splitlines(keepends=True):
            if line.lstrip().startswith('©'):
                excluded.update(range(pos,pos+len(line.rstrip('\r\n'))))
            pos += len(line)
        for i in excluded:
            if masked[i] not in '\r\n':
                masked[i] = ' '
        masked = ''.join(masked)
        forced = {}
        for hint in hints:
            if not isinstance(hint.get('text'),str) or not hint['text']:
                raise ValueError('Language hint text must be nonempty')
            language = hint['language']
            if language not in self.languages:
                raise ValueError('Hint language not enabled: ' + language)
            matches = [m for m in re.finditer(re.escape(hint['text']), masked)
                       if ('line' not in hint or text.count('\n',0,m.start())+1 == hint['line'])
                       and ('char_start' not in hint or m.start() == hint['char_start'])]
            if not matches:
                raise ValueError('Language hint does not match source: ' + hint['text'])
            for match in matches:
                for i in range(match.start(),match.end()):
                    if i in forced and forced[i] != language:
                        raise ValueError('Overlapping language hints disagree')
                    forced[i] = language
        words, syllables, diagnostics = [], [], []

        def add_word(surface, start, language, source, confidence, specs):
            word = dict(id=len(words), text=surface, char_start=start, char_end=start+len(surface),
                        line=text.count('\n',0,start), language=language, language_source=source,
                        language_confidence=confidence, syllable_ids=[])
            words.append(word)
            for spec in specs:
                tokens = spec['model_phones']
                if not tokens or set(tokens)-self.aligner.vocab.keys():
                    raise ValueError(f'Unsupported model phones in {surface}: {tokens}')
                si = len(word['syllable_ids'])
                word['syllable_ids'].append(len(syllables))
                syllables.append(dict(spec, id=len(syllables), word_id=word['id'], syllable_index=si,
                                      language=language, line=word['line']))

        offset = 0
        for line in masked.splitlines(keepends=True):
            covered={i for match in WORD_RE.finditer(line) for i in range(match.start(),match.end())}
            unsupported=[(offset+i,c) for i,c in enumerate(line) if i not in covered and regex.match(r'[\p{L}\p{N}]',c)]
            if unsupported:
                raise ValueError(f'Unsupported lyric characters at {unsupported[:8]}; write numerals as sung words and use the supported language scripts')
            detected = self.detector.detect_multiple_languages_of(line) if line.strip() else []
            latin_line=''.join(c if regex.match(r"[\p{Latin}\s'’.,!?]",c) else ' ' for c in line)
            latin_detected=self.detector.detect_multiple_languages_of(latin_line) if latin_line.strip() else []
            for match in WORD_RE.finditer(line):
                a,b = offset+match.start(), offset+match.end()
                cuts = [a]+[i for i in range(a+1,b) if forced.get(i)!=forced.get(i-1)]+[b]
                for start,end in zip(cuts,cuts[1:]):
                    surface = text[start:end]
                    source, confidence = 'text_detection', None
                    if start in forced:
                        language, source, confidence = forced[start], 'explicit_hint', 1.0
                    elif regex.search(r'\p{Hangul}', surface):
                        language, source, confidence = 'ko','script',1.0
                    elif regex.search(r'[\p{Hiragana}\p{Katakana}]', surface):
                        language, source, confidence = 'ja','script',1.0
                    else:
                        latin=bool(regex.search(r'\p{Latin}',surface))
                        active_segments=latin_detected if latin else detected
                        overlaps = [(max(0,min(end-offset,s.end_index)-max(start-offset,s.start_index)),s) for s in active_segments]
                        overlap, segment = max(overlaps,key=lambda x:x[0]) if overlaps else (0,None)
                        language = segment.language.iso_code_639_1.name.lower() if overlap else None
                        if regex.search(r'\p{Latin}',surface) and language not in FOREIGN_LANGUAGES:
                            language = None
                        if language is None:
                            choices = self.detector.compute_language_confidence_values(surface)
                            language = choices[0].language.iso_code_639_1.name.lower()
                            confidence = choices[0].value
                        else:
                            fragment = (latin_line if latin else line)[segment.start_index:segment.end_index]
                            confidence = next((c.value for c in self.detector.compute_language_confidence_values(fragment)
                                               if c.language.iso_code_639_1.name.lower()==language),0.)
                    if language not in self.languages:
                        raise ValueError('Detected language is not enabled: '+language)
                    if source != 'explicit_hint' and regex.search(r'\p{Han}',surface) and language not in {'ja','zh'}:
                        allowed=[x for x in ('ja','zh') if x in self.languages]
                        if not allowed:
                            raise ValueError('Han text requires Japanese or Chinese')
                        choices=self.detector.compute_language_confidence_values(surface)
                        choice=next(c for c in choices if c.language.iso_code_639_1.name.lower() in allowed)
                        language=choice.language.iso_code_639_1.name.lower();confidence=choice.value
                        source='script_filtered_text_detection'
                    if source == 'text_detection':
                        diagnostics.append(dict(text=surface,char_start=start,language=language,confidence=confidence,
                                                status='language_inferred_not_verified'))
                    if language == 'ja':
                        parsed = parse_lyrics(surface, self.readings)
                        reader = Reader(self.readings)
                        local_pos = 0
                        for token in reader.tagger(surface):
                            wa = surface.index(token.surface,local_pos); wb=wa+len(token.surface);local_pos=wb
                            moras = [m for m in parsed.moras if wa <= min(m.char_idx) < wb]
                            specs=[]
                            for mora in moras:
                                ts=self.aligner.mora_tokens(mora,None)
                                indices=[start+i for i in mora.char_idx]
                                if specs and mora.kind in {'hatsuon','sokuon','choon'}:
                                    specs[-1]['model_phones'] += ts
                                    specs[-1]['reading'] += mora.kana
                                    specs[-1]['source_chars']=sorted(set(specs[-1]['source_chars']+indices))
                                elif ts:
                                    specs.append(dict(reading=mora.kana,ipa=None,model_phones=ts,source_chars=indices,
                                                      nucleus_token=next((i for i,t in enumerate(ts) if t in 'aiueo'),0),
                                                      pronunciation_source='japanese_reader',syllable_basis='japanese_moras_merged_specials'))
                            if not specs:
                                raise ValueError(f'Japanese reader produced no syllables for {token.surface!r}; specify the sung language or spelling')
                            add_word(token.surface,start+wa,'ja',source,confidence,specs)
                    else:
                        pronunciation=self.g2p.pronounce(surface,language)
                        parts=pronunciation.parts or syllabify(pronunciation.phones,language)
                        specs=[]
                        for part in parts:
                            mapper=getattr(self.aligner,'ipa_tokens',model_tokens)
                            mapped=[mapper(p) for p in part]
                            ni=next(i for i,p in enumerate(part) if nucleus(p))
                            specs.append(dict(reading=''.join(part),ipa=part,model_phones=[t for group in mapped for t in group],
                                              nucleus_token=sum(len(g) for g in mapped[:ni]),
                                              source_chars=list(range(start,end)),pronunciation_source=pronunciation.source,
                                              pronunciation_variants=pronunciation.variants,
                                              syllable_basis='pinyin_syllables' if language=='zh' else 'ipa_nuclei_heuristic',
                                              phone_mapping='multilingual_ipa_inventory' if hasattr(self.aligner,'ipa_tokens') else 'approximate_japanese_inventory'))
                        add_word(surface,start,language,source,confidence,specs)
            offset += len(line)
        if not syllables:
            raise ValueError('No pronounceable syllables')
        return dict(text=text,words=words,syllables=syllables,language_diagnostics=diagnostics,
                    source_offset_unit='unicode_code_point')
