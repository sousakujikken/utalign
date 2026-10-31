"""Language-aware words and phonetic syllables with lossless source positions.

Japanese uses the existing reader; foreign languages use local non-eSpeak G2P.
IPA is retained; Japanese-model approximations are a separate representation.

The original lyric text is never rewritten. Readers and language detection see
a separate analysis text (NFC, narrow width normalization; see
utalign.lyric_input) and every word/syllable/mora reports source code-point
positions. Japanese syllables are formed over the whole contiguous Japanese run,
so morphological token boundaries never reset long vowels, sokuon or hatsuon.
Unresolved input is reported with line/column/reason instead of being dropped.
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


def _token_boundaries(surface, reader):
    """Morphological token boundaries (local offsets, including 0 and len)."""
    bounds = {0, len(surface)}
    local_pos = 0
    for token in reader.tagger(surface):
        start = surface.find(token.surface, local_pos)
        if start < 0 or not token.surface:
            continue
        local_pos = start + len(token.surface)
        bounds.update((start, local_pos))
    return bounds


# Kana and marks used only in Japanese spelling (々/〆 are Han by script, ー is Common).
JAPANESE_CUE_RE = regex.compile(r'[\p{Hiragana}\p{Katakana}ー々〆ゝゞヽヾ]')
SPECIAL_MORAS = {'hatsuon', 'sokuon', 'choon'}
INFERRED_SOURCES = {'text_detection', 'script_filtered_text_detection'}
_NON_CJK_RE = regex.compile(r'[\p{Latin}\p{Hangul}]')


def _merge_override_runs(runs, text):
    """Join reading-override spans (e.g. '2' in 夜に2人) with touching CJK runs."""
    out = []
    for run in sorted(runs):
        if out:
            last = out[-1]
            plain = last if run[2] else run
            touching = run[0] == last[1] and (run[2] or last[2]) and not _NON_CJK_RE.match(text[plain[0]])
            if run[0] < last[1] or touching:
                last[1] = max(last[1], run[1])
                last[2] = last[2] or run[2]
                continue
        out.append(list(run))
    return out


def input_warnings(document, limit=5):
    """Human-readable counts of inferred/estimated mappings for meta.warnings."""
    warnings = []
    diagnostics = document.get('language_diagnostics') or []
    if diagnostics:
        examples = ', '.join(f"{d['line']}:{d['column']} {d['text']!r}->{d['language']}" for d in diagnostics[:limit])
        # Diagnostics are per text span (run, segment or override neighbour), not per display word.
        warnings.append(f"{len(diagnostics)} text spans use an inferred, unverified language ({examples}); "
                        "add language hints where the sung language differs")
    from utalign.lyrics import envelope_ruby_warning
    groups = {}
    for s in document.get('syllables', []):
        for m in s.get('moras') or []:
            if m.get('envelope'):
                groups.setdefault(tuple(m['source_chars']), []).append(m['kana'])
    ruby_warning = envelope_ruby_warning((document['text'][k[0]:k[-1] + 1], ''.join(v)) for k, v in groups.items())
    if ruby_warning:
        warnings.append(ruby_warning)
    estimated = [s for s in document.get('syllables', []) if s.get('split_estimated')]
    if estimated:
        examples = ', '.join(f"{s['reading']}@{s['source_chars'][0]}" for s in estimated[:limit])
        warnings.append(f"{len(estimated)} syllables use an estimated kanji reading split ({examples}); "
                        "check ruby/character timing or add readings overrides")
    return warnings


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
        from utalign.lyrics import KANA_ANY, KANJI, Reader, _token_groups, override_spans, parse_lyrics
        from utalign.lyric_input import (LINE_BREAK_CHARS, LyricsInputError, analysis_text, annotation_indices,
                                         coverage_issues, line_column, line_spans, make_issue)
        norm = analysis_text(text)
        atext = norm.text
        excluded = annotation_indices(atext)
        masked = ''.join(' ' if i in excluded and c not in LINE_BREAK_CHARS else c for i, c in enumerate(atext))
        reader = Reader(self.readings)
        forced = {}
        for hint in hints:
            if not isinstance(hint.get('text'),str) or not hint['text']:
                raise ValueError('Language hint text must be nonempty')
            language = hint['language']
            if language not in self.languages:
                raise ValueError('Hint language not enabled: ' + language)
            needle = analysis_text(hint['text']).text
            # Line numbers use the same delimiters as diagnostics (CRLF/LF/CR/U+2028...), on source positions.
            # A match may not start or end inside a grapheme or a reordered/expanded source cluster.
            matches = [m for m in re.finditer(re.escape(needle), masked)
                       if norm.is_boundary(m.start()) and norm.is_boundary(m.end())
                       and ('line' not in hint or line_column(text, norm.source_span(m.start(),m.end())[0])[0] == hint['line'])
                       and ('char_start' not in hint or norm.source_span(m.start(),m.end())[0] == hint['char_start'])]
            if not matches:
                raise ValueError('Language hint does not match source: ' + hint['text'])
            for match in matches:
                for i in range(match.start(),match.end()):
                    if i in forced and forced[i] != language:
                        raise ValueError('Overlapping language hints disagree')
                    forced[i] = language
        words, syllables, diagnostics, issues = [], [], [], []
        reported = set()   # analysis indices already explained by an issue
        represented = set()  # analysis indices read by some syllable
        token_at = {}      # source index -> morphological token lacking a reading

        def report(a, b, reason, message, suggestion=None):
            s, e = norm.source_span(a, b)
            issues.append(make_issue(text, s, e, reason, message, suggestion))
            reported.update(range(a, b))

        def add_word(a, b, language, source, confidence, specs, analysis_read):
            s, e = norm.source_span(a, b)
            for spec in specs:
                tokens = spec['model_phones']
                if not tokens or set(tokens)-self.aligner.vocab.keys():
                    report(a, b, 'model_phones_unsupported', f'Unsupported model phones in {text[s:e]}: {tokens}')
                    return
            represented.update(analysis_read)
            word = dict(id=len(words), text=text[s:e], char_start=s, char_end=e,
                        line=line_column(text, s)[0]-1, language=language, language_source=source,
                        language_confidence=confidence, syllable_ids=[])
            words.append(word)
            for spec in specs:
                si = len(word['syllable_ids'])
                word['syllable_ids'].append(len(syllables))
                syllables.append(dict(spec, id=len(syllables), word_id=word['id'], syllable_index=si,
                                      language=language, line=word['line']))

        def resolve(surface, a, b, line, offset, detected, latin_line, latin_detected):
            """Run-level language; (language, source, confidence) or None after reporting."""
            source, confidence = 'text_detection', None
            if regex.search(r'\p{Hangul}', surface):
                language, source, confidence = 'ko','script',1.0
            elif JAPANESE_CUE_RE.search(surface):
                language, source, confidence = 'ja','script',1.0
            else:
                latin=bool(regex.search(r'\p{Latin}',surface))
                active_segments=latin_detected if latin else detected
                overlaps = [(max(0,min(b-offset,s.end_index)-max(a-offset,s.start_index)),s) for s in active_segments]
                overlap, segment = max(overlaps,key=lambda x:x[0]) if overlaps else (0,None)
                language = segment.language.iso_code_639_1.name.lower() if overlap else None
                if latin and language not in FOREIGN_LANGUAGES:
                    language = None
                if language is None:
                    choices = self.detector.compute_language_confidence_values(surface)
                    if not choices:
                        report(a, b, 'language_undetermined', 'the sung language could not be determined',
                               'add a language hint for this word')
                        return None
                    language = choices[0].language.iso_code_639_1.name.lower()
                    confidence = choices[0].value
                else:
                    fragment = (latin_line if latin else line)[segment.start_index:segment.end_index]
                    confidence = next((c.value for c in self.detector.compute_language_confidence_values(fragment)
                                       if c.language.iso_code_639_1.name.lower()==language),0.)
            enabled = ', '.join(self.languages)
            if language not in self.languages:
                # Lyric content in a script/language the configuration excludes: an input issue with a position.
                s, e = norm.source_span(a, b)
                report(a, b, 'language_not_enabled',
                       f'{text[s:e]!r} is written as {language}, which is not among the enabled languages ({enabled})',
                       f"enable '{language}', or add a language hint naming an enabled language if it is sung in one")
                return None
            if regex.search(r'\p{Han}',surface) and language not in {'ja','zh'}:
                allowed=[x for x in ('ja','zh') if x in self.languages]
                if not allowed:
                    report(a, b, 'language_not_enabled',
                           f'Han text needs Japanese or Chinese, which are not among the enabled languages ({enabled})',
                           "enable 'ja' or 'zh'")
                    return None
                choices=self.detector.compute_language_confidence_values(surface)
                choice=next((c for c in choices if c.language.iso_code_639_1.name.lower() in allowed),None)
                if choice is None:
                    report(a, b, 'language_undetermined', 'Han text could not be assigned to Japanese or Chinese',
                           "add a language hint such as 'ja: ...' or 'zh: ...'")
                    return None
                language=choice.language.iso_code_639_1.name.lower();confidence=choice.value
                source='script_filtered_text_detection'
            return language, source, confidence

        def japanese(a, b, source, confidence):
            seg_text = atext[a:b]
            # parse_lyrics returns positions in *its* source, i.e. seg_text = analysis indices local to this run.
            # This holds whether or not re-normalizing seg_text changes anything (a substring of NFC text is NFC,
            # and the analysis text has no halfwidth kana or selectors left), because the nested call maps its own
            # analysis back to seg_text.
            parsed = parse_lyrics(seg_text, reader=reader)
            for local, token in parsed.unread_tokens.items():
                for si in norm.source_indices([a+local]):
                    token_at[si] = token
            before = len(issues)
            for problem in parsed.override_problems:
                # A readings override that cannot be mapped safely is never replaced by the spelled reading.
                ls, le = parsed.analysis.source_span(problem['analysis_start'], problem['analysis_end'])
                report(a+ls, a+le, 'reading_override_unmappable', problem['message'], problem['suggestion'])
            specs = []
            for mora in parsed.moras:
                indices = [a+i for i in mora.char_idx]
                tokens = list(self.aligner.mora_tokens(mora, None))
                entry = dict(kana=mora.kana, kind=mora.kind, model_phones=tokens, analysis=indices,
                             split_estimated=mora.split_estimated, envelope=mora.envelope)
                if specs and mora.kind in SPECIAL_MORAS and not mora.run_start:
                    # Same contiguous Japanese run: attach regardless of morphological token boundaries
                    # and of readings-override chunk boundaries (OKー with 'OK おっけ' -> ケー).
                    # run_start marks a real break (punctuation/symbol inside an override span).
                    spec = specs[-1]
                    spec['model_phones'] += tokens
                    spec['reading'] += mora.kana
                    spec['analysis'].update(indices)
                    spec['moras'].append(entry)
                elif mora.kind == 'choon':
                    report(min(indices), max(indices)+1, 'no_preceding_sound',
                           'long-vowel mark has no preceding sound in the same word',
                           'write the vowel to be sung or attach the mark to the preceding word')
                elif not tokens:
                    # Never attach an unknown, phone-less reading like a long vowel.
                    report(min(indices), max(indices)+1, 'no_model_phones',
                           f'reading {mora.kana!r} cannot be represented completely by the acoustic model '
                           'inventory (no phone sequence covering the whole mora)',
                           "add a readings override line such as '<surface> <kana reading>'")
                else:
                    # A run-initial ッ/ン stays a standalone syllable (genuine interjection/nasal).
                    specs.append(dict(reading=mora.kana, analysis=set(indices), moras=[entry], model_phones=list(tokens),
                                      nucleus_token=next((i for i,t in enumerate(tokens) if t in 'aiueo'),0)))
            if not specs:
                if len(issues) == before and not regex.search(r'[\p{Han}\p{Hiragana}\p{Katakana}ー]', seg_text):
                    s, e = norm.source_span(a, b)
                    report(a, b, 'japanese_reader_no_syllables',
                           f'Japanese reader produced no syllables for {text[s:e]!r}; specify the sung language or spelling')
                return
            bounds = _token_boundaries(seg_text, reader)
            for spec in specs:
                lo, hi = min(spec['analysis'])-a, max(spec['analysis'])-a
                bounds = {x for x in bounds if not lo < x <= hi}
            edges = sorted(bounds)

            def finish(spec):
                # envelope: a whole-spelling readings override ('100 ひゃく'); its moras belong to every character
                # of that spelling, with no per-character reading or onset.
                moras = [dict(kana=m['kana'], kind=m['kind'], model_phones=m['model_phones'],
                              source_chars=norm.source_indices(m['analysis']), split_estimated=m['split_estimated'],
                              **({'envelope': True} if m['envelope'] else {}))
                         for m in spec['moras']]
                return dict(reading=spec['reading'], ipa=None, model_phones=spec['model_phones'],
                            source_chars=norm.source_indices(spec['analysis']), nucleus_token=spec['nucleus_token'],
                            pronunciation_source='japanese_reader', syllable_basis='japanese_moras_merged_specials',
                            moras=moras, split_estimated=any(m['split_estimated'] for m in moras))
            for wa, wb in zip(edges, edges[1:]):
                own = [spec for spec in specs if wa <= min(spec['analysis'])-a < wb]
                if own:
                    word_source, word_confidence = source, confidence
                    if source == 'readings_override_context' and all(i in override_key_at for i in range(a+wa, a+wb)):
                        # This word is exactly the explicit override; only its neighbours are inferred.
                        word_source, word_confidence = 'readings_override', 1.0
                    add_word(a+wa, a+wb, 'ja', word_source, word_confidence, [finish(spec) for spec in own],
                             {i for spec in own for i in spec['analysis']})

        def foreign(a, b, language, source, confidence):
            s, e = norm.source_span(a, b)
            try:
                pronunciation=self.g2p.pronounce(atext[a:b],language)
                parts=pronunciation.parts or syllabify(pronunciation.phones,language)
                mapper=getattr(self.aligner,'ipa_tokens',model_tokens)
                specs=[]
                for part in parts:
                    mapped=[mapper(p) for p in part]
                    ni=next((i for i,p in enumerate(part) if nucleus(p)),None)
                    if ni is None:
                        raise ValueError('Pronunciation has no syllabic nucleus: '+' '.join(part))
                    specs.append(dict(reading=''.join(part),ipa=part,model_phones=[t for group in mapped for t in group],
                                      nucleus_token=sum(len(g) for g in mapped[:ni]),
                                      source_chars=list(range(s,e)),pronunciation_source=pronunciation.source,
                                      pronunciation_variants=pronunciation.variants,
                                      syllable_basis='pinyin_syllables' if language=='zh' else 'ipa_nuclei_heuristic',
                                      phone_mapping='multilingual_ipa_inventory' if hasattr(self.aligner,'ipa_tokens') else 'approximate_japanese_inventory'))
            except ValueError as exc:
                # Recorded with its position and re-raised below together with every other unresolved span.
                report(a, b, 'pronunciation_failed', f'{language} pronunciation failed: {exc}',
                       f"check the sung spelling, or add a language hint such as 'en: {text[s:e]}' if {language} is not the sung language")
                return
            add_word(a,b,language,source,confidence,specs,range(a,b))

        def override_language(a, b):
            """Run language when a readings override (a Japanese kana reading) applies inside [a, b).

            An explicit language hint wins over the override for the characters it covers, so only
            unhinted override characters count. The override confirms Japanese only for its own span.
            Neighbouring characters merged into the same run (第 and 章 around '1') are confirmed only by
            Japanese spelling cues; otherwise Japanese is an inference from the reading context and is
            diagnosed (with the neighbours' own spans), never labelled as confirmed.
            Returns (language, source, confidence); None when every override character is covered by a
            language hint (the run is resolved normally); False after reporting a configuration contradiction
            (a Japanese readings override while 'ja' is not enabled: never silently enabled or discarded).
            """
            explicit = [i for i in range(a, b) if i in override_key_at and i not in forced]
            if not explicit:
                return None
            if 'ja' not in self.languages:
                s, e = norm.source_span(a, b)
                keys = ', '.join(sorted({repr(override_key_at[i]) for i in explicit}))
                report(a, b, 'language_not_enabled',
                       f'the readings override for {keys} gives a Japanese reading for {text[s:e]!r}, but Japanese '
                       f"is not among the enabled languages ({', '.join(self.languages)})",
                       "enable 'ja', remove that readings override line, or add a language hint naming an "
                       'enabled language for this word')
                return False
            rest = [i for i in range(a, b) if i not in override_key_at and i not in forced]
            if not rest:
                return 'ja', 'readings_override', 1.0
            if JAPANESE_CUE_RE.search(''.join(atext[i] for i in rest)):
                return 'ja', 'script', 1.0
            spans = []
            for i in rest:
                if spans and spans[-1][1] == i:
                    spans[-1][1] = i + 1
                else:
                    spans.append([i, i + 1])
            for x, y in spans:
                s, e = norm.source_span(x, y)
                where = make_issue(text, s, e, 'language_inferred_not_verified', '')
                diagnostics.append(dict(text=text[s:e], char_start=s, char_end=e, line=where['line'],
                                        column=where['column'], language='ja', confidence=None,
                                        basis='readings_override_context', status='language_inferred_not_verified',
                                        suggestion=f"read as Japanese because it is written together with a Japanese "
                                                   f"readings override; add a language hint (e.g. 'ja: {text[s:e]}') "
                                                   f"to confirm, or a readings override for the whole word"))
            return 'ja', 'readings_override_context', None

        cjk_keys = [k for k in reader.override if all(KANA_ANY.match(c) or KANJI.match(c) for c in k)]

        def inner_override_matches(a, b):
            """Kana/kanji override keys the Japanese reader would actually apply inside a run.

            Uses the reader's own tiling (lyrics._token_groups: whole morphological tokens, longest key
            first, left to right, non-overlapping), never the union of every possible match: with keys
            日本世界 and 世界人民 on 日本|世界|人民 only 日本世界 is applied, and 人民 stays a dictionary
            reading that is not explicit. Never part of a token ('本世' does not match inside 日本|世界).
            Applied keys entirely under a language hint are skipped.
            """
            surface = atext[a:b]
            if not any(k in surface and k != surface for k in cjk_keys) or _NON_CJK_RE.match(surface[0]):
                return []
            found, pos = [], 0
            for surf, _, ov in _token_groups(surface, reader):
                if (ov is not None and surf in cjk_keys
                        and any(a+i not in forced for i in range(pos, pos+len(surf)))):
                    found.append((a+pos, a+pos+len(surf), surf))
                pos += len(surf)
            return found

        def inner_override_conflict(a, b, inner, language):
            """A matching override inside a run whose language is not Japanese: positioned, never silent."""
            s, e = norm.source_span(a, b)
            where = '; '.join(f"{k!r} at line {line_column(text, norm.source_span(x, y)[0])[0]} "
                              f"col {line_column(text, norm.source_span(x, y)[0])[1]}" for x, y, k in inner)
            if 'ja' not in self.languages:
                report(a, b, 'language_not_enabled',
                       f'the readings override ({where}) gives a Japanese reading inside {text[s:e]!r}, but Japanese '
                       f"is not among the enabled languages ({', '.join(self.languages)})",
                       f"enable 'ja', remove that readings override line, or add a language hint for the whole word "
                       f"(e.g. '{language}: {text[s:e]}')")
            else:
                report(a, b, 'reading_override_language_ambiguous',
                       f'the readings override ({where}) matches a whole token inside {text[s:e]!r}, but the language '
                       f'of {text[s:e]!r} was inferred as {language}; the supplied Japanese reading is neither applied '
                       f'nor silently replaced by a {language} pronunciation',
                       f"add a language hint for the whole word: 'ja: {text[s:e]}' to use the readings override, "
                       f"or '{language}: {text[s:e]}' to sing it as {language}")

        def process_run(a, b, has_override, line, offset, detected, latin_line, latin_detected):
            surface = atext[a:b]
            run_language = None
            if any(i not in forced for i in range(a, b)):
                run_language = override_language(a, b) if has_override else None
                if run_language is False:
                    return
                inner = [] if has_override else inner_override_matches(a, b)
                if run_language is None and inner:
                    unhinted = {i for i in range(a, b) if i not in forced}
                    if unhinted <= {i for x, y, _ in inner for i in range(x, y)}:
                        # The run is wholly spelled by consecutive override keys (日本|世界): explicit Japanese.
                        override_key_at.update((i, k) for x, y, k in inner for i in range(x, y))
                        run_language = override_language(a, b)
                        if run_language is False:
                            return
                if run_language is None:
                    run_language = resolve(surface, a, b, line, offset, detected, latin_line, latin_detected)
                    if run_language is None:
                        return
                    if inner and run_language[0] != 'ja':
                        inner_override_conflict(a, b, inner, run_language[0])
                        return
            segments = []
            cuts = [a]+[i for i in range(a+1,b) if forced.get(i)!=forced.get(i-1)]+[b]
            for start, end in zip(cuts, cuts[1:]):
                segment = [start, end, forced[start], 'explicit_hint', 1.0] if start in forced else [start, end, *run_language]
                if segments and segments[-1][2] == segment[2]:
                    # A hint that agrees with the surrounding language confirms it; it never re-cuts the word.
                    segments[-1][1] = end
                    if segment[3] == 'explicit_hint':
                        segments[-1][3:5] = ['explicit_hint', 1.0]
                else:
                    segments.append(segment)
            if len(segments) > 1:
                bounds = _token_boundaries(surface, reader) if not _NON_CJK_RE.match(surface[0]) else {0, len(surface)}
                inside = [seg[0] for seg in segments[1:] if seg[0]-a not in bounds]
                if inside:
                    s, e = norm.source_span(a, b)
                    report(a, b, 'hint_splits_word',
                           f'a language hint boundary splits the word {text[s:e]!r}; hints must cover whole words',
                           'extend the hint to the whole word, or remove it')
                    return
            for start, end, language, source, confidence in segments:
                if source in INFERRED_SOURCES:
                    s, e = norm.source_span(start, end)
                    where = make_issue(text, s, e, 'language_inferred_not_verified', '')
                    diagnostics.append(dict(text=text[s:e], char_start=s, char_end=e, line=where['line'],
                                            column=where['column'], language=language, confidence=confidence,
                                            basis=source, status='language_inferred_not_verified',
                                            suggestion=f"add a language hint (e.g. '{language}: {text[s:e]}') to confirm, "
                                                       f"or give the sung language if it is not {language}"))
                if language == 'ja':
                    japanese(start, end, source, confidence)
                else:
                    foreign(start, end, language, source, confidence)

        override = override_spans(masked, reader.override, excluded)
        override_key_at = {i: k for s, e, k in override for i in range(s, e)}
        # Same line delimiters as diagnostics/results (lyric_input.line_spans); the analysis text keeps them as-is.
        for offset, _, end_of_line in line_spans(masked):
            line = masked[offset:end_of_line]
            if not line:
                continue
            detected = self.detector.detect_multiple_languages_of(line) if line.strip() else []
            latin_line=''.join(c if regex.match(r"[\p{Latin}\s'’.,!?]",c) else ' ' for c in line)
            latin_detected=self.detector.detect_multiple_languages_of(latin_line) if latin_line.strip() else []
            runs = [[offset+m.start(), offset+m.end(), False] for m in WORD_RE.finditer(line)]
            for run in runs:
                # A whole kana/kanji word spelled exactly like a readings override key is an explicit
                # Japanese reading too (the reader applies it at token boundaries): not left to detection.
                if atext[run[0]:run[1]] in reader.override:
                    run[2] = True
                    override_key_at.update((i, atext[run[0]:run[1]]) for i in range(run[0], run[1]))
            runs += [[s, e, True] for s, e, _ in override if offset <= s and e <= end_of_line]
            for a, b, has_override in _merge_override_runs(runs, atext):
                process_run(a, b, has_override, line, offset, detected, latin_line, latin_detected)
        # A source point is represented only if every analysis char made from it was read
        # (reading あ does not cover an uncomposable U+3099 that follows it).
        covered = norm.covered_source(represented)
        issues += coverage_issues(text, covered, norm.source_indices(excluded), 'multilingual', token_at,
                                  norm.source_indices(reported))
        if issues:
            raise LyricsInputError(issues)
        if not syllables:
            raise ValueError('No pronounceable syllables')
        return dict(text=text,words=words,syllables=syllables,language_diagnostics=diagnostics,
                    normalization=norm.changes(),source_offset_unit='unicode_code_point')
