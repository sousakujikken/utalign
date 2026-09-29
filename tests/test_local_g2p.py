"""Independent pronunciation regressions for the replacement backends."""
import builtins
import socket
import subprocess

import pytest

from utalign.multilingual.frontend import Frontend, model_tokens, nucleus, syllabify
from utalign.multilingual.g2p import LocalG2P, ipa_segments
from test_multilingual import FakeAligner


@pytest.fixture(scope='module')
def g2p():
    return LocalG2P()


@pytest.mark.parametrize('lang,text,count', [
    ('en','SNS',3), ('en','NASA',2), ('en','activationist',5),
    ('en',"glorpish's",3), ('de','Woher',2), ('de','Freund',1),
    ('de','Feuer',2), ('de','gehen',2), ('fr','beaucoup',2),
    ('fr',"l’amour",2), ('fr','oiseau',2), ('fr','fille',1),
    ('es','canción',2), ('es','aire',2), ('es','país',2),
    ('es','hoy',1), ('es','muy',1), ('es','ciudad',2),
    ('it','più',1), ('it','ciao',1), ('it','Paolo',3),
    ('it','paese',3), ('it','mio',2), ('it','mai',1),
    ('zh','你好世界',4), ('zh','中国音乐',4), ('zh','女儿',2),
    ('ko','안녕하세요',5), ('ko','국물',2), ('ko','같이',2),
])
def test_known_syllable_counts_and_complete_phone_mapping(g2p, lang, text, count):
    pronunciation = g2p.pronounce(text, lang)
    parts = pronunciation.parts or syllabify(pronunciation.phones, lang)
    assert len(parts) == count
    for part in parts:
        assert any(nucleus(phone) for phone in part)
        assert all(model_tokens(phone) for phone in part)


def test_pronunciation_details_not_just_nonempty_output(g2p):
    assert g2p.pronounce('SNS','en').source == 'english_initials'
    assert g2p.pronounce('activationist','en').source == 'g2pe_neural'
    assert 'h' in g2p.pronounce('Woher','de').phones
    assert 'ŋ' in g2p.pronounce('국물','ko').phones  # nasal assimilation
    assert 't͡ɕʰ' in g2p.pronounce('같이','ko').phones  # palatalization
    assert g2p.pronounce('重庆','zh').parts[0][0] == 'ʈ͡ʂʰ'  # chóng, not zhòng


def test_ipa_boundaries_do_not_split_affricates_or_diphthongs():
    assert ipa_segments('t͡ʃaɪ̯') == ['t͡ʃ','aɪ̯']
    assert ipa_segments('i̯ɛ') == ['i̯','ɛ']
    assert not nucleus('i̯')
    assert nucleus('aɪ̯') and nucleus('n̩')
    assert model_tokens('t͡ʃ') == ['ch']
    assert model_tokens('ɔ̃') == ['o','N']


def test_no_network_subprocess_or_espeak_is_needed(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if 'espeak' in name:
            raise AssertionError('eSpeak import is forbidden')
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('External process/network is forbidden')

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    pieces = [('夢','ja'),('hello','en'),('Woher','de'),('bonjour','fr'),
              ('你好','zh'),('사랑','ko'),('hola','es'),('ciao','it')]
    document = Frontend(FakeAligner()).parse(' '.join(t for t,_ in pieces),
                  [dict(text=t,language=lang) for t,lang in pieces])
    assert len(document['words']) == 8
    assert all(s['pronunciation_source'] != 'espeak_ng' for s in document['syllables'])


def test_japanese_reader_does_not_initialize_foreign_resources(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Japanese must not call a foreign G2P')
    monkeypatch.setattr(LocalG2P, 'pronounce', forbidden)
    assert Frontend(FakeAligner(),languages=('ja',)).parse('こんにちは')['syllables']


def test_unknown_spelling_fails_instead_of_silently_dropping_letters(g2p):
    with pytest.raises(ValueError, match='Unsupported spelling'):
        g2p.pronounce('helloЖ','de')
    with pytest.raises(ValueError, match='No Mandarin reading'):
        g2p.pronounce('☃','zh')
