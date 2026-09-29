import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from utalign import config
from utalign.pipeline import AlignOptions, run_align, run_export
from utalign.multilingual.pipeline import use_multilingual, build_result
from utalign.multilingual.frontend import Frontend
from utalign.multilingual.proxy_readings import apply_proxy_readings
from utalign.multilingual.hints import language_hints, proxy_readings
from utalign.midi import Note
from utalign.project import Project, create_project, run_project_align
from utalign.utavista import result_to_lyrics_timing, self_check
from test_multilingual import FakeAligner, log_emissions


@pytest.mark.parametrize('text,expected', [('こんにちは世界',False),('[Chorus]\nこんにちは\n© hello',False),
                         ('こんにちは hello',True),('사랑',True),('你好世界',True)])
def test_auto_route(text,expected):
    assert use_multilingual(text,AlignOptions()) is expected


def test_invalid_options_do_not_fall_back_or_ignore_hints():
    for options in [AlignOptions(alignment_mode='typo'),AlignOptions(languages=('xx',)),
                    AlignOptions(midi_offset=float('nan')),AlignOptions(search_band=-1),
                    AlignOptions(alignment_mode='japanese',proxy_readings=[{}])]:
        with pytest.raises(ValueError):use_multilingual('hello',options)


def test_project_hint_round_trip_and_option_forwarding(tmp_path):
    pr=create_project('mixed',config.DEFAULTS,pdir=tmp_path)
    audio=tmp_path/'vocal.wav';audio.touch();midi=tmp_path/'vocal.mid';midi.touch();lyrics=tmp_path/'lyrics.txt';lyrics.write_text('knock out')
    pr.apply_patch(dict(inputs=dict(audio=str(audio),midi=str(midi),lyrics=str(lyrics)),
                        options=dict(languages='ja,en',alignment_mode='multilingual',midi_offset='0.12'),
                        language_hints='en: knock out',proxy_readings='knock out = ノッ / キャウ'))
    restored=Project(pr.path).summary()
    assert restored['proxy_readings']=='knock out = ノッ / キャウ'
    with patch('utalign.pipeline.run_align',return_value=dict(timing_unit='syllable',n_syllables=2,low_support=0)) as run, patch('utalign.config.models_dir',return_value=tmp_path):
        run_project_align(pr,config.DEFAULTS,lambda _:None)
        opt=run.call_args.args[5]
        assert opt.languages==('ja','en') and opt.midi_offset==.12
        assert opt.proxy_readings==[dict(text='knock out',syllables=[['ノッ'],['キャウ']])]
        assert opt.language_hints==[dict(text='knock out',language='en')]
    assert Project(pr.path).summary()['last_run']['n_syllables']==2


def test_plain_hints_validate_and_retain_word_syllable_structure():
    assert proxy_readings('Everything = エ|ブリ|シン')[0]['syllables']==[['エ','ブリ','シン']]
    assert language_hints('fr: Bonjour')[0]['language']=='fr'
    with pytest.raises(ValueError):language_hints('Bonjour')
    with pytest.raises(ValueError):proxy_readings('knock out')


def test_unrecognized_letters_and_numbers_cannot_be_silently_omitted():
    frontend=Frontend(FakeAligner(),languages=('ja','en'))
    for text in ['hello 123','hello Привет']:
        with pytest.raises(ValueError,match='Unsupported lyric characters'):frontend.parse(text)


def test_decomposed_french_spelling_keeps_source_offsets():
    text='cafe\u0301'
    result=Frontend(FakeAligner(),languages=('fr',)).parse(text)
    assert result['words'][0]['text']==text and result['words'][0]['char_end']==5
    assert len(result['syllables'])==2


def test_complete_pipeline_export_preserves_syllables_and_does_not_fabricate_letter_onsets(tmp_path):
    model=FakeAligner()
    lyrics=tmp_path/'lyrics.txt';lyrics.write_text('🌋hello 世界')
    front=Frontend(model,languages=('ja','en'))
    doc=front.parse(lyrics.read_text(),[dict(text='hello',language='en'),dict(text='世界',language='ja')])
    notes=[];events=[]
    for i,s in enumerate(doc['syllables']):
        frame=10+i*30
        notes.append(Note(i,frame*.02,(frame+24)*.02,60,0,0))
        events.extend((frame+2*q,frame+2*q+1,model.vocab[p]) for q,p in enumerate(s['model_phones']))
    em=log_emissions(30*len(notes)+30,len(model.vocab),events)
    model.compute_emissions=lambda wav,cache:em
    model.log=lambda _:None
    audio=tmp_path/'audio.wav';audio.write_bytes(b'input');midi=tmp_path/'input.mid';midi.write_bytes(b'input')
    # Match the acoustic model's convolution frame count, without loading weights.
    wav=np.ones((len(em)-1)*320+400,dtype=np.float32)
    out=tmp_path/'out';stages=[]
    opt=AlignOptions(write_playback=False,languages=('ja','en'),midi_offset=0.,language_hints=[dict(text='hello',language='en'),dict(text='世界',language='ja')])
    with patch('utalign.aligners.make_aligner',return_value=model),patch('utalign.audio.load_audio_16k',return_value=wav),patch('utalign.midi.load_notes',return_value=notes):
        meta=run_align(audio,lyrics,midi,out,tmp_path/'work',opt,lambda _:None,stages.append)
    result=json.loads((out/'result.json').read_text())
    assert result['timing_unit']=='syllable' and meta['n_syllables']==5
    assert result['summary']['omitted_syllables']==0
    assert result['words'][0]['char_start']==1
    assert not (out/'audio.wav').exists()
    exported=out/'export.json';info=run_export(out/'result.json',exported,ruby=True,log=lambda _:None)
    assert info['issues']==[]
    export=json.loads(exported.read_text())
    hello=export['phrases'][0]['words'][0]
    assert hello['text']=='🌋hello'
    letters=hello['chars'][1:]
    assert len({(c['startMs'],c['endMs']) for c in letters})==1
    assert all('ruby' not in c and c['timingBasis']=='word_envelope' for c in letters)
    assert len(hello['syllables'])==2
    assert hello['syllables'][0]['startMs']<hello['syllables'][1]['startMs']
    assert [e['stage'] for e in stages]==['model','lyrics','midi','audio','emission','align','match','write']


def test_failed_new_alignment_does_not_replace_previous_result(tmp_path):
    out=tmp_path/'out';out.mkdir();(out/'result.json').write_text('previous')
    lyrics=tmp_path/'lyrics.txt';lyrics.write_text('hello')
    with patch('utalign.aligners.make_aligner',return_value=FakeAligner()):
        with pytest.raises(ValueError,match='star'):
            run_align(tmp_path/'audio',lyrics,tmp_path/'midi',out,tmp_path/'work',AlignOptions(star=True))
    assert (out/'result.json').read_text()=='previous'
