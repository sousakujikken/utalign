"""Production entry point and compatibility view for multilingual syllables."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import regex

from .frontend import LANGUAGES


def use_multilingual(text, opt):
    if opt.alignment_mode not in {'auto', 'japanese', 'multilingual'}:
        raise ValueError('解析方式は auto / japanese / multilingual を指定してください')
    if not opt.languages or set(opt.languages)-set(LANGUAGES):
        raise ValueError('対応言語: '+','.join(LANGUAGES))
    if not np.isfinite(opt.search_band) or opt.search_band<=0:
        raise ValueError('探索範囲は正の有限値にしてください')
    if opt.midi_offset is not None and not np.isfinite(opt.midi_offset):
        raise ValueError('MIDIオフセットは有限値にしてください')
    if opt.alignment_mode=='japanese':
        if opt.language_hints or opt.proxy_readings:
            raise ValueError('言語ヒント・音節の近似読みには多言語音節モードを選んでください')
        return False
    if opt.alignment_mode=='multilingual' or opt.language_hints or opt.proxy_readings or 'ja' not in opt.languages:
        return True
    from ..lyrics import BRACKET_RE
    content=BRACKET_RE.sub(' ',text)
    content='\n'.join(line for line in content.splitlines() if not line.lstrip().startswith('©'))
    if regex.search(r'[\p{Latin}\p{Hangul}]',content):
        return True
    # Chinese lyrics using only Han characters must not disappear into a Japanese reading.
    if 'zh' in opt.languages and regex.search(r'\p{Han}',content) and not regex.search(r'[\p{Hiragana}\p{Katakana}]',content):
        from lingua import Language, LanguageDetectorBuilder
        detector=LanguageDetectorBuilder.from_languages(Language.JAPANESE,Language.CHINESE).build()
        return detector.detect_language_of(content)==Language.CHINESE
    return False


def build_result(document, syllables, notes, offset, summary, meta, duration):
    """Canonical syllables plus legacy chars/moras used by lyrics-timing/1.0.

    A foreign spelling's characters share its word envelope. They are never
    presented as independently measured phoneme or syllable boundaries.
    """
    words=[dict(w) for w in document['words']]
    for w in words:
        ss=[syllables[i] for i in w['syllable_ids']]
        w.update(start=ss[0]['start'],end=ss[-1]['end'])
    char_ids={};note_ids={}
    moras=[]
    for s in syllables:
        for ci in s['source_chars']:char_ids.setdefault(ci,[]).append(s['id'])
        for ni in s['note_ids']:note_ids.setdefault(ni,[]).append(s['id'])
        moras.append(dict(s,id=s['id'],kana=s['proxy_reading'],romaji=' '.join(s['model_phones']),
                          kind='syllable',text=words[s['word_id']]['text'],char_idx=s['source_chars'],sung=True,
                          source=s['timing_basis'],end_kind=s['end_basis'],ctc_start=s['start'],
                          phrase=notes[s['note_ids'][0]].phrase,split_estimated=False,seg=s['word_id']))
    chars=[];line=0
    for i,ch in enumerate(document['text']):
        ids=char_ids.get(i,[]);ss=[syllables[k] for k in ids]
        language=ss[0]['language'] if ss else None
        chars.append(dict(index=i,char=ch,line=line,mora_ids=ids,syllable_ids=ids,pronounceable=bool(ids),sung=bool(ids),
                          start=min(s['start'] for s in ss) if ss else None,end=max(s['end'] for s in ss) if ss else None,
                          language=language,split_estimated=False,
                          timing_basis='word_envelope' if ss and language!='ja' else 'source_syllable_envelope'))
        if ch=='\n':line+=1
    lines=[];pos=0
    for li,text in enumerate(document['text'].split('\n')):
        ss=[s for s in syllables if s['line']==li]
        lines.append(dict(line=li,text=text,char_start=pos,char_end=pos+len(text),
                          start=ss[0]['start'] if ss else None,end=ss[-1]['end'] if ss else None,
                          n_syllables=len(ss),n_moras=len(ss),n_sung=len(ss)))
        pos+=len(text)+1
    ns=[dict(id=n.id,start=n.start+offset,end=n.end+offset,pitch=n.pitch,track=n.track,phrase=n.phrase,
             midi_start=n.start,syllable_ids=note_ids.get(i,[]),mora_ids=note_ids.get(i,[])) for i,n in enumerate(notes)]
    return dict(document,schema='utalign-result/2',timing_unit='syllable',words=words,syllables=syllables,
                meta=meta,summary=summary,duration=duration,audio='audio.wav',midi_offset=offset,
                chars=chars,moras=moras,notes=ns,lines=lines,phrases=[],
                limitations=['Language selection and syllabification are pronunciation hypotheses.',
                             'Proxy sounds need not exactly reproduce the original language.',
                             'Complete MIDI assignment does not guarantee acoustic accuracy.',
                             'Display ends include MIDI sustain; foreign spelling characters use word envelopes.'])


def run_multilingual(audio, lyrics, midi, out, work, opt, log, progress):
    from importlib.metadata import version
    from .. import __version__
    from ..aligners import make_aligner
    from ..audio import load_audio_16k
    from ..midi import load_notes
    from ..timing import xcorr_offset
    from ..output import decode_audio, write_peaks, write_click
    from ..pipeline import WEB_DIR, VIEWER_FILES
    from .frontend import Frontend
    from .proxy_readings import apply_proxy_readings
    from .joint_alignment import align_syllables

    if opt.star:
        raise ValueError('多言語音節モードでは star をオフにしてください')
    started=time.monotonic()
    def stage(name):progress(dict(event='stage',stage=name,elapsed=round(time.monotonic()-started,2)))
    stage('model')
    model=make_aligner(opt.model,models_dir=opt.models_dir,device=opt.device,star=False)
    model.log=log
    if not getattr(model,'is_phoneme',False):
        raise ValueError('多言語音節には日本語音素モデル (推奨: japanese-hubert-base-phoneme-ctc-v4) を選んでください')
    stage('lyrics')
    document=Frontend(model,opt.languages,readings=opt.readings).parse(Path(lyrics).read_bytes().decode('utf-8-sig'),opt.language_hints)
    apply_proxy_readings(document,model,opt.proxy_readings)
    log(f"[lyrics] {len(document['words'])} words, {len(document['syllables'])} multilingual syllables")
    stage('midi')
    notes=load_notes(Path(midi),min_rest=opt.min_rest,tracks=opt.midi_tracks,exclude_drums=not opt.keep_drums,log=log)
    stage('audio')
    wav=load_audio_16k(Path(audio))
    if len(wav)<16000 or not np.isfinite(wav).all() or not np.any(wav):
        raise ValueError('1秒以上の有効なボーカル音声が必要です')
    offset=opt.midi_offset if opt.midi_offset is not None else xcorr_offset(wav,notes)
    if all(n.end+offset<=0 or n.start+offset>=len(wav)/16000 for n in notes):
        raise ValueError('MIDIと音声が重なりません。MIDIオフセットを確認してください')
    stage('emission')
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    em=model.compute_emissions(wav,work)
    from ..aligners.base import num_frames
    if em.shape!=(num_frames(len(wav)),len(model.vocab)) or not np.isfinite(em).all():
        raise ValueError('音声のフレーム数とCTCの出力が一致しません')
    stage('align');stage('match')
    syllables,summary=align_syllables(em,document,notes,offset,model,wav,band=opt.search_band)
    warnings=[]
    if summary['low_support']:
        warnings.append(f"{summary['low_support']}音節は音声との一致が弱いため確認してください")
    if model.device_fallback:warnings.append('GPU推論に失敗したためCPUで解析しました: '+model.device_fallback)
    meta=dict(utalign_version=__version__,
              pronunciation_backends={name:version(name) for name in ('epitran','pypinyin')},
              pronunciation_resources='utalign.multilingual/resources/manifest.json',
              alignment_mode='multilingual',timing_unit='syllable',model=model.model_id,vocab=model.vocab_kind,
              device=model.device,device_fallback=model.device_fallback,offset=float(offset),
              offset_source='explicit' if opt.midi_offset is not None else 'onset_correlation',
              generated=time.strftime('%Y-%m-%dT%H:%M:%S'),n_syllables=len(syllables),low_support=summary['low_support'],
              n_mora_skipped=0,n_note_skipped=len(summary['unused_notes']),warnings=warnings,languages=list(opt.languages),
              audio=str(audio),lyrics=str(lyrics),midi=str(midi),
              inputs_sha256={str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [audio,lyrics,midi]},
              options=dict(language_hints=opt.language_hints,proxy_readings=opt.proxy_readings,search_band=opt.search_band))
    result=build_result(document,syllables,notes,offset,summary,meta,len(wav)/16000)
    stage('write')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if opt.write_playback:
        write_peaks(wav,out/'peaks.json');decode_audio(Path(audio),out/'audio.wav')
        write_click(wav,[s['start'] for s in syllables],out/'click.wav')
        for name in VIEWER_FILES:
            shutil.copy(WEB_DIR/name,out/('index.html' if name=='viewer.html' else name))
    meta['elapsed']=time.monotonic()-started
    payload=json.dumps(result,ensure_ascii=False,indent=1,allow_nan=False)
    # Replace the authoritative result only after all computations/serialization succeed.
    temporary=out/'result.json.tmp';temporary.write_text(payload,encoding='utf-8');temporary.replace(out/'result.json')
    log(f"[done] {len(syllables)} syllables, {summary['low_support']} low support; {out/'result.json'}")
    return meta
