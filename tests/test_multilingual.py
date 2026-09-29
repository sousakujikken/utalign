"""Contracts for multilingual syllables, joint note allocation, and constrained CTC."""
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from utalign.multilingual.frontend import Frontend, model_tokens, syllabify
from utalign.multilingual.joint_alignment import assign_notes, constrained_path, align_syllables


def log_emissions(frames, size, events):
    probabilities=np.full((frames,size),.001,dtype=np.float32)
    probabilities[:,0]=.99
    for a,b,label in events:
        probabilities[a:b,:]=.001;probabilities[a:b,label]=.99
    probabilities/=probabilities.sum(axis=1,keepdims=True)
    return np.log(probabilities)


class FakeAligner:
    from utalign.aligners.hf_ctc import HFCTCAligner
    mora_tokens=HFCTCAligner.mora_tokens
    vocab={p:i for i,p in enumerate('_ a i u e o N cl b by ch d dy f g gy h hy j k ky m my n ny p py r ry s sh t ts v w y z I U'.split())}
    blank=0
    is_phoneme=True
    model_id='test-phoneme-model'
    vocab_kind='phoneme'
    device='cpu'
    device_fallback=None


class FrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=FakeAligner()
        cls.front=Frontend(cls.model)

    def test_all_eight_languages_have_phonetic_syllables(self):
        cases=[('ja','こんにちは世界',7),('en','Shining artists',4),('de','Woher das Magma',5),
               ('fr','Bonjour liberté',5),('zh','你好世界',4),('ko','안녕하세요 사랑',7),
               ('es','Hola corazón',5),('it','Ciao amore',4)]
        for lang,text,count in cases:
            with self.subTest(language=lang):
                d=self.front.parse(text,[dict(text=text,language=lang)])
                self.assertEqual(len(d['syllables']),count)
                self.assertTrue(all(w['language']==lang for w in d['words']))
                self.assertTrue(all(w['syllable_ids'] for w in d['words']))
                self.assertTrue(all(s['model_phones'] for s in d['syllables']))
                for w in d['words']:
                    self.assertEqual(text[w['char_start']:w['char_end']],w['text'])

    def test_mixed_line_emoji_repeats_and_language_hints_preserve_positions(self):
        text='🌋こんにちは Bonjour你好 사랑 hola amore Woher hello hello'
        pieces=[('こんにちは','ja'),('Bonjour','fr'),('你好','zh'),('사랑','ko'),('hola','es'),('amore','it'),('Woher','de'),('hello','en')]
        d=self.front.parse(text,[dict(text=t,language=l) for t,l in pieces])
        self.assertEqual({w['language'] for w in d['words']},set(('ja','fr','zh','ko','es','it','de','en')))
        hello=[w for w in d['words'] if w['text']=='hello']
        self.assertEqual(len(hello),2)
        self.assertNotEqual(hello[0]['char_start'],hello[1]['char_start'])
        self.assertEqual(d['words'][0]['char_start'],1)

    def test_source_ipa_is_not_replaced_by_japanese_approximation(self):
        d=self.front.parse('liberté',[dict(text='liberté',language='fr')])
        self.assertEqual(len(d['words'][0]['syllable_ids']),3)
        self.assertTrue(any(set(s['reading']) & {'ʁ','ʀ'} for s in d['syllables']))
        self.assertTrue(all(not set(s['model_phones']) & {'ʁ','ʀ'} for s in d['syllables']))
        with self.assertRaisesRegex(ValueError,'Unmapped IPA'):
            model_tokens('☃')

    def test_special_japanese_moras_are_not_spurious_syllables(self):
        d=self.front.parse('ラン ボヘー',[dict(text='ラン ボヘー',language='ja')])
        self.assertEqual([s['reading'] for s in d['syllables']],['ラン','ボ','ヘー'])
        self.assertEqual(d['syllables'][0]['model_phones'],['r','a','N'])

    def test_phonetic_diphthongs_and_korean_glides(self):
        self.assertEqual(len(syllabify(['tʃ','ao̯'],'it')),1)
        self.assertEqual(syllabify(['a','n','n','j','ʌ','ŋ'],'ko'),[['a','n'],['n','j','ʌ','ŋ']])

    def test_script_filter_never_sends_han_to_european_voice(self):
        d=self.front.parse('私 volcano')
        self.assertIn(d['words'][0]['language'],['ja','zh'])

    def test_annotation_and_hint_validation(self):
        d=self.front.parse('[Chorus]\nhello\n© AUTHOR')
        self.assertEqual([w['text'] for w in d['words']],['hello'])
        with self.assertRaisesRegex(ValueError,'does not match'):
            self.front.parse('hello',[dict(text='missing',language='en')])
        with self.assertRaisesRegex(ValueError,'disagree'):
            self.front.parse('hello',[dict(text='hello',language='en'),dict(text='hello',language='de')])

    def test_kana_proxy_does_not_resegment_source_syllables(self):
        from utalign.multilingual.proxy_readings import apply_proxy_readings
        d=self.front.parse('knock out knock out',[dict(text='knock out',language='en')])
        apply_proxy_readings(d,self.model,[dict(text='knock out',syllables=[['ノッ'],['キャウ']])])
        self.assertEqual(len(d['syllables']),4)
        self.assertEqual([s['proxy_reading'] for s in d['syllables']],['ノッ','キャウ']*2)
        self.assertEqual(d['syllables'][1]['model_phones'],['ky','a','u'])
        self.assertEqual(d['words'][1]['text'],'out')
        self.assertEqual(d['words'][1]['syllable_ids'],[1])
        self.assertEqual(d['syllables'][1]['proxy_source'],'explicit_sung_approximation')
        self.assertIsNotNone(d['syllables'][1]['ipa'])

    def test_proxy_hint_cannot_remove_or_invent_source_syllables(self):
        from utalign.multilingual.proxy_readings import apply_proxy_readings
        for readings in [[['ノッ']], [['ノッ'],['キャ','ウ']]]:
            d=self.front.parse('knock out',[dict(text='knock out',language='en')])
            with self.assertRaisesRegex(ValueError,'source words|original syllable'):
                apply_proxy_readings(d,self.model,[dict(text='knock out',syllables=readings)])

    def test_unreadable_japanese_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError,'no syllables'):
            self.front.parse('SNS',[dict(text='SNS',language='ja')])


class JointTests(unittest.TestCase):
    def test_constrained_ctc_ignores_stronger_outside_peak(self):
        em=log_emissions(60,3,[(4,7,1),(20,24,1),(35,39,2)])
        em[20:24,1]-=.5
        targets=np.array([1,2],np.int32)
        path,score=constrained_path(em,targets,0,np.array([15,30]),np.array([30,50]))
        self.assertGreater(score,-1e19)
        self.assertGreaterEqual(np.flatnonzero(path==1)[0],15)
        self.assertGreaterEqual(np.flatnonzero(path==3)[0],30)

    def test_impossible_midi_window_is_not_reported_as_success(self):
        em=log_emissions(10,2,[(2,5,1)])
        _,score=constrained_path(em,np.array([1,1],np.int32),0,np.array([2,2]),np.array([3,3]))
        self.assertLess(score,-1e19)

    def test_melisma_consumes_multiple_notes_without_deleting_a_syllable(self):
        em=log_emissions(100,3,[(11,17,1),(50,56,2)])
        first,last,_,_,cost=assign_notes(em,np.array([1,2],np.int32),np.array([0,1,2]),np.array([0,1]),
            np.array([.2,.6,1.]),np.array([.5,.9,1.4]),np.array([.2,1.]),np.ones(3,bool))
        self.assertLess(cost,1e19)
        self.assertEqual(list(first),[0,2]);self.assertEqual(list(last),[1,2])

    def test_two_syllables_can_share_one_note(self):
        em=log_emissions(70,4,[(10,13,1),(22,26,2),(44,49,3)])
        first,last,indices,counts,cost=assign_notes(em,np.array([1,2,3],np.int32),np.array([0,1,2,3]),np.array([0,1,2]),
            np.array([.2,.85]),np.array([.65,1.2]),np.array([.2,.44,.88]),np.ones(2,bool))
        self.assertLess(cost,1e19)
        self.assertEqual(list(first),[0,0,1]);self.assertEqual(list(counts),[2,2,1])
        self.assertEqual(list(indices),[0,1,0])

    def test_full_joint_pass_exports_phone_nuclei_and_note_ids(self):
        em=log_emissions(70,3,[(10,16,1),(40,48,2)])
        document=dict(syllables=[dict(id=0,word_id=0,model_phones=['a'],nucleus_token=0),
                                 dict(id=1,word_id=0,model_phones=['i'],nucleus_token=0)])
        notes=[SimpleNamespace(start=.2,end=.6),SimpleNamespace(start=.8,end=1.2)]
        out,summary=align_syllables(em,document,notes,0.,SimpleNamespace(vocab={'_':0,'a':1,'i':2},blank=0))
        self.assertEqual(summary['omitted_syllables'],0)
        self.assertEqual([s['note_ids'] for s in out],[[0],[1]])
        self.assertTrue(all(s['phone_spans'] and s['nucleus_start']==s['start'] for s in out))
        self.assertTrue(all(s['timing_basis']=='midi_window_constrained_ctc' for s in out))

    def test_all_eight_languages_pass_the_joint_graph_with_known_phone_events(self):
        # Artificial emissions test the alignment graph, not real singing accuracy.
        model=FakeAligner()
        pieces=[('夢','ja'),('hello','en'),('hallo','de'),('bonjour','fr'),('你好','zh'),('사랑','ko'),('hola','es'),('amore','it')]
        text=' '.join(t for t,_ in pieces)
        doc=Frontend(model).parse(text,[dict(text=t,language=lang) for t,lang in pieces])
        events=[];notes=[]
        for i,s in enumerate(doc['syllables']):
            frame=10+i*30
            notes.append(SimpleNamespace(start=frame*.02,end=(frame+24)*.02))
            for q,phone in enumerate(s['model_phones']):
                events.append((frame+2*q,frame+2*q+1,model.vocab[phone]))
        # Real model's blank ID may be nonzero; remap the artificial generator's blank.
        em=log_emissions(30*len(notes)+30,len(model.vocab),events)
        self.assertEqual(model.blank,0)
        out,summary=align_syllables(em,doc,notes,0.,model)
        self.assertEqual(len(out),len(doc['syllables']))
        self.assertEqual(summary['omitted_syllables'],0)
        self.assertTrue(all(s['note_ids']==[i] for i,s in enumerate(out)))
        self.assertTrue(all(abs(s['start']-notes[i].start)<=.021 for i,s in enumerate(out)))


if __name__=='__main__':unittest.main()
