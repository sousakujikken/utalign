"""Offline CMUdict + NumPy g2pE (Apache-2.0) English pronunciations.

GRU inference adapted from Kyubyong Park / Jongseok Kim. Resource revisions,
hashes and original license texts are preserved in resources/. No downloads.
"""
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

ARPABET = dict(zip(
    "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH".split(),
    "ɑ æ ʌ ɔ aʊ aɪ ɛ ɝ eɪ ɪ i oʊ ɔɪ ʊ u b tʃ d ð f ɡ h dʒ k l m n ŋ p ɹ s ʃ t θ v w j z ʒ".split()))
LETTER_NAMES = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', (
    'EY1|B IY1|S IY1|D IY1|IY1|EH1 F|JH IY1|EY1 CH|AY1|JH EY1|K EY1|EH1 L|EH1 M|'
    'EH1 N|OW1|P IY1|K Y UW1|AA1 R|EH1 S|T IY1|Y UW1|V IY1|D AH1 B AH0 L Y UW0|'
    'EH1 K S|W AY1|Z IY1').split('|')))


def arpabet_to_ipa(phones):
    result = []
    for phone in phones:
        base = re.sub(r"[012]$", "", phone)
        if base not in ARPABET:
            raise ValueError(f"Unknown English phone: {phone}")
        result.append("ə" if phone == "AH0" else "ɚ" if phone == "ER0" else ARPABET[base])
    if not any(re.search(r"[012]$", p) or p == "UW" for p in phones):
        raise ValueError("English pronunciation has no vowel")
    return result


class NeuralG2P:
    def __init__(self,path):
        self.weights=dict(np.load(path,allow_pickle=False))
        self.graphemes=['<pad>','<unk>','</s>']+list('abcdefghijklmnopqrstuvwxyz')
        self.phones=['<pad>','<unk>','<s>','</s>']+('AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 AW0 AW1 AW2 AY0 AY1 AY2 B CH D DH EH0 EH1 EH2 ER0 ER1 ER2 EY0 EY1 EY2 F G HH IH0 IH1 IH2 IY0 IY1 IY2 JH K L M N NG OW0 OW1 OW2 OY0 OY1 OY2 P R S SH T TH UH0 UH1 UH2 UW UW0 UW1 UW2 V W Y Z ZH').split()

    def cell(self,x,h,prefix):
        w=self.weights
        a=x@w[prefix+'_w_ih'].T+w[prefix+'_b_ih'];b=h@w[prefix+'_w_hh'].T+w[prefix+'_b_hh']
        ar,az,an=np.split(a,3,axis=-1);br,bz,bn=np.split(b,3,axis=-1)
        sigmoid=lambda v:1/(1+np.exp(-np.clip(v,-80,80)))
        reset=sigmoid(ar+br);update=sigmoid(az+bz)
        return (1-update)*np.tanh(an+reset*bn)+update*h

    def predict(self,word):
        if not re.fullmatch('[a-z]{1,48}',word):raise ValueError('Neural G2P expects one ASCII English word of at most 48 letters')
        w=self.weights;h=np.zeros((1,w['enc_w_hh'].shape[1]),np.float32)
        for char in list(word)+['</s>']:h=self.cell(w['enc_emb'][[self.graphemes.index(char)]],h,'enc')
        token=2;out=[]
        for _ in range(64):
            h=self.cell(w['dec_emb'][[token]],h,'dec');token=int((h@w['fc_w'].T+w['fc_b']).argmax())
            if token==3:
                arpabet_to_ipa(out)
                return out
            if token<4:raise ValueError('G2P emitted an unknown/special phoneme')
            out.append(self.phones[token])
        raise ValueError('G2P did not finish; manual reading needed')


class EnglishG2P:
    def __init__(self):
        self.root = Path(__file__).with_name("resources")
        self.dictionary = {}
        self.neural = None
        for item in json.loads((self.root / "manifest.json").read_text(encoding="utf-8")):
            if hashlib.sha256((self.root / item["file"]).read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Pronunciation resource hash mismatch: " + item["file"])
        for line in (self.root / "cmudict-cmudict.dict").read_text(encoding="utf-8").splitlines():
            fields = line.split("#", 1)[0].split()
            if len(fields) >= 2:
                key = re.sub(r"\(\d+\)$", "", fields[0].lower())
                self.dictionary.setdefault(key, []).append(fields[1:])

    def lookup(self, word):
        key = unicodedata.normalize("NFC", word).lower().replace("’", "'")
        if key in self.dictionary:
            variants = self.dictionary[key]
            return variants[0], "cmudict", len(variants)
        if re.fullmatch('[A-Z]{2,8}', word):
            # Unknown capitals are initials, not an out-of-dictionary word.
            # Known acronyms (NASA, NATO, ...) retain their dictionary reading.
            letters = [LETTER_NAMES[c].split() for c in word]
            return [phone for letter in letters for phone in letter], "english_initials", 1
        if key.endswith("'s"):
            base, _, _ = self.lookup(key[:-2])
            tail = re.sub(r"[012]$", "", base[-1])
            suffix = ["IH0", "Z"] if tail in {"S","Z","SH","ZH","CH","JH"} else ["S"] if tail in {"P","T","K","F","TH"} else ["Z"]
            return base + suffix, "english_possessive_rule", 1
        # Accented loanwords use an explicit, approximate normalized spelling.
        plain = "".join(c for c in unicodedata.normalize("NFD", key) if not unicodedata.combining(c))
        if not re.fullmatch("[a-z]{1,48}", plain):
            raise ValueError(f"Cannot infer English pronunciation for {word!r}; check the language hint or sung spelling")
        if self.neural is None:
            self.neural = NeuralG2P(self.root / "g2p-checkpoint20.npz")
        return self.neural.predict(plain), "g2pe_neural", 1
