"""Hugging Face の日本語 CTC モデル (Wav2Vec2ForCTC / HubertForCTC) による強制アライメント.

語彙は vocab.json から自動判定:
  phoneme … 'cl' と 'N' を持つ OpenJTalk 系音素 (prj-beatrice/japanese-hubert-base-phoneme-ctc-*)
  hira / kata … かな 1 文字 = 1 トークン
整列は torchaudio.functional.forced_align (BSD)。モデル重みは models_dir から読み、Hub には接続しない。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .base import HOP, SR, Aligner, UnitSpan, chunked_emissions

VOWELS = "aiueo"


def kata2hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in s)


class HFCTCAligner(Aligner):
    name = "hf_ctc"
    supports_star = True

    def __init__(self, model_id: str, models_dir: Optional[Path] = None, device: str = "auto",
                 star: bool = False, star_penalty: float = 1.0):
        from ..config import pick_device
        from ..models import download, is_downloaded, local_path
        self.model_id = model_id
        self.device = pick_device(device)
        self.star = star
        self.star_penalty = star_penalty
        self.path = local_path(model_id, models_dir)
        if not is_downloaded(model_id, models_dir):
            download(model_id, models_dir)
        vocab_file = self.path / "vocab.json"
        if vocab_file.exists():
            self.vocab: dict[str, int] = json.loads(vocab_file.read_text(encoding="utf-8"))
        else:
            from transformers import AutoTokenizer
            self.vocab = AutoTokenizer.from_pretrained(str(self.path)).get_vocab()
        cfg = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
        self.blank = cfg.get("pad_token_id")
        if self.blank is None:
            for k in ("<pad>", "[PAD]", "PAD", "<blk>", "<blank>"):
                if k in self.vocab:
                    self.blank = self.vocab[k]; break
        if self.blank is None:
            self.blank = 0
        self.has_hira = "あ" in self.vocab
        self.has_kata = "ア" in self.vocab
        self.is_phoneme = "cl" in self.vocab and "N" in self.vocab
        self.vocab_kind = "phoneme" if self.is_phoneme else ("hira" if self.has_hira else ("kata" if self.has_kata else "unknown"))
        if self.vocab_kind == "unknown":
            raise ValueError(f"{model_id}: 音素でもかなでもない語彙のため使えません")
        self._model = None
        self._fe = None

    # ---------------------------------------------------------------- tokens
    def mora_tokens(self, mora, next_mora) -> list[str]:
        if self.is_phoneme:
            if mora.kind == "hatsuon":
                return ["N"]
            if mora.kind == "sokuon":
                return ["cl"]
            if mora.kind == "choon":
                return []
            r = mora.romaji
            if not r:
                return []
            if r[-1] not in VOWELS:
                return [c for c in r if c in self.vocab]
            cons, v = r[:-1], r[-1]
            if not cons:
                return [v]
            if cons in self.vocab:
                return [cons, v]
            return [c for c in cons if c in self.vocab] + [v]
        s = mora.kana
        if mora.kind == "choon" and "ー" not in self.vocab:
            return []
        if self.has_hira and not self.has_kata:
            s = kata2hira(s)
        toks = []
        for ch in s:
            if ch in self.vocab:
                toks.append(ch)
            elif kata2hira(ch) in self.vocab:
                toks.append(kata2hira(ch))
        return toks

    # ---------------------------------------------------------------- emission
    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCTC
            try:
                from transformers import AutoFeatureExtractor
                self._fe = AutoFeatureExtractor.from_pretrained(str(self.path))
            except Exception:
                from transformers import Wav2Vec2FeatureExtractor
                self._fe = Wav2Vec2FeatureExtractor(sampling_rate=SR, do_normalize=True, return_attention_mask=False)
            self._model = AutoModelForCTC.from_pretrained(str(self.path)).to(self.device).eval()
        return self._model, self._fe

    def compute_emissions(self, wav: np.ndarray, cache_dir: Optional[Path] = None) -> np.ndarray:
        import torch
        model, fe = self._load()

        def run(seg: np.ndarray) -> np.ndarray:
            x = fe(seg, sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
            with torch.inference_mode():
                return torch.log_softmax(model(x).logits[0].float(), dim=-1).cpu().numpy()
        return chunked_emissions(wav, run, f"hf_ctc:{self.model_id}", cache_dir)

    # ---------------------------------------------------------------- align
    def align(self, emission: np.ndarray, units) -> list[Optional[UnitSpan]]:
        import torch
        import torchaudio.functional as F
        em = emission
        star_id = None
        if self.star:
            nb = np.delete(em, self.blank, axis=1)
            star = nb.max(axis=1) - self.star_penalty
            em = np.concatenate([em, star[:, None]], axis=1)
            star_id = emission.shape[1]
        targets: list[int] = []
        ranges: list[Optional[tuple[int, int]]] = []
        for u in units:
            if u.is_star:
                if star_id is not None:
                    ranges.append((len(targets), len(targets) + 1)); targets.append(star_id)
                else:
                    ranges.append(None)
                continue
            ids = [self.vocab[t] for t in u.tokens if t in self.vocab]
            if not ids:
                ranges.append(None); continue
            ranges.append((len(targets), len(targets) + len(ids))); targets += ids
        lp = torch.from_numpy(np.ascontiguousarray(em))[None]
        tg = torch.tensor(targets, dtype=torch.int32)[None]
        ali, scores = F.forced_align(lp, tg, blank=self.blank)
        spans = F.merge_tokens(ali[0], scores[0].exp(), blank=self.blank)
        assert len(spans) == len(targets), (len(spans), len(targets))
        ratio = HOP / SR
        out: list[Optional[UnitSpan]] = []
        for u, rg in zip(units, ranges):
            if rg is None or u.is_star:
                out.append(None); continue
            sp = spans[rg[0]:rg[1]]
            out.append(UnitSpan(unit=u.id, start=sp[0].start * ratio, end=sp[-1].end * ratio,
                                score=float(np.mean([s.score for s in sp]))))
        return out
