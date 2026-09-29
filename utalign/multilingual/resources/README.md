# Bundled English pronunciation resources

These files are shipped in the wheel so that G2P needs no runtime downloads.
`manifest.json` records the upstream commit, source URL, size, and SHA-256.
`EnglishG2P` checks the hashes before loading them. No song-specific entries
or eSpeak output are included.

* `cmudict-cmudict.dict`: CMUSphinx CMUdict, copyright Carnegie Mellon
  University. BSD-style two-condition license in `cmudict-LICENSE`.
  https://github.com/cmusphinx/cmudict
* `g2p-checkpoint20.npz`: NumPy checkpoint from Kyubyong Park and Jongseok Kim's
  g2pE. Apache-2.0, complete license in `g2p-LICENSE.txt`.
  https://github.com/Kyubyong/g2p

`../english.py` adapts g2pE's GRU inference: it uses NumPy only, validates decoder
termination and phone output, and omits NLTK, POS tagging, number normalization,
and runtime downloads. CMUdict alternatives are counted, but the first is used;
contextual heteronym disambiguation is not claimed. Unknown capitalized initials
use letter names; other unknown English words use the bundled model.

Other languages use installed packages, not these English assets:

* Epitran 1.35.2: MIT license text in its wheel. Packaged `SimpleEpitran` rules
  for German, French, Spanish, Italian, Korean and Mandarin pinyin only.
  https://github.com/dmort27/epitran/blob/master/LICENSE.txt
* pypinyin 0.55.0: MIT license and packaged character/phrase readings.
  https://github.com/mozillazg/python-pinyin/blob/master/LICENSE.txt
* Epitran's transitive dependencies are recorded in `uv.lock` and
  `THIRD_PARTY_NOTICES.md`. libmarisa is dual licensed; the BSD-2-Clause option
  is used (not its alternative LGPL license).

The adapters add independently implemented pronunciation/syllabification rules
for initials, directional German compounds, French eau, and Romance glides.
These remain pronunciation hypotheses. No eSpeak library, command, generated
pronunciation dictionary, or fallback is used by the production package.
