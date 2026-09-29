# Third-party notices

utalign は下記のパッケージに依存します。この一覧は `uv.lock` で固定した版を `uv sync` した環境から
`tools/generate_third_party_notices.py` で生成したものです。各パッケージはそれぞれのライセンスに従います。
utalign の wheel はこれらを同梱せず、インストール時に利用者の環境で PyPI から取得されます。

既定モデル `prj-beatrice/japanese-hubert-base-phoneme-ctc-v4` (Apache-2.0) は初回実行時に Hugging Face Hub から
取得され、utalign は再配布しません。

手動で編集しないでください。

- annotated-doc@0.0.5 — MIT — https://github.com/fastapi/annotated-doc
- anyio@4.15.1 — MIT — https://github.com/agronholm/anyio
- certifi@2026.7.22 — Mozilla Public License 2.0 (MPL 2.0) — https://github.com/certifi/python-certifi
- cffi@2.1.1 — MIT-0 — https://github.com/python-cffi/cffi
- charset-normalizer@3.5.1 — MIT
- click@8.5.0 — BSD-3-Clause — https://github.com/pallets/click/
- cloudpickle@3.1.2 — BSD License — https://github.com/cloudpipe/cloudpickle
- decorator@5.3.1 — BSD-2-Clause
- filelock@4.0.0 — MIT — https://github.com/tox-dev/py-filelock
- fsspec@2026.7.0 — BSD-3-Clause — https://github.com/fsspec/filesystem_spec
- fugashi@1.5.2 — MIT AND BSD-3-Clause — https://github.com/polm/fugashi
- h11@0.16.0 — MIT License — https://github.com/python-hyper/h11
- hf-xet@1.6.0 — Apache-2.0 — https://github.com/huggingface/xet-core
- httpcore@1.0.9 — BSD-3-Clause — https://www.encode.io/httpcore/
- httpx@0.28.1 — BSD License — https://github.com/encode/httpx
- huggingface_hub@1.32.0 — Apache Software License — https://github.com/huggingface/huggingface_hub
- idna@3.19 — BSD-3-Clause — https://github.com/kjd/idna
- iniconfig@2.3.0 — MIT — https://github.com/pytest-dev/iniconfig
- Jinja2@3.1.6 — BSD License — https://github.com/pallets/jinja/
- joblib@1.6.0 — BSD-3-Clause — https://joblib.readthedocs.io
- lazy-loader@0.5 — BSD-3-Clause — https://github.com/scientific-python/lazy-loader
- librosa@1.0.0 — ISC License (ISCL) — https://github.com/librosa/librosa
- llvmlite@0.49.0 — BSD-2-Clause AND Apache-2.0 WITH LLVM-exception — https://github.com/numba/llvmlite
- markdown-it-py@4.2.0 — MIT License — https://github.com/executablebooks/markdown-it-py
- MarkupSafe@3.0.3 — BSD-3-Clause — https://github.com/pallets/markupsafe/
- mdurl@0.1.2 — MIT License — https://github.com/executablebooks/mdurl
- mido@1.3.3 — MIT License — https://github.com/mido/mido
- mpmath@1.3.0 — BSD License — https://github.com/fredrik-johansson/mpmath
- msgpack@1.2.2 — Apache-2.0 — https://msgpack.org/
- narwhals@2.26.0 — MIT — https://github.com/narwhals-dev/narwhals
- networkx@3.6.1 — BSD-3-Clause — https://networkx.org/
- numba@0.67.0 — BSD License — https://numba.pydata.org
- numpy@2.5.3 — BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 — https://numpy.org
- packaging@26.3 — Apache-2.0 OR BSD-2-Clause — https://github.com/pypa/packaging
- platformdirs@4.11.9 — MIT — https://github.com/tox-dev/platformdirs
- pluggy@1.6.0 — MIT License
- pooch@1.9.0 — BSD-3-Clause — https://github.com/fatiando/pooch
- pycparser@3.0 — BSD-3-Clause — https://github.com/eliben/pycparser
- Pygments@2.21.0 — BSD-2-Clause — https://pygments.org
- pytest@9.1.1 — MIT — https://docs.pytest.org/en/latest/
- PyYAML@6.0.3 — MIT License — https://github.com/yaml/pyyaml
- regex@2026.9.10 — Apache-2.0 AND CNRI-Python — https://github.com/mrabarnett/mrab-regex
- requests@2.34.2 — Apache Software License — https://github.com/psf/requests
- rich@15.0.0 — MIT License — https://github.com/Textualize/rich
- safetensors@0.8.0 — Apache Software License — https://github.com/huggingface/safetensors
- scikit-learn@1.9.1 — BSD-3-Clause — https://scikit-learn.org
- scipy@1.18.1 — BSD License — https://scipy.org/
- setuptools@84.0.0 — MIT — https://github.com/pypa/setuptools
- shellingham@1.5.4 — ISC License (ISCL) — https://github.com/sarugaku/shellingham
- soundfile@0.14.0 — BSD License — https://github.com/bastibe/python-soundfile
- soxr@1.1.0 — LGPL-2.1-or-later — https://github.com/dofuuz/python-soxr
- sympy@1.14.0 — BSD License — https://github.com/sympy/sympy
- threadpoolctl@3.7.0 — BSD-3-Clause — https://github.com/joblib/threadpoolctl
- tokenizers@0.23.2 — Apache Software License — https://github.com/huggingface/tokenizers
- torch@2.14.0 — Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT — https://pytorch.org
- torchaudio@2.11.0 — BSD License — https://github.com/pytorch/audio
- tqdm@4.70.1 — MPL-2.0 AND MIT — https://tqdm.github.io
- transformers@5.17.0 — Apache 2.0 License — https://github.com/huggingface/transformers
- typer@0.27.2 — MIT — https://github.com/fastapi/typer
- typing_extensions@4.16.0 — PSF-2.0 — https://github.com/python/typing_extensions
- unidic-lite@1.0.8 — MIT License — https://github.com/polm/unidic-lite
- urllib3@2.8.0 — MIT
