#!/usr/bin/env python3
"""THIRD_PARTY_NOTICES.md を生成する。

対象は現在の環境 (venv) にインストールされている utalign の依存パッケージ全体
(`uv sync` / `pip install -e .` 後に実行する)。各パッケージの版・ライセンス・配布元を
importlib.metadata から取り、名前順に並べる。utalign 自身は除く。

使い方: ./venv/bin/python tools/generate_third_party_notices.py [--check]
"""
from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "THIRD_PARTY_NOTICES.md"
SELF = {"utalign"}


def license_of(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    expr = meta.get("License-Expression")
    if expr:
        return expr.strip()
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    classifiers = [c for c in classifiers if c and c != "OSI Approved"]
    if classifiers:
        return " / ".join(dict.fromkeys(classifiers))
    lic = (meta.get("License") or "").strip()
    if lic and "\n" not in lic and len(lic) <= 80:
        return lic
    return "(see package metadata)"


def homepage_of(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    for url in meta.get_all("Project-URL") or []:
        label, _, target = url.partition(",")
        if label.strip().lower() in ("homepage", "source", "repository", "source code"):
            return target.strip()
    return (meta.get("Home-page") or "").strip()


def render() -> str:
    rows = []
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").strip()
        if not name or name.lower() in SELF:
            continue
        rows.append((name.lower(), name, dist.version, license_of(dist), homepage_of(dist)))
    rows.sort()
    lines = [
        "# Third-party notices",
        "",
        "utalign は下記のパッケージに依存します。この一覧は `uv.lock` で固定した版を `uv sync` した環境から",
        "`tools/generate_third_party_notices.py` で生成したものです。各パッケージはそれぞれのライセンスに従います。",
        "utalign の wheel はこれらを同梱せず、インストール時に利用者の環境で PyPI から取得されます。",
        "",
        "既定モデル `prj-beatrice/japanese-hubert-base-phoneme-ctc-v4` (Apache-2.0) は初回実行時に Hugging Face Hub から",
        "取得され、utalign は再配布しません。",
        "",
        "手動で編集しないでください。",
        "",
    ]
    for _, name, version, lic, url in rows:
        lines.append(f"- {name}@{version} — {lic}" + (f" — {url}" if url else ""))
    return "\n".join(lines) + "\n"


def main() -> None:
    text = render()
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            sys.exit("THIRD_PARTY_NOTICES.md is stale; run tools/generate_third_party_notices.py")
        print(f"[notices] up to date ({text.count(chr(10) + '- ')} packages)")
        return
    OUT.write_text(text, encoding="utf-8")
    print(f"[notices] wrote {OUT} ({text.count(chr(10) + '- ')} packages)")


if __name__ == "__main__":
    main()
