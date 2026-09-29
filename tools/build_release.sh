#!/bin/sh
# Release 資産を dist/ に生成する: wheel、requirements.lock (固定版 + ハッシュ)、THIRD_PARTY_NOTICES.md。
# 前提: uv が PATH にあり、`uv sync --extra dev` 済み (venv/)。
set -eu
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-venv}"
rm -rf dist
uv lock
uv sync --extra dev
uv build --wheel --out-dir dist
uv export --format requirements-txt --no-dev --no-emit-project --output-file dist/requirements.lock
./venv/bin/python tools/generate_third_party_notices.py
cp THIRD_PARTY_NOTICES.md dist/
echo "[release] dist/:"; ls -la dist
