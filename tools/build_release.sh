#!/bin/sh
# Release 資産を生成する: wheel、requirements.lock (固定版 + ハッシュ)、THIRD_PARTY_NOTICES.md、
# release-inputs.txt (ソース revision と固定した入力)、SHA256SUMS。
#
# 使い方: tools/build_release.sh <出力ディレクトリ (存在しないか空)>
#
# - コミット済みの HEAD を git archive で取り出して、その内容だけからビルドする (作業ツリーの未コミット変更・
#   build/ や *.egg-info の残骸は入らない)。作業ツリーが clean でなければ中止する。
# - uv.lock は更新しない (uv lock を実行しない、export は --frozen)。依存のインストールも行わない。
# - ビルドバックエンドは setuptools==84.2.1 に固定する (uv build --build-constraint)。
# - 既存のファイルは消さない。出力先が空でなければ中止する。
# - THIRD_PARTY_NOTICES.md は依存集合が変わっていないため、コミット済みの監査済みファイルをそのまま添付する。
#   依存を変えたときは環境を用意して tools/generate_third_party_notices.py で再生成・確認してからコミットすること。
set -eu

BUILD_BACKEND="setuptools==84.2.1"

OUT="${1:-}"
if [ -z "$OUT" ]; then
  echo "usage: tools/build_release.sh <empty-output-dir>" >&2
  exit 2
fi
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
  echo "[release] output directory is not empty: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain --untracked-files=all)" ]; then
  echo "[release] working tree is not clean; commit or remove changes first" >&2
  exit 1
fi
REV="$(git rev-parse HEAD)"
EPOCH="$(git log -1 --format=%ct HEAD)"
VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml | head -n 1)"
if ! grep -qx "__version__ = \"$VERSION\"" utalign/__init__.py; then
  echo "[release] pyproject.toml version $VERSION does not match utalign/__init__.py" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM
mkdir "$TMP/src"
git archive --format=tar "$REV" | tar -x -C "$TMP/src"
printf '%s\n' "$BUILD_BACKEND" > "$TMP/build-constraints.txt"

# wheel のタイムスタンプをコミット時刻に揃える
export SOURCE_DATE_EPOCH="$EPOCH"

uv build --wheel --python 3.13 --build-constraint "$TMP/build-constraints.txt" --out-dir "$OUT" "$TMP/src"
uv export --project "$TMP/src" --frozen --no-dev --no-emit-project --no-header \
  --format requirements-txt --output-file "$OUT/requirements.lock"
cp "$TMP/src/THIRD_PARTY_NOTICES.md" "$OUT/"

LOCK_SHA="$(shasum -a 256 "$TMP/src/uv.lock" | cut -d ' ' -f 1)"
cat > "$OUT/release-inputs.txt" <<EOF
name: utalign
version: $VERSION
source_revision: $REV
source_date_epoch: $EPOCH
uv_lock_sha256: $LOCK_SHA
build_backend: $BUILD_BACKEND
build_python: 3.13
uv: $(uv --version)
requirements_export: uv export --frozen --no-dev --no-emit-project --no-header --format requirements-txt
third_party_notices: committed file at source_revision (not regenerated)
EOF

# 生成物のハッシュ (ファイル名のみ、ローカルの絶対パスは含めない)
(cd "$OUT" && find . -maxdepth 1 -type f ! -name SHA256SUMS ! -name .gitignore | sed 's|^\./||' | LC_ALL=C sort \
  | while IFS= read -r f; do shasum -a 256 "$f"; done > SHA256SUMS)

echo "[release] $VERSION ($REV) -> $OUT"
ls -la "$OUT"
