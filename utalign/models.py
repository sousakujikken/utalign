"""アライメントモデルの登録簿とローカル保管.

- 登録簿 (REGISTRY): 検証済みモデルのメタ情報 (ライセンス・語彙種別・サイズ)。
- 保管: models_dir/<repo id を -- で連結した名前>/ に HF スナップショットを丸ごと置く。
  以後の読み込みはこのフォルダからのみ行い、HF Hub には接続しない (オフラインで動く)。
- 登録簿にない HF repo id も指定できる (語彙は vocab.json から自動判定)。
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from . import config as _cfg


@dataclass
class ModelInfo:
    id: str                      # HF repo id
    kind: str                    # hf_ctc
    vocab: str                   # phoneme | hira | kata | romaji | auto
    license: str
    size_mb: int
    commercial: bool
    note: str = ""


REGISTRY: list[ModelInfo] = [
    ModelInfo("prj-beatrice/japanese-hubert-base-phoneme-ctc-v4", "hf_ctc", "phoneme", "Apache-2.0", 378, True,
              "推奨。音素 CTC (ReazonSpeech で学習)。検証 2 曲で残差中央値 14〜21ms、emission が最速"),
    ModelInfo("prj-beatrice/japanese-hubert-base-phoneme-ctc-v3", "hf_ctc", "phoneme", "Apache-2.0", 378, True,
              "v4 の予備 (同等精度)"),
    ModelInfo("TKU410410103/hubert-large-japanese-asr", "hf_ctc", "hira", "Apache-2.0", 1262, True,
              "ひらがな語彙。精度同等だが emission が約 25 倍遅い"),
    ModelInfo("TKU410410103/wav2vec2-base-japanese-asr", "hf_ctc", "hira", "Apache-2.0", 378, True,
              "ひらがな語彙、軽量。残差 150ms 超がやや多い"),
    ModelInfo("vumichien/wav2vec2-large-xlsr-japanese-hiragana", "hf_ctc", "hira", "Apache-2.0", 2524, True,
              "ひらがな語彙 (Common Voice)"),
]

def registry_by_id() -> dict[str, ModelInfo]:
    return {m.id: m for m in REGISTRY}


def local_name(model_id: str) -> str:
    return model_id.replace("/", "--")


def local_path(model_id: str, mdir: Optional[Path] = None) -> Path:
    mdir = mdir or _cfg.models_dir()
    return mdir / local_name(model_id)


def _dir_size_mb(p: Path) -> int:
    if not p.exists():
        return 0
    return int(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6)


def is_downloaded(model_id: str, mdir: Optional[Path] = None) -> bool:
    p = local_path(model_id, mdir)
    if not p.exists() or not (p / "config.json").exists():
        return False
    return any(p.glob("*.safetensors")) or (p / "pytorch_model.bin").exists()


def list_models(mdir: Optional[Path] = None) -> list[dict[str, Any]]:
    """登録簿 + models_dir に置かれた未登録モデルを、保管状態付きで返す."""
    mdir = mdir or _cfg.models_dir()
    out = []
    seen = set()
    for m in REGISTRY:
        d = asdict(m)
        d.update(downloaded=is_downloaded(m.id, mdir), path=str(local_path(m.id, mdir)),
                 local_size_mb=_dir_size_mb(local_path(m.id, mdir)))
        out.append(d); seen.add(m.id)
    for p in sorted(mdir.iterdir()) if mdir.exists() else []:
        if not p.is_dir():
            continue
        mid = p.name.replace("--", "/")
        if mid in seen or not (p / "config.json").exists():
            continue
        out.append({"id": mid, "kind": "hf_ctc", "vocab": "auto", "license": "(未確認)", "size_mb": _dir_size_mb(p),
                    "commercial": False, "note": "models_dir に手動配置", "downloaded": True, "path": str(p),
                    "local_size_mb": _dir_size_mb(p)})
    return out


def download(model_id: str, mdir: Optional[Path] = None, log: Callable[[str], None] = print) -> Path:
    """モデルを models_dir に取得する (既にあれば何もしない)."""
    mdir = mdir or _cfg.models_dir()
    dst = local_path(model_id, mdir)
    if is_downloaded(model_id, mdir):
        log(f"[models] already present: {dst}")
        return dst
    from huggingface_hub import snapshot_download
    log(f"[models] downloading {model_id} -> {dst}")
    snapshot_download(model_id, local_dir=str(dst),
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.md", "LICENSE*"],
                      ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.ckpt"])
    if not any(dst.glob("*.safetensors")):
        # safetensors が無い古いリポジトリは .bin を取る
        snapshot_download(model_id, local_dir=str(dst), allow_patterns=["*.json", "*.bin", "*.txt", "*.md"])
    (dst / "UTALIGN_SOURCE.json").write_text(json.dumps({"repo_id": model_id}, ensure_ascii=False), encoding="utf-8")
    log(f"[models] done: {dst} ({_dir_size_mb(dst)} MB)")
    return dst


def delete(model_id: str, mdir: Optional[Path] = None) -> None:
    p = local_path(model_id, mdir)
    if p.exists():
        shutil.rmtree(p)


def _check_distinct(src: Path, dst: Path) -> None:
    """src と dst が同一または包含関係なら拒否する (削除で唯一の重みを失わないため)."""
    rs, rd = src.resolve(), dst.resolve()
    if rs == rd:
        raise ValueError(f"コピー元とコピー先が同じです: {rs}")
    if rd in rs.parents:
        raise ValueError(f"コピー先がコピー元を含んでいます: {rd}")
    if rs in rd.parents:
        raise ValueError(f"コピー先がコピー元の中にあります: {rd}")


def _copy_replace(src: Path, dst: Path) -> None:
    """src を dst にコピーする. 既存の dst はコピーが成功してから入れ替える."""
    _check_distinct(src, dst)
    tmp = dst.with_name(dst.name + ".utalign-tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if dst.exists():
        old = dst.with_name(dst.name + ".utalign-old")
        if old.exists():
            shutil.rmtree(old)
        dst.rename(old)
        tmp.rename(dst)
        shutil.rmtree(old)
    else:
        tmp.rename(dst)


def import_local(src: Path, model_id: Optional[str] = None, mdir: Optional[Path] = None) -> str:
    """別の場所にあるモデルフォルダ (config.json + safetensors) を models_dir にコピーして登録する."""
    src = Path(src).expanduser()
    if not (src / "config.json").exists():
        raise FileNotFoundError(f"config.json が見つかりません: {src}")
    mid = model_id or src.name.replace("--", "/")
    dst = local_path(mid, mdir)
    _copy_replace(src, dst)
    return mid


def export_local(model_id: str, dst_dir: Path, mdir: Optional[Path] = None) -> Path:
    """保管済みモデルを別フォルダにコピーする (バックアップ用). コピー先が保管場所そのものや元フォルダの中なら拒否."""
    src = local_path(model_id, mdir)
    if not src.exists():
        raise FileNotFoundError(f"モデルが保管されていません: {src}")
    dst = Path(dst_dir).expanduser() / src.name
    _copy_replace(src, dst)
    return dst
