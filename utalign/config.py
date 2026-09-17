"""アプリ設定 (保存場所・既定値).

設定ファイル: $UTALIGN_HOME/config.json
  既定の UTALIGN_HOME: macOS = ~/Library/Application Support/utalign, それ以外 = ~/.local/share/utalign
モデルの保管場所 (models_dir) とプロジェクト置き場 (projects_dir) はここから相対でも絶対でも指定できる。
旧名 (utaalign) のホームが残っていて新ホームが無い場合は初回に自動で移動する。
"""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Optional

DEFAULT_MODEL = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4"
APP_NAME = "utalign"
LEGACY_APP_NAME = "utaalign"


def _default_home(name: str) -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / name
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / name


def app_home() -> Path:
    env = os.environ.get("UTALIGN_HOME")
    if env:
        return Path(env).expanduser().resolve()
    home = _default_home(APP_NAME)
    legacy = _default_home(LEGACY_APP_NAME)
    if not home.exists() and legacy.exists():
        try:
            legacy.rename(home)
        except OSError:
            pass
    return home


DEFAULTS: dict[str, Any] = {
    "models_dir": "models",            # app_home 相対 (絶対パスも可)
    "projects_dir": "projects",        # 同上
    "model": DEFAULT_MODEL,            # hf_ctc のモデル ID (HF repo id または登録済みローカル名)
    "device": "auto",                  # auto | cpu | mps | cuda
    "port": 8791,
    "min_rest": 0.2,                   # フレーズ分割の休符長 (秒)
    "utavista_dir": "",                # utavista2 のリポジトリ (検証器を使う場合)
    "export_ruby": True,
    "export_strip_spaces": False,
    "recent_dirs": [],                 # ファイル選択ダイアログの最近使ったフォルダ
}


def config_path() -> Path:
    return app_home() / "config.json"


def load_config() -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    p = config_path()
    if p.exists():
        try:
            loaded = json.loads(p.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in loaded.items() if k in DEFAULTS})
        except Exception:
            pass
    return cfg


def save_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = {k: cfg.get(k, v) for k, v in DEFAULTS.items()}
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def resolve_dir(cfg: dict[str, Any], key: str) -> Path:
    raw = str(cfg.get(key) or DEFAULTS[key])
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = app_home() / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir(cfg: dict[str, Any] | None = None) -> Path:
    return resolve_dir(cfg or load_config(), "models_dir")


def projects_dir(cfg: dict[str, Any] | None = None) -> Path:
    return resolve_dir(cfg or load_config(), "projects_dir")


def pick_device(pref: str = "auto") -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# PyTorch の MPS 畳み込み (aten/native/mps/operations/Convolution.mm) は macOS 15.1 未満のとき
# 出力テンソルの全次元 (チャネルだけでなく時間軸も) が 65,536 以下でないと NotImplementedError を投げる。
MPS_CONV_MAX_OUTPUT = 65536
MPS_CONV_LIMIT_FIXED_IN = (15, 1)


def macos_version() -> Optional[tuple[int, ...]]:
    """macOS のバージョン (例 (14, 6, 1))。macOS 以外や取得できないときは None."""
    if sys.platform != "darwin":
        return None
    raw = platform.mac_ver()[0]
    try:
        return tuple(int(x) for x in raw.split(".")) if raw else None
    except ValueError:
        return None


def mps_conv_limited(ver: Optional[tuple[int, ...]] = None) -> bool:
    """この macOS の MPS に畳み込み出力 65,536 上限があるか (macOS < 15.1 で True)."""
    if ver is None:
        ver = macos_version()
    return ver is not None and ver < MPS_CONV_LIMIT_FIXED_IN
