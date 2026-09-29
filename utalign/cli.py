"""CLI: align / export-utavista / utavista-align / doctor / serve / models / config.

UTAVISTA からは `utalign doctor --json` (環境確認) と
`utalign utavista-align ... --progress-json` (解析 + JSON 書き出し) を子プロセスとして呼ぶ。
--progress-json のとき stdout には 1 行 1 JSON のイベントのみを書き、人間向けログは
{"event":"log"} に包む (ライブラリの print は stderr に逃がす)。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from . import __version__
from . import config as _cfg


def _align_options(a: argparse.Namespace, cfg: dict[str, Any], write_playback: bool = True):
    from .pipeline import AlignOptions
    readings = Path(a.readings) if a.readings else None
    return AlignOptions(model=a.model or cfg["model"], device=a.device or cfg["device"],
                        star=a.star, min_rest=a.min_rest,
                        midi_tracks=[int(x) for x in a.midi_tracks.split(",")] if a.midi_tracks else None,
                        keep_drums=a.keep_drums, readings=readings, models_dir=_cfg.models_dir(cfg),
                        write_playback=write_playback)


def _add_align_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--audio", required=True); p.add_argument("--lyrics", required=True); p.add_argument("--midi", required=True)
    p.add_argument("--readings", default=None, help="読みの上書きファイル (省略時は使わない)")
    p.add_argument("--model", default=None, help="hf_ctc のモデル ID (既定は config の model)")
    p.add_argument("--device", default=None, help="auto | cpu | mps | cuda")
    p.add_argument("--star", action="store_true", help="行境界に歌詞外発声を吸収する star トークンを入れる")
    p.add_argument("--min-rest", type=float, default=0.2)
    p.add_argument("--midi-tracks", default=None, help="使う MIDI トラック番号 (カンマ区切り、省略で全部)")
    p.add_argument("--keep-drums", action="store_true", help="drum 系トラック / ch10 を除外しない")


def cmd_align(a: argparse.Namespace) -> None:
    from .pipeline import run_align
    cfg = _cfg.load_config()
    run_align(Path(a.audio), Path(a.lyrics), Path(a.midi), Path(a.out), Path(a.work), _align_options(a, cfg))


def cmd_export_utavista(a: argparse.Namespace) -> None:
    from .pipeline import run_export
    run_export(Path(a.result), Path(a.out), ruby=a.ruby, strip_spaces=a.strip_spaces)


class _JsonEvents:
    """--progress-json: 本来の stdout にイベント JSON を 1 行ずつ書き、それ以外の出力は stderr へ."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.stdout = sys.stdout

    def emit(self, ev: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.stdout.write(json.dumps(ev, ensure_ascii=False) + "\n")
        self.stdout.flush()

    def log(self, s: str) -> None:
        if self.enabled:
            self.emit({"event": "log", "text": str(s)})
        else:
            print(s, flush=True)

    def redirect(self):
        return contextlib.redirect_stdout(sys.stderr) if self.enabled else contextlib.nullcontext()


def cmd_utavista_align(a: argparse.Namespace) -> None:
    from .pipeline import run_utavista
    ev = _JsonEvents(a.progress_json)
    cfg = _cfg.load_config()
    opt = _align_options(a, cfg, write_playback=a.viewer)
    try:
        with ev.redirect():
            ev.emit({"event": "start", "version": __version__, "model": opt.model, "device": _cfg.pick_device(opt.device)})
            res = run_utavista(Path(a.audio), Path(a.lyrics), Path(a.midi), Path(a.out_json), Path(a.work), opt,
                               ruby=a.ruby, strip_spaces=a.strip_spaces, out_dir=Path(a.out_dir) if a.out_dir else None,
                               log=ev.log, progress=ev.emit)
        ev.emit({"event": "done", "result": res})
    except Exception as e:  # noqa: BLE001
        import traceback
        msg = f"{type(e).__name__}: {e}"
        if ev.enabled:
            ev.emit({"event": "error", "message": msg, "traceback": traceback.format_exc()})
        else:
            traceback.print_exc()
        sys.exit(1)


def doctor_info(cfg: Optional[dict[str, Any]] = None, probe_device: bool = True) -> dict[str, Any]:
    from . import models as _models
    from .audio import _afconvert_available
    cfg = cfg or _cfg.load_config()
    mdir = _cfg.models_dir(cfg)
    model = cfg["model"]
    info: dict[str, Any] = {
        "name": "utalign", "version": __version__, "python": platform.python_version(), "platform": sys.platform,
        "executable": sys.executable, "home": str(_cfg.app_home()), "config": str(_cfg.config_path()),
        "modelsDir": str(mdir), "model": model, "modelDownloaded": _models.is_downloaded(model, mdir),
        "modelPath": str(_models.local_path(model, mdir)), "afconvert": _afconvert_available(),
        "devicePreference": cfg["device"],
        # macOS のバージョンと、その MPS に畳み込み出力 65,536 上限 (macOS < 15.1) があるか。
        # utalign 自体は窓幅を上限内に収めているので上限があっても MPS で動くが、UTAVISTA の案内表示用に出す。
        "macos": ".".join(map(str, mv)) if (mv := _cfg.macos_version()) else None,
        "mpsConvLimited": _cfg.mps_conv_limited(),
    }
    if probe_device:
        try:
            info["device"] = _cfg.pick_device(cfg["device"])
            import torch
            info["torch"] = torch.__version__
        except Exception as e:  # noqa: BLE001
            info["device"] = None; info["deviceError"] = f"{type(e).__name__}: {e}"
    return info


def cmd_doctor(a: argparse.Namespace) -> None:
    info = doctor_info(probe_device=not a.no_device)
    if a.json:
        print(json.dumps(info, ensure_ascii=False))
        return
    for k, v in info.items():
        print(f"{k:16s} {v}")


def cmd_serve(a: argparse.Namespace) -> None:
    from .server import serve
    cfg = _cfg.load_config()
    serve(port=a.port or int(cfg["port"]), open_browser=not a.no_browser, static_dir=Path(a.dir) if a.dir else None)


def cmd_models(a: argparse.Namespace) -> None:
    from . import models
    cfg = _cfg.load_config()
    mdir = _cfg.models_dir(cfg)
    if a.action == "list":
        print(f"models_dir: {mdir}")
        for m in models.list_models(mdir):
            mark = "✓" if m["downloaded"] else " "
            print(f" [{mark}] {m['id']:55s} {m['license']:14s} {m['vocab']:8s} {m['size_mb']:5d}MB  {m['note']}")
    elif a.action == "download":
        models.download(a.model or cfg["model"], mdir, log=lambda s: print(s, flush=True))
    elif a.action == "delete":
        models.delete(a.model, mdir); print("deleted", a.model)
    elif a.action == "import":
        mid = models.import_local(Path(a.path), a.model, mdir); print("imported as", mid)
    elif a.action == "export":
        print("copied to", models.export_local(a.model or cfg["model"], Path(a.path), mdir))


def cmd_config(a: argparse.Namespace) -> None:
    cfg = _cfg.load_config()
    if a.set:
        for kv in a.set:
            k, _, v = kv.partition("=")
            if k not in _cfg.DEFAULTS:
                sys.exit(f"unknown key: {k} (keys: {', '.join(_cfg.DEFAULTS)})")
            d = _cfg.DEFAULTS[k]
            cfg[k] = (v.lower() in ("1", "true", "yes")) if isinstance(d, bool) else type(d)(v) if isinstance(d, (int, float)) else v
        cfg = _cfg.save_config(cfg)
    print(f"config: {_cfg.config_path()}")
    print(f"models_dir -> {_cfg.models_dir(cfg)}\nprojects_dir -> {_cfg.projects_dir(cfg)}")
    print(json.dumps(cfg, ensure_ascii=False, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="utalign")
    ap.add_argument("--version", action="version", version=f"utalign {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("align", help="歌詞・音声・MIDI から文字タイミングを推定 (result.json + ビューア)")
    _add_align_args(p)
    p.add_argument("--out", default="out"); p.add_argument("--work", default="work")
    p.set_defaults(func=cmd_align)

    p = sub.add_parser("export-utavista", help="result.json から UTAVISTA lyrics-timing/1.0 JSON を出力")
    p.add_argument("--result", default="out/result.json"); p.add_argument("--out", default="out/utavista-lyrics-timing.json")
    p.add_argument("--strip-spaces", action="store_true"); p.add_argument("--ruby", action="store_true")
    p.set_defaults(func=cmd_export_utavista)

    p = sub.add_parser("utavista-align", help="解析 + UTAVISTA JSON 書き出しを一括で行う (UTAVISTA 連携用)")
    _add_align_args(p)
    p.add_argument("--out-json", required=True, help="UTAVISTA lyrics-timing/1.0 JSON の出力先")
    p.add_argument("--work", required=True, help="16kHz 音声 / emission キャッシュと result.json の置き場")
    p.add_argument("--out-dir", default=None, help="result.json の置き場 (既定: <work>/out)")
    p.add_argument("--ruby", dest="ruby", action="store_true", default=True, help="漢字にルビを付与 (既定)")
    p.add_argument("--no-ruby", dest="ruby", action="store_false")
    p.add_argument("--strip-spaces", action="store_true", help="phrase.text から空白を除く")
    p.add_argument("--viewer", action="store_true", help="再生用 WAV とビューアも out-dir に書く")
    p.add_argument("--progress-json", action="store_true", help="stdout に 1 行 1 JSON の進捗イベントを書く")
    p.set_defaults(func=cmd_utavista_align)

    p = sub.add_parser("doctor", help="環境・モデル取得状況を表示 (UTAVISTA の検出用に --json)")
    p.add_argument("--json", action="store_true"); p.add_argument("--no-device", action="store_true", help="torch を読み込まない")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("serve", help="GUI アプリをローカル配信")
    p.add_argument("--port", type=int, default=None); p.add_argument("--no-browser", action="store_true")
    p.add_argument("--dir", default=None, help="(互換) 旧形式の out/ フォルダをビューアで開く")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("models", help="モデルの一覧 / 取得 / 削除 / 取り込み / 書き出し")
    p.add_argument("action", choices=["list", "download", "delete", "import", "export"])
    p.add_argument("model", nargs="?", default=None); p.add_argument("--path", default=None)
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("config", help="設定の表示・変更 (--set key=value)")
    p.add_argument("--set", action="append", default=None)
    p.set_defaults(func=cmd_config)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
