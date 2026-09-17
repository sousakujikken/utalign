"""ローカル GUI アプリのサーバ (標準ライブラリのみ).

  GET  /                          アプリ (utalign/web/index.html)
  GET  /static/<file>             アプリの静的ファイル
  GET  /projects/<slug>/out/<f>   解析出力 (Range 対応: <audio> のシークに必要)
  API (JSON):
    GET/PUT  /api/config
    GET      /api/models            POST /api/models/download {id}   POST /api/models/delete {id}
             POST /api/models/import {path,id}   POST /api/models/export {id,path}
    GET      /api/fs?path=&kind=    フォルダ閲覧 (ローカルのファイル選択用)
    GET      /api/projects          POST /api/projects {name}
    GET      /api/projects/<slug>   PUT /api/projects/<slug> {name?,options?,readings?,inputs?}  DELETE
    POST     /api/projects/<slug>/upload/<kind>   (multipart/form-data, file)
    POST     /api/projects/<slug>/align           -> job
    POST     /api/projects/<slug>/export          -> job
    GET      /api/jobs/<id>?from=N  ログの差分取得
    GET      /api/jobs              実行中ジョブ
"""
from __future__ import annotations

import functools
import http.server
import json
import mimetypes
import os
import re
import sys
import threading
import urllib.parse
import webbrowser
from email.parser import BytesParser
from email.policy import HTTP
from pathlib import Path
from typing import Any, Optional

from . import config as _cfg
from . import models as _models
from . import project as _proj

WEB_DIR = Path(__file__).resolve().parent / "web"
JOBS = _proj.JobManager()


class _Range:
    """Range リクエストのヘルパ (ファイルを部分送信)."""

    @staticmethod
    def send_file(h: http.server.BaseHTTPRequestHandler, path: Path, ctype: Optional[str] = None) -> None:
        if not path.is_file():
            h.send_error(404); return
        size = path.stat().st_size
        ctype = ctype or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        rng = h.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if rng:
            try:
                unit, _, spec = rng.partition("=")
                lo, _, hi = spec.partition("-")
                start = int(lo) if lo else max(0, size - int(hi))
                end = int(hi) if (lo and hi) else size - 1
                end = min(end, size - 1)
                if unit != "bytes" or start > end:
                    raise ValueError
                status = 206
            except ValueError:
                h.send_error(416); return
        h.send_response(status)
        h.send_header("Content-Type", ctype)
        h.send_header("Accept-Ranges", "bytes")
        h.send_header("Content-Length", str(end - start + 1))
        h.send_header("Cache-Control", "no-cache")
        if status == 206:
            h.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        h.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                buf = f.read(min(1 << 16, remaining))
                if not buf:
                    break
                h.wfile.write(buf); remaining -= len(buf)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    static_dir: Optional[Path] = None   # 互換モード (旧 out/ をそのまま配信)

    def log_message(self, fmt, *args):  # noqa: D102
        pass

    # ---------------------------------------------------------------- helpers
    def _json(self, obj: Any, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, msg: str, status: int = 400) -> None:
        self._json({"error": msg}, status)

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _read_json(self) -> dict[str, Any]:
        raw = self._read_body()
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _read_multipart(self) -> list[tuple[str, str, bytes]]:
        ctype = self.headers.get("Content-Type", "")
        raw = self._read_body()
        msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw)
        out = []
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            fname = part.get_filename() or ""
            out.append((name or "", fname, part.get_payload(decode=True) or b""))
        return out

    def _route(self, method: str) -> None:
        url = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(url.path)
        q = dict(urllib.parse.parse_qsl(url.query))
        try:
            if path.startswith("/api/"):
                self._api(method, path[5:].rstrip("/"), q)
            elif method == "GET":
                self._static(path)
            else:
                self._error("not found", 404)
        except FileNotFoundError as e:
            self._error(str(e), 404)
        except (ValueError, KeyError) as e:
            self._error(f"{type(e).__name__}: {e}", 400)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._error(f"{type(e).__name__}: {e}", 500)

    def do_GET(self): self._route("GET")
    def do_POST(self): self._route("POST")
    def do_PUT(self): self._route("PUT")
    def do_DELETE(self): self._route("DELETE")

    # ---------------------------------------------------------------- static
    def _static(self, path: str) -> None:
        if self.static_dir is not None:
            # 互換モード: 旧 out/ ディレクトリをそのまま配信 (index.html はビューア)
            rel = path.lstrip("/") or "index.html"
            f = (self.static_dir / rel).resolve()
            if not str(f).startswith(str(self.static_dir.resolve())):
                self._error("forbidden", 403); return
            if rel == "index.html" and not f.exists():
                f = WEB_DIR / "viewer.html"
            if not f.exists() and (WEB_DIR / rel).exists():
                f = WEB_DIR / rel
            _Range.send_file(self, f); return
        if path in ("/", "/index.html"):
            _Range.send_file(self, WEB_DIR / "index.html", "text/html; charset=utf-8"); return
        if path.startswith("/static/"):
            f = (WEB_DIR / path[8:]).resolve()
            if not str(f).startswith(str(WEB_DIR)):
                self._error("forbidden", 403); return
            _Range.send_file(self, f); return
        m = re.match(r"^/projects/([^/]+)/(out|inputs)/([^/]+)$", path)
        if m:
            pr = _proj.get_project(m.group(1), _cfg.projects_dir())
            f = (pr.path / m.group(2) / m.group(3)).resolve()
            if not str(f).startswith(str(pr.path.resolve())):
                self._error("forbidden", 403); return
            _Range.send_file(self, f); return
        self._error("not found", 404)

    # ---------------------------------------------------------------- API
    def _api(self, method: str, route: str, q: dict[str, str]) -> None:
        cfg = _cfg.load_config()
        parts = route.split("/")
        # ---- config
        if route == "config":
            if method == "GET":
                self._json({**cfg, "_paths": {"config": str(_cfg.config_path()), "models_dir": str(_cfg.models_dir(cfg)),
                                             "projects_dir": str(_cfg.projects_dir(cfg)), "device": _cfg.pick_device(cfg["device"])}})
            elif method == "PUT":
                body = self._read_json()
                cfg.update({k: v for k, v in body.items() if k in _cfg.DEFAULTS})
                cfg = _cfg.save_config(cfg)
                self._json({**cfg, "_paths": {"config": str(_cfg.config_path()), "models_dir": str(_cfg.models_dir(cfg)),
                                             "projects_dir": str(_cfg.projects_dir(cfg)), "device": _cfg.pick_device(cfg["device"])}})
            else:
                self._error("method", 405)
            return
        # ---- models
        if parts[0] == "models":
            mdir = _cfg.models_dir(cfg)
            if len(parts) == 1 and method == "GET":
                self._json({"models_dir": str(mdir), "models": _models.list_models(mdir)}); return
            if method == "POST" and len(parts) == 2:
                body = self._read_json()
                mid = body.get("id") or cfg["model"]
                if parts[1] == "download":
                    job = JOBS.submit("download", None, lambda log: str(_models.download(mid, mdir, log)))
                    self._json(job.to_dict(), 202); return
                if parts[1] == "delete":
                    _models.delete(mid, mdir); self._json({"ok": True}); return
                if parts[1] == "import":
                    new = _models.import_local(Path(body["path"]), body.get("id") or None, mdir); self._json({"ok": True, "id": new}); return
                if parts[1] == "export":
                    dst = _models.export_local(mid, Path(body["path"]), mdir); self._json({"ok": True, "path": str(dst)}); return
            self._error("not found", 404); return
        # ---- filesystem browse
        if route == "fs" and method == "GET":
            self._json(_browse(q.get("path", ""), q.get("kind", ""), cfg)); return
        # ---- jobs
        if parts[0] == "jobs":
            if len(parts) == 1:
                self._json([j.to_dict() for j in JOBS.active()]); return
            job = JOBS.get(parts[1])
            if not job:
                self._error("job not found", 404); return
            self._json(job.to_dict(int(q.get("from", 0)))); return
        # ---- projects
        if parts[0] == "projects":
            pdir = _cfg.projects_dir(cfg)
            if len(parts) == 1:
                if method == "GET":
                    self._json(_proj.list_projects(pdir)); return
                if method == "POST":
                    body = self._read_json()
                    pr = _proj.create_project(body.get("name") or "新しいプロジェクト", cfg, pdir)
                    for k in _proj.INPUT_KINDS:
                        if body.get("inputs", {}).get(k):
                            _proj.set_input_path(pr, k, body["inputs"][k])
                    self._json(pr.summary(), 201); return
            pr = _proj.get_project(parts[1], pdir)
            if len(parts) == 2:
                if method == "GET":
                    self._json(pr.summary()); return
                if method == "PUT":
                    pr.apply_patch(self._read_json())
                    self._json(pr.summary()); return
                if method == "DELETE":
                    _proj.delete_project(pr.slug, pdir); self._json({"ok": True}); return
            if len(parts) == 4 and parts[2] == "upload" and method == "POST":
                kind = parts[3]
                files = [(n, f, d) for n, f, d in self._read_multipart() if f]
                if not files:
                    self._error("file がありません"); return
                dst = _proj.save_upload(pr, kind, files[0][1], files[0][2])
                self._json({"ok": True, "path": str(dst), "project": pr.summary()}); return
            if len(parts) == 3 and method == "POST":
                if parts[2] == "align":
                    if any(j.kind == "align" and j.project == pr.slug for j in JOBS.active()):
                        self._error("このプロジェクトの解析は実行中です", 409); return
                    job = JOBS.submit("align", pr.slug, lambda log: _proj.run_project_align(pr, cfg, log))
                    self._json(job.to_dict(), 202); return
                if parts[2] == "export":
                    job = JOBS.submit("export", pr.slug, lambda log: _proj.run_project_export(pr, cfg, log))
                    self._json(job.to_dict(), 202); return
        self._error("not found", 404)


def _browse(path: str, kind: str, cfg: dict[str, Any]) -> dict[str, Any]:
    exts = {"audio": _proj.AUDIO_EXT, "lyrics": _proj.LYRICS_EXT, "midi": _proj.MIDI_EXT, "dir": ()}.get(kind, None)
    roots = [{"name": "ホーム", "path": str(Path.home())},
             {"name": "デスクトップ", "path": str(Path.home() / "Desktop")},
             {"name": "書類", "path": str(Path.home() / "Documents")},
             {"name": "ダウンロード", "path": str(Path.home() / "Downloads")},
             {"name": "iCloud Drive", "path": str(Path.home() / "Library/Mobile Documents/com~apple~CloudDocs")},
             {"name": "プロジェクト置き場", "path": str(_cfg.projects_dir(cfg))},
             {"name": "モデル保管場所", "path": str(_cfg.models_dir(cfg))}]
    if sys.platform == "darwin" and Path("/Volumes").exists():
        roots += [{"name": f"ボリューム: {v.name}", "path": str(v)} for v in sorted(Path("/Volumes").iterdir()) if v.is_dir()]
    roots = [r for r in roots if Path(r["path"]).exists()]
    for r in cfg.get("recent_dirs", [])[:8]:
        if Path(r).exists() and all(r != x["path"] for x in roots):
            roots.append({"name": Path(r).name or r, "path": r, "recent": True})
    if not path:
        return {"path": "", "parent": None, "dirs": [], "files": [], "roots": roots}
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.is_file():
        p = p.parent
    dirs, files = [], []
    try:
        for c in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if c.name.startswith("."):
                continue
            if c.is_dir():
                dirs.append({"name": c.name, "path": str(c)})
            elif exts is None or (exts and c.suffix.lower() in exts):
                try:
                    files.append({"name": c.name, "path": str(c), "size": c.stat().st_size})
                except OSError:
                    pass
    except PermissionError:
        pass
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "dirs": dirs, "files": files, "roots": roots}


def serve(port: int = 8791, open_browser: bool = True, static_dir: Optional[Path] = None) -> None:
    Handler.static_dir = Path(static_dir).resolve() if static_dir else None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    print(f"utalign app: {url}   (config: {_cfg.config_path()})")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
