"""プロジェクト (入力ファイル + 設定 + 出力) の管理と、バックグラウンドジョブ.

projects_dir/<slug>/
  project.json   名前・入力パス・解析オプション・最終実行結果
  inputs/        アップロードされた入力ファイル (パス指定の場合はコピーしない)
  out/           result.json, audio.wav, click.wav, peaks.json, utavista-lyrics-timing.json, ビューア
  work/          16kHz 音声と emission のキャッシュ
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from . import config as _cfg

_SAVE_LOCK = threading.RLock()   # project.json の読み書きを直列化する

INPUT_KINDS = ("audio", "lyrics", "midi")
AUDIO_EXT = (".mp3", ".wav", ".m4a", ".flac", ".aif", ".aiff", ".ogg")
LYRICS_EXT = (".txt", ".md")
MIDI_EXT = (".mid", ".midi")


def slugify(name: str) -> str:
    """フォルダ名兼 URL パス要素. パス区切り・URL 予約文字 (# % ? & + ; = など)・制御文字を _ に置き換える."""
    s = re.sub(r"[\\/:*?\"<>|#%&+;=,\s\x00-\x1f\x7f]+", "_", name.strip())
    s = s.strip("._") or "project"
    return s[:80]


def default_options(cfg: dict[str, Any]) -> dict[str, Any]:
    return {"model": cfg["model"], "device": cfg["device"], "min_rest": cfg["min_rest"],
            "midi_tracks": "", "keep_drums": False, "star": False,
            "export_ruby": cfg.get("export_ruby", True), "export_strip_spaces": cfg.get("export_strip_spaces", False)}


class Project:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.slug = self.path.name
        self.file = self.path / "project.json"
        self.data: dict[str, Any] = {}
        if self.file.exists():
            self.data = json.loads(self.file.read_text(encoding="utf-8"))

    # ---- paths
    @property
    def out(self) -> Path: return self.path / "out"
    @property
    def work(self) -> Path: return self.path / "work"
    @property
    def inputs_dir(self) -> Path: return self.path / "inputs"

    def input_path(self, kind: str) -> Optional[Path]:
        v = (self.data.get("inputs") or {}).get(kind)
        if not v:
            return None
        p = Path(v)
        return p if p.is_absolute() else self.path / p

    def reload(self) -> None:
        with _SAVE_LOCK:
            if self.file.exists():
                self.data = json.loads(self.file.read_text(encoding="utf-8"))

    def save(self) -> None:
        with _SAVE_LOCK:
            self.path.mkdir(parents=True, exist_ok=True)
            self.data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.file.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")

    def update_saved(self, patch: dict[str, Any]) -> None:
        """ディスク上の最新データに patch だけを反映して保存する (ジョブ完了時など、古い data で上書きしないため)."""
        with _SAVE_LOCK:
            self.reload()
            self.data.update(patch)
            self.save()

    def apply_patch(self, body: dict[str, Any]) -> None:
        """PUT の内容 (name / options / readings / inputs) を、再読込〜保存まで同一ロック内で反映する."""
        with _SAVE_LOCK:
            self.reload()
            if "name" in body:
                self.data["name"] = str(body["name"]).strip() or self.slug
            if "options" in body:
                self.data["options"] = {**(self.data.get("options") or {}), **(body["options"] or {})}
            if "readings" in body:
                self.data["readings"] = str(body["readings"])
            for k, v in (body.get("inputs") or {}).items():
                if k in INPUT_KINDS:
                    self.data.setdefault("inputs", {})[k] = str(Path(v).expanduser()) if v else None
            self.save()

    def summary(self) -> dict[str, Any]:
        inputs = {k: (str(self.input_path(k)) if self.input_path(k) else None) for k in INPUT_KINDS}
        exists = {k: bool(inputs[k] and Path(inputs[k]).exists()) for k in INPUT_KINDS}
        result = self.out / "result.json"
        export = self.out / "utavista-lyrics-timing.json"
        d = {"slug": self.slug, "name": self.data.get("name", self.slug), "created": self.data.get("created"),
             "updated": self.data.get("updated"), "inputs": inputs, "inputs_exist": exists,
             "options": self.data.get("options", {}), "readings": self.data.get("readings", ""),
             "has_result": result.exists(), "has_export": export.exists(),
             "last_run": self.data.get("last_run"), "last_export": self.data.get("last_export"),
             "path": str(self.path)}
        return d


def list_projects(pdir: Optional[Path] = None) -> list[dict[str, Any]]:
    pdir = pdir or _cfg.projects_dir()
    out = []
    for p in sorted(pdir.iterdir()) if pdir.exists() else []:
        if (p / "project.json").exists():
            out.append(Project(p).summary())
    out.sort(key=lambda d: d.get("updated") or "", reverse=True)
    return out


def get_project(slug: str, pdir: Optional[Path] = None) -> Project:
    pdir = pdir or _cfg.projects_dir()
    p = pdir / slug
    if not (p / "project.json").exists():
        raise FileNotFoundError(f"project not found: {slug}")
    return Project(p)


def create_project(name: str, cfg: dict[str, Any], pdir: Optional[Path] = None) -> Project:
    pdir = pdir or _cfg.projects_dir()
    base = slugify(name); slug = base; k = 2
    while (pdir / slug).exists():
        slug = f"{base}-{k}"; k += 1
    pr = Project(pdir / slug)
    pr.data = {"name": name.strip() or slug, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "inputs": {},
               "options": default_options(cfg), "readings": ""}
    pr.save()
    return pr


def delete_project(slug: str, pdir: Optional[Path] = None) -> None:
    pr = get_project(slug, pdir)
    shutil.rmtree(pr.path)


def set_input_path(pr: Project, kind: str, path: str) -> None:
    if kind not in INPUT_KINDS:
        raise ValueError(kind)
    with _SAVE_LOCK:
        pr.reload()
        pr.data.setdefault("inputs", {})[kind] = str(Path(path).expanduser()) if path else None
        pr.save()


def save_upload(pr: Project, kind: str, filename: str, data: bytes) -> Path:
    if kind not in INPUT_KINDS:
        raise ValueError(kind)
    pr.inputs_dir.mkdir(parents=True, exist_ok=True)
    name = Path(filename).name or f"{kind}.bin"
    dst = pr.inputs_dir / name
    dst.write_bytes(data)
    with _SAVE_LOCK:
        pr.reload()
        pr.data.setdefault("inputs", {})[kind] = str(Path("inputs") / name)
        pr.save()
    return dst


# ---------------------------------------------------------------- jobs
@dataclass
class Job:
    id: str
    kind: str
    project: Optional[str]
    status: str = "queued"            # queued | running | done | error
    log: list[str] = field(default_factory=list)
    result: Any = None
    error: Optional[str] = None
    started: Optional[float] = None
    finished: Optional[float] = None

    def to_dict(self, log_from: int = 0) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "project": self.project, "status": self.status,
                "log": self.log[log_from:], "log_len": len(self.log), "result": self.result, "error": self.error,
                "started": self.started, "finished": self.finished}


class JobManager:
    """1 本のワーカースレッドで順に実行する (GPU / モデル読み込みの競合を避ける)."""

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._queue: list[tuple[Job, Callable[[Callable[[str], None]], Any]]] = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, kind: str, project: Optional[str], fn: Callable[[Callable[[str], None]], Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, project=project)
        with self._cv:
            self.jobs[job.id] = job
            self._queue.append((job, fn))
            self._cv.notify()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def active(self) -> list[Job]:
        return [j for j in self.jobs.values() if j.status in ("queued", "running")]

    def _worker(self) -> None:
        while True:
            with self._cv:
                while not self._queue:
                    self._cv.wait()
                job, fn = self._queue.pop(0)
            job.status = "running"; job.started = time.time()
            try:
                job.result = fn(lambda s: job.log.append(str(s)))
                job.status = "done"
            except Exception as e:  # noqa: BLE001
                job.error = f"{type(e).__name__}: {e}"
                job.log.append(traceback.format_exc())
                job.status = "error"
            job.finished = time.time()


def run_project_align(pr: Project, cfg: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    from .pipeline import AlignOptions, run_align
    pr.reload()
    for k in INPUT_KINDS:
        p = pr.input_path(k)
        if not p or not p.exists():
            raise FileNotFoundError(f"{k} の入力ファイルがありません: {p}")
    o = {**default_options(cfg), **(pr.data.get("options") or {})}
    readings = None
    if pr.data.get("readings", "").strip():
        readings = pr.path / "readings.override.txt"
        readings.write_text(pr.data["readings"], encoding="utf-8")
    tracks = [int(x) for x in re.split(r"[,\s]+", str(o.get("midi_tracks") or "")) if x.strip().isdigit()] or None
    opt = AlignOptions(model=o["model"], device=o.get("device", "auto"), star=bool(o.get("star")),
                       min_rest=float(o.get("min_rest", 0.2)), midi_tracks=tracks, keep_drums=bool(o.get("keep_drums")),
                       readings=readings, models_dir=_cfg.models_dir(cfg))
    meta = run_align(pr.input_path("audio"), pr.input_path("lyrics"), pr.input_path("midi"), pr.out, pr.work, opt, log)
    last_run = {k: meta.get(k) for k in ("generated", "model", "vocab", "offset", "residual_median",
                                          "residual_p90", "residual_over150", "n_mora_skipped", "n_note_skipped",
                                          "repeated_lines", "repeat_mismatches", "warnings", "elapsed")}
    pr.update_saved({"last_run": last_run, "last_export": None})
    return last_run


def run_project_export(pr: Project, cfg: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
    from .pipeline import run_export, run_utavista_validator
    pr.reload()
    o = {**default_options(cfg), **(pr.data.get("options") or {})}
    result = pr.out / "result.json"
    if not result.exists():
        raise FileNotFoundError("先に解析を実行してください")
    dst = pr.out / "utavista-lyrics-timing.json"
    info = run_export(result, dst, ruby=bool(o.get("export_ruby", True)), strip_spaces=bool(o.get("export_strip_spaces")), log=log)
    info["validator"] = None
    udir = (cfg.get("utavista_dir") or "").strip()
    if udir and Path(udir).expanduser().exists():
        import soundfile as sf
        dur_ms = int(sf.info(str(pr.out / "audio.wav")).duration * 1000)
        info["validator"] = run_utavista_validator(Path(udir).expanduser(), dst, pr.input_path("lyrics"), dur_ms, log)
    info["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    pr.update_saved({"last_export": info})
    return info
