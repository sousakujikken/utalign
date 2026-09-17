"""音声デコード (ffmpeg 不使用).

デコーダの優先順:
  1. soundfile (libsndfile) — WAV / AIFF / CAF / FLAC / OGG (Vorbis) / MP3 を追加依存なしで読む
  2. macOS の afconvert (CoreAudio) — M4A / AAC / ALAC など libsndfile が読めない形式を
     一時 WAV に変換してから soundfile で読む (macOS 以外では利用不可)

プロジェクト方針により ffmpeg は使わない (ライセンス条件が平易でないため。用途も macOS に限定)。

afconvert の AAC / ALAC デコーダは CoreAudio の XPC サービス (AudioComponentRegistrar) への
mach-lookup を必要とする。これを遮断する実行サンドボックス内 (sandbox-exec の deny default 等) では
`ExtAudioFileSetProperty ('cfmt') failed ('fmt?')` で失敗する。これは CoreAudio XPC の制約であり、
サンドボックス外の macOS では M4A / AAC / ALAC を正常にデコードできる (tests/test_audio.py)。

解析用 16kHz モノラル (load_audio_16k) と再生用 44.1kHz ステレオ WAV (write_playback_wav) は
同じ decode() を通るため、両者の時刻は一致する。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SR_ANALYSIS = 16000
SR_PLAYBACK = 44100

AFCONVERT = "/usr/bin/afconvert"

# afconvert がコーデックの XPC サービス (AudioComponentRegistrar) に到達できないときのエラー断片
_AFCONVERT_CODEC_UNAVAILABLE = ("'cfmt'", "'fmt?'", "(cfmt)", "(fmt?)")
AFCONVERT_SANDBOX_HINT = (
    "afconvert の AAC/ALAC デコーダが CoreAudio の XPC サービス AudioComponentRegistrar に到達できません。"
    "mach サービスを遮断する実行サンドボックス内 (sandbox-exec 等) では利用できないため、"
    "サンドボックス外の macOS で実行してください")


class AudioDecodeError(RuntimeError):
    pass


def _read_soundfile(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return y, int(sr)


def _afconvert_path() -> str | None:
    if sys.platform != "darwin":
        return None
    if Path(AFCONVERT).exists():
        return AFCONVERT
    return shutil.which("afconvert")


def _afconvert_available() -> bool:
    return _afconvert_path() is not None


def _read_afconvert(path: Path) -> tuple[np.ndarray, int]:
    exe = _afconvert_path() or AFCONVERT
    with tempfile.TemporaryDirectory(prefix="utalign-afconvert-") as td:
        tmp = Path(td) / "decoded.wav"
        r = subprocess.run([exe, "-f", "WAVE", "-d", "LEF32", str(path), str(tmp)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not tmp.exists():
            msg = (r.stderr or r.stdout).strip()
            if any(k in msg for k in _AFCONVERT_CODEC_UNAVAILABLE):
                msg += f" ({AFCONVERT_SANDBOX_HINT})"
            raise AudioDecodeError(f"afconvert が失敗しました: {msg}")
        return _read_soundfile(tmp)


def decode(path: Path) -> tuple[np.ndarray, int]:
    """音声ファイル -> (float32 [n, ch], sample_rate)."""
    path = Path(path)
    if not path.exists():
        raise AudioDecodeError(f"音声ファイルがありません: {path}")
    try:
        return _read_soundfile(path)
    except Exception as e:  # noqa: BLE001 — libsndfile 非対応形式は afconvert に回す
        sf_err = e
    if _afconvert_available():
        try:
            return _read_afconvert(path)
        except AudioDecodeError as e:
            raise AudioDecodeError(
                f"デコードできません ({path.suffix or '拡張子なし'}): soundfile: {sf_err} / {e}") from e
    raise AudioDecodeError(
        f"デコードできません ({path.suffix or '拡張子なし'}): {sf_err}. "
        "WAV / FLAC / OGG / MP3 は libsndfile で、M4A / AAC / ALAC は macOS の afconvert で読めます")


def to_mono(y: np.ndarray) -> np.ndarray:
    return y.mean(axis=1).astype(np.float32) if y.ndim == 2 else y.astype(np.float32)


def resample(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return y.astype(np.float32)
    import soxr
    return soxr.resample(y, sr, target_sr, quality="HQ").astype(np.float32)


def load_audio_16k(path: Path) -> np.ndarray:
    y, sr = decode(path)
    return resample(to_mono(y), sr, SR_ANALYSIS)


def duration_ms(path: Path) -> int:
    y, sr = decode(path)
    return int(round(len(y) / sr * 1000))


def write_playback_wav(src: Path, dst: Path) -> None:
    """再生用 WAV (44.1kHz ステレオ float32) を生成する. 解析と同じデコーダなので時刻が一致する."""
    import soundfile as sf
    y, sr = decode(src)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    elif y.shape[1] > 2:
        y = y[:, :2]
    y = resample(y, sr, SR_PLAYBACK)
    sf.write(str(dst), np.clip(y, -1, 1), SR_PLAYBACK, subtype="FLOAT")
