"""utalign.audio の回帰テスト (実 M4A / AAC / ALAC のデコード、サンドボックス外 macOS).

tests/fixtures/ の音源はすべて同じ元音 (2 秒 44.1kHz ステレオ、L=440Hz / R=660Hz、振幅 0.3) を
エンコードしたもの (エンコーダは afconvert と、独立実装として ffmpeg。実行時に ffmpeg は使わない)。
AAC は非可逆なので、長さ・支配周波数・実効値で判定する。
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from utalign import audio

FIXTURES = Path(__file__).parent / "fixtures"
SRC_SR = 44100
SRC_SEC = 2.0
SRC_AMP = 0.3
SRC_FREQ = (440.0, 660.0)

LOSSY = [
    "tone_aac_afconvert.m4a",       # afconvert -f m4af -d aac
    "tone_aac_ffmpeg.m4a",          # ffmpeg -c:a aac (エンコードのみ)
    "tone_aac_ffmpeg.aac",          # ffmpeg -c:a aac -f adts
    "tone_aac_ffmpeg_48k_mono.m4a", # ffmpeg -c:a aac -ar 48000 -ac 1
]
LOSSLESS = ["tone_alac_afconvert.m4a"]  # afconvert -f m4af -d alac

darwin_only = pytest.mark.skipif(
    sys.platform != "darwin" or not audio._afconvert_available(), reason="afconvert は macOS のみ")


def _reference(sr: int, channels: int) -> np.ndarray:
    t = np.arange(int(sr * SRC_SEC)) / sr
    chans = [SRC_AMP * np.sin(2 * np.pi * f * t) for f in SRC_FREQ[:channels]]
    if channels == 1:  # ffmpeg -ac 1 は L/R 平均
        chans = [0.5 * (SRC_AMP * np.sin(2 * np.pi * SRC_FREQ[0] * t)
                        + SRC_AMP * np.sin(2 * np.pi * SRC_FREQ[1] * t))]
    return np.stack(chans, 1).astype(np.float32)


def _dominant_hz(x: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / sr)[int(np.argmax(spec))])


def _check_lossy(y: np.ndarray, sr: int, expect_sr: int, expect_ch: int) -> None:
    assert sr == expect_sr
    assert y.ndim == 2 and y.shape[1] == expect_ch
    assert y.dtype == np.float32
    n = int(expect_sr * SRC_SEC)
    # エンコーダの priming / padding 分を許容 (AAC のフレーム 1024 × 数個)
    assert abs(len(y) - n) <= 4096, f"length {len(y)} vs {n}"
    mid = y[n // 4: n * 3 // 4]
    rms = np.sqrt((mid ** 2).mean(axis=0))
    if expect_ch == 2:
        for ch, f in enumerate(SRC_FREQ):
            assert abs(_dominant_hz(mid[:, ch], sr) - f) < 5, f"ch{ch}"
            assert abs(rms[ch] - SRC_AMP / np.sqrt(2)) < 0.03
    else:
        hz = _dominant_hz(mid[:, 0], sr)
        assert min(abs(hz - f) for f in SRC_FREQ) < 5


@darwin_only
@pytest.mark.parametrize("name", LOSSY)
def test_decode_lossy_aac(name: str) -> None:
    y, sr = audio.decode(FIXTURES / name)
    if "48k_mono" in name:
        _check_lossy(y, sr, 48000, 1)
    else:
        _check_lossy(y, sr, SRC_SR, 2)


@darwin_only
@pytest.mark.parametrize("name", LOSSLESS)
def test_decode_lossless_alac_is_exact(name: str) -> None:
    y, sr = audio.decode(FIXTURES / name)
    ref = _reference(SRC_SR, 2)
    assert sr == SRC_SR and y.shape == ref.shape
    # 元音は PCM_16 で書いたので量子化誤差 (1/32768) まで
    assert np.abs(y - ref).max() < 1.5 / 32768


@darwin_only
def test_load_audio_16k_and_duration() -> None:
    p = FIXTURES / "tone_aac_afconvert.m4a"
    y = audio.load_audio_16k(p)
    assert y.ndim == 1 and abs(len(y) - int(audio.SR_ANALYSIS * SRC_SEC)) <= 2048
    assert abs(audio.duration_ms(p) - int(SRC_SEC * 1000)) <= 100


@darwin_only
def test_write_playback_wav(tmp_path: Path) -> None:
    import soundfile as sf
    dst = tmp_path / "playback.wav"
    audio.write_playback_wav(FIXTURES / "tone_aac_ffmpeg_48k_mono.m4a", dst)
    y, sr = sf.read(str(dst), dtype="float32", always_2d=True)
    assert sr == audio.SR_PLAYBACK and y.shape[1] == 2
    assert abs(len(y) - int(audio.SR_PLAYBACK * SRC_SEC)) <= 4096


@darwin_only
def test_afconvert_decodes_real_m4a_directly() -> None:
    """afconvert 経路そのもので実 M4A が読めること."""
    for name in LOSSY[:2] + LOSSLESS:
        y, sr = audio._read_afconvert(FIXTURES / name)
        assert sr == SRC_SR and y.shape[1] == 2


def _fake_afconvert(tmp_path: Path, stderr: str) -> str:
    exe = tmp_path / "afconvert"
    exe.write_text(f"#!/bin/sh\necho \"{stderr}\" >&2\nexit 1\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def test_error_message_explains_codec_unavailable(tmp_path: Path, monkeypatch) -> None:
    """afconvert が 'cfmt'/'fmt?' で落ちたら、CoreAudio XPC (サンドボックス) 制約として説明する."""
    monkeypatch.setattr(audio, "AFCONVERT", _fake_afconvert(
        tmp_path, "Error: ExtAudioFileSetProperty ('cfmt') failed ('fmt?')"))
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(audio.AudioDecodeError) as ei:
        audio.decode(FIXTURES / "tone_aac_ffmpeg.m4a")
    msg = str(ei.value)
    assert "AudioComponentRegistrar" in msg and "サンドボックス" in msg and "soundfile" in msg


def test_error_message_without_afconvert(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(audio.AudioDecodeError) as ei:
        audio.decode(FIXTURES / "tone_aac_afconvert.m4a")
    assert "afconvert" in str(ei.value)


def test_missing_file() -> None:
    with pytest.raises(audio.AudioDecodeError):
        audio.decode(FIXTURES / "nope.m4a")


@darwin_only
@pytest.mark.skipif(not Path("/usr/bin/sandbox-exec").exists(), reason="sandbox-exec が無い")
def test_sandbox_blocks_afconvert_codec(tmp_path: Path) -> None:
    """レビュー指摘の再現: mach-lookup を遮断した実行サンドボックス内では afconvert の AAC デコーダが
    CoreAudio XPC に到達できず 'cfmt'/'fmt?' で失敗する。decode() はそれを制約として説明する.
    (サンドボックス外では同じファイルが test_decode_lossy_aac で読める)"""
    profile = tmp_path / "deny-mach.sb"
    profile.write_text(
        "(version 1)\n(deny default)\n(allow process*)\n(allow sysctl-read)\n(allow file-read*)\n"
        f"(allow file-write* (subpath \"{tmp_path.resolve()}\") (subpath \"/private/tmp\") (subpath \"/private/var/folders\"))\n")
    src = FIXTURES / "tone_aac_afconvert.m4a"
    r = subprocess.run(["/usr/bin/sandbox-exec", "-f", str(profile), audio.AFCONVERT,
                        "-f", "WAVE", "-d", "LEF32", str(src), str(tmp_path / "o.wav")],
                       capture_output=True, text=True)
    if r.returncode == 0:
        pytest.skip("この環境の sandbox-exec では afconvert の AAC デコードが遮断されない")
    assert "cfmt" in r.stderr and "fmt?" in r.stderr, r.stderr
    code = (
        "import json; from pathlib import Path; from utalign import audio\n"
        "try:\n"
        f"    audio.decode(Path({str(src)!r})); print(json.dumps({{'ok': True}}))\n"
        "except audio.AudioDecodeError as e:\n"
        "    print(json.dumps({'ok': False, 'msg': str(e)}))\n")
    env = dict(os.environ, TMPDIR=str(tmp_path), PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    r2 = subprocess.run(["/usr/bin/sandbox-exec", "-f", str(profile), sys.executable, "-c", code],
                        capture_output=True, text=True, env=env)
    assert r2.returncode == 0, r2.stderr[-2000:]
    res = json.loads(r2.stdout.strip().splitlines()[-1])
    assert res["ok"] is False
    assert "AudioComponentRegistrar" in res["msg"] and "fmt?" in res["msg"]
