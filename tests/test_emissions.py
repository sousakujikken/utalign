"""chunked_emissions の窓幅 / キャッシュキーと、MPS で推論できないときの CPU フォールバック (issue #1).

macOS 15.1 未満の PyTorch MPS は畳み込み出力の各次元を 65,536 以下に制限する。HuBERT の最初の Conv1d
(kernel 10, stride 5) に入れるセグメントは 327,685 サンプル (約 20.48 秒) 以下でなければならない。
"""
from __future__ import annotations

import numpy as np
import pytest

from utalign import config as _cfg
from utalign.aligners import base
from utalign.aligners.base import HOP, SR, MPS_MAX_SEG_SAMPLES, chunked_emissions, num_frames

V = 4


def _fake_run(seen: list[int]):
    def run(seg: np.ndarray) -> np.ndarray:
        seen.append(len(seg))
        return np.zeros((num_frames(len(seg)), V), dtype=np.float32)
    return run


def test_mps_limit_constant_matches_first_conv():
    # (n - 10) // 5 + 1 <= 65536  <=>  n <= 327,685
    assert MPS_MAX_SEG_SAMPLES == 327_685
    assert (MPS_MAX_SEG_SAMPLES - 10) // 5 + 1 == _cfg.MPS_CONV_MAX_OUTPUT
    assert (MPS_MAX_SEG_SAMPLES + 5 - 10) // 5 + 1 > _cfg.MPS_CONV_MAX_OUTPUT


def test_default_window_fits_mps_limit():
    chunk = int(base.CHUNK_SEC * SR) // HOP * HOP
    ov = int(base.OVERLAP_SEC * SR) // HOP * HOP
    assert chunk + 2 * ov <= MPS_MAX_SEG_SAMPLES


@pytest.mark.parametrize("seconds", [0.5, 15.0, 16.0, 20.0, 24.0, 37.3, 200.0])
def test_segments_never_exceed_mps_limit_and_cover_all_frames(seconds):
    wav = np.zeros(int(SR * seconds), dtype=np.float32)
    seen: list[int] = []
    em = chunked_emissions(wav, _fake_run(seen), "test", None)
    assert seen and max(seen) <= MPS_MAX_SEG_SAMPLES
    assert em.shape == (num_frames(len(wav)), V)
    assert not (em == -1e4).any(), "埋まっていないフレームがある"


def test_old_20s_window_would_exceed_limit():
    wav = np.zeros(SR * 60, dtype=np.float32)
    seen: list[int] = []
    chunked_emissions(wav, _fake_run(seen), "test", None, chunk_sec=20.0, overlap_sec=2.0)
    assert max(seen) > MPS_MAX_SEG_SAMPLES


def test_cache_key_includes_window(tmp_path):
    wav = np.zeros(SR * 30, dtype=np.float32)
    chunked_emissions(wav, _fake_run([]), "m", tmp_path, chunk_sec=16.0)
    chunked_emissions(wav, _fake_run([]), "m", tmp_path, chunk_sec=20.0)
    files = sorted(tmp_path.glob("emission_*.npy"))
    assert len(files) == 2
    # 同じ窓なら再計算しない
    seen: list[int] = []
    chunked_emissions(wav, _fake_run(seen), "m", tmp_path, chunk_sec=16.0)
    assert seen == []
    assert len(list(tmp_path.glob("emission_*.npy"))) == 2


@pytest.mark.parametrize("ver,expected", [
    ((14, 6, 1), True), ((15, 0), True), ((15, 0, 1), True),
    ((15, 1), False), ((15, 1, 1), False), ((16, 0), False),
])
def test_mps_conv_limited(ver, expected):
    assert _cfg.mps_conv_limited(ver) is expected


def test_mps_conv_limited_uses_running_os(monkeypatch):
    monkeypatch.setattr(_cfg, "macos_version", lambda: (14, 6, 1))
    assert _cfg.mps_conv_limited() is True
    monkeypatch.setattr(_cfg, "macos_version", lambda: None)   # macOS 以外
    assert _cfg.mps_conv_limited() is False


def test_macos_version_parse(monkeypatch):
    monkeypatch.setattr(_cfg.sys, "platform", "darwin")
    monkeypatch.setattr(_cfg.platform, "mac_ver", lambda: ("14.6.1", ("", "", ""), "arm64"))
    assert _cfg.macos_version() == (14, 6, 1)
    monkeypatch.setattr(_cfg.platform, "mac_ver", lambda: ("", ("", "", ""), ""))
    assert _cfg.macos_version() is None
    monkeypatch.setattr(_cfg.sys, "platform", "linux")
    assert _cfg.macos_version() is None


class _FakeModel:
    def __init__(self):
        self.device = "mps"

    def to(self, device):
        self.device = device
        return self


def test_hf_ctc_falls_back_to_cpu_on_not_implemented():
    from utalign.aligners.hf_ctc import HFCTCAligner
    al = HFCTCAligner.__new__(HFCTCAligner)
    al.model_id = "fake"; al.device = "mps"; al.device_fallback = None
    al._model = _FakeModel(); al._fe = None
    logs: list[str] = []
    al.log = logs.append
    calls: list[str] = []

    def infer(seg):
        calls.append(al.device)
        if al.device == "mps":
            raise NotImplementedError("Output channels > 65536 not supported at the MPS device.")
        return np.zeros((num_frames(len(seg)), V), dtype=np.float32)
    al._load = lambda: (al._model, al._fe)
    al._infer = infer
    wav = np.zeros(SR * 40, dtype=np.float32)
    em = al.compute_emissions(wav, None)
    assert em.shape == (num_frames(len(wav)), V)
    assert calls[0] == "mps" and all(c == "cpu" for c in calls[1:]) and len(calls) >= 3
    assert al.device == "cpu" and al._model.device == "cpu"
    assert al.device_fallback and "NotImplementedError" in al.device_fallback
    assert logs and "CPU" in logs[0]


def test_hf_ctc_does_not_swallow_error_on_cpu():
    from utalign.aligners.hf_ctc import HFCTCAligner
    al = HFCTCAligner.__new__(HFCTCAligner)
    al.model_id = "fake"; al.device = "cpu"; al.device_fallback = None
    al._model = _FakeModel().to("cpu"); al._fe = None
    al._load = lambda: (al._model, al._fe)

    def infer(seg):
        raise NotImplementedError("boom")
    al._infer = infer
    with pytest.raises(NotImplementedError):
        al.compute_emissions(np.zeros(SR * 5, dtype=np.float32), None)
    assert al.device_fallback is None


def test_doctor_reports_macos_and_mps_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("UTALIGN_HOME", str(tmp_path))
    from utalign.cli import doctor_info
    info = doctor_info(probe_device=False)
    assert "macos" in info and "mpsConvLimited" in info
    assert isinstance(info["mpsConvLimited"], bool)
    assert info["mpsConvLimited"] == _cfg.mps_conv_limited()
    if info["macos"] is not None:
        assert tuple(int(x) for x in info["macos"].split(".")) == _cfg.macos_version()
