"""HTTP API 层测试：上传、错误响应、结果下载。"""
from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from fastapi.testclient import TestClient

from app.main import app
from tests.test_align import event_track, make_device


client = TestClient(app)


def wav_upload_bytes(x: np.ndarray, sr: int, ch: int = 1) -> bytes:
    y = np.clip(x, -1, 1)
    if ch == 2:
        d = np.empty(2 * len(y))
        d[0::2] = y * 0.9
        d[1::2] = y * 1.1
        pcm = (d * 32767).astype("<i2")
    else:
        pcm = (y * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@pytest.fixture
def pair_bytes():
    sr = 48000
    base = event_track(100.0, sr, seed=21)
    rec = make_device(base, sr, 44100, 6.0, 1.0005, 80.0, channels=2, seed=22)
    ref_b = wav_upload_bytes(base * 0.7, sr, ch=1)
    rec_b = wav_upload_bytes(rec.samples, 44100, ch=1)
    return ref_b, rec_b


def test_health():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_align_happy_path(pair_bytes):
    ref_b, rec_b = pair_bytes
    r = client.post(
        "/api/v1/align",
        files={"reference": ("ref.wav", ref_b, "audio/wav"),
               "recording": ("rec.wav", rec_b, "audio/wav")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.999 <= body["mapping"]["b"] <= 1.001
    assert abs(body["mapping"]["a_s"] - 6.0) < 0.05
    assert body["overlap"]["duration_s"] >= 30.0
    assert body["output"]["sample_rate_hz"] == 48000
    assert body["output"]["channels"] == 1

    n = body["output"]["frames"]
    for url in (body["output"]["reference_wav_url"],
                body["output"]["recording_wav_url"]):
        d = client.get(url)
        assert d.status_code == 200
        assert d.headers["content-type"] == "audio/wav"
        with wave.open(io.BytesIO(d.content)) as wf:
            assert wf.getframerate() == 48000
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getnframes() == n


def test_missing_file():
    r = client.post("/api/v1/align",
                    files={"reference": ("r.wav", b"x", "audio/wav")})
    assert r.status_code in (400, 422)
    assert r.json()["error"]["code"] in ("INVALID_REQUEST",)


def test_not_wav():
    r = client.post(
        "/api/v1/align",
        files={"reference": ("r.wav", b"this is definitely not a wav", "audio/wav"),
               "recording": ("x.wav", b"neither is this one at all!!", "audio/wav")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_WAV"


def test_bad_format_8bit(tmp_path):
    sr = 16000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(sr)
        wf.writeframes(np.zeros(sr * 2, dtype=np.uint8).tobytes())
    good = wav_upload_bytes(event_track(60.0, sr, seed=31), sr)
    r = client.post(
        "/api/v1/align",
        files={"reference": ("r.wav", good, "audio/wav"),
               "recording": ("x.wav", buf.getvalue(), "audio/wav")},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_rejection_different_content():
    sr = 16000
    a = wav_upload_bytes(event_track(60.0, sr, seed=41), sr)
    b = wav_upload_bytes(event_track(60.0, sr, seed=42) * 0 +
                         np.random.default_rng(0).standard_normal(sr * 60) * 0.01, sr)
    r = client.post(
        "/api/v1/align",
        files={"reference": ("r.wav", a, "audio/wav"),
               "recording": ("x.wav", b, "audio/wav")},
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] in (
        "NO_COMMON_EVENT", "AMBIGUOUS_ALIGNMENT", "OVERLAP_TOO_SHORT", "SILENCE")
