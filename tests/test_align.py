"""对齐管线与 DSP 的边界条件测试。"""
from __future__ import annotations

import io
import struct
import wave

import numpy as np
import pytest

from app import dsp
from app.align import align_recordings
from app.config import settings
from app.errors import AlignError, InvalidWav
from app.wavio import MonoAudio, read_wav, write_wav_16bit


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def event_track(duration_s: float, sr: int, seed: int = 3,
                event_every: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    x = np.zeros(n, dtype=np.float64)
    t = 1.0
    while t < duration_s - 1:
        length = int((0.08 + rng.random() * 0.3) * sr)
        start = int(t * sr)
        tt = np.arange(length) / sr
        burst = (0.6 * rng.standard_normal(length)
                 + 0.4 * np.sin(2 * np.pi * (400 + rng.random() * 3000) * tt))
        x[start:start + length] += burst * np.hanning(length)
        t += event_every * (0.7 + 0.6 * rng.random())
    x += rng.standard_normal(n) * 0.002
    return x.astype(np.float32)


def make_device(base: np.ndarray, base_sr: int, sr: int,
                a: float, b: float, duration: float,
                gain: float = 0.5, snr_db: float = 25.0,
                channels: int = 1, seed: int = 5) -> MonoAudio:
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    start_pos = a * base_sr
    ratio = b * base_sr / sr
    y = dsp.resample_arbitrary_range(base.astype(np.float32), ratio, start_pos, n)
    valid = ((np.arange(n) / sr * b + a) >= 0)
    valid &= ((np.arange(n) / sr * b + a) < len(base) / base_sr)
    y[~valid] = 0.0
    y *= gain
    power = np.mean(y[valid] ** 2) + 1e-12
    y += rng.standard_normal(n) * np.sqrt(power / 10 ** (snr_db / 10))
    y = np.clip(y, -1, 1)
    if channels == 2:
        interleaved = np.empty(2 * n)
        interleaved[0::2] = y * 0.9
        interleaved[1::2] = y * 1.1
        pcm = (interleaved * 32767).astype("<i2")
    else:
        pcm = (y * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return read_wav(buf.getvalue(), "recording")


@pytest.fixture
def reference():
    sr = 48000
    x = event_track(120.0, sr)
    return MonoAudio(samples=x, sample_rate=sr, channels=1)


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------

def test_align_typical(reference):
    a_true, b_true = 8.0, 1.0004
    rec = make_device(reference.samples, 48000, 44100, a_true, b_true, 90.0)
    r = align_recordings(reference, rec)
    assert abs(r.a - a_true) < 0.03
    assert abs(r.b - b_true) < 2e-5
    assert r.overlap_seconds >= 30.0
    assert len(r.ref_out) == len(r.ali_out)
    assert r.out_sample_rate == 48000


def test_negative_offset_and_stereo(reference):
    """待对齐录音开始更早（a<0），且为立体声 16k。"""
    a_true, b_true = -10.0, 0.9997
    rec = make_device(reference.samples, 48000, 16000, a_true, b_true, 110.0,
                      channels=2)
    r = align_recordings(reference, rec)
    assert abs(r.a - a_true) < 0.03
    assert abs(r.b - b_true) < 2e-5
    assert r.overlap_start_ali >= 0.0
    assert r.overlap_start_ref >= -1e-6
    assert len(r.ref_out) == len(r.ali_out)


def test_b_equal_one(reference):
    rec = make_device(reference.samples, 48000, 32000, 5.0, 1.0, 80.0)
    r = align_recordings(reference, rec)
    assert abs(r.b - 1.0) < 5e-6
    assert abs(r.a - 5.0) < 0.03


# ---------------------------------------------------------------------------
# 拒绝路径
# ---------------------------------------------------------------------------

def test_overlap_too_short(reference):
    # 待对齐录音只有 25s 落在参考内
    rec = make_device(reference.samples, 48000, 48000, 90.0, 1.0, 30.0)
    with pytest.raises(AlignError) as ei:
        align_recordings(reference, rec)
    assert ei.value.code == "OVERLAP_TOO_SHORT"


def test_b_out_of_range(reference):
    # 真实漂移 0.998（-0.2%）超出 0.999~1.001
    rec = make_device(reference.samples, 48000, 48000, 2.0, 0.998, 90.0)
    with pytest.raises(AlignError) as ei:
        align_recordings(reference, rec)
    assert ei.value.code == "B_OUT_OF_RANGE"


def test_silence_rejected():
    sr = 16000
    ref = MonoAudio(samples=np.zeros(int(60 * sr), dtype=np.float32),
                    sample_rate=sr, channels=1)
    rec = MonoAudio(samples=event_track(60.0, sr, seed=9),
                    sample_rate=sr, channels=1)
    with pytest.raises(AlignError) as ei:
        align_recordings(ref, rec)
    assert ei.value.code == "SILENCE"


def test_different_content_rejected(reference):
    sr = 48000
    other = event_track(90.0, sr, seed=99)  # 独立生成、无共同事件
    rec = MonoAudio(samples=other, sample_rate=sr, channels=1)
    with pytest.raises(AlignError):
        align_recordings(reference, rec)


def test_periodic_content_rejected():
    """两台设备录到同一节拍器（0.5s 周期、80s 重叠）：滞后不唯一，必须拒绝。"""
    sr = 16000
    rng = np.random.default_rng(1)

    def metronome(dur, phase, gain):
        n = int(dur * sr)
        x = np.zeros(n, dtype=np.float64)
        for t0 in np.arange(0, dur, 0.5) + phase:
            i = int(t0 * sr)
            length = int(0.02 * sr)
            if 0 <= i < n:
                tt = np.arange(length) / sr
                x[i:i + length] += (np.sin(2 * np.pi * 1200 * tt)
                                    * np.hanning(length) * gain)
        x += rng.standard_normal(n) * 0.002
        return MonoAudio(samples=x.astype(np.float32), sample_rate=sr, channels=1)

    # 两条时长相同（90s），真实偏移 4.32s，重叠超过 80s，粗对齐必落在重叠区内
    ref = metronome(90.0, 0.0, 0.5)
    rec = metronome(90.0, -4.32, 0.2)
    with pytest.raises(AlignError) as ei:
        align_recordings(ref, rec)
    assert ei.value.code in ("AMBIGUOUS_ALIGNMENT",)


def test_tone_no_common_event(reference):
    """纯正弦（与参考无共同事件）也应被拒绝。"""
    sr = 16000
    t = np.arange(int(60 * sr)) / sr
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)
            + np.random.default_rng(1).standard_normal(len(t)) * 0.001)
    rec = MonoAudio(samples=tone.astype(np.float32),
                    sample_rate=sr, channels=1)
    with pytest.raises(AlignError) as ei:
        align_recordings(reference, rec)
    assert ei.value.code in ("AMBIGUOUS_ALIGNMENT", "NO_COMMON_EVENT",
                             "OVERLAP_TOO_SHORT")


# ---------------------------------------------------------------------------
# WAV 校验
# ---------------------------------------------------------------------------

def wav_bytes(sr=16000, ch=1, sw=2, n=16000, comp=False, data=None,
              framerate=None, extensible=False):
    if data is None:
        data = (np.random.default_rng(0).standard_normal(n * ch) * 10000).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(sw)
        wf.setframerate(framerate or sr)
        wf.writeframes(data)
    raw = bytearray(buf.getvalue())
    if comp:  # 把格式标记从 PCM(1) 改成 float(3)
        raw[20:22] = (3).to_bytes(2, "little")
    if extensible:
        # 构造 WAVE_FORMAT_EXTENSIBLE + PCM 子类型 GUID（fmt 扩展为 40 字节）
        rate = framerate or sr
        block = ch * sw
        fmt = struct.pack("<HHIIHHH", 0xFFFE, ch, rate, rate * block, block, sw * 8, 22)
        fmt += struct.pack("<H", sw * 8) + struct.pack("<I", 0)
        fmt += bytes.fromhex("0100000000001000800000aa00389b71")
        out = bytes(raw[:12])
        out += b"fmt " + struct.pack("<I", len(fmt)) + fmt
        # 找到原 data 块并追加
        idx = bytes(raw).find(b"data")
        out += bytes(raw)[idx:]
        raw = bytearray(out)
        # 修正 RIFF 长度
        raw[4:8] = struct.pack("<I", len(raw) - 8)
    return bytes(raw)


def test_valid_wav_stereo_48k():
    a = read_wav(wav_bytes(48000, ch=2, n=48000), "recording")
    assert a.sample_rate == 48000 and a.channels == 2
    assert abs(a.duration - 1.0) < 1e-6


def test_extensible_pcm_wav_accepted():
    """手机/专业录音机常见的 WAVE_FORMAT_EXTENSIBLE + PCM 子类型必须接受。"""
    a = read_wav(wav_bytes(44100, ch=2, n=44100, extensible=True), "recording")
    assert a.sample_rate == 44100 and a.channels == 2
    assert abs(a.duration - 1.0) < 1e-6


def test_not_wav_rejected():
    with pytest.raises(InvalidWav) as ei:
        read_wav(b"not a wav file" * 100, "reference")
    assert ei.value.code == "INVALID_WAV"


def test_24bit_rejected():
    data = b"\x00" * (16000 * 3)
    with pytest.raises(InvalidWav) as ei:
        read_wav(wav_bytes(sw=3, data=data), "reference")
    assert ei.value.code == "UNSUPPORTED_FORMAT"


def test_float_wav_rejected():
    with pytest.raises(InvalidWav):
        read_wav(wav_bytes(comp=True), "reference")


def test_samplerate_out_of_range():
    with pytest.raises(InvalidWav):
        read_wav(wav_bytes(framerate=8000), "reference")
    with pytest.raises(InvalidWav):
        read_wav(wav_bytes(framerate=96000), "reference")


def test_duration_out_of_range():
    with pytest.raises(InvalidWav):
        read_wav(wav_bytes(sr=16000, n=8000), "reference")   # 0.5s
    with pytest.raises(InvalidWav):
        read_wav(wav_bytes(sr=16000, n=16000 * 700), "reference")  # 700s


def test_truncated_wav_rejected():
    good = wav_bytes(n=16000)
    with pytest.raises(InvalidWav) as ei:
        read_wav(good[:-2000], "reference")
    assert ei.value.code in ("INVALID_WAV",)


# ---------------------------------------------------------------------------
# DSP 原语
# ---------------------------------------------------------------------------

def test_resample_poly_ratio_accuracy():
    sr = 48000
    t = np.arange(int(5 * sr)) / sr
    x = np.sin(2 * np.pi * 1500 * t).astype(np.float32)
    y = dsp.resample_arbitrary(x, 48000 / 44100)
    assert abs(len(y) / 44100 - 5.0) < 0.01
    amp = np.ptp(y[int(1 * 44100): int(2 * 44100)]) / 2
    assert amp > 0.99


def test_resample_range_matches_full():
    sr = 48000
    x = event_track(10.0, sr, seed=11)
    ratio = 48000 / 44100
    full = dsp.resample_arbitrary(x, ratio)
    # start_pos 是输入轴坐标：输出 j 对应输入 start_pos + j*ratio
    seg = dsp.resample_arbitrary_range(x, ratio, 2.0 * 48000, 44100)
    i0 = int(round(2.0 * 48000 / ratio))
    inner = slice(100, 44100 - 100)
    err = np.max(np.abs(full[i0: i0 + 44100][inner] - seg[inner]))
    assert err < 1e-4


def test_gcc_phat_finds_lag():
    sr = 16000
    x = event_track(4.0, sr, seed=4)
    shift = 1234
    # 事件在 y 中提前 shift 个样本 -> 测得 lag=+shift
    y = x[shift:]
    x = x[:len(y)]
    lag, q = dsp.gcc_phat(x, y, 4000, sr=sr)
    assert abs(lag - shift) < 1.0
    assert q > 1.3


def test_write_wav_roundtrip(tmp_path):
    sr = 16000
    x = event_track(2.0, sr, seed=8)
    p = tmp_path / "o.wav"
    write_wav_16bit(str(p), x, sr)
    a = read_wav(p.read_bytes(), "reference")
    assert a.sample_rate == sr
    assert len(a.samples) == len(x)
