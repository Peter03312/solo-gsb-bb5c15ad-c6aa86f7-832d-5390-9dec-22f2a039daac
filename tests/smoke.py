"""端到端冒烟测试：合成一对含共同声音事件的录音，验证对齐参数与输出音轨。

用法：python -m tests.smoke
"""
from __future__ import annotations

import io
import sys
import wave

import numpy as np

sys.path.insert(0, ".")

from app.align import align_recordings  # noqa: E402
from app.config import settings  # noqa: E402
from app.errors import AlignError  # noqa: E402
from app.wavio import MonoAudio, read_wav  # noqa: E402


def make_event_signal(duration_s: float, sr: int, seed: int = 7) -> np.ndarray:
    """非周期、含多个显著事件的信号：噪声突发 + 若干窄带瞬态，适合唯一对齐。"""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    x = np.zeros(n, dtype=np.float64)

    # 稀疏事件：带通噪声爆发（约每 2~4s 一个，时长 0.1~0.5s）
    t = 0.5
    while t < duration_s - 0.5:
        length = int((0.1 + rng.random() * 0.4) * sr)
        start = int(t * sr)
        burst = rng.standard_normal(length)
        env = np.hanning(length)
        # 随机中心频率的窄带成分
        f0 = 300 + rng.random() * 3000
        tt = np.arange(length) / sr
        burst = burst * 0.6 + np.sin(2 * np.pi * f0 * tt + rng.uniform(0, 6.28)) * 0.4
        x[start:start + length] += burst * env
        t += 2 + rng.random() * 2

    # 弱底噪
    x += rng.standard_normal(n) * 0.003
    return x.astype(np.float32)


def simulate_recording(base: np.ndarray, base_sr: int,
                       target_sr: int, b_true: float, a_true: float,
                       gain: float, noise_snr_db: float,
                       channels: int, seed: int) -> MonoAudio:
    """按 t参考 = a + b*t待对齐 生成一条设备录音。"""
    rng = np.random.default_rng(seed)
    from app import dsp

    ref_dur = len(base) / base_sr
    # 待对齐设备时长 120s，覆盖部分参考
    ali_dur = 120.0
    n_ali_native = int(ali_dur * target_sr)
    # 待对齐样本 n（秒 c）对应参考时间 a + b*c，从参考信号重采样取值
    c = np.arange(n_ali_native) / target_sr
    ref_pos = a_true + b_true * c
    valid = (ref_pos >= 0) & (ref_pos < ref_dur - 1e-6)
    y = np.zeros(n_ali_native, dtype=np.float32)
    # 参考信号按位置重采样到待对齐采样网格：
    # 输出样本 n（待对齐秒 c=n/target_sr）对应参考位置 (a+b*c)*base_sr，
    # 每个输出样本在输入轴前进 b*base_sr/target_sr 个样本，即 ratio 等于该步长。
    start_pos = a_true * base_sr
    ratio = b_true * base_sr / target_sr
    y = dsp.resample_arbitrary_range(
        base.astype(np.float32), ratio, start_pos, n_ali_native)
    y[~valid] = 0.0

    # 增益差
    y *= gain
    # 加噪
    sig_power = np.mean(y[valid] ** 2) + 1e-12
    noise_power = sig_power / (10 ** (noise_snr_db / 10))
    y += rng.standard_normal(n_ali_native) * np.sqrt(noise_power)

    if channels == 2:
        # 轻微声道差，均值即单声道
        l = y * 0.9
        r = y * 1.1
        y_int = np.empty(2 * n_ali_native)
        y_int[0::2] = l
        y_int[1::2] = r
        pcm = np.clip(y_int, -1, 1)
    else:
        pcm = np.clip(y, -1, 1)

    pcm16 = (pcm * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(target_sr)
        wf.writeframes(pcm16.tobytes())
    audio = read_wav(buf.getvalue(), "recording")
    return audio


def main() -> int:
    ref_sr = 48000
    ali_sr = 44100
    ref = make_event_signal(150.0, ref_sr, seed=1)

    a_true, b_true = 12.5, 1.0006
    ref_audio = MonoAudio(samples=(ref * 0.8).astype(np.float32)
                          - np.mean(ref * 0.8), sample_rate=ref_sr, channels=1)
    rec_audio = simulate_recording(ref, ref_sr, ali_sr, b_true, a_true,
                                   gain=0.35, noise_snr_db=18,
                                   channels=2, seed=11)

    result = align_recordings(ref_audio, rec_audio)

    a_err = abs(result.a - a_true)
    b_err = abs(result.b - b_true)
    print(f"true a={a_true:.4f} b={b_true:.7f}")
    print(f"est  a={result.a:.4f} b={result.b:.7f}  "
          f"(|Δa|={a_err*1000:.1f}ms, |Δb|={b_err:.2e})")
    print(f"overlap ref [{result.overlap_start_ref:.2f},{result.overlap_end_ref:.2f}] "
          f"ali [{result.overlap_start_ali:.2f},{result.overlap_end_ali:.2f}] "
          f"len={result.overlap_seconds:.2f}s segs={len(result.segments)} "
          f"resid={result.residual_rms_ms:.2f}ms peak={result.coarse_peak:.3f}")
    print(f"outputs: sr={result.out_sample_rate} frames={len(result.ref_out)}/"
          f"{len(result.ali_out)} ({len(result.ref_out)/result.out_sample_rate:.2f}s)")

    ok = True
    if a_err > 0.03:
        print(f"FAIL: a 误差 {a_err*1000:.0f}ms 超过 30ms")
        ok = False
    if b_err > 15e-6:
        print(f"FAIL: b 误差 {b_err:.2e} 超过 1.5e-5")
        ok = False
    if result.overlap_seconds < settings.min_overlap_s:
        print("FAIL: 重叠不足 30s")
        ok = False
    if len(result.ref_out) != len(result.ali_out):
        print("FAIL: 两条输出长度不同")
        ok = False

    # 输出音轨内容对齐验证：互相关峰值应在零附近
    from app import dsp
    r = result.ref_out.astype(np.float64)
    s = result.ali_out.astype(np.float64)
    lag, q = dsp.gcc_phat(r[: 16 * result.out_sample_rate],
                          s[: 16 * result.out_sample_rate],
                          int(0.01 * result.out_sample_rate),
                          sr=result.out_sample_rate)
    print(f"输出首 16s PHAT 残留时延: {lag/result.out_sample_rate*1000:.2f}ms q={q:.2f}")
    if abs(lag) > result.out_sample_rate * 0.005:
        print("FAIL: 输出音轨样本级对齐残差超过 5ms")
        ok = False

    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
