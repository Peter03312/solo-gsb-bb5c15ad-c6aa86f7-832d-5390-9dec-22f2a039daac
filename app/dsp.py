"""音频信号处理原语。

- resample_arbitrary：任意比率带限（窗化 sinc）重采样
- envelope_normalized_xcorr：粗时延（低采样率能量包络的归一化互相关）
- gcc_phat：分段精细时延（PHAT 加权广义互相关，抛物线插值）
- silence check：整段 RMS + 低能量帧比例
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly

from .config import settings


# ---------------------------------------------------------------------------
# 重采样
# ---------------------------------------------------------------------------

def resample_to_rate(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """整数/有理数比率重采样（用于分析域统一到 analysis_rate）。"""
    import math as _m

    g = _m.gcd(int(src_rate), int(dst_rate))
    up = int(dst_rate) // g
    down = int(src_rate) // g
    return resample_poly(x, up, down).astype(np.float32)


def resample_arbitrary(x: np.ndarray, ratio: float, half_taps: int = 16) -> np.ndarray:
    """把信号按输出/输入样本间隔比 ratio 做带限重采样。

    ratio = dt_out / dt_in，即输出样本 n 对应输入位置 n*ratio。
    反镜像截止频率取 min(1, 1/ratio) 并留 1.5% 保护带。
    """
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"非法重采样比率: {ratio}")
    if abs(ratio - 1.0) < 1e-12:
        return x.astype(np.float32, copy=True)

    cutoff = 0.4925 * min(1.0, 1.0 / ratio)
    n_out = int(math.floor((len(x) - 1) / ratio)) + 1
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)

    padded = np.concatenate([np.zeros(half_taps, dtype=np.float64),
                             x.astype(np.float64, copy=False),
                             np.zeros(half_taps, dtype=np.float64)])
    out = np.empty(n_out, dtype=np.float32)
    win = np.hanning(2 * half_taps + 1)

    # 分块处理，控制临时内存
    chunk = 65536
    for start in range(0, n_out, chunk):
        end = min(start + chunk, n_out)
        n = np.arange(start, end, dtype=np.float64)
        pos = n * ratio                       # 在原信号中的连续位置
        k0 = np.floor(pos).astype(np.int64)
        frac = pos - k0                       # 小数偏移
        kk = np.arange(-half_taps, half_taps + 1)[None, :]
        d = frac[:, None] - kk                # 相对 sinc 中心
        kernel = np.sinc(d) * win[None, :]
        # 抗镜像：sinc 低通，截止 cutoff（相对输入奈奎斯特）
        if cutoff < 0.4999:
            kernel *= 2.0 * cutoff
        kernel /= kernel.sum(axis=1, keepdims=True)
        idx = (k0[:, None] + half_taps + kk)
        seg = padded[idx]
        out[start:end] = (seg * kernel).sum(axis=1).astype(np.float32)
    return out


def resample_arbitrary_range(x: np.ndarray, ratio: float,
                             start_pos: float, n_out: int,
                             half_taps: int = 16) -> np.ndarray:
    """带限重采样，只生成输出区间 [0, n_out)。

    输出样本 j 对应输入连续位置 start_pos + j*ratio。
    区间超出输入边界的样本补零。
    """
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"非法重采样比率: {ratio}")
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)

    cutoff = 0.4925 * min(1.0, 1.0 / ratio)
    padded = np.concatenate([np.zeros(half_taps, dtype=np.float64),
                             x.astype(np.float64, copy=False),
                             np.zeros(half_taps, dtype=np.float64)])
    out = np.empty(n_out, dtype=np.float32)
    win = np.hanning(2 * half_taps + 1)

    chunk = 65536
    for start in range(0, n_out, chunk):
        end = min(start + chunk, n_out)
        n = np.arange(start, end, dtype=np.float64)
        pos = start_pos + n * ratio
        k0 = np.floor(pos).astype(np.int64)
        frac = pos - k0
        kk = np.arange(-half_taps, half_taps + 1)[None, :]
        d = frac[:, None] - kk
        kernel = np.sinc(d) * win[None, :]
        if cutoff < 0.4999:
            kernel *= 2.0 * cutoff
        kernel /= kernel.sum(axis=1, keepdims=True)
        idx = (k0[:, None] + half_taps + kk)
        valid = (idx >= 0) & (idx < len(padded))
        seg = np.where(valid, padded[np.clip(idx, 0, len(padded) - 1)], 0.0)
        out[start:end] = (seg * kernel).sum(axis=1).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 对齐唯一性：事件片段相似度矩阵
# ---------------------------------------------------------------------------

def _event_positions(x: np.ndarray, sr: int, frame_s: float = 0.02,
                     min_gap_s: float = 0.25) -> list[int]:
    fl = max(64, int(round(frame_s * sr)))
    n = len(x) // fl
    if n == 0:
        return []
    e = np.sum(x[: n * fl].reshape(n, fl).astype(np.float64) ** 2, axis=1)
    thresh = np.median(e) * 20.0 + np.mean(e) * 0.05
    out: list[int] = []
    gap = int(min_gap_s * sr)
    for i in np.where(e > thresh)[0]:
        p = i * fl
        if not out or p - out[-1] > gap:
            out.append(int(p))
    return out


def event_snippets(x: np.ndarray, positions: list[int], sr: int,
                   pre_s: float = 0.01, post_s: float = 0.14) -> np.ndarray:
    """以事件点为基准截取归一化短波形（含少量前置，覆盖冲击起始）。"""
    a, b = int(pre_s * sr), int(post_s * sr)
    rows = []
    for p in positions:
        if p - a >= 0 and p + b < len(x):
            s = x[p - a: p + b].astype(np.float64)
            s -= s.mean()
            nrm = np.linalg.norm(s)
            if nrm > 1e-9:
                rows.append(s / nrm)
    return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 1))


def event_match_ratios(ref_a: np.ndarray, ali_a: np.ndarray, sr: int,
                       a: float,
                       offsets_s: tuple[float, ...] = (-2.0, -1.0, -0.5,
                                                      0.5, 1.0, 2.0),
                       pairing_tol_s: float = 0.15,
                       max_events: int = 160) -> tuple[list[float], int]:
    """在已知粗映射 t参考 ≈ a + t待对齐 下评估事件匹配的唯一性。

    对每个待对齐事件：按映射找到参考侧对应事件（真实匹配），再在 ±0.5/1/2s
    等候选周期偏移处找“错配”。比较真实匹配与最强错配的归一化波形相似度：

    - 节拍器/蜂鸣等雷同周期内容：错配和真匹配几乎一样强 -> 比值 ≈ 1；
    - 独立声音事件：错配接近 0 -> 比值远大于 1。

    只有在某个候选偏移处确实存在事件（时间差 < pairing_tol_s）时才计入，
    因此稀疏事件录音可能只有少量有效观测，样本不足时交由其它门限把关。

    返回 (各事件 真匹配/最强错配 比值列表, 有效观测数)。
    """
    pr = _event_positions(ref_a, sr)
    pa = _event_positions(ali_a, sr)
    if len(pa) > max_events:
        sel = np.linspace(0, len(pa) - 1, max_events).round().astype(int)
        pa = [pa[i] for i in sorted(set(sel))]
    if len(pr) > max_events:
        sel = np.linspace(0, len(pr) - 1, max_events).round().astype(int)
        pr = [pr[i] for i in sorted(set(sel))]

    # event_snippets 会丢弃边界不完整或零能量的位置，时间轴必须与矩阵行对齐
    def _snippets_and_times(sig, positions):
        a, b = int(0.01 * sr), int(0.14 * sr)
        rows, kept = [], []
        for p in positions:
            if p - a >= 0 and p + b < len(sig):
                s = sig[p - a: p + b].astype(np.float64)
                s -= s.mean()
                nrm = np.linalg.norm(s)
                if nrm > 1e-9:
                    rows.append(s / nrm)
                    kept.append(p)
        return np.asarray(rows, dtype=np.float64), np.asarray(kept, dtype=np.float64) / sr

    sx, tx = _snippets_and_times(ref_a, pr)
    sy, ty = _snippets_and_times(ali_a, pa)
    if len(sx) < 3 or len(sy) < 3:
        return [], 0
    width = min(sx.shape[1], sy.shape[1])
    sim = np.abs(sx[:, :width] @ sy[:, :width].T)

    ratios: list[float] = []
    for j, t in enumerate(ty):
        pred = a + t
        i_true = int(np.argmin(np.abs(tx - pred)))
        if abs(tx[i_true] - pred) >= pairing_tol_s:
            continue
        wrong = []
        for off in offsets_s:
            i2 = int(np.argmin(np.abs(tx - (pred + off))))
            if 0 <= i2 < sim.shape[0] and abs(tx[i2] - (pred + off)) < pairing_tol_s:
                wrong.append(float(sim[i2, j]))
        if wrong:
            ratios.append(float(sim[i_true, j]) / (max(wrong) + 1e-9))
    return ratios, len(ratios)


# ---------------------------------------------------------------------------
# 静音检测
# ---------------------------------------------------------------------------

def silence_report(x: np.ndarray, sr: int) -> tuple[bool, float, float]:
    """返回 (是否静音, 整段 RMS dBFS, 低能量帧比例)。"""
    if len(x) == 0:
        return True, -np.inf, 1.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-20))
    rms_db = 20.0 * math.log10(rms + 1e-12)

    frame = max(256, int(round(sr * 0.05)))
    hop = frame // 2
    if len(x) < frame:
        framed = x[None, :]
    else:
        n = 1 + (len(x) - frame) // hop
        framed = np.lib.stride_tricks.as_strided(
            x, shape=(n, frame), strides=(x.strides[0] * hop, x.strides[0])
        )
    frame_rms = np.sqrt(np.mean(framed.astype(np.float64) ** 2, axis=1) + 1e-20)
    floor = 10.0 ** (settings.silence_dbfs / 20.0)
    low_ratio = float(np.mean(frame_rms < floor))

    silent = (rms < 10.0 ** (settings.silence_dbfs / 20.0)
              and low_ratio > settings.silence_frame_ratio)
    return silent, rms_db, low_ratio


# ---------------------------------------------------------------------------
# 粗对齐：能量包络互相关
# ---------------------------------------------------------------------------

def downsample_envelope(x: np.ndarray, src_rate: int, env_rate: int) -> np.ndarray:
    """RMS 能量包络，速率 env_rate Hz，去均值，鲁棒于音量差。"""
    win = max(1, int(round(src_rate / env_rate)))
    n = len(x) // win
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    trimmed = x[: n * win].reshape(n, win)
    env = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    env = env.astype(np.float32)
    env -= env.mean()
    norm = np.linalg.norm(env) + 1e-12
    return env / norm


def envelope_xcorr(ref_env: np.ndarray,
                   ali_env: np.ndarray,
                   env_rate: int,
                   min_overlap_s: float = 7.5) -> tuple[float, float]:
    """包络 NCCF（按每个 lag 的重叠窗能量做归一化）。

    返回 (lag_seconds, peak)，lag>0 表示同一事件在待对齐录音中出现得更早
    （即 t参考 ≈ lag + t待对齐）。
    """
    nr, na = len(ref_env), len(ali_env)
    n = nr + na - 1
    nfft = 1 << int(np.ceil(np.log2(max(1, n))))

    def _corr(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        c = np.fft.irfft(np.fft.rfft(u, nfft) * np.conj(np.fft.rfft(v, nfft)), nfft)
        return np.concatenate([c[-(na - 1):], c[:nr]])

    cross = _corr(ref_env, ali_env)
    # 每个 lag 下两路各自重叠窗的能量和（与 cross 同索引约定）
    sum_r2 = _corr(ref_env ** 2, np.ones(na, dtype=np.float64))
    sum_a2 = _corr(np.ones(nr, dtype=np.float64), ali_env ** 2)
    denom = np.sqrt(np.maximum(sum_r2 * sum_a2, 1e-12))
    corr = cross / denom

    # 至少四分之一的最小重叠长度，防止边缘单窗伪峰被误判为对齐
    lags_all = np.arange(na - 1, -nr, -1)
    overlap_n = np.minimum(nr, na) - np.abs(lags_all)
    min_overlap_n = min(max(10, int(min_overlap_s * env_rate)),
                        max(2, min(nr, na) - 1))
    corr = np.where(overlap_n >= min_overlap_n, corr, -np.inf)
    if not np.isfinite(corr).any():
        return 0.0, 0.0

    idx = int(np.argmax(corr))
    peak = float(corr[idx])
    lag = idx - (na - 1)
    return lag / env_rate, peak


# ---------------------------------------------------------------------------
# 精对齐：GCC-PHAT
# ---------------------------------------------------------------------------

def gcc_phat(ref: np.ndarray,
             ali: np.ndarray,
             max_lag: int,
             sr: int = 16000,
             floor: float = 1e-6) -> tuple[float, float]:
    """两段等长（或近似等长）信号的 PHAT 加权广义互相关。

    返回 (lag_samples, quality)：
      lag>0 表示同一事件在 ali 中比在 ref 中出现得更早
      （满足 t参考 ≈ lag + t待对齐）。
      quality = 主峰值 / 次峰（滞后搜索窗内），越大越尖锐；周期性内容时趋近 1。
    ref/ali 假定已归一化到同一分析采样率，长度相同。
    """
    vals, lags, nfft = _phat_corr(ref, ali, max_lag, floor)

    idx = int(np.argmax(vals))
    peak = float(vals[idx])

    # 次峰：排除主峰附近 ±5ms 的主瓣区域，其余最大绝对值即为干扰峰
    excl = max(3, int(0.005 * sr))
    mask = np.ones(len(vals), dtype=bool)
    mask[max(0, idx - excl): min(len(vals), idx + excl + 1)] = False
    side = float(np.max(np.abs(vals[mask]))) if mask.any() else 0.0
    quality = peak / (side + 1e-9)

    # 抛物线插值得到亚样本 lag
    lag_float = float(lags[idx])
    if 0 < idx < len(vals) - 1:
        y0, y1, y2 = vals[idx - 1], vals[idx], vals[idx + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            if abs(delta) < 1.0:
                lag_float += float(np.clip(delta, -1, 1))
    return lag_float, quality


def _phat_corr(ref: np.ndarray, ali: np.ndarray, max_lag: int,
               floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray, int]:
    m = min(len(ref), len(ali))
    if m < 64:
        lags = np.arange(-max_lag, max_lag + 1)
        return np.zeros(len(lags)), lags, 1
    ref = ref[:m].astype(np.float64)
    ali = ali[:m].astype(np.float64)
    ref -= ref.mean()
    ali -= ali.mean()

    nfft = 1 << int(np.ceil(np.log2(2 * m)))
    spec = np.fft.rfft(ref, nfft) * np.conj(np.fft.rfft(ali, nfft))
    mag = np.abs(spec)
    # 轻地板 + 轻微正则，避免噪声段把相位放大到失真
    spec /= (mag + floor * (np.max(mag) + floor))
    corr = np.fft.irfft(spec, nfft)

    lags = np.arange(-max_lag, max_lag + 1)
    vals = corr[np.mod(lags, nfft)]
    return vals, lags, nfft
