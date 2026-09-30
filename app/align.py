"""多设备录音时间对齐核心流程。

估计线性映射   t参考 = a + b * t待对齐   （均以秒计）：
  1. 能量包络归一化互相关做粗时延（对音量差稳健）；
  2. 在预测重叠区内均匀取若干 6s 段，GCC-PHAT 测每段局部时延；
  3. 对 (t待对齐, t参考) 做加权直线拟合得到 a、b，迭代两次；
  4. 质量门限：段数、峰值尖锐度、拟合残差、b 范围、重叠长度、静音、唯一性。

任一硬性条件不满足都抛出 AlignError，绝不产出伪造音轨。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import dsp
from .config import settings
from .errors import AlignError
from .wavio import MonoAudio


@dataclass
class SegmentObs:
    c: float          # 待对齐侧段中心时间（s）
    p: float          # 对应参考侧段中心时间（s）
    lag: float        # 测得的段内时延（s，正=待对齐更晚）
    quality: float    # PHAT 主峰/次峰
    rms_db: float


@dataclass
class AlignResult:
    a: float
    b: float
    overlap_start_ref: float
    overlap_end_ref: float
    overlap_start_ali: float
    overlap_end_ali: float
    overlap_seconds: float
    coarse_peak: float
    segments: list[SegmentObs] = field(default_factory=list)
    residual_rms_ms: float = 0.0
    ref_out: np.ndarray | None = None
    ali_out: np.ndarray | None = None
    out_sample_rate: int = 0


def _reject(code: str, message: str, **detail) -> AlignError:
    return AlignError(code, message, **detail)


# ---------------------------------------------------------------------------
# 静音检查
# ---------------------------------------------------------------------------

def _check_silence(audio: MonoAudio, analysis: np.ndarray, role: str, label: str) -> None:
    silent, rms_db, low_ratio = dsp.silence_report(analysis, settings.analysis_rate)
    if silent:
        raise _reject(
            "SILENCE",
            f"{label}为静音或近静音（整段 RMS={rms_db:.1f} dBFS，"
            f"低能量帧占比 {low_ratio * 100:.0f}%），无法对齐。",
            role=role, rms_dbfs=round(rms_db, 2), low_energy_frame_ratio=round(low_ratio, 3),
        )


# ---------------------------------------------------------------------------
# 分段精细时延
# ---------------------------------------------------------------------------

def _seg_rms_db(seg: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-20))
    return 20.0 * math.log10(rms + 1e-12)


def _find_events(x: np.ndarray, sr: int, frame_s: float = 0.02,
                 min_gap_s: float = 1.0) -> list[int]:
    """返回能量爆发点的样本位置（帧能量阈值 + 最小间隔的非极大抑制）。"""
    fl = max(64, int(round(frame_s * sr)))
    n = len(x) // fl
    if n == 0:
        return []
    e = np.sum(x[: n * fl].reshape(n, fl).astype(np.float64) ** 2, axis=1)
    noise = np.median(e) + 1e-12
    thresh = noise * 25.0 + np.mean(e) * 0.05
    cand = np.where(e > thresh)[0]
    events: list[int] = []
    for i in cand:
        lo, hi = max(0, i - 2), min(n, i + 3)
        if e[i] == np.max(e[lo:hi]):
            pos = i * fl
            if not events or pos - events[-1] >= int(min_gap_s * sr):
                events.append(int(pos))
    return events


def _collect_segments(ref_a: np.ndarray, ali_a: np.ndarray,
                      a: float, b: float,
                      search_ms: float) -> list[SegmentObs]:
    """在待对齐信号上找能量事件，映射到参考侧取短窗做 PHAT。

    短窗（默认 1.2s）围绕单个声音事件，避免长窗内大量静音白化后把相关峰展宽。
    """
    sr = settings.analysis_rate
    win_n = int(round(settings.segment_s * sr))
    search = max(1, int(round(search_ms * sr / 1000.0)))
    floor_db = settings.silence_dbfs + 6.0

    events = _find_events(ali_a, sr)
    obs: list[SegmentObs] = []
    for ia0 in events:
        ia = ia0 - win_n // 2
        if ia < 0 or ia + win_n > len(ali_a):
            continue
        sa = ali_a[ia: ia + win_n]
        if _seg_rms_db(sa) < floor_db:
            continue
        c = ia0 / sr
        p = a + b * c
        ir = int(round(p * sr)) - win_n // 2
        if ir < 0 or ir + win_n > len(ref_a):
            continue
        sf = ref_a[ir: ir + win_n]
        lag, q = dsp.gcc_phat(sf, sa, search, sr=sr)
        obs.append(SegmentObs(c=float(c), p=float(p), lag=lag / sr,
                              quality=float(q), rms_db=_seg_rms_db(sa)))
    return obs


def _fit_line(obs: list[SegmentObs]) -> tuple[float, float, float, float]:
    """加权拟合 p' = a + b*c（p' = p + lag）。

    返回 (a, b, residual_rms_ms, mean_quality)。
    """
    c = np.array([o.c for o in obs], dtype=np.float64)
    pp = np.array([o.p + o.lag for o in obs], dtype=np.float64)
    q = np.array([o.quality for o in obs], dtype=np.float64)
    w = np.clip(q, 0.2, None) ** 2
    w /= w.sum()

    cw = float(np.sum(w * c))
    pw = float(np.sum(w * pp))
    var_c = float(np.sum(w * (c - cw) ** 2))
    if var_c < 1e-12:
        raise _reject("AMBIGUOUS_ALIGNMENT",
                      "有效观测段的时间跨度太小，无法估计时钟漂移 b。")
    b = float(np.sum(w * (c - cw) * (pp - pw)) / var_c)
    a = pw - b * cw
    resid = pp - (a + b * c)
    rms_ms = float(np.sqrt(np.mean(resid ** 2))) * 1000.0
    return a, b, rms_ms, float(np.average(q, weights=w))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def align_recordings(ref: MonoAudio, ali: MonoAudio) -> AlignResult:
    ar = settings.analysis_rate
    ref_a = dsp.resample_to_rate(ref.samples, ref.sample_rate, ar)
    ali_a = dsp.resample_to_rate(ali.samples, ali.sample_rate, ar)

    _check_silence(ref, ref_a, "reference", "参考录音")
    _check_silence(ali, ali_a, "recording", "待对齐录音")

    ref_dur = len(ref_a) / ar
    ali_dur = len(ali_a) / ar

    # --- 1) 粗对齐：能量包络 ---------------------------------------------
    er = settings.envelope_rate
    ref_env = dsp.downsample_envelope(ref_a, ar, er)
    ali_env = dsp.downsample_envelope(ali_a, ar, er)
    lag0, coarse_peak = dsp.envelope_xcorr(
        ref_env, ali_env, er, min_overlap_s=settings.min_overlap_s / 4)
    # t参考 = a0 + t待对齐（b 初始为 1）；lag0<0 表示 ali 中事件更早（a0 为负）
    a0 = lag0

    # --- 粗对齐可信度检查（先于重叠判断，避免随机峰导致错误归因）------------
    if coarse_peak < 0.10:
        raise _reject(
            "NO_COMMON_EVENT",
            f"两条录音未发现可信的共同声音事件（包络相关峰值 {coarse_peak:.3f}），"
            "可能内容完全不同、为周期性信号或噪声过强。",
            coarse_correlation=round(coarse_peak, 4),
        )

    # --- 粗重叠预检（唯一性检查优先于重叠判断，见下方 1b）------------------
    ov_s_ali = max(0.0, -a0)
    ov_e_ali = min(ali_dur, (ref_dur - a0))
    coarse_overlap = ov_e_ali - ov_s_ali

    # --- 1b) 对齐唯一性：节拍器/蜂鸣等雷同周期内容在错误周期位置的波形匹配
    # 与真实位置几乎一样强，滞后不唯一。必须先于重叠预检，否则粗对齐可能锁到
    # 一个重叠不足的周期伪峰而给出误导性拒绝原因。
    ratios, n_uniq = dsp.event_match_ratios(ref_a, ali_a, ar, a0)
    if n_uniq >= settings.min_segments:
        med_uniq = float(np.median(ratios))
        n_ambig = int(np.sum(np.array(ratios) < settings.min_event_uniqueness))
        if med_uniq < settings.min_event_uniqueness or n_ambig > len(ratios) // 3:
            raise _reject(
                "AMBIGUOUS_ALIGNMENT",
                f"在 {len(ratios)} 个可检验的声音事件中，{n_ambig} 个在偏离整数"
                "周期的位置上存在同样强的波形匹配（真实匹配/周期错配相似度"
                f"中位数 {med_uniq:.2f}），内容呈周期性重复（如节拍、蜂鸣），"
                "时间对齐不唯一，拒绝输出。",
                event_uniqueness_median=round(med_uniq, 3),
                ambiguous_events=n_ambig,
                checked_events=len(ratios),
            )

    if coarse_overlap < settings.min_overlap_s:
        raise _reject(
            "OVERLAP_TOO_SHORT",
            f"粗对齐估计的共同重叠仅 {coarse_overlap:.1f}s，"
            f"少于要求的 {settings.min_overlap_s:g}s。",
            estimated_overlap_s=round(coarse_overlap, 2),
            coarse_correlation=round(coarse_peak, 4),
        )

    # --- 2/3) 分段 PHAT + 加权拟合，迭代精化 ------------------------------
    a, b = a0, 1.0
    obs: list[SegmentObs] = []
    residual_ms = 0.0
    for it in range(max(1, settings.refine_iterations)):
        search_ms = (settings.seg_search_ms_first if it == 0
                     else settings.seg_search_ms_next)
        obs = _collect_segments(ref_a, ali_a, a, b, search_ms)
        if len(obs) < settings.min_segments:
            raise _reject(
                "AMBIGUOUS_ALIGNMENT",
                f"重叠区内仅找到 {len(obs)} 个有效对齐段"
                f"（至少需要 {settings.min_segments} 个），"
                "内容可能过于安静或为周期性信号，对齐不唯一。",
                valid_segments=len(obs),
            )
        a, b, residual_ms, mean_q = _fit_line(obs)

    # --- 4) 质量门限 -------------------------------------------------------
    qs = np.array([o.quality for o in obs])
    med_q = float(np.median(qs))
    min_q = float(np.min(qs))
    if med_q < settings.min_peak_ratio or min_q < settings.min_seg_peak_ratio:
        raise _reject(
            "AMBIGUOUS_ALIGNMENT",
            "各段互相关峰不够尖锐（主峰/次峰中位数 "
            f"{med_q:.2f}、最小值 {min_q:.2f}），内容疑似周期性或噪声过强，"
            "对齐结果不唯一，拒绝输出。",
            median_peak_ratio=round(med_q, 3),
            min_peak_ratio=round(min_q, 3),
            valid_segments=len(obs),
        )
    if residual_ms > settings.max_residual_ms:
        raise _reject(
            "AMBIGUOUS_ALIGNMENT",
            f"分段时延相对线性模型的残差 RMS 为 {residual_ms:.1f}ms，"
            f"超过 {settings.max_residual_ms:g}ms 容差，映射模型不可信（"
            "可能存在非匀速播放/多处不同内容）。",
            residual_rms_ms=round(residual_ms, 2),
        )

    # 留 5e-7 数值容差吸收亚样本估计误差（约为允许漂移量的 0.0005%），
    # 不改变 0.999/1.001 硬约束语义
    eps = 5e-7
    if not (settings.b_min - eps <= b <= settings.b_max + eps):
        raise _reject(
            "B_OUT_OF_RANGE",
            f"估计的漂移系数 b={b:.6f} 超出允许范围 "
            f"[{settings.b_min}, {settings.b_max}]，拒绝对齐。",
            b=b, b_min=settings.b_min, b_max=settings.b_max,
        )

    # --- 最终重叠区间 -----------------------------------------------------
    ov_s_ali = max(0.0, -a / b)
    ov_e_ali = min(ali_dur, (ref_dur - a) / b)
    ov_s_ref = a + b * ov_s_ali
    ov_e_ref = a + b * ov_e_ali
    overlap = ov_e_ref - ov_s_ref
    if overlap < settings.min_overlap_s:
        raise _reject(
            "OVERLAP_TOO_SHORT",
            f"最终共同重叠仅 {overlap:.1f}s，少于要求的 {settings.min_overlap_s:g}s。",
            estimated_overlap_s=round(overlap, 2),
        )

    # --- 生成输出：参考采样率、单声道、裁到共同重叠 ------------------------
    ref_out, ali_out, out_sr = _render_overlap(ref, ali, a, b,
                                               ov_s_ref, ov_e_ref,
                                               ov_s_ali, ov_e_ali)

    return AlignResult(
        a=a, b=b,
        overlap_start_ref=ov_s_ref, overlap_end_ref=ov_e_ref,
        overlap_start_ali=ov_s_ali, overlap_end_ali=ov_e_ali,
        overlap_seconds=overlap,
        coarse_peak=coarse_peak,
        segments=obs, residual_rms_ms=residual_ms,
        ref_out=ref_out, ali_out=ali_out, out_sample_rate=out_sr,
    )


def _render_overlap(ref: MonoAudio, ali: MonoAudio,
                    a: float, b: float,
                    ov_s_ref: float, ov_e_ref: float,
                    ov_s_ali: float, ov_e_ali: float) -> tuple[np.ndarray, np.ndarray, int]:
    """在原生采样率上裁切并重采样到参考采样率，两条输出严格等长（样本对样本）。"""
    out_sr = ref.sample_rate
    n_out = int(round((ov_e_ref - ov_s_ref) * out_sr))
    if n_out <= 0:
        raise _reject("OVERLAP_TOO_SHORT", "共同重叠区间长度为 0。")

    # 参考侧：仅裁切（无漂移）。对齐到最近样本，边界补零避免咔哒声
    r0 = int(round(ov_s_ref * ref.sample_rate))
    r1 = r0 + n_out
    ref_out = np.empty(n_out, dtype=np.float32)
    ref_valid_l = max(0, r0)
    ref_valid_r = min(len(ref.samples), r1)
    ref_out[:] = 0.0
    if ref_valid_r > ref_valid_l:
        ref_out[ref_valid_l - r0: ref_valid_r - r0] = ref.samples[ref_valid_l:ref_valid_r]

    # 待对齐侧：输出样本 m 对应参考时间 ov_s_ref + m/out_sr，
    # 待对齐时间 c = (ref_t - a)/b，原生位置 c * ali.sample_rate。
    # 一次任意比率带限重采样同时完成名义采样率差与漂移 b 的补偿。
    start_pos = (ov_s_ref - a) / b * ali.sample_rate          # m=0 的原生位置
    ratio = ali.sample_rate / (out_sr * b)                    # 输出间隔/输入间隔
    ali_out = dsp.resample_arbitrary_range(
        ali.samples, ratio, start_pos, n_out
    )

    return ref_out, ali_out, out_sr
