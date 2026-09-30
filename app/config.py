"""服务运行参数，全部可通过环境变量覆盖。"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


@dataclass(frozen=True)
class Settings:
    # 服务监听
    host: str = os.getenv("ALIGN_HOST", "0.0.0.0")
    port: int = _get_int("ALIGN_PORT", 8000)

    # 输入 WAV 约束
    min_duration_s: float = _get_float("ALIGN_MIN_DURATION_S", 1.0)
    max_duration_s: float = _get_float("ALIGN_MAX_DURATION_S", 600.0)
    min_sample_rate: int = _get_int("ALIGN_MIN_SAMPLE_RATE", 16000)
    max_sample_rate: int = _get_int("ALIGN_MAX_SAMPLE_RATE", 48000)
    max_upload_bytes: int = _get_int("ALIGN_MAX_UPLOAD_MB", 120) * 1024 * 1024

    # 对齐参数约束
    b_min: float = _get_float("ALIGN_B_MIN", 0.999)
    b_max: float = _get_float("ALIGN_B_MAX", 1.001)
    min_overlap_s: float = _get_float("ALIGN_MIN_OVERLAP_S", 30.0)

    # 静音判定（分析采样率下的整段 RMS 与低能量帧比例，二者同时满足才判静音）
    silence_dbfs: float = _get_float("ALIGN_SILENCE_DBFS", -66.0)
    silence_frame_ratio: float = _get_float("ALIGN_SILENCE_FRAME_RATIO", 0.98)

    # 分析与质量门限
    analysis_rate: int = _get_int("ALIGN_ANALYSIS_RATE", 16000)
    envelope_rate: int = _get_int("ALIGN_ENVELOPE_RATE", 50)
    segment_s: float = _get_float("ALIGN_SEGMENT_S", 1.2)
    min_segments: int = _get_int("ALIGN_MIN_SEGMENTS", 4)
    min_peak_ratio: float = _get_float("ALIGN_MIN_PEAK_RATIO", 1.30)
    min_seg_peak_ratio: float = _get_float("ALIGN_MIN_SEG_PEAK_RATIO", 1.15)
    # 事件一对一匹配唯一性：最佳/次佳波形相似度比值（周期性雷同内容趋近 1）
    min_event_uniqueness: float = _get_float("ALIGN_MIN_EVENT_UNIQUENESS", 2.0)
    max_residual_ms: float = _get_float("ALIGN_MAX_RESIDUAL_MS", 10.0)
    refine_iterations: int = _get_int("ALIGN_REFINE_ITERATIONS", 2)
    seg_search_ms_first: float = _get_float("ALIGN_SEG_SEARCH_MS_FIRST", 300.0)
    seg_search_ms_next: float = _get_float("ALIGN_SEG_SEARCH_MS_NEXT", 30.0)

    # 结果保留（默认在项目根 data/jobs；容器内建议挂载卷到 /data/jobs）
    output_dir: str = os.getenv(
        "ALIGN_OUTPUT_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "data", "jobs"),
    )
    job_ttl_seconds: int = _get_int("ALIGN_JOB_TTL_SECONDS", 3600)
    job_max_count: int = _get_int("ALIGN_JOB_MAX_COUNT", 200)


settings = Settings()
