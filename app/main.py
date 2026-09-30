"""多设备录音时间对齐 HTTP 服务。

POST /api/v1/align   multipart 上传 reference + recording
GET  /api/v1/jobs/{job_id}/files/{name}   下载对齐后的 WAV
GET  /healthz        健康检查
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from .align import align_recordings
from .config import settings
from .errors import AlignError
from .wavio import read_wav, write_wav_16bit

app = FastAPI(
    title="多设备录音时间对齐服务",
    description="上传参考录音与待对齐录音，估计 t参考 = a + b*t待对齐，"
                "并输出重采样、裁切到共同重叠区间的两条单声道 16 位 PCM WAV。",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# 作业存储（本地目录 + TTL 清理）
# ---------------------------------------------------------------------------

def _jobs_root() -> Path:
    root = Path(settings.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _purge_old_jobs() -> None:
    root = _jobs_root()
    now = time.time()
    for d in root.iterdir():
        meta = d / "result.json"
        try:
            if d.is_dir() and (not meta.exists()
                               or now - meta.stat().st_mtime > settings.job_ttl_seconds):
                for f in d.iterdir():
                    f.unlink()
                d.rmdir()
        except OSError:
            pass
    dirs = sorted([d for d in root.iterdir() if d.is_dir()],
                  key=lambda d: d.stat().st_mtime)
    for d in dirs[: max(0, len(dirs) - settings.job_max_count)]:
        try:
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
        except OSError:
            pass


@app.on_event("startup")
def _startup() -> None:
    _jobs_root()
    _purge_old_jobs()


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _read_upload(f: UploadFile, role: str) -> bytes:
    """同步读取上传文件（在线程池端点中调用），超过上限立即拒绝。"""
    label = "参考录音" if role == "reference" else "待对齐录音"
    limit = settings.max_upload_bytes
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = f.file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise AlignError(
                "UNSUPPORTED_FORMAT",
                f"{label}大小超过上限 {limit // 1024 // 1024}MB。",
                role=role, max_bytes=limit,
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise AlignError("INVALID_WAV", f"{label}为空，请上传有效的 WAV 文件。", role=role)
    return data


def _segments_json(result) -> list[dict[str, Any]]:
    return [
        {
            "recording_time_s": round(o.c, 4),
            "reference_time_s": round(o.p + o.lag, 4),
            "segment_lag_s": round(o.lag, 6),
            "peak_ratio": round(o.quality, 3),
        }
        for o in result.segments
    ]


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"service": "audio-alignment", "docs": "/docs", "health": "/healthz"}


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status": "ok", "output_dir": settings.output_dir}


@app.post("/api/v1/align")
def align(reference: UploadFile = File(..., description="参考录音 WAV"),
          recording: UploadFile = File(..., description="待对齐录音 WAV")):
    """同步端点：CPU 密集的对齐计算在线程池中执行，不阻塞事件循环。"""
    _purge_old_jobs()

    ref_bytes = _read_upload(reference, "reference")
    rec_bytes = _read_upload(recording, "recording")

    # 严格校验（无效/不支持格式在此明确拒绝）
    ref_audio = read_wav(ref_bytes, "reference")
    rec_audio = read_wav(rec_bytes, "recording")

    result = align_recordings(ref_audio, rec_audio)

    job_id = uuid.uuid4().hex
    job_dir = _jobs_root() / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    ref_path = job_dir / "reference_aligned.wav"
    rec_path = job_dir / "recording_aligned.wav"
    write_wav_16bit(str(ref_path), result.ref_out, result.out_sample_rate)
    write_wav_16bit(str(rec_path), result.ali_out, result.out_sample_rate)

    body = {
        "job_id": job_id,
        "mapping": {
            "equation": "t_reference = a + b * t_recording",
            "a_s": round(result.a, 6),
            "b": round(result.b, 8),
        },
        "overlap": {
            "start_reference_s": round(result.overlap_start_ref, 6),
            "end_reference_s": round(result.overlap_end_ref, 6),
            "start_recording_s": round(result.overlap_start_ali, 6),
            "end_recording_s": round(result.overlap_end_ali, 6),
            "duration_s": round(result.overlap_seconds, 6),
        },
        "quality": {
            "coarse_envelope_correlation": round(result.coarse_peak, 4),
            "residual_rms_ms": round(result.residual_rms_ms, 3),
            "segments": _segments_json(result),
        },
        "output": {
            "sample_rate_hz": result.out_sample_rate,
            "channels": 1,
            "sample_format": "PCM_16",
            "frames": len(result.ref_out),
            "reference_wav_url": f"/api/v1/jobs/{job_id}/files/reference_aligned.wav",
            "recording_wav_url": f"/api/v1/jobs/{job_id}/files/recording_aligned.wav",
        },
        "inputs": {
            "reference": {"sample_rate_hz": ref_audio.sample_rate,
                          "channels": ref_audio.channels,
                          "duration_s": round(ref_audio.duration, 3)},
            "recording": {"sample_rate_hz": rec_audio.sample_rate,
                          "channels": rec_audio.channels,
                          "duration_s": round(rec_audio.duration, 3)},
        },
    }
    (job_dir / "result.json").write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    return body


@app.get("/api/v1/jobs/{job_id}/files/{name}")
def download(job_id: str, name: str):
    if name not in ("reference_aligned.wav", "recording_aligned.wav") or not _safe_id(job_id):
        return JSONResponse(status_code=404,
                            content={"error": {"code": "NOT_FOUND", "message": "文件不存在。"}})
    path = _jobs_root() / job_id / name
    if not path.is_file():
        return JSONResponse(status_code=404,
                            content={"error": {"code": "NOT_FOUND",
                                               "message": "文件不存在或已过期。"}})
    return Response(content=path.read_bytes(), media_type="audio/wav",
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


def _safe_id(job_id: str) -> bool:
    return bool(job_id) and all(c in "0123456789abcdef" for c in job_id) and len(job_id) == 32


# ---------------------------------------------------------------------------
# 统一错误处理
# ---------------------------------------------------------------------------

@app.exception_handler(AlignError)
async def align_error_handler(_request, exc: AlignError):
    return JSONResponse(status_code=exc.http_status, content=exc.to_dict())


@app.exception_handler(RequestValidationError)
async def request_error_handler(_request, exc: RequestValidationError):
    """表单校验错误（缺少文件等）。"""
    return JSONResponse(
        status_code=400,
        content={"error": {
            "code": "INVALID_REQUEST",
            "message": "请求必须以 multipart/form-data 同时上传 reference 与 recording 两个 WAV 文件。",
            "detail": exc.errors(),
        }},
    )
