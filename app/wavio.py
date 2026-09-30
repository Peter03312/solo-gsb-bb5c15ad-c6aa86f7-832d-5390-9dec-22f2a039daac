"""16 位 PCM WAV 的严格读取与写出。

只接受：WAV(RIFF) 容器、PCM 或 WAVE_FORMAT_EXTENSIBLE(PCM 子类型)、
16 bit、单声道或立体声、16k~48kHz、时长 1~10 分钟。其余一律明确拒绝。

解析器手写 RIFF 块扫描，不依赖 wave 模块对格式标记的宽松/严格假设，
可以正确处理带 LIST 等附加块以及现代录音设备常见的 extensible PCM。
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .config import settings
from .errors import InvalidWav

# WAVE_FORMAT_EXTENSIBLE 中 KSDATAFORMAT_SUBTYPE_PCM 的 GUID（小端字节序）
_PCM_SUBTYPE_GUID = bytes.fromhex("0100000000001000800000aa00389b71")
_WAVE_FORMAT_PCM = 0x0001
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE


@dataclass(frozen=True)
class MonoAudio:
    """单声道 float32 信号（-1~1）及其原生采样率。"""

    samples: np.ndarray
    sample_rate: int
    channels: int  # 原始声道数（1 或 2）

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


def _invalid(role: str, reason: str, **detail) -> InvalidWav:
    where = "参考录音" if role == "reference" else "待对齐录音"
    return InvalidWav(
        "INVALID_WAV",
        f"{where}不是有效的 WAV 文件：{reason}",
        role=role,
        **detail,
    )


def _unsupported(role: str, reason: str, **detail) -> InvalidWav:
    where = "参考录音" if role == "reference" else "待对齐录音"
    return InvalidWav(
        "UNSUPPORTED_FORMAT",
        f"{where}格式不受支持：{reason}",
        role=role,
        **detail,
    )


def _scan_wav(data: bytes, role: str) -> tuple[bytes, dict]:
    """扫描 RIFF/WAVE，返回 (data 块字节, fmt 描述)。"""
    if len(data) < 44 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise _invalid(role, "缺少 RIFF/WAVE 标识，仅接受 WAV 容器。")
    if struct.unpack("<I", data[4:8])[0] != len(data) - 8:
        # 长度声明不匹配仍尝试按实际字节读取（部分设备写入长度不规范）
        pass

    pos = 12
    fmt: dict | None = None
    audio_data: bytes | None = None
    n = len(data)
    while pos + 8 <= n:
        cid = data[pos: pos + 4]
        (csize,) = struct.unpack("<I", data[pos + 4: pos + 8])
        body = pos + 8

        if cid == b"fmt ":
            if csize < 16 or body + csize > n:
                raise _invalid(role, "fmt 块长度异常或文件被截断。")
            f = data[body: body + min(csize, 40)]
            (tag, channels, framerate, _avg, _block,
             bits) = struct.unpack("<HHIIHH", f[:16])
            fmt_info = {"tag": tag, "channels": channels,
                        "sample_rate": framerate, "bits": bits}
            if tag == _WAVE_FORMAT_EXTENSIBLE and csize >= 40:
                if len(f) < 40:
                    raise _invalid(role, "extensible fmt 块不完整。")
                valid_bits = struct.unpack("<H", f[18:20])[0]
                guid = f[24:40]
                fmt_info["valid_bits"] = valid_bits
                fmt_info["guid"] = guid
            fmt = fmt_info
        elif cid == b"data":
            if body > n:
                raise _invalid(role, "data 块声明位置超出文件，文件被截断。")
            available = n - body
            if csize > available:
                raise _invalid(
                    role,
                    f"数据被截断：data 块声明 {csize} 字节，实际仅有 {available} 字节。",
                    declared_bytes=csize, actual_bytes=available,
                )
            audio_data = data[body: body + csize]

        # RIFF 块按偶数字节对齐；块越界即结束扫描
        nxt = body + csize + (csize & 1)
        if csize > n - body:
            break
        pos = nxt

    if fmt is None:
        raise _invalid(role, "缺少 fmt 块。")
    if audio_data is None:
        raise _invalid(role, "缺少 data 块。")
    return audio_data, fmt


def read_wav(data: bytes, role: str) -> MonoAudio:
    """读取并校验上传的 WAV 字节，返回降混后的单声道信号。"""
    raw, fmt = _scan_wav(data, role)

    tag = fmt["tag"]
    if tag == _WAVE_FORMAT_PCM:
        is_pcm = True
    elif tag == _WAVE_FORMAT_EXTENSIBLE:
        if fmt.get("guid") != _PCM_SUBTYPE_GUID:
            raise _unsupported(role, "extensible WAV 不是 PCM 子类型（可能是浮点或压缩编码）",
                               format_tag=tag)
        if fmt.get("valid_bits", 16) != 16:
            raise _unsupported(role,
                               f"extensible PCM 有效位深 {fmt.get('valid_bits')} 位，仅支持 16 位",
                               valid_bits=fmt.get("valid_bits"))
        is_pcm = True
    else:
        name = {0x0003: "IEEE_FLOAT", 0x0055: "MP3", 0x0057: "GSM",
                0x0002: "ADPCM", 0x0011: "IMA_ADPCM"}.get(tag, f"0x{tag:04x}")
        raise _unsupported(role, f"仅支持无压缩 PCM，检测到编码 {name}", format_tag=tag)

    if not is_pcm:  # pragma: no cover - 逻辑兜底
        raise _unsupported(role, "非 PCM 编码。")

    bits = fmt["bits"]
    if bits != 16:
        raise _unsupported(role, f"仅支持 16 位 PCM，检测到 {bits} 位", sample_width_bits=bits)

    nchannels = fmt["channels"]
    if nchannels not in (1, 2):
        raise _unsupported(role, f"仅支持单声道或立体声，检测到 {nchannels} 声道",
                           channels=nchannels)

    framerate = fmt["sample_rate"]
    if not (settings.min_sample_rate <= framerate <= settings.max_sample_rate):
        raise _unsupported(
            role,
            f"采样率 {framerate} Hz 超出 {settings.min_sample_rate}~{settings.max_sample_rate} Hz 范围",
            sample_rate=framerate,
        )

    expected_bytes_minus_trunc = (len(raw) // (nchannels * 2)) * nchannels * 2
    nframes = len(raw) // (nchannels * 2)
    if nframes == 0:
        raise _invalid(role, "data 块为空。")
    if len(raw) % (nchannels * 2) != 0:
        # 尾部不足一个完整样本帧：明确报错而不是静默截断
        raise _invalid(
            role,
            "data 块长度与声道数/位深不匹配，文件可能被截断。",
            data_bytes=len(raw), frame_bytes=nchannels * 2,
        )

    interleaved = np.frombuffer(raw[:expected_bytes_minus_trunc],
                                dtype="<i2").astype(np.float32) / 32768.0
    if nchannels == 2:
        frames = interleaved.reshape(nframes, 2)
        mono = frames.mean(axis=1)  # 立体声先合成单声道
    else:
        mono = interleaved

    duration = nframes / framerate
    if not (settings.min_duration_s <= duration <= settings.max_duration_s):
        raise _unsupported(
            role,
            f"时长 {duration:.2f}s 超出 {settings.min_duration_s:g}~{settings.max_duration_s:g}s 范围",
            duration_s=round(duration, 3),
        )

    # 去掉可能的直流偏移，不改变事件对齐关系
    mono -= mono.mean()
    return MonoAudio(samples=np.ascontiguousarray(mono, dtype=np.float32),
                     sample_rate=framerate, channels=nchannels)


def write_wav_16bit(path: str, samples: np.ndarray, sample_rate: int) -> None:
    """把 float 单声道信号写成标准 16 位 PCM WAV（格式标记 PCM=1）。"""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    data_bytes = pcm.tobytes()
    n = len(data_bytes)
    header = b"RIFF" + struct.pack("<I", 36 + n) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
                                   sample_rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", n)
    with open(path, "wb") as f:
        f.write(header)
        f.write(data_bytes)
