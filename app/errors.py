"""对齐过程中对外暴露的错误类型。

code 供调用方程序化处理，message 为中文说明，detail 携带可诊断字段。
"""
from __future__ import annotations

from typing import Any


class AlignError(Exception):
    """所有可预期拒绝情况的基类。"""

    http_status = 422

    def __init__(self, code: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "detail": self.detail}}


class InvalidWav(AlignError):
    """文件不是受支持的 16 位 PCM WAV。"""

    http_status = 400
