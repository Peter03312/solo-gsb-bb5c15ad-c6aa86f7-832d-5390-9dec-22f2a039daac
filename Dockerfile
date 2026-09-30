FROM python:3.11-slim

# 仅需最小运行时（numpy/scipy 走 manylinux wheel，无需编译工具链）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# 对齐结果输出目录（compose 会挂载命名卷）
RUN mkdir -p /data/jobs
ENV ALIGN_OUTPUT_DIR=/data/jobs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import json,urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=3); sys.exit(0 if json.load(r)['status']=='ok' else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
