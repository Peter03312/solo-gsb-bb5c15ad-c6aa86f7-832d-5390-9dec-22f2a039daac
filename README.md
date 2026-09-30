# 多设备录音时间对齐服务

上传一条**参考录音**和一条**待对齐录音**，服务根据共同的声音事件估计线性时间映射

```
t参考 = a + b × t待对齐        （时间单位：秒）
```

其中 `b`（两台设备时钟速率比）严格限制在 **0.999～1.001**。同时输出两条已经
**按参考采样率重采样、裁切到共同重叠区间**的单声道 16 位 PCM WAV，可直接逐样本比对。

---

## 1. 功能概览

- 输入：两条 WAV —— 16 位 PCM、单声道/立体声（立体声自动平均为单声道）、
  采样率 16k～48 kHz、时长 1～10 分钟。
- 允许两条录音存在**音量差与加性噪声**：粗对齐用能量包络归一化互相关（不受幅度
  缩放影响），精对齐用 **GCC-PHAT**（广义互相关相位变换）在事件短窗内测亚样本级时延。
- 在整条重叠区间内均匀采集多个声音事件的局部时延，对 `(t待对齐, t参考)` 做加权直线拟合，
  迭代 2 次得到 `a`、`b`，并以残差 RMS 和相关峰尖锐度校验模型可信度。
- **以下情况明确拒绝（HTTP 4xx + 中文原因），绝不输出伪造音轨：**
  - 不是 WAV / 非 PCM / 非 16 位 / 声道或采样率或时长超出范围 / 文件截断；
  - 任一录音为静音或近静音（`SILENCE`）；
  - 找不到可信共同事件（`NO_COMMON_EVENT`）；
  - 共同重叠不足 30 秒（`OVERLAP_TOO_SHORT`）；
  - 内容呈周期性、互相关峰不尖锐或线性模型残差过大，对齐不唯一
    （`AMBIGUOUS_ALIGNMENT`）；
  - 估计的 `b` 超出 0.999～1.001（`B_OUT_OF_RANGE`）。

## 2. 一键启动（Docker Compose）

前置条件：已安装 Docker 与 Docker Compose 插件（`docker compose version` 可用）。

```bash
docker compose up -d --build      # 构建镜像、安装依赖并初始化服务
docker compose ps                 # 查看状态（health: healthy）
docker compose logs -f            # 查看日志
docker compose down               # 停止
```

服务启动后监听 **`http://0.0.0.0:8000`**：

| 用途 | 地址 |
|---|---|
| 健康检查 | `GET http://localhost:8000/healthz` |
| 对齐接口 | `POST http://localhost:8000/api/v1/align` |
| 交互式文档（Swagger） | `http://localhost:8000/docs` |
| OpenAPI Schema | `http://localhost:8000/openapi.json` |

> 宿主机端口可用环境变量修改：`ALIGN_PORT=9000 docker compose up -d --build`。

### 不使用容器的本地运行

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 3. 配置

所有参数都有默认值，可用环境变量覆盖（Docker 下写在 `docker-compose.yml` 的
`environment:` 里即可）：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `ALIGN_HOST` / `ALIGN_PORT` | `0.0.0.0` / `8000` | 监听地址与容器内端口 |
| `ALIGN_OUTPUT_DIR` | `/data/jobs`（容器内） | 结果 WAV 与元数据保存目录 |
| `ALIGN_MAX_UPLOAD_MB` | `120` | 单文件上传大小上限（MB） |
| `ALIGN_MIN_DURATION_S` / `ALIGN_MAX_DURATION_S` | `1` / `600` | 允许的录音时长（秒） |
| `ALIGN_MIN_SAMPLE_RATE` / `ALIGN_MAX_SAMPLE_RATE` | `16000` / `48000` | 允许的采样率（Hz） |
| `ALIGN_B_MIN` / `ALIGN_B_MAX` | `0.999` / `1.001` | 漂移系数允许范围 |
| `ALIGN_MIN_OVERLAP_S` | `30` | 共同重叠最小秒数 |
| `ALIGN_SILENCE_DBFS` | `-66` | 静音 RMS 门限（dBFS） |
| `ALIGN_SILENCE_FRAME_RATIO` | `0.98` | 低能量帧占比门限（与 RMS 同时满足才判静音） |
| `ALIGN_ANALYSIS_RATE` | `16000` | 内部分析采样率（Hz） |
| `ALIGN_MIN_SEGMENTS` | `4` | 拟合漂移所需的最少有效事件段 |
| `ALIGN_MIN_PEAK_RATIO` | `1.30` | 相关峰尖锐度中位数门限（主峰/次峰） |
| `ALIGN_MIN_EVENT_UNIQUENESS` | `2.0` | 真实事件匹配/整数周期错配的波形相似度比值门限，低于则判定周期性不唯一 |
| `ALIGN_MAX_RESIDUAL_MS` | `10` | 线性模型残差 RMS 上限（毫秒） |
| `ALIGN_JOB_TTL_SECONDS` | `3600` | 结果保留时长（秒），超时自动清理 |
| `ALIGN_JOB_MAX_COUNT` | `200` | 最多保留的作业数，超出按最旧删除 |

## 4. API 调用示例

### curl

```bash
curl -X POST http://localhost:8000/api/v1/align \
  -F "reference=@reference.wav;type=audio/wav" \
  -F "recording=@device2.wav;type=audio/wav"
```

- `reference`：参考录音 WAV（输出音轨采用它的采样率）。
- `recording`：待对齐录音 WAV。

### Python

```python
import requests

with open("reference.wav", "rb") as fr, open("device2.wav", "rb") as fd:
    resp = requests.post(
        "http://localhost:8000/api/v1/align",
        files={"reference": ("reference.wav", fr, "audio/wav"),
               "recording": ("device2.wav", fd, "audio/wav")},
        timeout=300,
    )
resp.raise_for_status()
body = resp.json()

a = body["mapping"]["a_s"]          # 偏移（秒）
b = body["mapping"]["b"]            # 时钟速率比，0.999~1.001
ov = body["overlap"]
print(a, b, ov["duration_s"])

base = "http://localhost:8000"
for url, name in ((body["output"]["reference_wav_url"], "reference_aligned.wav"),
                  (body["output"]["recording_wav_url"], "recording_aligned.wav")):
    with open(name, "wb") as f:
        f.write(requests.get(base + url).content)
```

### 成功响应示例（HTTP 200）

```json
{
  "job_id": "5ef2d2f6c8c641219285f3ecc049061b",
  "mapping": {
    "equation": "t_reference = a + b * t_recording",
    "a_s": -9.249995,
    "b": 1.00070004
  },
  "overlap": {
    "start_reference_s": 0.0,
    "end_reference_s": 85.816509,
    "start_recording_s": 9.243525,
    "end_recording_s": 95.0,
    "duration_s": 85.816509
  },
  "quality": {
    "coarse_envelope_correlation": 0.9454,
    "residual_rms_ms": 0.015,
    "segments": [
      {"recording_time_s": 10.38, "reference_time_s": 1.1373,
       "segment_lag_s": 0.00003, "peak_ratio": 12.364}
    ]
  },
  "output": {
    "sample_rate_hz": 48000,
    "channels": 1,
    "sample_format": "PCM_16",
    "frames": 4119192,
    "reference_wav_url": "/api/v1/jobs/5ef2.../files/reference_aligned.wav",
    "recording_wav_url": "/api/v1/jobs/5ef2.../files/recording_aligned.wav"
  }
}
```

两条输出 WAV **等长、同为参考采样率、同为单声道 16 位 PCM**，第 `n` 个样本在两条
音轨中表示同一物理时刻（分别取自 `overlap.start_reference_s` 与
`overlap.start_recording_s` 起）。

### 拒绝响应示例（HTTP 400 / 422）

```json
{
  "error": {
    "code": "B_OUT_OF_RANGE",
    "message": "估计的漂移系数 b=0.998012 超出允许范围 [0.999, 1.001]，拒绝对齐。",
    "detail": {"b": 0.998012, "b_min": 0.999, "b_max": 1.001}
  }
}
```

错误码：`INVALID_WAV` / `UNSUPPORTED_FORMAT` / `INVALID_REQUEST`（HTTP 400）；
`SILENCE` / `NO_COMMON_EVENT` / `OVERLAP_TOO_SHORT` / `AMBIGUOUS_ALIGNMENT` /
`B_OUT_OF_RANGE`（HTTP 422）。

## 5. 方法说明

1. **读取与校验**：Python `wave` 解析 RIFF/WAVE 头，严格要求 PCM、16 bit、1/2 声道、
   16k～48 kHz、1～10 分钟，同时检查数据块未截断；立体声按两声道平均降混，去直流。
2. **分析域统一**：用 `scipy.signal.resample_poly` 把两条录音重采样到 16 kHz 单声道
   （仅用于参数估计，不改变输出质量）。
3. **粗对齐**：50 Hz RMS 能量包络的归一化互相关（按各滞后位置的重叠能量归一化），
   得到秒级偏移 `a₀`；要求相关峰显著且粗重叠 ≥ 30 s。
4. **周期性歧义检测**：检测两侧各自的声音事件，截取每个事件 150 ms 的归一化波形。
   在粗映射给出的预测位置测量真实事件匹配强度，并与 ±0.5/1/2 s 等整数周期候选
   位置的错配强度比较。节拍器、蜂鸣等雷同周期内容在错配位置的强度与真匹配几乎
   相同（比值 ≈ 1），而音色各异的真实声音事件错配强度接近 0（比值通常 ≥ 10），
   据此在精拟合前拒绝“锁错周期”的对齐。节奏规律但每次击打音色不同的内容（常见
   音乐/语音）不受影响。
5. **精对齐**：在待对齐录音中检测能量事件点（阈值 + 非极大抑制，最小间隔 1 s），
   依据当前映射取参考侧 1.2 s 短窗，用 GCC-PHAT 测亚样本级局部时延；首轮搜索窗
   ±300 ms，收敛后 ±30 ms。
6. **漂移拟合**：对全部事件点 `(t待对齐, t参考 + 局部时延)` 以相关峰质量加权拟合直线
   得到 `a`、`b`，迭代 2 次；校验有效段数、峰尖锐度中位数与最小值、残差 RMS。
7. **参数与重叠门限**：`b ∉ [0.999, 1.001]` 拒绝；按映射求两条录音时间轴的共同区间，
   不足 30 s 拒绝。
8. **渲染输出**：参考音轨直接裁切；待对齐音轨用窗化 sinc 带限插值器，按
   `native_pos = (t参考 − a)/b × f待对齐` 一次性完成采样率差与 `b` 漂移补偿。两条音轨
   输出相同样本数（参考采样率、单声道、16 位 PCM），边界补零避免咔哒声。

## 6. 目录结构与测试

```
app/
  config.py    # 环境变量配置
  errors.py    # 错误类型与错误码
  wavio.py     # 严格的 WAV 读取/校验与 16 位写出
  dsp.py       # 带限重采样、包络 NCCF、GCC-PHAT、静音检测
  align.py     # 对齐管线（粗对齐→分段 PHAT→加权拟合→门限→渲染）
  main.py      # FastAPI 路由、作业存储与清理、错误处理
Dockerfile
docker-compose.yml
tests/         # 25 项 pytest（正常路径、各类拒绝、WAV 校验、HTTP 层）
```

```bash
pip install pytest
python3 -m pytest tests/ -q       # 单元/接口测试
python3 -m tests.smoke            # 合成录音端到端冒烟测试
```
