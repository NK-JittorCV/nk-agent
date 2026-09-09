# GeoAgent-Jittor

`ghost233lism/GeoAgent`（Qwen2.5-VL-7B）的纯 Jittor 单图推理实现。最终推理进程不安装、导入或调用 PyTorch/Transformers；模型层、视觉编码、KV cache 与贪心解码均由 Jittor 执行。

> 模型与数据采用 CC BY-NC 4.0，仅限非商业用途。模型文件不会提交到本仓库。

## 环境

- Python 3.11（由 `uv` 管理）
- Jittor 1.3.11.0
- NVIDIA CUDA GPU；建议普通图片至少留出 40GB 显存
- 首次运行会编译 Jittor CUDA 内核，耗时可能较长

```bash
proxy_on
uv sync
```

项目配置已把 uv 缓存放在项目内的 `.uv-cache/`。Jittor 编译缓存默认由 CLI 放在 `.jittor-cache/`。

## 1. 下载并转换权重

转换工具使用隔离的可选环境，允许 PyTorch；输出是 Jittor 能直接读取的四个 BF16 `.bin` 分片。中断后重新运行会复用已下载和已转换的分片。

```bash
proxy_on
uv run --isolated --extra convert geoagent-jittor-convert \
  --model-id ghost233lism/GeoAgent \
  --source-dir checkpoints/GeoAgent-hf \
  --output-dir checkpoints/GeoAgent-jittor
```

若原始 Hugging Face 仓库已在本地：

```bash
uv run --isolated --extra convert geoagent-jittor-convert \
  --local-only \
  --source-dir /path/to/GeoAgent \
  --output-dir checkpoints/GeoAgent-jittor
```

## 2. 单图推理

```bash
uv run geoagent-jittor \
  --model-path checkpoints/GeoAgent-jittor \
  --image /path/to/street.jpg \
  --max-new-tokens 2048
```

默认使用官方 GeoAgent system/user prompt、Qwen 官方动态分辨率上限、BF16 权重、FP32 attention/RMSNorm 累加、KV cache 和确定性贪心生成。FP32 累加用于避免 BF16 舍入误差在 GeoAgent 的尖锐注意力分布中被放大；因此本实现优先保证可用性和输出质量，而非速度。可用 `--attention-chunk-size` 降低注意力峰值显存（值越小越慢）。合法 JSON（包括常见的 Markdown JSON 代码围栏）会格式化输出；否则保留模型原文并给出警告。

## 开发检查

```bash
uv sync --group dev
uv run --group dev ruff check .
uv run --group dev pytest -q
```

## 实现来源与许可

架构和预处理公式依据 Hugging Face Transformers 4.55.4 中 Apache-2.0 许可的 Qwen2.5-VL 实现重新表达为 Jittor。仓库代码采用 Apache-2.0；GeoAgent 权重、其输出及上游项目仍受原作者 CC BY-NC 4.0 条款约束。
