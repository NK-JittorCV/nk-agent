<div align="center">

# GeoAgent: Jittor Implementation

Pure-Jittor inference for **GeoAgent: Learning to Geolocate Everywhere with Reinforced Geographic Characteristics**

[Project Page](https://ghost233lism.github.io/GeoAgent-page/) ·
[Paper](https://huggingface.co/papers/2602.12617) ·
[Model Weights](https://huggingface.co/ghost233lism/GeoAgent) ·
[Original Code](https://github.com/HVision-NKU/GeoAgent)

<p>
  <img src="https://img.shields.io/badge/Python-3.10--3.12-blue.svg" alt="Python 3.10–3.12">
  <img src="https://img.shields.io/badge/Jittor-1.3.11-orange.svg" alt="Jittor 1.3.11">
  <img src="https://img.shields.io/badge/License-Apache--2.0-green.svg" alt="Apache-2.0">
</p>

</div>

## Overview

[GeoAgent](https://github.com/HVision-NKU/GeoAgent) is a vision-language model for image geolocation. Given a single image, it analyzes geographic clues, produces an interpretable reasoning chain, and predicts a fine-grained location.

This directory provides a pure [Jittor](https://github.com/Jittor/jittor) implementation of GeoAgent based on Qwen2.5-VL-7B. The inference process does not install, import, or invoke PyTorch or Transformers: the language model, vision encoder, multimodal RoPE, KV cache, and greedy decoding are all implemented with Jittor. PyTorch is used only in the optional, isolated checkpoint-conversion environment.

### Features

- Pure-Jittor single-image inference with BF16 weights.
- Official GeoAgent prompts and Qwen2.5-VL dynamic image resolution.
- FP32 accumulation for attention and RMSNorm to improve numerical stability.
- KV-cached deterministic greedy decoding.
- Resumable download and shard-by-shard checkpoint conversion.
- Configurable chunked attention for lower peak memory usage.
- Structured JSON output formatting with raw-text fallback.

## Installation

### Requirements

- Linux and an NVIDIA CUDA GPU (40 GB or more GPU memory is recommended)
- Python `>=3.10,<3.13`
- Jittor `1.3.11.0`
- [uv](https://docs.astral.sh/uv/)

The first run compiles Jittor CUDA kernels and can therefore take longer than later runs.

```bash
git clone https://github.com/NK-JittorCV/nk-agent.git
cd nk-agent/GeoAgent
uv sync
```

The uv package cache and Jittor compilation cache are kept inside `.uv-cache/` and `.jittor-cache/`, respectively.

## Model Weights

The original GeoAgent checkpoint is available from the [GeoAgent model page on Hugging Face](https://huggingface.co/ghost233lism/GeoAgent). Convert it into four BF16 shards that can be loaded directly by Jittor:

```bash
uv run --isolated --extra convert geoagent-jittor-convert \
  --model-id ghost233lism/GeoAgent \
  --source-dir checkpoints/GeoAgent-hf \
  --output-dir checkpoints/GeoAgent-jittor
```

The command resumes interrupted downloads and reuses completed shards. If the original Hugging Face checkpoint is already available locally, skip downloading it:

```bash
uv run --isolated --extra convert geoagent-jittor-convert \
  --local-only \
  --source-dir /path/to/GeoAgent \
  --output-dir checkpoints/GeoAgent-jittor
```

## Quick Inference

```bash
uv run geoagent-jittor \
  --model-path checkpoints/GeoAgent-jittor \
  --image examples/test.jpg \
  --max-new-tokens 2048
```

The command prints formatted JSON when the model returns valid JSON, including the common Markdown JSON fence. Otherwise, the original model response is preserved and a warning is emitted.

### Main Options

| Option | Description |
|---|---|
| `--model-path` | Directory containing the converted Jittor checkpoint. |
| `--image` | Path to one local input image. |
| `--max-new-tokens` | Maximum number of newly generated tokens. |
| `--dtype` | Model dtype: `bfloat16` (default), `float16`, or `float32`. |
| `--attention-chunk-size` | Smaller values reduce attention peak memory at the cost of speed. |
| `--jittor-home` | Custom writable directory for the Jittor compilation cache. |

## Jittor Inference Speed

The following is a single-run reference measurement of this implementation. It intentionally does not compare against PyTorch. Actual latency varies with image resolution, output length, GPU load, and whether Jittor kernels have already been compiled.

| Item | Setting / Result |
|---|---|
| GPU | NVIDIA RTX 6000D, 96 GB (Compute Capability 12.0) |
| Software | Python 3.11, Jittor 1.3.11.0, CUDA 12.9 |
| Precision | BF16 weights; FP32 attention/RMSNorm accumulation |
| Input | `examples/test.jpg`, 960 × 640, 782 image tokens |
| Generation | 128 new tokens, token-limit stop |
| Weight loading | 26.7 s |
| Generation | 58.9 s (about 2.17 new tokens/s) |
| End-to-end process | 93.2 s |

The Jittor compilation cache was already warm. The generation timing covers visual/text prefill plus autoregressive decoding, so the derived throughput is a workload-level figure rather than a steady-state decode-only rate. End-to-end time additionally includes process and framework startup, preprocessing, checkpoint loading, and output rendering.

## Development

```bash
uv sync --group dev
uv run --group dev ruff check .
uv run --group dev pytest -q
```

## Citation

```bibtex
@article{jin2026geoagent,
  title={GeoAgent: Learning to Geolocate Everywhere with Reinforced Geographic Characteristics},
  author={Jin, Modi and Zhang, Yiming and Sun, Boyuan and Zhang, Dingwen and Cheng, Ming-Ming and Hou, Qibin},
  journal={arXiv preprint arXiv:2602.12617},
  year={2026}
}
```

## License and Acknowledgements

This Jittor implementation is released under the Apache-2.0 license. GeoAgent weights, model outputs, and upstream data remain subject to the original [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/) and are intended for non-commercial use.

The architecture and preprocessing formulas were reimplemented from the Apache-2.0-licensed Qwen2.5-VL implementation in Hugging Face Transformers 4.55.4. We thank the authors of [GeoAgent](https://github.com/HVision-NKU/GeoAgent), [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), and [Jittor](https://github.com/Jittor/jittor).
