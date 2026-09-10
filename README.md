<div align="center">

# Jittor Implementations of AI Agents

An extensible collection of agent models implemented with the [Jittor](https://github.com/Jittor/jittor) deep learning framework

<p>
  <img src="https://img.shields.io/badge/framework-Jittor-orange.svg" alt="Jittor">
  <img src="https://img.shields.io/badge/Agent%20Projects-1-blue.svg" alt="Agent projects">
  <img src="https://img.shields.io/badge/PR-welcome-55EB99.svg" alt="PR welcome">
</p>

</div>

This repository brings modern AI agent models to Jittor. Each project lives in a self-contained subdirectory with its own environment, model-weight instructions, inference entry point, performance notes, tests, and license information. GeoAgent is the first implementation; more agent projects will be added over time.

## Projects

### GeoAgent

[GeoAgent](https://ghost233lism.github.io/GeoAgent-page/) is a vision-language agent for worldwide image geolocation. It reasons from visual geographic clues—including language, road infrastructure, architecture, and natural landscapes—to produce an interpretable chain of thought and a fine-grained location prediction.

The Jittor implementation includes the Qwen2.5-VL-7B vision encoder and language model, multimodal position encoding, KV-cached decoding, official GeoAgent prompts, checkpoint conversion, and a command-line interface for single-image inference. Its runtime path is pure Jittor and does not depend on PyTorch or Transformers.

| Project | Task | Base Model | Status | Resources |
|---|---|---|---|---|
| **GeoAgent** | Image geolocation | Qwen2.5-VL-7B | Inference available | [Jittor Code](./GeoAgent) · [Project Page](https://ghost233lism.github.io/GeoAgent-page/) · [Model Weights](https://huggingface.co/ghost233lism/GeoAgent) · [Original Code](https://github.com/HVision-NKU/GeoAgent) |

<p align="center">
  <img src="GeoAgent/examples/test.jpg" width="60%" alt="GeoAgent example input">
</p>

For installation, checkpoint conversion, inference commands, and measured Jittor speed, see the [GeoAgent documentation](./GeoAgent/README.md).

## About Jittor

[Jittor](https://cg.cs.tsinghua.edu.cn/jittor/) is a high-performance deep learning framework built around just-in-time compilation and meta-operators. It dynamically compiles and optimizes computation graphs for the target hardware, while retaining a concise Python interface for model development.

The implementations in this repository use Jittor for model construction and inference, including framework-native operators, memory management, and CUDA execution. Project-specific conversion tools may use an isolated environment to translate upstream checkpoints, but the final inference runtime remains Jittor-native.

## Repository Layout

```text
nk-agent/
├── README.md
└── GeoAgent/
    ├── README.md
    ├── src/
    ├── tests/
    └── examples/
```

Future agent implementations will be added as peer directories alongside `GeoAgent/` and listed in the project table above.

## Contributing

Issues and pull requests are welcome. When contributing a new agent, keep its dependencies, usage documentation, checkpoint instructions, tests, and licensing notes inside the corresponding project directory.

## License

Licensing is documented independently within each project. Model weights, datasets, and upstream code may use licenses different from the Jittor implementation; consult the relevant project README before use.
