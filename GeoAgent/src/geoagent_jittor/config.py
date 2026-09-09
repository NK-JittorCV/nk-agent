"""Configuration objects for the Jittor Qwen2.5-VL implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class VisionConfig:
    depth: int = 32
    hidden_size: int = 1280
    hidden_act: str = "silu"
    intermediate_size: int = 3420
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    tokens_per_second: int = 2
    window_size: int = 112
    out_hidden_size: int = 3584
    fullatt_block_indexes: list[int] = field(default_factory=lambda: [7, 15, 23, 31])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisionConfig:
        names = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in names})


@dataclass(slots=True)
class TextConfig:
    vocab_size: int = 152064
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 128000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    use_cache: bool = True
    use_sliding_window: bool = False
    sliding_window: int | None = None
    max_window_layers: int = 28
    layer_types: list[str] = field(default_factory=list)
    attention_dropout: float = 0.0
    rope_scaling: dict[str, Any] = field(
        default_factory=lambda: {"mrope_section": [16, 24, 24], "rope_type": "default"}
    )

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if self.rope_scaling is None:
            self.rope_scaling = {"mrope_section": [16, 24, 24], "rope_type": "default"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextConfig:
        names = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in names})


@dataclass(slots=True)
class GeoAgentConfig:
    text: TextConfig
    vision: VisionConfig
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    eos_token_ids: tuple[int, ...] = (151645, 151643)
    pad_token_id: int = 151643
    torch_dtype: str = "bfloat16"
    min_pixels: int = 3136
    max_pixels: int = 12845056
    image_mean: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    image_std: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> GeoAgentConfig:
        model_dir = Path(model_dir)
        with (model_dir / "config.json").open(encoding="utf-8") as handle:
            raw = json.load(handle)

        text_raw = raw.get("text_config") or raw
        vision_raw = raw.get("vision_config") or {}

        processor_path = model_dir / "preprocessor_config.json"
        processor: dict[str, Any] = {}
        if processor_path.exists():
            with processor_path.open(encoding="utf-8") as handle:
                processor = json.load(handle)

        generation_path = model_dir / "generation_config.json"
        generation: dict[str, Any] = {}
        if generation_path.exists():
            with generation_path.open(encoding="utf-8") as handle:
                generation = json.load(handle)
        eos = generation.get("eos_token_id", raw.get("eos_token_id", [151645, 151643]))
        if isinstance(eos, int):
            eos = [eos]

        return cls(
            text=TextConfig.from_dict(text_raw),
            vision=VisionConfig.from_dict(vision_raw),
            image_token_id=int(raw.get("image_token_id", 151655)),
            video_token_id=int(raw.get("video_token_id", 151656)),
            vision_start_token_id=int(raw.get("vision_start_token_id", 151652)),
            vision_end_token_id=int(raw.get("vision_end_token_id", 151653)),
            vision_token_id=int(raw.get("vision_token_id", 151654)),
            eos_token_ids=tuple(int(item) for item in eos),
            pad_token_id=int(raw.get("pad_token_id", 151643)),
            torch_dtype=str(raw.get("torch_dtype", text_raw.get("torch_dtype", "bfloat16"))),
            min_pixels=int(processor.get("min_pixels", 3136)),
            max_pixels=int(processor.get("max_pixels", 12845056)),
            image_mean=tuple(processor.get("image_mean", (0.48145466, 0.4578275, 0.40821073))),
            image_std=tuple(processor.get("image_std", (0.26862954, 0.26130258, 0.27577711))),
        )

