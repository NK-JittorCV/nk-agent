from __future__ import annotations

import numpy as np
from PIL import Image

from geoagent_jittor.config import GeoAgentConfig, TextConfig, VisionConfig
from geoagent_jittor.processing import GeoAgentProcessor, smart_resize
from geoagent_jittor.prompts import build_chat_prompt


def tiny_config() -> GeoAgentConfig:
    return GeoAgentConfig(
        text=TextConfig(
            vocab_size=32,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=3,
            num_key_value_heads=1,
            rope_scaling={"mrope_section": [1, 1, 2]},
        ),
        vision=VisionConfig(
            depth=2,
            hidden_size=24,
            intermediate_size=48,
            num_heads=3,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
            window_size=4,
            out_hidden_size=24,
            fullatt_block_indexes=[1],
        ),
        min_pixels=16,
        max_pixels=256,
        image_token_id=22,
        vision_start_token_id=20,
        vision_end_token_id=21,
    )


def test_smart_resize_matches_patch_factor():
    assert smart_resize(101, 203, factor=28, min_pixels=3136, max_pixels=12_845_056) == (112, 196)
    assert smart_resize(4, 4, factor=4, min_pixels=16, max_pixels=256) == (4, 4)


def test_image_preprocessing_layout(tmp_path):
    image_path = tmp_path / "image.png"
    pixels = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    Image.fromarray(pixels).save(image_path)
    processor = object.__new__(GeoAgentProcessor)
    processor.config = tiny_config()

    flattened, grid = processor.preprocess_image(image_path)

    assert grid.tolist() == [[1, 2, 2]]
    assert flattened.shape == (4, 24)
    assert flattened.dtype == np.float32
    assert np.isfinite(flattened).all()


def test_prompt_repeats_exact_image_placeholder_count():
    prompt = build_chat_prompt(7)
    assert prompt.count("<|image_pad|>") == 7
    assert prompt.endswith("<|im_start|>assistant\n")

