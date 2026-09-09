from __future__ import annotations

import json

from geoagent_jittor.config import GeoAgentConfig


def test_loads_nested_huggingface_config(tmp_path):
    config = {
        "image_token_id": 25,
        "video_token_id": 26,
        "vision_start_token_id": 22,
        "vision_end_token_id": 23,
        "vision_token_id": 24,
        "pad_token_id": 3,
        "torch_dtype": "bfloat16",
        "text_config": {
            "vocab_size": 64,
            "hidden_size": 24,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "rope_scaling": {"mrope_section": [1, 1, 2]},
        },
        "vision_config": {
            "depth": 2,
            "hidden_size": 24,
            "intermediate_size": 48,
            "num_heads": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 24,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"min_pixels": 16, "max_pixels": 256}), encoding="utf-8"
    )
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [4, 3]}), encoding="utf-8")

    loaded = GeoAgentConfig.from_directory(tmp_path)

    assert loaded.text.hidden_size == 24
    assert loaded.vision.patch_size == 2
    assert loaded.image_token_id == 25
    assert loaded.eos_token_ids == (4, 3)
    assert loaded.max_pixels == 256

