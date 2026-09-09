from __future__ import annotations

import jittor as jt
import numpy as np

from geoagent_jittor.config import GeoAgentConfig, TextConfig, VisionConfig
from geoagent_jittor.modeling import GeoAgentForConditionalGeneration, TextModel, get_rope_index


def tiny_config() -> GeoAgentConfig:
    return GeoAgentConfig(
        text=TextConfig(
            vocab_size=32,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=3,
            num_key_value_heads=1,
            max_position_embeddings=128,
            rope_theta=10_000.0,
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
        image_token_id=22,
        video_token_id=23,
        vision_start_token_id=20,
        vision_end_token_id=21,
        vision_token_id=24,
        eos_token_ids=(31,),
        pad_token_id=0,
        torch_dtype="float32",
        min_pixels=16,
        max_pixels=256,
    )


def initialize_for_test(model: GeoAgentForConditionalGeneration) -> None:
    rng = np.random.default_rng(7)
    for _, parameter in model.named_parameters():
        values = rng.normal(0.0, 0.02, size=parameter.shape).astype(np.float32)
        parameter.update(jt.array(values))
    jt.sync_all()


def test_rope_index_for_one_merged_image_token():
    config = tiny_config()
    ids = np.asarray([[1, 20, 22, 21, 2]], dtype=np.int32)
    positions, delta = get_rope_index(ids, np.asarray([[1, 2, 2]], dtype=np.int32), config)
    assert positions.shape == (3, 1, 5)
    assert positions[:, 0, 2].tolist() == [2, 2, 2]
    assert delta == 0


def test_multimodal_rope_interleaves_temporal_height_width_sections():
    config = tiny_config().text
    model = TextModel(config, dtype="float32", attention_chunk_size=2)
    position_ids = np.asarray([[[1, 2]], [[3, 4]], [[5, 6]]], dtype=np.int32)

    cos, sin = model._position_embeddings(jt.array(position_ids), "float32")

    frequencies = position_ids[..., None].astype(np.float32) * model._inv_freq.reshape(1, 1, 1, -1)
    embeddings = np.concatenate((frequencies, frequencies), axis=-1)
    sections = config.rope_scaling["mrope_section"] * 2
    boundaries = np.cumsum([0, *sections]).tolist()
    selected = np.concatenate(
        [embeddings[index % 3, :, :, boundaries[index] : boundaries[index + 1]] for index in range(6)],
        axis=-1,
    )
    np.testing.assert_allclose(cos.numpy(), np.cos(selected)[:, None, :, :], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sin.numpy(), np.sin(selected)[:, None, :, :], rtol=1e-6, atol=1e-6)


def test_tiny_multimodal_prefill_and_cached_decode():
    jt.flags.use_cuda = 0
    config = tiny_config()
    model = GeoAgentForConditionalGeneration(config, dtype="float32", attention_chunk_size=2)
    initialize_for_test(model)
    model.eval()

    ids = np.asarray([[1, 20, 22, 21, 2]], dtype=np.int32)
    pixels = np.random.default_rng(3).normal(size=(4, 24)).astype(np.float32)
    grid = np.asarray([[1, 2, 2]], dtype=np.int32)
    output = model.prefill(ids, pixels, grid)

    assert tuple(output.logits.shape) == (1, 1, 32)
    assert len(output.past_key_values) == 2
    assert tuple(output.past_key_values[0][0].shape) == (1, 1, 5, 8)
    next_logits, next_cache = model.decode_step(
        5,
        absolute_position=ids.shape[1] + output.rope_delta,
        past_key_values=output.past_key_values,
    )
    assert tuple(next_logits.shape) == (1, 1, 32)
    assert tuple(next_cache[0][0].shape) == (1, 1, 6, 8)
    assert np.isfinite(next_logits.numpy()).all()
