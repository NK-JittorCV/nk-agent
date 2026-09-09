"""Qwen2.5-VL inference layers implemented with Jittor only.

The parameter tree intentionally mirrors the Hugging Face checkpoint names:
``visual.*``, ``model.*`` and ``lm_head.*``.  This keeps conversion and
shard-by-shard loading simple and auditable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

import jittor as jt
import numpy as np
from jittor import nn

from .config import GeoAgentConfig, TextConfig, VisionConfig


def _empty_parameter(shape: Iterable[int], dtype: str) -> jt.Var:
    value = jt.empty(tuple(int(item) for item in shape), dtype=dtype)
    value.stop_grad()
    return value


class Linear(nn.Module):
    """A non-initializing linear layer with PyTorch-compatible weight layout."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool, dtype: str) -> None:
        self.weight = _empty_parameter((out_features, in_features), dtype)
        self.bias = _empty_parameter((out_features,), dtype) if bias else None

    def execute(self, value: jt.Var) -> jt.Var:
        # PyTorch's CUDA linear kernel accumulates a BF16/FP16 GEMM and its
        # bias in FP32 before casting the result once.  Expressing this as a
        # low-precision matmul followed by ``+ bias`` rounds twice in Jittor.
        # A one-ULP Q/K error is enough to change GeoAgent's very sharp
        # attention distributions, so preserve the fused-linear numerics.
        if self.bias is not None and self.weight.dtype != "float32":
            output = nn.matmul_transpose(value.float32(), self.weight.float32())
            return (output + self.bias.float32()).cast(self.weight.dtype)
        output = nn.matmul_transpose(value, self.weight)
        return output + self.bias if self.bias is not None else output


class GELU(nn.Module):
    def execute(self, value: jt.Var) -> jt.Var:
        return nn.gelu(value)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, dtype: str) -> None:
        self.weight = _empty_parameter((hidden_size,), dtype)
        self.eps = float(eps)

    def execute(self, hidden_states: jt.Var) -> jt.Var:
        input_dtype = hidden_states.dtype
        value = hidden_states.float32()
        variance = (value * value).mean(dim=-1, keepdims=True)
        value = value * jt.rsqrt(variance + self.eps)
        return value.cast(input_dtype) * self.weight


def rotate_half(value: jt.Var) -> jt.Var:
    midpoint = value.shape[-1] // 2
    return jt.concat((-value[..., midpoint:], value[..., :midpoint]), dim=-1)


def repeat_kv(value: jt.Var, repeats: int) -> jt.Var:
    if repeats == 1:
        return value
    return jt.repeat_interleave(value, repeats, dim=1)


def _softmax_fp32(value: jt.Var, output_dtype: str) -> jt.Var:
    return nn.softmax(value.float32(), dim=-1).cast(output_dtype)


def _attention_chunks(
    query: jt.Var,
    key: jt.Var,
    value: jt.Var,
    *,
    scale: float,
    query_offset: int = 0,
    causal: bool,
    chunk_size: int,
) -> jt.Var:
    """Memory-bounded, FP32-accumulating attention.

    Qwen2.5-VL is unusually sensitive to rounding its attention scores to
    BF16 before softmax.  CUDA SDPA keeps these reductions in FP32; doing the
    same here avoids large changes in the selected token while the returned
    context retains the model dtype.
    """
    query_length = int(query.shape[2])
    key_length = int(key.shape[2])
    outputs: list[jt.Var] = []
    key_t = key.float32().transpose(0, 1, 3, 2)
    value_fp32 = value.float32()
    for start in range(0, query_length, chunk_size):
        end = min(start + chunk_size, query_length)
        query_chunk = query[:, :, start:end, :].float32()
        scores = jt.matmul(query_chunk, key_t) * scale
        if causal:
            query_positions = np.arange(query_offset + start, query_offset + end, dtype=np.int64)[:, None]
            key_positions = np.arange(key_length, dtype=np.int64)[None, :]
            mask = np.where(key_positions > query_positions, -1.0e30, 0.0).astype(np.float32)
            scores = scores + jt.array(mask).reshape(1, 1, end - start, key_length)
        probs = _softmax_fp32(scores, "float32")
        outputs.append(jt.matmul(probs, value_fp32).cast(query.dtype))
    return outputs[0] if len(outputs) == 1 else jt.concat(outputs, dim=2)


class VisionPatchProjection(nn.Module):
    def __init__(self, config: VisionConfig, dtype: str) -> None:
        self.weight = _empty_parameter(
            (
                config.hidden_size,
                config.in_channels,
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            ),
            dtype,
        )

    def execute(self, flattened_patches: jt.Var) -> jt.Var:
        weight = self.weight.reshape(self.weight.shape[0], -1)
        return nn.matmul_transpose(flattened_patches.cast(self.weight.dtype), weight)


class VisionPatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig, dtype: str) -> None:
        self.proj = VisionPatchProjection(config, dtype)

    def execute(self, flattened_patches: jt.Var) -> jt.Var:
        return self.proj(flattened_patches)


class VisionMLP(nn.Module):
    def __init__(self, config: VisionConfig, dtype: str) -> None:
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=True, dtype=dtype)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=True, dtype=dtype)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=True, dtype=dtype)

    def execute(self, hidden_states: jt.Var) -> jt.Var:
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class VisionAttention(nn.Module):
    def __init__(self, config: VisionConfig, dtype: str, chunk_size: int) -> None:
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim**-0.5
        self.chunk_size = int(chunk_size)
        self.qkv = Linear(config.hidden_size, config.hidden_size * 3, bias=True, dtype=dtype)
        self.proj = Linear(config.hidden_size, config.hidden_size, bias=True, dtype=dtype)

    def execute(
        self,
        hidden_states: jt.Var,
        cu_seqlens: list[int],
        cos: jt.Var,
        sin: jt.Var,
    ) -> jt.Var:
        sequence_length = int(hidden_states.shape[0])
        qkv = self.qkv(hidden_states).reshape(sequence_length, 3, self.num_heads, self.head_dim)
        query = qkv[:, 0, :, :]
        key = qkv[:, 1, :, :]
        value = qkv[:, 2, :, :]

        input_dtype = query.dtype
        cos = cos.unsqueeze(1).float32()
        sin = sin.unsqueeze(1).float32()
        query = (query.float32() * cos + rotate_half(query.float32()) * sin).cast(input_dtype)
        key = (key.float32() * cos + rotate_half(key.float32()) * sin).cast(input_dtype)

        segment_outputs: list[jt.Var] = []
        for segment_start, segment_end in pairwise(cu_seqlens):
            if segment_end <= segment_start:
                continue
            q_segment = query[segment_start:segment_end].transpose(1, 0, 2).unsqueeze(0)
            k_segment = key[segment_start:segment_end].transpose(1, 0, 2).unsqueeze(0)
            v_segment = value[segment_start:segment_end].transpose(1, 0, 2).unsqueeze(0)
            attended = _attention_chunks(
                q_segment,
                k_segment,
                v_segment,
                scale=self.scale,
                causal=False,
                chunk_size=self.chunk_size,
            )
            segment_outputs.append(attended.squeeze(0).transpose(1, 0, 2))
        output = segment_outputs[0] if len(segment_outputs) == 1 else jt.concat(segment_outputs, dim=0)
        return self.proj(output.reshape(sequence_length, -1))


class VisionBlock(nn.Module):
    def __init__(self, config: VisionConfig, dtype: str, chunk_size: int) -> None:
        self.norm1 = RMSNorm(config.hidden_size, 1e-6, dtype)
        self.norm2 = RMSNorm(config.hidden_size, 1e-6, dtype)
        self.attn = VisionAttention(config, dtype, chunk_size)
        self.mlp = VisionMLP(config, dtype)

    def execute(self, hidden_states: jt.Var, cu_seqlens: list[int], cos: jt.Var, sin: jt.Var) -> jt.Var:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, cos, sin)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class PatchMerger(nn.Module):
    def __init__(self, config: VisionConfig, dtype: str) -> None:
        merged_size = config.hidden_size * config.spatial_merge_size**2
        self.merged_size = merged_size
        self.ln_q = RMSNorm(config.hidden_size, 1e-6, dtype)
        self.mlp = nn.Sequential(
            Linear(merged_size, merged_size, bias=True, dtype=dtype),
            GELU(),
            Linear(merged_size, config.out_hidden_size, bias=True, dtype=dtype),
        )

    def execute(self, hidden_states: jt.Var) -> jt.Var:
        return self.mlp(self.ln_q(hidden_states).reshape(-1, self.merged_size))


def _vision_positions(grid_thw: np.ndarray, merge_size: int) -> np.ndarray:
    positions: list[np.ndarray] = []
    for temporal, height, width in grid_thw.tolist():
        h_ids = np.broadcast_to(np.arange(height, dtype=np.int32)[:, None], (height, width))
        h_ids = h_ids.reshape(height // merge_size, merge_size, width // merge_size, merge_size)
        h_ids = h_ids.transpose(0, 2, 1, 3).reshape(-1)
        w_ids = np.broadcast_to(np.arange(width, dtype=np.int32)[None, :], (height, width))
        w_ids = w_ids.reshape(height // merge_size, merge_size, width // merge_size, merge_size)
        w_ids = w_ids.transpose(0, 2, 1, 3).reshape(-1)
        positions.append(np.tile(np.stack((h_ids, w_ids), axis=-1), (temporal, 1)))
    return np.concatenate(positions, axis=0)


def _window_layout(grid_thw: np.ndarray, config: VisionConfig) -> tuple[np.ndarray, list[int], list[int]]:
    window_indices: list[np.ndarray] = []
    cumulative_window_lengths = [0]
    cumulative_full_lengths = [0]
    window_index_offset = 0
    window_side = config.window_size // config.spatial_merge_size // config.patch_size
    merge_unit = config.spatial_merge_size**2

    for temporal, height, width in grid_thw.tolist():
        llm_h = height // config.spatial_merge_size
        llm_w = width // config.spatial_merge_size
        index = np.arange(temporal * llm_h * llm_w, dtype=np.int32).reshape(temporal, llm_h, llm_w)
        pad_h = (window_side - llm_h % window_side) % window_side
        pad_w = (window_side - llm_w % window_side) % window_side
        padded = np.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)
        windows_h = (llm_h + pad_h) // window_side
        windows_w = (llm_w + pad_w) // window_side
        padded = padded.reshape(temporal, windows_h, window_side, windows_w, window_side)
        padded = padded.transpose(0, 1, 3, 2, 4).reshape(temporal, windows_h * windows_w, window_side, window_side)
        lengths = (padded != -100).sum(axis=(2, 3)).reshape(-1)
        valid = padded.reshape(-1)
        window_indices.append(valid[valid != -100] + window_index_offset)
        for length in lengths:
            if int(length):
                cumulative_window_lengths.append(cumulative_window_lengths[-1] + int(length) * merge_unit)
        image_tokens = int(temporal * height * width)
        cumulative_full_lengths.append(cumulative_full_lengths[-1] + image_tokens)
        window_index_offset += int(temporal * llm_h * llm_w)

    return np.concatenate(window_indices), cumulative_window_lengths, cumulative_full_lengths


class VisionTransformer(nn.Module):
    def __init__(self, config: VisionConfig, dtype: str, attention_chunk_size: int) -> None:
        self.config = config
        self.patch_embed = VisionPatchEmbed(config, dtype)
        self.blocks = nn.ModuleList(
            [VisionBlock(config, dtype, attention_chunk_size) for _ in range(config.depth)]
        )
        self.merger = PatchMerger(config, dtype)
        self._inv_freq = np.asarray(
            1.0
            / (
                10000.0
                ** (
                    np.arange(0, (config.hidden_size // config.num_heads) // 2, 2, dtype=np.float32)
                    / ((config.hidden_size // config.num_heads) // 2)
                )
            ),
            dtype=np.float32,
        )

    def execute(self, pixel_values: jt.Var, grid_thw: np.ndarray) -> jt.Var:
        hidden_states = self.patch_embed(pixel_values)
        merge_unit = self.config.spatial_merge_size**2
        sequence_length = int(hidden_states.shape[0])

        position_indices = _vision_positions(grid_thw, self.config.spatial_merge_size)
        positions = jt.array(position_indices).float32().unsqueeze(-1)
        inv_freq = jt.array(self._inv_freq).reshape(1, 1, -1)
        rotary = (positions * inv_freq).reshape(sequence_length, -1)

        window_index, window_cu, full_cu = _window_layout(grid_thw, self.config)
        window_index_var = jt.array(window_index.astype(np.int32))
        hidden_states = hidden_states.reshape(sequence_length // merge_unit, merge_unit, -1)[window_index_var]
        hidden_states = hidden_states.reshape(sequence_length, -1)
        rotary = rotary.reshape(sequence_length // merge_unit, merge_unit, -1)[window_index_var]
        rotary = rotary.reshape(sequence_length, -1)
        rotary = jt.concat((rotary, rotary), dim=-1)
        cos, sin = jt.cos(rotary), jt.sin(rotary)

        full_layers = set(self.config.fullatt_block_indexes)
        for index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, full_cu if index in full_layers else window_cu, cos, sin)

        hidden_states = self.merger(hidden_states)
        reverse_index = np.argsort(window_index).astype(np.int32)
        return hidden_states[jt.array(reverse_index)]


class TextMLP(nn.Module):
    def __init__(self, config: TextConfig, dtype: str) -> None:
        self.gate_proj = Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=dtype)
        self.up_proj = Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=dtype)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, bias=False, dtype=dtype)

    def execute(self, hidden_states: jt.Var) -> jt.Var:
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class TextAttention(nn.Module):
    def __init__(self, config: TextConfig, dtype: str, attention_chunk_size: int) -> None:
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.chunk_size = int(attention_chunk_size)
        self.q_proj = Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True, dtype=dtype)
        self.k_proj = Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True, dtype=dtype)
        self.v_proj = Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=True, dtype=dtype)
        self.o_proj = Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False, dtype=dtype)

    def execute(
        self,
        hidden_states: jt.Var,
        cos: jt.Var,
        sin: jt.Var,
        past_key_value: tuple[jt.Var, jt.Var] | None,
    ) -> tuple[jt.Var, tuple[jt.Var, jt.Var]]:
        batch_size, query_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).reshape(batch_size, query_length, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).reshape(batch_size, query_length, self.num_key_value_heads, self.head_dim)
        value = self.v_proj(hidden_states).reshape(batch_size, query_length, self.num_key_value_heads, self.head_dim)
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        query = query * cos + rotate_half(query) * sin
        key = key * cos + rotate_half(key) * sin
        past_length = 0
        if past_key_value is not None:
            past_length = int(past_key_value[0].shape[2])
            key = jt.concat((past_key_value[0], key), dim=2)
            value = jt.concat((past_key_value[1], value), dim=2)
        present = (key, value)

        expanded_key = repeat_kv(key, self.num_key_value_groups)
        expanded_value = repeat_kv(value, self.num_key_value_groups)
        output = _attention_chunks(
            query,
            expanded_key,
            expanded_value,
            scale=self.scale,
            query_offset=past_length,
            causal=True,
            chunk_size=self.chunk_size,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, query_length, -1)
        return self.o_proj(output), present


class DecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, dtype: str, attention_chunk_size: int) -> None:
        self.self_attn = TextAttention(config, dtype, attention_chunk_size)
        self.mlp = TextMLP(config, dtype)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)

    def execute(
        self,
        hidden_states: jt.Var,
        cos: jt.Var,
        sin: jt.Var,
        past_key_value: tuple[jt.Var, jt.Var] | None,
    ) -> tuple[jt.Var, tuple[jt.Var, jt.Var]]:
        attention_output, present = self.self_attn(
            self.input_layernorm(hidden_states), cos, sin, past_key_value
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, dtype: str) -> None:
        self.weight = _empty_parameter((vocab_size, hidden_size), dtype)

    def execute(self, token_ids: jt.Var) -> jt.Var:
        return self.weight[token_ids]


class TextModel(nn.Module):
    def __init__(self, config: TextConfig, dtype: str, attention_chunk_size: int) -> None:
        self.config = config
        self.embed_tokens = TokenEmbedding(config.vocab_size, config.hidden_size, dtype)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, dtype, attention_chunk_size) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        head_dim = config.hidden_size // config.num_attention_heads
        self._inv_freq = np.asarray(
            1.0 / (config.rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)),
            dtype=np.float32,
        )

    def _position_embeddings(self, position_ids: jt.Var, dtype: str) -> tuple[jt.Var, jt.Var]:
        positions = position_ids.float32().unsqueeze(-1)
        inv_freq = jt.array(self._inv_freq).reshape(1, 1, 1, -1)
        frequencies = positions * inv_freq
        embeddings = jt.concat((frequencies, frequencies), dim=-1)
        cos_all, sin_all = jt.cos(embeddings), jt.sin(embeddings)

        sections = [int(value) for value in self.config.rope_scaling["mrope_section"]] * 2
        boundaries = np.cumsum([0, *sections]).tolist()
        cos_parts = [
            cos_all[index % 3, :, :, boundaries[index] : boundaries[index + 1]]
            for index in range(len(sections))
        ]
        sin_parts = [
            sin_all[index % 3, :, :, boundaries[index] : boundaries[index + 1]]
            for index in range(len(sections))
        ]
        cos = jt.concat(cos_parts, dim=-1).unsqueeze(1).cast(dtype)
        sin = jt.concat(sin_parts, dim=-1).unsqueeze(1).cast(dtype)
        return cos, sin

    def execute(
        self,
        *,
        input_ids: jt.Var | None = None,
        inputs_embeds: jt.Var | None = None,
        position_ids: jt.Var,
        past_key_values: list[tuple[jt.Var, jt.Var]] | None = None,
    ) -> tuple[jt.Var, list[tuple[jt.Var, jt.Var]]]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        cos, sin = self._position_embeddings(position_ids, hidden_states.dtype)
        new_cache: list[tuple[jt.Var, jt.Var]] = []
        for index, layer in enumerate(self.layers):
            layer_past = None if past_key_values is None else past_key_values[index]
            hidden_states, present = layer(hidden_states, cos, sin, layer_past)
            new_cache.append(present)
        return self.norm(hidden_states), new_cache


def get_rope_index(
    input_ids: np.ndarray,
    image_grid_thw: np.ndarray,
    config: GeoAgentConfig,
) -> tuple[np.ndarray, int]:
    """Calculate Qwen2.5-VL 3-D positions for a batch of one and one image."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("only batch size 1 is supported")
    ids = input_ids[0].tolist()
    image_positions = np.flatnonzero(input_ids[0] == config.image_token_id)
    if image_positions.size == 0:
        positions = np.arange(input_ids.shape[1], dtype=np.int32)
        return np.broadcast_to(positions, (3, 1, positions.size)).copy(), 0
    if len(image_grid_thw) != 1:
        raise ValueError("single-image inference expects exactly one image grid")
    first_image = int(image_positions[0])
    if image_positions.tolist() != list(range(first_image, first_image + image_positions.size)):
        raise ValueError("image placeholder tokens must be contiguous")

    temporal, height, width = (int(value) for value in image_grid_thw[0])
    llm_h = height // config.vision.spatial_merge_size
    llm_w = width // config.vision.spatial_merge_size
    expected = temporal * llm_h * llm_w
    if image_positions.size != expected:
        raise ValueError(f"image token count {image_positions.size} does not match grid feature count {expected}")

    pieces: list[np.ndarray] = []
    if first_image:
        pieces.append(np.broadcast_to(np.arange(first_image, dtype=np.int32), (3, first_image)).copy())
    start = first_image
    t_index = np.repeat(np.arange(temporal, dtype=np.int32), llm_h * llm_w)
    h_index = np.tile(np.repeat(np.arange(llm_h, dtype=np.int32), llm_w), temporal)
    w_index = np.tile(np.arange(llm_w, dtype=np.int32), temporal * llm_h)
    vision_positions = np.stack((t_index, h_index, w_index), axis=0) + start
    pieces.append(vision_positions)

    consumed = first_image + expected
    next_start = int(vision_positions.max()) + 1
    if consumed < len(ids):
        tail_length = len(ids) - consumed
        tail = np.arange(tail_length, dtype=np.int32) + next_start
        pieces.append(np.broadcast_to(tail, (3, tail_length)).copy())

    positions = np.concatenate(pieces, axis=1)[:, None, :]
    delta = int(positions.max()) + 1 - len(ids)
    return positions, delta


def _replace_image_embeddings(
    token_ids: np.ndarray,
    token_embeddings: jt.Var,
    image_embeddings: jt.Var,
    image_token_id: int,
) -> jt.Var:
    positions = np.flatnonzero(token_ids[0] == image_token_id)
    if positions.size != image_embeddings.shape[0]:
        raise ValueError(
            f"image feature/token mismatch: {image_embeddings.shape[0]} features for {positions.size} placeholders"
        )
    start, end = int(positions[0]), int(positions[-1]) + 1
    parts: list[jt.Var] = []
    if start:
        parts.append(token_embeddings[:, :start, :])
    parts.append(image_embeddings.unsqueeze(0).cast(token_embeddings.dtype))
    if end < token_embeddings.shape[1]:
        parts.append(token_embeddings[:, end:, :])
    return parts[0] if len(parts) == 1 else jt.concat(parts, dim=1)


@dataclass(slots=True)
class PrefillOutput:
    logits: jt.Var
    past_key_values: list[tuple[jt.Var, jt.Var]]
    rope_delta: int
    prompt_length: int


class GeoAgentForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: GeoAgentConfig,
        *,
        dtype: str = "bfloat16",
        attention_chunk_size: int = 256,
    ) -> None:
        self.config = config
        self.dtype = dtype
        self.visual = VisionTransformer(config.vision, dtype, attention_chunk_size)
        self.model = TextModel(config.text, dtype, attention_chunk_size)
        self.lm_head = Linear(config.text.hidden_size, config.text.vocab_size, bias=False, dtype=dtype)

    def prefill(
        self,
        input_ids: np.ndarray,
        pixel_values: np.ndarray,
        image_grid_thw: np.ndarray,
    ) -> PrefillOutput:
        ids = jt.array(input_ids.astype(np.int32, copy=False))
        token_embeddings = self.model.embed_tokens(ids)
        image_embeddings = self.visual(jt.array(pixel_values).cast(self.dtype), image_grid_thw)
        inputs_embeds = _replace_image_embeddings(
            input_ids, token_embeddings, image_embeddings, self.config.image_token_id
        )
        positions, rope_delta = get_rope_index(input_ids, image_grid_thw, self.config)
        hidden_states, cache = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=jt.array(positions.astype(np.int32, copy=False)),
            past_key_values=None,
        )
        logits = self.lm_head(hidden_states[:, -1:, :])
        return PrefillOutput(logits, cache, rope_delta, input_ids.shape[1])

    def decode_step(
        self,
        token_id: int,
        *,
        absolute_position: int,
        past_key_values: list[tuple[jt.Var, jt.Var]],
    ) -> tuple[jt.Var, list[tuple[jt.Var, jt.Var]]]:
        ids = jt.array(np.asarray([[token_id]], dtype=np.int32))
        positions = jt.array(np.full((3, 1, 1), absolute_position, dtype=np.int32))
        hidden_states, cache = self.model(
            input_ids=ids,
            position_ids=positions,
            past_key_values=past_key_values,
        )
        return self.lm_head(hidden_states[:, -1:, :]), cache

    def generate(
        self,
        input_ids: np.ndarray,
        pixel_values: np.ndarray,
        image_grid_thw: np.ndarray,
        *,
        max_new_tokens: int,
    ) -> list[int]:
        generated: list[int] = []
        with jt.no_grad():
            output = self.prefill(input_ids, pixel_values, image_grid_thw)
            logits = output.logits
            cache = output.past_key_values
            next_position = output.prompt_length + output.rope_delta
            for _ in range(max_new_tokens):
                next_token = int(np.asarray(logits.float32().numpy()).reshape(-1).argmax())
                if next_token in self.config.eos_token_ids:
                    break
                generated.append(next_token)
                logits, cache = self.decode_step(
                    next_token,
                    absolute_position=next_position,
                    past_key_values=cache,
                )
                next_position += 1
        return generated
