"""Torch-free tokenizer and image preprocessing for one Qwen2.5-VL image."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tokenizers import Tokenizer

from .config import GeoAgentConfig
from .prompts import build_chat_prompt


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Match Qwen2-VL's aspect-preserving, patch-aligned resize rule."""
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image dimensions: {height}x{width}")
    ratio = max(height, width) / min(height, width)
    if ratio > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {ratio}")

    resized_h = round(height / factor) * factor
    resized_w = round(width / factor) * factor
    if resized_h * resized_w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_h = max(factor, math.floor(height / beta / factor) * factor)
        resized_w = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_h * resized_w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_h = math.ceil(height * beta / factor) * factor
        resized_w = math.ceil(width * beta / factor) * factor
    return resized_h, resized_w


@dataclass(slots=True)
class PreparedInput:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    pixel_values: np.ndarray
    image_grid_thw: np.ndarray
    prompt: str


class GeoAgentProcessor:
    """A deliberately small replacement for Transformers' Qwen2.5-VL processor."""

    def __init__(self, model_dir: str | Path, config: GeoAgentConfig | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.config = config or GeoAgentConfig.from_directory(self.model_dir)
        tokenizer_path = self.model_dir / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"missing tokenizer: {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def preprocess_image(self, image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        vision = self.config.vision
        factor = vision.patch_size * vision.spatial_merge_size
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            height, width = image.height, image.width
            resized_h, resized_w = smart_resize(
                height,
                width,
                factor=factor,
                min_pixels=self.config.min_pixels,
                max_pixels=self.config.max_pixels,
            )
            image = image.resize((resized_w, resized_h), resample=Image.Resampling.BICUBIC)
            pixels = np.asarray(image, dtype=np.float32)

        pixels = pixels / np.float32(255.0)
        mean = np.asarray(self.config.image_mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.config.image_std, dtype=np.float32).reshape(1, 1, 3)
        pixels = ((pixels - mean) / std).transpose(2, 0, 1)

        frames = pixels[np.newaxis, ...]
        temporal = vision.temporal_patch_size
        if frames.shape[0] % temporal:
            repeats = temporal - frames.shape[0] % temporal
            frames = np.concatenate([frames, np.repeat(frames[-1:], repeats, axis=0)], axis=0)

        channel = frames.shape[1]
        grid_t = frames.shape[0] // temporal
        grid_h = resized_h // vision.patch_size
        grid_w = resized_w // vision.patch_size
        merge = vision.spatial_merge_size
        patch = vision.patch_size
        patches = frames.reshape(
            grid_t,
            temporal,
            channel,
            grid_h // merge,
            merge,
            patch,
            grid_w // merge,
            merge,
            patch,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        patches = np.ascontiguousarray(
            patches.reshape(grid_t * grid_h * grid_w, channel * temporal * patch * patch),
            dtype=np.float32,
        )
        grid = np.asarray([[grid_t, grid_h, grid_w]], dtype=np.int32)
        return patches, grid

    def prepare(self, image_path: str | Path) -> PreparedInput:
        pixel_values, grid = self.preprocess_image(image_path)
        merge_length = self.config.vision.spatial_merge_size**2
        image_token_count = int(np.prod(grid[0], dtype=np.int64) // merge_length)
        prompt = build_chat_prompt(image_token_count)
        ids = np.asarray(self.tokenizer.encode(prompt, add_special_tokens=False).ids, dtype=np.int32)[None, :]
        actual_count = int(np.count_nonzero(ids == self.config.image_token_id))
        if actual_count != image_token_count:
            raise ValueError(f"image placeholder mismatch: prompt has {actual_count}, expected {image_token_count}")
        return PreparedInput(
            input_ids=ids,
            attention_mask=np.ones_like(ids, dtype=np.int32),
            pixel_values=pixel_values,
            image_grid_thw=grid,
            prompt=prompt,
        )

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

