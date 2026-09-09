"""Strict, shard-by-shard loading of converted checkpoints into Jittor."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import jittor as jt

from .modeling import GeoAgentForConditionalGeneration

LOGGER = logging.getLogger(__name__)
INDEX_FILENAME = "jittor_model.bin.index.json"


def load_converted_weights(model: GeoAgentForConditionalGeneration, model_dir: str | Path) -> None:
    model_dir = Path(model_dir)
    index_path = model_dir / INDEX_FILENAME
    if not index_path.is_file():
        raise FileNotFoundError(
            f"missing converted checkpoint index: {index_path}; run geoagent-jittor-convert first"
        )
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map: dict[str, str] = index.get("weight_map", {})
    if not weight_map:
        raise ValueError(f"empty or invalid checkpoint index: {index_path}")

    parameters = dict(model.named_parameters())
    expected = set(parameters)
    indexed = set(weight_map)
    missing_from_index = expected - indexed
    unexpected_in_index = indexed - expected
    if missing_from_index or unexpected_in_index:
        details = []
        if missing_from_index:
            details.append(f"missing={sorted(missing_from_index)[:20]}")
        if unexpected_in_index:
            details.append(f"unexpected={sorted(unexpected_in_index)[:20]}")
        raise ValueError("checkpoint/model parameter mismatch: " + "; ".join(details))

    shard_names = list(dict.fromkeys(weight_map.values()))
    loaded_names: set[str] = set()
    for shard_index, shard_name in enumerate(shard_names, start=1):
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"checkpoint shard is missing: {shard_path}")
        LOGGER.info("Loading weight shard %d/%d: %s", shard_index, len(shard_names), shard_path.name)
        state = jt.load(str(shard_path))
        if not isinstance(state, dict):
            raise TypeError(f"checkpoint shard did not contain a state dictionary: {shard_path}")
        for name, value in state.items():
            if name not in parameters:
                raise KeyError(f"unexpected parameter {name!r} in {shard_path.name}")
            target = parameters[name]
            if tuple(target.shape) != tuple(value.shape):
                raise ValueError(
                    f"shape mismatch for {name}: model {tuple(target.shape)}, checkpoint {tuple(value.shape)}"
                )
            if value.dtype != target.dtype:
                value = value.cast(target.dtype)
            target.update(value)
            loaded_names.add(name)
        jt.sync_all(True)
        del state

    missing = expected - loaded_names
    if missing:
        raise ValueError(f"checkpoint did not load {len(missing)} model parameters: {sorted(missing)[:20]}")
    LOGGER.info("Loaded %d parameters from %d shards", len(loaded_names), len(shard_names))

