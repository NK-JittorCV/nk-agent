"""Download GeoAgent and convert safetensors shards to Jittor-readable .bin files."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_ID = "ghost233lism/GeoAgent"
OUTPUT_INDEX = "jittor_model.bin.index.json"
SUPPORT_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


def _require_conversion_dependencies():
    try:
        import torch
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError(
            "conversion dependencies are missing; run with `uv run --isolated --extra convert "
            "geoagent-jittor-convert ...`"
        ) from error
    return torch, snapshot_download, safe_open


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _download(model_id: str, source_dir: Path, revision: str | None) -> str | None:
    _, snapshot_download, _ = _require_conversion_dependencies()
    source_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s to %s (existing files are resumed)", model_id, source_dir)
    resolved = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=source_dir,
        allow_patterns=[*SUPPORT_FILES, "model*.safetensors", "model.safetensors.index.json", "README.md"],
    )
    snapshot_marker = Path(resolved) / ".cache" / "huggingface" / "download"
    return revision or (snapshot_marker.name if snapshot_marker.exists() else None)


def convert_checkpoint(source_dir: Path, output_dir: Path, *, model_id: str, revision: str | None) -> Path:
    torch, _, safe_open = _require_conversion_dependencies()
    source_index_path = source_dir / "model.safetensors.index.json"
    if not source_index_path.is_file():
        raise FileNotFoundError(f"missing source weight index: {source_index_path}")
    with source_index_path.open(encoding="utf-8") as handle:
        source_index = json.load(handle)
    source_map: dict[str, str] = source_index["weight_map"]
    source_shards = list(dict.fromkeys(source_map.values()))
    output_dir.mkdir(parents=True, exist_ok=True)

    output_map: dict[str, str] = {}
    shard_metadata: dict[str, dict[str, object]] = {}
    for shard_number, source_name in enumerate(source_shards, start=1):
        source_path = source_dir / source_name
        if not source_path.is_file():
            raise FileNotFoundError(f"missing source shard: {source_path}")
        output_name = source_name.replace(".safetensors", ".bin")
        output_path = output_dir / output_name
        names = [name for name, filename in source_map.items() if filename == source_name]

        if output_path.is_file() and output_path.stat().st_size > 0:
            LOGGER.info("Reusing converted shard %d/%d: %s", shard_number, len(source_shards), output_name)
        else:
            LOGGER.info("Converting shard %d/%d: %s", shard_number, len(source_shards), source_name)
            tensors: dict[str, object] = {}
            with safe_open(source_path, framework="pt", device="cpu") as reader:
                checkpoint_names = set(reader.keys())
                if checkpoint_names != set(names):
                    raise ValueError(f"tensor list does not agree with source index for {source_name}")
                for name in names:
                    tensors[name] = reader.get_tensor(name)
            temporary = output_path.with_suffix(output_path.suffix + ".partial")
            torch.save(tensors, temporary)
            os.replace(temporary, output_path)
            del tensors

        for name in names:
            output_map[name] = output_name
        shard_metadata[output_name] = {
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
            "tensor_count": len(names),
        }

    for filename in SUPPORT_FILES:
        source_path = source_dir / filename
        if source_path.is_file():
            shutil.copy2(source_path, output_dir / filename)

    index = {
        "metadata": {
            "format": "pytorch-zip-readable-by-jittor",
            "model_id": model_id,
            "revision": revision,
            "source_total_size": source_index.get("metadata", {}).get("total_size"),
            "shards": shard_metadata,
        },
        "weight_map": output_map,
    }
    index_path = output_dir / OUTPUT_INDEX
    temporary_index = index_path.with_suffix(index_path.suffix + ".partial")
    with temporary_index.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_index, index_path)
    LOGGER.info("Converted %d tensors into %d shards at %s", len(output_map), len(source_shards), output_dir)
    return index_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=None, help="Optional Hugging Face commit/tag")
    parser.add_argument("--source-dir", type=Path, default=Path("checkpoints/GeoAgent-hf"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/GeoAgent-jittor"))
    parser.add_argument("--local-only", action="store_true", help="Do not contact Hugging Face")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    revision = args.revision
    if not args.local_only:
        revision = _download(args.model_id, args.source_dir, args.revision) or revision
    convert_checkpoint(args.source_dir, args.output_dir, model_id=args.model_id, revision=revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
