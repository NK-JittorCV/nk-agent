"""Command-line interface for pure-Jittor GeoAgent single-image inference."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .bootstrap import configure_jittor_environment
from .config import GeoAgentConfig
from .processing import GeoAgentProcessor

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="Converted GeoAgent directory")
    parser.add_argument("--image", type=Path, required=True, help="Local input image")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--attention-chunk-size", type=int, default=256)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--jittor-home",
        type=Path,
        default=Path(".jittor-cache"),
        help="Writable Jittor compilation cache directory",
    )
    return parser


def _unwrap_json_fence(response: str) -> str:
    """Remove the Markdown fence that this checkpoint commonly emits."""
    stripped = response.strip()
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].strip().lower() in {"```", "```json"}:
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        return "\n".join(lines).strip()
    return stripped


def _render_output(response: str) -> None:
    json_payload = _unwrap_json_fence(response)
    try:
        parsed = json.loads(json_payload)
    except json.JSONDecodeError as error:
        LOGGER.warning("Model output is not valid JSON (%s); preserving raw response", error)
        print(response)
        return
    print(json.dumps(parsed, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.attention_chunk_size <= 0:
        raise ValueError("--attention-chunk-size must be positive")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    configure_jittor_environment(args.jittor_home)
    # Imports are intentionally delayed until JITTOR_HOME is configured.  The
    # runtime environment does not import torch or transformers.
    import jittor as jt

    from .modeling import GeoAgentForConditionalGeneration
    from .weights import load_converted_weights

    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("pure Jittor runtime unexpectedly imported torch or transformers")
    jt.flags.use_cuda = 1

    config = GeoAgentConfig.from_directory(args.model_path)
    processor = GeoAgentProcessor(args.model_path, config)
    LOGGER.info("Preprocessing %s", args.image)
    prepared = processor.prepare(args.image)
    LOGGER.info(
        "Image grid is %s (%d language-model image tokens)",
        prepared.image_grid_thw.tolist(),
        int((prepared.input_ids == config.image_token_id).sum()),
    )

    LOGGER.info("Constructing Qwen2.5-VL in Jittor (%s)", args.dtype)
    model = GeoAgentForConditionalGeneration(
        config,
        dtype=args.dtype,
        attention_chunk_size=args.attention_chunk_size,
    )
    model.eval()
    load_converted_weights(model, args.model_path)
    LOGGER.info("Generating up to %d tokens", args.max_new_tokens)
    generated = model.generate(
        prepared.input_ids,
        prepared.pixel_values,
        prepared.image_grid_thw,
        max_new_tokens=args.max_new_tokens,
    )
    stop_reason = "EOS" if len(generated) < args.max_new_tokens else "token limit"
    LOGGER.info("Generated %d tokens (stop: %s)", len(generated), stop_reason)
    response = processor.decode(generated).strip()
    if not response:
        raise RuntimeError("model generated no text")
    _render_output(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
