"""Environment preparation that must happen before importing Jittor."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path

_OLD_ARCH_CAP = "    max_arch = 90\n    if nvcc_version < [11,]:"
_NATIVE_ARCH_CAP = '    max_arch = 120 if nvcc_version >= [12,8] else 90\n    if nvcc_version < [11,]:'


def _configure_native_cuda_arch(cache_home: Path) -> None:
    """Overlay Jittor with a narrowly patched compiler for CUDA 12.8+ GPUs.

    Jittor 1.3.11 caps generated kernels at ``sm_90`` even when its CUDA
    runtime detects a newer GPU.  On Blackwell this can silently produce
    incorrect tensor values.  Keep site-packages untouched: copy the small
    Python package into the requested cache, patch the single cap, and import
    that private copy for this process.
    """
    spec = find_spec("jittor")
    if spec is None or not spec.submodule_search_locations:
        return
    package_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
    compiler_path = package_dir / "compiler.py"
    source = compiler_path.read_text(encoding="utf-8")
    if _NATIVE_ARCH_CAP in source or _OLD_ARCH_CAP not in source:
        return

    source_digest = sha256(source.encode()).hexdigest()[:12]
    overlays_dir = cache_home / "compat"
    overlay_root = overlays_dir / f"jittor-native-arch-{source_digest}"
    marker = overlay_root / ".complete"
    if not marker.is_file():
        overlays_dir.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(tempfile.mkdtemp(prefix=".jittor-native-arch-", dir=overlays_dir))
        try:
            temporary_package = temporary_root / "jittor"
            shutil.copytree(
                package_dir,
                temporary_package,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            patched = source.replace(_OLD_ARCH_CAP, _NATIVE_ARCH_CAP, 1)
            (temporary_package / "compiler.py").write_text(patched, encoding="utf-8")
            (temporary_root / ".complete").write_text("Jittor native CUDA architecture patch\n", encoding="utf-8")
            try:
                temporary_root.rename(overlay_root)
            except FileExistsError:
                shutil.rmtree(temporary_root)
        except BaseException:
            shutil.rmtree(temporary_root, ignore_errors=True)
            raise
    sys.path.insert(0, str(overlay_root))


def _configure_conda_cuda_shim(cache_home: Path) -> None:
    """Work around Jittor assuming every nvcc installation uses ``lib64``."""
    if "nvcc_path" in os.environ:
        return
    conventional_nvcc = Path("/usr/local/cuda/bin/nvcc")
    nvcc = str(conventional_nvcc) if conventional_nvcc.is_file() else shutil.which("nvcc")
    if nvcc is None:
        return
    cuda_prefix = Path(nvcc).resolve().parent.parent
    if (cuda_prefix / "lib64" / "libcudart.so").is_file():
        os.environ["nvcc_path"] = nvcc
        return

    include_candidates = (
        cuda_prefix / "include",
        cuda_prefix / "targets" / "x86_64-linux" / "include",
    )
    library_candidates = (
        cuda_prefix / "lib",
        cuda_prefix / "targets" / "x86_64-linux" / "lib",
    )
    include_dir = next((path for path in include_candidates if (path / "cuda.h").is_file()), None)
    library_dir = next((path for path in library_candidates if (path / "libcudart.so").is_file()), None)
    if include_dir is None or library_dir is None:
        return

    shim = cache_home / "cuda-layout"
    bin_dir = shim / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    wrapper = bin_dir / "nvcc"
    wrapper.write_text(f'#!/bin/sh\nexec "{nvcc}" "$@"\n', encoding="utf-8")
    wrapper.chmod(0o755)
    for name, target in (("include", include_dir), ("lib64", library_dir)):
        link = shim / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    os.environ["nvcc_path"] = str(wrapper)


def configure_jittor_environment(cache_home: str | Path, *, enable_cuda: bool = True) -> None:
    cache_home = Path(cache_home).resolve()
    os.environ.setdefault("JITTOR_HOME", str(cache_home))
    if "python_config_path" not in os.environ:
        candidates = (
            Path(sys.base_prefix) / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}-config",
            Path(sys.base_prefix) / "bin" / "python3-config",
        )
        for candidate in candidates:
            if candidate.is_file():
                os.environ["python_config_path"] = str(candidate)
                break
    if enable_cuda:
        # GeoAgent implements its patch projection with matmul and does not use
        # convolution operators.  Tell Jittor to load only cuBLAS/curand so a
        # system cuDNN development-header layout cannot block CUDA startup.
        os.environ.setdefault("conv_opt", "1")
        # CUTT 1.2 does not ship kernels for Blackwell (sm_120).  Jittor's
        # built-in transpose is slower but portable and sufficient here.
        os.environ.setdefault("use_cutt", "0")
        _configure_native_cuda_arch(cache_home)
        _configure_conda_cuda_shim(cache_home)
    else:
        os.environ["nvcc_path"] = ""
