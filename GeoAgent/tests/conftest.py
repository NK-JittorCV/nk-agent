from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JITTOR_HOME", str((Path(__file__).parent / ".jittor-cache").resolve()))
os.environ.setdefault("use_cuda", "0")
os.environ.setdefault("nvcc_path", "")
os.environ.setdefault(
    "python_config_path",
    str(Path(sys.base_prefix) / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}-config"),
)
