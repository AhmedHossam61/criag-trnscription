"""Environment checks: ffmpeg, CUDA, and each backend's imports."""

from __future__ import annotations

import shutil
import sys

import numpy as np


def check_env(backend_name: str) -> int:
    """Report on ffmpeg, CUDA, VRAM and the selected backend's imports."""
    problems = 0
    print("Core:")
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        print(f"  {tool:<10} {path or 'MISSING -- install ffmpeg'}")
        problems += path is None
    print(f"  numpy      {np.__version__}")
    print(f"  python     {sys.version.split()[0]}")

    print("\nGPU:")
    try:
        import torch
        print(f"  torch      {torch.__version__}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / (1024 ** 3)
            print(f"  device     {props.name} ({vram:.1f} GB, sm_{props.major}{props.minor})")
            if vram < 7.0:
                print("  ! under 7 GB free -- lower --batch-size to 1")
        else:
            print("  ! CUDA not available; this will run on CPU and be far too slow")
            problems += 1
    except ImportError:
        print("  torch      not installed")
        problems += 1

    print(f"\nBackend '{backend_name}':")
    needed = {
        "qwencleo": ["qwencleo_asr", "qwen_asr"],
        "qwen3asr": ["transformers"],
        "cohere": ["transformers"],
        "faster-whisper": ["faster_whisper"],
    }[backend_name]
    for module in needed:
        try:
            mod = __import__(module)
            print(f"  {module:<14} {getattr(mod, '__version__', 'ok')}")
        except ImportError as exc:
            print(f"  {module:<14} MISSING ({exc})")
            problems += 1

    if backend_name == "cohere":
        try:
            from huggingface_hub import whoami
            print(f"  hf auth        {whoami()['name']}")
        except Exception:  # noqa: BLE001
            print("  hf auth        not logged in -- weights are gated, run `hf auth login`")
            problems += 1

    print("\n" + ("All good." if not problems else f"{problems} problem(s) above."))
    return problems


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"error: {tool} not found on PATH. Install ffmpeg first.")
