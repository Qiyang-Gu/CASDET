# -*- coding: utf-8 -*-
"""Check the server environment and the configured input paths.

Usage:
    python scripts/check_env.py
"""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _version(name: str) -> str:
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", "installed")
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({exc})"


def main() -> None:
    print("Paper environment: Windows 10 (build 19045), Python 3.12.0, "
          "PyTorch 2.3.1 (CUDA 12.1), transformers 4.57.6")
    print("Python:", sys.version.split()[0], "|", platform.platform())
    print("PyTorch:", _version("torch"))
    try:
        import torch
        print("CUDA available:", torch.cuda.is_available())
        count = torch.cuda.device_count()
        print("CUDA devices:", count)
        for i in range(count):
            print(f"  cuda:{i}: {torch.cuda.get_device_name(i)}")
    except Exception as exc:  # noqa: BLE001
        print("CUDA check failed:", exc)
    print("transformers:", _version("transformers"))
    print("accelerate:", _version("accelerate"))
    print("sentencepiece:", _version("sentencepiece"))
    print("PyMuPDF:", _version("pymupdf"))
    print("matplotlib:", _version("matplotlib"))

    try:
        from matplotlib import font_manager
        for family in ("Arial", "Helvetica"):
            try:
                path = font_manager.findfont(family, fallback_to_default=False)
                print(f"Figure font: {family} ({path})")
                break
            except Exception:  # noqa: BLE001
                continue
        else:
            print("Figure font: WARNING - Arial/Helvetica not found; "
                  "Matplotlib will use DejaVu Sans.")
    except Exception as exc:  # noqa: BLE001
        print("Font check failed:", exc)

    for name in ("main", "intervention", "ablation"):
        path = ROOT / "log" / "configs" / f"{name}.yaml"
        print(f"\n[{name}] {path}")
        if not path.exists():
            print("  missing configuration")
            continue
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for key, model_cfg in (cfg.get("models") or {}).items():
            target = ROOT / str(model_cfg.get("name", ""))
            state = "OK" if target.exists() else f"MISSING {target}"
            print(f"  model {key:<12} {state}")
        for dataset in cfg.get("datasets") or []:
            for key, rel in (dataset.get("files") or {}).items():
                target = ROOT / str(rel)
                state = "OK" if target.exists() else f"MISSING {target}"
                print(f"  data  {key:<12} {state}")

    print("\nEnvironment check complete.")


if __name__ == "__main__":
    main()
