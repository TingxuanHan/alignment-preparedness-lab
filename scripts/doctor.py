"""Read only preflight; optional tiny CUDA check. Never enumerates credentials."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(args):
    if not shutil.which(args[0]):
        return {"ok": False, "message": f"{args[0]} not found"}
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"ok": result.returncode == 0,
                "output": (result.stdout if result.returncode == 0 else result.stderr).strip()[:4000]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "message": str(error)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    packages = {}
    for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "kind": "environment_preflight_not_benchmark",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "recommended_python": (3, 10) <= sys.version_info[:2] <= (3, 12),
        "git": command(["git", "--version"]),
        "gh": command(["gh", "--version"]),
        "gpu_driver": command(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                               "--format=csv,noheader"]),
        "disk_free_gib": round(shutil.disk_usage(Path.cwd()).free / 2**30, 1),
        "packages": packages,
        "cuda_check": "not_requested",
    }
    ok = True
    if args.require_gpu:
        try:
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
                raise RuntimeError("CUDA unavailable; inspect driver, GPU allocation and CUDA wheel.")
            device = torch.cuda.current_device()
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 unsupported on the selected GPU.")
            with torch.inference_mode():
                a = torch.ones((256, 256), device=f"cuda:{device}", dtype=torch.bfloat16)
                out = a @ a
                torch.cuda.synchronize()
                if not bool(torch.isfinite(out).all()) or out[0, 0].item() != 256.0:
                    raise RuntimeError("CUDA matrix check failed.")
            report["cuda_check"] = {"ok": True, "runtime": torch.version.cuda,
                                    "visible_count": torch.cuda.device_count(),
                                    "selected_name": torch.cuda.get_device_name(device),
                                    "bf16_matrix_check": "passed"}
        except Exception as error:
            report["cuda_check"] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            ok = False
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
