"""One normal prompt on one GPU; explicitly downloads the requested model."""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", help="Model commit; when omitted resolve current main once and record it")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.max_new_tokens <= 512:
        parser.error("max-new-tokens must be between 1 and 512 for this smoke check")
    output = args.output or Path("runs") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-model-smoke.json")
    if output.exists():
        parser.error("Output already exists; choose a new filename")
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Expose exactly one allocated GPU for this check using CUDA_VISIBLE_DEVICES.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("Selected GPU does not support the BF16 configuration.")
    info = HfApi().model_info(args.model, revision=args.revision or "main")
    revision = info.sha
    print(f"Loading {args.model} at {revision}. First execution may download model weights.")
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, trust_remote_code=False, torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa",
    ).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    prompt = "用两句话解释监督微调与偏好优化有什么区别。"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt").to("cuda:0")
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    tokens = generated[0, inputs["input_ids"].shape[1]:]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    report = {
        "kind": "single_prompt_smoke_not_safety_result",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "model": args.model, "model_revision": revision,
        "enable_thinking": False, "dtype": "bfloat16", "do_sample": False,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "load_seconds": load_seconds, "generation_seconds": seconds,
        "generated_tokens": len(tokens),
        "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
        "prompt": prompt, "output": tokenizer.decode(tokens, skip_special_tokens=True),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(report["output"])
    print(f"Saved {output}. Loading/inference passed; no training or safety evaluation ran.")


if __name__ == "__main__":
    main()
