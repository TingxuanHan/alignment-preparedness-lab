"""Validate the planning schema; missing revisions remain explicitly unready."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate(plan):
    if plan.get("schema_version") != 1:
        raise ValueError("Unsupported schema_version")
    arms = {a["id"]: a for a in plan["arms"]}
    if set(arms) != {"8b_base", "8b_sft", "8b_dpo"} or len(plan["arms"]) != 3:
        raise ValueError("Require one base, SFT and DPO arm")
    if arms["8b_sft"]["init"] != arms["8b_dpo"]["init"] or arms["8b_sft"]["init"] != "primary_original":
        raise ValueError("SFT/DPO must start at the same original checkpoint")
    if arms["8b_dpo"]["reference"] != "primary_original_frozen":
        raise ValueError("Reference must be fixed")
    if plan["common"]["enable_thinking"] is not False:
        raise ValueError("This protocol fixes non-thinking mode")
    for key in ("max_length", "per_device_train_batch_size", "gradient_accumulation_steps"):
        if plan["common"][key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not plan["data"]["drop_unsafe_chosen"] or plan["data"]["preference_axis"] != "safety":
        raise ValueError("Safety label/filter protocol mismatch")
    if plan["evaluation"]["report_errors_separately"] is not True:
        raise ValueError("Evaluation errors must remain separate")
    pending = []
    for label, value in (
        ("primary model revision", plan["models"]["primary"]["revision"]),
        ("replication model revision", plan["models"]["replication"]["revision"]),
        ("data revision", plan["data"]["revision"]),
        ("split manifest hash", plan["data"]["split_manifest_sha256"]),
        ("evaluation subset hash", plan["evaluation"]["subset_manifest_sha256"]),
        ("judge revision", plan["evaluation"]["judge_revision"]),
    ):
        if not value:
            pending.append(label)
    return pending


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.json")
    parser.add_argument("--ready", action="store_true", help="Fail if version fields are incomplete")
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.config.read_text(encoding="utf-8"))
        pending = validate(plan)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"INVALID: {error}")
        return 1
    print("PASS: planning schema. This does not validate an executable trainer.")
    if pending:
        print("NOT READY for frozen experiments: " + ", ".join(pending))
    else:
        print("Version fields filled; still validate data, trainer and evaluation implementation.")
    return int(args.ready and bool(pending))


if __name__ == "__main__":
    raise SystemExit(main())
