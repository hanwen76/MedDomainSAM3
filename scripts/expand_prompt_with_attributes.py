#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prompt_system import PromptSystem


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Expand prompts with anchored attributes. "
            "Supports input JSON as either case->prompt mapping or dataset list with prompt field."
        )
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--attributes-json", type=Path, required=True)
    parser.add_argument("--aliases-json", type=Path, default=None)
    parser.add_argument("--topk-attrs", type=int, default=3)
    parser.add_argument(
        "--unknown-policy",
        type=str,
        default="keep_raw",
        choices=["keep_raw", "error", "canonical_only"],
        help="How to handle prompts that cannot be matched to attributes.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="attrs_then_class",
        choices=["attrs_then_class", "class_then_attrs"],
    )
    parser.add_argument("--separator", type=str, default=", ")
    parser.add_argument("--save-report", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    payload_obj = json.loads(args.input_json.read_text())
    prompt_system = PromptSystem.from_paths(
        attributes_json=args.attributes_json,
        aliases_json=args.aliases_json,
    )
    if not prompt_system.is_enabled():
        raise ValueError("attributes-json/aliases-json produced an empty prompt system")

    output, details, known, unknown = prompt_system.expand_json_payload(
        payload_obj,
        topk_attrs=args.topk_attrs,
        unknown_policy=args.unknown_policy,
        fmt=args.format,
        separator=args.separator,
    )
    if isinstance(payload_obj, dict):
        input_type = "mapping"
    else:
        input_type = "list"

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2))

    report = {
        "input_json": str(args.input_json),
        "output_json": str(args.output_json),
        "input_type": input_type,
        "num_entries": len(details),
        "matched_entries": known,
        "unknown_entries": unknown,
        "topk_attrs": args.topk_attrs,
        "format": args.format,
        "unknown_policy": args.unknown_policy,
        "prompt_system": prompt_system.manifest(),
        "details": details,
    }
    if args.save_report is not None:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(json.dumps(report, indent=2))

    print(
        f"Expanded prompts saved to {args.output_json} | "
        f"matched={known}, unknown={unknown}, total={len(details)}"
    )


if __name__ == "__main__":
    main()
