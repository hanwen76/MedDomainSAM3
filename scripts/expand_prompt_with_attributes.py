#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


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


def norm_text(s: str):
    s = s.strip().lower()
    s = re.sub(r"[\s_\-]+", " ", s)
    return s


def load_json(path: Path):
    return json.loads(path.read_text())


def build_lookup(attributes: dict, aliases: dict | None):
    canonical_norm_to_canonical = {norm_text(k): k for k in attributes.keys()}
    alias_norm_to_canonical = {}
    if aliases:
        for alias, canonical in aliases.items():
            alias_norm_to_canonical[norm_text(str(alias))] = str(canonical)
    return canonical_norm_to_canonical, alias_norm_to_canonical


def resolve_canonical(raw_prompt, canonical_norm_to_canonical, alias_norm_to_canonical):
    key = norm_text(raw_prompt)
    if key in alias_norm_to_canonical:
        return alias_norm_to_canonical[key]
    if key in canonical_norm_to_canonical:
        return canonical_norm_to_canonical[key]
    return None


def compose_prompt(canonical: str, attrs: list[str], fmt: str, sep: str):
    attrs = [a.strip() for a in attrs if str(a).strip()]
    if len(attrs) == 0:
        return canonical
    if fmt == "class_then_attrs":
        return sep.join([canonical] + attrs)
    return sep.join(attrs + [canonical])


def expand_one_prompt(
    raw_prompt: str,
    attributes: dict,
    canonical_norm_to_canonical: dict,
    alias_norm_to_canonical: dict,
    topk_attrs: int,
    unknown_policy: str,
    fmt: str,
    sep: str,
):
    canonical = resolve_canonical(raw_prompt, canonical_norm_to_canonical, alias_norm_to_canonical)
    if canonical is None:
        if unknown_policy == "error":
            raise KeyError(f"Unknown prompt: {raw_prompt!r}")
        if unknown_policy == "canonical_only":
            return raw_prompt, raw_prompt, [], False
        return raw_prompt, raw_prompt, [], False

    attrs = attributes.get(canonical, [])
    if topk_attrs > 0:
        attrs = attrs[:topk_attrs]
    expanded = compose_prompt(canonical, attrs, fmt, sep)
    return raw_prompt, expanded, attrs, True


def expand_mapping_payload(
    payload: dict,
    attributes: dict,
    canonical_norm_to_canonical: dict,
    alias_norm_to_canonical: dict,
    topk_attrs: int,
    unknown_policy: str,
    fmt: str,
    sep: str,
):
    output = {}
    details = {}
    known = 0
    unknown = 0
    for key, raw_prompt in payload.items():
        if raw_prompt is None:
            output[key] = None
            details[key] = {"raw_prompt": None, "expanded_prompt": None, "matched": False}
            unknown += 1
            continue
        raw_prompt = str(raw_prompt)
        raw, expanded, attrs, matched = expand_one_prompt(
            raw_prompt=raw_prompt,
            attributes=attributes,
            canonical_norm_to_canonical=canonical_norm_to_canonical,
            alias_norm_to_canonical=alias_norm_to_canonical,
            topk_attrs=topk_attrs,
            unknown_policy=unknown_policy,
            fmt=fmt,
            sep=sep,
        )
        output[key] = expanded
        details[key] = {
            "raw_prompt": raw,
            "expanded_prompt": expanded,
            "attrs_used": attrs,
            "matched": matched,
        }
        if matched:
            known += 1
        else:
            unknown += 1
    return output, details, known, unknown


def expand_dataset_list_payload(
    payload: list,
    attributes: dict,
    canonical_norm_to_canonical: dict,
    alias_norm_to_canonical: dict,
    topk_attrs: int,
    unknown_policy: str,
    fmt: str,
    sep: str,
):
    output = []
    details = []
    known = 0
    unknown = 0
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"datasets-json[{idx}] must be an object")
        if "prompt" not in item:
            raise ValueError(f"datasets-json[{idx}] missing prompt field")
        raw_prompt = str(item["prompt"])
        raw, expanded, attrs, matched = expand_one_prompt(
            raw_prompt=raw_prompt,
            attributes=attributes,
            canonical_norm_to_canonical=canonical_norm_to_canonical,
            alias_norm_to_canonical=alias_norm_to_canonical,
            topk_attrs=topk_attrs,
            unknown_policy=unknown_policy,
            fmt=fmt,
            sep=sep,
        )
        new_item = dict(item)
        new_item["prompt_raw"] = raw
        new_item["prompt"] = expanded
        output.append(new_item)
        details.append(
            {
                "index": idx,
                "raw_prompt": raw,
                "expanded_prompt": expanded,
                "attrs_used": attrs,
                "matched": matched,
            }
        )
        if matched:
            known += 1
        else:
            unknown += 1
    return output, details, known, unknown


def main():
    args = parse_args()
    payload = load_json(args.input_json)
    attributes = load_json(args.attributes_json)
    aliases = load_json(args.aliases_json) if args.aliases_json is not None else {}

    if not isinstance(attributes, dict) or len(attributes) == 0:
        raise ValueError("attributes-json must be a non-empty object: canonical_prompt -> [attrs]")
    for key, value in attributes.items():
        if not isinstance(value, list):
            raise ValueError(f"attributes-json[{key!r}] must be a list of attributes")

    canonical_norm_to_canonical, alias_norm_to_canonical = build_lookup(attributes, aliases)

    if isinstance(payload, dict):
        output, details, known, unknown = expand_mapping_payload(
            payload=payload,
            attributes=attributes,
            canonical_norm_to_canonical=canonical_norm_to_canonical,
            alias_norm_to_canonical=alias_norm_to_canonical,
            topk_attrs=args.topk_attrs,
            unknown_policy=args.unknown_policy,
            fmt=args.format,
            sep=args.separator,
        )
        input_type = "mapping"
    elif isinstance(payload, list):
        output, details, known, unknown = expand_dataset_list_payload(
            payload=payload,
            attributes=attributes,
            canonical_norm_to_canonical=canonical_norm_to_canonical,
            alias_norm_to_canonical=alias_norm_to_canonical,
            topk_attrs=args.topk_attrs,
            unknown_policy=args.unknown_policy,
            fmt=args.format,
            sep=args.separator,
        )
        input_type = "list"
    else:
        raise ValueError("input-json must be either a JSON object or a JSON list")

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
