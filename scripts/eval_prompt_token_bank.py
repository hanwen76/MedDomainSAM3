#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_medical_static_memory import (
    collect_pairs,
    dice_score,
    iou_score,
    load_image_model,
    load_mask_tensor,
    predict_mask_with_text,
    resolve_text_prompt,
)


class PromptTokenBank(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_prompts: int,
        tokens_per_prompt: int = 1,
        prompt_scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_prompts = int(num_prompts)
        self.tokens_per_prompt = int(tokens_per_prompt)
        self.prompt_scale = float(prompt_scale)
        self.memory_tokens = nn.Parameter(
            torch.randn(self.num_prompts, self.tokens_per_prompt, self.hidden_dim) * 0.02
        )
        self.current_prompt_ids = None

    def set_prompt_ids(self, prompt_ids: torch.Tensor):
        self.current_prompt_ids = prompt_ids

    def build_prompt(self, img_feats, txt_feats, txt_masks=None):
        batch_size = txt_feats.shape[1]
        if self.current_prompt_ids is None:
            prompt_ids = torch.zeros(batch_size, dtype=torch.long, device=txt_feats.device)
        else:
            prompt_ids = self.current_prompt_ids.to(txt_feats.device).long()
            if prompt_ids.numel() != batch_size:
                raise ValueError(
                    f"prompt id count ({prompt_ids.numel()}) != batch size ({batch_size})"
                )
        tokens = self.memory_tokens[prompt_ids]  # [batch, T, D]
        prompt = tokens.permute(1, 0, 2).contiguous() * self.prompt_scale
        prompt_mask = torch.zeros(
            (batch_size, self.tokens_per_prompt),
            dtype=torch.bool,
            device=prompt.device,
        )
        return prompt, prompt_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate prompt-token-bank memory on top of SAM3 baseline."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--token-bank-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-prompt", type=str, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-object-threshold", type=float, default=0.2)
    parser.add_argument("--memory-trigger-threshold", type=float, default=0.8)
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"],
    )
    parser.add_argument(
        "--prompt-match-mode",
        type=str,
        choices=["exact", "normalized"],
        default="normalized",
        help="How to map user prompt to prompt_id in token-bank config.",
    )
    parser.add_argument(
        "--unknown-prompt-fallback",
        type=str,
        choices=["error", "baseline_only"],
        default="baseline_only",
    )
    return parser.parse_args()


def load_metadata(path: Path | None):
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("metadata-json must contain a JSON object mapping case stem to text")
    return payload


def norm_text(s: str):
    return re.sub(r"[\s_\-]+", " ", s.strip().lower())


def build_prompt_id_lookup(prompt_to_id: dict, mode: str):
    if mode == "exact":
        return {str(k): int(v) for k, v in prompt_to_id.items()}
    return {norm_text(str(k)): int(v) for k, v in prompt_to_id.items()}


def resolve_prompt_id(prompt: str, lookup: dict, mode: str):
    key = prompt if mode == "exact" else norm_text(prompt)
    return lookup.get(key)


def summarize_metrics(results, key):
    if not results:
        return {"dice": 0.0, "iou": 0.0}
    return {
        "dice": float(sum(item[key]["dice"] for item in results) / len(results)),
        "iou": float(sum(item[key]["iou"] for item in results) / len(results)),
    }


def main():
    args = parse_args()
    metadata = load_metadata(args.metadata_json)
    pairs, missing_masks = collect_pairs(args.image_dir, args.mask_dir, args.extensions)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError("No image/mask pairs found. Check filenames and directories.")

    payload = torch.load(args.token_bank_ckpt, map_location="cpu")
    cfg = payload.get("config", {})
    prompt_to_id = cfg.get("prompt_to_id", {})
    if not isinstance(prompt_to_id, dict) or len(prompt_to_id) == 0:
        raise RuntimeError("token-bank-ckpt missing config.prompt_to_id")
    num_prompts = int(cfg.get("num_prompts", len(prompt_to_id)))
    tokens_per_prompt = int(cfg.get("tokens_per_prompt", 1))
    lookup = build_prompt_id_lookup(prompt_to_id, mode=args.prompt_match_mode)

    baseline_model, baseline_processor = load_image_model(args.checkpoint_path, args.device)
    memory_model, memory_processor = load_image_model(args.checkpoint_path, args.device)
    token_bank = PromptTokenBank(
        hidden_dim=memory_model.hidden_dim,
        num_prompts=num_prompts,
        tokens_per_prompt=tokens_per_prompt,
    ).to(args.device)
    state = payload.get("memory_prompt_builder", payload)
    missing_keys, unexpected_keys = token_bank.load_state_dict(state, strict=False)
    if missing_keys:
        print(f"Missing keys while loading prompt token bank: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys while loading prompt token bank: {unexpected_keys}")
    memory_model.memory_prompt_builder = token_bank
    memory_model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    unknown_prompt_count = 0
    for idx, (image_path, mask_path) in enumerate(pairs, start=1):
        text_prompt = resolve_text_prompt(image_path.stem, metadata, args.text_prompt)
        if text_prompt is None:
            raise ValueError("text mode requires --text-prompt or --metadata-json")

        gt_mask = load_mask_tensor(mask_path, args.image_size, args.device)
        baseline_pred, baseline_score = predict_mask_with_text(
            baseline_model, baseline_processor, image_path, text_prompt, args.image_size
        )
        baseline_pred = baseline_pred.to(gt_mask.device)
        if baseline_score < args.no_object_threshold:
            baseline_pred = torch.zeros_like(baseline_pred)

        prompt_id = resolve_prompt_id(text_prompt, lookup, args.prompt_match_mode)
        unknown_prompt = prompt_id is None
        if unknown_prompt:
            unknown_prompt_count += 1
            if args.unknown_prompt_fallback == "error":
                raise KeyError(
                    f"Prompt {text_prompt!r} not found in token-bank prompt_to_id "
                    f"(match mode: {args.prompt_match_mode})"
                )

        if baseline_score < args.no_object_threshold:
            mem_pred = torch.zeros_like(baseline_pred)
            mem_score = baseline_score
            strategy = "empty"
        elif baseline_score >= args.memory_trigger_threshold:
            mem_pred = baseline_pred.clone()
            mem_score = baseline_score
            strategy = "baseline_passthrough"
        elif unknown_prompt:
            mem_pred = baseline_pred.clone()
            mem_score = baseline_score
            strategy = "unknown_prompt_baseline_fallback"
        else:
            memory_model.memory_prompt_builder.set_prompt_ids(
                torch.tensor([prompt_id], dtype=torch.long, device=args.device)
            )
            mem_pred, mem_score = predict_mask_with_text(
                memory_model, memory_processor, image_path, text_prompt, args.image_size
            )
            mem_pred = mem_pred.to(gt_mask.device)
            strategy = "memory_refine"

        results.append(
            {
                "case_id": image_path.stem,
                "prompt": text_prompt,
                "prompt_id": None if unknown_prompt else int(prompt_id),
                "baseline": {
                    "dice": dice_score(baseline_pred, gt_mask),
                    "iou": iou_score(baseline_pred, gt_mask),
                    "score": baseline_score,
                },
                "prompt_token_memory": {
                    "dice": dice_score(mem_pred, gt_mask),
                    "iou": iou_score(mem_pred, gt_mask),
                    "score": mem_score,
                    "strategy": strategy,
                },
            }
        )

        if idx % 10 == 0 or idx == len(pairs):
            print(f"Processed {idx}/{len(pairs)} cases")

    summary = {
        "num_cases": len(results),
        "prompt_match_mode": args.prompt_match_mode,
        "unknown_prompt_count": unknown_prompt_count,
        "baseline_mean": summarize_metrics(results, "baseline"),
        "prompt_token_memory_mean": summarize_metrics(results, "prompt_token_memory"),
        "cases": results,
        "missing_masks": missing_masks,
        "token_bank_prompts": sorted(prompt_to_id.keys()),
    }
    summary_path = args.output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved metrics to {summary_path}")


if __name__ == "__main__":
    main()
