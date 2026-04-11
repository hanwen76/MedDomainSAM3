#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_medical_static_memory import (
    collect_pairs,
    dice_score,
    iou_score,
    load_rgb_image_pil,
    load_image_model,
    load_image_model_with_free_memory,
    load_image_model_with_memory,
    load_mask_tensor,
    predict_mask_with_text,
    resolve_text_prompt,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ablation for baseline/static/free/hybrid memory on SAM3 image branch "
            "for single-task text-guided medical segmentation."
        )
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--static-memory-bank-path", type=Path, required=True)
    parser.add_argument("--free-memory-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-prompt", type=str, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--static-memory-topk", type=int, default=1)
    parser.add_argument("--static-memory-text-weight", type=float, default=1.0)
    parser.add_argument("--free-memory-num-tokens", type=int, default=4)
    parser.add_argument("--no-object-threshold", type=float, default=0.2)
    parser.add_argument("--memory-trigger-threshold", type=float, default=0.8)
    parser.set_defaults(use_gate=True)
    parser.add_argument("--use-gate", dest="use_gate", action="store_true")
    parser.add_argument("--disable-gate", dest="use_gate", action="store_false")
    parser.add_argument(
        "--hybrid-fusion",
        type=str,
        default="score_weighted",
        choices=["avg", "score_weighted"],
    )
    parser.add_argument("--hybrid-score-temperature", type=float, default=0.2)
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"],
    )
    return parser.parse_args()


def load_metadata(path: Path | None):
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("metadata-json must contain a JSON object mapping case stem to text")
    return payload


@torch.inference_mode()
def predict_prob_with_text(model, processor, image_path: Path, text_prompt: str, image_size: int):
    image = load_rgb_image_pil(image_path)
    state = processor.set_image(image)
    output = processor.set_text_prompt(text_prompt, state)
    masks_logits = output["masks_logits"]  # [N, 1, h, w]
    scores = output["scores"]  # [N]
    if masks_logits.numel() == 0:
        empty = torch.zeros(1, 1, image_size, image_size, device=processor.device)
        return empty, empty, 0.0
    best_idx = torch.argmax(scores)
    logits = masks_logits[best_idx : best_idx + 1]
    logits = F.interpolate(
        logits,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    probs = logits.sigmoid()
    return logits, probs, float(scores[best_idx].item())


def maybe_apply_gate(score, baseline_pred, baseline_logits, no_object_thr, trigger_thr):
    if score < no_object_thr:
        return "empty", torch.zeros_like(baseline_pred), torch.zeros_like(baseline_logits)
    if score >= trigger_thr:
        return "baseline_passthrough", baseline_pred.clone(), baseline_logits.clone()
    return "memory_refine", None, None


def summarize_metrics(results, key):
    if not results:
        return {"dice": 0.0, "iou": 0.0}
    return {
        "dice": float(sum(item[key]["dice"] for item in results) / len(results)),
        "iou": float(sum(item[key]["iou"] for item in results) / len(results)),
    }


def fuse_probs(static_probs, free_probs, static_score, free_score, mode: str, temp: float):
    if mode == "avg":
        return 0.5 * static_probs + 0.5 * free_probs
    s = torch.tensor([static_score, free_score], device=static_probs.device, dtype=static_probs.dtype)
    w = torch.softmax(s / max(temp, 1e-6), dim=0)
    return w[0] * static_probs + w[1] * free_probs


def pack_result(pred, gt, score, strategy):
    return {
        "dice": dice_score(pred, gt),
        "iou": iou_score(pred, gt),
        "score": float(score),
        "strategy": strategy,
    }


def main():
    args = parse_args()
    metadata = load_metadata(args.metadata_json)
    pairs, missing_masks = collect_pairs(args.image_dir, args.mask_dir, args.extensions)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError("No image/mask pairs found. Check filenames and directories.")

    baseline_model, baseline_processor = load_image_model(args.checkpoint_path, args.device)
    static_model, static_processor = load_image_model_with_memory(
        args.checkpoint_path,
        args.device,
        static_memory_bank_path=args.static_memory_bank_path,
        static_memory_topk=args.static_memory_topk,
        static_memory_text_weight=args.static_memory_text_weight,
    )
    free_model, free_processor = load_image_model_with_free_memory(
        args.checkpoint_path,
        args.device,
        free_memory_ckpt=args.free_memory_ckpt,
        free_memory_num_tokens=args.free_memory_num_tokens,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_results = []
    for idx, (image_path, mask_path) in enumerate(pairs, start=1):
        text_prompt = resolve_text_prompt(image_path.stem, metadata, args.text_prompt)
        if text_prompt is None:
            raise ValueError("Please provide --text-prompt or --metadata-json.")
        gt_mask = load_mask_tensor(mask_path, args.image_size, args.device)

        baseline_logits, baseline_probs, baseline_score = predict_prob_with_text(
            baseline_model, baseline_processor, image_path, text_prompt, args.image_size
        )
        # For strict parity with previous baseline numbers, use the original binary path.
        baseline_pred, baseline_score = predict_mask_with_text(
            baseline_model, baseline_processor, image_path, text_prompt, args.image_size
        )
        baseline_pred = baseline_pred.to(gt_mask.device)

        if baseline_score < args.no_object_threshold:
            baseline_pred = torch.zeros_like(baseline_pred)
            baseline_logits = torch.zeros_like(baseline_logits)

        # static memory
        if args.use_gate:
            static_strategy, static_pred, static_logits = maybe_apply_gate(
                baseline_score,
                baseline_pred,
                baseline_logits,
                args.no_object_threshold,
                args.memory_trigger_threshold,
            )
            if static_pred is None:
                static_pred, static_score = predict_mask_with_text(
                    static_model, static_processor, image_path, text_prompt, args.image_size
                )
                static_pred = static_pred.to(gt_mask.device)
                static_logits, static_probs, _ = predict_prob_with_text(
                    static_model, static_processor, image_path, text_prompt, args.image_size
                )
            else:
                static_score = baseline_score
                static_probs = static_logits.sigmoid()
        else:
            static_pred, static_score = predict_mask_with_text(
                static_model, static_processor, image_path, text_prompt, args.image_size
            )
            static_pred = static_pred.to(gt_mask.device)
            static_logits, static_probs, _ = predict_prob_with_text(
                static_model, static_processor, image_path, text_prompt, args.image_size
            )
            static_strategy = "always_memory"

        # free memory
        if args.use_gate:
            free_strategy, free_pred, free_logits = maybe_apply_gate(
                baseline_score,
                baseline_pred,
                baseline_logits,
                args.no_object_threshold,
                args.memory_trigger_threshold,
            )
            if free_pred is None:
                free_pred, free_score = predict_mask_with_text(
                    free_model, free_processor, image_path, text_prompt, args.image_size
                )
                free_pred = free_pred.to(gt_mask.device)
                free_logits, free_probs, _ = predict_prob_with_text(
                    free_model, free_processor, image_path, text_prompt, args.image_size
                )
            else:
                free_score = baseline_score
                free_probs = free_logits.sigmoid()
        else:
            free_pred, free_score = predict_mask_with_text(
                free_model, free_processor, image_path, text_prompt, args.image_size
            )
            free_pred = free_pred.to(gt_mask.device)
            free_logits, free_probs, _ = predict_prob_with_text(
                free_model, free_processor, image_path, text_prompt, args.image_size
            )
            free_strategy = "always_memory"

        # hybrid
        if args.use_gate:
            hybrid_strategy, hybrid_pred, _ = maybe_apply_gate(
                baseline_score,
                baseline_pred,
                baseline_logits,
                args.no_object_threshold,
                args.memory_trigger_threshold,
            )
            if hybrid_pred is None:
                hybrid_probs = fuse_probs(
                    static_probs=static_probs,
                    free_probs=free_probs,
                    static_score=static_score,
                    free_score=free_score,
                    mode=args.hybrid_fusion,
                    temp=args.hybrid_score_temperature,
                )
                hybrid_pred = (hybrid_probs > 0.5).float().to(gt_mask.device)
                hybrid_score = max(static_score, free_score)
            else:
                hybrid_score = baseline_score
        else:
            hybrid_probs = fuse_probs(
                static_probs=static_probs,
                free_probs=free_probs,
                static_score=static_score,
                free_score=free_score,
                mode=args.hybrid_fusion,
                temp=args.hybrid_score_temperature,
            )
            hybrid_pred = (hybrid_probs > 0.5).float().to(gt_mask.device)
            hybrid_score = max(static_score, free_score)
            hybrid_strategy = "always_memory_fusion"

        case_results.append(
            {
                "case_id": image_path.stem,
                "baseline": pack_result(
                    baseline_pred, gt_mask, baseline_score, "baseline"
                ),
                "static_memory": pack_result(
                    static_pred, gt_mask, static_score, static_strategy
                ),
                "free_memory": pack_result(
                    free_pred, gt_mask, free_score, free_strategy
                ),
                "hybrid_memory": pack_result(
                    hybrid_pred, gt_mask, hybrid_score, hybrid_strategy
                ),
            }
        )

        if idx % 10 == 0 or idx == len(pairs):
            print(f"Processed {idx}/{len(pairs)} cases")

    summary = {
        "num_cases": len(case_results),
        "use_gate": bool(args.use_gate),
        "hybrid_fusion": args.hybrid_fusion,
        "baseline_mean": summarize_metrics(case_results, "baseline"),
        "static_memory_mean": summarize_metrics(case_results, "static_memory"),
        "free_memory_mean": summarize_metrics(case_results, "free_memory"),
        "hybrid_memory_mean": summarize_metrics(case_results, "hybrid_memory"),
        "cases": case_results,
        "missing_masks": missing_masks,
    }
    out_path = args.output_dir / "metrics.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
