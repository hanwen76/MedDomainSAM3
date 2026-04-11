#!/usr/bin/env python3

"""
Evaluate a federated free memory tokens model (FedAvg) on held-out sites.
Uses the same evaluation logic as eval_medical_static_memory.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import torch
from pathlib import Path

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
    load_image_model_with_free_memory,
    load_rgb_image_pil,
    load_metadata,
    resolve_text_prompt,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a federated free memory tokens model on held-out sites."
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--fedavg-model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-site", type=str, default=None)
    parser.add_argument("--run-all-holdouts", action="store_true")
    parser.add_argument("--site-names", nargs="+", default=None)
    parser.add_argument("--text-prompt", type=str, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--limit-per-site", type=int, default=None)
    parser.add_argument("--no-object-threshold", type=float, default=0.2)
    parser.add_argument("--memory-trigger-threshold", type=float, default=0.8)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"],
    )
    return parser.parse_args()


def discover_sites(dataset_root: Path, site_names: list[str] | None):
    if site_names is not None:
        sites = [dataset_root / name for name in site_names]
    else:
        sites = [path for path in sorted(dataset_root.iterdir()) if path.is_dir()]

    valid_sites = []
    for site_dir in sites:
        image_dir = site_dir / "data_npy"
        mask_dir = site_dir / "label_npy"
        if image_dir.is_dir() and mask_dir.is_dir():
            valid_sites.append(site_dir)
    return valid_sites


def evaluate_holdout_site(
    holdout_site: Path,
    checkpoint_path: Path,
    fedavg_model_path: Path,
    output_dir: Path,
    metadata_map: dict,
    text_prompt: str | None,
    image_size: int,
    device: str,
    extensions,
    limit_per_site: int | None,
    no_object_threshold: float,
    memory_trigger_threshold: float,
    save_predictions: bool,
):
    """Evaluate baseline and fedavg model on a holdout site."""
    image_dir = holdout_site / "val_data_npy"
    mask_dir = holdout_site / "val_label_npy"
    pairs, missing_masks = collect_pairs(image_dir, mask_dir, extensions)
    if limit_per_site is not None:
        pairs = pairs[:limit_per_site]
    if not pairs:
        raise RuntimeError(f"No pairs found for holdout site {holdout_site.name}")

    # Load models
    baseline_model, baseline_processor = load_image_model(checkpoint_path, device)

    # Load fedavg model
    fedavg_model, fedavg_processor = load_image_model_with_free_memory(
        checkpoint_path,
        device,
        free_memory_ckpt=fedavg_model_path,
        free_memory_num_tokens=4,  # will be overridden by checkpoint config if needed
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_predictions:
        (output_dir / "baseline").mkdir(parents=True, exist_ok=True)
        (output_dir / "fedavg").mkdir(parents=True, exist_ok=True)

    case_results = []
    for idx, (image_path, mask_path) in enumerate(pairs, start=1):
        gt_mask = load_mask_tensor(mask_path, image_size, device)
        case_text_prompt = resolve_text_prompt(image_path.stem, metadata_map, text_prompt)
        if case_text_prompt is None:
            raise ValueError("text mode requires --text-prompt or --metadata-json with per-case prompts")

        # Baseline prediction
        baseline_pred, baseline_score = predict_mask_with_text(
            baseline_model,
            baseline_processor,
            image_path,
            case_text_prompt,
            image_size,
        )
        baseline_pred = baseline_pred.to(gt_mask.device)
        if baseline_score < no_object_threshold:
            baseline_pred = torch.zeros_like(baseline_pred)

        # Fedavg prediction
        if baseline_score < no_object_threshold:
            fedavg_pred = torch.zeros_like(baseline_pred)
            fedavg_score = baseline_score
            strategy = "empty"
        elif baseline_score >= memory_trigger_threshold:
            fedavg_pred = baseline_pred.clone()
            fedavg_score = baseline_score
            strategy = "baseline_passthrough"
        else:
            fedavg_pred, fedavg_score = predict_mask_with_text(
                fedavg_model,
                fedavg_processor,
                image_path,
                case_text_prompt,
                image_size,
            )
            fedavg_pred = fedavg_pred.to(gt_mask.device)
            strategy = "fedavg_memory_refine"

        result = {
            "case_id": image_path.stem,
            "baseline": {
                "dice": dice_score(baseline_pred, gt_mask),
                "iou": iou_score(baseline_pred, gt_mask),
                "score": baseline_score,
            },
            "fedavg": {
                "dice": dice_score(fedavg_pred, gt_mask),
                "iou": iou_score(fedavg_pred, gt_mask),
                "score": fedavg_score,
                "strategy": strategy,
            },
        }
        case_results.append(result)

        if save_predictions:
            save_mask_tensor(baseline_pred, output_dir / "baseline" / f"{image_path.stem}.png")
            save_mask_tensor(fedavg_pred, output_dir / "fedavg" / f"{image_path.stem}.png")

        if idx % 10 == 0 or idx == len(pairs):
            print(f"[{holdout_site.name}] Processed {idx}/{len(pairs)} cases")

    def summarize_metrics(results, key):
        if not results:
            return {"dice": 0.0, "iou": 0.0}
        return {
            "dice": float(sum(item[key]["dice"] for item in results) / len(results)),
            "iou": float(sum(item[key]["iou"] for item in results) / len(results)),
        }

    summary = {
        "holdout_site": holdout_site.name,
        "num_cases": len(case_results),
        "baseline_mean": summarize_metrics(case_results, "baseline"),
        "fedavg_mean": summarize_metrics(case_results, "fedavg"),
        "cases": case_results,
        "missing_masks": missing_masks,
    }
    summary_path = output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved holdout metrics to {summary_path}")
    return summary


def predict_mask_with_text(model, processor, image_path, text_prompt, image_size):
    """Predict mask using text prompt. Imported from eval_medical_static_memory."""
    import torch.nn.functional as F
    image = load_rgb_image_pil(image_path)
    state = processor.set_image(image)
    output = processor.set_text_prompt(text_prompt, state)
    masks = output["masks_logits"]
    scores = output["scores"]
    if masks.numel() == 0:
        return (
            torch.zeros(1, 1, image_size, image_size, device=processor.device),
            0.0,
        )
    best_idx = torch.argmax(scores)
    mask = masks[best_idx : best_idx + 1]
    mask = F.interpolate(
        mask,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return (mask > 0.5).float(), float(scores[best_idx].item())


def load_mask_tensor(path, image_size, device):
    """Load and preprocess mask. Imported from eval_medical_static_memory."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    if str(path).lower().endswith(".npy"):
        mask_np = np.load(path)
        mask_np = np.asarray(mask_np)
        if mask_np.ndim == 3:
            if mask_np.shape[0] == 1:
                mask_np = mask_np[0]
            elif mask_np.shape[-1] == 1:
                mask_np = mask_np[..., 0]
            else:
                mask_np = mask_np[..., 0]
        mask_np = np.nan_to_num(mask_np).astype(np.float32)
        mask_np = (mask_np > 0).astype(np.float32)
    else:
        mask = Image.open(path).convert("L")
        mask_np = np.array(mask, dtype=np.float32)
        if mask_np.max() > 1:
            mask_np = mask_np / 255.0
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    mask_tensor = F.interpolate(mask_tensor, size=(image_size, image_size), mode="nearest")
    mask_tensor = (mask_tensor > 0.5).float().to(device)
    return mask_tensor


def save_mask_tensor(mask_tensor, path):
    """Save mask tensor as image. Imported from eval_medical_static_memory."""
    import numpy as np
    from PIL import Image
    mask = (mask_tensor[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask).save(path)


def main():
    args = parse_args()
    site_dirs = discover_sites(args.dataset_root, args.site_names)
    if len(site_dirs) < 1:
        raise RuntimeError("Need at least one valid site directory under dataset-root.")

    metadata_map = load_metadata(args.metadata_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_all_holdouts:
        holdout_sites = site_dirs
    else:
        if args.holdout_site is None:
            raise ValueError("Please provide --holdout-site, or use --run-all-holdouts.")
        holdout_sites = [site_dir for site_dir in site_dirs if site_dir.name == args.holdout_site]
        if not holdout_sites:
            raise ValueError(f"Holdout site {args.holdout_site!r} not found under {args.dataset_root}")

    all_results = []
    for holdout_site in holdout_sites:
        print(f"\n=== Evaluating on holdout site: {holdout_site.name} ===")
        holdout_output_dir = args.output_dir / holdout_site.name
        summary = evaluate_holdout_site(
            holdout_site=holdout_site,
            checkpoint_path=args.checkpoint_path,
            fedavg_model_path=args.fedavg_model,
            output_dir=holdout_output_dir,
            metadata_map=metadata_map,
            text_prompt=args.text_prompt,
            image_size=args.image_size,
            device=args.device,
            extensions=args.extensions,
            limit_per_site=args.limit_per_site,
            no_object_threshold=args.no_object_threshold,
            memory_trigger_threshold=args.memory_trigger_threshold,
            save_predictions=args.save_predictions,
        )
        all_results.append({
            "holdout_site": holdout_site.name,
            "baseline_mean": summary["baseline_mean"],
            "fedavg_mean": summary["fedavg_mean"],
            "num_cases": summary["num_cases"],
        })

    if len(all_results) > 1:
        overall = {
            "dataset_root": str(args.dataset_root),
            "fedavg_model": str(args.fedavg_model),
            "num_holdouts": len(all_results),
            "holdouts": all_results,
            "mean_baseline_dice": float(
                sum(item["baseline_mean"]["dice"] for item in all_results) / len(all_results)
            ),
            "mean_fedavg_dice": float(
                sum(item["fedavg_mean"]["dice"] for item in all_results) / len(all_results)
            ),
            "mean_baseline_iou": float(
                sum(item["baseline_mean"]["iou"] for item in all_results) / len(all_results)
            ),
            "mean_fedavg_iou": float(
                sum(item["fedavg_mean"]["iou"] for item in all_results) / len(all_results)
            ),
        }
        overall_path = args.output_dir / "eval_summary.json"
        overall_path.write_text(json.dumps(overall, indent=2))
        print(f"Saved eval summary to {overall_path}")


if __name__ == "__main__":
    main()
