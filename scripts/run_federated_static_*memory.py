#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_static_memory_bank import (
    aggregate_to_prototypes,
    build_image_transform,
    collect_pairs,
    encode_pair,
    load_image_tensor,
    load_mask_tensor,
    load_metadata,
    load_tracker,
)
from eval_medical_static_memory import (
    dice_score,
    iou_score,
    load_image_model,
    load_image_model_with_memory,
    predict_mask_with_text,
    resolve_text_prompt,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Federated leave-one-site-out static-memory evaluation for SAM3. "
            "Build a shared static memory bank from n-1 client sites and evaluate on the held-out site."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-site", type=str, default=None)
    parser.add_argument("--run-all-holdouts", action="store_true")
    parser.add_argument("--site-names", nargs="+", default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--text-prompt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--limit-per-site", type=int, default=None)
    parser.add_argument("--prototype-count", type=int, default=8)
    parser.add_argument(
        "--prototype-grouping",
        type=str,
        default="global",
        choices=["global", "per_text"],
    )
    parser.add_argument("--prototype-iters", type=int, default=15)
    parser.add_argument(
        "--local-prototype-count",
        type=int,
        default=0,
        help="If > 0, compress each client into local prototypes before global aggregation.",
    )
    parser.add_argument("--static-memory-topk", type=int, default=1)
    parser.add_argument("--static-memory-text-weight", type=float, default=1.0)
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


def summarize_metrics(results, key):
    if not results:
        return {"dice": 0.0, "iou": 0.0}
    return {
        "dice": float(sum(item[key]["dice"] for item in results) / len(results)),
        "iou": float(sum(item[key]["iou"] for item in results) / len(results)),
    }


def save_bank(
    output_path: Path,
    memory_features: torch.Tensor,
    memory_pos_enc: torch.Tensor,
    memory_keys: torch.Tensor,
    memory_text_keys: torch.Tensor | None,
    metadata: list[dict],
    config: dict,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_features": memory_features.cpu(),
            "memory_pos_enc": memory_pos_enc.cpu(),
            "memory_keys": memory_keys.cpu(),
            "memory_text_keys": None if memory_text_keys is None else memory_text_keys.cpu(),
            "metadata": metadata,
            "config": config,
        },
        output_path,
    )


def maybe_prototype_compress(
    memory_features: torch.Tensor,
    memory_pos_enc: torch.Tensor,
    memory_keys: torch.Tensor,
    memory_text_keys: torch.Tensor | None,
    metadata: list[dict],
    prototype_count: int,
    prototype_grouping: str,
    prototype_iters: int,
):
    return aggregate_to_prototypes(
        memory_features=memory_features,
        memory_pos_enc=memory_pos_enc,
        memory_keys=memory_keys,
        memory_text_keys=memory_text_keys,
        metadata=metadata,
        prototype_count=prototype_count,
        prototype_grouping=prototype_grouping,
        prototype_iters=prototype_iters,
    )


def build_local_memory_from_site(
    site_dir: Path,
    tracker,
    image_transform,
    image_size: int,
    device: str,
    metadata_map: dict,
    text_prompt: str | None,
    extensions,
    limit_per_site: int | None,
    local_prototype_count: int,
    prototype_grouping: str,
    prototype_iters: int,
):
    image_dir = site_dir / "data_npy"
    mask_dir = site_dir / "label_npy"
    pairs, missing_masks = collect_pairs(image_dir, mask_dir, extensions)
    if limit_per_site is not None:
        pairs = pairs[:limit_per_site]
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found for site {site_dir.name}")

    memory_features = []
    memory_pos_enc = []
    memory_keys = []
    memory_text_keys = []
    metadata = []

    for image_path, mask_path in pairs:
        image_tensor = load_image_tensor(image_path, image_transform, device)
        mask_tensor = load_mask_tensor(mask_path, image_size, device)
        case_text_prompt = resolve_text_prompt(image_path.stem, metadata_map, text_prompt)
        feats, pos_enc, keys, text_keys = encode_pair(
            tracker,
            image_tensor,
            mask_tensor,
            text_prompt=case_text_prompt,
        )
        memory_features.append(feats)
        memory_pos_enc.append(pos_enc)
        memory_keys.append(keys)
        if text_keys is not None:
            memory_text_keys.append(text_keys)
        metadata.append(
            {
                "site_name": site_dir.name,
                "case_id": image_path.stem,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "text_prompt": case_text_prompt,
            }
        )

    memory_features = torch.cat(memory_features, dim=0)
    memory_pos_enc = torch.cat(memory_pos_enc, dim=0)
    memory_keys = torch.cat(memory_keys, dim=0)
    memory_text_keys = (
        torch.cat(memory_text_keys, dim=0) if memory_text_keys else None
    )

    if local_prototype_count > 0:
        (
            memory_features,
            memory_pos_enc,
            memory_keys,
            memory_text_keys,
            metadata,
        ) = maybe_prototype_compress(
            memory_features,
            memory_pos_enc,
            memory_keys,
            memory_text_keys,
            metadata,
            prototype_count=local_prototype_count,
            prototype_grouping=prototype_grouping,
            prototype_iters=prototype_iters,
        )
        for item in metadata:
            item["site_name"] = site_dir.name

    return {
        "memory_features": memory_features,
        "memory_pos_enc": memory_pos_enc,
        "memory_keys": memory_keys,
        "memory_text_keys": memory_text_keys,
        "metadata": metadata,
        "num_pairs": len(pairs),
        "missing_masks": missing_masks,
    }


def merge_client_memories(client_memories):
    all_features = torch.cat([item["memory_features"] for item in client_memories], dim=0)
    all_pos_enc = torch.cat([item["memory_pos_enc"] for item in client_memories], dim=0)
    all_keys = torch.cat([item["memory_keys"] for item in client_memories], dim=0)
    any_text = any(item["memory_text_keys"] is not None for item in client_memories)
    if any_text:
        text_dim = next(
            item["memory_text_keys"].shape[-1]
            for item in client_memories
            if item["memory_text_keys"] is not None
        )
        all_text_keys = torch.cat(
            [
                item["memory_text_keys"]
                if item["memory_text_keys"] is not None
                else torch.zeros(
                    item["memory_features"].shape[0],
                    text_dim,
                )
                for item in client_memories
            ],
            dim=0,
        )
    else:
        all_text_keys = None
    all_metadata = []
    for item in client_memories:
        all_metadata.extend(item["metadata"])
    return all_features, all_pos_enc, all_keys, all_text_keys, all_metadata


def evaluate_holdout_site(
    holdout_site: Path,
    checkpoint_path: Path | None,
    shared_bank_path: Path,
    output_dir: Path,
    metadata_map: dict,
    text_prompt: str | None,
    image_size: int,
    device: str,
    extensions,
    limit_per_site: int | None,
    static_memory_topk: int,
    static_memory_text_weight: float,
    no_object_threshold: float,
    memory_trigger_threshold: float,
    save_predictions: bool,
):
    image_dir = holdout_site / "val_data_npy"
    mask_dir = holdout_site / "val_label_npy"
    pairs, missing_masks = collect_pairs(image_dir, mask_dir, extensions)
    if limit_per_site is not None:
        pairs = pairs[:limit_per_site]
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found for holdout site {holdout_site.name}")

    baseline_image_model, baseline_processor = load_image_model(checkpoint_path, device)
    static_image_model, static_processor = load_image_model_with_memory(
        checkpoint_path,
        device,
        static_memory_bank_path=shared_bank_path,
        static_memory_topk=static_memory_topk,
        static_memory_text_weight=static_memory_text_weight,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_predictions:
        (output_dir / "baseline").mkdir(parents=True, exist_ok=True)
        (output_dir / "static_memory").mkdir(parents=True, exist_ok=True)

    case_results = []
    for idx, (image_path, mask_path) in enumerate(pairs, start=1):
        gt_mask = load_mask_tensor(mask_path, image_size, device)
        case_text_prompt = resolve_text_prompt(image_path.stem, metadata_map, text_prompt)
        if case_text_prompt is None:
            raise ValueError(
                "text mode requires --text-prompt or --metadata-json with per-case prompts"
            )

        baseline_pred, baseline_score = predict_mask_with_text(
            baseline_image_model,
            baseline_processor,
            image_path,
            case_text_prompt,
            image_size,
        )
        baseline_pred = baseline_pred.to(gt_mask.device)
        if baseline_score < no_object_threshold:
            baseline_pred = torch.zeros_like(baseline_pred)

        if baseline_score < no_object_threshold:
            static_pred = torch.zeros_like(baseline_pred)
            static_score = baseline_score
            memory_strategy = "empty"
        elif baseline_score >= memory_trigger_threshold:
            static_pred = baseline_pred.clone()
            static_score = baseline_score
            memory_strategy = "baseline_passthrough"
        else:
            static_pred, static_score = predict_mask_with_text(
                static_image_model,
                static_processor,
                image_path,
                case_text_prompt,
                image_size,
            )
            static_pred = static_pred.to(gt_mask.device)
            memory_strategy = "memory_refine"

        result = {
            "case_id": image_path.stem,
            "baseline": {
                "dice": dice_score(baseline_pred, gt_mask),
                "iou": iou_score(baseline_pred, gt_mask),
                "score": baseline_score,
            },
            "static_memory": {
                "dice": dice_score(static_pred, gt_mask),
                "iou": iou_score(static_pred, gt_mask),
                "score": static_score,
                "strategy": memory_strategy,
            },
        }
        case_results.append(result)

        if save_predictions:
            from eval_medical_static_memory import save_mask

            save_mask(baseline_pred, output_dir / "baseline" / f"{image_path.stem}.png")
            save_mask(static_pred, output_dir / "static_memory" / f"{image_path.stem}.png")

        if idx % 10 == 0 or idx == len(pairs):
            print(f"[{holdout_site.name}] Processed {idx}/{len(pairs)} cases")

    summary = {
        "holdout_site": holdout_site.name,
        "num_cases": len(case_results),
        "baseline_mean": summarize_metrics(case_results, "baseline"),
        "static_memory_mean": summarize_metrics(case_results, "static_memory"),
        "cases": case_results,
        "missing_masks": missing_masks,
    }
    summary_path = output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved holdout metrics to {summary_path}")
    return summary


def run_single_holdout(args, holdout_site: Path, site_dirs, metadata_map):
    train_sites = [site_dir for site_dir in site_dirs if site_dir.name != holdout_site.name]
    if not train_sites:
        raise RuntimeError("Need at least one training site besides the held-out site.")

    print(
        f"Running federated static-memory leave-one-site-out with holdout={holdout_site.name} "
        f"and train_sites={[site.name for site in train_sites]}"
    )
    enable_text_encoder = args.metadata_json is not None or args.text_prompt is not None
    tracker = load_tracker(args.checkpoint_path, args.device, enable_text_encoder=enable_text_encoder)
    image_transform = build_image_transform(args.image_size)

    client_memories = []
    for site_dir in train_sites:
        print(f"Encoding local client memory for site {site_dir.name}")
        local_memory = build_local_memory_from_site(
            site_dir=site_dir,
            tracker=tracker,
            image_transform=image_transform,
            image_size=args.image_size,
            device=args.device,
            metadata_map=metadata_map,
            text_prompt=args.text_prompt,
            extensions=args.extensions,
            limit_per_site=args.limit_per_site,
            local_prototype_count=args.local_prototype_count,
            prototype_grouping=args.prototype_grouping,
            prototype_iters=args.prototype_iters,
        )
        client_memories.append(local_memory)

    (
        memory_features,
        memory_pos_enc,
        memory_keys,
        memory_text_keys,
        metadata,
    ) = merge_client_memories(client_memories)

    (
        memory_features,
        memory_pos_enc,
        memory_keys,
        memory_text_keys,
        metadata,
    ) = maybe_prototype_compress(
        memory_features,
        memory_pos_enc,
        memory_keys,
        memory_text_keys,
        metadata,
        prototype_count=args.prototype_count,
        prototype_grouping=args.prototype_grouping,
        prototype_iters=args.prototype_iters,
    )

    holdout_output_dir = args.output_dir / holdout_site.name
    shared_bank_path = holdout_output_dir / "shared_static_memory.pt"
    save_bank(
        shared_bank_path,
        memory_features,
        memory_pos_enc,
        memory_keys,
        memory_text_keys,
        metadata,
        config={
            "federated": True,
            "train_sites": [site.name for site in train_sites],
            "holdout_site": holdout_site.name,
            "prototype_count": args.prototype_count,
            "prototype_grouping": args.prototype_grouping,
            "prototype_iters": args.prototype_iters,
            "local_prototype_count": args.local_prototype_count,
        },
    )
    print(f"Saved shared federated memory bank to {shared_bank_path}")

    return evaluate_holdout_site(
        holdout_site=holdout_site,
        checkpoint_path=args.checkpoint_path,
        shared_bank_path=shared_bank_path,
        output_dir=holdout_output_dir,
        metadata_map=metadata_map,
        text_prompt=args.text_prompt,
        image_size=args.image_size,
        device=args.device,
        extensions=args.extensions,
        limit_per_site=args.limit_per_site,
        static_memory_topk=args.static_memory_topk,
        static_memory_text_weight=args.static_memory_text_weight,
        no_object_threshold=args.no_object_threshold,
        memory_trigger_threshold=args.memory_trigger_threshold,
        save_predictions=args.save_predictions,
    )


def main():
    args = parse_args()
    site_dirs = discover_sites(args.dataset_root, args.site_names)
    if len(site_dirs) < 2:
        raise RuntimeError("Need at least two valid client/site directories under dataset-root.")

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
        summary = run_single_holdout(args, holdout_site, site_dirs, metadata_map)
        all_results.append(
            {
                "holdout_site": holdout_site.name,
                "baseline_mean": summary["baseline_mean"],
                "static_memory_mean": summary["static_memory_mean"],
                "num_cases": summary["num_cases"],
            }
        )

    if len(all_results) > 1:
        overall = {
            "dataset_root": str(args.dataset_root),
            "num_holdouts": len(all_results),
            "holdouts": all_results,
            "mean_baseline_dice": float(
                sum(item["baseline_mean"]["dice"] for item in all_results) / len(all_results)
            ),
            "mean_static_memory_dice": float(
                sum(item["static_memory_mean"]["dice"] for item in all_results) / len(all_results)
            ),
            "mean_baseline_iou": float(
                sum(item["baseline_mean"]["iou"] for item in all_results) / len(all_results)
            ),
            "mean_static_memory_iou": float(
                sum(item["static_memory_mean"]["iou"] for item in all_results) / len(all_results)
            ),
        }
        overall_path = args.output_dir / "loso_summary.json"
        overall_path.write_text(json.dumps(overall, indent=2))
        print(f"Saved LOSO summary to {overall_path}")


if __name__ == "__main__":
    main()
