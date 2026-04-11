#!/usr/bin/env python3

"""
Federated learning with free memory tokens using FedAvg aggregation.
Each client trains its own free memory tokens locally, then the weights are aggregated via FedAvg.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model_builder import build_sam3_image_model


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Federated learning with free memory tokens using FedAvg aggregation. "
            "Each client trains its own free memory tokens locally, then weights are aggregated."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-site", type=str, default=None)
    parser.add_argument("--run-all-holdouts", action="store_true")
    parser.add_argument("--site-names", nargs="+", default=None)
    parser.add_argument("--text-prompt", type=str, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=1008)
    # Free memory token args
    parser.add_argument("--num-tokens", type=int, default=4)
    # Training args
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of federated rounds (each round: local train 1 epoch + FedAvg aggregation).",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--limit-per-site", type=int, default=None)
    parser.add_argument(
        "--token-l2-weight",
        type=float,
        default=1e-4,
        help="L2 regularization weight for free memory tokens.",
    )
    parser.add_argument(
        "--token-diversity-weight",
        type=float,
        default=1e-3,
        help="Diversity regularization weight to discourage token collapse.",
    )
    # Evaluation args
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


def collect_pairs(image_dir: Path, mask_dir: Path, extensions):
    ext_set = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    image_paths = {}
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in ext_set:
            image_paths[path.stem] = path
    pairs = []
    missing_masks = []
    for stem, image_path in sorted(image_paths.items()):
        mask_path = None
        for ext in ext_set:
            candidate = mask_dir / f"{stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
        if mask_path is None:
            alt = list(mask_dir.rglob(f"{stem}.*"))
            for candidate in alt:
                if candidate.suffix.lower() in ext_set:
                    mask_path = candidate
                    break
        if mask_path is None:
            missing_masks.append(stem)
            continue
        pairs.append((image_path, mask_path))
    return pairs, missing_masks


def load_metadata(path: Path | None):
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("metadata-json must be a dict mapping case stem to text")
    return payload


def build_transform(image_size: int):
    return v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(image_size, image_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_rgb_image_pil(path: Path):
    if path.suffix.lower() != ".npy":
        return Image.open(path).convert("RGB")

    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]
    else:
        raise ValueError(f"Unsupported npy image shape {arr.shape} for {path}")

    arr = np.nan_to_num(arr)
    if np.issubdtype(arr.dtype, np.floating):
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        if arr_max <= 1.0 and arr_min >= 0.0:
            arr = arr * 255.0
        elif arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def load_image(path: Path, transform, device: str):
    image = load_rgb_image_pil(path)
    tensor = transform(v2.functional.to_image(image)).unsqueeze(0).to(device)
    return tensor


def load_mask(path: Path, image_size: int, device: str):
    if path.suffix.lower() == ".npy":
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
    mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    mask = F.interpolate(mask, size=(image_size, image_size), mode="nearest")
    return (mask > 0.5).float().to(device)


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor):
    probs = logits.sigmoid()
    numer = 2 * (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    return 1 - (numer / denom).mean()


def token_l2_regularization(tokens: torch.Tensor):
    return (tokens**2).mean()


def token_diversity_regularization(tokens: torch.Tensor):
    if tokens.shape[0] <= 1:
        return tokens.new_tensor(0.0)
    normalized = F.normalize(tokens, dim=-1)
    sim = torch.matmul(normalized, normalized.T)
    identity = torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
    off_diag = sim - identity
    return (off_diag**2).mean()


def resolve_text(stem: str, metadata: dict, default_text: str | None):
    text = metadata.get(stem, default_text)
    if text is None:
        raise ValueError("Missing text prompt for case and no default provided")
    return text


def train_one_epoch(
    model,
    pairs: list,
    metadata: dict,
    args,
    find_stage,
    geometric_prompt,
    transform,
):
    """Train free memory tokens for one epoch on the client's data."""
    optimizer = torch.optim.AdamW(model.memory_prompt_builder.parameters(), lr=args.lr)

    epoch_loss = 0.0
    perm = torch.randperm(len(pairs))
    for idx in perm.tolist():
        image_path, mask_path = pairs[idx]
        image = load_image(image_path, transform, args.device)
        target = load_mask(mask_path, args.image_size, args.device)
        text_prompt = resolve_text(image_path.stem, metadata, args.text_prompt)

        backbone_out = model.backbone.forward_image(image)
        backbone_out.update(model.backbone.forward_text([text_prompt], device=args.device))
        out = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_stage,
            find_target=None,
            geometric_prompt=geometric_prompt,
        )
        pred_logits = out["pred_masks"]
        pred_scores = (
            out["pred_logits"].sigmoid()
            * out["presence_logit_dec"].sigmoid().unsqueeze(1)
        ).squeeze(-1)
        best_idx = torch.argmax(pred_scores, dim=1)
        selected_logits = pred_logits[
            torch.arange(pred_logits.shape[0], device=pred_logits.device), best_idx
        ]
        selected_logits = selected_logits.unsqueeze(1)
        selected_logits = F.interpolate(
            selected_logits,
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=False,
        )

        bce = F.binary_cross_entropy_with_logits(selected_logits, target)
        dice = dice_loss_from_logits(selected_logits, target)
        tokens = model.memory_prompt_builder.memory_tokens
        reg_l2 = token_l2_regularization(tokens)
        reg_div = token_diversity_regularization(tokens)
        loss = (
            dice
            + 0.3 * bce
            + args.token_l2_weight * reg_l2
            + args.token_diversity_weight * reg_div
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss.item())

    epoch_loss /= max(len(pairs), 1)
    return epoch_loss


def fedavg_aggregate(client_states: list[dict], client_weights: list[float]) -> dict:
    """
    Federated Averaging (FedAvg) aggregation.

    Args:
        client_states: List of state_dicts from each client
        client_weights: List of weights (e.g., number of samples) for each client

    Returns:
        Aggregated state_dict
    """
    total_weight = sum(client_weights)
    aggregated = {}

    for key in client_states[0].keys():
        weighted_sum = None
        for state, weight in zip(client_states, client_weights):
            w = weight / total_weight
            if weighted_sum is None:
                weighted_sum = state[key] * w
            else:
                weighted_sum = weighted_sum + state[key] * w
        aggregated[key] = weighted_sum

    return aggregated


def build_find_stage_and_prompt(device):
    find_stage = FindStage(
        img_ids=torch.tensor([0], device=device, dtype=torch.long),
        text_ids=torch.tensor([0], device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    geometric_prompt = Prompt(
        box_embeddings=torch.zeros(0, 1, 4, device=device),
        box_mask=torch.zeros(1, 0, device=device, dtype=torch.bool),
    )
    return find_stage, geometric_prompt


def run_single_holdout(args, holdout_site: Path, site_dirs, metadata_map):
    train_sites = [site_dir for site_dir in site_dirs if site_dir.name != holdout_site.name]
    if not train_sites:
        raise RuntimeError("Need at least one training site besides the held-out site.")

    print(
        f"Running federated free memory tokens with holdout={holdout_site.name} "
        f"and train_sites={[site.name for site in train_sites]}"
    )

    transform = build_transform(args.image_size)
    find_stage, geometric_prompt = build_find_stage_and_prompt(args.device)

    # Collect client data (pairs list and sample count for each site)
    client_data = []
    for site_dir in train_sites:
        image_dir = site_dir / "data_npy"
        mask_dir = site_dir / "label_npy"
        pairs, _ = collect_pairs(image_dir, mask_dir, args.extensions)
        if args.limit_per_site is not None:
            pairs = pairs[:args.limit_per_site]
        if pairs:
            client_data.append({"site_dir": site_dir, "pairs": pairs, "num_samples": len(pairs)})
            print(f"  Site {site_dir.name}: {len(pairs)} samples")

    if len(client_data) == 0:
        raise RuntimeError("No clients had valid data for training.")

    # Initialize global model
    global_model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint_path),
        load_from_HF=False,
        device=args.device,
        eval_mode=True,
        use_free_memory_tokens=True,
        free_memory_num_tokens=args.num_tokens,
    )
    global_state = global_model.memory_prompt_builder.state_dict()
    del global_model
    torch.cuda.empty_cache()

    client_weights = [c["num_samples"] for c in client_data]
    total_weight = sum(client_weights)
    print(f"\n=== Starting Federated Training: {args.epochs} rounds ===")
    print(f"Client weights: {client_weights}")

    all_round_histories = []

    for round_idx in range(args.epochs):
        print(f"\n--- Round {round_idx + 1}/{args.epochs} ---")
        round_losses = []

        # Each client trains locally for 1 epoch with the global model
        client_states = []
        for client in client_data:
            print(f"  Training {client['site_dir'].name}...", end=" ")

            # Build fresh model and load global weights
            model = build_sam3_image_model(
                checkpoint_path=str(args.checkpoint_path),
                load_from_HF=False,
                device=args.device,
                eval_mode=True,
                use_free_memory_tokens=True,
                free_memory_num_tokens=args.num_tokens,
            )
            model.memory_prompt_builder.load_state_dict(global_state, strict=False)
            model.eval()

            for param in model.parameters():
                param.requires_grad = False
            for param in model.memory_prompt_builder.parameters():
                param.requires_grad = True
            model.memory_prompt_builder.train()

            # Train 1 epoch locally
            epoch_loss = train_one_epoch(
                model=model,
                pairs=client["pairs"],
                metadata=metadata_map,
                args=args,
                find_stage=find_stage,
                geometric_prompt=geometric_prompt,
                transform=transform,
            )

            client_states.append(model.memory_prompt_builder.state_dict())
            round_losses.append(epoch_loss)
            print(f"loss={epoch_loss:.6f}")

            del model
            torch.cuda.empty_cache()

        # FedAvg aggregation
        print(f"  Aggregating {len(client_states)} clients via FedAvg...")
        global_state = fedavg_aggregate(client_states, client_weights)

        avg_loss = sum(round_losses) / len(round_losses)
        all_round_histories.append({"round": round_idx + 1, "avg_loss": avg_loss})
        print(f"  Round {round_idx + 1} avg loss: {avg_loss:.6f}")

    # Save aggregated model
    holdout_output_dir = args.output_dir / holdout_site.name
    holdout_output_dir.mkdir(parents=True, exist_ok=True)

    aggregated_path = holdout_output_dir / "fedavg_free_memory_tokens.pt"
    torch.save({
        "memory_prompt_builder": global_state,
        "config": {
            "num_tokens": args.num_tokens,
            "epochs": args.epochs,
            "lr": args.lr,
            "image_size": args.image_size,
            "token_l2_weight": args.token_l2_weight,
            "token_diversity_weight": args.token_diversity_weight,
            "train_sites": [site.name for site in train_sites],
            "holdout_site": holdout_site.name,
            "client_weights": client_weights,
            "aggregation": "fedavg",
        },
        "round_histories": all_round_histories,
    }, aggregated_path)
    print(f"Saved aggregated model to {aggregated_path}")

    # Evaluate on holdout site
    return evaluate_holdout_site(
        args=args,
        holdout_site=holdout_site,
        holdout_output_dir=holdout_output_dir,
        aggregated_state=global_state,
        metadata_map=metadata_map,
    )


def evaluate_holdout_site(
    args,
    holdout_site: Path,
    holdout_output_dir: Path,
    aggregated_state: dict,
    metadata_map: dict,
):
    from eval_medical_static_memory import (
        dice_score,
        iou_score,
        load_image_model,
        predict_mask_with_text,
    )

    image_dir = holdout_site / "val_data_npy"
    mask_dir = holdout_site / "val_label_npy"
    pairs, missing_masks = collect_pairs(image_dir, mask_dir, args.extensions)
    if args.limit_per_site is not None:
        pairs = pairs[:args.limit_per_site]
    if not pairs:
        raise RuntimeError(f"No pairs found for holdout site {holdout_site.name}")

    # Load baseline model
    baseline_model, baseline_processor = load_image_model(args.checkpoint_path, args.device)

    # Load aggregated free memory tokens into a fresh model
    aggregated_model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint_path),
        load_from_HF=False,
        device=args.device,
        eval_mode=True,
        use_free_memory_tokens=True,
        free_memory_num_tokens=args.num_tokens,
    )
    aggregated_model.memory_prompt_builder.load_state_dict(aggregated_state, strict=False)
    aggregated_model.eval()

    holdout_output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        (holdout_output_dir / "baseline").mkdir(parents=True, exist_ok=True)
        (holdout_output_dir / "fedavg").mkdir(parents=True, exist_ok=True)

    case_results = []
    for idx, (image_path, mask_path) in enumerate(pairs, start=1):
        gt_mask = load_mask(mask_path, args.image_size, args.device)
        text_prompt = resolve_text(image_path.stem, metadata_map, args.text_prompt)

        # Baseline prediction
        baseline_pred, baseline_score = predict_mask_with_text(
            baseline_model,
            baseline_processor,
            image_path,
            text_prompt,
            args.image_size,
        )
        baseline_pred = baseline_pred.to(gt_mask.device)
        if baseline_score < args.no_object_threshold:
            baseline_pred = torch.zeros_like(baseline_pred)

        # FedAvg model prediction
        if baseline_score >= args.memory_trigger_threshold:
            fedavg_pred = baseline_pred.clone()
            fedavg_score = baseline_score
            strategy = "baseline_passthrough"
        else:
            fedavg_pred, fedavg_score = predict_mask_with_text(
                aggregated_model,
                None,  # processor not needed for free memory tokens model
                image_path,
                text_prompt,
                args.image_size,
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

        if args.save_predictions:
            from eval_medical_static_memory import save_mask
            save_mask(baseline_pred, holdout_output_dir / "baseline" / f"{image_path.stem}.png")
            save_mask(fedavg_pred, holdout_output_dir / "fedavg" / f"{image_path.stem}.png")

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
    summary_path = holdout_output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved holdout metrics to {summary_path}")
    return summary


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
        all_results.append({
            "holdout_site": holdout_site.name,
            "baseline_mean": summary["baseline_mean"],
            "fedavg_mean": summary["fedavg_mean"],
            "num_cases": summary["num_cases"],
        })

    if len(all_results) > 1:
        overall = {
            "dataset_root": str(args.dataset_root),
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
        overall_path = args.output_dir / "loso_summary.json"
        overall_path.write_text(json.dumps(overall, indent=2))
        print(f"Saved LOSO summary to {overall_path}")


if __name__ == "__main__":
    main()
