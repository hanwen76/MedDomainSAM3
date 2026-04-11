#!/usr/bin/env python3

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from sam3.model_builder import build_tracker


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SAM3 medical segmentation with and without static memory."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--static-memory-bank-path", type=Path, default=None)
    parser.add_argument("--free-memory-ckpt", type=Path, default=None)
    parser.add_argument("--text-prompt", type=str, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--static-memory-text-weight", type=float, default=1.0)
    parser.add_argument("--static-memory-fusion-alpha", type=float, default=0.15)
    parser.add_argument(
        "--no-object-threshold",
        type=float,
        default=0.2,
        help="If the best text grounding score is below this threshold, predict an empty mask.",
    )
    parser.add_argument(
        "--memory-trigger-threshold",
        type=float,
        default=0.8,
        help="If the baseline score is below this threshold (and above no-object threshold), allow memory refinement.",
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="text",
        choices=["text", "box"],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--static-memory-topk", type=int, default=4)
    parser.add_argument("--free-memory-num-tokens", type=int, default=4)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(IMAGE_SUFFIXES),
        help="Image suffixes to consider when pairing files by stem.",
    )
    return parser.parse_args()


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


def load_metadata(metadata_json: Path | None):
    if metadata_json is None:
        return {}
    payload = json.loads(metadata_json.read_text())
    if isinstance(payload, dict):
        return payload
    raise ValueError("metadata-json must contain a JSON object mapping case stem to text")


def build_image_transform(image_size: int):
    return v2.Compose(
        [
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


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


def load_image_tensor(path: Path, transform, device: str):
    image = load_rgb_image_pil(path)
    tensor = v2.functional.to_image(image)
    tensor = transform(tensor).unsqueeze(0).to(device)
    return tensor


def load_mask_tensor(path: Path, image_size: int, device: str):
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
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    mask_tensor = F.interpolate(mask_tensor, size=(image_size, image_size), mode="nearest")
    mask_tensor = (mask_tensor > 0.5).float().to(device)
    return mask_tensor


def build_box_prompt(mask_tensor: torch.Tensor, image_size: int):
    mask_2d = mask_tensor[0, 0] > 0.5
    coords = torch.nonzero(mask_2d, as_tuple=False)
    if coords.numel() == 0:
        return None

    y_min = coords[:, 0].min().float()
    x_min = coords[:, 1].min().float()
    y_max = coords[:, 0].max().float()
    x_max = coords[:, 1].max().float()

    points = torch.tensor(
        [[[x_min, y_min], [x_max, y_max]]],
        dtype=torch.float32,
        device=mask_tensor.device,
    )
    labels = torch.tensor([[2, 3]], dtype=torch.int32, device=mask_tensor.device)
    return {
        "point_coords": points.clamp(0, image_size - 1),
        "point_labels": labels,
    }


def resolve_text_prompt(case_stem: str, metadata_map: dict, default_text_prompt: str | None):
    return metadata_map.get(case_stem, default_text_prompt)


def dice_score(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred = pred_mask.float().flatten()
    gt = gt_mask.float().flatten()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * (pred * gt).sum() / denom).item())


def iou_score(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred = pred_mask.bool()
    gt = gt_mask.bool()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0
    inter = (pred & gt).sum()
    return float((inter.float() / union.float()).item())


def load_tracker(
    checkpoint_path: Path | None,
    device: str,
    static_memory_bank_path=None,
    static_memory_topk=4,
    static_memory_text_weight=1.0,
    static_memory_fusion_alpha=0.15,
    enable_text_encoder=False,
):
    tracker = build_tracker(
        apply_temporal_disambiguation=False,
        with_backbone=True,
        use_static_memory=static_memory_bank_path is not None,
        static_memory_bank_path=(
            None if static_memory_bank_path is None else str(static_memory_bank_path)
        ),
        static_memory_topk=static_memory_topk,
        static_memory_text_weight=static_memory_text_weight,
        static_memory_fusion_alpha=static_memory_fusion_alpha,
        enable_text_encoder=enable_text_encoder,
    )
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        tracker_state = {
            key.replace("tracker.", ""): value
            for key, value in ckpt.items()
            if key.startswith("tracker.")
        }
        tracker_state.update(
            {
                key.replace(
                    "detector.backbone.vision_backbone.",
                    "backbone.vision_backbone.",
                ): value
                for key, value in ckpt.items()
                if key.startswith("detector.backbone.vision_backbone.")
            }
        )
        if enable_text_encoder:
            tracker_state.update(
                {
                    key.replace(
                        "detector.backbone.language_backbone.",
                        "backbone.language_backbone.",
                    ): value
                    for key, value in ckpt.items()
                    if key.startswith("detector.backbone.language_backbone.")
                }
            )
        missing_keys, unexpected_keys = tracker.load_state_dict(
            tracker_state,
            strict=False,
        )
        if missing_keys:
            print(f"Missing keys while loading tracker checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys while loading tracker checkpoint: {unexpected_keys}")
    else:
        print("Warning: evaluating without a tracker checkpoint.")

    tracker = tracker.to(device)
    tracker.eval()
    return tracker


def load_image_model(checkpoint_path: Path | None, device: str):
    model = build_sam3_image_model(
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        load_from_HF=False,
        device=device,
        eval_mode=True,
    )
    processor = Sam3Processor(model, device=device)
    return model, processor


def load_image_model_with_memory(
    checkpoint_path: Path | None,
    device: str,
    static_memory_bank_path: Path,
    static_memory_topk: int = 4,
    static_memory_text_weight: float = 1.0,
):
    model = build_sam3_image_model(
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        load_from_HF=False,
        device=device,
        eval_mode=True,
        use_memory_prompt=True,
        static_memory_bank_path=str(static_memory_bank_path),
        static_memory_topk=static_memory_topk,
        static_memory_text_weight=static_memory_text_weight,
    )
    processor = Sam3Processor(model, device=device)
    return model, processor


def load_image_model_with_free_memory(
    checkpoint_path: Path | None,
    device: str,
    free_memory_ckpt: Path,
    free_memory_num_tokens: int = 4,
):
    model = build_sam3_image_model(
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        load_from_HF=False,
        device=device,
        eval_mode=True,
        use_free_memory_tokens=True,
        free_memory_num_tokens=free_memory_num_tokens,
    )
    payload = torch.load(free_memory_ckpt, map_location="cpu")
    state = payload.get("memory_prompt_builder", payload)
    missing_keys, unexpected_keys = model.memory_prompt_builder.load_state_dict(
        state,
        strict=False,
    )
    if missing_keys:
        print(f"Missing keys while loading free memory tokens: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys while loading free memory tokens: {unexpected_keys}")
    processor = Sam3Processor(model, device=device)
    return model, processor


@torch.inference_mode()
def predict_mask(tracker, image_tensor, point_inputs):
    backbone_out = tracker.forward_image(image_tensor)
    _, current_vision_feats, current_vision_pos_embeds, feat_sizes = tracker._prepare_backbone_features(backbone_out)
    current_out = tracker.track_step(
        frame_idx=0,
        is_init_cond_frame=True,
        current_vision_feats=current_vision_feats,
        current_vision_pos_embeds=current_vision_pos_embeds,
        feat_sizes=feat_sizes,
        image=image_tensor,
        point_inputs=point_inputs,
        mask_inputs=None,
        output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
        num_frames=1,
        run_mem_encoder=False,
        use_prev_mem_frame=False,
    )
    pred_logits = current_out["pred_masks_high_res"]
    pred_mask = (pred_logits > 0).float()
    return pred_mask


@torch.inference_mode()
def predict_mask_with_text(
    model,
    processor,
    image_path: Path,
    text_prompt: str,
    image_size: int,
):
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


def save_mask(mask_tensor: torch.Tensor, path: Path):
    mask = (mask_tensor[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask).save(path)


def summarize_metrics(results, key):
    if not results:
        return {"dice": 0.0, "iou": 0.0}
    return {
        "dice": float(sum(item[key]["dice"] for item in results) / len(results)),
        "iou": float(sum(item[key]["iou"] for item in results) / len(results)),
    }


def main():
    args = parse_args()
    pairs, missing_masks = collect_pairs(args.image_dir, args.mask_dir, args.extensions)
    metadata_map = load_metadata(args.metadata_json)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError("No image/mask pairs found. Check filenames and directories.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        (args.output_dir / "baseline").mkdir(parents=True, exist_ok=True)
        (args.output_dir / "static_memory").mkdir(parents=True, exist_ok=True)

    image_transform = build_image_transform(args.image_size)
    baseline_tracker = None
    baseline_image_model = None
    baseline_processor = None
    if args.prompt_mode == "box":
        baseline_tracker = load_tracker(args.checkpoint_path, args.device)
    else:
        baseline_image_model, baseline_processor = load_image_model(
            args.checkpoint_path,
            args.device,
        )
    static_tracker = None
    static_image_model = None
    static_processor = None
    if args.free_memory_ckpt is not None:
        static_image_model, static_processor = load_image_model_with_free_memory(
            args.checkpoint_path,
            args.device,
            free_memory_ckpt=args.free_memory_ckpt,
            free_memory_num_tokens=args.free_memory_num_tokens,
        )
    elif args.static_memory_bank_path is not None:
        if args.prompt_mode == "text":
            static_image_model, static_processor = load_image_model_with_memory(
                args.checkpoint_path,
                args.device,
                static_memory_bank_path=args.static_memory_bank_path,
                static_memory_topk=args.static_memory_topk,
                static_memory_text_weight=args.static_memory_text_weight,
            )
        else:
            static_tracker = load_tracker(
                args.checkpoint_path,
                args.device,
                static_memory_bank_path=args.static_memory_bank_path,
                static_memory_topk=args.static_memory_topk,
                static_memory_text_weight=args.static_memory_text_weight,
                static_memory_fusion_alpha=args.static_memory_fusion_alpha,
                enable_text_encoder=(
                    args.prompt_mode == "text"
                    or args.text_prompt is not None
                    or args.metadata_json is not None
                ),
            )

    case_results = []
    for idx, (image_path, mask_path) in enumerate(pairs, start=1):
        image_tensor = load_image_tensor(image_path, image_transform, args.device)
        gt_mask = load_mask_tensor(mask_path, args.image_size, args.device)
        text_prompt = resolve_text_prompt(image_path.stem, metadata_map, args.text_prompt)
        point_inputs = (
            build_box_prompt(gt_mask, args.image_size)
            if args.prompt_mode == "box"
            else None
        )
        if args.prompt_mode == "box" and point_inputs is None:
            print(f"Skipping empty mask: {image_path.name}")
            continue
        if args.prompt_mode == "text" and text_prompt is None:
            raise ValueError(
                "text mode requires --text-prompt or --metadata-json with per-case prompts"
            )

        if args.prompt_mode == "box":
            baseline_pred = predict_mask(baseline_tracker, image_tensor, point_inputs)
            baseline_score = None
        else:
            baseline_pred, baseline_score = predict_mask_with_text(
                baseline_image_model,
                baseline_processor,
                image_path,
                text_prompt,
                args.image_size,
            )
            baseline_pred = baseline_pred.to(gt_mask.device)

        if args.prompt_mode == "text" and baseline_score is not None:
            if baseline_score < args.no_object_threshold:
                baseline_pred = torch.zeros_like(baseline_pred)

        result = {
            "case_id": image_path.stem,
            "baseline": {
                "dice": dice_score(baseline_pred, gt_mask),
                "iou": iou_score(baseline_pred, gt_mask),
            },
        }
        if baseline_score is not None:
            result["baseline"]["score"] = baseline_score

        if static_tracker is not None or static_image_model is not None:
            if args.prompt_mode == "text":
                if baseline_score is not None and baseline_score < args.no_object_threshold:
                    static_pred = torch.zeros_like(baseline_pred)
                    static_score = baseline_score
                    memory_strategy = "empty"
                elif (
                    baseline_score is not None
                    and baseline_score >= args.memory_trigger_threshold
                ):
                    static_pred = baseline_pred.clone()
                    static_score = baseline_score
                    memory_strategy = "baseline_passthrough"
                else:
                    static_pred, static_score = predict_mask_with_text(
                        static_image_model,
                        static_processor,
                        image_path,
                        text_prompt,
                        args.image_size,
                    )
                    static_pred = static_pred.to(gt_mask.device)
                    memory_strategy = "memory_refine"
            else:
                static_tracker.set_static_memory_text_prompt(text_prompt)
                static_pred = predict_mask(static_tracker, image_tensor, point_inputs)
                static_score = None
                memory_strategy = "tracker_memory"
            result["static_memory"] = {
                "dice": dice_score(static_pred, gt_mask),
                "iou": iou_score(static_pred, gt_mask),
            }
            if static_score is not None:
                result["static_memory"]["score"] = static_score
            result["static_memory"]["strategy"] = memory_strategy
        else:
            static_pred = None

        case_results.append(result)
        if args.save_predictions:
            save_mask(baseline_pred, args.output_dir / "baseline" / f"{image_path.stem}.png")
            if static_pred is not None:
                save_mask(
                    static_pred,
                    args.output_dir / "static_memory" / f"{image_path.stem}.png",
                )

        if idx % 10 == 0 or idx == len(pairs):
            print(f"Processed {idx}/{len(pairs)} cases")

    summary = {
        "num_cases": len(case_results),
        "prompt_mode": args.prompt_mode,
        "baseline_mean": summarize_metrics(case_results, "baseline"),
        "static_memory_mean": (
            summarize_metrics(case_results, "static_memory")
            if (static_tracker is not None or static_image_model is not None)
            else None
        ),
        "cases": case_results,
        "missing_masks": missing_masks,
    }
    summary_path = args.output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved metrics to {summary_path}")


if __name__ == "__main__":
    main()
