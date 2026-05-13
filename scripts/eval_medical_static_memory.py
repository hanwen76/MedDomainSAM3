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
from prompt_system import PromptSystem


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
    parser.add_argument("--task-encoder-pool-image-dir", type=Path, default=None)
    parser.add_argument("--task-encoder-pool-mask-dir", type=Path, default=None)
    parser.add_argument("--task-encoder-sample-count", type=int, default=16)
    parser.add_argument("--task-encoder-seed", type=int, default=0)
    parser.add_argument("--task-encoder-batch-size", type=int, default=4)
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
        "--image-stem-suffix",
        type=str,
        default="",
        help=(
            "Optional suffix stripped from image filenames before pairing, "
            "e.g. '_image' maps case001_image.npy to case001."
        ),
    )
    parser.add_argument(
        "--image-stem-prefix",
        type=str,
        default="",
        help=(
            "Optional prefix stripped from image filenames before pairing, "
            "e.g. 'image_' maps image_case001.npy to case001."
        ),
    )
    parser.add_argument(
        "--mask-stem-suffix",
        type=str,
        default="",
        help=(
            "Optional suffix stripped from mask filenames before pairing, "
            "e.g. '_mask' maps case001_mask.npy to case001."
        ),
    )
    parser.add_argument(
        "--mask-stem-prefix",
        type=str,
        default="",
        help=(
            "Optional prefix stripped from mask filenames before pairing, "
            "e.g. 'mask_' maps mask_case001.npy to case001."
        ),
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=sorted(IMAGE_SUFFIXES),
        help="Image suffixes to consider when pairing files by stem.",
    )
    parser.add_argument(
        "--prompt-system-mode",
        type=str,
        default="raw",
        choices=["raw", "canonical", "expanded"],
        help="How to normalize prompts before inference.",
    )
    parser.add_argument("--attributes-json", type=Path, default=None)
    parser.add_argument("--aliases-json", type=Path, default=None)
    parser.add_argument("--prompt-topk-attrs", type=int, default=3)
    parser.add_argument(
        "--prompt-format",
        type=str,
        default="attrs_then_class",
        choices=["attrs_then_class", "class_then_attrs"],
    )
    parser.add_argument("--prompt-separator", type=str, default=", ")
    parser.add_argument(
        "--pairing-mode",
        type=str,
        default="stem",
        choices=["stem", "sequential"],
        help="How to pair images and masks. Sequential pairing matches sorted files by order.",
    )
    return parser.parse_args()


def normalize_stem(stem: str, prefix_to_strip: str = "", suffix_to_strip: str = ""):
    if prefix_to_strip and stem.startswith(prefix_to_strip):
        stem = stem[len(prefix_to_strip) :]
    if suffix_to_strip and stem.endswith(suffix_to_strip):
        stem = stem[: -len(suffix_to_strip)]
    return stem


def natural_sort_key(path: Path):
    import re

    parts = re.split(r"(\d+)", str(path))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def collect_file_list(directory: Path, extensions):
    ext_set = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    files = [path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in ext_set]
    return sorted(files, key=natural_sort_key)


def collect_pairs(
    image_dir: Path,
    mask_dir: Path,
    extensions,
    image_stem_prefix: str = "",
    image_stem_suffix: str = "",
    mask_stem_prefix: str = "",
    mask_stem_suffix: str = "",
    pairing_mode: str = "stem",
):
    if pairing_mode not in {"stem", "sequential"}:
        raise ValueError("--pairing-mode must be one of: stem, sequential")

    if pairing_mode == "sequential":
        image_paths = collect_file_list(image_dir, extensions)
        mask_paths = collect_file_list(mask_dir, extensions)
        if not image_paths or not mask_paths:
            return [], []
        pair_count = min(len(image_paths), len(mask_paths))
        if len(image_paths) != len(mask_paths):
            print(
                f"Warning: sequential pairing found {len(image_paths)} images and {len(mask_paths)} masks; "
                f"using first {pair_count} pairs."
            )
        return list(zip(image_paths[:pair_count], mask_paths[:pair_count])), []

    ext_set = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    image_paths = {}
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in ext_set:
            key = normalize_stem(path.stem, image_stem_prefix, image_stem_suffix)
            if key in image_paths:
                raise ValueError(
                    f"Duplicate image key {key!r}: {image_paths[key]} and {path}. "
                    "Adjust --image-stem-prefix/--image-stem-suffix or rename files."
                )
            image_paths[key] = path

    mask_paths = {}
    for path in mask_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in ext_set:
            key = normalize_stem(path.stem, mask_stem_prefix, mask_stem_suffix)
            if key in mask_paths:
                raise ValueError(
                    f"Duplicate mask key {key!r}: {mask_paths[key]} and {path}. "
                    "Adjust --mask-stem-prefix/--mask-stem-suffix or rename files."
                )
            mask_paths[key] = path

    pairs = []
    missing_masks = []
    for key, image_path in sorted(image_paths.items()):
        mask_path = mask_paths.get(key)
        if mask_path is None:
            missing_masks.append(key)
            continue
        pairs.append((image_path, mask_path))
    if missing_masks:
        preview = ", ".join(missing_masks[:10])
        suffix = " ..." if len(missing_masks) > 10 else ""
        print(
            f"Warning: skipped {len(missing_masks)} images without matched masks "
            f"after stem normalization: {preview}{suffix}"
        )
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


def load_image_batch(paths, transform, device: str):
    return torch.cat([load_image_tensor(path, transform, device) for path in paths], dim=0)


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


def load_mask_batch(paths, image_size: int, device: str):
    return torch.cat([load_mask_tensor(path, image_size, device) for path in paths], dim=0)


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


def load_image_model_with_task_encoder(
    checkpoint_path: Path | None,
    device: str,
    free_memory_ckpt: Path | None = None,
    free_memory_num_tokens: int = 4,
):
    model = build_sam3_image_model(
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        load_from_HF=False,
        device=device,
        eval_mode=True,
        use_memory_prompt=True,
        use_free_memory_tokens=free_memory_ckpt is not None,
        free_memory_num_tokens=free_memory_num_tokens,
    )
    if free_memory_ckpt is not None:
        _load_free_memory_state_into_model(model, free_memory_ckpt)
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
    if any(key.startswith("prompt_tuning_builder.") for key in state):
        state = {
            key.replace("prompt_tuning_builder.", "", 1): value
            for key, value in state.items()
            if key.startswith("prompt_tuning_builder.")
        }
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


def _load_free_memory_state_into_model(model, free_memory_ckpt: Path):
    payload = torch.load(free_memory_ckpt, map_location="cpu")
    state = payload.get("memory_prompt_builder", payload)
    target_builder = getattr(model, "memory_prompt_builder", None)
    if target_builder is None:
        raise RuntimeError("Model does not have a memory_prompt_builder")
    if hasattr(target_builder, "prompt_tuning_builder") and target_builder.prompt_tuning_builder is not None:
        target_builder = target_builder.prompt_tuning_builder
    if any(key.startswith("prompt_tuning_builder.") for key in state):
        state = {
            key.replace("prompt_tuning_builder.", "", 1): value
            for key, value in state.items()
            if key.startswith("prompt_tuning_builder.")
        }
    missing_keys, unexpected_keys = target_builder.load_state_dict(state, strict=False)
    if missing_keys:
        print(f"Missing keys while loading free memory tokens: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys while loading free memory tokens: {unexpected_keys}")


def load_image_model_with_memory_and_free_memory(
    checkpoint_path: Path | None,
    device: str,
    static_memory_bank_path: Path,
    free_memory_ckpt: Path,
    static_memory_topk: int = 4,
    static_memory_text_weight: float = 1.0,
    free_memory_num_tokens: int = 4,
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
        use_free_memory_tokens=True,
        free_memory_num_tokens=free_memory_num_tokens,
    )
    _load_free_memory_state_into_model(model, free_memory_ckpt)
    processor = Sam3Processor(model, device=device)
    return model, processor


def sample_task_pool(pairs, sample_count: int, seed: int):
    if sample_count <= 0 or sample_count >= len(pairs):
        return list(pairs)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pairs), size=sample_count, replace=False)
    return [pairs[int(index)] for index in indices]


@torch.inference_mode()
def encode_task_pool(
    model,
    pairs,
    transform,
    device: str,
    image_size: int,
    batch_size: int,
):
    task_embeddings = []
    memory_keys = []
    was_training = model.training
    model.eval()
    for batch in batch_items(pairs, batch_size):
        image_paths = [image_path for image_path, _ in batch]
        mask_paths = [mask_path for _, mask_path in batch]
        images = load_image_batch(image_paths, transform, device)
        masks = load_mask_batch(mask_paths, image_size, device)
        backbone_out = model.backbone.forward_image(images)
        img_ids = torch.arange(len(batch), device=device, dtype=torch.long)
        _, img_feats, _, feat_sizes = model._get_img_feats(backbone_out, img_ids)
        top_feat = img_feats[-1]
        height, width = feat_sizes[-1]
        channels = top_feat.shape[-1]
        pix_feat = top_feat.permute(1, 2, 0).reshape(len(batch), channels, height, width)
        support_mask = F.interpolate(masks.float(), size=(height, width), mode="nearest")
        mask_sum = support_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        foreground = (pix_feat * support_mask).sum(dim=(2, 3), keepdim=True) / mask_sum
        foreground = foreground.flatten(1)
        context = pix_feat.mean(dim=(2, 3))
        task_embeddings.append(torch.stack([foreground, context], dim=1).detach().cpu())
        memory_keys.append(context.detach().cpu())
    if was_training:
        model.train()
    return torch.cat(task_embeddings, dim=0), torch.cat(memory_keys, dim=0)


def batch_items(items, batch_size: int):
    if batch_size <= 0:
        raise ValueError("--task-encoder-batch-size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def inject_task_pool(model, task_embeddings: torch.Tensor, memory_keys: torch.Tensor, device: str):
    builder = getattr(model, "memory_prompt_builder", None)
    if builder is None:
        raise RuntimeError("Model does not have a memory_prompt_builder")
    task_builder = getattr(builder, "task_builder", builder)
    if not hasattr(task_builder, "set_task_support"):
        raise RuntimeError("memory_prompt_builder does not support task-pool injection")
    task_builder.set_task_support(
        task_embeddings.to(device),
        memory_keys.to(device),
    )


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
    pairs, missing_masks = collect_pairs(
        args.image_dir,
        args.mask_dir,
        args.extensions,
        image_stem_prefix=args.image_stem_prefix,
        image_stem_suffix=args.image_stem_suffix,
        mask_stem_prefix=args.mask_stem_prefix,
        mask_stem_suffix=args.mask_stem_suffix,
        pairing_mode=args.pairing_mode,
    )
    metadata_map = load_metadata(args.metadata_json)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError("No image/mask pairs found. Check filenames and directories.")
    use_task_encoder = (
        args.task_encoder_pool_image_dir is not None
        or args.task_encoder_pool_mask_dir is not None
    )
    if use_task_encoder and (
        args.task_encoder_pool_image_dir is None
        or args.task_encoder_pool_mask_dir is None
    ):
        raise ValueError(
            "--task-encoder-pool-image-dir and --task-encoder-pool-mask-dir must be provided together"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        (args.output_dir / "baseline").mkdir(parents=True, exist_ok=True)
        (args.output_dir / "static_memory").mkdir(parents=True, exist_ok=True)

    prompt_system = PromptSystem.from_paths(
        attributes_json=args.attributes_json,
        aliases_json=args.aliases_json,
    )
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
    if use_task_encoder:
        static_image_model, static_processor = load_image_model_with_task_encoder(
            args.checkpoint_path,
            args.device,
            free_memory_ckpt=args.free_memory_ckpt,
            free_memory_num_tokens=args.free_memory_num_tokens,
        )
        task_pairs, task_missing_masks = collect_pairs(
            args.task_encoder_pool_image_dir,
            args.task_encoder_pool_mask_dir,
            args.extensions,
            image_stem_prefix=args.image_stem_prefix,
            image_stem_suffix=args.image_stem_suffix,
            mask_stem_prefix=args.mask_stem_prefix,
            mask_stem_suffix=args.mask_stem_suffix,
            pairing_mode=args.pairing_mode,
        )
        if not task_pairs:
            raise RuntimeError("No image/mask pairs found in task encoder pool.")
        sampled_task_pairs = sample_task_pool(
            task_pairs,
            args.task_encoder_sample_count,
            args.task_encoder_seed,
        )
        task_embeddings, memory_keys = encode_task_pool(
            static_image_model,
            sampled_task_pairs,
            image_transform,
            args.device,
            args.image_size,
            args.task_encoder_batch_size,
        )
        inject_task_pool(static_image_model, task_embeddings, memory_keys, args.device)
        print(
            f"Injected task encoder pool with {len(sampled_task_pairs)} sampled train cases"
        )
    elif args.free_memory_ckpt is not None and args.static_memory_bank_path is not None:
        static_image_model, static_processor = load_image_model_with_memory_and_free_memory(
            args.checkpoint_path,
            args.device,
            static_memory_bank_path=args.static_memory_bank_path,
            free_memory_ckpt=args.free_memory_ckpt,
            static_memory_topk=args.static_memory_topk,
            static_memory_text_weight=args.static_memory_text_weight,
            free_memory_num_tokens=args.free_memory_num_tokens,
        )
    elif args.free_memory_ckpt is not None:
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
        if text_prompt is not None:
            text_prompt = prompt_system.transform_prompt(
                text_prompt,
                mode=args.prompt_system_mode,
                topk_attrs=args.prompt_topk_attrs,
                fmt=args.prompt_format,
                separator=args.prompt_separator,
            )
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
                    memory_strategy = "task_encoder_refine" if use_task_encoder else "memory_refine"
            else:
                static_tracker.set_static_memory_text_prompt(text_prompt)
                static_pred = predict_mask(static_tracker, image_tensor, point_inputs)
                static_score = None
                memory_strategy = "tracker_memory"
            result["static_memory"] = {
                "dice": dice_score(static_pred, gt_mask),
                "iou": iou_score(static_pred, gt_mask),
            }
            result["task_encoder"] = result["static_memory"]
            if static_score is not None:
                result["static_memory"]["score"] = static_score
                result["task_encoder"]["score"] = static_score
            result["static_memory"]["strategy"] = memory_strategy
            result["task_encoder"]["strategy"] = memory_strategy
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
        "task_encoder_mean": (
            summarize_metrics(case_results, "task_encoder")
            if use_task_encoder
            else None
        ),
        "cases": case_results,
        "missing_masks": missing_masks,
        "task_encoder_pool": (
            {
                "image_dir": str(args.task_encoder_pool_image_dir),
                "mask_dir": str(args.task_encoder_pool_mask_dir),
                "sample_count": args.task_encoder_sample_count,
                "seed": args.task_encoder_seed,
                "missing_masks": task_missing_masks,
            }
            if use_task_encoder
            else None
        ),
    }
    summary_path = args.output_dir / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved metrics to {summary_path}")


if __name__ == "__main__":
    main()
