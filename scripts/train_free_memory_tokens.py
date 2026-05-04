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

from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model_builder import build_sam3_image_model
from prompt_system import PromptSystem


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train free memory tokens as an ablation on top of frozen SAM3."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--text-prompt", type=str, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--limit", type=int, default=None)
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
    parser.add_argument(
        "--prompt-system-mode",
        type=str,
        default="raw",
        choices=["raw", "canonical", "expanded"],
        help="How to normalize prompts before training.",
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
    return parser.parse_args()


def normalize_stem(stem: str, prefix_to_strip: str = "", suffix_to_strip: str = ""):
    if prefix_to_strip and stem.startswith(prefix_to_strip):
        stem = stem[len(prefix_to_strip) :]
    if suffix_to_strip and stem.endswith(suffix_to_strip):
        stem = stem[: -len(suffix_to_strip)]
    return stem


def collect_pairs(
    image_dir: Path,
    mask_dir: Path,
    image_stem_prefix: str = "",
    image_stem_suffix: str = "",
    mask_stem_prefix: str = "",
    mask_stem_suffix: str = "",
):
    image_paths = {}
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            key = normalize_stem(path.stem, image_stem_prefix, image_stem_suffix)
            if key in image_paths:
                raise ValueError(
                    f"Duplicate image key {key!r}: {image_paths[key]} and {path}. "
                    "Adjust --image-stem-prefix/--image-stem-suffix or rename files."
                )
            image_paths[key] = path

    mask_paths = {}
    for path in mask_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
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
        if mask_path is not None:
            pairs.append((image_path, mask_path))
        else:
            missing_masks.append(key)
    if missing_masks:
        preview = ", ".join(missing_masks[:10])
        suffix = " ..." if len(missing_masks) > 10 else ""
        print(
            f"Warning: skipped {len(missing_masks)} images without matched masks "
            f"after stem normalization: {preview}{suffix}"
        )
    return pairs


def load_metadata(path: Path | None):
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("metadata-json must be a dict mapping case stem to text")
    return payload


def build_transform(image_size: int):
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


def load_image(path: Path, transform, device: str):
    image = load_rgb_image_pil(path)
    tensor = transform(v2.functional.to_image(image)).unsqueeze(0).to(device)
    return tensor


def load_image_batch(paths, transform, device: str):
    return torch.cat([load_image(path, transform, device) for path in paths], dim=0)


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


def load_mask_batch(paths, image_size: int, device: str):
    return torch.cat([load_mask(path, image_size, device) for path in paths], dim=0)


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


def batch_items(items, batch_size: int):
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main():
    args = parse_args()
    pairs = collect_pairs(
        args.image_dir,
        args.mask_dir,
        image_stem_prefix=args.image_stem_prefix,
        image_stem_suffix=args.image_stem_suffix,
        mask_stem_prefix=args.mask_stem_prefix,
        mask_stem_suffix=args.mask_stem_suffix,
    )
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError("No image/mask pairs found.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    metadata = load_metadata(args.metadata_json)
    prompt_system = PromptSystem.from_paths(
        attributes_json=args.attributes_json,
        aliases_json=args.aliases_json,
    )
    transform = build_transform(args.image_size)

    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint_path),
        load_from_HF=False,
        device=args.device,
        eval_mode=True,
        use_free_memory_tokens=True,
        free_memory_num_tokens=args.num_tokens,
    )
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    if model.memory_prompt_builder is None:
        raise RuntimeError("Free memory prompt builder was not created")
    for param in model.memory_prompt_builder.parameters():
        param.requires_grad = True
    model.memory_prompt_builder.train()

    optimizer = torch.optim.AdamW(model.memory_prompt_builder.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in batch_items(pairs, args.batch_size):
            image_paths = [image_path for image_path, _ in batch]
            mask_paths = [mask_path for _, mask_path in batch]
            text_prompts = []
            for image_path in image_paths:
                text_prompt = resolve_text(image_path.stem, metadata, args.text_prompt)
                text_prompt = prompt_system.transform_prompt(
                    text_prompt,
                    mode=args.prompt_system_mode,
                    topk_attrs=args.prompt_topk_attrs,
                    fmt=args.prompt_format,
                    separator=args.prompt_separator,
                )
                text_prompts.append(text_prompt)

            batch_size = len(batch)
            image = load_image_batch(image_paths, transform, args.device)
            target = load_mask_batch(mask_paths, args.image_size, args.device)
            backbone_out = model.backbone.forward_image(image)
            backbone_out.update(model.backbone.forward_text(text_prompts, device=args.device))
            find_stage = FindStage(
                img_ids=torch.arange(batch_size, device=args.device, dtype=torch.long),
                text_ids=torch.arange(batch_size, device=args.device, dtype=torch.long),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
            geometric_prompt = Prompt(
                box_embeddings=torch.zeros(0, batch_size, 4, device=args.device),
                box_mask=torch.zeros(batch_size, 0, device=args.device, dtype=torch.bool),
            )
            out = model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_stage,
                find_target=None,
                geometric_prompt=geometric_prompt,
            )
            pred_logits = out["pred_masks"]
            pred_scores = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
            best_idx = torch.argmax(pred_scores, dim=1)
            selected_logits = pred_logits[torch.arange(pred_logits.shape[0], device=pred_logits.device), best_idx]
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
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {epoch_loss:.6f}")

    payload = {
        "memory_prompt_builder": model.memory_prompt_builder.state_dict(),
        "config": {
            "num_tokens": args.num_tokens,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "image_size": args.image_size,
            "token_l2_weight": args.token_l2_weight,
            "token_diversity_weight": args.token_diversity_weight,
            "prompt_system_mode": args.prompt_system_mode,
            "prompt_topk_attrs": args.prompt_topk_attrs,
            "prompt_format": args.prompt_format,
            "prompt_separator": args.prompt_separator,
            "prompt_system": prompt_system.manifest(),
        },
        "history": history,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_path)
    print(f"Saved free memory token checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
