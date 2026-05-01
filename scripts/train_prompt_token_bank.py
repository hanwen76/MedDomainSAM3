#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


class PromptTokenBank(nn.Module):
    """
    One token-group per prompt.

    During training/inference, samples are routed by prompt_id, and only the
    corresponding prompt token-group is used.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_prompts: int,
        tokens_per_prompt: int = 1,
        init_std: float = 0.02,
        prompt_scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_prompts = int(num_prompts)
        self.tokens_per_prompt = int(tokens_per_prompt)
        self.prompt_scale = float(prompt_scale)
        # [num_prompts, tokens_per_prompt, hidden_dim]
        self.memory_tokens = nn.Parameter(
            torch.randn(self.num_prompts, self.tokens_per_prompt, self.hidden_dim)
            * init_std
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
        prompt = tokens.permute(1, 0, 2).contiguous() * self.prompt_scale  # [T, b, D]
        prompt_mask = torch.zeros(
            (batch_size, self.tokens_per_prompt),
            dtype=torch.bool,
            device=prompt.device,
        )
        return prompt, prompt_mask


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a prompt-token bank on mixed datasets. "
            "Same prompt shares one token-group, different prompts use different token-groups."
        )
    )
    parser.add_argument("--datasets-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--tokens-per-prompt", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--loader-mode",
        type=str,
        default="sequential",
        choices=["sequential", "shuffled"],
        help=(
            "sequential: iterate datasets in the order provided by datasets-json; "
            "shuffled: mix all samples and shuffle each epoch."
        ),
    )
    parser.add_argument("--token-l2-weight", type=float, default=1e-4)
    parser.add_argument("--token-diversity-weight", type=float, default=1e-3)
    parser.add_argument("--prompt-diversity-weight", type=float, default=1e-3)
    parser.add_argument(
        "--prompt-system-mode",
        type=str,
        default="raw",
        choices=["raw", "canonical", "expanded"],
        help="How to normalize prompts before grouping and training.",
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


def parse_datasets_json(path: Path):
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError(
            "datasets-json must be a non-empty list. "
            "Each item: {image_dir, mask_dir, prompt}"
        )
    datasets = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"datasets-json[{i}] must be an object")
        image_dir = Path(str(item.get("image_dir", "")))
        mask_dir = Path(str(item.get("mask_dir", "")))
        prompt = str(item.get("prompt", "")).strip()
        if not image_dir.is_dir():
            raise ValueError(f"datasets-json[{i}] invalid image_dir: {image_dir}")
        if not mask_dir.is_dir():
            raise ValueError(f"datasets-json[{i}] invalid mask_dir: {mask_dir}")
        if not prompt:
            raise ValueError(f"datasets-json[{i}] missing prompt")
        datasets.append({"image_dir": image_dir, "mask_dir": mask_dir, "prompt": prompt})
    return datasets


def collect_pairs(image_dir: Path, mask_dir: Path):
    image_paths = {}
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            image_paths[path.stem] = path
    pairs = []
    for stem, image_path in sorted(image_paths.items()):
        mask_path = None
        for ext in IMAGE_SUFFIXES:
            candidate = mask_dir / f"{stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    return pairs


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


def l2_regularization(x: torch.Tensor):
    return (x**2).mean()


def diversity_regularization(x: torch.Tensor):
    x2 = x.reshape(-1, x.shape[-1])
    if x2.shape[0] <= 1:
        return x.new_tensor(0.0)
    normalized = F.normalize(x2, dim=-1)
    sim = torch.matmul(normalized, normalized.T)
    identity = torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
    off_diag = sim - identity
    return (off_diag**2).mean()


def inter_prompt_diversity(memory_tokens: torch.Tensor):
    # memory_tokens: [num_prompts, T, D]
    proto = memory_tokens.mean(dim=1)  # [num_prompts, D]
    if proto.shape[0] <= 1:
        return memory_tokens.new_tensor(0.0)
    proto = F.normalize(proto, dim=-1)
    sim = torch.matmul(proto, proto.T)
    identity = torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
    off_diag = sim - identity
    return (off_diag**2).mean()


def main():
    args = parse_args()
    datasets = parse_datasets_json(args.datasets_json)
    rng = random.Random(args.shuffle_seed)
    prompt_system = PromptSystem.from_paths(
        attributes_json=args.attributes_json,
        aliases_json=args.aliases_json,
    )

    prompt_to_id = {}
    for d in datasets:
        normalized_prompt = prompt_system.transform_prompt(
            d["prompt"],
            mode=args.prompt_system_mode,
            topk_attrs=args.prompt_topk_attrs,
            fmt=args.prompt_format,
            separator=args.prompt_separator,
        )
        d["prompt_raw"] = d["prompt"]
        d["prompt"] = normalized_prompt
        if normalized_prompt not in prompt_to_id:
            prompt_to_id[normalized_prompt] = len(prompt_to_id)

    samples = []
    dataset_samples = []
    for d in datasets:
        pairs = collect_pairs(d["image_dir"], d["mask_dir"])
        if args.limit_per_dataset is not None:
            pairs = pairs[: args.limit_per_dataset]
        if len(pairs) == 0:
            raise RuntimeError(
                f"No image/mask pairs found for dataset image_dir={d['image_dir']}"
            )
        pid = prompt_to_id[d["prompt"]]
        cur = []
        for image_path, mask_path in pairs:
            item = {
                "prompt": d["prompt"],
                "prompt_raw": d["prompt_raw"],
                "prompt_id": pid,
                "image_path": image_path,
                "mask_path": mask_path,
                "image_dir": str(d["image_dir"]),
                "mask_dir": str(d["mask_dir"]),
            }
            samples.append(item)
            cur.append(item)
        dataset_samples.append(
            {
                "prompt": d["prompt"],
                "prompt_id": pid,
                "image_dir": str(d["image_dir"]),
                "mask_dir": str(d["mask_dir"]),
                "samples": cur,
            }
        )

    if len(samples) == 0:
        raise RuntimeError("No training samples found from datasets-json.")

    transform = build_transform(args.image_size)
    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint_path),
        load_from_HF=False,
        device=args.device,
        eval_mode=True,
        use_free_memory_tokens=False,
    )
    token_bank = PromptTokenBank(
        hidden_dim=model.hidden_dim,
        num_prompts=len(prompt_to_id),
        tokens_per_prompt=args.tokens_per_prompt,
    ).to(args.device)
    model.memory_prompt_builder = token_bank

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.memory_prompt_builder.parameters():
        param.requires_grad = True
    model.memory_prompt_builder.train()

    optimizer = torch.optim.AdamW(model.memory_prompt_builder.parameters(), lr=args.lr)
    find_stage = FindStage(
        img_ids=torch.tensor([0], device=args.device, dtype=torch.long),
        text_ids=torch.tensor([0], device=args.device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    geometric_prompt = Prompt(
        box_embeddings=torch.zeros(0, 1, 4, device=args.device),
        box_mask=torch.zeros(1, 0, device=args.device, dtype=torch.bool),
    )

    history = []
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        if args.loader_mode == "shuffled":
            worklist = list(samples)
            rng.shuffle(worklist)
        else:
            worklist = []
            for ds in dataset_samples:
                worklist.extend(ds["samples"])

        for item in worklist:
            image = load_image(item["image_path"], transform, args.device)
            target = load_mask(item["mask_path"], args.image_size, args.device)

            model.memory_prompt_builder.set_prompt_ids(
                torch.tensor([item["prompt_id"]], device=args.device, dtype=torch.long)
            )

            backbone_out = model.backbone.forward_image(image)
            backbone_out.update(
                model.backbone.forward_text([item["prompt"]], device=args.device)
            )
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
            ].unsqueeze(1)
            selected_logits = F.interpolate(
                selected_logits,
                size=(args.image_size, args.image_size),
                mode="bilinear",
                align_corners=False,
            )

            bce = F.binary_cross_entropy_with_logits(selected_logits, target)
            dice = dice_loss_from_logits(selected_logits, target)
            tokens = model.memory_prompt_builder.memory_tokens
            loss = (
                dice
                + 0.3 * bce
                + args.token_l2_weight * l2_regularization(tokens)
                + args.token_diversity_weight * diversity_regularization(tokens)
                + args.prompt_diversity_weight * inter_prompt_diversity(tokens)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        epoch_loss /= max(len(worklist), 1)
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {epoch_loss:.6f}")

    payload = {
        "memory_prompt_builder": model.memory_prompt_builder.state_dict(),
        "config": {
            "num_prompts": len(prompt_to_id),
            "tokens_per_prompt": args.tokens_per_prompt,
            "epochs": args.epochs,
            "lr": args.lr,
            "image_size": args.image_size,
            "token_l2_weight": args.token_l2_weight,
            "token_diversity_weight": args.token_diversity_weight,
            "prompt_diversity_weight": args.prompt_diversity_weight,
            "loader_mode": args.loader_mode,
            "prompt_to_id": prompt_to_id,
            "prompt_system_mode": args.prompt_system_mode,
            "prompt_topk_attrs": args.prompt_topk_attrs,
            "prompt_format": args.prompt_format,
            "prompt_separator": args.prompt_separator,
            "prompt_system": prompt_system.manifest(),
            "datasets": [
                {
                    "image_dir": str(d["image_dir"]),
                    "mask_dir": str(d["mask_dir"]),
                    "prompt": d["prompt"],
                    "prompt_raw": d["prompt_raw"],
                }
                for d in datasets
            ],
        },
        "history": history,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_path)
    print(f"Saved prompt token bank checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
