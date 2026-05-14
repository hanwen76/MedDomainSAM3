#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model_builder import build_sam3_image_model
from prompt_system import PromptSystem
from train_free_memory_tokens import (
    batch_items,
    build_transform,
    collect_pairs,
    dice_loss_from_logits,
    load_image_batch,
    load_mask_batch,
    load_metadata,
    resolve_text,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the SAM3 task encoder using support samples from the train dataset."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--text-prompt", type=str, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--num-task-tokens", type=int, default=4)
    parser.add_argument("--support-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=1)
    parser.add_argument("--support-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-stem-prefix", type=str, default="")
    parser.add_argument("--image-stem-suffix", type=str, default="")
    parser.add_argument("--mask-stem-prefix", type=str, default="")
    parser.add_argument("--mask-stem-suffix", type=str, default="")
    parser.add_argument(
        "--pairing-mode",
        type=str,
        default="stem",
        choices=["stem", "sequential"],
    )
    parser.add_argument(
        "--prompt-system-mode",
        type=str,
        default="raw",
        choices=["raw", "canonical", "expanded"],
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
    parser.add_argument("--task-l2-weight", type=float, default=1e-5)
    return parser.parse_args()


def get_task_builder(model):
    builder = getattr(model, "memory_prompt_builder", None)
    task_builder = getattr(builder, "task_builder", builder)
    if task_builder is None or not hasattr(task_builder, "set_task_support"):
        raise RuntimeError("Model does not have a trainable task encoder builder")
    return task_builder


def sample_support_pairs(pairs, query_batch, support_size: int, rng: np.random.Generator):
    query_paths = {str(image_path) for image_path, _ in query_batch}
    candidates = [pair for pair in pairs if str(pair[0]) not in query_paths]
    if not candidates:
        candidates = list(pairs)
    replace = support_size > len(candidates)
    indices = rng.choice(len(candidates), size=support_size, replace=replace)
    return [candidates[int(index)] for index in indices]


@torch.no_grad()
def encode_support_pairs(model, pairs, transform, device: str, image_size: int, batch_size: int):
    task_embeddings = []
    memory_keys = []
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
        task_embeddings.append(torch.stack([foreground, context], dim=1))
        memory_keys.append(context)
    return torch.cat(task_embeddings, dim=0), torch.cat(memory_keys, dim=0)


def make_text_prompts(image_paths, metadata, default_prompt, prompt_system, args):
    prompts = []
    for image_path in image_paths:
        text = resolve_text(image_path.stem, metadata, default_prompt)
        text = prompt_system.transform_prompt(
            text,
            mode=args.prompt_system_mode,
            topk_attrs=args.prompt_topk_attrs,
            fmt=args.prompt_format,
            separator=args.prompt_separator,
        )
        prompts.append(text)
    return prompts


def task_l2_regularization(task_builder):
    penalty = None
    for param in task_builder.parameters():
        value = (param**2).mean()
        penalty = value if penalty is None else penalty + value
    if penalty is None:
        raise RuntimeError("Task builder has no trainable parameters")
    return penalty


def main():
    args = parse_args()
    pairs = collect_pairs(
        args.image_dir,
        args.mask_dir,
        image_stem_prefix=args.image_stem_prefix,
        image_stem_suffix=args.image_stem_suffix,
        mask_stem_prefix=args.mask_stem_prefix,
        mask_stem_suffix=args.mask_stem_suffix,
        pairing_mode=args.pairing_mode,
    )
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if len(pairs) < 2:
        raise RuntimeError("Task encoder training needs at least two image/mask pairs")

    rng = np.random.default_rng(args.seed)
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
        use_memory_prompt=True,
        memory_prompt_num_tokens=args.num_task_tokens,
    )
    model.eval()
    task_builder = get_task_builder(model)

    # Initialize support dimensions before constructing the optimizer.
    init_support = sample_support_pairs(pairs, pairs[:1], args.support_size, rng)
    task_embeddings, memory_keys = encode_support_pairs(
        model,
        init_support,
        transform,
        args.device,
        args.image_size,
        args.support_batch_size,
    )
    task_builder.set_task_support(task_embeddings, memory_keys)

    for param in model.parameters():
        param.requires_grad = False
    for param in task_builder.parameters():
        param.requires_grad = True
    task_builder.train()
    optimizer = torch.optim.AdamW(task_builder.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        order = rng.permutation(len(pairs))
        epoch_loss = 0.0
        step_count = 0
        ordered_pairs = [pairs[int(index)] for index in order]
        for query_batch in batch_items(ordered_pairs, args.query_batch_size):
            support_pairs = sample_support_pairs(
                pairs,
                query_batch,
                args.support_size,
                rng,
            )
            task_embeddings, memory_keys = encode_support_pairs(
                model,
                support_pairs,
                transform,
                args.device,
                args.image_size,
                args.support_batch_size,
            )
            task_builder.set_task_support(task_embeddings, memory_keys)

            image_paths = [image_path for image_path, _ in query_batch]
            mask_paths = [mask_path for _, mask_path in query_batch]
            images = load_image_batch(image_paths, transform, args.device)
            targets = load_mask_batch(mask_paths, args.image_size, args.device)
            text_prompts = make_text_prompts(
                image_paths,
                metadata,
                args.text_prompt,
                prompt_system,
                args,
            )
            batch_size = len(query_batch)
            backbone_out = model.backbone.forward_image(images)
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

            bce = F.binary_cross_entropy_with_logits(selected_logits, targets)
            dice = dice_loss_from_logits(selected_logits, targets)
            loss = dice + 0.3 * bce + args.task_l2_weight * task_l2_regularization(task_builder)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            step_count += 1

        mean_loss = epoch_loss / max(step_count, 1)
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {mean_loss:.6f}")

    payload = {
        "task_encoder": task_builder.state_dict(),
        "config": {
            "num_task_tokens": args.num_task_tokens,
            "support_size": args.support_size,
            "query_batch_size": args.query_batch_size,
            "support_batch_size": args.support_batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "image_size": args.image_size,
            "prompt_system_mode": args.prompt_system_mode,
            "seed": args.seed,
            "prompt_system": prompt_system.manifest(),
        },
        "history": history,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_path)
    print(f"Saved task encoder checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
