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

from sam3.model_builder import build_tracker
from prompt_system import PromptSystem


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a static medical memory bank for SAM3."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--default-text-prompt", type=str, default=None)
    parser.add_argument(
        "--prompt-system-mode",
        type=str,
        default="raw",
        choices=["raw", "canonical", "expanded"],
        help="How to normalize prompts before memory bank construction.",
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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--prototype-count",
        type=int,
        default=0,
        help="If > 0, aggregate case memories into this many prototypes.",
    )
    parser.add_argument(
        "--prototype-grouping",
        type=str,
        default="global",
        choices=["global", "per_text"],
        help="How to group memories before prototype aggregation.",
    )
    parser.add_argument(
        "--prototype-iters",
        type=int,
        default=15,
        help="Number of refinement iterations for cosine k-means style prototype aggregation.",
    )
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
    mask_tensor = F.interpolate(
        mask_tensor,
        size=(image_size, image_size),
        mode="nearest",
    )
    mask_tensor = (mask_tensor > 0.5).float().to(device)
    return mask_tensor


def load_tracker(checkpoint_path: Path | None, device: str, enable_text_encoder: bool):
    tracker = build_tracker(
        apply_temporal_disambiguation=False,
        with_backbone=True,
        use_static_memory=False,
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
        print("Warning: building static memory bank without a tracker checkpoint.")

    tracker = tracker.to(device)
    tracker.eval()
    return tracker


@torch.inference_mode()
def encode_pair(tracker, image_tensor, mask_tensor, text_prompt=None):
    backbone_out = tracker.forward_image(image_tensor)
    _, current_vision_feats, _, feat_sizes = tracker._prepare_backbone_features(backbone_out)

    batch_size = current_vision_feats[-1].size(1)
    channels = tracker.hidden_dim
    height, width = feat_sizes[-1]
    pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(batch_size, channels, height, width)

    mask_for_mem = mask_tensor
    if tracker.sigmoid_scale_for_mem_enc != 1.0:
        mask_for_mem = mask_for_mem * tracker.sigmoid_scale_for_mem_enc
    if tracker.sigmoid_bias_for_mem_enc != 0.0:
        mask_for_mem = mask_for_mem + tracker.sigmoid_bias_for_mem_enc

    maskmem_out = tracker.maskmem_backbone(
        pix_feat,
        mask_for_mem,
        skip_mask_sigmoid=True,
    )
    memory_features = maskmem_out["vision_features"].detach().cpu()
    memory_pos_enc = maskmem_out["vision_pos_enc"][-1].detach().cpu()
    memory_keys = F.adaptive_avg_pool2d(memory_features, output_size=1).flatten(1)
    mask_for_task = F.interpolate(
        mask_tensor.float(),
        size=pix_feat.shape[-2:],
        mode="nearest",
    )
    mask_sum = mask_for_task.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    foreground_token = (pix_feat * mask_for_task).sum(dim=(2, 3), keepdim=True) / mask_sum
    context_token = pix_feat.mean(dim=(2, 3), keepdim=True)
    task_embeddings = torch.stack(
        [
            foreground_token.flatten(1).squeeze(0).detach().cpu(),
            context_token.flatten(1).squeeze(0).detach().cpu(),
        ],
        dim=0,
    )
    memory_text_keys = None
    if text_prompt is not None and getattr(tracker.backbone, "language_backbone", None) is not None:
        text_outputs = tracker.backbone.forward_text([text_prompt], device=image_tensor.device)
        language_features = text_outputs["language_features"]
        memory_text_keys = language_features.mean(dim=0).detach().cpu()
    return memory_features, memory_pos_enc, memory_keys, memory_text_keys, task_embeddings


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def _cosine_kmeans_indices(keys: torch.Tensor, k: int, num_iters: int) -> torch.Tensor:
    num_items = keys.shape[0]
    if k >= num_items:
        return torch.arange(num_items)

    keys = _normalize_rows(keys)
    centers = [0]
    sim = torch.matmul(keys, keys[0:1].T).squeeze(1)
    min_dist = 1.0 - sim
    for _ in range(1, k):
        next_idx = torch.argmax(min_dist).item()
        centers.append(next_idx)
        sim = torch.matmul(keys, keys[next_idx : next_idx + 1].T).squeeze(1)
        min_dist = torch.minimum(min_dist, 1.0 - sim)

    centers = keys[torch.tensor(centers)]
    for _ in range(num_iters):
        sim = torch.matmul(keys, centers.T)
        assign = sim.argmax(dim=1)
        new_centers = []
        for cluster_idx in range(k):
            members = keys[assign == cluster_idx]
            if members.numel() == 0:
                new_centers.append(centers[cluster_idx])
            else:
                new_centers.append(_normalize_rows(members.mean(dim=0, keepdim=True))[0])
        centers = torch.stack(new_centers, dim=0)
    sim = torch.matmul(keys, centers.T)
    return sim.argmax(dim=1)


def aggregate_to_prototypes(
    memory_features: torch.Tensor,
    memory_pos_enc: torch.Tensor,
    memory_keys: torch.Tensor,
    memory_text_keys: torch.Tensor | None,
    task_embeddings: torch.Tensor | None,
    metadata: list[dict],
    prototype_count: int,
    prototype_grouping: str,
    prototype_iters: int,
):
    if prototype_count <= 0 or memory_features.shape[0] <= prototype_count:
        return memory_features, memory_pos_enc, memory_keys, memory_text_keys, task_embeddings, metadata

    if prototype_grouping == "per_text":
        groups = {}
        for idx, item in enumerate(metadata):
            group_name = item.get("text_prompt") or "__default__"
            groups.setdefault(group_name, []).append(idx)
    else:
        groups = {"__global__": list(range(len(metadata)))}

    proto_features = []
    proto_pos_enc = []
    proto_keys = []
    proto_text_keys = []
    proto_task_embeddings = []
    proto_metadata = []

    for group_name, group_indices in groups.items():
        group_features = memory_features[group_indices]
        group_pos_enc = memory_pos_enc[group_indices]
        group_keys = memory_keys[group_indices]
        group_text_keys = (
            None if memory_text_keys is None else memory_text_keys[group_indices]
        )
        group_task_embeddings = (
            None if task_embeddings is None else task_embeddings[group_indices]
        )

        group_k = min(prototype_count, len(group_indices))
        if group_k == len(group_indices):
            assignments = torch.arange(len(group_indices))
        elif group_k == 1:
            assignments = torch.zeros(len(group_indices), dtype=torch.long)
        else:
            assignments = _cosine_kmeans_indices(
                group_keys,
                k=group_k,
                num_iters=prototype_iters,
            )

        for cluster_idx in range(group_k):
            member_mask = assignments == cluster_idx
            member_ids_local = torch.nonzero(member_mask, as_tuple=False).flatten()
            member_ids_global = [group_indices[i] for i in member_ids_local.tolist()]
            cluster_features = group_features[member_mask]
            cluster_pos_enc = group_pos_enc[member_mask]
            cluster_keys = group_keys[member_mask]

            proto_features.append(cluster_features.mean(dim=0, keepdim=True))
            proto_pos_enc.append(cluster_pos_enc.mean(dim=0, keepdim=True))
            proto_keys.append(_normalize_rows(cluster_keys.mean(dim=0, keepdim=True)))
            if group_text_keys is not None:
                proto_text_keys.append(group_text_keys[member_mask].mean(dim=0, keepdim=True))
            if group_task_embeddings is not None:
                proto_task_embeddings.append(
                    group_task_embeddings[member_mask].mean(dim=0, keepdim=True)
                )
            proto_metadata.append(
                {
                    "prototype_group": group_name,
                    "prototype_index": cluster_idx,
                    "num_members": len(member_ids_global),
                    "member_indices": member_ids_global,
                    "text_prompt": None if group_name == "__global__" else group_name,
                }
            )

    return (
        torch.cat(proto_features, dim=0),
        torch.cat(proto_pos_enc, dim=0),
        torch.cat(proto_keys, dim=0),
        torch.cat(proto_text_keys, dim=0) if proto_text_keys else None,
        torch.cat(proto_task_embeddings, dim=0) if proto_task_embeddings else None,
        proto_metadata,
    )


def main():
    args = parse_args()
    pairs, missing_masks = collect_pairs(args.image_dir, args.mask_dir, args.extensions)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    if not pairs:
        raise RuntimeError("No image/mask pairs found. Check filenames and directories.")

    metadata_map = load_metadata(args.metadata_json)
    enable_text_encoder = args.metadata_json is not None or args.default_text_prompt is not None
    tracker = load_tracker(args.checkpoint_path, args.device, enable_text_encoder=enable_text_encoder)
    image_transform = build_image_transform(args.image_size)
    prompt_system = PromptSystem.from_paths(
        attributes_json=args.attributes_json,
        aliases_json=args.aliases_json,
    )

    memory_features = []
    memory_pos_enc = []
    memory_keys = []
    memory_text_keys = []
    task_embeddings = []
    metadata = []

    for idx, (image_path, mask_path) in enumerate(pairs, start=1):
        image_tensor = load_image_tensor(image_path, image_transform, args.device)
        mask_tensor = load_mask_tensor(mask_path, args.image_size, args.device)
        text_prompt = metadata_map.get(image_path.stem, args.default_text_prompt)
        if text_prompt is not None:
            text_prompt = prompt_system.transform_prompt(
                text_prompt,
                mode=args.prompt_system_mode,
                topk_attrs=args.prompt_topk_attrs,
                fmt=args.prompt_format,
                separator=args.prompt_separator,
            )
        feats, pos_enc, keys, text_keys, task_tokens = encode_pair(
            tracker,
            image_tensor,
            mask_tensor,
            text_prompt=text_prompt,
        )
        memory_features.append(feats)
        memory_pos_enc.append(pos_enc)
        memory_keys.append(keys)
        if text_keys is not None:
            memory_text_keys.append(text_keys)
        task_embeddings.append(task_tokens)
        metadata.append(
            {
                "index": idx - 1,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "text_prompt": text_prompt,
            }
        )
        if idx % 10 == 0 or idx == len(pairs):
            print(f"Encoded {idx}/{len(pairs)} cases")

    payload = {
        "memory_features": torch.cat(memory_features, dim=0),
        "memory_pos_enc": torch.cat(memory_pos_enc, dim=0),
        "memory_keys": torch.cat(memory_keys, dim=0),
        "memory_text_keys": (
            torch.cat(memory_text_keys, dim=0)
            if len(memory_text_keys) == len(metadata)
            else None
        ),
        "task_embeddings": torch.stack(task_embeddings, dim=0),
        "metadata": metadata,
        "config": {
            "image_size": args.image_size,
            "device": args.device,
            "num_entries": len(metadata),
        },
    }

    (
        payload["memory_features"],
        payload["memory_pos_enc"],
        payload["memory_keys"],
        payload["memory_text_keys"],
        payload["task_embeddings"],
        payload["metadata"],
    ) = aggregate_to_prototypes(
        memory_features=payload["memory_features"],
        memory_pos_enc=payload["memory_pos_enc"],
        memory_keys=payload["memory_keys"],
        memory_text_keys=payload["memory_text_keys"],
        task_embeddings=payload["task_embeddings"],
        metadata=payload["metadata"],
        prototype_count=args.prototype_count,
        prototype_grouping=args.prototype_grouping,
        prototype_iters=args.prototype_iters,
    )
    payload["config"].update(
        {
            "prototype_count": args.prototype_count,
            "prototype_grouping": args.prototype_grouping,
            "prototype_iters": args.prototype_iters,
            "num_entries_after_aggregation": int(payload["memory_features"].shape[0]),
        }
    )
    if args.prototype_count > 0:
        print(
            "Aggregated memories into "
            f"{payload['memory_features'].shape[0]} prototypes "
            f"using {args.prototype_grouping} grouping"
        )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_path)
    print(f"Saved static memory bank to {args.output_path}")

    if missing_masks:
        missing_path = args.output_path.with_suffix(".missing_masks.json")
        missing_path.write_text(json.dumps(missing_masks, indent=2))
        print(f"Saved stems with missing masks to {missing_path}")


if __name__ == "__main__":
    main()
