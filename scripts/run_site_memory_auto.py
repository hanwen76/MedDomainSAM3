#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end site pipeline: train free memory tokens on train split, "
            "build static memory from 5% val support, then run hybrid ablation on val query."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--site", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)

    parser.add_argument("--text-prompt", type=str, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)

    parser.add_argument("--val-support-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-size", type=int, default=1008)

    parser.add_argument("--free-memory-num-tokens", type=int, default=4)
    parser.add_argument("--free-epochs", type=int, default=5)
    parser.add_argument("--free-lr", type=float, default=1e-2)
    parser.add_argument("--free-limit", type=int, default=None)
    parser.add_argument("--token-l2-weight", type=float, default=1e-4)
    parser.add_argument("--token-diversity-weight", type=float, default=1e-3)

    parser.add_argument("--prototype-count", type=int, default=8)
    parser.add_argument("--prototype-grouping", type=str, default="global", choices=["global", "per_text"])
    parser.add_argument("--prototype-iters", type=int, default=15)

    parser.add_argument("--static-memory-topk", type=int, default=4)
    parser.add_argument("--static-memory-text-weight", type=float, default=1.0)
    parser.add_argument("--no-object-threshold", type=float, default=0.2)
    parser.add_argument("--memory-trigger-threshold", type=float, default=0.8)
    parser.add_argument("--hybrid-fusion", type=str, default="score_weighted", choices=["avg", "score_weighted"])
    parser.add_argument("--hybrid-score-temperature", type=float, default=0.2)

    parser.add_argument("--extensions", nargs="+", default=sorted(IMAGE_SUFFIXES))
    return parser.parse_args()


def _ext_set(extensions):
    return {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}


def collect_pairs(image_dir: Path, mask_dir: Path, extensions):
    ext_set = _ext_set(extensions)
    image_paths = {}
    for path in image_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in ext_set:
            image_paths[path.stem] = path

    pairs = []
    for stem, image_path in sorted(image_paths.items()):
        mask_path = None
        for ext in ext_set:
            candidate = mask_dir / f"{stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
        if mask_path is None:
            for candidate in mask_dir.rglob(f"{stem}.*"):
                if candidate.suffix.lower() in ext_set:
                    mask_path = candidate
                    break
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    return pairs


def resolve_split_dirs(site_dir: Path):
    train_image_dir = site_dir / "data_npy"
    train_mask_dir = site_dir / "label_npy"
    val_image_dir = site_dir / "val_data_npy"
    val_mask_dir = site_dir / "val_label_npy"

    if all(path.is_dir() for path in [train_image_dir, train_mask_dir, val_image_dir, val_mask_dir]):
        return train_image_dir, train_mask_dir, val_image_dir, val_mask_dir

    raise FileNotFoundError(
        "Expected directory layout: <dataset-root>/<site>/train/images, <dataset-root>/<site>/train/masks, "
        "<dataset-root>/<site>/val/images, <dataset-root>/<site>/val/masks"
    )


def compute_support_count(total: int, ratio: float):
    if total <= 0:
        return 0
    count = max(1, int(math.ceil(total * ratio)))
    if total > 1:
        count = min(count, total - 1)
    return count


def materialize_subset(pairs, out_dir: Path):
    image_out = out_dir / "images"
    mask_out = out_dir / "masks"
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    for image_path, mask_path in pairs:
        image_link = image_out / image_path.name
        mask_link = mask_out / mask_path.name

        if image_link.exists() or image_link.is_symlink():
            image_link.unlink()
        if mask_link.exists() or mask_link.is_symlink():
            mask_link.unlink()

        os.symlink(image_path.resolve(), image_link)
        os.symlink(mask_path.resolve(), mask_link)

    return image_out, mask_out


def run_cmd(cmd):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    site_dir = args.dataset_root / args.site
    if not site_dir.is_dir():
        raise FileNotFoundError(f"Site directory not found: {site_dir}")

    train_image_dir, train_mask_dir, val_image_dir, val_mask_dir = resolve_split_dirs(site_dir)

    train_pairs = collect_pairs(train_image_dir, train_mask_dir, args.extensions)
    val_pairs = collect_pairs(val_image_dir, val_mask_dir, args.extensions)

    if not train_pairs:
        raise RuntimeError(f"No train pairs found under: {train_image_dir} and {train_mask_dir}")
    if len(val_pairs) < 2:
        raise RuntimeError("Validation split needs at least 2 pairs to build support/query subsets.")

    rng = random.Random(args.seed)
    rng.shuffle(val_pairs)

    support_count = compute_support_count(len(val_pairs), args.val_support_ratio)
    support_pairs = val_pairs[:support_count]
    query_pairs = val_pairs[support_count:]

    if not query_pairs:
        raise RuntimeError("Validation query subset is empty after split. Increase val size or lower support ratio.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root or (PROJECT_ROOT / "outputs" / "site_memory_auto" / f"{args.site}_{timestamp}")
    output_root.mkdir(parents=True, exist_ok=True)

    work_dir = output_root / "work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    support_image_dir, support_mask_dir = materialize_subset(support_pairs, work_dir / "val_support")
    query_image_dir, query_mask_dir = materialize_subset(query_pairs, work_dir / "val_query")

    free_ckpt = output_root / "artifacts" / "free_memory_tokens.pt"
    static_bank = output_root / "artifacts" / "static_memory_bank.pt"
    eval_output = output_root / "eval"
    eval_output.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train_free_memory_tokens.py"),
        "--image-dir",
        str(train_image_dir),
        "--mask-dir",
        str(train_mask_dir),
        "--output-path",
        str(free_ckpt),
        "--checkpoint-path",
        str(args.checkpoint_path),
        "--device",
        args.device,
        "--image-size",
        str(args.image_size),
        "--num-tokens",
        str(args.free_memory_num_tokens),
        "--epochs",
        str(args.free_epochs),
        "--lr",
        str(args.free_lr),
        "--token-l2-weight",
        str(args.token_l2_weight),
        "--token-diversity-weight",
        str(args.token_diversity_weight),
    ]
    if args.free_limit is not None:
        train_cmd.extend(["--limit", str(args.free_limit)])
    if args.text_prompt is not None:
        train_cmd.extend(["--text-prompt", args.text_prompt])
    if args.metadata_json is not None:
        train_cmd.extend(["--metadata-json", str(args.metadata_json)])

    build_static_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "build_static_memory_bank.py"),
        "--image-dir",
        str(support_image_dir),
        "--mask-dir",
        str(support_mask_dir),
        "--output-path",
        str(static_bank),
        "--checkpoint-path",
        str(args.checkpoint_path),
        "--device",
        args.device,
        "--image-size",
        str(args.image_size),
        "--prototype-count",
        str(args.prototype_count),
        "--prototype-grouping",
        args.prototype_grouping,
        "--prototype-iters",
        str(args.prototype_iters),
    ]
    if args.text_prompt is not None:
        build_static_cmd.extend(["--default-text-prompt", args.text_prompt])
    if args.metadata_json is not None:
        build_static_cmd.extend(["--metadata-json", str(args.metadata_json)])

    eval_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "eval_memory_hybrid_ablation.py"),
        "--image-dir",
        str(query_image_dir),
        "--mask-dir",
        str(query_mask_dir),
        "--checkpoint-path",
        str(args.checkpoint_path),
        "--static-memory-bank-path",
        str(static_bank),
        "--free-memory-ckpt",
        str(free_ckpt),
        "--output-dir",
        str(eval_output),
        "--device",
        args.device,
        "--image-size",
        str(args.image_size),
        "--static-memory-topk",
        str(args.static_memory_topk),
        "--static-memory-text-weight",
        str(args.static_memory_text_weight),
        "--free-memory-num-tokens",
        str(args.free_memory_num_tokens),
        "--no-object-threshold",
        str(args.no_object_threshold),
        "--memory-trigger-threshold",
        str(args.memory_trigger_threshold),
        "--hybrid-fusion",
        args.hybrid_fusion,
        "--hybrid-score-temperature",
        str(args.hybrid_score_temperature),
    ]
    if args.text_prompt is not None:
        eval_cmd.extend(["--text-prompt", args.text_prompt])
    if args.metadata_json is not None:
        eval_cmd.extend(["--metadata-json", str(args.metadata_json)])

    run_cmd(train_cmd)
    run_cmd(build_static_cmd)
    run_cmd(eval_cmd)

    manifest = {
        "dataset_root": str(args.dataset_root),
        "site": args.site,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "val_support_pairs": len(support_pairs),
        "val_query_pairs": len(query_pairs),
        "val_support_ratio": args.val_support_ratio,
        "seed": args.seed,
        "paths": {
            "output_root": str(output_root),
            "free_memory_ckpt": str(free_ckpt),
            "static_memory_bank": str(static_bank),
            "eval_dir": str(eval_output),
            "eval_metrics_json": str(eval_output / "metrics.json"),
        },
    }
    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Manifest: {manifest_path}")
    print(f"Metrics:  {eval_output / 'metrics.json'}")


if __name__ == "__main__":
    main()
