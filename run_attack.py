import argparse
import json
import os

import torch

from m_ifgsm import (
    NERF_CLASSES,
    load_clip_model,
    build_text_features,
    mirror_object_structure,
    process_object_split,
)

ALL_SPLITS = ["train", "test", "val"]


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="M-IFGSM adversarial attack on nerf_synthetic dataset",
    )
    parser.add_argument("--dataset_root", required=True,
                        help="Root of nerf_synthetic dataset")
    parser.add_argument("--output_dir",   required=True,
                        help="Where adversarial images are saved")
    parser.add_argument("--epsilon",      type=float, required=True,
                        help="Perturbation magnitude (pixel units 0-255)")
    parser.add_argument("--classes",      nargs="+", default=NERF_CLASSES,
                        help="Object classes to process (default: all 8)")
    parser.add_argument("--num_iter",     type=int,   default=100,
                        help="Max gradient iterations")
    parser.add_argument("--loss_thresh",  type=float, default=20.0,
                        help="Early-stopping loss threshold tau")
    parser.add_argument("--mask_dir",     default=None,
                        help="Optional: root of pre-computed SAM masks")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-iteration progress")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("  M-IFGSM Adversarial Attack -- nerf_synthetic")
    print("=" * 65)
    print(f"  Dataset  : {args.dataset_root}")
    print(f"  Output   : {args.output_dir}")
    print(f"  Device   : {args.device}")
    print(f"  Classes  : {args.classes}")
    print(f"  Epsilon  : {args.epsilon}")
    print(f"  N iter   : {args.num_iter}")
    print(f"  Tau      : {args.loss_thresh}")
    print("=" * 65)

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading CLIP ViT-B/16 ...")
    clip_model, _ = load_clip_model(args.device)
    text_features = build_text_features(NERF_CLASSES, clip_model, args.device)
    print(f"  Text features : {text_features.shape}")

    attack_log = {}

    for obj in args.classes:
        obj_dir = os.path.join(args.dataset_root, obj)
        if not os.path.isdir(obj_dir):
            print(f"\n  [SKIP] '{obj}' not found at {obj_dir}")
            continue

        out_object_dir = os.path.join(args.output_dir, obj)

        print(f"\n  [{obj}] Copying static files ...")
        mirror_object_structure(obj_dir, out_object_dir)

        attack_log[obj] = {}

        for split in ALL_SPLITS:
            split_src = os.path.join(obj_dir, split)
            json_src  = os.path.join(obj_dir, f"transforms_{split}.json")

            if not os.path.isdir(split_src) or not os.path.exists(json_src):
                print(f"  [{obj}/{split}] not found -- skipping")
                continue

            results = process_object_split(
                object_dir     = obj_dir,
                object_name    = obj,
                out_object_dir = out_object_dir,
                clip_model     = clip_model,
                text_features  = text_features,
                class_names    = NERF_CLASSES,
                split          = split,
                epsilon        = args.epsilon,
                num_iterations = args.num_iter,
                loss_threshold = args.loss_thresh,
                mask_dir       = args.mask_dir,
                device         = args.device,
                verbose        = args.verbose,
            )
            attack_log[obj][split] = results
            print(f"  [OK] {obj}/{split} -- {len(results)} images -> "
                  f"{out_object_dir}/{split}/")

    # Save attack log
    log_path = os.path.join(args.output_dir, "attack_log.json")
    with open(log_path, "w") as f:
        json.dump(attack_log, f, indent=2)
    print(f"\n  Attack log -> {log_path}")

    # Print output structure
    print("\n  Output dataset structure:")
    for obj in args.classes:
        out_obj = os.path.join(args.output_dir, obj)
        if not os.path.isdir(out_obj):
            continue
        print(f"    {obj}/")
        for item in sorted(os.listdir(out_obj)):
            item_path = os.path.join(out_obj, item)
            if os.path.isdir(item_path):
                n = len([f for f in os.listdir(item_path)
                          if f.lower().endswith((".png", ".jpg"))])
                print(f"      {item}/  ({n} images)")
            else:
                print(f"      {item}  ({os.path.getsize(item_path)//1024} KB)")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()