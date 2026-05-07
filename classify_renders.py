"""

    # Classify against the 8 nerf_synthetic classes
    python classify_renders.py \
        --image_dir  /path/to/rendered_chairs \
        --true_label chair

    # Classify against custom class list
    python classify_renders.py \
        --image_dir  /path/to/rendered_chairs \
        --true_label chair \
        --classes    chair table sofa bench stool desk cabinet shelf

    # Save results to JSON
    python classify_renders.py \
        --image_dir  /path/to/rendered_chairs \
        --true_label chair \
        --output     results.json
"""

import os
import json
import argparse
from pathlib import Path

import torch
import clip
import numpy as np
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NERF_CLASSES = [
    "chair", "drums", "ficus", "hotdog",
    "lego", "materials", "mic", "ship",
]

CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SIZE = (224, 224)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# CLIP setup
# ---------------------------------------------------------------------------

def load_clip(device):
    model, _ = clip.load("ViT-B/16", device=device)
    model.eval()
    return model


def build_text_features(class_names, model, device):
    prompts = [f"a photo of a {c}" for c in class_names]
    tokens  = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
    return feats / feats.norm(dim=-1, keepdim=True)   # [C, D]


# ---------------------------------------------------------------------------
# Single-image classification
# ---------------------------------------------------------------------------

_norm      = transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
_to_tensor = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


def classify_image(img_pil, model, text_features, class_names, device, top_k=3):
    img_tensor = _to_tensor(img_pil.convert("RGB")).unsqueeze(0).to(device)
    img_norm   = _norm(img_tensor)

    with torch.no_grad():
        img_feats = model.encode_image(img_norm)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()
        logits = logit_scale * (img_feats @ text_features.T)   # [1, C]
        probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu()  # [C]

    topk_probs, topk_idx = torch.topk(probs, min(top_k, len(class_names)))

    return {
        "top1_name":   class_names[topk_idx[0].item()],
        "top1_conf":   topk_probs[0].item(),
        "topk_names":  [class_names[i.item()] for i in topk_idx],
        "topk_confs":  topk_probs.tolist(),
        "all_probs":   {c: probs[i].item() for i, c in enumerate(class_names)},
    }

def classify_folder(
    image_dir:   str,
    true_label:  str,
    class_names: list,
    device:      str,
    top_k:       int = 3,
    verbose:     bool = True,
) -> list:
    model         = load_clip(device)
    text_features = build_text_features(class_names, model, device)
    true_idx      = class_names.index(true_label)

    # Gather image files, sorted for reproducibility
    files = sorted([
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in IMAGE_EXTS
    ])

    if not files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    if verbose:
        print(f"\n  Classifying {len(files)} images  |  "
              f"true label: \"{true_label}\"  |  top-k: 1, {top_k}")
        print()
        col = max(len(f) for f in files) + 2
        hdr = (f"  {'#':<4} {'Filename':<{col}} "
               f"{'True conf':>10}  {'Top-1':>7}  {f'Top-{top_k}':>7}  "
               f"Top-1 prediction")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2 + 20))

    results = []
    for i, fname in enumerate(files):
        img_path = os.path.join(image_dir, fname)
        img_pil  = Image.open(img_path)

        res = classify_image(img_pil, model, text_features, class_names, device, top_k)

        true_conf = res["all_probs"][true_label]
        top1_hit  = res["top1_name"] == true_label
        topk_hit  = true_label in res["topk_names"]

        row = {
            "filename":   fname,
            "true_label": true_label,
            "true_conf":  round(true_conf, 4),
            "top1_hit":   top1_hit,
            "topk_hit":   topk_hit,
            "top1_name":  res["top1_name"],
            "top1_conf":  round(res["top1_conf"], 4),
            "topk_names": res["topk_names"],
            "topk_confs": [round(c, 4) for c in res["topk_confs"]],
            "all_probs":  {k: round(v, 4) for k, v in res["all_probs"].items()},
        }
        results.append(row)

        if verbose:
            hit1 = "YES" if top1_hit else "NO "
            hitk = "YES" if topk_hit else "NO "
            pred = f"{res['top1_name']} ({res['top1_conf']:.4f})"
            print(f"  {i+1:<4} {fname:<{col}} "
                  f"{true_conf:>10.4f}  {hit1:>7}  {hitk:>7}  {pred}")

    return results


def print_summary(results, top_k=3):
    true_label   = results[0]["true_label"]
    n            = len(results)
    top1_hits    = sum(r["top1_hit"] for r in results)
    topk_hits    = sum(r["topk_hit"] for r in results)
    avg_true_conf = np.mean([r["true_conf"] for r in results])
    avg_top1_conf = np.mean([r["top1_conf"] for r in results])

    print()
    print("  Summary")
    print("  " + "-" * 52)
    print(f"  True label                  : {true_label}")
    print(f"  Total images                : {n}")
    print(f"  Avg confidence (true class) : {avg_true_conf:.4f}")
    print(f"  Avg confidence (top-1 pred) : {avg_top1_conf:.4f}")
    print(f"  Top-1 accuracy              : {top1_hits} / {n}"
          f"  ({100*top1_hits/n:.1f}%)")
    print(f"  Top-{top_k} accuracy"
          f"              : {topk_hits} / {n}"
          f"  ({100*topk_hits/n:.1f}%)")
    print()

    # Per-class breakdown: what did the model predict when it was wrong?
    wrong = [r for r in results if not r["top1_hit"]]
    if wrong:
        from collections import Counter
        wrong_preds = Counter(r["top1_name"] for r in wrong)
        print(f"  Misclassified as:")
        for cls, cnt in wrong_preds.most_common():
            print(f"    {cls:<20} {cnt} time(s)")
        print()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify rendered images with CLIP and report accuracy metrics"
    )
    parser.add_argument("--image_dir",  required=True,
                        help="Folder containing the rendered PNG/JPG images")
    parser.add_argument("--true_label", required=True,
                        help="Ground-truth class name, e.g. 'chair'")
    parser.add_argument("--classes",    nargs="+", default=NERF_CLASSES,
                        help="Full list of candidate class names for zero-shot CLIP")
    parser.add_argument("--top_k",      type=int,  default=3,
                        help="k for Top-k accuracy (default: 3)")
    parser.add_argument("--output",     default=None,
                        help="Optional path to save results as JSON")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.true_label not in args.classes:
        raise ValueError(
            f"true_label '{args.true_label}' not in class list: {args.classes}"
        )

    results = classify_folder(
        image_dir   = args.image_dir,
        true_label  = args.true_label,
        class_names = args.classes,
        device      = args.device,
        top_k       = args.top_k,
        verbose     = True,
    )

    print_summary(results, top_k=args.top_k)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved -> {args.output}")


if __name__ == "__main__":
    main()