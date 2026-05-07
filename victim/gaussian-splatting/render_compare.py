#!/usr/bin/env python3
"""
render_compare.py

Renders ALL test-split views from a trained 3DGS model and saves:
  - renders/       rendered images
  - gt/            ground truth images
  - comparison/    side-by-side (GT | Render)
  - metrics_summary.txt  per-image PSNR / SSIM / L1

Usage:
  python render_compare.py \
      -s <DATA_PATH> \
      -m <MODEL_PATH> \
      --ply <path/to/victim_model.ply>
"""

import os
import sys
import torch
import torchvision
import numpy as np
from argparse import ArgumentParser
from PIL import Image

from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, get_combined_args


def make_side_by_side(gt_tensor, render_tensor):
    separator_width = 4
    _, H, W = gt_tensor.shape
    separator = torch.ones(3, H, separator_width, device=gt_tensor.device)
    return torch.cat([gt_tensor, separator, render_tensor], dim=2)


def render_and_compare(dataset, pipeline, ply_path=None, iteration=30000,
                       num_images=None, output_dir=None):
    """
    Render ALL test cameras (scene.getTestCameras()).
    Falls back to training cameras if no test cameras are found.

    num_images=None means use all available cameras.
    """

    if output_dir is None:
        output_dir = os.path.join(dataset.model_path, "render_comparison")
    os.makedirs(output_dir, exist_ok=True)

    renders_dir    = os.path.join(output_dir, "renders")
    gt_dir         = os.path.join(output_dir, "gt")
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(renders_dir,    exist_ok=True)
    os.makedirs(gt_dir,         exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    # ── Load Gaussians ─────────────────────────────────────────────────────
    gaussians = GaussianModel(dataset.sh_degree)

    if ply_path and os.path.exists(ply_path):
        print(f"Loading cameras from scene: {dataset.model_path}")
        scene = Scene(dataset, gaussians, shuffle=False)
        print(f"Loading trained model from PLY: {ply_path}")
        gaussians.load_ply(ply_path)
    else:
        print(f"Loading model from scene: {dataset.model_path}, "
              f"iteration {iteration}")
        scene = Scene(dataset, gaussians,
                      load_iteration=iteration, shuffle=False)

    print(f"Model loaded: {gaussians.get_xyz.shape[0]:,} Gaussians")

    # ── Choose cameras -- prefer test split ────────────────────────────────
    test_cameras  = scene.getTestCameras()
    train_cameras = scene.getTrainCameras()

    if len(test_cameras) > 0:
        cameras    = test_cameras
        split_name = "test"
    else:
        cameras    = train_cameras
        split_name = "train (no test cameras found)"

    total_cams = len(cameras)
    print(f"Using {split_name} cameras: {total_cams} total")

    # Use all cameras unless num_images is explicitly set
    if num_images is None or num_images >= total_cams:
        indices    = list(range(total_cams))
        n_render   = total_cams
    else:
        indices  = np.linspace(0, total_cams - 1, num_images, dtype=int).tolist()
        n_render = num_images

    print(f"Rendering {n_render} / {total_cams} views")

    # ── Background ─────────────────────────────────────────────────────────
    bg_color   = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # ── Render loop ────────────────────────────────────────────────────────
    results = []

    print(f"\n  Rendering {n_render} comparison images (from PLY)")
    print(f"  Data:  {dataset.source_path}")
    print(f"  PLY:   {ply_path}")
    print("=" * 50)

    for i, idx in enumerate(indices):
        cam      = cameras[idx]
        gt_image = cam.original_image[0:3, :, :].cuda()

        with torch.no_grad():
            render_pkg = render(cam, gaussians, pipeline, background)
            rendered   = torch.clamp(render_pkg["render"], 0.0, 1.0)

        l1_val   = l1_loss(rendered, gt_image).mean().item()
        psnr_val = psnr(rendered, gt_image).mean().item()
        ssim_val = ssim(rendered, gt_image).item()

        results.append({
            "index":      int(idx),
            "image_name": getattr(cam, "image_name", f"cam_{idx}"),
            "l1":         l1_val,
            "psnr":       psnr_val,
            "ssim":       ssim_val,
        })

        # Save images using image_name so filenames match the dataset
        img_name = getattr(cam, "image_name", f"{i:04d}_view{idx:04d}")
        torchvision.utils.save_image(
            rendered,
            os.path.join(renders_dir, f"{img_name}.png")
        )
        torchvision.utils.save_image(
            gt_image,
            os.path.join(gt_dir, f"{img_name}.png")
        )
        comparison = make_side_by_side(gt_image, rendered)
        torchvision.utils.save_image(
            comparison,
            os.path.join(comparison_dir, f"{img_name}.png")
        )

        print(f"  [{i+1:>4}/{n_render}] {img_name}: "
              f"PSNR={psnr_val:.2f}  SSIM={ssim_val:.4f}  L1={l1_val:.4f}")

    # ── Summary ────────────────────────────────────────────────────────────
    mean_psnr = np.mean([r["psnr"] for r in results])
    mean_ssim = np.mean([r["ssim"] for r in results])
    mean_l1   = np.mean([r["l1"]   for r in results])

    summary_path = os.path.join(output_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Render Comparison Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model     : {dataset.model_path}\n")
        if ply_path:
            f.write(f"PLY       : {ply_path}\n")
        f.write(f"Split     : {split_name}\n")
        f.write(f"Gaussians : {gaussians.get_xyz.shape[0]:,}\n")
        f.write(f"Views     : {n_render} / {total_cams}\n\n")

        f.write(f"{'View':>6s}  {'Name':<24s}  "
                f"{'PSNR':>8s}  {'SSIM':>8s}  {'L1':>8s}\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"{r['index']:>6d}  {r['image_name']:<24s}  "
                    f"{r['psnr']:>8.2f}  {r['ssim']:>8.4f}  "
                    f"{r['l1']:>8.4f}\n")

        f.write("\nMean metrics:\n")
        f.write(f"  PSNR : {mean_psnr:.4f}\n")
        f.write(f"  SSIM : {mean_ssim:.4f}\n")
        f.write(f"  L1   : {mean_l1:.4f}\n")

    print(f"\n{'=' * 50}")
    print(f"  Split used : {split_name}")
    print(f"  Views      : {n_render}")
    print(f"  Mean PSNR  : {mean_psnr:.4f}")
    print(f"  Mean SSIM  : {mean_ssim:.4f}")
    print(f"  Mean L1    : {mean_l1:.4f}")
    print(f"{'=' * 50}")
    print(f"\nDone! Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render and compare with ground truth")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--iteration",  type=int,  default=30000)
    parser.add_argument("--ply",        type=str,  default=None,
                        help="Path to victim_model.ply")
    parser.add_argument("--num_images", type=int,  default=None,
                        help="Number of images (default: all test cameras)")
    parser.add_argument("--output_dir", type=str,  default=None)
    parser.add_argument("--quiet",      action="store_true")

    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)

    render_and_compare(
        dataset    = lp.extract(args),
        pipeline   = pp.extract(args),
        ply_path   = args.ply,
        iteration  = args.iteration,
        num_images = args.num_images,
        output_dir = args.output_dir,
    )