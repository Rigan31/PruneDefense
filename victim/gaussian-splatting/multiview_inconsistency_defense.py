#
# Multi-View Inconsistency Score (MVIS) Defense for 3D Gaussian Splatting
#
# This defense identifies and prunes Gaussians that contribute inconsistently
# across training views — a hallmark of adversarial artifacts from per-image
# poisoning attacks. Unlike PUP 3DGS which sums per-view sensitivities to find
# the LEAST important Gaussians, we compute the VARIANCE of per-view contributions
# to find Gaussians that serve view-specific adversarial signals.
#
# Usage:
#   Integrated into training loop (after densification steps):
#     python train_with_mvis_defense.py -s <data_path> -m <model_output_path> \
#         --mvis_prune_percent 0.1 --mvis_start_iter 15500 --mvis_interval 1000
#
#   Post-hoc on a trained model:
#     python multiview_inconsistency_defense.py \
#         --model_path <path_to_trained_model> \
#         --data_path <path_to_dataset> \
#         --prune_percent 0.1 --output_path <output_ply>
#

import os
import sys
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scene import Scene, GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from argparse import Namespace
from utils.loss_utils import l1_loss, ssim


def compute_per_gaussian_per_view_contribution(
    gaussians: GaussianModel,
    viewpoint_cameras: list,
    pipeline_params,
    background: torch.Tensor,
    contribution_type: str = "alpha_weighted_error",
) -> torch.Tensor:
    n_gaussians = gaussians.get_xyz.shape[0]
    n_views = len(viewpoint_cameras)
    
    # We'll accumulate per-Gaussian contributions across views
    contributions = torch.zeros(n_gaussians, n_views, device="cuda")
    
    for view_idx, viewpoint_cam in enumerate(tqdm(viewpoint_cameras, desc="Computing per-view contributions")):
        # Render the full scene
        render_pkg = render(viewpoint_cam, gaussians, pipeline_params, background)
        rendered_image = render_pkg["render"]  # [3, H, W]
        
        # Get ground truth
        gt_image = viewpoint_cam.original_image.cuda()  # [3, H, W]
        
        # Per-pixel error map
        pixel_error = (rendered_image - gt_image).abs().mean(dim=0)  # [H, W]
        
        if contribution_type == "alpha_weighted_error":
            # Use the radii to determine which Gaussians are visible in this view
            # radii > 0 means the Gaussian is visible
            radii = render_pkg["radii"]  # [N_gaussians]
            visibility_mask = radii > 0  # [N_gaussians]
            
            # For visible Gaussians, we need their per-pixel alpha contribution
            # The render function gives us per-pixel accumulated alpha, but not
            # per-Gaussian. We approximate using a per-Gaussian rendering approach.
            #
            # Approach: For each visible Gaussian, estimate its contribution as
            # the product of its opacity and the mean error in its projected region.
            # This is an efficient approximation that avoids N separate renders.
            
            # Get 2D projected means from the render package
            screenspace_points = render_pkg["viewspace_points"]  # [N, 2] or similar
            
            # Get opacity values
            opacities = gaussians.get_opacity.squeeze()  # [N]
            
            H, W = pixel_error.shape
            
            # For each visible Gaussian, sample the error at its projected location
            if screenspace_points.shape[-1] >= 2:
                # Normalize screen coordinates to [-1, 1] for grid_sample
                pts_2d = screenspace_points.detach()[:, :2]  # [N, 2]
                
                # Normalize to [-1, 1]
                pts_normalized = torch.zeros_like(pts_2d)
                pts_normalized[:, 0] = (pts_2d[:, 0] / W) * 2 - 1  # x
                pts_normalized[:, 1] = (pts_2d[:, 1] / H) * 2 - 1  # y
                
                # Sample error map at Gaussian projected locations
                # grid_sample expects [N, H_out, W_out, 2] grid
                error_map = pixel_error.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                grid = pts_normalized.unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]
                
                # Clamp to valid range
                grid = grid.clamp(-1, 1)
                
                sampled_errors = torch.nn.functional.grid_sample(
                    error_map, grid, mode='bilinear', padding_mode='border', align_corners=True
                )  # [1, 1, N, 1]
                sampled_errors = sampled_errors.squeeze()  # [N]
                
                # Contribution = opacity * sampled_error * visibility
                contribution = opacities.detach() * sampled_errors * visibility_mask.float()
                contributions[:, view_idx] = contribution
            
        elif contribution_type == "visibility_weighted_loss":
            # Simpler approach: just use visibility (radii > 0) weighted by
            # the overall view reconstruction error
            radii = render_pkg["radii"]
            visibility_mask = (radii > 0).float()
            
            view_loss = pixel_error.mean()
            contributions[:, view_idx] = visibility_mask * view_loss
    
    return contributions


def compute_multiview_inconsistency_score(
    contributions: torch.Tensor,
    method: str = "coefficient_of_variation",
) -> torch.Tensor:
    n_gaussians, n_views = contributions.shape
    
    # Only consider views where the Gaussian is visible (contribution > 0)
    # For Gaussians visible in very few views, that itself is suspicious
    visibility_count = (contributions > 0).sum(dim=1).float()  # [N]
    
    if method == "coefficient_of_variation":
        mean_contrib = contributions.mean(dim=1)  # [N]
        std_contrib = contributions.std(dim=1)    # [N]
        
        # Coefficient of variation: std / mean
        # High CV = inconsistent contributions across views
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        cv = std_contrib / (mean_contrib + eps)
        
        # Gaussians visible in very few views get a bonus penalty
        # (if visible in < 10% of views, that's suspicious)
        few_view_penalty = torch.where(
            visibility_count < max(n_views * 0.1, 2),
            torch.ones_like(cv) * 2.0,  # high penalty
            torch.zeros_like(cv)
        )
        
        scores = cv + few_view_penalty
    
    elif method == "max_ratio":
        mean_contrib = contributions.mean(dim=1)
        max_contrib = contributions.max(dim=1).values
        eps = 1e-8
        scores = max_contrib / (mean_contrib + eps)
    
    elif method == "entropy":
        # Normalize contributions to a probability distribution per Gaussian
        eps = 1e-8
        contrib_sum = contributions.sum(dim=1, keepdim=True) + eps
        prob = contributions / contrib_sum  # [N, V]
        
        # Shannon entropy: uniform distribution = high entropy = consistent
        # Peaked distribution = low entropy = inconsistent
        log_prob = torch.log(prob + eps)
        entropy = -(prob * log_prob).sum(dim=1)  # [N]
        
        # Invert: we want HIGH score = inconsistent
        max_entropy = np.log(n_views)
        scores = max_entropy - entropy
    
    elif method == "gini":
        # Gini coefficient: 0 = perfect equality, 1 = perfect inequality
        sorted_contrib, _ = contributions.sort(dim=1)
        n = n_views
        index = torch.arange(1, n + 1, device=contributions.device).float()
        
        contrib_sum = contributions.sum(dim=1, keepdim=True) + 1e-8
        scores = (2 * (index * sorted_contrib).sum(dim=1)) / (n * contrib_sum.squeeze()) - (n + 1) / n
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return scores


def prune_gaussians_by_score(
    gaussians: GaussianModel,
    scores: torch.Tensor,
    prune_percent: float = 0.1,
    score_threshold: float = None,
):
    n_gaussians = scores.shape[0]
    
    if score_threshold is not None:
        prune_mask = scores > score_threshold
    else:
        # Prune the top prune_percent by score
        n_prune = int(n_gaussians * prune_percent)
        if n_prune == 0:
            print("Warning: prune_percent too low, no Gaussians to prune")
            return torch.zeros(n_gaussians, dtype=torch.bool, device=scores.device)
        
        threshold = torch.topk(scores, n_prune).values[-1]
        prune_mask = scores >= threshold
    
    n_pruned = prune_mask.sum().item()
    print(f"MVIS Defense: Pruning {n_pruned}/{n_gaussians} Gaussians "
          f"({100 * n_pruned / n_gaussians:.2f}%)")
    
    # Keep mask is the inverse
    keep_mask = ~prune_mask
    
    # Use the GaussianModel's built-in pruning via prune_points
    gaussians.prune_points(prune_mask)
    
    return prune_mask


def mvis_prune_in_training_loop(
    gaussians: GaussianModel,
    scene: Scene,
    pipeline_params,
    background: torch.Tensor,
    prune_percent: float = 0.1,
    scoring_method: str = "coefficient_of_variation",
    max_views_for_scoring: int = 50,
):
    print(f"\n{'='*60}")
    print(f"MVIS Defense: Computing Multi-View Inconsistency Scores")
    print(f"  Gaussians before: {gaussians.get_xyz.shape[0]}")
    print(f"  Scoring method: {scoring_method}")
    print(f"  Prune percent: {prune_percent * 100:.1f}%")
    print(f"{'='*60}")
    
    # Get training cameras
    train_cameras = scene.getTrainCameras()
    
    # Subsample views for efficiency if needed
    if len(train_cameras) > max_views_for_scoring:
        indices = np.random.choice(len(train_cameras), max_views_for_scoring, replace=False)
        selected_cameras = [train_cameras[i] for i in indices]
    else:
        selected_cameras = list(train_cameras)
    
    print(f"  Using {len(selected_cameras)} views for scoring")
    
    # Step 1: Compute per-Gaussian per-view contributions
    with torch.no_grad():
        contributions = compute_per_gaussian_per_view_contribution(
            gaussians, selected_cameras, pipeline_params, background,
            contribution_type="alpha_weighted_error"
        )
    
    # Step 2: Compute inconsistency scores
    scores = compute_multiview_inconsistency_score(contributions, method=scoring_method)
    
    # Log score statistics
    print(f"  Score stats: min={scores.min():.4f}, max={scores.max():.4f}, "
          f"mean={scores.mean():.4f}, std={scores.std():.4f}")
    
    # Step 3: Prune
    prune_mask = prune_gaussians_by_score(gaussians, scores, prune_percent=prune_percent)
    
    print(f"  Gaussians after: {gaussians.get_xyz.shape[0]}")
    print(f"{'='*60}\n")
    
    return scores, prune_mask



def posthoc_mvis_prune(args):
    """
    Load a pretrained 3DGS model, compute MVIS scores, prune, and save.
    """
    from plyfile import PlyData, PlyElement
    
    print("=" * 60)
    print("MVIS Post-Hoc Pruning Defense")
    print("=" * 60)
    
    # Setup params
    parser_model = ModelParams(ArgumentParser(description=""))
    parser_pipe = PipelineParams(ArgumentParser(description=""))
    
    # Build namespace for scene loading
    model_args = Namespace(
        source_path=args.data_path,
        model_path=args.model_path,
        images="images",
        resolution=1,
        white_background=False,
        data_device="cuda",
        eval=True,
        sh_degree=3,
    )
    
    pipe_args = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )
    
    background = torch.tensor(
        [1, 1, 1] if args.white_background else [0, 0, 0],
        dtype=torch.float32, device="cuda"
    )
    
    # Load Gaussians
    gaussians = GaussianModel(3)
    
    # Load scene to get cameras
    scene = Scene(model_args, gaussians, shuffle=False)
    
    # Load the PLY file (must be after Scene creation to avoid overwrite)
    ply_path = args.ply_path if args.ply_path else os.path.join(
        args.model_path, "point_cloud", "iteration_30000", "point_cloud.ply"
    )
    print(f"Loading PLY from: {ply_path}")
    gaussians.load_ply(ply_path)
    print(f"Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    
    # Get training cameras
    train_cameras = scene.getTrainCameras()
    print(f"Using {len(train_cameras)} training views")
    
    # Compute contributions
    print("\nComputing per-view contributions...")
    with torch.no_grad():
        contributions = compute_per_gaussian_per_view_contribution(
            gaussians, list(train_cameras), pipe_args, background,
            contribution_type="alpha_weighted_error"
        )
    
    # Compute scores using all methods for analysis
    methods = ["coefficient_of_variation", "max_ratio", "entropy", "gini"]
    all_scores = {}
    for method in methods:
        scores = compute_multiview_inconsistency_score(contributions, method=method)
        all_scores[method] = scores.cpu().numpy()
        print(f"  {method}: min={scores.min():.4f}, max={scores.max():.4f}, "
              f"mean={scores.mean():.4f}, std={scores.std():.4f}")
    
    # Use the selected method for pruning
    prune_scores = compute_multiview_inconsistency_score(
        contributions, method=args.scoring_method
    )
    
    # Prune
    prune_mask = prune_gaussians_by_score(
        gaussians, prune_scores, prune_percent=args.prune_percent
    )
    
    # Save pruned model
    output_path = args.output_path if args.output_path else os.path.join(
        args.model_path, "point_cloud", "mvis_pruned", "point_cloud.ply"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gaussians.save_ply(output_path)
    print(f"\nSaved pruned model to: {output_path}")
    
    # Save scores for analysis
    scores_path = output_path.replace(".ply", "_scores.npz")
    np.savez(
        scores_path,
        **all_scores,
        prune_mask=prune_mask.cpu().numpy(),
        contributions=contributions.cpu().numpy(),
    )
    print(f"Saved scores to: {scores_path}")
    
    return all_scores, prune_mask


if __name__ == "__main__":
    parser = ArgumentParser(description="MVIS Post-Hoc Pruning Defense")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained 3DGS model directory")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training dataset")
    parser.add_argument("--ply_path", type=str, default=None,
                        help="Path to specific PLY file (default: auto-detect)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path for pruned PLY")
    parser.add_argument("--prune_percent", type=float, default=0.1,
                        help="Fraction of Gaussians to prune (default: 0.1)")
    parser.add_argument("--scoring_method", type=str, default="coefficient_of_variation",
                        choices=["coefficient_of_variation", "max_ratio", "entropy", "gini"],
                        help="Scoring method for inconsistency")
    parser.add_argument("--white_background", action="store_true",
                        help="Use white background")
    
    args = parser.parse_args()
    posthoc_mvis_prune(args)