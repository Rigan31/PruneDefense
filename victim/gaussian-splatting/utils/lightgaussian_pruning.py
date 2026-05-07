import torch
import numpy as np
from gaussian_renderer import render


def prune_list(gaussians, scene, pipe, background):
    """
    Compute per-Gaussian importance scores by rendering all training views
    and accumulating the opacity-weighted contribution (Global Significance Score).

    This is the core of LightGaussian's pruning criterion:
    For each training view, we render and collect per-Gaussian:
      - gaussian_list: accumulated count of how many views each Gaussian contributes to
      - imp_list: accumulated opacity-weighted importance across all views

    The importance is computed inside the rasterizer as the alpha (opacity)
    contribution of each Gaussian during splatting. Since poison-splat's codebase
    uses the standard diff-gaussian-rasterization (not LightGaussian's modified one
    that returns per-Gaussian importance directly), we approximate importance using
    opacity and the visibility filter from the render output.

    Args:
        gaussians: GaussianModel instance
        scene: Scene instance with training cameras
        pipe: Pipeline parameters
        background: Background color tensor

    Returns:
        gaussian_list: [N] tensor, per-Gaussian view count
        imp_list: [N] tensor, per-Gaussian accumulated importance score
    """
    viewpoint_stack = scene.getTrainCameras().copy()

    gaussian_list = torch.zeros(gaussians.get_xyz.shape[0], device="cuda")
    imp_list = torch.zeros(gaussians.get_xyz.shape[0], device="cuda")

    for viewpoint_cam in viewpoint_stack:
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        visibility_filter = render_pkg["visibility_filter"]  # [N] bool
        radii = render_pkg["radii"]                          # [N] int

        # Accumulate view count for visible Gaussians
        gaussian_list[visibility_filter] += 1

        # Importance = opacity * (radii > 0 indicates on-screen contribution)
        # This approximates LightGaussian's per-Gaussian alpha accumulation
        # using opacity as a proxy for rendering contribution
        opacity = gaussians.get_opacity.squeeze(-1).detach()  # [N]
        radii_float = radii.float().detach()

        # Importance: opacity-weighted screen coverage for visible Gaussians
        imp = opacity * radii_float
        imp_list[visibility_filter] += imp[visibility_filter]

    return gaussian_list, imp_list


def calculate_v_imp_score(gaussians, imp_list, v_pow=0.1):
    """
    Calculate Volume-weighted Importance Score (v_imp_score) from LightGaussian.

    This combines:
      1. Gaussian volume (product of scales) - penalizes tiny Gaussians
      2. Importance from multi-view rendering (from prune_list)

    The formula is:
      volume = prod(scaling, dim=1)
      normalized_volume = (volume / kth_percentile_volume) ^ v_pow
      v_imp_score = normalized_volume * imp_list

    Gaussians that are both small in volume AND have low rendering importance
    get the lowest scores and are pruned first.

    This is particularly effective against Poison-Splat because:
    - Poison-induced Gaussians tend to be very small (high-frequency noise fitting)
    - They often have low cross-view importance (inconsistent contributions)

    Args:
        gaussians: GaussianModel instance
        imp_list: [N] tensor from prune_list
        v_pow: Power for volume normalization (default 0.1, controls volume influence)
               Lower v_pow = less volume influence, higher = more volume influence

    Returns:
        v_list: [N] tensor, combined volume-importance score for each Gaussian
    """
    volume = torch.prod(gaussians.get_scaling, dim=1)  # [N]

    # Find the 90th percentile volume as normalization reference
    index = int(len(volume) * 0.9)
    sorted_volume, _ = torch.sort(volume, descending=True, dim=0)
    kth_percent_largest = sorted_volume[index]

    # Normalize volume and raise to v_pow
    # Small v_pow (e.g., 0.1) means volume has moderate influence
    # This prevents volume from dominating the score
    v_list = torch.pow(volume / (kth_percent_largest + 1e-8), v_pow)

    # Combine with importance: score = volume_factor * rendering_importance
    v_list = v_list * imp_list

    return v_list


class LightGaussianPruningDefense:
    """
    Defense mechanism that integrates LightGaussian's pruning into the
    Poison-Splat training loop.

    Strategy:
    - At specified iterations (prune_iterations), compute Global Significance
      scores for all Gaussians using multi-view rendering.
    - Prune the bottom `prune_percent` of Gaussians ranked by the
      volume-weighted importance score (v_imp_score).
    - Apply prune_decay for subsequent pruning rounds (prune less aggressively
      over time as the model stabilizes).

    This is more principled than a hard Gaussian count cap because it
    selectively removes Gaussians that contribute least to reconstruction
    quality, which are also most likely to be poison-induced artifacts.

    Args:
        prune_iterations: List of iterations at which to prune
        prune_percent: Fraction of Gaussians to remove at each prune step (e.g., 0.5 = 50%)
        prune_decay: Multiplicative decay applied to prune_percent at each successive
                     prune step. E.g., with decay=0.8, prune_percent becomes
                     0.5, 0.4, 0.32, ... at successive prune iterations.
        v_pow: Power for volume normalization in v_imp_score (default 0.1)
        max_gaussians: Hard safety cap. If exceeded at a prune iteration,
                       forces pruning down to this limit regardless of score.
    """

    def __init__(
        self,
        prune_iterations=None,
        prune_percent=0.5,
        prune_decay=1.0,
        v_pow=0.1,
        max_gaussians=500000,
    ):
        if prune_iterations is None:
            # Default: prune at iterations 20000 and 25000
            # (after densification ends at 15000 in vanilla 3DGS)
            self.prune_iterations = [15500, 20000, 25000]
        else:
            self.prune_iterations = prune_iterations

        self.prune_percent = prune_percent
        self.prune_decay = prune_decay
        self.v_pow = v_pow
        self.max_gaussians = max_gaussians

    def should_prune(self, iteration):
        """Check if pruning should happen at this iteration."""
        return iteration in self.prune_iterations

    def prune(self, gaussians, scene, pipe, background, iteration):
        """
        Execute LightGaussian-style pruning at the current iteration.

        Args:
            gaussians: GaussianModel instance
            scene: Scene instance (needed to render all training views)
            pipe: Pipeline parameters
            background: Background color tensor
            iteration: Current training iteration

        Returns:
            n_pruned: Number of Gaussians pruned (0 if not a prune iteration)
        """
        if iteration not in self.prune_iterations:
            return 0

        n_before = gaussians.get_xyz.shape[0]
        i = self.prune_iterations.index(iteration)

        print(f"\n[DEFENSE] LightGaussian pruning at iteration {iteration}")
        print(f"[DEFENSE] Gaussians before pruning: {n_before}")

        # Step 1: Compute per-Gaussian importance via multi-view rendering
        gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)

        # Step 2: Compute volume-weighted importance score
        v_list = calculate_v_imp_score(gaussians, imp_list, self.v_pow)

        # Step 3: Determine pruning ratio with decay
        effective_prune_percent = (self.prune_decay ** i) * self.prune_percent

        # Step 4: Prune Gaussians with lowest v_imp_score
        # Sort scores ascending, mark bottom effective_prune_percent for removal
        n_to_prune = int(n_before * effective_prune_percent)

        if n_to_prune > 0:
            # Get threshold: Gaussians below this score get pruned
            sorted_scores, _ = torch.sort(v_list, descending=False)
            threshold = sorted_scores[min(n_to_prune, len(sorted_scores) - 1)]

            prune_mask = v_list <= threshold

            # Safety: never prune everything
            n_remaining = (~prune_mask).sum().item()
            if n_remaining < 100:
                print(f"[DEFENSE] WARNING: Would prune too many. Keeping top 100.")
                _, top_indices = torch.topk(v_list, 100)
                prune_mask = torch.ones(n_before, dtype=torch.bool, device="cuda")
                prune_mask[top_indices] = False

            gaussians.prune_points(prune_mask)

        n_after = gaussians.get_xyz.shape[0]
        n_pruned = n_before - n_after

        print(f"[DEFENSE] Effective prune ratio: {effective_prune_percent:.2%}")
        print(f"[DEFENSE] Gaussians pruned: {n_pruned}")
        print(f"[DEFENSE] Gaussians remaining: {n_after}")

        return n_pruned

    def enforce_hard_cap(self, gaussians, scene, pipe, background):
        """
        Emergency pruning if Gaussian count exceeds max_gaussians.
        Uses the same v_imp_score mechanism but prunes down to max_gaussians.

        This can be called at any iteration as a safety valve.
        """
        n_current = gaussians.get_xyz.shape[0]
        if n_current <= self.max_gaussians:
            return 0

        print(f"\n[DEFENSE] HARD CAP triggered: {n_current} > {self.max_gaussians}")

        gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
        v_list = calculate_v_imp_score(gaussians, imp_list, self.v_pow)

        # Keep only top max_gaussians by score
        n_to_keep = self.max_gaussians
        _, top_indices = torch.topk(v_list, n_to_keep)
        prune_mask = torch.ones(n_current, dtype=torch.bool, device="cuda")
        prune_mask[top_indices] = False

        gaussians.prune_points(prune_mask)

        n_after = gaussians.get_xyz.shape[0]
        print(f"[DEFENSE] Hard cap pruned to: {n_after}")

        return n_current - n_after