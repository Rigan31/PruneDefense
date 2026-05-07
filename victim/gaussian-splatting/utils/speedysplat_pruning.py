import torch
import numpy as np
from gaussian_renderer import render


class SpeedySplatPruningDefense:
    def __init__(
        self,
        soft_prune_percent=0.1,
        hard_prune_percent=0.3,
        hard_prune_iterations=None,
        score_accumulation_views=5,
        max_gaussians=500000,
    ):
        self.soft_prune_percent = soft_prune_percent
        self.hard_prune_percent = hard_prune_percent
        self.score_accumulation_views = score_accumulation_views
        self.max_gaussians = max_gaussians

        if hard_prune_iterations is None:
            self.hard_prune_iterations = [16000, 20000, 25000]
        else:
            self.hard_prune_iterations = hard_prune_iterations

        # Accumulators for gradient-based sensitivity score
        self._grad_mean_accum = None
        self._grad_scale_accum = None
        self._accum_count = 0

    def reset_accumulators(self, n_gaussians):
        """Reset gradient accumulators (call after pruning/densification changes count)."""
        self._grad_mean_accum = torch.zeros(n_gaussians, 3, device="cuda")
        self._grad_scale_accum = torch.zeros(n_gaussians, 3, device="cuda")
        self._accum_count = 0

    def accumulate_gradients(self, gaussians):
        n = gaussians.get_xyz.shape[0]

        # Initialize or resize accumulators if needed
        if self._grad_mean_accum is None or self._grad_mean_accum.shape[0] != n:
            self.reset_accumulators(n)

        # Accumulate squared gradients (Fisher diagonal approximation)
        if gaussians._xyz.grad is not None:
            grad_xyz = gaussians._xyz.grad.detach()
            self._grad_mean_accum += grad_xyz ** 2

        if gaussians._scaling.grad is not None:
            grad_scale = gaussians._scaling.grad.detach()
            self._grad_scale_accum += grad_scale ** 2

        self._accum_count += 1

    def compute_efficient_pruning_score(self, gaussians):
        eps = 1e-12

        if self._accum_count == 0 or self._grad_mean_accum is None:
            # Fallback: use opacity as score if no gradients accumulated yet
            return gaussians.get_opacity.squeeze(-1).detach()

        # Average the accumulated squared gradients
        avg_grad_mean_sq = self._grad_mean_accum / self._accum_count  # [N, 3]
        avg_grad_scale_sq = self._grad_scale_accum / self._accum_count  # [N, 3]

        # Log determinant of diagonal matrix = sum of log of diagonal entries
        # Concatenate mean and scale grads for the 6 spatial parameters
        log_score = (
            torch.sum(torch.log(avg_grad_mean_sq + eps), dim=1) +
            torch.sum(torch.log(avg_grad_scale_sq + eps), dim=1)
        )  # [N]

        return log_score

    def soft_prune(self, gaussians, iteration):
        n_before = gaussians.get_xyz.shape[0]

        if n_before < 100:
            self.reset_accumulators(n_before)
            return 0

        # Check if accumulators match current Gaussian count
        # (densification may have changed the count since last accumulation)
        if (self._grad_mean_accum is None or
            self._grad_mean_accum.shape[0] != n_before or
            self._accum_count == 0):
            # No valid scores available — reset and skip this round
            self.reset_accumulators(n_before)
            return 0

        scores = self.compute_efficient_pruning_score(gaussians)

        n_to_prune = int(n_before * self.soft_prune_percent)
        if n_to_prune < 1:
            self.reset_accumulators(n_before)
            return 0

        # Ensure we keep at least 100 Gaussians
        n_to_prune = min(n_to_prune, n_before - 100)

        # Get threshold: Gaussians below this score get pruned
        sorted_scores, _ = torch.sort(scores)
        threshold = sorted_scores[n_to_prune - 1]

        # NOTE: prune_points expects a VALID points mask (True = KEEP)
        # so we create keep_mask where True means the Gaussian survives
        keep_mask = scores > threshold

        # Safety check
        n_remaining = keep_mask.sum().item()
        if n_remaining < 100:
            self.reset_accumulators(n_before)
            return 0

        gaussians.prune_points(~keep_mask)

        # Reset accumulators after pruning changes Gaussian count
        self.reset_accumulators(gaussians.get_xyz.shape[0])

        n_pruned = n_before - gaussians.get_xyz.shape[0]
        return n_pruned

    def hard_prune(self, gaussians, scene, pipe, background, iteration):
        """
        Hard Pruning: Called at specified iterations after densification ends.

        Uses the gradient sensitivity scores already accumulated during normal
        training iterations (via accumulate_gradients called each step).
        No separate backward pass is needed — the training loop gradients
        provide a strong signal for which Gaussians are geometrically important.

        Prunes the bottom hard_prune_percent of Gaussians by sensitivity score.

        Args:
            gaussians: GaussianModel instance
            scene: Scene instance
            pipe: Pipeline parameters
            background: Background color tensor
            iteration: Current training iteration

        Returns:
            n_pruned: Number of Gaussians pruned
        """
        if iteration not in self.hard_prune_iterations:
            return 0

        n_before = gaussians.get_xyz.shape[0]
        print(f"\n[DEFENSE] SpeedySplat Hard Pruning at iteration {iteration}")
        print(f"[DEFENSE] Gaussians before: {n_before}")

        # Check we have valid accumulated gradients from training
        if (self._grad_mean_accum is None or
            self._grad_mean_accum.shape[0] != n_before or
            self._accum_count == 0):
            print(f"[DEFENSE] WARNING: No valid gradient accumulation. Skipping hard prune.")
            self.reset_accumulators(n_before)
            return 0

        print(f"[DEFENSE] Using {self._accum_count} accumulated gradient steps for scoring")

        # Compute scores from accumulated training gradients
        scores = self.compute_efficient_pruning_score(gaussians)

        # Prune bottom hard_prune_percent
        n_to_prune = int(n_before * self.hard_prune_percent)
        n_to_prune = min(n_to_prune, n_before - 100)

        if n_to_prune > 0:
            sorted_scores, _ = torch.sort(scores)
            threshold = sorted_scores[n_to_prune - 1]
            prune_mask = scores <= threshold  # True = remove

            n_remaining = (~prune_mask).sum().item()
            if n_remaining >= 100:
                gaussians.prune_points(prune_mask)

        # Reset accumulators for fresh accumulation going forward
        self.reset_accumulators(gaussians.get_xyz.shape[0])

        n_after = gaussians.get_xyz.shape[0]
        n_pruned = n_before - n_after

        print(f"[DEFENSE] Hard prune ratio: {self.hard_prune_percent:.0%}")
        print(f"[DEFENSE] Gaussians pruned: {n_pruned}")
        print(f"[DEFENSE] Gaussians remaining: {n_after}")

        return n_pruned

    def enforce_hard_cap(self, gaussians, scene, pipe, background):
        """
        Emergency pruning if Gaussian count exceeds max_gaussians.
        Uses the Efficient Pruning Score to keep only the top max_gaussians.
        """
        n_current = gaussians.get_xyz.shape[0]
        if n_current <= self.max_gaussians:
            return 0

        print(f"\n[DEFENSE] HARD CAP triggered: {n_current} > {self.max_gaussians}")

        scores = self.compute_efficient_pruning_score(gaussians)

        n_to_keep = self.max_gaussians
        _, top_indices = torch.topk(scores, n_to_keep)
        prune_mask = torch.ones(n_current, dtype=torch.bool, device="cuda")
        prune_mask[top_indices] = False

        gaussians.prune_points(prune_mask)
        self.reset_accumulators(gaussians.get_xyz.shape[0])

        n_after = gaussians.get_xyz.shape[0]
        print(f"[DEFENSE] Hard cap pruned to: {n_after}")

        return n_current - n_after