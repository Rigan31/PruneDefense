import torch
from typing import Optional, List, Dict


class SmartPruningDefense:
    def __init__(
        self,
        max_gaussians: int = 500_000,
        soft_budget: int = 300_000,
        pruning_interval: int = 500,
        pruning_start_iter: int = 1000,
        growth_rate_threshold: float = 2.0,
        aggressive_prune_ratio: float = 0.5,
        normal_prune_ratio: float = 0.1,
        score_type: str = "gradient",
        verbose: bool = True,
    ):
        self.max_gaussians = max_gaussians
        self.soft_budget = min(soft_budget, max_gaussians)
        self.pruning_interval = pruning_interval
        self.pruning_start_iter = pruning_start_iter
        self.growth_rate_threshold = growth_rate_threshold
        self.aggressive_prune_ratio = aggressive_prune_ratio
        self.normal_prune_ratio = normal_prune_ratio
        self.score_type = score_type
        self.verbose = verbose

        # Internal state
        self._prev_count: Optional[int] = None
        self._acc_grad_xyz: Optional[torch.Tensor] = None
        self._acc_grad_scale: Optional[torch.Tensor] = None
        self._acc_steps: int = 0
        self._count_history: List[int] = []
        self._prune_history: List[Dict] = []
        self._attack_warnings: int = 0

    def log(self, msg: str):
        if self.verbose:
            print(f"[SMART-DEFENSE] {msg}")

    # ------------------------------------------------------------------ #
    #  Gradient accumulation (call every iter after loss.backward())      #
    # ------------------------------------------------------------------ #
    def accumulate_gradients(self, gaussians) -> None:
        """
        Accumulate |grad(xyz)| and |grad(scaling)| for importance scoring.
        Call AFTER loss.backward() and BEFORE optimizer.step().
        """
        xyz_grad = gaussians._xyz.grad
        if xyz_grad is None:
            return

        g_xyz = xyz_grad.detach().abs()
        scale_grad = gaussians._scaling.grad
        g_scale = scale_grad.detach().abs() if scale_grad is not None \
                  else torch.zeros_like(g_xyz)

        n = g_xyz.shape[0]

        # Reset if count changed (densification happened)
        if self._acc_grad_xyz is None or self._acc_grad_xyz.shape[0] != n:
            self._acc_grad_xyz = g_xyz.clone()
            self._acc_grad_scale = g_scale.clone()
            self._acc_steps = 1
        else:
            self._acc_grad_xyz += g_xyz
            self._acc_grad_scale += g_scale
            self._acc_steps += 1

    # ------------------------------------------------------------------ #
    #  Importance scoring                                                 #
    # ------------------------------------------------------------------ #
    def compute_importance_score(self, gaussians) -> torch.Tensor:
        """Compute per-Gaussian importance. Higher = more important = keep."""
        if self.score_type == "gradient":
            return self._score_gradient(gaussians)
        elif self.score_type == "opacity_volume":
            return self._score_opacity_volume(gaussians)
        elif self.score_type == "hybrid":
            return self._score_hybrid(gaussians)
        else:
            raise ValueError(f"Unknown score_type: {self.score_type}")

    def _score_gradient(self, gaussians) -> torch.Tensor:
        """
        Gradient-based score (approximates PUP 3D-GS Fisher sensitivity).
        score_i = opacity_i * (||avg_grad_xyz_i|| + ||avg_grad_scale_i||)
        """
        n_current = gaussians.get_xyz.shape[0]
        opacity = gaussians.get_opacity.detach().squeeze(-1)  # [n_current]

        if (self._acc_grad_xyz is not None and
                self._acc_steps > 0 and
                self._acc_grad_xyz.shape[0] == n_current):
            # Sizes match: use accumulated gradient score
            avg_xyz = self._acc_grad_xyz / self._acc_steps
            avg_scale = self._acc_grad_scale / self._acc_steps
            sensitivity = torch.norm(avg_xyz, dim=-1) + torch.norm(avg_scale, dim=-1)
            return opacity * sensitivity
        else:
            # Size mismatch (densification changed count) or no grads yet:
            # fall back to opacity * volume and reset accumulators
            if (self._acc_grad_xyz is not None and
                    self._acc_grad_xyz.shape[0] != n_current):
                self._acc_grad_xyz = None
                self._acc_grad_scale = None
                self._acc_steps = 0
            return self._score_opacity_volume(gaussians)

    def _score_opacity_volume(self, gaussians) -> torch.Tensor:
        """Simple opacity * volume fallback."""
        opacity = gaussians.get_opacity.detach().squeeze(-1)
        scales = gaussians.get_scaling.detach()
        volume = torch.prod(scales, dim=-1)
        return opacity * volume

    def _score_hybrid(self, gaussians) -> torch.Tensor:
        """70% gradient + 30% opacity_volume, both normalized."""
        gs = self._score_gradient(gaussians)
        vs = self._score_opacity_volume(gaussians)
        gn = gs / (gs.max() + 1e-8)
        vn = vs / (vs.max() + 1e-8)
        return 0.7 * gn + 0.3 * vn

    # ------------------------------------------------------------------ #
    #  Attack detection                                                   #
    # ------------------------------------------------------------------ #
    def detect_attack(self, current_count: int) -> bool:
        self._count_history.append(current_count)
        if self._prev_count is None or self._prev_count == 0:
            self._prev_count = current_count
            return False
        rate = current_count / self._prev_count
        self._prev_count = current_count
        if rate > self.growth_rate_threshold:
            self._attack_warnings += 1
            self.log(f"ATTACK WARNING #{self._attack_warnings}: "
                     f"growth={rate:.2f}x  count={current_count}")
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Main pruning entry point                                           #
    # ------------------------------------------------------------------ #
    def should_prune(self, iteration: int, count: int) -> bool:
        if iteration < self.pruning_start_iter:
            return False
        if count > self.max_gaussians:
            return True
        if iteration % self.pruning_interval == 0:
            return True
        return False

    def prune(self, gaussians, iteration: int) -> int:
        """
        Main entry point. Call every iteration after densification.
        Returns number of Gaussians removed (0 if nothing happened).
        """
        count = gaussians.get_xyz.shape[0]
        if not self.should_prune(iteration, count):
            return 0

        attack = self.detect_attack(count)

        # Decide how much to prune
        if count > self.max_gaussians:
            n_remove = count - self.soft_budget
            ratio = min(n_remove / count, 0.8)
            reason = "BUDGET"
        elif attack:
            ratio = self.aggressive_prune_ratio
            reason = "ATTACK"
        else:
            ratio = self.normal_prune_ratio
            reason = "PERIODIC"

        if ratio <= 0:
            return 0

        ratio = min(max(ratio, 0.0), 0.9)

        # Score and prune
        scores = self.compute_importance_score(gaussians)
        threshold = torch.quantile(scores, ratio)
        mask = scores <= threshold

        gaussians.prune_points(mask)

        # Reset accumulators
        self._acc_grad_xyz = None
        self._acc_grad_scale = None
        self._acc_steps = 0

        removed = mask.sum().item()
        remaining = gaussians.get_xyz.shape[0]
        self._prune_history.append({
            "iter": iteration, "reason": reason,
            "removed": removed, "remaining": remaining,
        })
        self.log(f"Iter {iteration} [{reason}]: pruned {removed}, "
                 f"remaining {remaining}")
        return removed

    # ------------------------------------------------------------------ #
    #  Summary                                                            #
    # ------------------------------------------------------------------ #
    def print_summary(self):
        self.log("=" * 60)
        self.log("DEFENSE SUMMARY")
        self.log(f"  Prune events:      {len(self._prune_history)}")
        self.log(f"  Attack warnings:   {self._attack_warnings}")
        peak = max(self._count_history) if self._count_history else 0
        final = self._count_history[-1] if self._count_history else 0
        self.log(f"  Peak count:        {peak}")
        self.log(f"  Final count:       {final}")
        self.log("=" * 60)