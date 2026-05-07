import math
import numpy as np
import torch
import random

from gaussian_renderer import render


class MVPIDefense:
    """
    Multi-View Photometric Inconsistency Defense with quadrant-stratified
    view sampling.

    Parameters
    ----------
    n_views        : total number of training views sampled per scoring pass
                     (2 per quadrant x 4 quadrants = 8 by default)
    prune_ratio    : fraction of Gaussians to prune per call (top-k by score)
    apply_from     : iteration to start applying the defense (warmup)
    apply_interval : how often to run (every N iterations)
    score_mode     : "variance" -- variance of per-view contributions
                     "max_min"  -- max minus min contribution across views
    verbose        : print statistics each time it runs
    """

    def __init__(
        self,
        n_views:        int   = 8,
        prune_ratio:    float = 0.05,
        apply_from:     int   = 1000,
        apply_interval: int   = 500,
        score_mode:     str   = "variance",
        verbose:        bool  = True,
    ):
        assert score_mode in ("variance", "max_min"), \
            "score_mode must be 'variance' or 'max_min'"

        self.n_views        = n_views
        self.prune_ratio    = prune_ratio
        self.apply_from     = apply_from
        self.apply_interval = apply_interval
        self.score_mode     = score_mode
        self.verbose        = verbose
        self._call_count    = 0

        # Cached quadrant assignment of training cameras. Built on first call.
        self._cam_quadrants = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def step(self, gaussians, scene, pipe, background, iteration):
        """
        Compute MVPI scores and prune the most inconsistent Gaussians.
        """
        if iteration < self.apply_from:
            return
        if (iteration - self.apply_from) % self.apply_interval != 0:
            return

        self._call_count += 1
        N = gaussians.get_xyz.shape[0]

        if self.verbose:
            print(f"\n  [MVPI Defense | iter={iteration} | call=#{self._call_count}]"
                  f"  N={N:,}  n_views={self.n_views}"
                  f"  prune_ratio={self.prune_ratio}")

        scores = self._compute_scores(gaussians, scene, pipe, background)
        self._prune_by_score(gaussians, scores)

    # ------------------------------------------------------------------
    # Quadrant assignment (stratified view sampling)
    # ------------------------------------------------------------------

    def _build_quadrants(self, cameras):
        """
        Assign each training camera to one of 4 azimuth quadrants around
        the scene center. Azimuth is measured in the XZ plane (treating Y
        as up, matching the 3DGS camera convention).

        Returns a list of 4 lists of cameras: [Q0, Q1, Q2, Q3].
        """
        positions = []
        for cam in cameras:
            c = cam.camera_center.detach().cpu().numpy()
            positions.append(c)
        positions = np.stack(positions, axis=0)  # [V, 3]
        center = positions.mean(axis=0)

        quadrants = [[], [], [], []]
        for cam, pos in zip(cameras, positions):
            dx = pos[0] - center[0]
            dz = pos[2] - center[2]
            angle = math.atan2(dz, dx)                    # in [-pi, pi]
            angle_deg = (math.degrees(angle) + 360) % 360  # in [0, 360)
            q = int(angle_deg // 90)                      # 0..3
            quadrants[q].append(cam)

        if self.verbose:
            print(f"  [MVPI] Camera quadrant assignment: "
                  f"Q0={len(quadrants[0])}  Q1={len(quadrants[1])}  "
                  f"Q2={len(quadrants[2])}  Q3={len(quadrants[3])}")
        return quadrants

    def _quadrant_of(self, cam):
        """Return which quadrant a camera belongs to (for debug printing)."""
        if self._cam_quadrants is None:
            return -1
        for q, cams_q in enumerate(self._cam_quadrants):
            if cam in cams_q:
                return q
        return -1

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _compute_scores(self, gaussians, scene, pipe, background):
        """
        For each Gaussian, compute its photometric inconsistency score
        across views sampled stratified over 4 azimuth quadrants.

        Returns
        -------
        scores : torch.Tensor shape (N,) on CUDA, higher = more suspicious
        """
        N       = gaussians.get_xyz.shape[0]
        device  = gaussians.get_xyz.device
        cameras = scene.getTrainCameras()

        # ---- Build or rebuild quadrant cache if needed ----
        total_cached = 0
        if self._cam_quadrants is not None:
            total_cached = sum(len(q) for q in self._cam_quadrants)
        if self._cam_quadrants is None or total_cached != len(cameras):
            self._cam_quadrants = self._build_quadrants(cameras)
        # ---------------------------------------------------

        # Pick (n_views / 4) views from each of the 4 quadrants.
        per_quadrant = max(1, self.n_views // 4)  # 2 when n_views == 8
        sampled = []
        for q in range(4):
            cams_q = self._cam_quadrants[q]
            if len(cams_q) == 0:
                continue
            k = min(per_quadrant, len(cams_q))
            sampled.extend(random.sample(cams_q, k))

        # If any quadrant was short, top up from the remaining pool.
        if len(sampled) < self.n_views:
            remaining = [c for c in cameras if c not in sampled]
            deficit = min(self.n_views - len(sampled), len(remaining))
            if deficit > 0:
                sampled.extend(random.sample(remaining, deficit))

        n_sample = len(sampled)
        per_view = []

        with torch.no_grad():
            for cam in sampled:
                gt_image = cam.original_image[0:3, :, :].to(device)  # (3, H, W)

                render_pkg        = render(cam, gaussians, pipe, background)
                rendered          = render_pkg["render"]               # (3, H, W)
                visibility_filter = render_pkg["visibility_filter"]    # (N,) bool

                H, W = gt_image.shape[1], gt_image.shape[2]

                # Per-pixel L1 error map
                l1_map = (rendered - gt_image).abs().mean(dim=0)       # (H, W)

                # Opacity of each Gaussian in probability space
                opacity_sigmoid = torch.sigmoid(
                    gaussians._opacity.squeeze(1)
                )                                                      # (N,)

                contrib = torch.zeros(N, device=device)
                vis_idx = torch.where(visibility_filter)[0]            # (M,)

                if len(vis_idx) == 0:
                    per_view.append(contrib)
                    continue

                # ---- Project visible Gaussians to screen space ----
                xyz_vis = gaussians.get_xyz[vis_idx]                   # (M, 3)

                R = torch.tensor(cam.R, dtype=torch.float32, device=device)
                T = torch.tensor(cam.T, dtype=torch.float32, device=device)

                # World -> camera coordinates
                xyz_cam = xyz_vis @ R.T + T                            # (M, 3)

                # Clamp depth to avoid division by zero
                z = xyz_cam[:, 2].clamp(min=1e-6)

                tan_fovx = torch.tan(torch.tensor(
                    cam.FoVx / 2, dtype=torch.float32, device=device))
                tan_fovy = torch.tan(torch.tensor(
                    cam.FoVy / 2, dtype=torch.float32, device=device))

                # Project to pixel coordinates
                px = (
                    (xyz_cam[:, 0] / z / tan_fovx) * (W / 2) + W / 2
                ).long().clamp(0, W - 1)

                py = (
                    (xyz_cam[:, 1] / z / tan_fovy) * (H / 2) + H / 2
                ).long().clamp(0, H - 1)

                # Sample per-pixel L1 error at each Gaussian's screen position
                local_error = l1_map[py, px]                           # (M,)

                # contribution(i, v) = opacity(i) * local_L1_error(i, v)
                contrib[vis_idx] = opacity_sigmoid[vis_idx] * local_error

                per_view.append(contrib)

        # Stack: (n_sample, N)
        stacked = torch.stack(per_view, dim=0)

        # Inconsistency score across views
        if self.score_mode == "variance":
            scores = stacked.var(dim=0)
        else:
            scores = stacked.max(dim=0).values - stacked.min(dim=0).values

        # ---- Debug printout: per-view contributions + quadrant of each view ----
        if self.verbose:
            topk_idx = torch.topk(scores, k=min(3, N)).indices.tolist()
            rand_idx = random.sample(range(N), min(2, N))
            show_idx = topk_idx + rand_idx

            # Label each column with the quadrant its view came from
            view_quadrants = [self._quadrant_of(cam) for cam in sampled]
            q_header = "  ".join(f"  Q{q}   " for q in view_quadrants)

            print(f"\n    [MVPI per-view contributions over {n_sample} views "
                  f"(stratified across 4 quadrants)]")
            print(f"    {'Gaussian':<14}{'score':<14}{q_header}")
            for i in show_idx:
                vals = stacked[:, i].tolist()
                vals_str = "  ".join(f"{v:7.4f}" for v in vals)
                tag = "(top)" if i in topk_idx else "(rand)"
                print(f"    {i:<8}{tag:<6}{scores[i].item():<14.5f}{vals_str}")
            print(f"    Score stats: mean={scores.mean().item():.5f}  "
                  f"std={scores.std().item():.5f}  "
                  f"max={scores.max().item():.5f}")
        # ------------------------------------------------------------------------

        return scores

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _prune_by_score(self, gaussians, scores):
        """
        Set the opacity of the top prune_ratio fraction of Gaussians
        (by MVPI score) to a very low value so they are removed by the
        next standard densify_and_prune call.
        """
        N       = scores.shape[0]
        n_prune = max(1, int(N * self.prune_ratio))

        _, top_idx = torch.topk(scores, k=n_prune, largest=True, sorted=False)

        if self.verbose:
            top_scores = scores[top_idx]
            all_mean   = scores.mean().item()
            all_std    = scores.std().item()
            print(f"    Score stats:  mean={all_mean:.5f}  std={all_std:.5f}")
            print(f"    Pruning {n_prune:,} / {N:,} Gaussians"
                  f"  (top score={top_scores.max().item():.5f})")

        # sigmoid(-10) ~ 4.5e-5 ~ 0 -- below min_opacity threshold
        with torch.no_grad():
            gaussians._opacity[top_idx] = -10.0

    # ------------------------------------------------------------------
    # Cumulative scorer (optional, unchanged from original)
    # ------------------------------------------------------------------

    def get_cumulative_scorer(self):
        """
        Returns a CumulativeMVPIScorer that accumulates per-Gaussian scores
        across multiple calls for a smoother signal.
        """
        return CumulativeMVPIScorer(n_views=self.n_views)

    def prune_from_scorer(self, gaussians, scorer):
        """Prune using accumulated scores from a CumulativeMVPIScorer."""
        scores = scorer.get_scores(gaussians.get_xyz.shape[0])
        if scores is not None:
            self._prune_by_score(gaussians, scores)


class CumulativeMVPIScorer:
    """
    Accumulates per-Gaussian MVPI scores across training iterations using
    an exponential moving average. Handles Gaussian count changes after
    densification by resetting when N changes.
    """

    def __init__(self, n_views=4, decay=0.9):
        self.n_views  = n_views
        self.decay    = decay
        self._scores  = None
        self._last_N  = None

    def update(self, gaussians, scene, pipe, background):
        """Compute one MVPI snapshot and update the running average."""
        N       = gaussians.get_xyz.shape[0]
        device  = gaussians.get_xyz.device
        cameras = scene.getTrainCameras()

        n_sample = min(self.n_views, len(cameras))
        sampled  = random.sample(list(cameras), n_sample)

        per_view = []
        with torch.no_grad():
            for cam in sampled:
                gt_image  = cam.original_image[0:3, :, :].to(device)
                render_pkg = render(cam, gaussians, pipe, background)
                rendered  = render_pkg["render"]
                vis       = render_pkg["visibility_filter"]
                H, W      = gt_image.shape[1], gt_image.shape[2]

                l1_map  = (rendered - gt_image).abs().mean(dim=0)
                opacity = torch.sigmoid(gaussians._opacity.squeeze(1))
                contrib = torch.zeros(N, device=device)
                vis_idx = torch.where(vis)[0]

                if len(vis_idx) > 0:
                    xyz_vis = gaussians.get_xyz[vis_idx]
                    R = torch.tensor(cam.R, dtype=torch.float32, device=device)
                    T = torch.tensor(cam.T, dtype=torch.float32, device=device)
                    xyz_cam = xyz_vis @ R.T + T
                    z = xyz_cam[:, 2].clamp(min=1e-6)
                    tan_fovx = torch.tan(torch.tensor(
                        cam.FoVx / 2, dtype=torch.float32, device=device))
                    tan_fovy = torch.tan(torch.tensor(
                        cam.FoVy / 2, dtype=torch.float32, device=device))
                    px = ((xyz_cam[:, 0] / z / tan_fovx) * (W / 2) + W / 2
                          ).long().clamp(0, W - 1)
                    py = ((xyz_cam[:, 1] / z / tan_fovy) * (H / 2) + H / 2
                          ).long().clamp(0, H - 1)
                    contrib[vis_idx] = opacity[vis_idx] * l1_map[py, px]

                per_view.append(contrib)

        stacked    = torch.stack(per_view, dim=0)
        new_scores = stacked.var(dim=0)

        if self._scores is None or self._last_N != N:
            self._scores = new_scores
        else:
            self._scores = self.decay * self._scores + (1 - self.decay) * new_scores

        self._last_N = N

    def get_scores(self, current_N):
        if self._scores is None or self._last_N != current_N:
            return None
        return self._scores

    def reset(self):
        self._scores = None
        self._last_N = None