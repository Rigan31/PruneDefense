import torch
import numpy as np
import os
import sys
import random
from random import randint
import uuid
import time
import re
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import multiprocessing
from gpuinfo import GPUInfo
from datetime import datetime
import matplotlib.pyplot as plt

# === DEFENSE: Import smart pruning defense ===
from defense.pup_pruning.pruning_defense import SmartPruningDefense


def gpu_monitor_worker(stop_event, log_file_handle, gpuid=0):
    while not stop_event.is_set():
        now = datetime.now()
        timestamp = datetime.timestamp(now)
        dt_object = datetime.fromtimestamp(timestamp)
        formatted_date = dt_object.strftime('%Y-%m-%d %H:%M:%S')
        percent, memory = GPUInfo.gpu_usage()
        if isinstance(percent, list):
            percent = [percent[gpuid]]
            memory = [memory[gpuid]]
        log_file_handle.write(f'[{formatted_date}] GPU:{gpuid} uses {percent}% and {memory} MB\n')
        log_file_handle.flush()
        time.sleep(0.2)
    print(f'GPU {gpuid} monitor stops')

def plot_record(file_name, record_name, xlabel='Iteration'):
    if not os.path.exists(file_name):
        return
    record = np.load(file_name)
    plt.figure()
    plt.plot(record, label=record_name)
    plt.xlabel(xlabel)
    plt.ylabel(record_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name.replace('.npy', '.png'))
    plt.close()

def fix_all_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def victim_training(dataset, opt, pipe, testing_iterations, saving_iterations,
                    checkpoint_iterations, checkpoint, debug_from, exp_run,
                    # === DEFENSE: pass defense config ===
                    defense_args=None):

    os.makedirs(f'{args.model_path}/exp_run_{exp_run}/', exist_ok=True)

    # ==============Tools for monitoring victim status==================
    record_gaussian_num = []
    record_iter_elapse = []
    record_l1 = []
    record_ssim = []
    record_psnr = []
    # === DEFENSE: track defense prune events ===
    record_defense_prunes = []  # list of (iteration, removed, remaining, reason)

    gpu_monitor_stop_event = multiprocessing.Event()
    gpu_log_file_handle = open(f'{args.model_path}/exp_run_{exp_run}/gpu.log', 'w')
    gpu_monitor_process = multiprocessing.Process(
        target=gpu_monitor_worker,
        args=(gpu_monitor_stop_event, gpu_log_file_handle, args.gpu))
    fix_all_random_seed()
    # =================================================================

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)  # set `shuffle=False` to fix randomness
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    first_iter += 1
    gpu_monitor_process.start()

    # ===================================================================
    # === DEFENSE: Initialize Smart Pruning Defense ===
    # ===================================================================
    defense = None
    if defense_args is not None and defense_args.get("enabled", False):
        defense = SmartPruningDefense(
            max_gaussians=defense_args.get("max_gaussians", 500_000),
            soft_budget=defense_args.get("soft_budget",
                                         int(defense_args.get("max_gaussians", 500_000) * 0.6)),
            pruning_interval=defense_args.get("prune_interval", 500),
            pruning_start_iter=defense_args.get("prune_start_iter", 1000),
            growth_rate_threshold=defense_args.get("growth_rate_threshold", 2.0),
            aggressive_prune_ratio=defense_args.get("aggressive_prune_ratio", 0.5),
            normal_prune_ratio=defense_args.get("normal_prune_ratio", 0.1),
            score_type=defense_args.get("score_type", "gradient"),
            verbose=True,
        )
        defense.log(f"Smart Pruning Defense ENABLED for exp_run_{exp_run}")
        defense.log(f"  max_gaussians={defense.max_gaussians}")
        defense.log(f"  soft_budget={defense.soft_budget}")
        defense.log(f"  score_type={defense.score_type}")
        defense.log(f"  prune_interval={defense.pruning_interval}")
        defense.log(f"  normal_prune_ratio={defense.normal_prune_ratio}")
        defense.log(f"  aggressive_prune_ratio={defense.aggressive_prune_ratio}")
    # ===================================================================

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        Lssim = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - Lssim)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            # ======================================================
            # === DEFENSE: Accumulate gradients BEFORE densification
            # === (gradients from loss.backward() match current count)
            # ======================================================
            if defense is not None:
                defense.accumulate_gradients(gaussians)
            # ======================================================

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor,
                                                   visibility_filter)

                if (iteration > opt.densify_from_iter and
                        iteration % opt.densification_interval == 0):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005,
                        scene.cameras_extent, size_threshold)

                if (iteration % opt.opacity_reset_interval == 0 or
                        (dataset.white_background and
                         iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            # ======================================================
            # === DEFENSE: Smart pruning (runs AFTER densification)
            # ======================================================
            if defense is not None:
                removed = defense.prune(gaussians, iteration)
                if removed > 0:
                    remaining = gaussians.get_xyz.shape[0]
                    reason = defense._prune_history[-1]["reason"] if defense._prune_history else "UNKNOWN"
                    record_defense_prunes.append(
                        (iteration, removed, remaining, reason))
            # ======================================================

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # ==================== Record iteration victim status =======================
            try:
                iter_elapse = iter_start.elapsed_time(iter_end)
                record_iter_elapse.append(iter_elapse)
            except:
                pass
            record_gaussian_num.append(gaussians._xyz.shape[0])
            record_psnr.append(psnr(image, gt_image).mean().item())
            record_l1.append(Ll1.item())
            record_ssim.append(Lssim.item())
            # ==========================================================================

    # ==================== Write Victim Records ==============================
    gpu_monitor_stop_event.set()
    gpu_monitor_process.join()
    gpu_log_file_handle.flush()
    gpu_log_file_handle.close()

    # Final quality evaluation on all training views
    SSIM_views = []
    PSNR_views = []
    viewpoint_stack = scene.getTrainCameras().copy()
    for camid, cam in enumerate(viewpoint_stack):
        gt_image = cam.original_image.cuda()
        render_image = render(cam, gaussians, pipe, bg)['render']
        SSIM_views.append(ssim(gt_image, render_image).item())
        PSNR_views.append(psnr(gt_image, render_image).mean().item())
    mean_SSIM = round(sum(SSIM_views) / len(SSIM_views), 4)
    mean_PSNR = round(sum(PSNR_views) / len(PSNR_views), 4)

    gaussians.save_ply(f'{args.model_path}/exp_run_{exp_run}/victim_model.ply')

    # Save numpy records + plots
    gaussian_num_record_numpy = np.array(record_gaussian_num)
    np.save(f'{args.model_path}/exp_run_{exp_run}/gaussian_num_record.npy', gaussian_num_record_numpy)
    plot_record(f'{args.model_path}/exp_run_{exp_run}/gaussian_num_record.npy', 'Number of Gaussians')

    iter_elapse_record_numpy = np.array(record_iter_elapse)
    np.save(f'{args.model_path}/exp_run_{exp_run}/iter_elapse_record.npy', iter_elapse_record_numpy)
    plot_record(f'{args.model_path}/exp_run_{exp_run}/iter_elapse_record.npy', 'Iteration Elapse Time [ms]', 'Time')

    psnr_record_numpy = np.array(record_psnr)
    np.save(f'{args.model_path}/exp_run_{exp_run}/psnr_record.npy', psnr_record_numpy)
    plot_record(f'{args.model_path}/exp_run_{exp_run}/psnr_record.npy', 'PSNR')

    l1_record_numpy = np.array(record_l1)
    np.save(f'{args.model_path}/exp_run_{exp_run}/l1_record.npy', l1_record_numpy)
    plot_record(f'{args.model_path}/exp_run_{exp_run}/l1_record.npy', 'L1 Loss')

    ssim_record_numpy = np.array(record_ssim)
    np.save(f'{args.model_path}/exp_run_{exp_run}/ssim_record.npy', ssim_record_numpy)
    plot_record(f'{args.model_path}/exp_run_{exp_run}/ssim_record.npy', 'SSIM')

    # Parse GPU log
    gpu_log = open(f'{args.model_path}/exp_run_{exp_run}/gpu.log', 'r')
    timestamps = []
    gpu_usage_percentage = []
    gpu_mem_cost = []
    for line in gpu_log:
        pattern = r'\[(.*?)\]'
        matches = re.findall(pattern, line)
        timestamps.append(matches[0])
        gpu_usage_percentage.append(int(matches[1]))
        gpu_mem_cost.append(int(matches[2]))

    plt.figure()
    plt.plot(gpu_mem_cost, label='GPU memory cost [MB]')
    plt.xlabel('Training time')
    plt.ylabel('GPU memory cost [MB]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{args.model_path}/exp_run_{exp_run}/gpu_mem_cost.png')
    plt.close()

    training_start_timestamp = timestamps[0]
    training_end_timestamp = timestamps[-1]
    training_start_time = datetime.strptime(training_start_timestamp, "%Y-%m-%d %H:%M:%S")
    training_end_time = datetime.strptime(training_end_timestamp, "%Y-%m-%d %H:%M:%S")
    training_time_diff = training_end_time - training_start_time
    training_time = training_time_diff.seconds / 60
    max_gaussian_nums = max(record_gaussian_num) / 1000 / 1000
    final_gaussian_nums = record_gaussian_num[-1] / 1000 / 1000
    max_GPU_mem = max(gpu_mem_cost)

    # === DEFENSE: Compute defense-specific stats ===
    total_pruned = sum(r[1] for r in record_defense_prunes)
    num_prune_events = len(record_defense_prunes)
    attack_warnings = defense._attack_warnings if defense is not None else 0

    # Write benchmark result
    result_log = open(f'{args.model_path}/exp_run_{exp_run}/benchmark_result.log', 'w')
    result_str = ''
    result_str += f"Max Gaussian Number: {max_gaussian_nums:.3f} M\n"
    result_str += f"Final Gaussian Number: {final_gaussian_nums:.3f} M\n"
    result_str += f"Max GPU mem: {int(max_GPU_mem)} MB\n"
    result_str += f"Training time: {training_time:.3f} min\n"
    result_str += f"SSIM: {mean_SSIM:.4f}\n"
    result_str += f"PSNR: {mean_PSNR:.4f}\n"
    # === DEFENSE: Log defense stats ===
    result_str += f"--- Defense Stats ---\n"
    result_str += f"Defense enabled: {defense is not None}\n"
    if defense is not None:
        result_str += f"Score type: {defense.score_type}\n"
        result_str += f"Max budget: {defense.max_gaussians}\n"
        result_str += f"Prune events: {num_prune_events}\n"
        result_str += f"Total Gaussians pruned: {total_pruned}\n"
        result_str += f"Attack warnings: {attack_warnings}\n"
    print(result_str)
    result_log.write(result_str)
    result_log.flush()
    result_log.close()

    # === DEFENSE: Save defense prune log ===
    if defense is not None and len(record_defense_prunes) > 0:
        defense_log_path = f'{args.model_path}/exp_run_{exp_run}/defense_prune_log.txt'
        with open(defense_log_path, 'w') as f:
            f.write("iteration,removed,remaining,reason\n")
            for (it, removed, remaining, reason) in record_defense_prunes:
                f.write(f"{it},{removed},{remaining},{reason}\n")
        defense.print_summary()
    # ======================================================================


def conclude_victim_multiple_runs(args):
    max_gaussian_nums_runs = []
    final_gaussian_nums_runs = []
    max_gpu_mem_runs = []
    training_time_runs = []
    ssim_runs = []
    psnr_runs = []
    total_pruned_runs = []

    for exp_run in range(1, args.exp_runs + 1):
        record_gaussian_num = np.load(f'{args.model_path}/exp_run_{exp_run}/gaussian_num_record.npy')

        gpu_log = open(f'{args.model_path}/exp_run_{exp_run}/gpu.log', 'r')
        timestamps = []
        gpu_usage_percentage = []
        gpu_mem_cost = []
        for line in gpu_log:
            pattern = r'\[(.*?)\]'
            matches = re.findall(pattern, line)
            timestamps.append(matches[0])
            gpu_usage_percentage.append(int(matches[1]))
            gpu_mem_cost.append(int(matches[2]))

        training_start_timestamp = timestamps[0]
        training_end_timestamp = timestamps[-1]
        training_start_time = datetime.strptime(training_start_timestamp, "%Y-%m-%d %H:%M:%S")
        training_end_time = datetime.strptime(training_end_timestamp, "%Y-%m-%d %H:%M:%S")
        training_time_diff = training_end_time - training_start_time
        training_time = training_time_diff.seconds / 60
        max_gaussian_nums = max(record_gaussian_num) / 1000 / 1000
        final_gaussian_nums = record_gaussian_num[-1] / 1000 / 1000
        max_GPU_mem = max(gpu_mem_cost)

        max_gaussian_nums_runs.append(max_gaussian_nums)
        final_gaussian_nums_runs.append(final_gaussian_nums)
        max_gpu_mem_runs.append(max_GPU_mem)
        training_time_runs.append(training_time)

        # === DEFENSE: Read per-run benchmark_result.log for PSNR/SSIM ===
        bench_log_path = f'{args.model_path}/exp_run_{exp_run}/benchmark_result.log'
        if os.path.exists(bench_log_path):
            with open(bench_log_path, 'r') as f:
                for line in f:
                    if line.startswith("PSNR:"):
                        psnr_runs.append(float(line.split(":")[1].strip()))
                    elif line.startswith("SSIM:"):
                        ssim_runs.append(float(line.split(":")[1].strip()))
                    elif line.startswith("Total Gaussians pruned:"):
                        total_pruned_runs.append(int(line.split(":")[1].strip()))

    max_gaussian_nums_runs_mean = np.mean(np.array(max_gaussian_nums_runs))
    max_gaussian_nums_runs_std = np.std(np.array(max_gaussian_nums_runs))
    final_gaussian_nums_runs_mean = np.mean(np.array(final_gaussian_nums_runs))
    final_gaussian_nums_runs_std = np.std(np.array(final_gaussian_nums_runs))
    max_gpu_mem_runs_mean = np.mean(np.array(max_gpu_mem_runs))
    max_gpu_mem_runs_std = np.std(np.array(max_gpu_mem_runs))
    training_time_runs_mean = np.mean(np.array(training_time_runs))
    training_time_runs_std = np.std(np.array(training_time_runs))

    result_log = open(f'{args.model_path}/benchmark_result.log', 'w')
    result_str = ''
    result_str += f"Max Gaussian Number: {max_gaussian_nums_runs_mean:.3f} M +- {max_gaussian_nums_runs_std:.3f} M\n"
    result_str += f"Final Gaussian Number: {final_gaussian_nums_runs_mean:.3f} M +- {final_gaussian_nums_runs_std:.3f} M\n"
    result_str += f"Max GPU mem: {int(max_gpu_mem_runs_mean)} MB +- {int(max_gpu_mem_runs_std)} MB\n"
    result_str += f"Training time: {training_time_runs_mean:.2f} min +- {training_time_runs_std:.2f} min\n"
    if psnr_runs:
        result_str += f"PSNR: {np.mean(psnr_runs):.4f} +- {np.std(psnr_runs):.4f}\n"
    if ssim_runs:
        result_str += f"SSIM: {np.mean(ssim_runs):.4f} +- {np.std(ssim_runs):.4f}\n"
    if total_pruned_runs:
        result_str += f"Total pruned (avg): {int(np.mean(total_pruned_runs))} +- {int(np.std(total_pruned_runs))}\n"
    print(result_str)
    result_log.write(result_str)
    result_log.flush()
    result_log.close()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="3DGS Victim Benchmark with Smart Pruning Defense")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--exp_runs", type=int, default=3)

    # === DEFENSE: Command-line arguments ===
    parser.add_argument("--defense_enabled", action="store_true", default=False,
                        help="Enable smart pruning defense")
    parser.add_argument("--max_gaussians", type=int, default=500_000,
                        help="Hard upper bound on Gaussian count")
    parser.add_argument("--defense_prune_interval", type=int, default=500,
                        help="Check for pruning every N iterations")
    parser.add_argument("--defense_prune_start", type=int, default=1000,
                        help="Start defense pruning after this iteration")
    parser.add_argument("--defense_normal_ratio", type=float, default=0.1,
                        help="Fraction to prune in periodic maintenance (0.0-1.0)")
    parser.add_argument("--defense_aggressive_ratio", type=float, default=0.5,
                        help="Fraction to prune when attack detected (0.0-1.0)")
    parser.add_argument("--defense_growth_threshold", type=float, default=2.0,
                        help="Growth rate to flag as attack")
    parser.add_argument("--defense_score_type", type=str, default="gradient",
                        choices=["gradient", "opacity_volume", "hybrid"],
                        help="Importance scoring method")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.makedirs(args.model_path, exist_ok=True)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # === DEFENSE: Build config dict from CLI args ===
    defense_args = None
    if args.defense_enabled:
        defense_args = {
            "enabled": True,
            "max_gaussians": args.max_gaussians,
            "soft_budget": int(args.max_gaussians * 0.6),
            "prune_interval": args.defense_prune_interval,
            "prune_start_iter": args.defense_prune_start,
            "growth_rate_threshold": args.defense_growth_threshold,
            "aggressive_prune_ratio": args.defense_aggressive_ratio,
            "normal_prune_ratio": args.defense_normal_ratio,
            "score_type": args.defense_score_type,
        }

    for exp_run in range(1, args.exp_runs + 1):
        victim_training(
            lp.extract(args), op.extract(args), pp.extract(args),
            args.test_iterations, args.save_iterations,
            args.checkpoint_iterations, args.start_checkpoint,
            args.debug_from, exp_run,
            defense_args=defense_args,
        )

    # All done
    print("\nTraining complete.")
    conclude_victim_multiple_runs(args)

    ## usage (WITHOUT defense - same as original):
    # python benchmark.py -s [data path] -m [output path] --gpu [x]
    #
    ## usage (WITH smart pruning defense):
    # python benchmark.py -s [data path] -m [output path] --gpu [x] \
    #     --defense_enabled --max_gaussians 300000 --defense_score_type gradient