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

# ======================================================================
# DEFENSE: Import LightGaussian-style pruning defense
# ======================================================================
from utils.lightgaussian_pruning import LightGaussianPruningDefense
# ======================================================================

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

def victim_training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, exp_run):
    os.makedirs(f'{args.model_path}/exp_run_{exp_run}/', exist_ok=True)
    # ==============Tools for monitoring victim status==================
    record_gaussian_num = []
    record_iter_elapse = []
    record_l1 = []
    record_ssim = []
    record_psnr = []
    record_pruned = []           # NEW: track pruning events
    gpu_monitor_stop_event = multiprocessing.Event()
    gpu_log_file_handle = open(f'{args.model_path}/exp_run_{exp_run}/gpu.log', 'w')
    gpu_monitor_process = multiprocessing.Process(target=gpu_monitor_worker, args=(gpu_monitor_stop_event, gpu_log_file_handle, args.gpu))
    fix_all_random_seed()
    # =================================================================
    

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False) # set `shuffle=False` to fix randomness
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    first_iter += 1
    gpu_monitor_process.start()

    # ==================================================================
    # DEFENSE: Initialize LightGaussian-style pruning defense
    #
    # Key parameters (adjust based on your attack/scene):
    #   prune_iterations: When to prune. We prune multiple times:
    #     - 15500: Right after densification ends (15000 in vanilla 3DGS)
    #       This catches the bulk of poison-induced Gaussians
    #     - 20000: Mid-training cleanup
    #     - 25000: Late-training refinement
    #   prune_percent: What fraction to prune each time (0.5 = 50%)
    #   prune_decay: Reduce prune aggressiveness over time
    #     With decay=0.6: prune 50%, then 30%, then 18%
    #   v_pow: Volume influence power (0.1 = moderate volume weight)
    #     Higher v_pow penalizes tiny Gaussians more (good vs poison)
    #   max_gaussians: Hard safety cap (emergency brake)
    #
    # For stronger defense against unconstrained attacks, increase
    # prune_percent to 0.6 and add more prune_iterations.
    # For constrained (eps=16/255) attacks, 0.3-0.5 is usually sufficient.
    # ==================================================================
    defense = LightGaussianPruningDefense(
        prune_iterations=args.prune_iterations,
        prune_percent=args.prune_percent,
        prune_decay=args.prune_decay,
        v_pow=args.v_pow,
        max_gaussians=args.max_gaussians,
    )

    print(f"\n[DEFENSE CONFIG]")
    print(f"  Prune iterations: {defense.prune_iterations}")
    print(f"  Prune percent: {defense.prune_percent}")
    print(f"  Prune decay: {defense.prune_decay}")
    print(f"  Volume power (v_pow): {defense.v_pow}")
    print(f"  Max Gaussians (hard cap): {defense.max_gaussians}")
    print()
    # ==================================================================


    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

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
            # if iteration % 10 == 0:
            #     print(f'[GPU: {args.gpu}]: Run {exp_run} iteration {iteration} loss {ema_loss_for_log:.3f}')
                
            # # Log and save
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # ===========================================================
            # DEFENSE: LightGaussian-style pruning at specified iterations
            # ===========================================================
            if defense.should_prune(iteration):
                n_pruned = defense.prune(
                    gaussians, scene, pipe, background, iteration
                )
                record_pruned.append((iteration, n_pruned))

            # DEFENSE: Hard cap safety valve (checked every 1000 iters)
            if iteration % 1000 == 0 and iteration > 0:
                current_count = gaussians.get_xyz.shape[0]
                if current_count > defense.max_gaussians:
                    n_pruned = defense.enforce_hard_cap(
                        gaussians, scene, pipe, background
                    )
                    record_pruned.append((iteration, n_pruned))
            # ===========================================================

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

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

            # Print progress every 1000 iterations
            if iteration % 1000 == 0:
                print(f'[GPU: {args.gpu}] Run {exp_run} iter {iteration} | '
                      f'loss {ema_loss_for_log:.4f} | '
                      f'#Gaussians {gaussians._xyz.shape[0]}')
            # ==========================================================================

    # ==================== Write Victim Records ==============================
    gpu_monitor_stop_event.set()
    gpu_monitor_process.join()
    gpu_log_file_handle.flush()
    gpu_log_file_handle.close()

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

    # DEFENSE: Save pruning event log
    if record_pruned:
        pruned_log_path = f'{args.model_path}/exp_run_{exp_run}/pruning_events.log'
        with open(pruned_log_path, 'w') as f:
            f.write("iteration,gaussians_pruned\n")
            for iter_num, n_pruned in record_pruned:
                f.write(f"{iter_num},{n_pruned}\n")
        print(f"\n[DEFENSE] Pruning log saved to {pruned_log_path}")

    
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
    max_GPU_mem = max(gpu_mem_cost)
    result_log = open(f'{args.model_path}/exp_run_{exp_run}/benchmark_result.log', 'w')
    result_str = ''
    result_str += f"Max Gaussian Number: {max_gaussian_nums:.3f} M\n"
    result_str += f"Max GPU mem: {int(max_GPU_mem)} MB\n"
    result_str += f"Training time: {training_time:.3f} min\n"
    result_str += f"SSIM: {mean_SSIM:.3f} \n"
    result_str += f"PSNR: {mean_PSNR:.3f} \n"
    # DEFENSE: Add defense config to results
    result_str += f"--- Defense Config ---\n"
    result_str += f"Prune iterations: {defense.prune_iterations}\n"
    result_str += f"Prune percent: {defense.prune_percent}\n"
    result_str += f"Prune decay: {defense.prune_decay}\n"
    result_str += f"V_pow: {defense.v_pow}\n"
    result_str += f"Max Gaussians cap: {defense.max_gaussians}\n"
    if record_pruned:
        result_str += f"Total pruning events: {len(record_pruned)}\n"
        total_pruned = sum(n for _, n in record_pruned)
        result_str += f"Total Gaussians pruned: {total_pruned}\n"
    print(result_str)
    result_log.write(result_str)
    result_log.flush()
    result_log.close()
    # ======================================================================

def conclude_victim_multiple_runs(args):
    max_gaussian_nums_runs = []
    max_gpu_mem_runs = []
    training_time_runs = []
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
        max_GPU_mem = max(gpu_mem_cost)

        max_gaussian_nums_runs.append(max_gaussian_nums)
        max_gpu_mem_runs.append(max_GPU_mem)
        training_time_runs.append(training_time)
    
    max_gaussian_nums_runs_mean = np.mean(np.array(max_gaussian_nums_runs))
    max_gaussian_nums_runs_std = np.std(np.array(max_gaussian_nums_runs))
    max_gpu_mem_runs_mean = np.mean(np.array(max_gpu_mem_runs))
    max_gpu_mem_runs_std = np.std(np.array(max_gpu_mem_runs))
    training_time_runs_mean = np.mean(np.array(training_time_runs))
    training_time_runs_std = np.std(np.array(training_time_runs))

    result_log = open(f'{args.model_path}/benchmark_result.log', 'w')
    result_str = ''
    result_str += f"Max Gaussian Number: {max_gaussian_nums_runs_mean:.3f} M +- {max_gaussian_nums_runs_std:.3f} M\n"
    result_str += f"Max GPU mem: {int(max_gpu_mem_runs_mean)} MB +- {int(max_gpu_mem_runs_std)} MB\n"
    result_str += f"Training time: {training_time_runs_mean:.2f} min +- {training_time_runs_std} min\n"
    print(result_str)
    result_log.write(result_str)
    result_log.flush()
    result_log.close()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="3DGS Victim Benchmark with LightGaussian Defense")
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
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--exp_runs", type=int, default=3)

    # ======================================================================
    # DEFENSE: LightGaussian pruning arguments
    # ======================================================================
    parser.add_argument("--prune_iterations", nargs="+", type=int,
                        default=[15500, 20000, 25000],
                        help="Iterations at which to perform LightGaussian pruning. "
                             "Default: [15500, 20000, 25000] (after densification ends)")
    parser.add_argument("--prune_percent", type=float, default=0.5,
                        help="Fraction of Gaussians to prune at each step (0.0-1.0). "
                             "Default: 0.5 (50%%). Higher = more aggressive defense.")
    parser.add_argument("--prune_decay", type=float, default=0.6,
                        help="Decay factor for prune_percent at successive steps. "
                             "Default: 0.6 (50%% -> 30%% -> 18%%). Set to 1.0 for no decay.")
    parser.add_argument("--v_pow", type=float, default=0.1,
                        help="Power for volume normalization in v_imp_score. "
                             "Default: 0.1. Higher = penalize tiny Gaussians more.")
    parser.add_argument("--max_gaussians", type=int, default=500000,
                        help="Hard cap on Gaussian count (emergency safety valve). "
                             "Default: 500000.")
    # ======================================================================

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.makedirs(args.model_path, exist_ok=True)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    for exp_run in range(1, args.exp_runs + 1):
        victim_training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, exp_run)

    # All done
    print("\nTraining complete.")

    conclude_victim_multiple_runs(args)

    ## usage:
    # python benchmark_lightgaussian_defense.py -s [data path] -m [output path] --gpu [x]
    #
    ## With custom defense parameters:
    # python benchmark_lightgaussian_defense.py -s [data path] -m [output path] --gpu [x] \
    #   --prune_iterations 15500 20000 25000 \
    #   --prune_percent 0.5 \
    #   --prune_decay 0.6 \
    #   --v_pow 0.1 \
    #   --max_gaussians 500000
    #
    ## For stronger defense against unconstrained attacks:
    # python benchmark_lightgaussian_defense.py -s [data path] -m [output path] --gpu [x] \
    #   --prune_iterations 10000 15500 20000 25000 \
    #   --prune_percent 0.6 \
    #   --prune_decay 0.8 \
    #   --v_pow 0.3 \
    #   --max_gaussians 300000