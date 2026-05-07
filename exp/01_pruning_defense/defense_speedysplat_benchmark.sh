#!/bin/bash

SCENES=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
# SCENES=("chair")

GPU=0
EXP_RUNS=1

# poisoned dataset
POISONED_DATA_ROOT="dataset/Nerf_Synthetic_eps16"
OUTPUT_ROOT="log/08_speedysplat_defense/nerf_synthetic_eps16_benchmark"


# Defense hyperparameters (SpeedySplat-style)
SOFT_PRUNE_PERCENT=0.1              # 10% pruned at each densification step
HARD_PRUNE_PERCENT=0.3              # 30% pruned after densification
HARD_PRUNE_ITERATIONS="16000 20000 25000"  # When to hard prune
SCORE_ACCUM_VIEWS=5                 # Views for soft prune score computation
MAX_GAUSSIANS=500000                # Hard safety cap

for SCENE in "${SCENES[@]}"; do
    echo "=============================================="
    echo "Benchmarking SpeedySplat defense on: ${SCENE}"
    echo "  Data:   ${POISONED_DATA_ROOT}/${SCENE}"
    echo "  Output: ${OUTPUT_ROOT}/${SCENE}"
    echo "  Runs:   ${EXP_RUNS}"
    echo "  Soft prune:      ${SOFT_PRUNE_PERCENT}"
    echo "  Hard prune:      ${HARD_PRUNE_PERCENT}"
    echo "  Hard prune iters: ${HARD_PRUNE_ITERATIONS}"
    echo "  Max Gaussians:   ${MAX_GAUSSIANS}"
    echo "=============================================="

    python victim/gaussian-splatting/benchmark_speedysplat_defense.py \
        -s "${POISONED_DATA_ROOT}/${SCENE}" \
        -m "${OUTPUT_ROOT}/${SCENE}/" \
        --gpu ${GPU} \
        --exp_runs ${EXP_RUNS} \
        --iterations 30000 \
        --test_iterations 7000 30000 \
        --save_iterations 7000 30000 \
        --soft_prune_percent ${SOFT_PRUNE_PERCENT} \
        --hard_prune_percent ${HARD_PRUNE_PERCENT} \
        --hard_prune_iterations ${HARD_PRUNE_ITERATIONS} \
        --score_accum_views ${SCORE_ACCUM_VIEWS} \
        --max_gaussians ${MAX_GAUSSIANS}

    echo ""
    echo "Done: ${SCENE}"
    echo "Per-run results:   ${OUTPUT_ROOT}/${SCENE}/exp_run_*/benchmark_result.log"
    echo "Pruning logs:      ${OUTPUT_ROOT}/${SCENE}/exp_run_*/pruning_events.log"
    echo "Aggregated result: ${OUTPUT_ROOT}/${SCENE}/benchmark_result.log"
    echo ""
done

echo "=============================================="
echo "All NeRF-Synthetic scenes complete."
echo "Results saved to: ${OUTPUT_ROOT}"
echo "=============================================="