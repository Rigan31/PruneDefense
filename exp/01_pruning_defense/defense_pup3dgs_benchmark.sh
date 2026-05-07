#!/bin/bash


SCENES=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
# SCENES=("chair")
GPU=0
EXP_RUNS=1

# poisoned dataset 
POISONED_DATA_ROOT="dataset/Nerf_Synthetic_eps16"
OUTPUT_ROOT="log/09_pup3dgs_defense/nerf_synthetic_eps16_benchmark"


# Defense hyperparameters
MAX_GAUSSIANS=300000       # NeRF-Synthetic scenes are simple, 300K is generous
PRUNE_INTERVAL=500
PRUNE_START=1000
NORMAL_RATIO=0.1
AGGRESSIVE_RATIO=0.5
GROWTH_THRESHOLD=2.0
SCORE_TYPE="gradient"

for SCENE in "${SCENES[@]}"; do
    echo "=============================================="
    echo "Benchmarking smart pruning defense on: ${SCENE}"
    echo "  Data:   ${POISONED_DATA_ROOT}/${SCENE}"
    echo "  Output: ${OUTPUT_ROOT}/${SCENE}"
    echo "  Runs:   ${EXP_RUNS}"
    echo "=============================================="

    python victim/gaussian-splatting/benchmark_pup3dgs_defense.py \
        -s "${POISONED_DATA_ROOT}/${SCENE}" \
        -m "${OUTPUT_ROOT}/${SCENE}/" \
        --gpu ${GPU} \
        --exp_runs ${EXP_RUNS} \
        --iterations 30000 \
        --test_iterations 7000 30000 \
        --save_iterations 7000 30000 \
        --defense_enabled \
        --max_gaussians ${MAX_GAUSSIANS} \
        --defense_prune_interval ${PRUNE_INTERVAL} \
        --defense_prune_start ${PRUNE_START} \
        --defense_normal_ratio ${NORMAL_RATIO} \
        --defense_aggressive_ratio ${AGGRESSIVE_RATIO} \
        --defense_growth_threshold ${GROWTH_THRESHOLD} \
        --defense_score_type ${SCORE_TYPE}

    echo ""
    echo "Done: ${SCENE}"
    echo "Per-run results:  ${OUTPUT_ROOT}/${SCENE}/exp_run_*/benchmark_result.log"
    echo "Aggregated result: ${OUTPUT_ROOT}/${SCENE}/benchmark_result.log"
    echo ""
done

echo "=============================================="
echo "All NeRF-Synthetic scenes complete."
echo "Results saved to: ${OUTPUT_ROOT}"
echo "=============================================="