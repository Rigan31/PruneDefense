#!/bin/bash

SCENES=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")
# SCENES=("chair")

GPU=0
EXP_RUNS=1

# poisoned dataset
POISONED_DATA_ROOT="dataset/Nerf_Synthetic_eps16"
OUTPUT_ROOT="log/07_lightgaussian_defense/nerf_synthetic_eps16_benchmark"


# Defense hyperparameters (LightGaussian-style)
PRUNE_ITERATIONS="15500 20000 25000"   # When to prune (after densification ends at 15000)
PRUNE_PERCENT=0.5                       # Fraction to prune each time (50%)
PRUNE_DECAY=0.6                         # Decay factor: 50% -> 30% -> 18% at successive steps
V_POW=0.1                               # Volume power in v_imp_score (0.1 = moderate)
MAX_GAUSSIANS=500000                    # Hard safety cap (emergency brake)

for SCENE in "${SCENES[@]}"; do
    echo "=============================================="
    echo "Benchmarking LightGaussian defense on: ${SCENE}"
    echo "  Data:   ${POISONED_DATA_ROOT}/${SCENE}"
    echo "  Output: ${OUTPUT_ROOT}/${SCENE}"
    echo "  Runs:   ${EXP_RUNS}"
    echo "  Prune iterations: ${PRUNE_ITERATIONS}"
    echo "  Prune percent:    ${PRUNE_PERCENT}"
    echo "  Prune decay:      ${PRUNE_DECAY}"
    echo "  V_pow:            ${V_POW}"
    echo "  Max Gaussians:    ${MAX_GAUSSIANS}"
    echo "=============================================="

    python victim/gaussian-splatting/benchmark_lightgaussian_defense.py \
        -s "${POISONED_DATA_ROOT}/${SCENE}" \
        -m "${OUTPUT_ROOT}/${SCENE}/" \
        --gpu ${GPU} \
        --exp_runs ${EXP_RUNS} \
        --iterations 30000 \
        --test_iterations 7000 30000 \
        --save_iterations 7000 30000 \
        --prune_iterations ${PRUNE_ITERATIONS} \
        --prune_percent ${PRUNE_PERCENT} \
        --prune_decay ${PRUNE_DECAY} \
        --v_pow ${V_POW} \
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