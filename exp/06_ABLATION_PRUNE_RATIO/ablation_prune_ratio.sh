#!/bin/bash

set -e
set -o pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT="*"
SCRIPTS="${ROOT}/victim/gaussian-splatting"
BENCHMARK_DEF="${SCRIPTS}/benchmark_mvpi_defense.py"
PYTHON="python"
GPU=0

LOG_ROOT="${ROOT}/log/15_ABLATION_PRUNE_RATIO"
mkdir -p "${LOG_ROOT}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_ROOT}/full_run_${TIMESTAMP}.log"


PRUNE_RATIOS=(0.01 0.02 0.05 0.10 0.15 0.20 0.30 0.50 0.75)


SCENE_DATA="${ROOT}/dataset/Nerf_Synthetic_eps16/lego"
SCENE_TAG="lego_eps16"

# ── Logging helper ─────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${MAIN_LOG}"
}

# ── Run one ratio ──────────────────────────────────────────────────────────
run_one_ratio() {
    local RHO=$1
    # Convert 0.05 -> "0.05" with two decimals so folder names sort cleanly
    local RHO_TAG=$(printf "%.2f" "${RHO}")
    local OUT_DIR="${LOG_ROOT}/rho_${RHO_TAG}_${SCENE_TAG}"
    local LABEL="rho_${RHO_TAG}_${SCENE_TAG}"

    log ""
    log "  ────────────────────────────────────────"
    log "  Prune ratio: ${RHO_TAG}"
    log "  Scene      : ${SCENE_TAG}"
    log "  Output     : ${OUT_DIR}"
    log "  ────────────────────────────────────────"

    # Skip if already completed
    if [ -f "${OUT_DIR}/exp_run_1/benchmark_result.log" ]; then
        log "  [SKIP] Already completed: ${LABEL}"
        return
    fi

    if [ ! -d "${SCENE_DATA}" ]; then
        log "  [ERROR] Scene data not found: ${SCENE_DATA}"
        return
    fi

    mkdir -p "${OUT_DIR}"

    # Run training. The MVPI defense reads the prune ratio from the
    # MVPI_PRUNE_RATIO environment variable (see benchmark_mvpi_defense.py).
    cd "${SCRIPTS}"
    MVPI_PRUNE_RATIO="${RHO}" \
    ${PYTHON} "${BENCHMARK_DEF}" \
        -s "${SCENE_DATA}" \
        -m "${OUT_DIR}" \
        --gpu ${GPU} \
        --exp_runs 1 \
        2>&1 | tee -a "${LOG_ROOT}/${LABEL}.log" "${MAIN_LOG}"

    log "  Done: ${LABEL}"
}


# ══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════

log "======================================================"
log "  ABLATION: MVPI PRUNE RATIO"
log "  Started   : $(date)"
log "  GPU       : ${GPU}"
log "  Scene     : ${SCENE_TAG}"
log "  Ratios    : ${PRUNE_RATIOS[*]}"
log "  Log       : ${MAIN_LOG}"
log "======================================================"

TOTAL_START=$(date +%s)

for RHO in "${PRUNE_RATIOS[@]}"; do
    run_one_ratio "${RHO}"
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( (TOTAL_END - TOTAL_START) / 60 ))
log ""
log "  All runs done in ~${TOTAL_ELAPSED} minutes."


# ══════════════════════════════════════════════════════════════════════════
# COLLECT RESULTS INTO CSV
# ══════════════════════════════════════════════════════════════════════════

log ""
log "======================================================"
log "  Collecting ablation results"
log "======================================================"

RESULTS_CSV="${LOG_ROOT}/ablation_results.csv"
echo "prune_ratio,max_gaussians_M,training_time_min,max_gpu_mem_MB,SSIM,PSNR" > "${RESULTS_CSV}"

for RHO in "${PRUNE_RATIOS[@]}"; do
    RHO_TAG=$(printf "%.2f" "${RHO}")
    LOG_FILE="${LOG_ROOT}/rho_${RHO_TAG}_${SCENE_TAG}/exp_run_1/benchmark_result.log"

    if [ ! -f "${LOG_FILE}" ]; then
        echo "${RHO_TAG},,,,," >> "${RESULTS_CSV}"
        log "  rho=${RHO_TAG}: missing benchmark_result.log"
        continue
    fi

    MAX_GAUSS=$(grep -i "Max Gaussian"   "${LOG_FILE}" | head -1 | grep -oP '[\d.]+(?=\s*M)' || echo "")
    TRAIN_TIME=$(grep -i "Training time" "${LOG_FILE}" | head -1 | grep -oP '[\d.]+(?=\s*min)' || echo "")
    GPU_MEM=$(grep -i "Max GPU mem"      "${LOG_FILE}" | head -1 | grep -oP '\d+(?=\s*MB)' || echo "")
    SSIM_VAL=$(grep -i "^SSIM"           "${LOG_FILE}" | head -1 | grep -oP '[\d.]+' || echo "")
    PSNR_VAL=$(grep -i "^PSNR"           "${LOG_FILE}" | head -1 | grep -oP '[\d.]+' || echo "")

    echo "${RHO_TAG},${MAX_GAUSS},${TRAIN_TIME},${GPU_MEM},${SSIM_VAL},${PSNR_VAL}" >> "${RESULTS_CSV}"
    log "  rho=${RHO_TAG}: G=${MAX_GAUSS}M T=${TRAIN_TIME}m GPU=${GPU_MEM}MB SSIM=${SSIM_VAL} PSNR=${PSNR_VAL}"
done

log "  CSV saved: ${RESULTS_CSV}"


log ""
log "======================================================"
log "  Ablation done."
log "  Finished : $(date)"
log "  Outputs:"
log "    CSV     : ${RESULTS_CSV}"
log "======================================================"