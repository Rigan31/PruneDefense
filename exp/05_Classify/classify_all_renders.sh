#!/bin/bash

set -e
set -o pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT="*"
CLASSIFY_SCRIPT="${ROOT}/classify_renders.py"   # adjust if it lives elsewhere
PYTHON="python"
GPU=0

# Filter: which datasets to process (default: both)
DATASET_FILTER="${1:-both}"

# Output log dir
LOG_ROOT="${ROOT}/log/14_CLASSIFY"
mkdir -p "${LOG_ROOT}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_ROOT}/full_run_${TIMESTAMP}.log"

SUMMARY_CSV="${LOG_ROOT}/classification_summary.csv"

# ── NeRF Synthetic class list ─────────────────────────────────────────────
NERF_CLASSES="chair drums ficus hotdog lego materials mic ship"

# ── Datasets to process ────────────────────────────────────────────────────
# Format: "DATASET_FOLDER_PREFIX|SHORT_TAG"
DATASETS=()
if [ "${DATASET_FILTER}" == "both" ] || [ "${DATASET_FILTER}" == "eps4" ]; then
    DATASETS+=("Nerf_Synthetic_adversarial_eps4|eps4")
fi
if [ "${DATASET_FILTER}" == "both" ] || [ "${DATASET_FILTER}" == "eps8" ]; then
    DATASETS+=("Nerf_Synthetic_adversarial_eps8|eps8")
fi

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "ERROR: Invalid filter '${DATASET_FILTER}'. Use: both | eps4 | eps8"
    exit 1
fi

# ── Where the trained models live ──────────────────────────────────────────
MVPI_LOG_ROOT="${ROOT}/log/13_MVPI_PRUNING"

# ── Logging helper ─────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${MAIN_LOG}"
}

# ── Initialize summary CSV ─────────────────────────────────────────────────
if [ ! -f "${SUMMARY_CSV}" ]; then
    echo "dataset,object,n_images,avg_true_conf,top1_correct,top1_total,top1_acc_pct,top3_correct,top3_total,top3_acc_pct" > "${SUMMARY_CSV}"
fi

# ── Process one model ──────────────────────────────────────────────────────
classify_one_scene() {
    local DATASET_PREFIX=$1   # e.g. Nerf_Synthetic_adversarial_eps4
    local DATASET_TAG=$2      # e.g. eps4
    local OBJ=$3              # e.g. chair

    local SCENE_FOLDER="${DATASET_PREFIX}_${OBJ}"
    local MODEL_DIR="${MVPI_LOG_ROOT}/${SCENE_FOLDER}"

    # Renders are at:
    #   <model_dir>/exp_run_1/render_comparison/renders/
    local RENDER_DIR="${MODEL_DIR}/exp_run_1/render_comparison/renders"

    local OUT_LOG="${LOG_ROOT}/${DATASET_TAG}_${OBJ}.log"
    local OUT_JSON="${LOG_ROOT}/${DATASET_TAG}_${OBJ}.json"

    log ""
    log "  ────────────────────────────────────────"
    log "  Scene  : ${DATASET_TAG} / ${OBJ}"
    log "  Render : ${RENDER_DIR}"
    log "  ────────────────────────────────────────"

    if [ ! -d "${RENDER_DIR}" ]; then
        log "  [SKIP] Render directory not found: ${RENDER_DIR}"
        echo "${DATASET_TAG},${OBJ},,,,,,,," >> "${SUMMARY_CSV}"
        return
    fi

    # Quick check that there are actually images in there
    local N_IMAGES
    N_IMAGES=$(find "${RENDER_DIR}" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | wc -l)
    if [ "${N_IMAGES}" -eq 0 ]; then
        log "  [SKIP] No image files in: ${RENDER_DIR}"
        echo "${DATASET_TAG},${OBJ},0,,,,,,," >> "${SUMMARY_CSV}"
        return
    fi
    log "  Found ${N_IMAGES} images"

    # Skip if already classified
    if [ -f "${OUT_JSON}" ]; then
        log "  [SKIP] Already classified (JSON exists): ${OUT_JSON}"
        return
    fi

    # Run classification — capture into the per-scene log
    cd "${ROOT}"
    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} "${CLASSIFY_SCRIPT}" \
        --image_dir  "${RENDER_DIR}" \
        --true_label "${OBJ}" \
        --classes    ${NERF_CLASSES} \
        --top_k      3 \
        --output     "${OUT_JSON}" \
        2>&1 | tee "${OUT_LOG}"

    # Parse summary numbers from the per-scene log file and append to CSV
    if [ -f "${OUT_LOG}" ]; then
        local N_IMG       AVG_CONF   T1_HIT  T1_TOT  T1_PCT  T3_HIT  T3_TOT  T3_PCT

        N_IMG=$(grep "Total images" "${OUT_LOG}" | head -1 | grep -oP '\d+' | tail -1)
        AVG_CONF=$(grep "Avg confidence (true class)" "${OUT_LOG}" | head -1 | grep -oP '[\d.]+' | tail -1)

        local T1_LINE T3_LINE
        T1_LINE=$(grep "Top-1 accuracy" "${OUT_LOG}" | head -1)
        T3_LINE=$(grep "Top-3 accuracy" "${OUT_LOG}" | head -1)

        T1_HIT=$(echo "${T1_LINE}" | grep -oP '\d+\s*/\s*\d+' | head -1 | grep -oP '^\d+')
        T1_TOT=$(echo "${T1_LINE}" | grep -oP '\d+\s*/\s*\d+' | head -1 | grep -oP '\d+$')
        T1_PCT=$(echo "${T1_LINE}" | grep -oP '\([\d.]+%' | grep -oP '[\d.]+')

        T3_HIT=$(echo "${T3_LINE}" | grep -oP '\d+\s*/\s*\d+' | head -1 | grep -oP '^\d+')
        T3_TOT=$(echo "${T3_LINE}" | grep -oP '\d+\s*/\s*\d+' | head -1 | grep -oP '\d+$')
        T3_PCT=$(echo "${T3_LINE}" | grep -oP '\([\d.]+%' | grep -oP '[\d.]+')

        echo "${DATASET_TAG},${OBJ},${N_IMG},${AVG_CONF},${T1_HIT},${T1_TOT},${T1_PCT},${T3_HIT},${T3_TOT},${T3_PCT}" >> "${SUMMARY_CSV}"

        log "  Done: ${OBJ}  |  Top-1: ${T1_HIT}/${T1_TOT} (${T1_PCT}%)  Top-3: ${T3_HIT}/${T3_TOT} (${T3_PCT}%)  AvgConf: ${AVG_CONF}"
    fi
}


# ══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════

log "======================================================"
log "  CLASSIFY MVPI RENDERS WITH CLIP"
log "  Started   : $(date)"
log "  GPU       : ${GPU}"
log "  Filter    : ${DATASET_FILTER}"
log "  Datasets  : ${DATASETS[*]}"
log "  Log       : ${MAIN_LOG}"
log "  Summary   : ${SUMMARY_CSV}"
log "======================================================"

TOTAL_START=$(date +%s)
TOTAL_DONE=0
TOTAL_SKIP=0

for ENTRY in "${DATASETS[@]}"; do
    IFS='|' read -r DATASET_PREFIX DATASET_TAG <<< "${ENTRY}"

    log ""
    log "######################################################"
    log "  Dataset: ${DATASET_PREFIX}  (tag: ${DATASET_TAG})"
    log "######################################################"

    for OBJ in ${NERF_CLASSES}; do
        SCENE_FOLDER="${DATASET_PREFIX}_${OBJ}"
        MODEL_DIR="${MVPI_LOG_ROOT}/${SCENE_FOLDER}"

        if [ ! -d "${MODEL_DIR}" ]; then
            log "  [SKIP] Model dir not found: ${MODEL_DIR}"
            TOTAL_SKIP=$((TOTAL_SKIP + 1))
            continue
        fi

        classify_one_scene "${DATASET_PREFIX}" "${DATASET_TAG}" "${OBJ}"
        TOTAL_DONE=$((TOTAL_DONE + 1))
    done
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( (TOTAL_END - TOTAL_START) / 60 ))

# ══════════════════════════════════════════════════════════════════════════
# PRINT FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════

log ""
log "======================================================"
log "  All classification done"
log "  Finished        : $(date)"
log "  Total time      : ~${TOTAL_ELAPSED} minutes"
log "  Scenes done     : ${TOTAL_DONE}"
log "  Scenes skipped  : ${TOTAL_SKIP}"
log "======================================================"
log ""
log "  Summary CSV: ${SUMMARY_CSV}"
log ""
log "  Per-dataset overall accuracy:"

for ENTRY in "${DATASETS[@]}"; do
    IFS='|' read -r DATASET_PREFIX DATASET_TAG <<< "${ENTRY}"

    awk -F',' -v ds="${DATASET_TAG}" '
        NR > 1 && $1 == ds && $5 != "" {
            n_img += $3
            sum_conf += $4 * $3
            sum_t1   += $5
            sum_t1t  += $6
            sum_t3   += $8
            sum_t3t  += $9
            count++
        }
        END {
            if (count == 0) { print "    " ds ": no scenes"; exit }
            avg_conf = sum_conf / n_img
            t1_acc   = (sum_t1t > 0) ? 100.0 * sum_t1 / sum_t1t : 0
            t3_acc   = (sum_t3t > 0) ? 100.0 * sum_t3 / sum_t3t : 0
            printf "    %s: %d scenes, %d images, AvgConf %.4f, Top-1 %.1f%%, Top-3 %.1f%%\n",
                   ds, count, n_img, avg_conf, t1_acc, t3_acc
        }
    ' "${SUMMARY_CSV}" | tee -a "${MAIN_LOG}"
done

log ""
log "  Per-scene log files: ${LOG_ROOT}/<eps>_<obj>.log"
log "  Per-scene JSONs    : ${LOG_ROOT}/<eps>_<obj>.json"
log "======================================================"