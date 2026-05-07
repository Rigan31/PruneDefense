#!/bin/bash
set -e
set -o pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT="*"
SCRIPTS="${ROOT}/victim/gaussian-splatting"
RENDER_SCRIPT="${SCRIPTS}/render_compare.py"
PYTHON="python"
GPU=0

# Filter: which experiments to render
EXPERIMENT_FILTER="${1:-both}"

# Output log directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RENDER_LOG_DIR="${ROOT}/log/13_RENDER"
mkdir -p "${RENDER_LOG_DIR}"
MAIN_LOG="${RENDER_LOG_DIR}/render_run_${TIMESTAMP}.log"

# ── Sanity check ───────────────────────────────────────────────────────────
if [ ! -f "${RENDER_SCRIPT}" ]; then
    echo "ERROR: render_compare.py not found at: ${RENDER_SCRIPT}"
    exit 1
fi

# ── Objects per dataset ────────────────────────────────────────────────────
NERF_SYN_OBJECTS="chair drums ficus hotdog lego materials mic ship"
MIP360_OBJECTS="bicycle flowers garden stump treehill room counter kitchen bonsai"

# ── All datasets ───────────────────────────────────────────────────────────
DATASETS=(
    "${ROOT}/dataset/MIP_Nerf_360|MIP_Nerf_360|MIP360"
    "${ROOT}/dataset/MIP_Nerf_360_eps16|MIP_Nerf_360_eps16|MIP360"
    "${ROOT}/dataset/MIP_Nerf_360_unbounded|MIP_Nerf_360_unbounded|MIP360"
    "${ROOT}/dataset/Nerf_Synthetic|Nerf_Synthetic|NERF_SYN"
    "${ROOT}/dataset/Nerf_Synthetic_adversarial_eps4|Nerf_Synthetic_adversarial_eps4|NERF_SYN"
    "${ROOT}/dataset/Nerf_Synthetic_adversarial_eps8|Nerf_Synthetic_adversarial_eps8|NERF_SYN"
    "${ROOT}/dataset/Nerf_Synthetic_eps16|Nerf_Synthetic_eps16|NERF_SYN"
    "${ROOT}/dataset/Nerf_Synthetic_unbounded|Nerf_Synthetic_unbounded|NERF_SYN"
)

# ── Experiment types to render ─────────────────────────────────────────────
EXPERIMENTS=()
if [ "${EXPERIMENT_FILTER}" == "both" ] || [ "${EXPERIMENT_FILTER}" == "no_defense" ]; then
    EXPERIMENTS+=("13_NO_DEFENSE|No Defense")
fi
if [ "${EXPERIMENT_FILTER}" == "both" ] || [ "${EXPERIMENT_FILTER}" == "mvpi" ]; then
    EXPERIMENTS+=("13_MVPI_PRUNING|MVPI Defense")
fi

if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    echo "ERROR: Invalid filter '${EXPERIMENT_FILTER}'. Use: both | mvpi | no_defense"
    exit 1
fi

# ── Logging helper ─────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${MAIN_LOG}"
}

# ── Render one model ───────────────────────────────────────────────────────
render_one_model() {
    local DATA_PATH=$1
    local MODEL_DIR=$2
    local LABEL=$3

    local PLY_PATH="${MODEL_DIR}/exp_run_1/victim_model.ply"
    local OUTPUT_DIR="${MODEL_DIR}/exp_run_1/render_comparison"

    log ""
    log "  Rendering: ${LABEL}"
    log "  Data     : ${DATA_PATH}"
    log "  PLY      : ${PLY_PATH}"
    log "  Output   : ${OUTPUT_DIR}"

    if [ ! -d "${DATA_PATH}" ]; then
        log "  [SKIP] Dataset not found: ${DATA_PATH}"
        return 1
    fi

    if [ ! -f "${PLY_PATH}" ]; then
        log "  [SKIP] Trained PLY not found: ${PLY_PATH}"
        return 1
    fi

    # Skip if already rendered
    if [ -f "${OUTPUT_DIR}/metrics_summary.txt" ]; then
        log "  [SKIP] Already rendered: ${OUTPUT_DIR}/metrics_summary.txt"
        return 0
    fi

    cd "${SCRIPTS}"

    CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} "${RENDER_SCRIPT}" \
        -s "${DATA_PATH}" \
        -m "${MODEL_DIR}" \
        --ply "${PLY_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --quiet \
        2>&1 | tee -a "${RENDER_LOG_DIR}/${LABEL}.log" "${MAIN_LOG}"

    log "  Done: ${LABEL}"
    return 0
}


# ══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════

log "======================================================"
log "  RENDER ALL MODELS (using render_compare.py)"
log "  Started   : $(date)"
log "  GPU       : ${GPU}"
log "  Filter    : ${EXPERIMENT_FILTER}"
log "  Script    : ${RENDER_SCRIPT}"
log "  Log       : ${MAIN_LOG}"
log "======================================================"

TOTAL_START=$(date +%s)
TOTAL_RENDERED=0
TOTAL_SKIPPED=0

for EXP_ENTRY in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r LOG_DIR_NAME EXP_DESC <<< "${EXP_ENTRY}"
    LOG_ROOT="${ROOT}/log/${LOG_DIR_NAME}"

    if [ ! -d "${LOG_ROOT}" ]; then
        log ""
        log "  [SKIP] Experiment dir not found: ${LOG_ROOT}"
        continue
    fi

    log ""
    log "######################################################"
    log "  Experiment: ${EXP_DESC}"
    log "  Log root  : ${LOG_ROOT}"
    log "######################################################"

    for ENTRY in "${DATASETS[@]}"; do
        IFS='|' read -r DPATH DNAME OBJTYPE <<< "${ENTRY}"

        if [ "${OBJTYPE}" == "MIP360" ]; then
            OBJECTS="${MIP360_OBJECTS}"
        else
            OBJECTS="${NERF_SYN_OBJECTS}"
        fi

        log ""
        log "  ──── Dataset: ${DNAME} ────"

        if [ ! -d "${DPATH}" ]; then
            log "  [SKIP] Dataset not found: ${DPATH}"
            continue
        fi

        for OBJ in ${OBJECTS}; do
            OBJ_DATA="${DPATH}/${OBJ}"
            FOLDER_NAME="${DNAME}_${OBJ}"
            MODEL_DIR="${LOG_ROOT}/${FOLDER_NAME}"
            LABEL="${LOG_DIR_NAME}_${FOLDER_NAME}"

            if [ ! -d "${MODEL_DIR}" ]; then
                TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
                continue
            fi

            if render_one_model "${OBJ_DATA}" "${MODEL_DIR}" "${LABEL}"; then
                TOTAL_RENDERED=$((TOTAL_RENDERED + 1))
            else
                TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
            fi
        done
    done
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( (TOTAL_END - TOTAL_START) / 60 ))

log ""
log "======================================================"
log "  All rendering done."
log "  Finished       : $(date)"
log "  Total time     : ~${TOTAL_ELAPSED} minutes"
log "  Models rendered: ${TOTAL_RENDERED}"
log "  Models skipped : ${TOTAL_SKIPPED}"
log ""
log "  Output structure:"
log "    log/<EXP>/<dataset>_<scene>/exp_run_1/render_comparison/"
log "      ├── renders/                    rendered images"
log "      ├── gt/                         ground truth"
log "      ├── comparison/                 side-by-side (GT | Render)"
log "      └── metrics_summary.txt         per-image PSNR/SSIM/L1"
log "======================================================"