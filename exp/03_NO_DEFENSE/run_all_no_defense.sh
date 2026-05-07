#!/bin/bash
set -e

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT="*"
SCRIPTS="${ROOT}/victim/gaussian-splatting"
BENCHMARK_DEF="${SCRIPTS}/benchmark.py"
PYTHON="python"
GPU=0

LOG_ROOT="${ROOT}/log/13_NO_DEFENSE"
mkdir -p "${LOG_ROOT}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_ROOT}/full_run_${TIMESTAMP}.log"

# ── Objects per dataset ────────────────────────────────────────────────────
NERF_SYN_OBJECTS="chair drums ficus hotdog lego materials mic ship"
MIP360_OBJECTS="bicycle flowers garden stump treehill room counter kitchen bonsai"


DATASETS=(
    "${ROOT}/dataset/MIP_Nerf_360|MIP_Nerf_360|MIP360"
    "${ROOT}/dataset/MIP_Nerf_360_eps16|MIP_Nerf_360_eps16|MIP360"
    "${ROOT}/dataset/Nerf_Synthetic|Nerf_Synthetic|NERF_SYN"
    "${ROOT}/dataset/Nerf_Synthetic_adversarial_eps4|Nerf_Synthetic_adversarial_eps4|NERF_SYN"
    "${ROOT}/dataset/Nerf_Synthetic_adversarial_eps8|Nerf_Synthetic_adversarial_eps8|NERF_SYN"
    "${ROOT}/dataset/Nerf_Synthetic_eps16|Nerf_Synthetic_eps16|NERF_SYN"
    "${ROOT}/dataset/Nerf_Synthetic_unbounded|Nerf_Synthetic_unbounded|NERF_SYN"
)

# ── Logging helper ─────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${MAIN_LOG}"
}

measure_render_speed() {
    local DATA_PATH=$1
    local MODEL_DIR=$2
    local OUT_FILE=$3   # append FPS line to this file

    log "    Measuring render speed for ${MODEL_DIR} ..."

    ${PYTHON} - "${DATA_PATH}" "${MODEL_DIR}" "${OUT_FILE}" << 'PYEOF'
import sys, time, torch
from argparse import Namespace

# Add gaussian-splatting to path
sys.path.insert(0, sys.argv[0].replace(sys.argv[0], "") or ".")

try:
    from gaussian_renderer import render
    from scene import Scene, GaussianModel
    from arguments import ModelParams, PipelineParams
    from argparse import ArgumentParser

    data_path = sys.argv[1]
    model_dir = sys.argv[2]
    out_file  = sys.argv[3]

    # Minimal argument setup
    parser = ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    # Parse with just the source and model path
    args = parser.parse_args(["-s", data_path, "-m", model_dir])
    dataset = lp.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)

    # Load the trained model
    import os, glob
    ply_path = os.path.join(model_dir, "exp_run_1", "victim_model.ply")
    if os.path.exists(ply_path):
        gaussians.load_ply(ply_path)
    else:
        print(f"  [WARN] PLY not found: {ply_path}")
        with open(out_file, 'a') as f:
            f.write("Render FPS       : N/A\n")
        sys.exit(0)

    bg_color = [1,1,1] if dataset.white_background else [0,0,0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    cams = scene.getTrainCameras()

    # Warmup
    for cam in cams[:3]:
        _ = render(cam, gaussians, pipe, background)
    torch.cuda.synchronize()

    # Timed pass
    start = time.time()
    n_rendered = 0
    for cam in cams:
        _ = render(cam, gaussians, pipe, background)
        n_rendered += 1
    torch.cuda.synchronize()
    elapsed = time.time() - start

    fps = n_rendered / elapsed if elapsed > 0 else 0
    print(f"  Render speed: {fps:.2f} FPS ({n_rendered} frames in {elapsed:.2f}s)")

    with open(out_file, 'a') as f:
        f.write(f"Render FPS       : {fps:.2f}\n")

except Exception as e:
    print(f"  [WARN] Render speed measurement failed: {e}")
    with open(sys.argv[3], 'a') as f:
        f.write("Render FPS       : ERROR\n")
PYEOF
}

# ── Train one object ───────────────────────────────────────────────────────
train_object() {
    local DATA_PATH=$1
    local OUT_DIR=$2
    local LABEL=$3

    log ""
    log "  Training : ${LABEL}"
    log "  Data     : ${DATA_PATH}"
    log "  Output   : ${OUT_DIR}"

    # Skip if data doesn't exist
    if [ ! -d "${DATA_PATH}" ]; then
        log "  [SKIP] Dataset path not found: ${DATA_PATH}"
        return
    fi

    mkdir -p "${OUT_DIR}"

    # Run training
    cd "${SCRIPTS}"
    ${PYTHON} "${BENCHMARK_DEF}" \
        -s "${DATA_PATH}" \
        -m "${OUT_DIR}" \
        --gpu ${GPU} \
        --exp_runs 1 \
        2>&1 | tee -a "${LOG_ROOT}/${LABEL}.log" "${MAIN_LOG}"

    # Measure render speed and append to benchmark_result.log
    local RESULT_LOG="${OUT_DIR}/exp_run_1/benchmark_result.log"
    if [ -f "${RESULT_LOG}" ]; then
        measure_render_speed "${DATA_PATH}" "${OUT_DIR}" "${RESULT_LOG}"
    fi

    log "  Done: ${LABEL}"
}


# ══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════

log "======================================================"
log "  13_NO_DEFENSE -- Full Benchmark"
log "  Started : $(date)"
log "  GPU     : ${GPU}"
log "  Log     : ${MAIN_LOG}"
log "======================================================"

TOTAL_START=$(date +%s)

for ENTRY in "${DATASETS[@]}"; do
    IFS='|' read -r DPATH DNAME OBJTYPE <<< "${ENTRY}"

    # Pick the right object list
    if [ "${OBJTYPE}" == "MIP360" ]; then
        OBJECTS="${MIP360_OBJECTS}"
    else
        OBJECTS="${NERF_SYN_OBJECTS}"
    fi

    log ""
    log "======================================================"
    log "  Dataset: ${DNAME}"
    log "  Path   : ${DPATH}"
    log "  Objects: ${OBJECTS}"
    log "======================================================"

    # Check dataset exists
    if [ ! -d "${DPATH}" ]; then
        log "  [SKIP] Dataset not found: ${DPATH}"
        continue
    fi

    for OBJ in ${OBJECTS}; do
        OBJ_DATA="${DPATH}/${OBJ}"
        FOLDER_NAME="${DNAME}_${OBJ}"
        OUT_DIR="${LOG_ROOT}/${FOLDER_NAME}"

        # Skip if already completed
        if [ -f "${OUT_DIR}/exp_run_1/benchmark_result.log" ]; then
            log "  [SKIP] Already completed: ${FOLDER_NAME}"
            continue
        fi

        train_object "${OBJ_DATA}" "${OUT_DIR}" "${FOLDER_NAME}"
    done
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( (TOTAL_END - TOTAL_START) / 60 ))
log ""
log "  All training done in ~${TOTAL_ELAPSED} minutes."


# ══════════════════════════════════════════════════════════════════════════
# COLLECT RESULTS INTO CSV
# ══════════════════════════════════════════════════════════════════════════

log ""
log "======================================================"
log "  Collecting results into CSV + LaTeX"
log "======================================================"

RESULTS_CSV="${LOG_ROOT}/results_table.csv"
echo "dataset,object,max_gaussians_M,training_time_min,max_gpu_mem_MB,render_fps,SSIM,PSNR" > "${RESULTS_CSV}"

for ENTRY in "${DATASETS[@]}"; do
    IFS='|' read -r DPATH DNAME OBJTYPE <<< "${ENTRY}"

    if [ "${OBJTYPE}" == "MIP360" ]; then
        OBJECTS="${MIP360_OBJECTS}"
    else
        OBJECTS="${NERF_SYN_OBJECTS}"
    fi

    for OBJ in ${OBJECTS}; do
        FOLDER_NAME="${DNAME}_${OBJ}"
        LOG_FILE="${LOG_ROOT}/${FOLDER_NAME}/exp_run_1/benchmark_result.log"

        if [ ! -f "${LOG_FILE}" ]; then
            echo "${DNAME},${OBJ},,,,,," >> "${RESULTS_CSV}"
            continue
        fi

        MAX_GAUSS=$(grep -i "Max Gaussian" "${LOG_FILE}" | head -1 | grep -oP '[\d.]+(?=\s*M)' || echo "")
        TRAIN_TIME=$(grep -i "Training time" "${LOG_FILE}" | head -1 | grep -oP '[\d.]+(?=\s*min)' || echo "")
        GPU_MEM=$(grep -i "Max GPU mem" "${LOG_FILE}" | head -1 | grep -oP '\d+(?=\s*MB)' || echo "")
        RENDER_FPS=$(grep -i "Render FPS" "${LOG_FILE}" | head -1 | grep -oP '[\d.]+' || echo "")
        SSIM_VAL=$(grep -i "^SSIM" "${LOG_FILE}" | head -1 | grep -oP '[\d.]+' || echo "")
        PSNR_VAL=$(grep -i "^PSNR" "${LOG_FILE}" | head -1 | grep -oP '[\d.]+' || echo "")

        echo "${DNAME},${OBJ},${MAX_GAUSS},${TRAIN_TIME},${GPU_MEM},${RENDER_FPS},${SSIM_VAL},${PSNR_VAL}" >> "${RESULTS_CSV}"
        log "  ${FOLDER_NAME}: G=${MAX_GAUSS}M T=${TRAIN_TIME}m GPU=${GPU_MEM}MB FPS=${RENDER_FPS} SSIM=${SSIM_VAL} PSNR=${PSNR_VAL}"
    done
done

log "  CSV saved: ${RESULTS_CSV}"


# ══════════════════════════════════════════════════════════════════════════
# GENERATE LATEX TABLE
# ══════════════════════════════════════════════════════════════════════════

LATEX_FILE="${LOG_ROOT}/results_table.tex"

cat > "${LATEX_FILE}" << 'TEXHEADER'
% Auto-generated from 13_NO_DEFENSE benchmark.
% \input{results_table.tex}
% Requires: \usepackage{booktabs}

\begin{table*}[t]
\centering
\caption{No-defense baseline benchmark across all datasets and scenes. Gaussians = peak count during training (millions). Time = total training time (minutes). GPU = peak GPU memory (MB). FPS = render speed (frames per second). SSIM and PSNR are evaluated on training views.}
\label{tab:mvpi_full_results}
\scriptsize
\begin{tabular}{ll rrrrrr}
\toprule
\textbf{Dataset} & \textbf{Scene} & \textbf{Gaussians (M)} & \textbf{Time (min)} & \textbf{GPU (MB)} & \textbf{FPS} & \textbf{SSIM} & \textbf{PSNR} \\
\midrule
TEXHEADER

PREV_DNAME=""
while IFS=',' read -r DNAME OBJ GAUSS TIME GPU_M FPS SSIM_V PSNR_V; do
    # Skip header
    [ "${DNAME}" == "dataset" ] && continue

    # Escape underscores for LaTeX
    DNAME_TEX=$(echo "${DNAME}" | sed 's/_/\\_/g')

    # Add midrule between datasets
    if [ -n "${PREV_DNAME}" ] && [ "${PREV_DNAME}" != "${DNAME}" ]; then
        echo "\\midrule" >> "${LATEX_FILE}"
    fi
    PREV_DNAME="${DNAME}"

    # Use — for missing values
    [ -z "${GAUSS}" ] && GAUSS="—"
    [ -z "${TIME}" ] && TIME="—"
    [ -z "${GPU_M}" ] && GPU_M="—"
    [ -z "${FPS}" ] && FPS="—"
    [ -z "${SSIM_V}" ] && SSIM_V="—"
    [ -z "${PSNR_V}" ] && PSNR_V="—"

    echo "${DNAME_TEX} & ${OBJ} & ${GAUSS} & ${TIME} & ${GPU_M} & ${FPS} & ${SSIM_V} & ${PSNR_V} \\\\" >> "${LATEX_FILE}"

done < "${RESULTS_CSV}"

cat >> "${LATEX_FILE}" << 'TEXFOOTER'
\bottomrule
\end{tabular}
\end{table*}
TEXFOOTER

log "  LaTeX saved: ${LATEX_FILE}"
log ""
log "======================================================"
log "  All done. Finished : $(date)"
log "  Total wall time    : ~${TOTAL_ELAPSED} minutes"
log ""
log "  Results:"
log "    CSV   : ${RESULTS_CSV}"
log "    LaTeX : ${LATEX_FILE}"
log "    Logs  : ${LOG_ROOT}/<dataset>_<object>/exp_run_1/"
log "======================================================"