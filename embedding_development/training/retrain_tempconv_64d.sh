#!/usr/bin/env bash
# retrain_tempconv_64d.sh
#
# Retrains TempConv-Cont and TempConv-Pred at embed_dim=64 for seeds 42-46.
# Outputs → models/cebra_64d/  and  models/cebra_pred_64d/
#
# Resume behaviour:
#   • Completed runs: marked by a .done file → skipped on restart.
#   • Interrupted mid-session: the Python script saves a per-session
#     .resume_session_XX.pt checkpoint → picked up automatically.
#   • Interrupted mid-seed-run: re-run the script; completed sessions are
#     skipped by the Python script's own skip logic.
#
# Usage (from embedding_development/):
#   bash training/retrain_tempconv_64d.sh                     # all seeds
#   bash training/retrain_tempconv_64d.sh 44 46               # specific seeds only
#   bash training/retrain_tempconv_64d.sh 2>&1 | tee logs/retrain_64d_master.log

set -euo pipefail

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"          # embedding_development/
LOG_DIR="${ROOT}/logs/retrain_64d"
DONE_DIR="${ROOT}/logs/retrain_64d/.done"

mkdir -p "${LOG_DIR}" "${DONE_DIR}"
cd "${ROOT}"                                    # scripts load ./outputs relative to here

# Seeds can be overridden via positional args: bash retrain_tempconv_64d.sh 44 46
if [[ $# -gt 0 ]]; then
    SEEDS=("$@")
else
    SEEDS=(42 43 44 45 46)
fi
EMBED_DIM=64
NUM_UNITS=32

# ── helper: run one training job ──────────────────────────────────────────────
run_job() {
    local arch="$1"          # "cont" or "pred"
    local seed="$2"
    local done_flag="${DONE_DIR}/${arch}_seed${seed}.done"

    if [[ -f "${done_flag}" ]]; then
        echo "[SKIP]  ${arch} seed=${seed} — already complete (${done_flag})"
        return 0
    fi

    local script models_dir log_file
    if [[ "${arch}" == "cont" ]]; then
        script="training/train_cebra.py"
        models_dir="models/cebra_64d/ensembles/seed${seed}"
    else
        script="training/train_cebra_non_contrastive.py"
        models_dir="models/cebra_pred_64d/ensembles/seed${seed}"
    fi
    log_file="${LOG_DIR}/${arch}_seed${seed}.log"
    local split_path="splits/split_seed${seed}.npy"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[START]  arch=${arch}  seed=${seed}  embed_dim=${EMBED_DIM}"
    echo "         models_dir=${models_dir}"
    echo "         log=${log_file}"
    echo "         $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local t0
    t0=$(date +%s)

    conda run -n analysisVR python3 "${script}" \
        --use_ensembles \
        --seed "${seed}" \
        --embed_dim "${EMBED_DIM}" \
        --num_units "${NUM_UNITS}" \
        --split_path "${split_path}" \
        --models_dir "${models_dir}" \
        2>&1 | tee "${log_file}"

    # tee exits 0 even if the python command fails through the pipe.
    # Use PIPESTATUS to catch the real exit code.
    local py_exit="${PIPESTATUS[0]}"
    local t1
    t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))

    if [[ "${py_exit}" -ne 0 ]]; then
        echo "[FAIL]   arch=${arch} seed=${seed} — exit ${py_exit}  (${mins}m${secs}s)"
        echo "[FAIL]   arch=${arch} seed=${seed} exit=${py_exit} elapsed=${elapsed}s  $(date '+%Y-%m-%d %H:%M:%S')" \
            >> "${LOG_DIR}/timing.log"
        return "${py_exit}"
    fi

    echo "[DONE]   arch=${arch} seed=${seed}  (${mins}m${secs}s)"
    echo "${arch} seed=${seed} elapsed=${elapsed}s (${mins}m${secs}s)  $(date '+%Y-%m-%d %H:%M:%S')" \
        >> "${LOG_DIR}/timing.log"

    touch "${done_flag}"
}

# ── main loop ─────────────────────────────────────────────────────────────────
echo "Retraining TempConv at embed_dim=${EMBED_DIM}"
echo "Seeds: ${SEEDS[*]}"
echo "Logs: ${LOG_DIR}/"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

TOTAL_START=$(date +%s)
FAILED=0

for seed in "${SEEDS[@]}"; do
    for arch in cont pred; do
        run_job "${arch}" "${seed}" || FAILED=$(( FAILED + 1 ))
    done
done

TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total wall time: ${TOTAL_MINS}m${TOTAL_SECS}s"
echo "Failed runs: ${FAILED}"
echo "Timing summary:"
cat "${LOG_DIR}/timing.log" 2>/dev/null || echo "  (none)"
echo "══════════════════════════════════════════════════════════════════"

[[ "${FAILED}" -eq 0 ]]
