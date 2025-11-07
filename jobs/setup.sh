#!/bin/bash -l
set -euo pipefail

: "${HPCVAULT:?Set HPCVAULT to the base persistent storage directory}"

REPO_NAME=TinyRecursiveModels
REPO_DIR="${HOME}/${REPO_NAME}"
SIF_DIR="${HPCVAULT}/${REPO_NAME}/containers"
SIF_PATH="${SIF_DIR}/pytorch.sif"
DATA_ROOT="${HPCVAULT}/${REPO_NAME}/data"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "[setup] Expected repository checkout at ${REPO_DIR}" >&2
  exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "[setup] Apptainer command not found" >&2
  exit 1
fi

mkdir -p "${SIF_DIR}" "${DATA_ROOT}" "${HPCVAULT}/${REPO_NAME}/job_logs" \
         "${HPCVAULT}/${REPO_NAME}/checkpoints" "${HPCVAULT}/${REPO_NAME}/hydra"

cd "${REPO_DIR}"

if [[ ! -d "${REPO_DIR}/.venv" ]] || [[ "${FORCE_VENV_REFRESH:-0}" == "1" ]]; then
  bash "${REPO_DIR}/scripts/setup_venv.sh"
fi

FORCE_REBUILD=0
if [[ "${1:-}" == "--force-build" ]]; then
  FORCE_REBUILD=1
fi

BUILD_IMAGE=0
if [[ ! -f "${SIF_PATH}" ]]; then
  BUILD_IMAGE=1
elif [[ ${FORCE_REBUILD} -eq 1 ]]; then
  BUILD_IMAGE=1
fi

if [[ ${BUILD_IMAGE} -eq 1 ]]; then
  echo "[setup] Building Apptainer image at ${SIF_PATH}" >&2
  BUILD_ARGS=()
  if [[ "${APPTAINER_NO_FAKEROOT:-0}" != "1" ]]; then
    BUILD_ARGS+=(--fakeroot)
  fi
  if [[ ${FORCE_REBUILD} -eq 1 ]]; then
    BUILD_ARGS+=(--force)
  fi
  apptainer build "${BUILD_ARGS[@]}" "${SIF_PATH}" apptainer/pytorch.def
else
  echo "[setup] Reusing existing Apptainer image at ${SIF_PATH}" >&2
fi

ARC_INPUT_PREFIX="${REPO_DIR}/kaggle/combined/arc-agi"

apptainer exec --nv --cleanenv \
  --bind "${HPCVAULT}:${HPCVAULT}","${REPO_DIR}:${REPO_DIR}" \
  --pwd "${REPO_DIR}" \
  --env-file hpcvault.env \
  --env PYTHONNOUSERSITE=1 \
  --env http_proxy="http://proxy:80" --env https_proxy="http://proxy:80" \
  --env PYTHONPATH= \
  --env DATA_ROOT="${DATA_ROOT}" \
  --env ARC_INPUT_PREFIX="${ARC_INPUT_PREFIX}" \
  "${SIF_PATH}" bash -lc "
    set -euo pipefail

    mkdir -p \"\${DATA_ROOT}/arc1concept-aug-1000\" \\
             \"\${DATA_ROOT}/arc2concept-aug-1000\" \\
             \"\${DATA_ROOT}/sudoku-extreme-1k-aug-1000\" \\
             \"\${DATA_ROOT}/maze-30x30-hard-1k\"

    python -m dataset.build_arc_dataset \\
      --input-file-prefix \"\${ARC_INPUT_PREFIX}\" \\
      --output-dir \"\${DATA_ROOT}/arc1concept-aug-1000\" \\
      --subsets training evaluation concept \\
      --test-set-name evaluation

    python -m dataset.build_arc_dataset \\
      --input-file-prefix \"\${ARC_INPUT_PREFIX}\" \\
      --output-dir \"\${DATA_ROOT}/arc2concept-aug-1000\" \\
      --subsets training2 evaluation2 concept \\
      --test-set-name evaluation2

    python dataset/build_sudoku_dataset.py \\
      --output-dir \"\${DATA_ROOT}/sudoku-extreme-1k-aug-1000\" \\
      --subsample-size 1000 \\
      --num-aug 1000

    python dataset/build_maze_dataset.py \\
      --output-dir \"\${DATA_ROOT}/maze-30x30-hard-1k\"
  "
