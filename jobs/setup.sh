#!/bin/bash -l
set -euo pipefail

: "${HPCVAULT:?Set HPCVAULT to the base persistent storage directory}"

REPO_NAME=TinyRecursiveModels
REPO_DIR="${HOME}/${REPO_NAME}"
SIF_DIR="${HPCVAULT}/${REPO_NAME}/containers"
SIF_PATH="${SIF_DIR}/pytorch.sif"
DATA_ROOT="${HPCVAULT}/${REPO_NAME}/data"
OVERLAY_DIR="${HPCVAULT}/${REPO_NAME}/overlays"
OVERLAY_PATH="${OVERLAY_DIR}/python-overlay.ext3"
OVERLAY_SIZE_MB="${OVERLAY_SIZE_MB:-16384}"
PYTHON_USER_BASE="${HPCVAULT}/${REPO_NAME}/python"
PIP_CACHE_DIR="${HPCVAULT}/${REPO_NAME}/pip-cache"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "[setup] Expected repository checkout at ${REPO_DIR}" >&2
  exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "[setup] Apptainer command not found" >&2
  exit 1
fi

mkdir -p "${SIF_DIR}" "${DATA_ROOT}" "${OVERLAY_DIR}" \
         "${HPCVAULT}/${REPO_NAME}/job_logs" \
         "${HPCVAULT}/${REPO_NAME}/checkpoints" "${HPCVAULT}/${REPO_NAME}/hydra" \
         "${PYTHON_USER_BASE}" "${PIP_CACHE_DIR}"

cd "${REPO_DIR}"

if [[ "${FORCE_OVERLAY_REFRESH:-0}" == "1" ]] && [[ -f "${OVERLAY_PATH}" ]]; then
  rm -f "${OVERLAY_PATH}"
fi

if [[ ! -f "${OVERLAY_PATH}" ]]; then
  echo "[setup] Creating Apptainer overlay at ${OVERLAY_PATH}" >&2
  apptainer overlay create --size "${OVERLAY_SIZE_MB}" "${OVERLAY_PATH}"
else
  echo "[setup] Reusing Apptainer overlay at ${OVERLAY_PATH}" >&2
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

COMMON_APPTAINER_ARGS_BASE=(
  --cleanenv
  --bind "${HPCVAULT}:${HPCVAULT}","${REPO_DIR}:${REPO_DIR}"
  --overlay "${OVERLAY_PATH}"
  --pwd "${REPO_DIR}"
  --env-file hpcvault.env
  --env http_proxy=http://proxy:80
  --env https_proxy=http://proxy:80
  --env PYTHONUSERBASE="${PYTHON_USER_BASE}"
  --env PIP_CACHE_DIR="${PIP_CACHE_DIR}"
  --env PIP_DISABLE_PIP_VERSION_CHECK=1
)

PYTHON_USER_SITE=$(apptainer exec "${COMMON_APPTAINER_ARGS_BASE[@]}" "${SIF_PATH}" \
  python -c 'import site, sys; sys.stdout.write(site.getusersitepackages())')

if [[ -z "${PYTHON_USER_SITE}" ]]; then
  echo "[setup] Failed to resolve python user site directory" >&2
  exit 1
fi

COMMON_APPTAINER_ARGS=("${COMMON_APPTAINER_ARGS_BASE[@]}" --env "PYTHONPATH=${PYTHON_USER_SITE}")

PIP_EXEC=(apptainer exec "${COMMON_APPTAINER_ARGS[@]}" "${SIF_PATH}")

"${PIP_EXEC[@]}" python -m pip install --user --upgrade pip wheel setuptools
"${PIP_EXEC[@]}" python -m pip install --user -r requirements.txt
"${PIP_EXEC[@]}" python -c "import adam_atan2_pytorch" >/dev/null

apptainer exec --nv "${COMMON_APPTAINER_ARGS[@]}" \
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
