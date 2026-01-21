#!/bin/bash -l
set -euo pipefail

: "${WORK:?Set WORK to the base work directory}"
: "${SCRATCH:?Set SCRATCH to the base scratch directory}"

REPO_NAME=TinyRecursiveModels
REPO_DIR="${HOME}/${REPO_NAME}"
WORK_ROOT="${WORK}/${REPO_NAME}"
SCRATCH_ROOT="${SCRATCH}/${REPO_NAME}"

SIF_DIR="${WORK_ROOT}/containers"
SIF_PATH="${SIF_DIR}/pytorch.sif"
DATA_ROOT="${WORK_ROOT}/data"
PYTHON_USER_BASE="${PYTHONUSERBASE:-${WORK_ROOT}/python}"
PYTHON_BIN=python3.10

BASE_CACHE_DIR="${SCRATCH_ROOT}"
HF_HOME="${BASE_CACHE_DIR}/hf"
HF_DATASETS_CACHE="${BASE_CACHE_DIR}/datasets"
HF_MODULES_CACHE="${BASE_CACHE_DIR}/modules"
TRANSFORMERS_CACHE="${BASE_CACHE_DIR}/transformers"
DEEPSPEED_CACHE_DIR="${BASE_CACHE_DIR}/deepspeed"
TRITON_CACHE_DIR="${BASE_CACHE_DIR}/triton"
WANDB_DIR="${BASE_CACHE_DIR}/wandb"
TORCH_HOME="${BASE_CACHE_DIR}/torch"
XDG_CACHE_HOME="${BASE_CACHE_DIR}/.cache"
PYTORCH_LIGHTNING_HOME="${BASE_CACHE_DIR}/lightning_logs"
HYDRA_BASE_DIR="${WORK_ROOT}/hydra"
CHECKPOINT_ROOT="${WORK_ROOT}/checkpoints"
JOB_LOG_DIR="${WORK_ROOT}/job_logs"
PIP_CACHE_DIR="${BASE_CACHE_DIR}/pip-cache"
TMPDIR="${SLURM_TMPDIR:-${BASE_CACHE_DIR}/.tmp}"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "[setup] Expected repository checkout at ${REPO_DIR}" >&2
  exit 1
fi

if ! command -v apptainer >/dev/null 2>&1; then
  echo "[setup] Apptainer command not found" >&2
  exit 1
fi

mkdir -p "${SIF_DIR}" "${DATA_ROOT}" \
         "${JOB_LOG_DIR}" "${CHECKPOINT_ROOT}" "${HYDRA_BASE_DIR}" \
         "${PYTHON_USER_BASE}" "${PIP_CACHE_DIR}" "${HF_HOME}" \
         "${HF_DATASETS_CACHE}" "${HF_MODULES_CACHE}" "${TRANSFORMERS_CACHE}" \
         "${DEEPSPEED_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
         "${WANDB_DIR}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" \
         "${PYTORCH_LIGHTNING_HOME}"

if [[ -z "${SLURM_TMPDIR:-}" ]]; then
  mkdir -p "${TMPDIR}"
fi

cd "${REPO_DIR}"


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
  --bind "${WORK}:${WORK}"
  --bind "${SCRATCH}:${SCRATCH}"
  --bind "${REPO_DIR}:${REPO_DIR}"
  --bind "${PYTHON_USER_BASE}:${PYTHON_USER_BASE}"
  --pwd "${REPO_DIR}"
  --env http_proxy=http://proxy:80
  --env https_proxy=http://proxy:80
  --env PYTHONUSERBASE="${PYTHON_USER_BASE}"
  --env PIP_CACHE_DIR="${PIP_CACHE_DIR}"
  --env PIP_DISABLE_PIP_VERSION_CHECK=1
  --env BASE_CACHE_DIR="${BASE_CACHE_DIR}"
  --env HF_HOME="${HF_HOME}"
  --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}"
  --env HF_MODULES_CACHE="${HF_MODULES_CACHE}"
  --env TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
  --env DEEPSPEED_CACHE_DIR="${DEEPSPEED_CACHE_DIR}"
  --env TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
  --env WANDB_DIR="${WANDB_DIR}"
  --env TORCH_HOME="${TORCH_HOME}"
  --env XDG_CACHE_HOME="${XDG_CACHE_HOME}"
  --env PYTORCH_LIGHTNING_HOME="${PYTORCH_LIGHTNING_HOME}"
  --env HYDRA_BASE_DIR="${HYDRA_BASE_DIR}"
  --env CHECKPOINT_ROOT="${CHECKPOINT_ROOT}"
  --env JOB_LOG_DIR="${JOB_LOG_DIR}"
  --env DATA_ROOT="${DATA_ROOT}"
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8
  --env FLASH_ATTENTION_DETERMINISTIC=0
  --env TMPDIR="${TMPDIR}"
)

if [[ -n "${CUDA_HOME:-}" ]]; then
  COMMON_APPTAINER_ARGS_BASE+=(
    --bind "${CUDA_HOME}:${CUDA_HOME}"
    --env CUDA_HOME="${CUDA_HOME}"
    --env CUDACXX="${CUDA_HOME}/bin/nvcc"
    --env PATH="${CUDA_HOME}/bin:\$PATH"
    --env LD_LIBRARY_PATH="${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH"
  )
fi

PYTHON_USER_SITE=$(apptainer exec "${COMMON_APPTAINER_ARGS_BASE[@]}" "${SIF_PATH}" \
  "${PYTHON_BIN}" -c 'import site, sys; sys.stdout.write(site.getusersitepackages())')

if [[ -z "${PYTHON_USER_SITE}" ]]; then
  echo "[setup] Failed to resolve python user site directory" >&2
  exit 1
fi

COMMON_APPTAINER_ARGS=("${COMMON_APPTAINER_ARGS_BASE[@]}" --env "PYTHONPATH=${PYTHON_USER_SITE}")

PIP_EXEC=(apptainer exec "${COMMON_APPTAINER_ARGS[@]}" "${SIF_PATH}")

"${PIP_EXEC[@]}" "${PYTHON_BIN}" -m pip install --user --upgrade pip wheel setuptools
"${PIP_EXEC[@]}" "${PYTHON_BIN}" -m pip install --user -r requirements.txt
"${PIP_EXEC[@]}" "${PYTHON_BIN}" -c "import adam_atan2_pytorch" >/dev/null


apptainer exec --nv "${COMMON_APPTAINER_ARGS[@]}" \
  --env DATA_ROOT="${DATA_ROOT}" \
  --env ARC_INPUT_PREFIX="${ARC_INPUT_PREFIX}" \
  "${SIF_PATH}" bash -lc "
    set -euo pipefail

    mkdir -p \"\${DATA_ROOT}/arc1concept-aug-1000\" \\
             \"\${DATA_ROOT}/arc2concept-aug-1000\" \\
             \"\${DATA_ROOT}/sudoku-extreme-1k-aug-1000\" \\
             \"\${DATA_ROOT}/maze-30x30-hard-1k\"

    python3.10 -m dataset.build_arc_dataset \\
      --input-file-prefix \"\${ARC_INPUT_PREFIX}\" \\
      --output-dir \"\${DATA_ROOT}/arc1concept-aug-1000\" \\
      --subsets training evaluation concept \\
      --test-set-name evaluation

    python3.10 -m dataset.build_arc_dataset \\
      --input-file-prefix \"\${ARC_INPUT_PREFIX}\" \\
      --output-dir \"\${DATA_ROOT}/arc2concept-aug-1000\" \\
      --subsets training2 evaluation2 concept \\
      --test-set-name evaluation2

    python3.10 dataset/build_sudoku_dataset.py \\
      --output-dir \"\${DATA_ROOT}/sudoku-extreme-1k-aug-1000\" \\
      --subsample-size 1000 \\
      --num-aug 1000

    python3.10 dataset/build_maze_dataset.py \\
      --output-dir \"\${DATA_ROOT}/maze-30x30-hard-1k\"
  "
