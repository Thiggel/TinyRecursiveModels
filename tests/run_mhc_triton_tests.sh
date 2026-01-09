#!/bin/bash -l

set -euo pipefail

: "${WORK:?Set WORK to the base work directory}"
: "${SCRATCH:?Set SCRATCH to the base scratch directory}"

REPO_NAME=TinyRecursiveModels
REPO_DIR="${HOME}/${REPO_NAME}"
WORK_ROOT="${WORK}/${REPO_NAME}"
SCRATCH_ROOT="${SCRATCH}/${REPO_NAME}"
SIF_PATH="${WORK_ROOT}/containers/pytorch.sif"

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
PYTHON_USER_BASE="${WORK_ROOT}/python"
PIP_CACHE_DIR="${BASE_CACHE_DIR}/pip-cache"
TMPDIR="${SLURM_TMPDIR:-${BASE_CACHE_DIR}/.tmp}"

if [[ ! -f "${SIF_PATH}" ]]; then
  echo "[mhc-triton-tests] Missing Apptainer image at ${SIF_PATH}. Run jobs/setup.sh first." >&2
  exit 1
fi

mkdir -p "${PIP_CACHE_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" \
         "${HF_MODULES_CACHE}" "${TRANSFORMERS_CACHE}" \
         "${DEEPSPEED_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
         "${WANDB_DIR}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" \
         "${PYTORCH_LIGHTNING_HOME}"

if [[ -z "${SLURM_TMPDIR:-}" ]]; then
  mkdir -p "${TMPDIR}"
fi

COMMON_APPTAINER_ARGS_BASE=(
  --cleanenv
  --bind "${WORK}:${WORK}"
  --bind "${SCRATCH}:${SCRATCH}"
  --bind "${REPO_DIR}:${REPO_DIR}"
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
  --env HYDRA_FULL_ERROR=1
  --env TMPDIR="${TMPDIR}"
)

PYTHON_USER_SITE=$(apptainer exec "${COMMON_APPTAINER_ARGS_BASE[@]}" "${SIF_PATH}" \
  python3.10 -c 'import site, sys; sys.stdout.write(site.getusersitepackages())')

if [[ -z "${PYTHON_USER_SITE}" ]]; then
  echo "[mhc-triton-tests] Failed to resolve python user site directory" >&2
  exit 1
fi

COMMON_APPTAINER_ARGS=("${COMMON_APPTAINER_ARGS_BASE[@]}" --env "PYTHONPATH=${PYTHON_USER_SITE}")

cd "${REPO_DIR}"

apptainer exec --nv "${COMMON_APPTAINER_ARGS[@]}" \
  "${SIF_PATH}" bash -lc "
    set -euo pipefail
    export PYTHONUSERBASE=\"${PYTHON_USER_BASE}\"
    export PIP_USER=1
    if ! python3.10 -c \"import pytest\" >/dev/null 2>&1; then
      if ! python3.10 -m pip install --user pytest; then
        TARGET_DIR=\"${PYTHON_USER_BASE}/lib/python3.10/site-packages\"
        mkdir -p \"\${TARGET_DIR}\"
        python3.10 -m pip install --target \"\${TARGET_DIR}\" pytest
      fi
    fi
    python3.10 -m pytest -q tests/models/recursive_reasoning/test_mhc_triton.py
  "
