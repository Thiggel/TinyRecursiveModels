unset SLURM_EXPORT_ENV

module purge
module load python/3.9-anaconda
module load gcc/12.1.0
module load cuda/12.1.1


export SCRATCH_LOCAL="$HPCVAULT"
export REPO_NAME="TinyRecursiveModels"

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

cd $SCRATCH_LOCAL
mkdir $REPO_NAME

cd $HOME/$REPO_NAME

# Base directory
export BASE_CACHE_DIR="$SCRATCH_LOCAL/$REPO_NAME"

# Hugging Face
export HF_HOME="$BASE_CACHE_DIR"
export HF_DATASETS_CACHE="$BASE_CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$BASE_CACHE_DIR/transformers"
export HF_MODULES_CACHE="$BASE_CACHE_DIR/modules"

# DeepSpeed
export DEEPSPEED_CACHE_DIR="$BASE_CACHE_DIR/deepspeed"
export TRITON_CACHE_DIR="$BASE_CACHE_DIR/triton"

# Weights & Biases
export WANDB_DIR="$BASE_CACHE_DIR/wandb"

# PyTorch Lightning
# Note: PyTorch Lightning doesn't use an environment variable,
# but you can use this in your Python code
export PYTORCH_LIGHTNING_HOME="$BASE_CACHE_DIR/lightning_logs"

export PYTHONUSERBASE="$BASE_CACHE_DIR/python"
export PYTHONPATH="${PYTHONUSERBASE}/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export PIP_CACHE_DIR="$BASE_CACHE_DIR/pip-cache"
export PIP_DISABLE_PIP_VERSION_CHECK=1

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=0

OVERLAY_DIR="${SCRATCH_LOCAL}/${REPO_NAME}/overlays"
OVERLAY_PATH="${OVERLAY_DIR}/python-overlay.ext3"
OVERLAY_SIZE_MB="${OVERLAY_SIZE_MB:-16384}"

mkdir -p "${OVERLAY_DIR}"

if command -v apptainer >/dev/null 2>&1; then
  if [[ ! -f "${OVERLAY_PATH}" ]] || [[ "${FORCE_OVERLAY_REFRESH:-0}" == "1" ]]; then
    if [[ "${FORCE_OVERLAY_REFRESH:-0}" == "1" ]] && [[ -f "${OVERLAY_PATH}" ]]; then
      rm -f "${OVERLAY_PATH}"
    fi
    echo "[environment] Creating Apptainer overlay at ${OVERLAY_PATH}" >&2
    apptainer overlay create --size "${OVERLAY_SIZE_MB}" "${OVERLAY_PATH}"
  else
    echo "[environment] Using existing Apptainer overlay at ${OVERLAY_PATH}" >&2
  fi
else
  echo "[environment] Apptainer not found on PATH; skipping overlay creation." >&2
fi

export TRM_OVERLAY_PATH="${OVERLAY_PATH}"

rm -rf "/home/vault/c107fa/c107fa12/TinyRecursiveModels/stored_tokens"
