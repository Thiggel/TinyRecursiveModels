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

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FLASH_ATTENTION_DETERMINISTIC=0

if [[ ! -d .venv ]] || [[ "${FORCE_VENV_REFRESH:-0}" == "1" ]]; then
  ./scripts/setup_venv.sh
fi

source .venv/bin/activate

rm -rf "/home/vault/c107fa/c107fa12/TinyRecursiveModels/stored_tokens"
