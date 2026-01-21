#!/bin/bash

# Source this file, then call: run_apptainer <command...>
# Example: run_apptainer python models/recursive_reasoning/mhc/mhc.py

run_apptainer() {
  if [[ $# -lt 1 ]]; then
    echo "usage: run_apptainer <command...>" >&2
    return 2
  fi

  if ! command -v apptainer >/dev/null 2>&1; then
    echo "[run_apptainer] Apptainer command not found" >&2
    return 1
  fi

  local repo_name repo_dir
  repo_name="${REPO_NAME:-TinyRecursiveModels}"
  if [[ -n "${REPO_DIR:-}" ]]; then
    repo_dir="${REPO_DIR}"
  else
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi

  local work_root scratch_root
  if [[ -n "${WORK:-}" ]]; then
    work_root="${WORK}/${repo_name}"
  else
    work_root="${repo_dir}"
  fi

  if [[ -n "${SCRATCH:-}" ]]; then
    scratch_root="${SCRATCH}/${repo_name}"
  else
    scratch_root="${work_root}"
  fi

  local sif_path
  sif_path="${SIF_PATH:-${work_root}/containers/pytorch.sif}"
  if [[ ! -f "${sif_path}" ]]; then
    echo "[run_apptainer] Missing Apptainer image at ${sif_path}. Run jobs/setup.sh first or set SIF_PATH." >&2
    return 1
  fi

  local python_user_base python_bin
  python_user_base="${PYTHONUSERBASE:-${work_root}/python}"
  python_bin="${PYTHON_BIN:-python3.10}"

  local base_cache_dir
  base_cache_dir="${BASE_CACHE_DIR:-${scratch_root}}"
  local hf_home hf_datasets_cache hf_modules_cache transformers_cache
  local deepspeed_cache_dir triton_cache_dir wandb_dir torch_home xdg_cache_home
  local pytorch_lightning_home hydra_base_dir checkpoint_root job_log_dir
  local pip_cache_dir tmpdir data_root

  hf_home="${HF_HOME:-${base_cache_dir}/hf}"
  hf_datasets_cache="${HF_DATASETS_CACHE:-${base_cache_dir}/datasets}"
  hf_modules_cache="${HF_MODULES_CACHE:-${base_cache_dir}/modules}"
  transformers_cache="${TRANSFORMERS_CACHE:-${base_cache_dir}/transformers}"
  deepspeed_cache_dir="${DEEPSPEED_CACHE_DIR:-${base_cache_dir}/deepspeed}"
  triton_cache_dir="${TRITON_CACHE_DIR:-${base_cache_dir}/triton}"
  wandb_dir="${WANDB_DIR:-${base_cache_dir}/wandb}"
  torch_home="${TORCH_HOME:-${base_cache_dir}/torch}"
  xdg_cache_home="${XDG_CACHE_HOME:-${base_cache_dir}/.cache}"
  pytorch_lightning_home="${PYTORCH_LIGHTNING_HOME:-${base_cache_dir}/lightning_logs}"
  hydra_base_dir="${HYDRA_BASE_DIR:-${work_root}/hydra}"
  checkpoint_root="${CHECKPOINT_ROOT:-${work_root}/checkpoints}"
  job_log_dir="${JOB_LOG_DIR:-job_logs}"
  pip_cache_dir="${PIP_CACHE_DIR:-${scratch_root}/pip-cache}"
  data_root="${DATA_ROOT:-${work_root}/data}"
  tmpdir="${TMPDIR:-${base_cache_dir}/.tmp}"

  mkdir -p "${python_user_base}" "${pip_cache_dir}" "${hf_home}" \
           "${hf_datasets_cache}" "${hf_modules_cache}" "${transformers_cache}" \
           "${deepspeed_cache_dir}" "${triton_cache_dir}" "${wandb_dir}" \
           "${torch_home}" "${xdg_cache_home}" "${pytorch_lightning_home}" \
           "${hydra_base_dir}" "${checkpoint_root}" "${tmpdir}"

  local http_proxy_env https_proxy_env
  http_proxy_env="${http_proxy:-http://proxy:80}"
  https_proxy_env="${https_proxy:-http://proxy:80}"

  local common_args_base
  common_args_base=(
    --cleanenv
    --bind "${work_root}:${work_root}"
    --bind "${scratch_root}:${scratch_root}"
    --bind "${repo_dir}:${repo_dir}"
    --bind "${python_user_base}:${python_user_base}"
    --pwd "${repo_dir}"
    --env "http_proxy=${http_proxy_env}"
    --env "https_proxy=${https_proxy_env}"
    --env "PYTHONUSERBASE=${python_user_base}"
    --env "PIP_CACHE_DIR=${pip_cache_dir}"
    --env PIP_DISABLE_PIP_VERSION_CHECK=1
    --env "BASE_CACHE_DIR=${base_cache_dir}"
    --env "HF_HOME=${hf_home}"
    --env "HF_DATASETS_CACHE=${hf_datasets_cache}"
    --env "HF_MODULES_CACHE=${hf_modules_cache}"
    --env "TRANSFORMERS_CACHE=${transformers_cache}"
    --env "DEEPSPEED_CACHE_DIR=${deepspeed_cache_dir}"
    --env "TRITON_CACHE_DIR=${triton_cache_dir}"
    --env "WANDB_DIR=${wandb_dir}"
    --env "TORCH_HOME=${torch_home}"
    --env "XDG_CACHE_HOME=${xdg_cache_home}"
    --env "PYTORCH_LIGHTNING_HOME=${pytorch_lightning_home}"
    --env "HYDRA_BASE_DIR=${hydra_base_dir}"
    --env "CHECKPOINT_ROOT=${checkpoint_root}"
    --env "JOB_LOG_DIR=${job_log_dir}"
    --env "DATA_ROOT=${data_root}"
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8
    --env FLASH_ATTENTION_DETERMINISTIC=0
    --env "TMPDIR=${tmpdir}"
  )

  if [[ -n "${CUDA_HOME:-}" ]]; then
    common_args_base+=(
      --bind "${CUDA_HOME}:${CUDA_HOME}"
      --env "CUDA_HOME=${CUDA_HOME}"
      --env "CUDACXX=${CUDA_HOME}/bin/nvcc"
      --env "PATH=${CUDA_HOME}/bin:\$PATH"
      --env "LD_LIBRARY_PATH=${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH"
    )
  fi

  local python_user_site
  python_user_site=$(apptainer exec "${common_args_base[@]}" "${sif_path}" \
    "${python_bin}" -c 'import site, sys; sys.stdout.write(site.getusersitepackages())')

  if [[ -z "${python_user_site}" ]]; then
    echo "[run_apptainer] Failed to resolve python user site directory" >&2
    return 1
  fi

  local common_args
  common_args=("${common_args_base[@]}" --env "PYTHONPATH=${python_user_site}")

  apptainer exec --nv "${common_args[@]}" "${sif_path}" "$@"
}
