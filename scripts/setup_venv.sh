#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${PROJECT_ROOT}/.venv"

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "[setup_venv] Creating virtual environment at ${VENV_PATH}" >&2
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
else
  echo "[setup_venv] Using existing virtual environment at ${VENV_PATH}" >&2
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

if [[ "${INSTALL_TORCH:-1}" == "1" ]]; then
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu126}"
  echo "[setup_venv] Installing PyTorch from ${TORCH_INDEX_URL}" >&2
  python -m pip install --pre --upgrade torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
fi

python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
python -m pip install --no-cache-dir --no-build-isolation adam-atan2

echo "[setup_venv] Virtual environment ready." >&2
