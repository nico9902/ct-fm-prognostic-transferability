#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"

# The Rad-JEPA-3D encoder depends on mamba-ssm/causal-conv1d CUDA extensions.
# Keep PyTorch on a CUDA runtime supported by the installed NVIDIA driver.
# CUDA 12.1 wheels run on newer 12.x drivers and have broad third-party wheel
# support, which makes this safer than accidentally installing cu130 wheels.
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

CAUSAL_CONV1D_VERSION="${CAUSAL_CONV1D_VERSION:-1.5.0.post8}"
MAMBA_SSM_VERSION="${MAMBA_SSM_VERSION:-2.2.4}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.46.3}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"

"${PYTHON}" -m pip install --upgrade pip setuptools wheel ninja packaging

"${PYTHON}" -m pip uninstall -y torch torchvision torchaudio mamba-ssm causal-conv1d transformers || true

"${PYTHON}" -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"

"${PYTHON}" -m pip install \
  "transformers==${TRANSFORMERS_VERSION}"

"${PYTHON}" -m pip install \
  "causal-conv1d==${CAUSAL_CONV1D_VERSION}" \
  "mamba-ssm==${MAMBA_SSM_VERSION}" \
  --no-build-isolation

if [[ -f "${REQUIREMENTS_FILE}" ]]; then
  "${PYTHON}" -m pip install -r "${REQUIREMENTS_FILE}"
  # Re-apply the compatible Transformers pin in case the requirements file
  # is changed to request a newer 5.x release incompatible with mamba-ssm 2.2.4.
  "${PYTHON}" -m pip install --upgrade --force-reinstall \
    "transformers==${TRANSFORMERS_VERSION}"
fi

"${PYTHON}" - <<'PY'
import torch
import transformers
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
import causal_conv1d
import mamba_ssm
print("causal_conv1d ok")
print("mamba_ssm ok")
PY
