#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Keural Mid-Level Vision Encoder — Virtual Environment Setup
# Target: 2× H200 141GB, CUDA 12.8, Python 3.12
#
# Usage:
#   chmod +x setup_venv.sh
#   bash setup_venv.sh
#
# After setup, activate with:
#   source venv_mid/bin/activate
# ─────────────────────────────────────────────────────────────────────────────

set -e   # exit on any error

VENV_NAME="venv_mid"
PYTHON="python3.12"
CUDA_VERSION="cu126"   # System torch is 2.7.1+cu126; venv inherits it via --system-site-packages

echo "=============================================="
echo " Keural Mid — venv setup"
echo " Python: $PYTHON"
echo " CUDA:   $CUDA_VERSION"
echo " Target: 2× H200 141GB"
echo "=============================================="

# ── Step 0: Verify prerequisites ─────────────────────────────────────────────
echo ""
echo "[0/6] Checking prerequisites..."

if ! command -v $PYTHON &> /dev/null; then
    echo "ERROR: $PYTHON not found. Install it first."
    exit 1
fi

CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -c2-)
echo "  Python: $($PYTHON --version)"
echo "  CUDA:   ${CUDA_VER:-not found (CPU mode only)}"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -2 | tr '\n' ' ')"

# Warn if CUDA < 12.1
if [[ ! -z "$CUDA_VER" ]]; then
    CUDA_MAJOR=$(echo $CUDA_VER | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VER | cut -d. -f2)
    if [[ $CUDA_MAJOR -lt 12 ]]; then
        echo "WARNING: CUDA $CUDA_VER detected. cu121 build requires CUDA >= 12.0"
        echo "  Change CUDA_VERSION to cu118 at the top of this script."
    fi
fi

# ── Step 1: Create venv ───────────────────────────────────────────────────────
echo ""
echo "[1/6] Creating virtual environment: $VENV_NAME ..."

if [ -d "$VENV_NAME" ]; then
    echo "  venv already exists — skipping creation."
else
    $PYTHON -m venv --system-site-packages $VENV_NAME
    echo "  Created $VENV_NAME/ (inherits system torch 2.7.1+cu126)"
fi

source $VENV_NAME/bin/activate
echo "  Activated: $(which python)"

# Upgrade pip first
pip install --upgrade pip setuptools wheel -q

# ── Step 2: Install PyTorch with CUDA ────────────────────────────────────────
echo ""
echo "[2/6] Verifying PyTorch (inherited from system — no install needed)..."
echo "  System has torch 2.7.1+cu126; venv inherits it via --system-site-packages."

python -c "
import torch
print(f'  torch version:  {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU count:      {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name}  ({props.total_memory // 1024**3} GB)')
"

# ── Step 3: Install flash-attention 2 ────────────────────────────────────────
echo ""
echo "[3/6] Verifying flash-attention (inherited from system — no install needed)..."
echo "  System has flash-attn 2.7.3; venv inherits it via --system-site-packages."

python -c "
try:
    import flash_attn
    print(f'  flash-attn version: {flash_attn.__version__}  OK')
except ImportError:
    print('  WARNING: flash-attn not found — will use PyTorch SDPA fallback')
"

# ── Step 4: Install core requirements ────────────────────────────────────────
echo ""
echo "[4/6] Installing packages missing from system..."
echo "  (transformers, torch, PIL, numpy etc. already present via --system-site-packages)"

# Only install what is genuinely absent from the system python3.12 environment
pip install \
    omegaconf==2.3.0 \
    webdataset==0.2.100 \
    --quiet
# img2dataset is a data-prep tool only; not imported by training code, skip here.

echo "  Missing packages installed."

# ── Step 5: Pre-download teacher model configs ────────────────────────────────
echo ""
echo "[5/6] Pre-caching teacher model configs (weights download at training time)..."

python -c "
from transformers import AutoConfig
import warnings
warnings.filterwarnings('ignore')

models = [
    'openai/clip-vit-large-patch14',
    'google/siglip-so400m-patch14-384',
]
for m in models:
    try:
        cfg = AutoConfig.from_pretrained(m)
        print(f'  OK: {m}')
    except Exception as e:
        print(f'  SKIP: {m} ({e})')
" 2>/dev/null || echo "  Config pre-cache skipped (no internet or HF token needed)."

# ── Step 6: Verify full environment ──────────────────────────────────────────
echo ""
echo "[6/6] Final verification..."

python -c "
import torch, transformers, PIL, yaml, safetensors, tqdm, numpy, pandas

print('  ── Package versions ──────────────────')
print(f'  torch:          {torch.__version__}')
print(f'  transformers:   {transformers.__version__}')
print(f'  Pillow:         {PIL.__version__}')
print(f'  numpy:          {numpy.__version__}')
print(f'  pandas:         {pandas.__version__}')
print(f'  safetensors:    {safetensors.__version__}')
print('')
print('  ── GPU status ────────────────────────')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU count:      {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name}  {p.total_memory // 1024**3} GB  (sm_{p.major}{p.minor})')

print('')
print('  ── Flash Attention ───────────────────')
try:
    import flash_attn
    print(f'  flash-attn: {flash_attn.__version__}  ENABLED')
except:
    print(f'  flash-attn: NOT installed (SDPA fallback active)')

print('')
print('  ── DDP / NCCL ───────────────────────')
import torch.distributed as dist
print(f'  NCCL available: {torch.distributed.is_nccl_available()}')
"

echo ""
echo "=============================================="
echo " Setup complete."
echo " Activate with:  source $VENV_NAME/bin/activate"
echo ""
echo " Run smoke test:"
echo "   cd /path/to/vision-encoder/mid"
echo "   python scripts/smoke_test_mid.py"
echo ""
echo " Launch 2× H200 training:"
echo "   torchrun --nproc_per_node=2 scripts/train_mid.py \\"
echo "       --config configs/keural_mid_2xh200.yaml \\"
echo "       --output-dir outputs/run"
echo "=============================================="
