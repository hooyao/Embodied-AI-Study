#!/bin/bash
# 云端环境检查脚本

echo "=== Host Python ==="
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())" 2>/dev/null || echo "No torch on host"

echo "=== CUDA toolkit ==="
/usr/local/cuda/bin/nvcc --version 2>/dev/null || echo "No nvcc"
ls /usr/local/cuda/lib64/libcudart* 2>/dev/null || echo "No CUDA runtime libs"

echo "=== Isaac Sim container Python ==="
sudo docker exec isaac-sim /isaac-sim/python.sh -c "
import sys
print('Python:', sys.version)
try:
    import torch
    print('PyTorch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
except ImportError:
    print('No PyTorch')
" 2>&1

echo "=== Isaac Sim standalone examples ==="
sudo docker exec isaac-sim ls /isaac-sim/standalone_examples/ 2>/dev/null

echo "=== pip packages in container (isaac/rl related) ==="
sudo docker exec isaac-sim /isaac-sim/python.sh -m pip list 2>/dev/null | grep -i -E "isaac|torch|gym|rl|rsl|stable" | head -30
