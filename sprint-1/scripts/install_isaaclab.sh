#!/bin/bash
# 在 Isaac Sim Docker 容器内安装 Isaac Lab 及 RL 依赖
set -e

echo "=== Step 1: Install Isaac Lab ==="
/isaac-sim/python.sh -m pip install isaaclab==2.3.2.post1 --extra-index-url https://pypi.nvidia.com 2>&1

echo ""
echo "=== Step 2: Install RL libraries ==="
/isaac-sim/python.sh -m pip install rsl_rl 2>&1 || echo "rsl_rl install failed, trying from source later"
/isaac-sim/python.sh -m pip install skrl 2>&1 || echo "skrl install failed"
/isaac-sim/python.sh -m pip install tensorboard 2>&1

echo ""
echo "=== Step 3: Verify installation ==="
/isaac-sim/python.sh -c "
import isaaclab
print(f'Isaac Lab version: {isaaclab.__version__}')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
try:
    import rsl_rl
    print(f'rsl_rl: available')
except ImportError:
    print('rsl_rl: not available')
try:
    import skrl
    print(f'skrl version: {skrl.__version__}')
except ImportError:
    print('skrl: not available')
print('Installation complete!')
"
