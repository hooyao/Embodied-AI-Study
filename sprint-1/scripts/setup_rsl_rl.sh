#!/bin/bash
# 安装 rsl_rl 从源码 + 克隆 Isaac Lab 源码获取训练脚本
set -e

echo "=== Install rsl_rl from source ==="
cd /tmp
git clone https://github.com/leggedrobotics/rsl_rl.git 2>/dev/null || (cd rsl_rl && git pull)
cd rsl_rl
/isaac-sim/python.sh -m pip install -e . 2>&1

echo ""
echo "=== Clone Isaac Lab source for training scripts ==="
cd /root
if [ ! -d "IsaacLab" ]; then
    git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git 2>&1
else
    echo "IsaacLab already cloned"
fi

echo ""
echo "=== List locomotion training scripts ==="
find /root/IsaacLab -path "*/locomotion*" -name "*.py" 2>/dev/null | head -20
find /root/IsaacLab -path "*velocity*" -name "*.py" 2>/dev/null | head -10

echo ""
echo "=== Verify all installations ==="
/isaac-sim/python.sh -c "
import isaaclab
print('isaaclab: OK')
import rsl_rl
print('rsl_rl: OK')
import skrl
print(f'skrl: OK (v{skrl.__version__})')
import torch
print(f'torch: OK (v{torch.__version__}, CUDA={torch.cuda.is_available()})')
import gymnasium
print(f'gymnasium: OK (v{gymnasium.__version__})')
print('All installations verified!')
"
