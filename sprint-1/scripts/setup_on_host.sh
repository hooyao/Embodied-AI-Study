#!/bin/bash
# 在宿主机上克隆仓库，然后复制到 Docker 容器内
set -e

echo "=== Install rsl_rl ==="
cd /home/ubuntu
if [ ! -d "rsl_rl" ]; then
    git clone --depth 1 https://github.com/leggedrobotics/rsl_rl.git
fi
sudo docker cp /home/ubuntu/rsl_rl isaac-sim:/tmp/rsl_rl
sudo docker exec isaac-sim /isaac-sim/python.sh -m pip install -e /tmp/rsl_rl 2>&1 | tail -5

echo ""
echo "=== Clone Isaac Lab source ==="
if [ ! -d "IsaacLab" ]; then
    git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git
fi

echo ""
echo "=== Find locomotion training configs ==="
find /home/ubuntu/IsaacLab -path "*/locomotion*" -name "*.py" 2>/dev/null | head -30
find /home/ubuntu/IsaacLab -path "*velocity*" -name "*.py" 2>/dev/null | head -10

echo ""
echo "=== Verify rsl_rl installation ==="
sudo docker exec isaac-sim /isaac-sim/python.sh -c "
import isaaclab; print('isaaclab: OK')
import rsl_rl; print('rsl_rl: OK')
import skrl; print(f'skrl: OK (v{skrl.__version__})')
import torch; print(f'torch: OK (v{torch.__version__}, CUDA={torch.cuda.is_available()})')
print('All verified!')
"
