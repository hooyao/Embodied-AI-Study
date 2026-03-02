#!/bin/bash
# ============================================================
# Isaac Sim Cloud Instance — Environment Discovery Script
# ============================================================
# Run on the remote HOST (not inside Docker).
# Outputs a structured report of the GPU, CUDA, Docker, Isaac
# Sim container, disk, and memory state.
# ============================================================

set -euo pipefail

section() { echo -e "\n=== $1 ==="; }

# ---------- GPU & Driver ----------
section "GPU & Driver"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
        || echo "nvidia-smi query failed"
    echo "---"
    nvidia-smi 2>/dev/null | head -20
else
    echo "nvidia-smi not found"
fi

# ---------- CUDA Toolkit ----------
section "CUDA Toolkit (Host)"
if [ -f /usr/local/cuda/bin/nvcc ]; then
    /usr/local/cuda/bin/nvcc --version 2>/dev/null | grep "release"
else
    echo "nvcc not found on host PATH"
fi
ls -d /usr/local/cuda* 2>/dev/null || echo "No /usr/local/cuda* directories"

# ---------- Docker ----------
section "Docker"
if sudo docker ps &>/dev/null; then
    sudo docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}"
else
    echo "Docker not accessible (no sudo or docker not running)"
fi

# ---------- Isaac Sim Container ----------
section "Isaac Sim Container"
CONTAINER_NAME="isaac-sim"

if sudo docker inspect "$CONTAINER_NAME" &>/dev/null; then
    echo "Container '$CONTAINER_NAME' exists"

    # Image
    IMAGE=$(sudo docker inspect --format='{{.Config.Image}}' "$CONTAINER_NAME" 2>/dev/null)
    echo "Image: $IMAGE"

    # Isaac Sim version
    sudo docker exec "$CONTAINER_NAME" cat /isaac-sim/VERSION 2>/dev/null \
        && echo "" \
        || echo "VERSION file not found"

    # Python version
    echo -n "Python: "
    sudo docker exec "$CONTAINER_NAME" /isaac-sim/python.sh -c \
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" \
        2>/dev/null || echo "unknown"

    # PyTorch
    echo -n "PyTorch: "
    sudo docker exec "$CONTAINER_NAME" /isaac-sim/python.sh -c \
        "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed"

    # CUDA from container
    echo -n "CUDA (container torch): "
    sudo docker exec "$CONTAINER_NAME" /isaac-sim/python.sh -c \
        "import torch; print(torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" \
        2>/dev/null || echo "unknown"

    # Container user
    echo -n "Container user: "
    sudo docker exec "$CONTAINER_NAME" whoami 2>/dev/null || echo "unknown"
else
    echo "Container '$CONTAINER_NAME' NOT found"
    echo "Looking for any Isaac Sim containers..."
    sudo docker ps -a --filter "ancestor=nvcr.io/nvidia/isaac-sim" \
        --format "{{.Names}} ({{.Image}}, {{.Status}})" 2>/dev/null \
        || echo "none found"
fi

# ---------- Host Python ----------
section "Host Python"
python3 --version 2>/dev/null || echo "python3 not available"
python3 -c "import torch; print('PyTorch:', torch.__version__)" 2>/dev/null \
    || echo "No torch on host"

# ---------- Disk & Memory ----------
section "Disk"
df -h / /home 2>/dev/null | grep -v tmpfs

section "Memory"
free -h 2>/dev/null

# ---------- Existing RL Packages ----------
section "Installed RL Packages (Container)"
if sudo docker inspect "$CONTAINER_NAME" &>/dev/null; then
    sudo docker exec "$CONTAINER_NAME" /isaac-sim/python.sh -m pip list 2>/dev/null \
        | grep -iE "isaac|torch|rsl|skrl|gym|stable.baselines|rl.games|tensorboard" \
        || echo "pip list failed or no RL packages found"
fi

echo -e "\n=== Discovery Complete ==="
