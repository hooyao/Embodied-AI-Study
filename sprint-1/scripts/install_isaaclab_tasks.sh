#!/bin/bash
# 安装 Isaac Lab 子包（tasks, rl, assets）
set -e

SITE_PKG="/isaac-sim/kit/python/lib/python3.11/site-packages"
ISAACLAB_SRC="${SITE_PKG}/isaaclab/source"

echo "=== Installing isaaclab sub-packages ==="

# 安装 isaaclab core (from source inside pip package)
echo "--- isaaclab core ---"
cd "${ISAACLAB_SRC}/isaaclab"
/isaac-sim/python.sh -m pip install -e . 2>&1 | tail -3

# 安装 isaaclab_assets
echo "--- isaaclab_assets ---"
cd "${ISAACLAB_SRC}/isaaclab_assets"
/isaac-sim/python.sh -m pip install -e . 2>&1 | tail -3

# 安装 isaaclab_rl
echo "--- isaaclab_rl ---"
cd "${ISAACLAB_SRC}/isaaclab_rl"
/isaac-sim/python.sh -m pip install -e . 2>&1 | tail -3

# 安装 isaaclab_tasks
echo "--- isaaclab_tasks ---"
cd "${ISAACLAB_SRC}/isaaclab_tasks"
/isaac-sim/python.sh -m pip install -e . 2>&1 | tail -3

echo ""
echo "=== Verify installations ==="
/isaac-sim/python.sh -c "
import isaaclab; print('isaaclab: OK')
import isaaclab_rl; print('isaaclab_rl: OK')
import isaaclab_assets; print('isaaclab_assets: OK')
import isaaclab_tasks; print('isaaclab_tasks: OK')
print('All sub-packages installed successfully!')
"
