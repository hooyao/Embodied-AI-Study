#!/bin/bash
# ============================================================
# Isaac Lab + RL Stack — One-Shot Installer
# ============================================================
# Run on the remote HOST. Installs everything into the
# isaac-sim Docker container.
#
# Usage: bash install_all.sh [--skip-isaac-lab] [--skip-rsl-rl]
# ============================================================

set -uo pipefail  # no -e: some sub-package installs fail harmlessly

CONTAINER="isaac-sim"
PY="/isaac-sim/python.sh"
SITE="/isaac-sim/kit/python/lib/python3.11/site-packages"

log()  { echo -e "\n>>> $1"; }
fail() { echo "FAILED: $1" >&2; exit 1; }

# ---------- Parse args ----------
SKIP_ISAACLAB=false
SKIP_RSLRL=false
for arg in "$@"; do
    case $arg in
        --skip-isaac-lab) SKIP_ISAACLAB=true ;;
        --skip-rsl-rl)    SKIP_RSLRL=true ;;
    esac
done

# ---------- Preflight ----------
log "Checking container is running..."
sudo docker inspect "$CONTAINER" &>/dev/null \
    || fail "Container '$CONTAINER' not found. Is Isaac Sim running?"

# ---------- Fix container DNS (China networks) ----------
log "Syncing container DNS with host..."
HOST_DNS=$(cat /etc/resolv.conf | grep "^nameserver" | head -2)
if [ -n "$HOST_DNS" ]; then
    sudo docker exec --user root "$CONTAINER" bash -c \
        "echo '$HOST_DNS' > /etc/resolv.conf" 2>/dev/null \
        && echo "  Container DNS updated" \
        || echo "  Warning: could not update container DNS (non-fatal)"
fi

# ---------- Isaac Lab ----------
if [ "$SKIP_ISAACLAB" = false ]; then
    log "Installing Isaac Lab 2.3.2..."
    sudo docker exec "$CONTAINER" $PY -m pip install \
        isaaclab==2.3.2.post1 \
        --extra-index-url https://pypi.nvidia.com \
        2>&1 | tail -5

    log "Installing Isaac Lab sub-packages..."
    # Skip 'isaaclab' core (editable install often fails on permissions, not needed)
    for pkg in isaaclab_assets isaaclab_rl isaaclab_tasks; do
        echo "  - $pkg"
        sudo docker exec "$CONTAINER" $PY -m pip install \
            -e "${SITE}/isaaclab/source/${pkg}" \
            2>&1 | tail -2
    done
else
    log "Skipping Isaac Lab (--skip-isaac-lab)"
fi

# ---------- rsl_rl ----------
if [ "$SKIP_RSLRL" = false ]; then
    log "Installing rsl-rl-lib==3.1.2..."
    sudo docker exec -e GIT_PYTHON_REFRESH=quiet "$CONTAINER" $PY -m pip install \
        "rsl-rl-lib==3.1.2" \
        2>&1 | tail -5
else
    log "Skipping rsl_rl (--skip-rsl-rl)"
fi

# ---------- skrl ----------
log "Installing skrl..."
sudo docker exec "$CONTAINER" $PY -m pip install skrl \
    2>&1 | tail -3

# ---------- Clone Isaac Lab source ----------
log "Cloning Isaac Lab source on host..."
cd /home/ubuntu
if [ ! -d IsaacLab ]; then
    git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git
    echo "  Cloned."
else
    echo "  Already exists."
fi

# ---------- Copy training scripts ----------
log "Deploying rsl_rl training scripts to container..."
sudo docker exec "$CONTAINER" mkdir -p /tmp/training/rsl_rl
for f in train.py play.py cli_args.py; do
    sudo docker cp "/home/ubuntu/IsaacLab/scripts/reinforcement_learning/rsl_rl/$f" \
        "$CONTAINER:/tmp/training/rsl_rl/"
done
echo "  Deployed: train.py, play.py, cli_args.py"

log "Installation complete!"
