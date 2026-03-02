---
name: isaac-sim-env-setup
description: >
  Automate SSH connection and Isaac Sim training environment setup on remote
  GPU cloud instances. Use this skill whenever the user mentions setting up a
  cloud GPU for Isaac Sim, configuring a remote training environment, preparing
  an Isaac Lab workspace, or connecting to a CompShare / cloud-rented GPU
  instance for robot simulation training. Also trigger when the user says things
  like "set up my cloud machine", "configure the remote server for training",
  "install Isaac Lab on the cloud", or "prepare the training environment".
  Covers SSH key setup, environment discovery, Isaac Lab + RL library
  installation inside Docker, and training script deployment.
---

# Isaac Sim Remote Environment Setup

Automate the full setup of an Isaac Sim training environment on a cloud GPU
instance via SSH. Currently tuned for the **CompShare "Isaac Sim Webrtc Ubuntu
22.04"** image, but the patterns apply to any instance where Isaac Sim runs
inside a Docker container.

## What this skill does

Given a remote server IP and credentials, it will:

1. **Establish SSH access** — generate an ed25519 key pair, upload the public
   key, and create an SSH config alias so future connections are passwordless.
2. **Discover the environment** — inspect GPU, CUDA, Docker containers, Isaac
   Sim version, Python, disk, and RAM.
3. **Install RL training stack** — install Isaac Lab, rsl_rl, skrl, and all
   sub-packages inside the running Isaac Sim Docker container.
4. **Deploy training scripts** — clone Isaac Lab source on the host, copy the
   rsl_rl train/play scripts into the container.
5. **Verify everything** — run a verification script to confirm all imports
   succeed and the GPU is accessible from the container Python.

After this skill completes, the user should be able to run a single training
command and get a working locomotion policy.

---

## Step 0 — Gather information

Before starting, collect from the user:

| Info needed | Example | Notes |
|-------------|---------|-------|
| Server IP | `117.50.89.136` | Required |
| SSH username | `ubuntu` | Default: `ubuntu` |
| SSH password | (user provides) | Needed once for key upload |
| SSH alias | `cloud-gpu` | Default: `cloud-gpu` |
| DNS servers | `100.90.90.90` | Optional, for faster network in China |

If the user says they already have SSH access, skip to Step 2.

---

## Step 1 — SSH connection

### 1.1 Generate a key pair (if none exists)

```bash
ssh-keygen -t ed25519 -C "isaac-sim-training" -f ~/.ssh/id_ed25519_cloud -N ""
```

Check first: `ls ~/.ssh/id_ed25519_cloud 2>/dev/null` — skip if the file exists.

### 1.2 Upload the public key

The local machine likely does not have `sshpass`. Use Python `paramiko` instead:

```python
import paramiko, os

pub_key_path = os.path.expanduser("~/.ssh/id_ed25519_cloud.pub")
with open(pub_key_path) as f:
    pub_key = f.read().strip()

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("<IP>", username="<USER>", password="<PASSWORD>", timeout=15)

for cmd in [
    "mkdir -p ~/.ssh",
    "chmod 700 ~/.ssh",
    f'echo "{pub_key}" >> ~/.ssh/authorized_keys',
    "chmod 600 ~/.ssh/authorized_keys",
]:
    client.exec_command(cmd)

client.close()
```

**Finding a working Python**: On Windows the bare `python3` alias often does
not work. Probe in this order:
1. `/c/Python/Python39/python` (or similar)
2. `/c/Users/<user>/AppData/Local/Programs/Python/Python*/python.exe`
3. `py` (Windows launcher)

Install paramiko into whichever interpreter you find:
`<python> -m pip install paramiko`

### 1.3 Configure SSH alias

Append to `~/.ssh/config`:

```
Host cloud-gpu
    HostName <IP>
    User <USER>
    IdentityFile ~/.ssh/id_ed25519_cloud
    StrictHostKeyChecking no
```

### 1.4 Verify

```bash
ssh cloud-gpu "echo 'SSH OK' && hostname && whoami"
```

---

## Step 2 — Discover the environment

Run `scripts/check_env.sh` on the remote host. Copy it up and execute:

```bash
scp scripts/check_env.sh cloud-gpu:/tmp/
ssh cloud-gpu "bash /tmp/check_env.sh"
```

Parse the output and present a summary table to the user. Key things to
confirm:

| Check | Expected for CompShare image |
|-------|------------------------------|
| GPU | RTX 4090 or RTX 5090 |
| Driver | 550+ |
| Isaac Sim container | `nvcr.io/nvidia/isaac-sim:5.1.0` running |
| Container Python | 3.11.x |
| Container PyTorch | 2.7.x+cu128 |
| CUDA available | True |
| Host disk free | > 50 GB |

### Important architecture insight

On the CompShare image, **Isaac Sim runs inside a Docker container** named
`isaac-sim`. The host OS has CUDA and a driver but no Python pip or torch.
All training work happens *inside* the container via
`sudo docker exec isaac-sim ...`.

The container's Python is at `/isaac-sim/python.sh` (a wrapper around
`/isaac-sim/kit/python/bin/python3`). Always use this instead of bare
`python3`.

---

## Step 3 — Install the RL training stack

### 3.1 Configure DNS (China-specific, but do it by default)

Faster DNS prevents pip timeouts. Set it on **both the host and inside the
container** — Docker copies the host's `/etc/resolv.conf` at container
creation, so changes to the host after that point do not propagate.

```bash
# Host DNS
ssh cloud-gpu 'echo "nameserver 100.90.90.90
nameserver 100.90.90.100" | sudo tee /etc/resolv.conf'

# Container DNS (requires --user root because the container user can't write /etc)
ssh cloud-gpu 'sudo docker exec --user root isaac-sim bash -c \
  "echo -e \"nameserver 100.90.90.90\nnameserver 100.90.90.100\" > /etc/resolv.conf"'
```

### 3.2 Install Isaac Lab (pip)

```bash
ssh cloud-gpu 'sudo docker exec isaac-sim \
  /isaac-sim/python.sh -m pip install \
  isaaclab==2.3.2.post1 \
  --extra-index-url https://pypi.nvidia.com'
```

This pulls ~150 MB of dependencies. Takes 2-5 min.

### 3.3 Install rsl_rl

**Critical version constraint**: Isaac Lab 2.3.2 requires **rsl-rl-lib==3.1.2**.
Version 5.0.0 has breaking config changes (`actor` key vs `policy` key). The
container has no git, so install from GitHub archive URL:

```bash
ssh cloud-gpu 'sudo docker exec -e GIT_PYTHON_REFRESH=quiet isaac-sim \
  /isaac-sim/python.sh -m pip install \
  "rsl-rl-lib==3.1.2"'
```

The `-e GIT_PYTHON_REFRESH=quiet` is essential — the container lacks git, and
without this flag, `import rsl_rl` crashes with a `GitPython` init error.

### 3.4 Install skrl (optional backup RL library)

```bash
ssh cloud-gpu 'sudo docker exec isaac-sim \
  /isaac-sim/python.sh -m pip install skrl'
```

### 3.5 Install Isaac Lab sub-packages

The `isaaclab` pip package bundles source for sub-packages but does not install
them automatically. Install from the bundled source:

```bash
SITE="/isaac-sim/kit/python/lib/python3.11/site-packages/isaaclab/source"

# Only install these three. Skip `isaaclab` core — its editable install
# fails on permissions and is not needed (the pip-installed isaaclab is enough).
for pkg in isaaclab_assets isaaclab_rl isaaclab_tasks; do
  ssh cloud-gpu "sudo docker exec isaac-sim \
    /isaac-sim/python.sh -m pip install -e ${SITE}/${pkg}"
done
```

**Note**: `isaaclab_assets` and `isaaclab_tasks` will show import errors if
tested outside a SimulationApp context — that is normal. They work correctly
inside training scripts.

### 3.6 Verify installation

Copy `scripts/verify_install.py` into the container and run it:

```bash
scp scripts/verify_install.py cloud-gpu:/tmp/
ssh cloud-gpu 'sudo docker cp /tmp/verify_install.py isaac-sim:/tmp/ && \
  sudo docker exec -e GIT_PYTHON_REFRESH=quiet isaac-sim \
  /isaac-sim/python.sh /tmp/verify_install.py'
```

Expected output:

```
=== Installation Verification ===
  isaaclab: OK
  rsl_rl: OK
  skrl: OK (v1.4.3)
  torch: OK (v2.7.0+cu128, CUDA=True)
    GPU: NVIDIA GeForce RTX 4090
  gymnasium: OK (v1.2.0)
  tensorboard: OK
All dependencies verified!
```

---

## Step 4 — Deploy training scripts

### 4.1 Clone Isaac Lab source on the host

The host has git; the container does not. Clone on the host, then copy what we
need into the container.

```bash
ssh cloud-gpu 'cd /home/ubuntu && \
  [ -d IsaacLab ] || git clone --depth 1 https://github.com/isaac-sim/IsaacLab.git'
```

### 4.2 Copy rsl_rl training scripts into the container

```bash
ssh cloud-gpu '
sudo docker exec isaac-sim mkdir -p /tmp/training/rsl_rl
for f in train.py play.py cli_args.py; do
  sudo docker cp /home/ubuntu/IsaacLab/scripts/reinforcement_learning/rsl_rl/$f \
    isaac-sim:/tmp/training/rsl_rl/
done
echo "Training scripts deployed"
sudo docker exec isaac-sim ls /tmp/training/rsl_rl/'
```

---

## Step 5 — Final validation

Run a quick headless smoke test to confirm Isaac Sim can load a robot:

```bash
scp scripts/smoke_test.py cloud-gpu:/tmp/
ssh cloud-gpu 'sudo docker cp /tmp/smoke_test.py isaac-sim:/tmp/ && \
  sudo docker exec isaac-sim \
  /isaac-sim/python.sh /tmp/smoke_test.py --headless --enable_cameras'
```

If the output contains lines like:
```
Mesh '/World/Go2/base_white/visuals' ...
articulation at /World/Go2/base ...
Simulation App Shutting Down
```
then the environment is fully ready.

Present the user with a summary:

```
Environment setup complete!

  SSH:        cloud-gpu → ubuntu@<IP>
  GPU:        RTX 4090 24GB
  Isaac Sim:  5.1.0 (Docker)
  Isaac Lab:  2.3.2
  rsl_rl:     3.1.2
  Training:   /tmp/training/rsl_rl/{train,play}.py

Quick-start training command:
  ssh cloud-gpu 'sudo docker exec \
    -e GIT_PYTHON_REFRESH=quiet -w /isaac-sim isaac-sim \
    /isaac-sim/python.sh /tmp/training/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --headless --num_envs 4096 --max_iterations 300'
```

---

## Key gotchas (read `references/troubleshooting.md` for full list)

1. **rsl_rl version**: Must be 3.1.2, not 5.0.0. The config schema changed.
2. **No git in container**: Always pass `-e GIT_PYTHON_REFRESH=quiet` to
   `docker exec`, or `import rsl_rl` will crash.
3. **Docker cp permissions**: Files copied with `docker cp` inherit host UID.
   The container user (`isaac-sim`, uid 1234) may not be able to chmod them.
   Prefer pip-installing from URLs or building wheels on the host.
4. **Nested SSH quoting**: Complex Python one-liners inside
   `ssh host 'docker exec ... python -c "..."'` break on nested quotes. Write
   a `.py` or `.sh` file, scp it up, and execute it instead.
5. **Isaac Sim startup is slow**: First SimulationApp launch takes 60-90s to
   load all extensions. Subsequent runs in the same container session are
   faster (~15s). Be patient with the smoke test.
6. **`isaaclab_assets` import error outside SimulationApp**: This is expected.
   Isaac Lab modules that touch the Omniverse runtime can only be imported
   after `SimulationApp()` is created.
7. **Working directory matters for training**: Run training from `/isaac-sim`
   (use `-w /isaac-sim` in docker exec) so that log output goes to
   `/isaac-sim/logs/`.
