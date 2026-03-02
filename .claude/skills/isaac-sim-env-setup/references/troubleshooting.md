# Troubleshooting Reference

All issues documented here were encountered during real setup sessions on the
CompShare "Isaac Sim Webrtc Ubuntu 22.04" image. Each entry includes the
symptom, root cause, and verified fix.

---

## Table of Contents

1. [rsl_rl version mismatch (KeyError: 'actor')](#1-rsl_rl-version-mismatch)
2. [GitPython ImportError in container](#2-gitpython-importerror)
3. [Docker cp permission denied](#3-docker-cp-permission-denied)
4. [Nested SSH quote escaping](#4-nested-ssh-quote-escaping)
5. [isaaclab.sim ModuleNotFoundError](#5-isaaclab-sim-import-error)
6. [Isaac Sim first launch is very slow](#6-slow-first-launch)
7. [play.py FileNotFoundError for checkpoint](#7-checkpoint-not-found)
8. [Host Python not found on Windows](#8-windows-python-not-found)
9. [pip install fails with DNS timeout](#9-dns-timeout)
10. [GPU memory conflict with WebRTC stream](#10-gpu-memory-conflict)
11. [Container DNS not synced with host](#11-container-dns-not-synced)
12. [isaaclab core editable install fails](#12-isaaclab-core-editable-install)

---

## 1. rsl_rl version mismatch

**Symptom**:
```
KeyError: 'actor'
```
at `rsl_rl/algorithms/ppo.py` inside `construct_algorithm`.

**Root cause**: rsl_rl 5.0.0 introduced a breaking config schema change. It
expects separate `actor` and `critic` config keys, while Isaac Lab 2.3.2
produces a `policy` key with `RslRlPpoActorCriticCfg`.

**Fix**:
```bash
sudo docker exec isaac-sim /isaac-sim/python.sh -m pip install "rsl-rl-lib==3.1.2"
```

Isaac Lab 2.3.2's `isaaclab_rl/setup.py` explicitly declares
`rsl-rl-lib==3.1.2` as a dependency. Always match this version.

---

## 2. GitPython ImportError

**Symptom**:
```
ImportError: Failed to initialize: Bad git executable.
The git executable must be specified in one of the following ways...
```

**Root cause**: The Isaac Sim Docker container does not ship with `git`. The
`rsl_rl` library imports `GitPython` for logging, which crashes if git is
absent.

**Fix**: Set the environment variable before any `docker exec` that touches
rsl_rl:
```bash
sudo docker exec -e GIT_PYTHON_REFRESH=quiet isaac-sim ...
```

This tells GitPython to silently skip initialization instead of raising.

---

## 3. Docker cp permission denied

**Symptom**:
```
chmod: changing permissions of '/tmp/rsl_rl/...': Operation not permitted
error: could not create 'rsl_rl_lib.egg-info': Permission denied
```

**Root cause**: `docker cp` preserves the host file ownership (UID/GID). The
container runs as user `isaac-sim` (uid 1234) and cannot modify files owned by
`ubuntu` (uid 1000) on the host.

**Fix options**:
1. **Preferred**: Install packages from PyPI or GitHub archive URLs, which
   downloads fresh with correct permissions:
   ```bash
   pip install "rsl-rl-lib==3.1.2"
   ```
2. **Alternative**: Create a tar archive on the host with `--owner=0
   --group=0`, copy, and extract inside the container.
3. **Last resort**: Run docker exec as root (`--user root`) for the chmod, but
   this can cause other ownership issues.

---

## 4. Nested SSH quote escaping

**Symptom**: Syntax errors when running Python code via SSH + Docker:
```
ssh host 'docker exec container python -c "print(f'{x}')"'
# → bash: syntax error near unexpected token `('
```

**Root cause**: Multiple levels of shell interpretation (local bash → SSH →
Docker → Python) mangle quotes.

**Fix**: Write a `.py` or `.sh` file, upload it with `scp` + `docker cp`, and
execute the file:
```bash
scp script.py cloud-gpu:/tmp/
ssh cloud-gpu 'sudo docker cp /tmp/script.py isaac-sim:/tmp/ && \
  sudo docker exec isaac-sim /isaac-sim/python.sh /tmp/script.py'
```

This eliminates all quoting issues and makes scripts reusable.

---

## 5. isaaclab.sim import error

**Symptom**:
```
ModuleNotFoundError: No module named 'isaaclab.sim'
```
when importing `isaaclab_assets` or `isaaclab_tasks` from a plain Python
script.

**Root cause**: Isaac Lab's `sim` module wraps Omniverse extensions that only
load after `SimulationApp()` is instantiated. Importing before that always
fails.

**Fix**: This is **expected behavior**, not an error. All Isaac Lab training
scripts follow this pattern:
```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
# NOW import Isaac Lab modules
import isaaclab_tasks  # works here
```

The `verify_install.py` script only tests packages that can be imported without
SimulationApp (isaaclab, rsl_rl, skrl, torch, gymnasium).

---

## 6. Slow first launch

**Symptom**: `SimulationApp` startup takes 60-90 seconds on first run, with
hundreds of `[ext: ...]` log lines.

**Root cause**: Isaac Sim loads ~120 Omniverse extensions on startup, including
physics, rendering, and asset management plugins. First run also populates
shader caches.

**Fix**: This is normal. Subsequent runs in the same container session take
~15s because caches are warm. For training, the startup cost is amortized over
300+ iterations.

---

## 7. Checkpoint not found

**Symptom**:
```
FileNotFoundError: Unable to find the file: model_299.pt
```
in play.py.

**Root cause**: play.py's `--checkpoint` argument is processed by
`retrieve_file_path()` which does not do relative path resolution. Passing just
`model_299.pt` fails.

**Fix**: Always use the **absolute path** inside the container:
```bash
--checkpoint "/isaac-sim/logs/rsl_rl/unitree_go2_flat/<RUN_DIR>/model_299.pt"
```

---

## 8. Windows Python not found

**Symptom**: `python3` or `python` returns "not found" or opens the Microsoft
Store on Windows.

**Root cause**: Windows App Execution Aliases redirect `python` and `python3`
to the Store. The actual interpreter is elsewhere.

**Fix**: Probe for available Pythons in order:
```bash
/c/Python/Python39/python            # common system install
/c/Users/<user>/AppData/Local/Programs/Python/Python*/python.exe
py                                    # Windows launcher
```

Use whichever resolves first.

---

## 9. DNS timeout

**Symptom**: `pip install` hangs or fails with connection timeout in China.

**Root cause**: Default DNS is slow or unreliable for reaching PyPI and
NVIDIA's package index.

**Fix**: Configure faster DNS on the host (does not affect container):
```bash
echo "nameserver 100.90.90.90
nameserver 100.90.90.100" | sudo tee /etc/resolv.conf
```

These are fast Chinese DNS resolvers. The change is ephemeral (lost on
reboot).

---

## 10. GPU memory conflict with WebRTC stream

**Symptom**: Isaac Sim training fails with OOM or the WebRTC streaming
freezes during headless training.

**Root cause**: The CompShare image runs Isaac Sim in WebRTC streaming mode
by default, using ~3 GB VRAM. Running a second `SimulationApp` (e.g., play.py
without `--headless`) competes for GPU memory.

**Fix options**:
1. **Always use `--headless`** for training and evaluation. This does not
   allocate rendering resources beyond what's needed for camera recording.
2. If you need live WebRTC visualization, stop the streaming container first
   or use `--livestream 1` in your script.
3. For evaluation, use `--video` flag to record an mp4, then download it.
   This is more reliable than trying to use the WebRTC stream.

---

## 11. Container DNS not synced with host

**Symptom**:
```
ERROR: Could not find a version that satisfies the requirement onnx>=1.18.0
```
or pip install hangs indefinitely inside the container, even though the host's
DNS was already configured correctly.

**Root cause**: Docker copies the host's `/etc/resolv.conf` into the container
at container creation time. If you change the host's DNS *after* the container
is already running, the container keeps the old (broken) DNS.

**Fix**: Update the container's DNS directly, using `--user root` because the
default container user (`isaac-sim`) cannot write to `/etc/resolv.conf`:
```bash
sudo docker exec --user root isaac-sim bash -c \
  "echo -e 'nameserver 100.90.90.90\nnameserver 100.90.90.100' > /etc/resolv.conf"
```

The `install_all.sh` script now does this automatically.

---

## 12. isaaclab core editable install fails

**Symptom**:
```
error: could not create 'isaaclab.egg-info': Permission denied
```
when running `pip install -e .../isaaclab/source/isaaclab`.

**Root cause**: The isaaclab pip package installs its source into
`site-packages/isaaclab/source/isaaclab/`. Running `pip install -e` there
tries to create `.egg-info` in a directory owned by root, which the container
user cannot write to.

**Fix**: Skip the `isaaclab` core sub-package entirely. The pip-installed
`isaaclab==2.3.2.post1` already provides the core module. Only install the
three sub-packages that are NOT covered by the main pip package:
`isaaclab_assets`, `isaaclab_rl`, `isaaclab_tasks`.
