"""检查 Isaac Lab 兼容性和现有扩展"""
import importlib
import subprocess
import os

# 检查已安装的 Isaac 相关模块
modules_to_check = [
    'omni.isaac.lab', 'isaaclab', 'isaacsim',
    'omni.isaac.core', 'omni.isaac.gym',
    'rsl_rl', 'stable_baselines3', 'rl_games',
    'skrl'
]

print("=== Python module check ===")
for mod in modules_to_check:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, '__version__', 'unknown')
        print(f"  {mod}: FOUND (version={ver})")
    except ImportError:
        print(f"  {mod}: NOT FOUND")
    except Exception as e:
        print(f"  {mod}: ERROR ({e})")

print("\n=== PyTorch info ===")
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

print("\n=== Isaac Sim extensions directory ===")
exts_dir = "/isaac-sim/exts/"
if os.path.exists(exts_dir):
    exts = sorted(os.listdir(exts_dir))
    rl_related = [e for e in exts if any(k in e.lower() for k in ['lab', 'gym', 'rl', 'robot', 'locomot'])]
    print(f"  Total extensions: {len(exts)}")
    print(f"  RL-related: {rl_related}")

print("\n=== Standalone examples ===")
examples_dir = "/isaac-sim/standalone_examples/"
if os.path.exists(examples_dir):
    for d in sorted(os.listdir(examples_dir)):
        print(f"  {d}/")
        subdir = os.path.join(examples_dir, d)
        if os.path.isdir(subdir):
            for f in sorted(os.listdir(subdir))[:5]:
                print(f"    {f}")
