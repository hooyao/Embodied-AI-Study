"""验证所有 RL 训练依赖是否安装成功"""
print("=== 验证安装 ===")

import isaaclab
print("  isaaclab: OK")

import rsl_rl
print(f"  rsl_rl: OK")

import skrl
print(f"  skrl: OK (v{skrl.__version__})")

import torch
print(f"  torch: OK (v{torch.__version__}, CUDA={torch.cuda.is_available()})")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    CUDA: {torch.version.cuda}")

import gymnasium
print(f"  gymnasium: OK (v{gymnasium.__version__})")

try:
    import tensorboard
    print("  tensorboard: OK")
except ImportError:
    print("  tensorboard: NOT FOUND")

print("\n所有依赖安装验证完成！")
