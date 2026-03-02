"""Sprint 1: 从 tensorboard 日志提取并绘制 reward 曲线
用法: python plot_reward.py
"""
import os
import sys

# 尝试导入必要的库
try:
    from tensorboard.backend.event_processing import event_accumulator
    import matplotlib
    matplotlib.use('Agg')  # 无 GUI 模式
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"需要安装依赖: pip install tensorboard matplotlib")
    print(f"缺少: {e}")
    sys.exit(1)

# 日志路径
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "go2_flat_ppo_2026-03-02")
log_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("events.out")]

if not log_files:
    print("未找到 tensorboard 日志文件")
    sys.exit(1)

log_path = RESULTS_DIR
print(f"读取日志: {log_path}")

# 加载 tensorboard 日志
ea = event_accumulator.EventAccumulator(log_path)
ea.Reload()

# 获取可用的 scalar tags
tags = ea.Tags()['scalars']
print(f"\n可用指标 ({len(tags)}):")
for tag in sorted(tags):
    print(f"  {tag}")

# 提取所有指标
metrics = {}
for tag in tags:
    events = ea.Scalars(tag)
    metrics[tag] = {
        'steps': [e.step for e in events],
        'values': [e.value for e in events],
    }

# 绘制图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Sprint 1: Unitree Go2 Flat PPO Training (Isaac Lab)', fontsize=14, fontweight='bold')

# 1. Mean Reward
reward_tag = 'Train/mean_reward'
if reward_tag in metrics:
    ax = axes[0, 0]
    data = metrics[reward_tag]
    ax.plot(data['steps'], data['values'], color='#2196F3', linewidth=1.5)
    ax.set_title('Mean Reward')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)

# 2. Episode Length
length_tag = 'Train/mean_episode_length'
if length_tag in metrics:
    ax = axes[0, 1]
    data = metrics[length_tag]
    ax.plot(data['steps'], data['values'], color='#4CAF50', linewidth=1.5)
    ax.set_title('Mean Episode Length')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Steps')
    ax.grid(True, alpha=0.3)

# 3. Value Loss
vloss_tag = 'Loss/value_function'
if vloss_tag in metrics:
    ax = axes[1, 0]
    data = metrics[vloss_tag]
    ax.plot(data['steps'], data['values'], color='#FF9800', linewidth=1.5)
    ax.set_title('Value Function Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)

# 4. Noise Std
noise_tag = 'Policy/mean_noise_std'
if noise_tag in metrics:
    ax = axes[1, 1]
    data = metrics[noise_tag]
    ax.plot(data['steps'], data['values'], color='#9C27B0', linewidth=1.5)
    ax.set_title('Mean Action Noise Std')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Std')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(RESULTS_DIR, 'training_curves.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n训练曲线已保存: {output_path}")

# 打印训练摘要
if reward_tag in metrics:
    rewards = metrics[reward_tag]['values']
    print(f"\n=== Training Summary ===")
    print(f"Initial reward: {rewards[0]:.2f}")
    print(f"Final reward: {rewards[-1]:.2f}")
    print(f"Max reward: {max(rewards):.2f}")
    print(f"Improvement: {rewards[-1] - rewards[0]:.2f} ({(rewards[-1]/max(rewards[0], 0.01) - 1)*100:.1f}%)")

if length_tag in metrics:
    lengths = metrics[length_tag]['values']
    print(f"Final episode length: {lengths[-1]:.1f}")
