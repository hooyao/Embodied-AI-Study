# Sprint 1: 仿真基建与基础 Locomotion

## 目标
- 在物理仿真器中渲染出机器人
- 通过 RL 训练出能在仿真中直线行走且不摔倒的 Policy

## 环境配置

### 云端实例
- GPU: NVIDIA RTX 4090 24GB
- 驱动: 575.57.08
- CUDA: 12.9 (host) / 12.8 (container PyTorch)
- 内存: 62GB
- 磁盘: 153GB 可用
- OS: Ubuntu 22.04 (host), Isaac Sim Docker Container

### 软件栈
- **Isaac Sim**: 5.1.0（Docker 容器: `nvcr.io/nvidia/isaac-sim:5.1.0`）
- **Isaac Lab**: 2.3.2.post1（pip 安装）
- **rsl_rl**: 3.1.2（ETH Zurich 的 RL 库）
- **skrl**: 1.4.3（备用 RL 库）
- **PyTorch**: 2.7.0+cu128
- **Python**: 3.11.13（容器内）

### SSH 配置
```
Host cloud-gpu
    HostName <YOUR_SERVER_IP>
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519_cloud
```

## 训练配置

### 任务
- **环境**: `Isaac-Velocity-Flat-Unitree-Go2-v0`
- **机器人**: Unitree Go2（四足机器人）
- **地形**: 平地（plane）
- **并行环境数**: 4096

### 算法: PPO (Proximal Policy Optimization)
- **网络**: Actor [128, 128, 128], Critic [128, 128, 128], ELU 激活
- **学习率**: 0.001（adaptive schedule）
- **PPO clip**: 0.2
- **Discount (γ)**: 0.99
- **GAE (λ)**: 0.95
- **最大迭代**: 300

### 观察空间 (48 维)
| 观察 | 维度 | 描述 |
|------|------|------|
| base_lin_vel | 3 | 基座线速度 |
| base_ang_vel | 3 | 基座角速度 |
| projected_gravity | 3 | 投影重力方向 |
| velocity_commands | 3 | 速度指令 |
| joint_pos | 12 | 关节位置 |
| joint_vel | 12 | 关节速度 |
| actions | 12 | 上一步动作 |

### 动作空间 (12 维)
- 12 个关节位置目标（4 条腿 × 3 关节/腿）

### 奖励函数
| 奖励项 | 权重 | 作用 |
|--------|------|------|
| track_lin_vel_xy_exp | 1.5 | 跟踪线速度指令 |
| track_ang_vel_z_exp | 0.75 | 跟踪角速度指令 |
| lin_vel_z_l2 | -2.0 | 抑制垂直速度 |
| ang_vel_xy_l2 | -0.05 | 抑制俯仰/翻滚 |
| dof_torques_l2 | -0.0002 | 抑制过大力矩 |
| dof_acc_l2 | -2.5e-7 | 抑制关节加速度 |
| action_rate_l2 | -0.01 | 动作平滑性 |
| feet_air_time | 0.25 | 鼓励步态 |
| flat_orientation_l2 | -2.5 | 保持水平 |

## 训练结果

### 训练曲线
- 初始奖励: -0.52
- 最终奖励: **35.47**
- 最高奖励: 35.47（第 299 次迭代）
- 收敛迭代: ~150（之后趋于稳定在 33-35 区间）
- 总训练时间: **314.58 秒**（约 5 分钟）
- 总步数: 29,491,200
- 训练速度: ~10万 steps/s

### 关键指标（最终）
| 指标 | 值 | 说明 |
|------|-----|------|
| Mean Reward | 35.47 | 远高于初始值 |
| Episode Length | 1000.0 | 达到最大值，不摔倒 |
| Noise Std | 0.35 | 从 1.0 降到 0.35，策略收敛 |
| error_vel_xy | 0.175 | 速度跟踪误差小 |
| error_vel_yaw | 0.340 | 偏航跟踪误差 |
| Timeout 终止率 | 99.56% | 几乎不因碰撞终止 |
| Base Contact 终止率 | 0.44% | 极低的摔倒率 |

### 训练曲线图
![Training Curves](../sprint-1/results/go2_flat_ppo_2026-03-02/training_curves.png)

### DoD 验证
- [x] 仿真器中成功渲染出机器人模型（Go2 加载并渲染，确认 mesh 和 articulation）
- [x] PPO 训练收敛，reward 曲线上升并趋于稳定（-0.52 → 35.47，~150 步后稳定）
- [x] 机器人在仿真中能直线行走不摔倒（episode 长度 = 1000，timeout 终止率 99.56%）

## 关键学习

### Isaac Lab 训练流程
1. SimulationApp 初始化 → 导入模块
2. 通过 gymnasium 注册环境（`gym.register`）
3. 配置 hydra 管理超参数
4. RslRlVecEnvWrapper 包装环境
5. OnPolicyRunner 执行训练循环
6. 日志记录到 tensorboard

### 遇到的问题
1. rsl_rl 5.0.0 与 Isaac Lab 2.3.2 配置格式不兼容 → 降级到 3.1.2
2. Docker 容器内无 git → `GIT_PYTHON_REFRESH=quiet` 环境变量
3. Docker cp 权限问题 → 使用 pip install URL 方式安装
4. 嵌套 SSH 命令引号转义 → 使用脚本文件方式

## 文件清单
- `sprint-1/configs/go2_flat_ppo.yaml` — 训练配置说明文件
- `sprint-1/scripts/check_env.sh` — 云端环境检查脚本
- `sprint-1/scripts/render_test.py` — 渲染验证脚本
- `sprint-1/scripts/install_isaaclab.sh` — Isaac Lab 安装脚本
- `sprint-1/scripts/install_isaaclab_tasks.sh` — Isaac Lab 子包安装脚本
- `sprint-1/scripts/verify_install.py` — 安装验证脚本
- `sprint-1/scripts/plot_reward.py` — 训练曲线绘制脚本
- `sprint-1/results/go2_flat_ppo_2026-03-02/` — 训练结果目录
  - `model_299.pt` — 最终模型权重
  - `model_0.pt` — 初始模型权重
  - `params/agent.yaml` — 算法配置
  - `params/env.yaml` — 环境配置
  - `events.out.tfevents.*` — TensorBoard 日志
  - `training_curves.png` — 训练曲线图
  - `rl-video-step-0.mp4` — 评估视频

## 云端文件位置
- 训练日志: `/isaac-sim/logs/rsl_rl/unitree_go2_flat/2026-03-02_06-29-29/`
- 训练脚本: `/tmp/training/rsl_rl/` (容器内)
- Isaac Lab 源码: `/home/ubuntu/IsaacLab/` (宿主机)
