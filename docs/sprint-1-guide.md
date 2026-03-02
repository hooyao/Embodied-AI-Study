# Sprint 1 复现指南：从零训练一个会走路的四足机器人

## 前提条件

本指南假设你已经完成了环境配置：
- 云端 Isaac Sim 5.1.0 Docker 容器正常运行
- 容器内已安装 `isaaclab 2.3.2`、`rsl-rl-lib 3.1.2`、`skrl 1.4.3`
- Isaac Lab 源码已克隆到宿主机 `/home/ubuntu/IsaacLab/`
- 训练脚本已复制到容器 `/tmp/training/rsl_rl/`（train.py, play.py, cli_args.py）
- SSH 免密登录已配置（`ssh cloud-gpu`）

所有训练命令都在**云端服务器**上执行，格式为：
```bash
ssh cloud-gpu '...'                    # 在宿主机执行
sudo docker exec isaac-sim ...         # 在容器内执行
```

---

## 第一部分：理解我们在做什么

### 1.1 目标

训练一个 **PPO（Proximal Policy Optimization）** 强化学习策略，让 **Unitree Go2**（四足机器人）在平地上稳定行走。

用人话说：我们要让一个虚拟的四条腿小狗学会走路，不摔倒，还能听指令转弯。

### 1.2 为什么选 Go2？

- Unitree Go2 是目前市面上最流行的四足机器人之一，价格相对亲民
- Isaac Lab 内置了 Go2 的完整模型（URDF/USD）和预配置环境
- 四足机器人是 locomotion 领域的经典入门对象：比双足简单（稳定性好），但比轮式复杂（需要协调步态）

### 1.3 为什么选 PPO？

PPO 是目前机器人 locomotion 领域最常用的 RL 算法，原因：
- **稳定**：clipped objective 防止策略更新幅度过大，训练不容易崩
- **高效**：on-policy 算法中 sample efficiency 最好的之一
- **并行友好**：可以同时跑几千个环境收集数据，充分利用 GPU
- **业界标准**：几乎所有 legged locomotion 论文都用 PPO 作为 baseline

### 1.4 整体流程

```
┌─────────────────────────────────────────────────────────┐
│                    训练循环（300 轮）                       │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│  │ 4096 个   │───▶│  收集经验  │───▶│  PPO 更新  │──┐      │
│  │ 并行环境   │    │ (24 步)   │    │ (5 epochs)│  │      │
│  └──────────┘    └──────────┘    └──────────┘  │      │
│       ▲                                         │      │
│       └─────────────────────────────────────────┘      │
│                                                          │
│  每轮产出：更新后的策略网络（actor + critic）               │
└─────────────────────────────────────────────────────────┘
```

---

## 第二部分：理解环境设计

在开始训练之前，你需要理解"环境"是怎么定义的。RL 中的环境 = 观察 + 动作 + 奖励 + 终止条件。

### 2.1 观察空间（Observation Space）—— 机器人能"看到"什么

策略网络的输入是一个 **48 维向量**，包含 7 种信息：

| 观察项 | 维度 | 含义 | 为什么需要 |
|--------|------|------|-----------|
| `base_lin_vel` | 3 | 基座线速度 (x, y, z) | 知道自己在往哪走、多快 |
| `base_ang_vel` | 3 | 基座角速度 (roll, pitch, yaw) | 知道自己在转吗、翻了吗 |
| `projected_gravity` | 3 | 重力方向在机体坐标系的投影 | 知道自己是不是歪了（相当于 IMU） |
| `velocity_commands` | 3 | 速度指令 (vx, vy, ω_yaw) | 知道人让它往哪走 |
| `joint_pos` | 12 | 12 个关节的当前角度（相对默认站姿） | 知道腿弯了多少 |
| `joint_vel` | 12 | 12 个关节的当前角速度 | 知道腿在动吗、多快 |
| `actions` | 12 | 上一步输出的动作 | 帮助策略保持动作连贯性 |

**重要细节 —— 观测噪声**：训练时会对观测加噪声（模拟真实传感器的不精确），这是 Sim2Real 的关键技术之一：

```
base_lin_vel:  ±0.1 均匀噪声
base_ang_vel:  ±0.2 均匀噪声
projected_gravity: ±0.05 均匀噪声
joint_pos:     ±0.01 均匀噪声
joint_vel:     ±1.5 均匀噪声
```

### 2.2 动作空间（Action Space）—— 机器人能"做"什么

策略输出一个 **12 维向量**，代表 12 个关节的**目标位置偏移量**。

Go2 有 4 条腿，每条腿 3 个关节：
```
每条腿：hip（髋关节）→ thigh（大腿）→ calf（小腿）
4 条腿：FL（左前）、FR（右前）、RL（左后）、RR（右后）
共计：4 × 3 = 12 个关节
```

动作会乘以一个 **scale = 0.25**，然后加到默认站姿角度上，作为 PD 控制器的目标位置。这个 scale 很重要：
- 太大（如 1.0）：动作幅度过大，机器人容易做出极端动作摔倒
- 太小（如 0.05）：机器人活动范围受限，学不会走路
- 0.25 是经验值，对 Go2 这种小型四足效果好

### 2.3 奖励函数（Reward Function）—— 什么算"做得好"

奖励函数是 RL 训练的灵魂。它告诉机器人什么行为是好的、什么是坏的。

Isaac Lab 使用**多项加权奖励**的方式，每项都有明确的物理含义：

#### 正奖励（鼓励做的事）

| 奖励项 | 权重 | 公式思路 | 物理含义 |
|--------|------|---------|---------|
| `track_lin_vel_xy_exp` | **+1.5** | exp(-‖v_cmd - v_actual‖²/σ²) | 跟踪线速度指令越准，奖励越高 |
| `track_ang_vel_z_exp` | **+0.75** | exp(-‖ω_cmd - ω_actual‖²/σ²) | 跟踪角速度指令越准，奖励越高 |
| `feet_air_time` | **+0.25** | 脚离地时间是否合理 | 鼓励交替抬腿（形成步态） |

#### 负奖励 / 惩罚（不希望做的事）

| 奖励项 | 权重 | 公式思路 | 物理含义 |
|--------|------|---------|---------|
| `flat_orientation_l2` | **-2.5** | ‖gravity_projection_xy‖² | 惩罚身体倾斜（保持水平） |
| `lin_vel_z_l2` | **-2.0** | v_z² | 惩罚上下弹跳 |
| `ang_vel_xy_l2` | **-0.05** | ω_x² + ω_y² | 惩罚俯仰和翻滚 |
| `action_rate_l2` | **-0.01** | ‖a_t - a_{t-1}‖² | 惩罚动作剧变（鼓励平滑运动） |
| `dof_torques_l2` | **-0.0002** | Σ τ² | 惩罚过大力矩（节能） |
| `dof_acc_l2` | **-2.5e-7** | Σ α² | 惩罚关节加速度（保护电机） |

**设计哲学**：
- 权重最大的正奖励是"跟踪速度指令"（1.5），这是主要目标
- 权重最大的负奖励是"保持水平"（-2.5）和"不弹跳"（-2.0），这是安全约束
- 力矩和加速度的惩罚权重很小，只是轻微引导，不会压过主目标

### 2.4 终止条件（Termination）—— 什么时候"游戏结束"

| 条件 | 类型 | 含义 |
|------|------|------|
| `time_out` | 超时 | episode 达到 20 秒（1000 步）自然结束 |
| `base_contact` | 失败 | 机器人的"base"（躯干）碰到地面 = 摔倒，立即结束 |

超时终止是**正常结束**（机器人走了 20 秒没摔倒），base_contact 是**失败终止**（摔了）。

### 2.5 速度指令（Command）

训练时，系统会随机生成速度指令让机器人跟踪：
```
前后速度 vx: [-1.0, 1.0] m/s
左右速度 vy: [-1.0, 1.0] m/s
转向角速度 ωz: [-1.0, 1.0] rad/s
```
每 10 秒重新采样一次指令。这样机器人学会的不只是直走，而是能跟踪任意方向的速度指令。

### 2.6 Domain Randomization（域随机化）

训练时会对环境参数做轻微随机化，增强策略的鲁棒性：
- **摩擦系数**：static=0.8, dynamic=0.6（固定值，未随机化）
- **基座质量偏移**：[-1.0, +3.0] kg（模拟负载变化）
- **初始位姿**：x, y ∈ [-0.5, 0.5] m，yaw ∈ [-π, π]

---

## 第三部分：理解 PPO 算法配置

### 3.1 网络结构

```
Actor（策略网络）：
  输入(48) → Linear(128) → ELU → Linear(128) → ELU → Linear(128) → ELU → Linear(12) → 输出

Critic（价值网络）：
  输入(48) → Linear(128) → ELU → Linear(128) → ELU → Linear(128) → ELU → Linear(1) → 输出
```

- **Actor** 输出 12 个关节的目标位置偏移（加上高斯噪声用于探索）
- **Critic** 输出对当前状态的价值估计（用于计算优势函数）
- 两个网络结构相同但**参数独立**
- 激活函数用 ELU（比 ReLU 平滑，有助于连续控制任务）

### 3.2 PPO 超参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `num_steps_per_env` | 24 | 每个环境走 24 步后做一次 PPO 更新 |
| `num_learning_epochs` | 5 | 每批数据重复学习 5 遍 |
| `num_mini_batches` | 4 | 把数据分成 4 份做 mini-batch 更新 |
| `learning_rate` | 0.001 | 初始学习率 |
| `schedule` | adaptive | 根据 KL 散度自动调节学习率 |
| `desired_kl` | 0.01 | 目标 KL 散度（控制策略更新幅度） |
| `gamma` | 0.99 | 折扣因子（agent 多重视未来奖励） |
| `lam` | 0.95 | GAE λ（优势估计的 bias-variance 权衡） |
| `clip_param` | 0.2 | PPO clip 范围（限制策略更新幅度） |
| `entropy_coef` | 0.01 | 熵正则化系数（鼓励探索） |
| `max_iterations` | 300 | 总共训练 300 轮 |
| `init_noise_std` | 1.0 | 初始动作噪声标准差（高 = 多探索） |

**每轮数据量**：4096 环境 × 24 步 = **98,304 个 transition**

### 3.3 数据流

```
一轮训练（1 iteration）的完整流程：

1. Collection（收集阶段）～0.9s
   ├── 4096 个环境并行运行 24 步
   ├── 收集 (obs, action, reward, done) 数据
   └── 产出 98,304 个 transition

2. Learning（学习阶段）～0.1s
   ├── 计算 GAE 优势估计
   ├── 将数据分成 4 个 mini-batch
   ├── 每个 mini-batch 做 5 次 PPO 更新
   └── 共 4 × 5 = 20 次梯度更新

3. 更新策略网络和价值网络
4. 记录日志（reward、episode length 等）
5. 每 50 轮保存一次 checkpoint
```

---

## 第四部分：动手训练

### 4.1 启动训练

```bash
ssh cloud-gpu 'sudo docker exec \
  -e GIT_PYTHON_REFRESH=quiet \
  -w /isaac-sim \
  isaac-sim \
  /isaac-sim/python.sh /tmp/training/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 300'
```

**参数解释**：

| 参数 | 值 | 为什么 |
|------|-----|-------|
| `-e GIT_PYTHON_REFRESH=quiet` | — | 容器内没有 git，跳过 GitPython 报错 |
| `-w /isaac-sim` | — | 设置工作目录，让日志保存到 `/isaac-sim/logs/` |
| `/isaac-sim/python.sh` | — | 使用 Isaac Sim 内置 Python（包含所有依赖） |
| `--task` | `Isaac-Velocity-Flat-Unitree-Go2-v0` | 选择 Go2 平地速度跟踪环境 |
| `--headless` | — | 不开渲染窗口，纯 GPU 计算，训练快 10 倍以上 |
| `--num_envs` | 4096 | 并行环境数。4090 可以轻松跑 4096 个 |
| `--max_iterations` | 300 | 训练 300 轮。Go2 平地任务 300 轮足够收敛 |

### 4.2 理解训练输出

训练开始后你会看到类似以下输出（每轮一次）：

```
################################################################################
                       Learning iteration 150/300

                       Computation: 98230 steps/s (collection: 0.876s, learning 0.125s)
             Mean action noise std: 0.45
          Mean value_function loss: 0.0100
               Mean surrogate loss: -0.0050
                 Mean entropy loss: 6.5000
                       Mean reward: 28.50
               Mean episode length: 995.00
Episode_Reward/track_lin_vel_xy_exp: 1.20
Episode_Reward/track_ang_vel_z_exp: 0.60
       Episode_Reward/lin_vel_z_l2: -0.03
      Episode_Reward/ang_vel_xy_l2: -0.08
     Episode_Reward/dof_torques_l2: -0.09
         Episode_Reward/dof_acc_l2: -0.06
     Episode_Reward/action_rate_l2: -0.07
      Episode_Reward/feet_air_time: -0.03
Episode_Reward/flat_orientation_l2: -0.01
      Episode_Termination/time_out: 0.995
  Episode_Termination/base_contact: 0.005
                   Total timesteps: 14745600
                    Iteration time: 1.00s
                      Time elapsed: 00:02:30
                               ETA: 00:02:30
```

**关键指标怎么看**：

| 指标 | 健康范围 | 说明 |
|------|---------|------|
| **Mean reward** | 持续上升，最终 30-36 | 最重要的指标。如果下降或停滞说明有问题 |
| **Mean episode length** | 接近 1000 | =1000 意味着跑满 20 秒不摔倒。初期可能只有 100-300 |
| **Mean action noise std** | 从 1.0 逐渐降到 0.3-0.4 | 策略从探索过渡到利用。降得太快→过早收敛；不降→没学到东西 |
| **value_function loss** | 初期高，后期 <0.01 | 价值网络的预测精度。越低越好 |
| **time_out 比例** | 越接近 1.0 越好 | 1.0 = 所有机器人都跑满 20 秒 |
| **base_contact 比例** | 越接近 0 越好 | 0 = 没有机器人摔倒 |
| **track_lin_vel_xy_exp** | 越高越好，理论最大 1.5 | 速度跟踪精度。>1.0 算不错 |
| **Computation steps/s** | 80k-100k+ | 训练吞吐量。太低说明 GPU 利用率不足 |

### 4.3 训练过程的三个阶段

**第一阶段（0-50 轮）—— 学会站立**
- reward 从负数迅速上升到 10+
- episode length 从 100-300 跳到 800+
- 机器人从到处乱摔到能站稳
- noise std 缓慢下降

**第二阶段（50-150 轮）—— 学会行走**
- reward 从 10 上升到 30+
- episode length 稳定在 950-1000
- 机器人开始能跟踪速度指令
- track_lin_vel_xy_exp 从 0.5 上升到 1.2+

**第三阶段（150-300 轮）—— 精细调优**
- reward 在 33-36 之间波动，趋于稳定
- episode length 稳定在 1000
- 速度跟踪误差继续减小
- noise std 降到 0.35 左右

### 4.4 预期训练时间

在 RTX 4090 上：
- 每轮约 1 秒
- 300 轮 ≈ **5 分钟**
- 总步数 ≈ 2940 万

### 4.5 Checkpoint 保存

训练会自动保存 checkpoint 到 `/isaac-sim/logs/rsl_rl/unitree_go2_flat/<timestamp>/`：
- `model_0.pt` — 初始随机策略
- `model_50.pt`, `model_100.pt`, ... — 每 50 轮保存
- `model_299.pt` — 最终模型
- `params/agent.yaml` — 算法配置
- `params/env.yaml` — 环境配置
- `events.out.tfevents.*` — TensorBoard 日志

---

## 第五部分：评估训练结果

### 5.1 查看训练日志目录

```bash
ssh cloud-gpu 'sudo docker exec isaac-sim \
  ls /isaac-sim/logs/rsl_rl/unitree_go2_flat/'
```

输出形如 `2026-03-02_06-29-29`，这就是你的训练 run 目录名。下面用 `<RUN_DIR>` 代替。

### 5.2 录制评估视频

```bash
ssh cloud-gpu 'sudo docker exec \
  -e GIT_PYTHON_REFRESH=quiet \
  -w /isaac-sim \
  isaac-sim \
  /isaac-sim/python.sh /tmp/training/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
    --headless \
    --enable_cameras \
    --video \
    --video_length 300 \
    --num_envs 16 \
    --checkpoint "/isaac-sim/logs/rsl_rl/unitree_go2_flat/<RUN_DIR>/model_299.pt"'
```

**参数解释**：

| 参数 | 值 | 为什么 |
|------|-----|-------|
| `--task` | `...-Play-v0` | 注意末尾是 **Play** 版本，环境数更少、关闭了随机化 |
| `--enable_cameras` | — | 启用相机渲染（录视频必需） |
| `--video` | — | 开启视频录制 |
| `--video_length` | 300 | 录制 300 帧（约 15 秒） |
| `--num_envs` | 16 | 评估时不需要太多环境 |
| `--checkpoint` | 绝对路径 | 指定加载哪个模型文件 |

视频会保存在：`/isaac-sim/logs/rsl_rl/unitree_go2_flat/<RUN_DIR>/videos/play/rl-video-step-0.mp4`

### 5.3 将视频下载到本地

```bash
# 从容器复制到宿主机
ssh cloud-gpu 'sudo docker cp \
  isaac-sim:/isaac-sim/logs/rsl_rl/unitree_go2_flat/<RUN_DIR>/videos/play/rl-video-step-0.mp4 \
  /home/ubuntu/'

# 从宿主机下载到本地
scp cloud-gpu:/home/ubuntu/rl-video-step-0.mp4 ./sprint-1/results/
```

### 5.4 下载模型和日志

```bash
# 创建本地目录
mkdir -p sprint-1/results/go2_flat_ppo/

# 下载最终模型
scp cloud-gpu:/home/ubuntu/IsaacLab/logs/rsl_rl/unitree_go2_flat/<RUN_DIR>/model_299.pt \
    sprint-1/results/go2_flat_ppo/

# 下载配置
scp -r cloud-gpu:/home/ubuntu/IsaacLab/logs/rsl_rl/unitree_go2_flat/<RUN_DIR>/params/ \
    sprint-1/results/go2_flat_ppo/

# 下载 TensorBoard 日志
scp cloud-gpu:/home/ubuntu/IsaacLab/logs/rsl_rl/unitree_go2_flat/<RUN_DIR>/events.out.* \
    sprint-1/results/go2_flat_ppo/
```

### 5.5 查看 TensorBoard 训练曲线

**方式 A：在云端启动 TensorBoard，本地端口转发**

```bash
# 终端 1：启动 TensorBoard（在容器内）
ssh cloud-gpu 'sudo docker exec -d isaac-sim \
  /isaac-sim/python.sh -m tensorboard.main \
  --logdir /isaac-sim/logs/rsl_rl/unitree_go2_flat/ \
  --host 0.0.0.0 --port 6006'

# 终端 2：本地端口转发
ssh -L 6006:localhost:6006 cloud-gpu

# 然后在本地浏览器打开：http://localhost:6006
```

**方式 B：本地用 Python 脚本绘图**

```bash
# 需要安装依赖
pip install tensorboard matplotlib

# 运行绘图脚本
python sprint-1/scripts/plot_reward.py
# 输出图片到 sprint-1/results/go2_flat_ppo/training_curves.png
```

---

## 第六部分：验收标准（Definition of Done）

完成以下三项即通过 Sprint 1：

### DoD 1: 仿真器中成功渲染出机器人模型

**验证方法**：训练日志中出现 Go2 的 mesh 加载信息
```
[Warning] [omni.hydra] Mesh '/World/.../Go2/base_white/visuals' ...
[Warning] [omni.physx.plugin] Detected an articulation at /World/.../Go2/base ...
```
看到这些就说明 Go2 的 USD 模型被正确加载到物理引擎中了。

### DoD 2: PPO 训练收敛，reward 曲线上升并趋于稳定

**验证方法**：查看训练曲线
- Mean reward 从负数上升到 30+
- 后 100 轮 reward 波动在 ±3 以内（稳定）
- Noise std 从 1.0 降到 0.4 以下

### DoD 3: 机器人在仿真中能直线行走不摔倒

**验证方法**：查看最后几轮训练输出
- `Mean episode length` ≈ 1000（跑满 20 秒）
- `Episode_Termination/time_out` > 0.99（99% 以上正常结束）
- `Episode_Termination/base_contact` < 0.01（不到 1% 摔倒）
- 评估视频中能看到 16 只 Go2 在平地上稳定行走

---

## 第七部分：常见问题排查

### Q: 训练时报 `KeyError: 'actor'`
**原因**：rsl_rl 版本不对。Isaac Lab 2.3.2 需要 `rsl-rl-lib==3.1.2`，不能用 5.0.0。
```bash
sudo docker exec isaac-sim /isaac-sim/python.sh -m pip install "rsl-rl-lib==3.1.2"
```

### Q: 报 `ImportError: Failed to initialize: Bad git executable`
**原因**：容器内没有 git，rsl_rl 依赖的 GitPython 会报错。
**解决**：在 docker exec 时加 `-e GIT_PYTHON_REFRESH=quiet`。

### Q: Reward 一直不上升，停在 0 附近
**可能原因**：
1. `num_envs` 太少（建议 ≥ 2048）
2. 学习率太高或太低（默认 0.001 通常没问题）
3. 奖励函数权重有误（用默认配置就好）

### Q: 训练很慢，每轮 > 3 秒
**可能原因**：
1. `num_envs` 太多导致 GPU 显存不足（4090 建议 4096）
2. 没有加 `--headless`（渲染会极大拖慢速度）
3. GPU 被其他进程占用（用 `nvidia-smi` 检查）

### Q: 找不到 checkpoint 文件
**注意**：play.py 的 `--checkpoint` 参数需要传**绝对路径**，不能只传文件名。

---

## 第八部分：实际训练结果参考

以下是我实际训练得到的数据，供你参考和对比：

| 指标 | 初始值 | 最终值（第 299 轮） |
|------|--------|-------------------|
| Mean Reward | -0.52 | **35.47** |
| Episode Length | ~200 | **1000.0** |
| Action Noise Std | 1.0 | 0.35 |
| track_lin_vel_xy | 0.14 | **1.43** |
| track_ang_vel_z | 0.25 | **0.64** |
| error_vel_xy | 2.05 | **0.175** |
| time_out 比例 | 0.21 | **0.996** |
| base_contact 比例 | 0.79 | **0.004** |
| 训练总时间 | — | **314 秒（5 分 15 秒）** |
| 训练总步数 | — | **29,491,200** |
| 训练速度 | — | **~10 万 steps/s** |

训练曲线图见 `sprint-1/results/go2_flat_ppo_2026-03-02/training_curves.png`，评估视频见同目录下的 `rl-video-step-0.mp4`。

---

## 快速命令参考

```bash
# ===== 完整的一键复现命令序列 =====

# 1. 训练（~5 分钟）
ssh cloud-gpu 'sudo docker exec \
  -e GIT_PYTHON_REFRESH=quiet -w /isaac-sim isaac-sim \
  /isaac-sim/python.sh /tmp/training/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --headless --num_envs 4096 --max_iterations 300'

# 2. 查看训练目录（记住 <RUN_DIR>）
ssh cloud-gpu 'sudo docker exec isaac-sim \
  ls -lt /isaac-sim/logs/rsl_rl/unitree_go2_flat/ | head -3'

# 3. 录制评估视频
ssh cloud-gpu 'sudo docker exec \
  -e GIT_PYTHON_REFRESH=quiet -w /isaac-sim isaac-sim \
  /isaac-sim/python.sh /tmp/training/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 \
  --headless --enable_cameras --video --video_length 300 --num_envs 16 \
  --checkpoint "/isaac-sim/logs/rsl_rl/unitree_go2_flat/<RUN_DIR>/model_299.pt"'

# 4. 下载视频和模型到本地
ssh cloud-gpu 'sudo docker cp \
  isaac-sim:/isaac-sim/logs/rsl_rl/unitree_go2_flat/<RUN_DIR>/videos/play/rl-video-step-0.mp4 \
  /home/ubuntu/'
scp cloud-gpu:/home/ubuntu/rl-video-step-0.mp4 ./sprint-1/results/
```
