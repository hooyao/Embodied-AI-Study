# Embodied AI Study

## Project Overview

This repository is a **personal learning workspace** for Embodied AI (具身智能), following a structured 2-month intensive study plan. The focus is on legged robot locomotion and whole-body control, using the **Simulation-First -> Sim2Real -> Hardware Deployment** pipeline.

## 开发环境

- **本地**：Windows 11（用于代码编辑、文档管理、git 操作）
- **训练**：云端租赁 GPU 实例（RTX 4090 / RTX 5090），Linux 环境
- **工作流**：本地编写代码 -> 同步到云端 -> 云端训练 -> 结果拉回本地

## Study Plan (5 Sprints)

### Sprint 1 (Week 1-2): Simulation Infrastructure & Basic Locomotion
- Setup: Ubuntu (WSL2/dual-boot), CUDA/cuDNN
- Tools: MuJoCo, NVIDIA Isaac Lab, `legged_gym`
- Goal: Train PPO policy for flat-ground walking
- DoD: Robot walks straight without falling in simulation

### Sprint 2 (Week 3-4): Motion Priors & Imitation (AMP & Mimic)
- Repos: `Noetix-Robotics/noetix_e1_lab`, `project-instinct`, `HybridRobotics/whole_body_tracking`
- Goal: Natural motion via AMP, depth perception as observation
- DoD: Reproduce AMP training, swap reference motions, train new policies

### Sprint 3 (Week 5): State Estimation & Sim2Real Preparation
- Repo: `InternRobotics/HIMLoco`
- Goal: Teacher-Student distillation, remove privileged information dependency
- DoD: Policy walks stably using only estimator (no privileged info)

### Sprint 4 (Week 6-7): Whole-Body Control & Teleoperation
- Repos: `LeCAR-Lab/BFM-Zero`, `NVlabs/GROOT-WholeBodyControl`
- Goal: Full-body coordination (arms + torso + legs), teleoperation mapping
- DoD: Robot reaches arbitrary 6D end-effector poses via whole-body coordination

### Sprint 5 (Week 8): Domain Randomization & Ecosystem Research
- Goal: Implement domain randomization, research Unitree developer ecosystem
- DoD: 90%+ success rate under extreme randomization; Unitree dev architecture doc

## Repository Structure

```
Embodied-AI-Study/
├── CLAUDE.md                 # This file - project conventions
├── docs/                     # Notes, papers, architecture diagrams
│   ├── concepts/             # Core concept explanations
│   ├── papers/               # Paper reading notes
│   └── sprint-logs/          # Weekly progress logs
├── envs/                     # Environment setup scripts & configs
├── sprint-1/                 # Basic locomotion experiments
│   ├── configs/              # Training configs (YAML/Python)
│   ├── scripts/              # Training & evaluation scripts
│   └── results/              # Trained weights, plots, videos
├── sprint-2/                 # AMP & Mimic experiments
├── sprint-3/                 # State estimation & Sim2Real
├── sprint-4/                 # Whole-body control
├── sprint-5/                 # Domain randomization
└── references/               # Cloned or linked external repos
```

## Key Technical Concepts

| Concept | Description |
|---------|-------------|
| MuJoCo | Physics engine optimized for contact dynamics & RL training |
| Isaac Lab | NVIDIA GPU-accelerated robot simulation framework |
| PPO | Proximal Policy Optimization - standard RL algorithm for locomotion |
| AMP | Adversarial Motion Priors - GAN-style natural motion enforcement |
| Mimic | Imitation learning from motion capture reference trajectories |
| Domain Randomization | Random env params (friction, mass, delay) for Sim2Real robustness |
| Teacher-Student | Train with privileged info (teacher), distill to sensor-only (student) |
| URDF | Unified Robot Description Format - standard robot model format |
| Estimator | Fuse noisy sensor data (IMU, encoders) to estimate true robot state |

## Conventions

- **默认语言**：本仓库的默认语言为**简体中文**，包括文档、笔记、commit message、代码注释等一律使用中文
- **实验管理**：每个实验在对应 sprint 目录下建立描述性子文件夹
- **配置文件**：使用 YAML 管理训练超参数，所有配置纳入版本控制
- **实验结果**：每次运行保存训练曲线、视频录制和最终权重
- **Git 提交**：格式 `sprint-N: <简要描述>`

## Key Repos to Track

- [legged_gym](https://github.com/leggedrobotics/legged_gym) - ETH Zurich baseline
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) - NVIDIA simulation
- [noetix_e1_lab](https://github.com/Noetix-Robotics/noetix_e1_lab) - AMP implementation
- [HIMLoco](https://github.com/InternRobotics/HIMLoco) - State estimation
- [BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero) - Whole-body teleoperation
- [unitree_sdk2](https://github.com/unitree-robotics/unitree_sdk2) - Unitree hardware SDK
