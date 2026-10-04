# 技术说明

本文记录项目的技术路线、关键设计与踩坑，供复现参考。

---

## 1. 背景

ROBOTAC「拳击力量挑战」赛项要求 Unitree G1（29 DOF）人形机器人复现人类拳击动作，
并对其击打力度做仿真量化验证。官方技术手册第 3.5 节给出「击打力度仿真验证」的基础思路
（motion / actual 两种靶子模式），本项目在其上完整实现并扩展。

**整体路线**：

```
人类拳击动作数据(.npz) → mjlab Tracking 任务 + PPO 训练 → 运动跟踪策略
                                                              ↓
                                   回放时注入拳靶 + ContactSensor → 力度验证
                                                              ↓
                                            导出 ONNX + 归一化参数 → 部署
```

---

## 2. Tracking 任务

本质是**运动模仿（imitation）**而非从零 RL：

- `MotionCommand` 按物理时间步提供参考轨迹（各 tracked body 的世界位姿 / 速度、关节位置）
- 策略目标 = 让机器人 tracked body 跟踪参考轨迹，误差越小奖励越高
- 奖励核 `exp(-error / std²)`（高斯核）

**跟踪的 14 个 body**（anchor = `torso_link`）：
`pelvis`、`torso_link`、左右 `hip_roll / knee / ankle_roll`、左右 `shoulder_roll / elbow / wrist_yaw`。

### 观测空间（154 维）
actor 组（策略输入）：command、motion_anchor_ori_b、base_ang_vel、joint_pos、joint_vel、actions 等；
No-State-Estimation 版本移除 `motion_anchor_pos_b` 与 `base_lin_vel`。
critic 组额外含全身 body 相对位姿（特权信息，不对称 actor-critic）。

### 动作空间
29 维关节位置目标 → 底层 PD 转力矩。

---

## 3. 奖励函数

| 奖励项 | 权重 | 作用 |
|--------|:----:|------|
| motion_global_root_pos | 1.5 | 躯干全局位置 |
| motion_global_root_ori | 0.5 | 躯干全局朝向 |
| motion_body_pos / _ori / _lin_vel / _ang_vel | 1.0 | 全身跟踪 |
| **motion_body_pos_arms / _ori_arms / _lin_vel_arms** | **4.0** | 手臂 6 关键点（4x）|
| action_rate_l2 | -0.1 | 动作变化率 |
| joint_limit | -10.0 | 关节限位 |
| self_collisions | -10.0 | 自碰撞（阈值 10N）|
| **fist_velocity** | **2.0** | 自定义拳速激励 |

**设计要点**：
- 腿是支撑、误差小、易收敛；手臂是末端、误差大、收敛慢 → 手臂单独加权
- 拳速激励权重需与跟踪项同量级：设到 3.0 会让策略牺牲跟踪精度、动作崩溃（force=0）
- `action_acc_l2`（加速度惩罚）会连出拳的正常加速一起压住 → 移除

---

## 4. 击打力度验证系统（核心）

### 拳靶
圆柱（半径 0.08 m，高 0.03 m，质量 2 kg），仅 `play=True` 时注入场景。

### 力传感器
```
ContactSensorCfg(
    primary=ContactMatch(mode="body", pattern="right_wrist_yaw_link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="punch_target", entity="punch_target"),
    fields=("found", "force"), reduce="netforce")
```

### 三种靶子跟踪模式
| 模式 | 靶子位置 | 说明 |
|------|---------|------|
| `motion` | 跟参考轨迹当前帧拳位 | 力**虚高**（穿透伪影）|
| `actual` | 跟机器人实际拳头 | 拳追不上靶 |
| `fixed` | 固定标定位 | **真实撞击力** |

**关键结论**：`motion` 模式靶子与参考拳位重叠，跟踪好时接触求解器判定穿透 → 反推几百牛的假力。
`fixed` 模式靶子独立静止，拳头必须真实撞上，读数才有物理意义。

### 靶位标定
三维网格扫描 → 取力最大点 → 附近 ±0.01 m 细扫收敛。
最优靶位 `(-0.220, -0.260, 0.880)`，峰值 303 N。
靶子小（r=8cm），偏 ≈1 cm 力度就从 ~300 N 掉到 ~110 N。

---

## 5. 踩坑记录

| 问题 | 根因 | 解决 |
|------|------|------|
| `free(): invalid pointer` 崩溃 | osmesa 软件渲染内存冲突 | 设 `MUJOCO_GL=egl` |
| `mjtEnableBit` 无 `mjENBL_MULTICCD` | mujoco 3.10 缺枚举，mujoco-warp 需要 | 运行时 monkey-patch 补位 |
| 机器人"像喝醉"乱动 | 训练/回放任务名不匹配 → 观测组不同 → 归一化器错配 | 统一任务名 |
| 靶子不出力 | 默认靶位离拳轨迹太远 | 网格扫描标定 |
| 靶子只在收拳段出现 | `threshold_seconds` 窗口太小 | 从 1.0 改 3.5 |
| 力传感器 `AttributeError` | MuJoCo 自动创建 `_found/_force` 子 sensor | `isinstance(sensor, ContactSensor)` 过滤 |

---

## 6. 复现步骤

见仓库根 `README.md` 的「快速开始」。核心：先装 unitree_rl_mjlab，
覆盖 `patches/`，再跑 `scripts/play_punch.py`（力度）与 `scripts/export_deploy.py`（导出）。
