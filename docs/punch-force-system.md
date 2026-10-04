# ROBOTAC 击打力度仿真验证 — 完整操作指南

> 技术手册 3.5 节的实现方案。基于 unitree_rl_mjlab + G1 机器人 + MuJoCo 仿真。

---

## 概述

这个方案的核心思路：

1. 在 G1 tracking 环境中添加一个**靶子实体**（固定在前方 0.5m，高度 1.0m）
2. 添加一个 **ContactSensor**，检测右手腕与靶面之间的接触力
3. 在仿真循环中实时读取力传感器数据，输出击打力度

三份文件：

| 文件 | 作用 |
|------|------|
| `env_cfgs_punch.py` | 替换原始 `env_cfgs.py`，添加靶子实体 + 力传感器配置 |
| `play_with_force.py` | 替换原始 `scripts/play.py`，在仿真中实时打印力度 + 统计报告 |
| `punching_target.xml` | 靶子的 MuJoCo XML 模型（备选方案，用 `spec_fn` 时不需要） |

---

## 操作步骤

### Step 0: 确认 unitree_rl_mjlab 已克隆并安装

```bash
cd ~/projects
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
cd unitree_rl_mjlab

# 创建虚拟环境
uv venv --python 3.12
source .venv/bin/activate

# 安装
uv pip install -e .
```

### Step 1: 替换 env_cfgs.py

将原始 `env_cfgs.py` 替换为 `env_cfgs_punch.py`：

```bash
# 备份原始文件
cp src/tasks/tracking/config/g1/env_cfgs.py \
   src/tasks/tracking/config/g1/env_cfgs_backup.py

# 替换
cp env_cfgs_punch.py src/tasks/tracking/config/g1/env_cfgs.py
```

**核心改动：**

1. **靶子实体**：通过 `spec_fn=get_punching_target_spec()` 在 G1 前方放置一个带 free joint 的靶面
2. **力传感器**：`ContactSensorCfg(name="right_hand_target_contact")` 检测右手腕 geom 与靶面 geom 的接触

```python
cfg.scene.entities = {
  "robot": get_g1_robot_cfg(),
  "punching_target": PUNCHING_TARGET_CFG,  # ← 新增
}

right_hand_target_contact = ContactSensorCfg(
  name="right_hand_target_contact",
  primary=ContactMatch(mode="geom", pattern=r"right_wrist_yaw_link_collision", entity="robot"),
  secondary=ContactMatch(mode="geom", pattern=r"target_pad_geom", entity="punching_target"),
  fields=("found", "force", "normal", "pos"),
  reduce="maxforce",
  num_slots=1,
  history_length=4,  # 与 decimation=4 匹配
)
```

### Step 2: 准备拳击动作文件

如果有 `boxing_motion.npz`：

```bash
# 直接指定
--motion_file=mjlab/motions/g1/boxing_motion.npz
```

如果没有现成的拳击动作 npz，需要先用 Video2Robot 生成：

1. 用手机录制一段出拳视频
2. 用 Video2Robot 提取骨骼关键点，导出 CSV
3. 用 `scripts/csv_to_npz.py` 转换为 npz

```bash
python scripts/csv_to_npz.py \
  --input boxing_motion.csv \
  --output mjlab/motions/g1/boxing_motion.npz
```

### Step 3: 训练拳击动作 tracking 策略（如果还没有 checkpoint）

```bash
python scripts/train.py Mjlab-Tracking-Flat-Unitree-G1 \
  --motion_file=mjlab/motions/g1/boxing_motion.npz \
  --num_envs=256 \
  --max_iterations=5000
```

训练完成后 checkpoint 保存在：
```
logs/rsl_rl/g1_tracking/YYYY-MM-DD/model_XXXX.pt
```

### Step 4: 运行力度验证

```bash
python play_with_force.py Mjlab-Tracking-Flat-Unitree-G1 \
  --checkpoint_file=logs/rsl_rl/g1_tracking/YYYY-MM-DD/model_XXXX.pt \
  --motion_file=mjlab/motions/g1/boxing_motion.npz \
  --force_threshold=10.0 \
  --max_steps=1000 \
  --viewer=native
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--force_threshold` | 认定"击中"的最低力 (N) | 10.0 |
| `--max_steps` | 最大仿真步数 | 1000 |
| `--print_force_every` | 每 N 步打印力度 | 1 |
| `--no_terminations` | 关闭终止条件 | True |
| `--viewer` | 可视化方式 (native/viser) | auto |

### Step 5: 查看结果

运行过程中会实时输出：

```
  Step   42 | Force:   85.34 N (Fx= 12.30, Fy= -3.40, Fz= 81.20) | Normal:  81.05 N | 💥 HIT!
           | Contact pos: (0.502, -0.015, 1.003) m
  Step   43 | Force:    0.00 N | No contact
```

仿真结束后会输出统计报告：

```
============================================================
击打力度仿真验证 — 统计报告
============================================================
仿真时长: 52.3s
击中次数 (力 ≥ 10N): 7
最大冲击力: 85.34 N
平均击打力: 32.15 N
力中位数 (P50): 25.60 N
力 P90: 72.10 N
有力步数: 23
最大力接触位置: (0.502, -0.015, 1.003) m
============================================================
```

---

## 技术细节

### 靶子设计

靶子分三层：

| 层级 | 名称 | 说明 |
|------|------|------|
| 底座 | `target_base` | 固定在地面，无关节 |
| 立柱 | `target_pole` | 连接底座和靶面 |
| 靶面 | `target_pad` | **free joint**，可被击打后移动 |

**为什么用 free joint 而不是 fixed？**

- free joint 让靶面在受到冲击后产生真实的物理响应（位移、振动）
- MuJoCo 的 ContactSensor 在有动态响应时测量的力更准确
- 如果靶面完全固定（welded），冲击力会瞬间传到地面，力传感器读数可能不稳定

### 力传感器配置

```python
ContactSensorCfg(
  name="right_hand_target_contact",
  primary=ContactMatch(mode="geom", pattern=..., entity="robot"),
  secondary=ContactMatch(mode="geom", pattern=..., entity="punching_target"),
  fields=("found", "force", "normal", "pos"),
  reduce="maxforce",       # 取力最大的接触点
  history_length=4,         # 与 decimation=4 匹配，防遗漏
)
```

**关键参数：**

- `fields=("found", "force", "normal", "pos")` — found 判断是否接触，force 是力向量，normal 是法向量（用于计算法向力），pos 是接触位置
- `reduce="maxforce"` — 多接触点时只保留力最大的那个
- `history_length=4` — 因为 decimation=4（4 个物理子步 = 1 个策略步），短暂碰撞可能发生在子步中，history 确保不遗漏

### 力的物理含义

输出数据中：

| 数据 | 含义 |
|------|------|
| `Force (N)` | 总接触力大小 = sqrt(fx² + fy² + fz²) |
| `Fx, Fy, Fz` | 力在全局坐标系下的三分量 |
| `Normal (N)` | 法向力分量（垂直于靶面方向的力），即"穿透力" |
| `Contact pos (m)` | 接触点在全局坐标系中的位置 |

---

## 如果 geom 名称不匹配怎么办

G1 机器人的 body/geom 名称可能因版本不同而有差异。如果 force sensor 找不到 contact，需要调整 pattern：

```bash
# 在 WSL 中启动 MuJoCo 模拟器查看 G1 的 geom 名称
python -c "
import mujoco
from mjlab.asset_zoo.robots import get_g1_robot_cfg
cfg = get_g1_robot_cfg()
spec = cfg.spec_fn()
model = spec.compile()
for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    if name and ('wrist' in name or 'elbow' in name or 'shoulder' in name or 'hand' in name):
        print(f'geom[{i}]: {name}')
"
```

根据输出调整 `env_cfgs_punch.py` 中的 pattern：

```python
# 示例：如果 geom 名称是 "right_wrist_yaw" 而不是 "right_wrist_yaw_link_collision"
primary=ContactMatch(
  mode="geom",
  pattern=r"right_wrist_yaw.*|right_elbow.*|right_shoulder_roll.*",
  entity="robot",
)
```

---

## 常见问题

### Q1: 传感器找不到（KeyError）
- 确认 `env_cfgs.py` 已替换为 `env_cfgs_punch.py`
- 确认靶子实体已正确加载（仿真窗口中能看到红色靶面）
- 检查 sensor name 是否和代码中的一致

### Q2: 力度始终为 0
- 检查 G1 右手腕 geom 名称是否匹配 pattern
- 用上面"geom 名称查看"的方法确认
- 确保 boxing motion 让右手确实向靶面方向运动

### Q3: 靶面位置不对
- 调整 `get_punching_target_spec()` 中的 `pos` 参数
- G1 的初始位置是 (0, 0, ~0.8)（站立高度），靶子放在 (0.5, 0, 0) 底座顶部约 1.0m
- 如果机器人不在 (0, 0) 起点，需要调整

### Q4: MuJoCo 编译报错
- 确保 `mujoco` Python 包版本 ≥ 3.2
- XML 语法错误时用 `mujoco.MjSpec.from_file("punching_target.xml")` 替代 `spec_fn`

### Q5: 训练时间太长
- 先用 `--agent=zero` 或 `--agent=random` 做力传感器测试（不需要 checkpoint）
- 确认力传感器能读出数据后，再训练正式策略

---

## 文件替换对照表

| 原始文件 | 替换文件 | 位置 |
|----------|----------|------|
| `src/tasks/tracking/config/g1/env_cfgs.py` | `env_cfgs_punch.py` | G1 配置 |
| `scripts/play.py` | `play_with_force.py` | 仿真验证脚本 |

> **注意**：替换前先备份原始文件！
