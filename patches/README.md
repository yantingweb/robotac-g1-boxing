# patches —— 对 unitree_rl_mjlab 的修改

本目录存放相对上游 [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) 改动的文件。
框架主体不改，只做局部扩展。使用方式：把下列文件覆盖到上游仓库的对应位置。

| 本目录文件 | 覆盖到（上游仓库内路径） |
|-----------|------------------------|
| `tracking_env_cfg.py` | `src/tasks/tracking/tracking_env_cfg.py` |
| `mdp_rewards.py` | `src/tasks/tracking/mdp/rewards.py` |
| `config_g1_env_cfgs.py` | `src/tasks/tracking/config/g1/env_cfgs.py` |
| `config_g1___init__.py` | `src/tasks/tracking/config/g1/__init__.py` |
| `config_g1_rl_cfg.py` | `src/tasks/tracking/config/g1/rl_cfg.py` |
| `scripts_play.py` | `scripts/play.py` |

## 各改动说明

### tracking_env_cfg.py —— 奖励函数
在默认运动跟踪奖励基础上：
- 手臂 6 个关键点（肩/肘/腕 × 左右）单独 **4x 加权**（`motion_body_pos_arms` / `_ori_arms` / `_lin_vel_arms`）
- 新增**拳速激励** `fist_velocity`
- 新增**自碰撞惩罚** `self_collisions`
- 移除 `action_acc_l2`（它会压住出拳的正常加速）

### mdp_rewards.py —— 自定义奖励
```python
def fist_velocity_reward(env, body_name, scale=1.0):
    """奖励指定 body 的线速度模长。"""
    ...

def self_collision_cost(env, sensor_name, force_threshold=10.0):
    """用接触力历史统计超阈值的子步数。"""
    ...
```

### config_g1_env_cfgs.py —— 注入拳靶
仅在 `play=True` 时向场景注入：
- 拳靶实体 `punch_target`（红色圆柱，半径 8cm）
- 力传感器 `fist_target_contact`（右腕 ↔ 靶面）

训练配置不受影响。

### config_g1___init__.py —— 任务注册
注册 `Unitree-G1-Tracking-No-State-Estimation`（无状态估计版），
与上游的 `Mjlab-*` 任务区别在于观测组不含基座线速度/锚点位置估计。

### scripts_play.py —— 回放脚本
- 顶部加入 mujoco 3.10 / mujoco-warp 兼容 monkey-patch
- 集成 `_PunchTargetTracker`（力度追踪）与 `_StepCallbackWrapper`
- 支持 `motion` / `actual` / `fixed` 三种靶子模式 + `result_file` 落盘
