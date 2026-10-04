# 模型产物

本目录收录训练得到的**里程碑模型**（不是全部 checkpoint）。每个子目录包含：

| 文件 | 说明 |
|------|------|
| `model_<N>.pt` | RSL-RL checkpoint（actor + critic + optimizer + 归一化统计） |
| `policy.onnx` | 标准推理图：输入 `obs[1,154]` → 输出 `actions[1,29]` |
| `<run>.onnx` | 含内嵌 motion 数据的推理图（motion tracking 播放用） |
| `obs_normalizer.json` | 观测归一化参数（mean / var / std / count），部署必备 |

## 模型清单

| 目录 | 说明 | 关键奖励配置 |
|------|------|-------------|
| `249999/` | **基准最优**（固定靶峰值 303 N） | 手臂 3x，`action_rate_l2=-0.3`，`action_acc_l2=-0.2`，`fist_velocity=1.5` |
| `299999/` | 平滑版（降低动作惩罚，拳速稳健） | 手臂 3x，`action_rate_l2=-0.1`，`action_acc_l2=-0.05`，`fist_velocity=1.5` |
| `330000/` | **最新**（追求力度上限） | 手臂 4x，`action_rate_l2=-0.1`，无 `action_acc_l2`，`fist_velocity=2.0` |

> 奖励函数的完整定义见 `../patches/tracking_env_cfg.py`。
> 使用这些模型回放/测力：见仓库根 `README.md` 的「快速开始」。
