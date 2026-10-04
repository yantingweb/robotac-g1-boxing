"""Export deployment package for a specific model checkpoint.

Usage:
  python export_deploy.py <checkpoint_path> <output_dir>
  
Example:
  python export_deploy.py /path/to/model_320000.pt /path/to/deployment/320000
"""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import numpy as np

# monkey-patch
import mujoco
if not hasattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD"):
    mujoco.mjtEnableBit.mjENBL_MULTICCD = 1 << 6

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking import mdp
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

import mjlab.tasks  # noqa
import src.tasks   # noqa


TASK_ID = "Unitree-G1-Tracking-No-State-Estimation"
MOTION_FILE = os.environ.get("MOTION_FILE", "data/Quick_Jab_train.npz")


def main():
    if len(sys.argv) < 3:
        print("Usage: python export_deploy.py <checkpoint_path> <output_dir>")
        sys.exit(1)

    checkpoint_path = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")

    configure_torch_backends()
    device = "cpu"

    # Load configs
    env_cfg = load_env_cfg(TASK_ID, play=True)
    ag_cfg = load_rl_cfg(TASK_ID)
    runner_cls = load_runner_cls(TASK_ID)

    # Set motion file for ONNX export (MotionTrackingOnPolicyRunner needs it)
    mc = env_cfg.commands["motion"]
    assert isinstance(mc, MotionCommandCfg)
    mc.motion_file = str(Path(MOTION_FILE).expanduser().resolve())
    env_cfg.scene.num_envs = 1

    print(f"Motion: {mc.motion_file}")

    # Create env and load model
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=ag_cfg.clip_actions)
    
    runner = runner_cls(env, asdict(ag_cfg), device=device)
    runner.load(str(checkpoint_path), map_location=device, strict=False)

    checkpoint_name = checkpoint_path.stem  # e.g. "model_320000"

    # 1. Copy the checkpoint .pt
    dst_pt = output_dir / f"{checkpoint_name}.pt"
    import shutil
    shutil.copy(checkpoint_path, dst_pt)
    print(f"[OK] Checkpoint: {dst_pt}")

    # 2. Extract obs_normalizer from checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "actor_state_dict" in ckpt:
        sd = ckpt["actor_state_dict"]
        obs_norm = {
            "mean": sd["obs_normalizer._mean"].tolist(),
            "var": sd["obs_normalizer._var"].tolist(),
            "std": sd["obs_normalizer._std"].tolist(),
            "count": sd["obs_normalizer.count"].item(),
        }
        dst_json = output_dir / "obs_normalizer.json"
        with open(dst_json, "w") as f:
            json.dump(obs_norm, f, indent=2)
        print(f"[OK] Obs normalizer: {dst_json}")
    else:
        print("[WARN] No actor_obs_normalizer in checkpoint")

    # 3. Export standard policy.onnx
    runner.export_policy_to_onnx(str(output_dir), "policy.onnx")
    print(f"[OK] Standard ONNX: {output_dir / 'policy.onnx'}")

    # 4. Export motion policy with embedded motion data
    #    Uses MotionTrackingOnPolicyRunner.export_motion_policy_to_onnx
    run_name = checkpoint_path.parent.name  # e.g. "2026-07-17_20-21-08"
    motion_onnx_name = f"{run_name}.onnx"
    runner.export_motion_policy_to_onnx(str(output_dir), motion_onnx_name)
    print(f"[OK] Motion ONNX: {output_dir / motion_onnx_name}")

    # 5. Copy motion NPZ
    dst_npz = output_dir / "Quick_Jab_train.npz"
    if Path(mc.motion_file).exists():
        shutil.copy(mc.motion_file, dst_npz)
        print(f"[OK] Motion NPZ: {dst_npz}")

    env.close()

    print(f"\nDeployment package ready at: {output_dir}")
    print("Files:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name} ({size / 1024 / 1024:.2f} MB)" if size > 1000000 else f"  {f.name} ({size} B)")


if __name__ == "__main__":
    main()
