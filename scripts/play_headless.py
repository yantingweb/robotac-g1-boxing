"""Headless play + video generation for trained RL models.
Simple manual sim loop, no viewer required. Saves directly to MP4.
"""

import json, sys, numpy as np, torch, tyro, imageio
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
if not hasattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD"):
    mujoco.mjtEnableBit.mjENBL_MULTICCD = 1 << 6

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.sensor import ContactSensor


@dataclass
class Cfg:
    motion_file: str
    checkpoint_file: str
    output_dir: str = "model_export/videos"
    device: str | None = None
    height: int = 640
    width: int = 960
    fps: int = 50


def run(task_id, cfg):
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    ag_cfg = load_rl_cfg(task_id)

    # Motion file
    mc = env_cfg.commands["motion"]
    assert isinstance(mc, MotionCommandCfg)
    mp = Path(cfg.motion_file).expanduser().resolve()
    assert mp.exists(), f"Motion not found: {mp}"
    mc.motion_file = str(mp)

    # Motion duration
    md = np.load(mp)
    T = md["body_pos_w"].shape[0]
    mfps = float(md["fps"].item())
    dt = env_cfg.decimation * env_cfg.sim.mujoco.timestep
    steps = int(T / mfps / dt) + 10
    print(f"[INFO] Motion: {T / mfps:.1f}s, {steps} sim steps")

    # Video settings
    env_cfg.viewer.height = cfg.height
    env_cfg.viewer.width = cfg.width
    env_cfg.scene.num_envs = 1

    # Checkpoint
    cp = Path(cfg.checkpoint_file)
    assert cp.exists(), f"Checkpoint not found: {cp}"
    print(f"[INFO] Checkpoint: {cp.name}")

    # Create env (no VideoRecorder - we capture frames manually)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    
    # Force tracking
    has_fist = any(
        isinstance(s, ContactSensor) and "fist_target" in n
        for n, s in env.scene.sensors.items()
    )

    env = RslRlVecEnvWrapper(env, clip_actions=ag_cfg.clip_actions)

    # Load policy
    runner = MjlabOnPolicyRunner(env, asdict(ag_cfg), device=device)
    runner.load(str(cp), map_location=device, strict=False)
    policy = runner.get_inference_policy(device=device)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    motion_name = mp.stem
    checkpoint_name = cp.stem
    video_path = output_dir / f"{checkpoint_name}_{motion_name}.mp4"

    # Track max force
    max_force = 0.0
    result_file = output_dir / f"force_{motion_name}.json"

    print(f"[INFO] Running {steps} steps...")
    obs, _ = env.reset()
    frames = []
    force_tracker_target = 0.50  # fixed target X position
    
    for step in range(steps):
        with torch.no_grad():
            action = policy(obs)
        obs = env.step(action)[0]
        
        # Capture frame from env render
        frame = env.unwrapped.render()
        if frame is not None:
            frames.append(frame)

        # Track force
        if has_fist:
            for n, s in env.unwrapped.scene.sensors.items():
                if "fist_target" in n and isinstance(s, ContactSensor):
                    if s.data.found is not None and s.data.force is not None:
                        f = s.data.found[0]
                        force = s.data.force[0]
                        for i in range(force.shape[0]):
                            if f[i] > 0:
                                fm = torch.norm(force[i]).item()
                                if fm > max_force:
                                    max_force = fm
                                    with open(result_file, "w") as fh:
                                        json.dump({"max_force_N": max_force}, fh)

        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{steps} - {len(frames)} frames, max_force={max_force:.0f}N")

    # Save video with imageio
    if frames:
        h, w = frames[0].shape[:2]
        imageio.mimsave(str(video_path), frames, fps=cfg.fps, codec='libx264')
        print(f"[INFO] Video: {video_path} ({len(frames)} frames, {w}x{h})")
    else:
        print("[ERROR] No frames captured!")

    env.close()
    print(f"[INFO] Max force: {max_force:.2f} N")
    return str(video_path)


def main():
    import mjlab.tasks  # noqa
    import src.tasks   # noqa

    all_tasks = list_tasks()
    chosen, rest = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False, return_unknown_args=True,
    )
    args = tyro.cli(Cfg, args=rest, prog=sys.argv[0] + f" {chosen}",
                    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff))
    run(chosen, args)


if __name__ == "__main__":
    main()
