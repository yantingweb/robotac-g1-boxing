import mujoco
if not hasattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD"):
    mujoco.mjtEnableBit.mjENBL_MULTICCD = 1 << 6

"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
import torch
import tyro

from rsl_rl.runners import OnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactSensor
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
    agent: Literal["zero", "random", "trained"] = "trained"
    motion_file: str | None = None
    wandb_run_path: str | None = None
    checkpoint_file: str | None = None
    num_envs: int | None = None
    device: str | None = None
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    camera: int | str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    punch_tip_offset: tuple[float, float, float] = (0.15, 0.02, 0.0)
    punch_track_mode: Literal["motion", "actual"] = "motion"
    punch_target: bool = True

    # Internal flag used by demo script.
    _demo_mode: tyro.conf.Suppress[bool] = False


class _PunchTargetTracker:
    """Tracks punch target position and reads contact force during play.

    Supports both tracking (single punch_target at end of motion) and
    punching (multiple targets) tasks.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        tip_offset: tuple[float, float, float] = (0.15, 0.02, 0.0),
        track_mode: str = "motion",
        fist_body_name: str = "right_wrist_yaw_link",
        threshold_seconds: float = 1.0,
    ):
        self.env = env
        self.threshold_seconds = threshold_seconds
        self.tip_offset = torch.tensor(tip_offset, device=env.device)
        self.track_mode = track_mode
        self.fist_body_name = fist_body_name

        # Collect all fist-target contact sensors (works for tracking & punching).
        # We filter by isinstance(ContactSensor) because MuJoCo also creates
        # sub-sensors (e.g. ..._found, ..._force) as BuiltinSensors which return raw
        # tensors and do not have the .found / .force attributes.
        self.sensors: list[tuple[str, Any]] = [
            (name, sensor)
            for name, sensor in env.scene.sensors.items()
            if isinstance(sensor, ContactSensor) and "fist_target" in name
        ]
        if not self.sensors:
            raise RuntimeError("No fist_target_contact sensors found in scene")

        # Optional: single punch_target entity (tracking mode).
        self.target = env.scene.entities.get("punch_target")
        self.motion_cmd = env.command_manager.get_term("motion")

        # Optional: single punch_target entity (tracking mode).
        self.target = env.scene.entities.get("punch_target")
        self.motion_cmd = env.command_manager.get_term("motion")

        if self.target is not None and self.motion_cmd is not None:
            body_names = self.motion_cmd.cfg.body_names
            if fist_body_name in body_names:
                self._fist_idx = body_names.index(fist_body_name)
            else:
                self._fist_idx = None
        else:
            self._fist_idx = None

        self.target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
        self.hidden_pos = torch.tensor([0.0, 0.0, -100.0], device=env.device)

        self.max_force = 0.0
        self.has_contact = False
        self._debug_printed = False

    def on_step(self):
        # Move punch_target in tracking mode (end of motion only).
        if self.target is not None and self._fist_idx is not None:
            steps_remaining = (
                self.motion_cmd.motion.time_step_total - self.motion_cmd.time_steps[0]
            )
            seconds_remaining = steps_remaining.item() * self.env.step_dt

            if seconds_remaining <= self.threshold_seconds:
                robot = self.env.scene["robot"]
                if self.fist_body_name in robot.body_names:
                    fist_idx = robot.body_names.index(self.fist_body_name)
                    wrist_pos = robot.data.body_link_pos_w[0, fist_idx]
                    wrist_quat = robot.data.body_link_quat_w[0, fist_idx]
                    actual_fist = wrist_pos + quat_apply(wrist_quat, self.tip_offset)
                else:
                    actual_fist = None

                if self.track_mode == "actual" and actual_fist is not None:
                    # Track the actual fist so the pad is always right in front
                    # of the real hand (minimises dist due to tracking error).
                    target_pos = actual_fist
                else:
                    # Default: follow the motion reference trajectory.
                    t = self.motion_cmd.time_steps[0]
                    ref_pos = self.motion_cmd.motion.body_pos_w[t, self._fist_idx]
                    ref_quat = self.motion_cmd.motion.body_quat_w[t, self._fist_idx]
                    world_offset = quat_apply(ref_quat, self.tip_offset)
                    target_pos = ref_pos + world_offset + self.env.scene.env_origins[0]

                pose = torch.cat([target_pos, self.target_quat]).unsqueeze(0)
                self.target.write_mocap_pose_to_sim(pose)

                # Debug: print target vs actual fist position once per episode
                if not self._debug_printed and actual_fist is not None:
                    dist = torch.norm(target_pos - actual_fist).item()
                    print(
                        f"[PunchDebug] target_pos={target_pos.cpu().numpy()}, "
                        f"actual_fist={actual_fist.cpu().numpy()}, dist={dist:.3f}m"
                    )
                    self._debug_printed = True
            else:
                pose = torch.cat([self.hidden_pos, self.target_quat]).unsqueeze(0)
                self.target.write_mocap_pose_to_sim(pose)
                self._debug_printed = False

        # Read contact forces from all fist_target_contact sensors.
        total_force = 0.0
        any_contact = False
        for _, sensor in self.sensors:
            data = sensor.data
            if data.found is None or data.force is None:
                continue
            found = data.found[0]   # first env
            force = data.force[0]   # [num_primaries, 3]
            for i in range(found.shape[0]):
                if found[i] > 0:
                    fmag = torch.norm(force[i]).item()
                    if fmag > 0:
                        any_contact = True
                        total_force = max(total_force, fmag)

        if any_contact and total_force > self.max_force:
            self.max_force = total_force
            self.has_contact = True
            print(f"[PunchForce] Contact force: {total_force:.2f} N")

    def on_reset(self):
        if self.has_contact:
            print(
                f"[PunchForce] Episode summary - Max force: {self.max_force:.2f} N"
            )
        self.max_force = 0.0
        self.has_contact = False
        self._debug_printed = False


class _StepCallbackWrapper:
    """Thin wrapper to inject a callback after each env step/reset."""

    def __init__(self, env, callback):
        self._env = env
        self._callback = callback

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def step(self, action):
        result = self._env.step(action)
        self._callback.on_step()
        return result

    def reset(self, **kwargs):
        result = self._env.reset(**kwargs)
        self._callback.on_reset()
        return result


def run_play(task_id: str, cfg: PlayConfig):
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    # Check if this is a tracking task by checking for motion command.
    is_tracking_task = "motion" in env_cfg.commands and isinstance(
        env_cfg.commands["motion"], MotionCommandCfg
    )

    # Remove punch target from tracking play if requested.
    if is_tracking_task and not cfg.punch_target:
        env_cfg.scene.entities.pop("punch_target", None)
        if hasattr(env_cfg.scene, "sensors") and env_cfg.scene.sensors is not None:
            env_cfg.scene.sensors = tuple(
                s for s in env_cfg.scene.sensors if getattr(s, "name", None) != "fist_target_contact"
            )

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    TRAINED_MODE = not DUMMY_MODE

    if is_tracking_task and cfg._demo_mode:
        # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        motion_cmd.sampling_mode = "uniform"

    if is_tracking_task:
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)

        if cfg.motion_file is not None:
            motion_path = Path(cfg.motion_file).expanduser().resolve()
            if not motion_path.exists():
              raise FileNotFoundError(f"Motion file not found: {motion_path}")
            motion_cmd.motion_file = str(motion_path)
        elif not motion_cmd.motion_file:
            raise ValueError(
                "Tracking tasks require --motion-file pointing to a local motion npz."
            )

        if cfg._demo_mode:
          # Demo mode: increase diversity
          motion_cmd.sampling_mode = "uniform"

        print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")

    log_dir: Path | None = None
    resume_path: Path | None = None

    if TRAINED_MODE:
        log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
        resume_path = Path(cfg.checkpoint_file)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
        if cfg.wandb_run_path is None:
            raise ValueError(
                "`wandb_run_path` is required when `checkpoint_file` is not provided."
            )
        resume_path, was_cached = get_wandb_checkpoint_path(
            log_root_path, Path(cfg.wandb_run_path)
        )
        # Extract run_id and checkpoint name from path for display.
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
            f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
        )
    log_dir = resume_path.parent

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    if cfg.video and DUMMY_MODE:
        print(
            "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
        )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    # Enable punch force monitoring if the scene contains fist_target_contact sensors.
    if any(
        isinstance(s, ContactSensor) and "fist_target" in name
        for name, s in env.scene.sensors.items()
    ):
        tracker = _PunchTargetTracker(
            env,
            tip_offset=cfg.punch_tip_offset,
            track_mode=cfg.punch_track_mode,
        )
        env = _StepCallbackWrapper(env, tracker)
        print(f"[INFO] Punch force monitoring enabled (track_mode={cfg.punch_track_mode})")

    if TRAINED_MODE and cfg.video:
        print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if DUMMY_MODE:
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
        if cfg.agent == "zero":
            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return torch.zeros(action_shape, device=env.unwrapped.device)
            policy = PolicyZero()
        else:
            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1
            policy = PolicyRandom()

    else:
        runner_cls = load_runner_cls(task_id) or OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(str(resume_path), map_location=device)
        policy = runner.get_inference_policy(device=device)

    # Handle "auto" viewer selection.
    if cfg.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved_viewer = "native" if has_display else "viser"
        del has_display
    else:
        resolved_viewer = cfg.viewer

    if resolved_viewer == "native":
        NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env, policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

    env.close()


def main():
    # Parse first argument to choose the task.
    # Import tasks to populate the registry.
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
    agent_cfg = load_rl_cfg(chosen_task)

    args = tyro.cli(
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(
            tyro.conf.AvoidSubcommands,
            tyro.conf.FlagConversionOff,
        ),
    )
    del remaining_args, agent_cfg

    run_play(chosen_task, args)


if __name__ == "__main__":
  main()
