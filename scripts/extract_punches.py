"""Extract punch segments from G1 Moves boxing motions.

For each motion, finds the window around peak fist forward velocity,
extracts the frames as a standalone NPZ for punch-only training/play.
"""
import numpy as np
import os

FIST_BODY_IDX = 29  # right wrist yaw link
WINDOW_BEFORE = 30   # frames before peak (0.5s @ 60fps)
WINDOW_AFTER = 60    # frames after peak (1.0s @ 60fps)
MIN_FORWARD_SPEED = 2.0  # m/s - only consider frames with vx > this

clips = {
    "Power_Burst": "M_ShortMove16",
    "Quick_Jab": "M_ShortMove12",
    "Blitz": "M_Move11",
    "Twist_Punch": "M_Move5",
    "Rapid_Punch": "M_Move7",
    "Double_Strike": "M_Move17",
    "Low_Punch": "M_Move2",
}

INPUT_DIR = "g1_moves_data/karate"
OUTPUT_DIR = "g1_moves_data/punch_segments"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 100)
print("PUNCH SEGMENT EXTRACTION")
print("=" * 100)

for name, clip in clips.items():
    npz_path = os.path.join(INPUT_DIR, name, "training", clip + ".npz")
    if not os.path.exists(npz_path):
        continue

    data = np.load(npz_path)
    joint_pos = data["joint_pos"]          # (T, 29)
    joint_vel = data["joint_vel"]          # (T, 29)
    body_pos = data["body_pos_w"]          # (T, 30, 3)
    body_quat = data["body_quat_w"]        # (T, 30, 4)
    body_lin_vel = data["body_lin_vel_w"]  # (T, 30, 3)
    body_ang_vel = data["body_ang_vel_w"]  # (T, 30, 3)
    fps = float(data["fps"].item())

    fist_vel = body_lin_vel[:, FIST_BODY_IDX, :]  # (T, 3)
    fist_vx = fist_vel[:, 0]       # forward velocity
    fist_speed = np.linalg.norm(fist_vel, axis=1)  # (T,)

    # Find the punch moment: max forward velocity while moving forward
    forward_mask = fist_vx > MIN_FORWARD_SPEED
    if not np.any(forward_mask):
        # Fallback: use max total speed
        peak_idx = int(np.argmax(fist_speed))
    else:
        forward_indices = np.where(forward_mask)[0]
        peak_idx = int(forward_indices[np.argmax(fist_speed[forward_indices])])

    peak_speed = fist_speed[peak_idx]
    peak_vx = fist_vx[peak_idx]
    peak_pos = body_pos[peak_idx, FIST_BODY_IDX, :]

    # Extract window around peak
    start = max(0, peak_idx - WINDOW_BEFORE)
    end = min(joint_pos.shape[0], peak_idx + WINDOW_AFTER)
    window_frames = end - start

    # Extract all arrays for this window
    seg_joint_pos = joint_pos[start:end]
    seg_joint_vel = joint_vel[start:end]
    seg_body_pos = body_pos[start:end]
    seg_body_quat = body_quat[start:end]
    seg_body_lin_vel = body_lin_vel[start:end]
    seg_body_ang_vel = body_ang_vel[start:end]

    # Fist position at strike (relative to world, but we need robot-relative)
    fist_at_strike = body_pos[peak_idx, FIST_BODY_IDX, :]
    pelvis_at_strike = body_pos[peak_idx, 0, :]  # body 0 = pelvis
    fist_rel = fist_at_strike - pelvis_at_strike

    # Save segment
    out_path = os.path.join(OUTPUT_DIR, name + "_punch.npz")
    np.savez_compressed(
        out_path,
        fps=np.array([fps]),
        joint_pos=seg_joint_pos,
        joint_vel=seg_joint_vel,
        body_pos_w=seg_body_pos,
        body_quat_w=seg_body_quat,
        body_lin_vel_w=seg_body_lin_vel,
        body_ang_vel_w=seg_body_ang_vel,
    )

    # Calculate suggested target position (in front of fist at strike)
    vel_dir = fist_vel[peak_idx] / (peak_speed + 1e-8)
    target_pos = fist_at_strike + 0.05 * vel_dir  # 5cm ahead in velocity direction

    print(f"\n{name}:")
    print(f"  Full motion: {joint_pos.shape[0]} frames ({joint_pos.shape[0]/fps:.1f}s)")
    print(f"  Punch window: frames [{start}, {end}) = {window_frames} frames ({window_frames/fps:.1f}s)")
    print(f"  Peak fist speed: {peak_speed:.2f} m/s (vx={peak_vx:.2f}) at frame {peak_idx} ({peak_idx/fps:.2f}s)")
    print(f"  Fist at strike: x={fist_at_strike[0]:.3f}, y={fist_at_strike[1]:.3f}, z={fist_at_strike[2]:.3f}")
    print(f"  Fist rel pelvis: x={fist_rel[0]:.3f}, y={fist_rel[1]:.3f}, z={fist_rel[2]:.3f}")
    print(f"  Suggested target: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
    print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")

print(f"\n{'='*100}")
print(f"Done. All segments saved to {OUTPUT_DIR}/")
