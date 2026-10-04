"""Analyze fist trajectory in each boxing motion to find optimal target position."""
import numpy as np
import os

# G1 body name index mapping (from env.yaml / g1_constants)
# The body_pos_w has shape (T, 30, 3) - we need to find the right wrist yaw link index
# Standard G1 mode 15 body order:
BODY_NAMES_29DOF = [
    "pelvis",          # 0
    "torso_link",      # 1
    "waist_yaw_link",  # 2 (sometimes)
]

# Actually, let's just load the NPZ and check which body index has the highest velocity
# during punching motions - that's likely the fist

clips = {
    "Quick_Jab": "M_ShortMove12",
    "Rapid_Punch": "M_Move7",
    "Twist_Punch": "M_Move5",
    "Low_Punch": "M_Move2",
    "Double_Strike": "M_Move17",
    "Power_Burst": "M_ShortMove16",
    "Blitz": "M_Move11",
}

print("=" * 100)
print("BOXING MOTION FIST TRAJECTORY ANALYSIS")
print("=" * 100)

for name, clip in clips.items():
    npz_path = f"g1_moves_data/karate/{name}/training/{clip}.npz"
    if not os.path.exists(npz_path):
        print(f"\n{name}: MISSING")
        continue

    data = np.load(npz_path)
    body_pos = data["body_pos_w"]  # (T, 30, 3)
    body_vel = data["body_lin_vel_w"]  # (T, 30, 3)
    joint_pos = data["joint_pos"]  # (T, 29)
    fps = float(data["fps"].item())
    dt = 1.0 / fps

    T, N, _ = body_pos.shape

    # Find the body with highest peak velocity (likely the fist)
    peak_vels = np.max(np.linalg.norm(body_vel, axis=2), axis=0)  # (30,)
    top_bodies = np.argsort(peak_vels)[::-1][:5]

    # Right wrist is typically the last few bodies in G1
    # Let's check body 28 (right_wrist_yaw) and nearby
    # In G1 mode 15: body indices map to the kinematic tree
    # Let's find the body that extends furthest in +X direction (punching direction)

    # Find peak extension in X for each body
    max_x = np.max(body_pos[:, :, 0], axis=0)  # (30,)
    furthest_bodies = np.argsort(max_x)[::-1][:5]

    print(f"\n{'='*80}")
    print(f"{name} ({clip}) — {T} frames, {T/fps:.1f}s")
    print(f"{'='*80}")

    print(f"\nTop 5 bodies by peak velocity:")
    for i, bidx in enumerate(top_bodies):
        print(f"  Body {bidx:2d}: peak_vel={peak_vels[bidx]:.2f} m/s, max_x={max_x[bidx]:.3f}m")

    print(f"\nTop 5 bodies by max X extension:")
    for i, bidx in enumerate(furthest_bodies):
        print(f"  Body {bidx:2d}: max_x={max_x[bidx]:.3f}m, peak_vel={peak_vels[bidx]:.2f} m/s")

    # For the body with highest velocity AND high X extension,
    # find the position at peak velocity
    # Try body indices that are likely the right hand
    for candidate in top_bodies[:3]:
        vel_mag = np.linalg.norm(body_vel[:, candidate, :], axis=1)
        peak_idx = np.argmax(vel_mag)
        pos_at_peak = body_pos[peak_idx, candidate, :]
        vel_at_peak = body_vel[peak_idx, candidate, :]

        # Find max X position and its frame
        x_traj = body_pos[:, candidate, 0]
        max_x_idx = np.argmax(x_traj)
        pos_at_max_x = body_pos[max_x_idx, candidate, :]

        print(f"\n  Body {candidate} analysis:")
        print(f"    Peak velocity: {vel_mag[peak_idx]:.2f} m/s at frame {peak_idx} ({peak_idx/fps:.2f}s)")
        print(f"    Position at peak vel: x={pos_at_peak[0]:.3f}, y={pos_at_peak[1]:.3f}, z={pos_at_peak[2]:.3f}")
        print(f"    Max X extension: {x_traj[max_x_idx]:.3f}m at frame {max_x_idx} ({max_x_idx/fps:.2f}s)")
        print(f"    Position at max X: x={pos_at_max_x[0]:.3f}, y={pos_at_max_x[1]:.3f}, z={pos_at_max_x[2]:.3f}")

        # Velocity at max X (impact velocity)
        vel_at_max_x = body_vel[max_x_idx, candidate, :]
        print(f"    Velocity at max X: vx={vel_at_max_x[0]:.2f}, vy={vel_at_max_x[1]:.2f}, vz={vel_at_max_x[2]:.2f} (|v|={np.linalg.norm(vel_at_max_x):.2f} m/s)")

        # Suggested target position (slightly beyond max X for impact)
        print(f"    >> SUGGESTED TARGET: ({pos_at_max_x[0]+0.02:.3f}, {pos_at_max_x[1]:.3f}, {pos_at_max_x[2]:.3f})")
