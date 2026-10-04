"""NPZ trimming template — cut full-cycle boxing NPZ to training-ready length.

Usage:
    python trim_npz.py input.npz output.npz --mode auto

Modes:
    auto     — detect fist forward speed peak, trim 0.5s before + keep until max extension
    manual   — specify start_frame, end_frame
    punch_only — extract only punch window around peak speed
"""
import numpy as np
import argparse
import os

FIST_BODY_IDX = 13  # right_wrist_yaw_link in G1 body_names


def trim_npz(input_path, output_path, mode="auto", start=None, end=None):
    data = np.load(input_path)
    fps = float(data["fps"].item())
    T = data["body_pos_w"].shape[0]
    
    if mode == "auto":
        # 1. Find peak fist forward speed
        fist_vel = data["body_lin_vel_w"][:, FIST_BODY_IDX, :]
        fist_speed = np.linalg.norm(fist_vel, axis=1)
        fist_vx = fist_vel[:, 0]
        
        # Peak where forward velocity > 1 m/s
        forward_mask = fist_vx > 1.0
        if np.any(forward_mask):
            peak = int(np.where(forward_mask)[0][np.argmax(fist_speed[forward_mask])])
        else:
            peak = int(np.argmax(fist_speed))
        
        # 2. Start: 0.5s before peak (or frame 0)
        start = max(0, peak - int(0.5 * fps))
        
        # 3. End: where fist reaches max forward extension (after peak)
        fist_x = data["body_pos_w"][:, FIST_BODY_IDX, 0]
        post_peak = fist_x[peak:]
        end = peak + int(np.argmax(post_peak))  # max X after peak
        end = min(T, end + int(0.1 * fps))  # add 0.1s buffer
        
        print(f"  Full: {T} frames ({T/fps:.1f}s)")
        print(f"  Peak speed: {fist_speed[peak]:.2f} m/s at frame {peak} ({peak/fps:.2f}s)")
        print(f"  Cut: [{start}, {end}) = {end-start} frames ({(end-start)/fps:.1f}s)")
    
    elif mode == "manual":
        assert start is not None and end is not None
        start, end = int(start), int(end)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Extract and save
    out = {}
    for key in data.files:
        arr = data[key]
        if arr.ndim >= 1 and arr.shape[0] == T:
            arr = arr[start:end]
        out[key] = arr
    
    np.savez_compressed(output_path, **out)
    
    orig_sz = os.path.getsize(input_path) / 1024
    out_sz = os.path.getsize(output_path) / 1024
    print(f"  Size: {orig_sz:.0f}K -> {out_sz:.0f}K")
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Input NPZ")
    p.add_argument("output", help="Output NPZ")
    p.add_argument("--mode", default="auto", choices=["auto", "manual"])
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    args = p.parse_args()
    trim_npz(args.input, args.output, args.mode, args.start, args.end)
