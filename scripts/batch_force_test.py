"""Batch force test for top boxing clips.

Runs each clip with fixed target at estimated optimal position,
captures max force, and reports ranking.
"""
import os
import json
import subprocess
import sys

TOP_CLIPS = [
    {
        "name": "Power_Burst",
        "clip": "M_ShortMove16",
        "npz": "g1_moves_data/karate/Power_Burst/training/M_ShortMove16.npz",
        "pt": "g1_moves_data/karate/Power_Burst/policy_154/M_ShortMove16_policy.pt",
        "target": (-0.489, 0.140, 0.988),
    },
    {
        "name": "Blitz",
        "clip": "M_Move11",
        "npz": "g1_moves_data/karate/Blitz/training/M_Move11.npz",
        "pt": "g1_moves_data/karate/Blitz/policy_154/M_Move11_policy.pt",
        "target": (-0.346, 0.380, 0.815),
    },
    {
        "name": "B_AttackKarate",
        "clip": "B_AttackKarate",
        "npz": "g1_moves_data/karate/B_AttackKarate/training/B_AttackKarate.npz",
        "pt": "g1_moves_data/karate/B_AttackKarate/policy_154/B_AttackKarate_policy.pt",
        "target": (-0.520, 0.103, 0.876),
    },
    {
        "name": "Twist_Punch",
        "clip": "M_Move5",
        "npz": "g1_moves_data/karate/Twist_Punch/training/M_Move5.npz",
        "pt": "g1_moves_data/karate/Twist_Punch/policy_154/M_Move5_policy.pt",
        "target": (-0.645, 0.338, 0.780),
    },
    {
        "name": "Spin_Punch",
        "clip": "M_Move4",
        "npz": "g1_moves_data/karate/Spin_Punch/training/M_Move4.npz",
        "pt": "g1_moves_data/karate/Spin_Punch/policy_154/M_Move4_policy.pt",
        "target": (-0.478, -0.144, 1.240),
    },
]

RESULTS_DIR = "/tmp/punch_force_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 100)
print("BATCH PUNCH FORCE TESTING - TOP 5 CLIPS")
print("=" * 100)

results = []

for i, clip in enumerate(TOP_CLIPS):
    result_file = os.path.join(RESULTS_DIR, "{}.json".format(clip["name"]))
    
    cmd = [
        "python3", "scripts/play.py",
        "Unitree-G1-Tracking-No-State-Estimation",
        "--motion-file", clip["npz"],
        "--checkpoint-file", clip["pt"],
        "--punch-target", "True",
        "--punch-track-mode", "fixed",
        "--fixed-target-pos",
        "{:.3f}".format(clip["target"][0]),
        "{:.3f}".format(clip["target"][1]),
        "{:.3f}".format(clip["target"][2]),
        "--result-file", result_file,
        "--video", "True",
        "--video-length", "200",
        "--viewer", "auto",
    ]
    
    print("\n[{}/{}] Testing {}...".format(i+1, len(TOP_CLIPS), clip["name"]))
    print("  Target: ({:.3f}, {:.3f}, {:.3f})".format(*clip["target"]))
    print("  CMD: {}".format(" ".join(cmd)))
    sys.stdout.flush()
    
    try:
        proc = subprocess.run(
            cmd,
            cwd=os.environ.get("UNITREE_RL_MJLAB_DIR", "."),
            env={**os.environ, "MUJOCO_GL": "egl"},
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        # Check result file
        if os.path.exists(result_file):
            with open(result_file) as f:
                force_data = json.load(f)
            max_force = force_data.get("max_force_N", 0)
            has_contact = force_data.get("has_contact", False)
            print("  RESULT: max_force={:.2f} N, has_contact={}".format(max_force, has_contact))
        else:
            max_force = 0
            has_contact = False
            print("  RESULT: no result file generated")
            
        # Check for PunchDebug/PunchForce in stderr
        for line in proc.stdout.split("\n"):
            if "Punch" in line or "punch" in line.lower():
                print("  OUTPUT:", line.strip())
        for line in proc.stderr.split("\n")[:5]:
            if line.strip():
                pass  # suppress stderr noise
        
        results.append({
            "name": clip["name"],
            "max_force": max_force,
            "has_contact": has_contact,
            "target": clip["target"],
        })
        
    except subprocess.TimeoutExpired:
        print("  TIMEOUT (120s)")
        results.append({
            "name": clip["name"],
            "max_force": 0,
            "has_contact": False,
            "target": clip["target"],
            "timeout": True,
        })
    except Exception as e:
        print("  ERROR:", e)
        results.append({
            "name": clip["name"],
            "max_force": 0,
            "has_contact": False,
            "error": str(e),
        })

# Final ranking
print("\n" + "=" * 100)
print("FINAL FORCE RANKING")
print("=" * 100)
results.sort(key=lambda r: -r.get("max_force", 0))
for i, r in enumerate(results):
    extras = ""
    if r.get("timeout"):
        extras = " [TIMEOUT]"
    elif r.get("error"):
        extras = " [ERROR: {}]".format(r["error"])
    print("{}. {}: {:.2f} N (contact={}){}".format(
        i+1, r["name"], r.get("max_force", 0), r.get("has_contact", False), extras
    ))
