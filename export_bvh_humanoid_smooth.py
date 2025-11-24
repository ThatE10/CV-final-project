import numpy as np
import math
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# ---------- config ----------
POSE_FILE = "pose3d.npy"                 # from VideoPose3D --viz-export
OUT_FILE  = "pose_humanoid_smooth.bvh"   # output BVH
FPS       = 30.0                         # your input video FPS
SMOOTH_WINDOW = 5                        # odd number, e.g. 5 or 7
# -----------------------------


# Human3.6M 17-joint skeleton (what VideoPose3D uses for 3D)
JOINT_NAMES = [
    "Hips",          # 0
    "RightUpLeg",    # 1
    "RightLeg",      # 2
    "RightFoot",     # 3
    "LeftUpLeg",     # 4
    "LeftLeg",       # 5
    "LeftFoot",      # 6
    "Spine",         # 7
    "Spine1",        # 8 (Thorax)
    "Neck",          # 9
    "Head",          # 10
    "LeftShoulder",  # 11
    "LeftArm",       # 12
    "LeftForeArm",   # 13
    "RightShoulder", # 14
    "RightArm",      # 15
    "RightForeArm"   # 16
]

# parent indices for each joint (-1 = root)
PARENTS = np.array([
    -1,  # Hips
     0,  # RightUpLeg
     1,  # RightLeg
     2,  # RightFoot
     0,  # LeftUpLeg
     4,  # LeftLeg
     5,  # LeftFoot
     0,  # Spine
     7,  # Spine1
     8,  # Neck
     9,  # Head
     8,  # LeftShoulder
    11,  # LeftArm
    12,  # LeftForeArm
     8,  # RightShoulder
    14,  # RightArm
    15   # RightForeArm
], dtype=int)

ROOT = 0
J = len(JOINT_NAMES)


def smooth_temporal(poses, window=5):
    """Simple moving average over time to reduce jitter."""
    assert window % 2 == 1, "SMOOTH_WINDOW must be odd"
    T = poses.shape[0]
    out = np.empty_like(poses)
    half = window // 2
    for t in range(T):
        s = max(0, t - half)
        e = min(T, t + half + 1)
        out[t] = poses[s:e].mean(axis=0)
    return out


def rotation_between(a, b):
    """Rotation matrix that maps vector a -> b."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    v = np.cross(a, b)
    c = np.dot(a, b)

    # almost the same direction
    if np.linalg.norm(v) < 1e-8 and c > 0.9999:
        return np.eye(3)

    # opposite direction (180°)
    if np.linalg.norm(v) < 1e-8 and c < -0.9999:
        # pick any axis not parallel to a
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v /= np.linalg.norm(v) + 1e-9
        return R.from_rotvec(math.pi * v).as_matrix()

    s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s ** 2 + 1e-9))


# ---------- load & prepare data ----------

pose_path = Path(POSE_FILE)
poses = np.load(pose_path).astype(np.float64)  # (T, 17, 3)
T, J_read, C = poses.shape
assert J_read == J and C == 3, f"Expected (T,17,3), got {poses.shape}"

print(f"Loaded {POSE_FILE} with shape {poses.shape}")

# smooth to kill some noise
poses = smooth_temporal(poses, window=SMOOTH_WINDOW)

# root trajectory (world space)
root0 = poses[0, ROOT].copy()
root_traj = poses[:, ROOT, :] - root0  # translate so first frame root is origin

# center skeleton so root is at origin each frame
poses_centered = poses - poses[:, ROOT:ROOT+1, :]

# "rest pose" = first (centered) frame
rest = poses_centered[0].copy()

# bone offsets in rest pose
offsets = np.zeros((J, 3), dtype=np.float64)
for j in range(J):
    p = PARENTS[j]
    if p < 0:
        offsets[j] = rest[j]  # usually ~0 for root
    else:
        offsets[j] = rest[j] - rest[p]

# ---------- compute local rotations per frame ----------

frames = []
prev_local = [np.eye(3) for _ in range(J)]

for t in range(T):
    local_rots = [np.eye(3) for _ in range(J)]

    for j in range(1, J):
        p = PARENTS[j]
        rest_vec = offsets[j]
        cur_vec = poses_centered[t, j] - poses_centered[t, p]

        if (np.linalg.norm(cur_vec) < 1e-4) or (np.linalg.norm(rest_vec) < 1e-4):
            # if detector freaks out, reuse previous frame's rotation
            local_rots[j] = prev_local[j]
        else:
            local_rots[j] = rotation_between(rest_vec, cur_vec)

    prev_local = [r.copy() for r in local_rots]

    # root: position + rotation (we keep rotation = identity for now)
    tx, ty, tz = root_traj[t]
    frame_vals = [tx, ty, tz, 0.0, 0.0, 0.0]  # Xpos Ypos Zpos Xrot Yrot Zrot

    # children: write *local* Euler angles
    for j in range(1, J):
        euler = R.from_matrix(local_rots[j]).as_euler("XYZ", degrees=True)
        frame_vals.extend(euler.tolist())

    frames.append(frame_vals)

num_channels = 6 + 3 * (J - 1)
assert len(frames[0]) == num_channels, "Channel count mismatch"

print(f"Generated {T} frames, {num_channels} channels per frame.")

# ---------- write BVH ----------

out_path = Path(OUT_FILE)
with out_path.open("w") as f:
    f.write("HIERARCHY\n")

    def write_joint(j, indent):
        ind = "  " * indent
        if PARENTS[j] < 0:
            f.write(f"{ind}ROOT {JOINT_NAMES[j]}\n")
        else:
            f.write(f"{ind}JOINT {JOINT_NAMES[j]}\n")
        f.write(f"{ind}{{\n")

        off = offsets[j]
        f.write(f"{ind}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}\n")

        if PARENTS[j] < 0:
            f.write(f"{ind}  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n")
        else:
            f.write(f"{ind}  CHANNELS 3 Xrotation Yrotation Zrotation\n")

        children = [k for k, p in enumerate(PARENTS) if p == j]
        if not children:
            # simple end site
            f.write(f"{ind}  End Site\n")
            f.write(f"{ind}  {{\n")
            f.write(f"{ind}    OFFSET 0.000000 0.000000 0.000000\n")
            f.write(f"{ind}  }}\n")
        else:
            for c in children:
                write_joint(c, indent + 1)

        f.write(f"{ind}}}\n")

    write_joint(ROOT, 0)

    # MOTION section
    f.write("MOTION\n")
    f.write(f"Frames: {T}\n")
    f.write(f"Frame Time: {1.0 / FPS:.6f}\n")

    for fr in frames:
        f.write(" ".join(f"{v:.6f}" for v in fr) + "\n")

print(f"Wrote BVH to {out_path.resolve()}")
