# make_custom_dataset.py
import numpy as np
import os
import cv2

# === paths ===
VIDEO    = r"D:\School\input.mp4"      # same video you used for YOLO
KPS_FILE = r"D:\School\kps2d.npy"      # (T, 17, 2) from yolo2d.py

OUT_FILE = r"D:\School\VideoPose3D\data\data_2d_custom_myvideos.npz"

# ---- load keypoints ----
kps = np.load(KPS_FILE)  # shape: (T, 17, 2)

def ffill_nan(arr):
    out = arr.copy()
    last = None
    for t in range(out.shape[0]):
        if np.isnan(out[t]).any():
            if last is not None:
                out[t] = last
        else:
            last = out[t]
    out[np.isnan(out)] = 0
    return out

kps = ffill_nan(kps)

# subject name = video filename (what we'll use as --viz-subject)
subject = os.path.basename(VIDEO)  # e.g. "input.mp4"

# ---- build positions_2d dict ----
positions_2d = {
    subject: {
        "custom": [kps]   # list of (T, 17, 2) arrays
    }
}

# ---- get video resolution for metadata ----
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO}")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# ---- build metadata dict in the format VideoPose3D expects ----
metadata = {
    "layout_name": "coco",
    "num_joints": 17,
    # left-right symmetry (COCO indexing)
    "keypoints_symmetry": [
        [1, 3, 5, 7, 9, 11, 13, 15],
        [2, 4, 6, 8, 10, 12, 14, 16],
    ],
    "video_metadata": {
        subject: {  # same key as positions_2d
            "w": w,
            "h": h,
        }
    },
}

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
np.savez_compressed(OUT_FILE, positions_2d=positions_2d, metadata=metadata)

print("Saved custom dataset to:", OUT_FILE)
print("Subject name:", subject)
print("Keypoints shape:", kps.shape)
print("Video size:", w, "x", h)
