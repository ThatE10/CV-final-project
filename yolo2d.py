# yolo2d.py - Fully fixed & improved version (copy-paste safe)
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm

# ================== CONFIG ==================
VIDEO_PATH = r"D:\School\NewMethod_BattleTestedPipeline\input.mp4"          # your video
OUT_NPY = "kps2d.npy"                        # [T, 17, 2]
OUT_CONF = "kps2d_conf.npy"                  # [T, 17]
OUT_VIDEO = "input_with_kpts.mp4"            # video with skeleton overlay
MODEL_NAME = "yolov8x-pose.pt"               # best accuracy (use yolov8m-pose.pt if too slow)
CONF_THRESHOLD = 0.3
# ===========================================

print("Loading model...")
model = YOLO(MODEL_NAME)

print("Opening video:", VIDEO_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (width, height))

SKELETON = [
    (0,1), (0,2), (1,3), (2,4), (0,5), (0,6),
    (5,7), (7,9), (9,11), (6,8), (8,10), (10,12),
    (5,6), (5,11), (6,12), (11,13), (13,15), (12,14), (14,16)
]

kps_list = []
conf_list = []

print("Processing frames...")
for frame_idx in tqdm(range(total_frames)):
    ok, frame = cap.read()
    if not ok:
        break

    results = model.predict(frame, verbose=False)[0]

    # Default: all NaN / zero conf
    keypoints_xy = np.full((17, 2), np.nan, dtype=np.float32)
    keypoints_conf = np.zeros(17, dtype=np.float32)

    if results.keypoints is not None and len(results.keypoints) > 0:
        # Select best person by average keypoint confidence
        pose_confs = [kp.conf.mean().cpu().item() for kp in results.keypoints]
        best_idx = int(np.argmax(pose_confs))

        keypoints_xy = results.keypoints[best_idx].xy[0].cpu().numpy().astype(np.float32)
        keypoints_conf = results.keypoints[best_idx].conf[0].cpu().numpy().astype(np.float32)

        # Threshold low-confidence points
        low_conf_mask = keypoints_conf < CONF_THRESHOLD
        keypoints_xy[low_conf_mask] = np.nan

    kps_list.append(keypoints_xy)
    conf_list.append(keypoints_conf)

    # Draw on frame
    draw_frame = frame.copy()
    for idx, (x, y) in enumerate(keypoints_xy):
        if not np.any(np.isnan([x, y])):
            cv2.circle(draw_frame, (int(x), int(y)), 6, (0, 255, 0), -1)

    for i, j in SKELETON:
        if not np.any(np.isnan(keypoints_xy[i])) and not np.any(np.isnan(keypoints_xy[j])):
            pt1 = (int(keypoints_xy[i][0]), int(keypoints_xy[i][1]))
            pt2 = (int(keypoints_xy[j][0]), int(keypoints_xy[j][1]))
            cv2.line(draw_frame, pt1, pt2, (0, 255, 255), 3)

    cv2.putText(draw_frame, f"Frame {frame_idx + 1}/{total_frames}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    video_writer.write(draw_frame)
    cv2.imshow("YOLOv8 Pose Preview", draw_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Save results
kps = np.stack(kps_list, axis=0)
conf = np.stack(conf_list, axis=0)

np.save(OUT_NPY, kps)
np.save(OUT_CONF, conf)

print("\nDone!")
print(f"Saved 2D keypoints   → {OUT_NPY}     shape: {kps.shape}")
print(f"Saved confidences    → {OUT_CONF}   shape: {conf.shape}")
print(f"Saved preview video  → {OUT_VIDEO}")