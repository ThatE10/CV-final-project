def debug_print(msg, force=False):
    """Print debug message if DEBUG is enabled"""
    if DEBUG or force:
        print(f"[DEBUG] {msg}")


def get_limbs_for_joints(num_joints):
    """Get skeleton limb connections based on number of joints"""
    if num_joints == 15:
        return LIMBS15
    elif num_joints == 17:
        return LIMBS17
    else:
        debug_print(f"Unknown joint count {num_joints}, using LIMBS15 as fallback", force=True)
        return LIMBS15  # !/usr/bin/env python3


"""
Webcam-based 3D Pose Estimation Inference
Adapted for single camera view with MPS device support
"""

import sys
import os
import json
import time

import torch
import torchvision
import numpy as np
import cv2
import torchvision.transforms as transforms

# ============================================================================
# CONFIGURATION - Modify these parameters as needed
# ============================================================================

# Paths
LIB_PATH = None
BACKBONE_FILE = './models/pose_resnet50_panoptic.pth.tar'
MODEL_FILE = 'models/pantopic_model_best.tar'
CONFIG_FILE = './models/campus.yaml'
CAM_FILE = 'calibration.json'
OUTPUT_DIR = 'output'

# Webcam settings
WEBCAM_ID = 0  # 0 for default webcam, change if you have multiple cameras
FRAME_WIDTH = 1920  # Set to None to use camera default
FRAME_HEIGHT = 1080  # Set to None to use camera default
FPS_TARGET = 30

# Model settings
DEVICE = 'mps'  # 'mps' for Apple Silicon, 'cuda' for NVIDIA GPU, 'cpu' for CPU
NUM_VIEWS = 1  # Single camera setup
SEQ_NAME = 'webcam_live'

# Display settings
SHOW_FPS = True
SHOW_HEATMAPS = False  # Set to True to display heatmap overlay
SAVE_OUTPUT = True  # Set to True to save annotated frames
DISPLAY_SCALE = 0.8  # Scale factor for display window (0.5 = 50% size)
MIN_CONFIDENCE = 0.1  # Minimum confidence to draw a keypoint

# Debug settings
DEBUG = True  # Enable detailed debugging output
DEBUG_EVERY_N_FRAMES = 1  # Print debug info every N frames

# ============================================================================
# SKELETON DEFINITIONS - Define limb connections for different joint counts
# ============================================================================

# 15 joints (Panoptic format)
LIMBS15 = [
    [0, 1], [1, 2], [2, 3],  # Right arm
    [0, 4], [4, 5], [5, 6],  # Left arm
    [0, 7], [7, 8], [8, 9],  # Spine
    [7, 10], [10, 11], [11, 12],  # Right leg
    [7, 13], [13, 14], [14, 15]  # Left leg
]

# 17 joints (COCO format)
LIMBS17 = [
    [0, 1], [1, 2], [2, 3],  # Right arm
    [0, 4], [4, 5], [5, 6],  # Left arm
    [0, 7], [7, 8], [8, 9],  # Spine
    [7, 10], [10, 11], [11, 12],  # Right leg
    [7, 13], [13, 14], [14, 15],  # Left leg
    [7, 16]  # Head/neck
]

# ============================================================================
# SETUP
# ============================================================================

if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

import lib.models as models
import matplotlib
from lib.core.config import config, update_config
from lib.utils.transforms import get_affine_transform, get_scale
from lib.utils.cameras import project_pose
from lib.utils.transforms import affine_transform_pts_cuda as do_transform
from lib.utils.vis import save_image_with_poses, save_2d_planes, is_valid_coord


def get_resize_transform(ori_image_size, image_size):
    """Obtain the resizing transform"""
    r = 0
    c = np.array([ori_image_size[0] / 2.0, ori_image_size[1] / 2.0])
    s = get_scale((ori_image_size[0], ori_image_size[1]), image_size)
    trans = get_affine_transform(c, s, r, image_size)
    return trans


def load_model(config, backbone_file, model_file, device):
    """Load the model and the Pose-ResNet backbone_model"""
    print('=> Loading models...')

    print(f'=> Creating backbone_model: {config.BACKBONE}')
    import lib.models.resnet as resnet_backbone_module
    backbone_model = resnet_backbone_module.get(config)

    print(f'=> Loading backbone_model weights from {backbone_file}')

    if device == 'mps':
        backbone_model.load_state_dict(torch.load(backbone_file, map_location='cpu'))
    else:
        backbone_model.load_state_dict(torch.load(backbone_file))

    backbone_model = backbone_model.to(device)
    backbone_model.eval()

    print(config.MODEL)
    print(":)" * 100)

    import lib.models.faster_voxelpose as faster_voxelpose
    model_module = faster_voxelpose

    if not hasattr(model_module, 'get'):
        raise ValueError(f"Model module '{config.MODEL}' does not have a 'get' function")

    print(f'=> Creating model: {config.MODEL}')
    config.DATASET.NUM_JOINTS = 15
    model = model_module.get(config)
    print(f'=> Loading model weights from {model_file}')

    if device == 'mps':
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(model_file))

    model = model.to(device)
    model.eval()

    print(f'=> Models loaded successfully on {device}')
    return backbone_model, model


def load_cameras(cam_file, seq_name):
    """Load camera calibration parameters"""
    if os.path.exists(cam_file):
        with open(cam_file) as f:
            cameras_data = json.load(f)
        print(f'=> Loaded camera calibration from {cam_file}')

        if seq_name in cameras_data:
            cameras = cameras_data
        elif any(k.isdigit() for k in cameras_data.keys()):
            cameras = {seq_name: cameras_data}
            print(f'=> Wrapped camera data with sequence name: {seq_name}')
        else:
            cameras = cameras_data

        cameras_with_int_keys = {}
        for seq, cams in cameras.items():
            cameras_with_int_keys[seq] = {}
            for cam_idx, cam_params in cams.items():
                idx = int(cam_idx) if isinstance(cam_idx, str) else cam_idx

                normalized_params = {}
                for key, value in cam_params.items():
                    normalized_key = key.upper() if key.lower() in ['t', 'r'] else key
                    normalized_params[normalized_key] = value

                if 'K' in normalized_params and 'fx' not in normalized_params:
                    K = normalized_params['K']
                    normalized_params['fx'] = K[0][0]
                    normalized_params['fy'] = K[1][1]
                    normalized_params['cx'] = K[0][2]
                    normalized_params['cy'] = K[1][2]

                if 'distCoef' in normalized_params:
                    distCoef = normalized_params['distCoef']
                    if 'k' not in normalized_params:
                        normalized_params['k'] = [distCoef[0], distCoef[1], distCoef[4] if len(distCoef) > 4 else 0.0]
                    if 'p' not in normalized_params:
                        normalized_params['p'] = [distCoef[2], distCoef[3]]

                cameras_with_int_keys[seq][idx] = normalized_params

        cameras = cameras_with_int_keys
        print(f'=> Converted camera indices to integers and normalized parameter names')

    else:
        print(f'=> Warning: Camera file {cam_file} not found, using default params')
        cameras = {
            seq_name: {
                0: {
                    'K': [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
                    'distCoef': [0.0, 0.0, 0.0, 0.0, 0.0],
                    'R': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    'T': [[0.0], [0.0], [3000.0]],
                    'fx': 1000.0,
                    'fy': 1000.0,
                    'cx': 960.0,
                    'cy': 540.0,
                    'k': [0.0, 0.0, 0.0],
                    'p': [0.0, 0.0]
                }
            }
        }
        print(f'=> Created default camera params for sequence: {seq_name}')

    return cameras


def create_default_calibration_file(filename, seq_name, num_cameras=1):
    """Create a default calibration file with proper format"""
    cameras = {seq_name: {}}

    for cam_idx in range(num_cameras):
        cameras[seq_name][str(cam_idx)] = {
            'K': [[1000.0, 0.0, 960.0],
                  [0.0, 1000.0, 540.0],
                  [0.0, 0.0, 1.0]],
            'distCoef': [0.0, 0.0, 0.0, 0.0, 0.0],
            'R': [[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]],
            'T': [[0.0], [0.0], [3000.0]],
            'fx': 1000.0,
            'fy': 1000.0,
            'cx': 960.0,
            'cy': 540.0,
            'k': [0.0, 0.0, 0.0],
            'p': [0.0, 0.0]
        }

    with open(filename, 'w') as f:
        json.dump(cameras, f, indent=2)

    print(f"=> Created default calibration file: {filename}")
    return cameras


def preprocess_frame(frame, transform, resize_transform_matrix):
    """Preprocess a single frame"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = transform(frame_rgb)
    return frame_tensor


def get_limbs_for_joints(num_joints):
    """Get skeleton limb connections based on number of joints"""
    if num_joints == 15:
        return LIMBS15
    elif num_joints == 17:
        return LIMBS17
    else:
        print(f"Warning: Unknown joint count {num_joints}, using LIMBS15 as fallback")
        return LIMBS15


def draw_poses_on_frame_3d(frame, poses_3d, cam_params, num_joints=15, min_confidence=MIN_CONFIDENCE, frame_idx=0):
    """
    Draw 3D poses projected onto 2D frame.

    Args:
        frame (np.array): The image to draw on
        poses_3d (np.array): 3D poses, shape [num_people, num_joints, 5] where last dim is [x, y, z, conf, ...]
        cam_params (dict): Camera parameters with 'fx', 'fy', 'cx', 'cy'
        num_joints (int): Number of joints
        min_confidence (float): Minimum confidence threshold
        frame_idx (int): Frame index for debugging

    Returns:
        np.array: Frame with poses drawn
    """
    if poses_3d is None or len(poses_3d) == 0:
        debug_print(f"Frame {frame_idx}: No poses to draw")
        return frame

    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    debug_print(f"Frame {frame_idx}: Drawing on {w}x{h} frame")

    limbs = get_limbs_for_joints(num_joints)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    num_people = len(poses_3d)
    debug_print(f"Frame {frame_idx}: Detected {num_people} people")

    keypoints_drawn = 0
    connections_drawn = 0

    for person_idx, pose_3d in enumerate(poses_3d):
        color = colors[person_idx % len(colors)]
        debug_print(f"Frame {frame_idx}: Processing person {person_idx}, pose shape: {pose_3d.shape}")

        # Project 3D points to 2D
        pose_2d = []
        valid_joints = 0
        for j in range(min(num_joints, len(pose_3d))):
            joint = pose_3d[j]

            # Extract coordinates
            x3d, y3d, z3d = joint[0], joint[1], joint[2]
            conf = joint[4] if len(joint) > 4 else 1.0

            debug_print(
                f"Frame {frame_idx}: Person {person_idx}, Joint {j}: pos=({x3d:.2f}, {y3d:.2f}, {z3d:.2f}), conf={conf:.3f}")

            if conf < min_confidence:
                debug_print(f"Frame {frame_idx}: Joint {j} filtered (conf {conf:.3f} < {min_confidence})")
                pose_2d.append(None)
                continue

            if z3d <= 0:
                debug_print(f"Frame {frame_idx}: Joint {j} filtered (z={z3d:.2f} <= 0)")
                pose_2d.append(None)
                continue

            # Project to 2D
            x2d = int(x3d / z3d * cam_params['fx'] + cam_params['cx'])
            y2d = int(y3d / z3d * cam_params['fy'] + cam_params['cy'])

            debug_print(f"Frame {frame_idx}: Joint {j} projected to ({x2d}, {y2d})")
            pose_2d.append((x2d, y2d, conf))
            valid_joints += 1

        debug_print(f"Frame {frame_idx}: Person {person_idx} has {valid_joints}/{num_joints} valid joints")

        # Draw keypoints
        for j, pt2d in enumerate(pose_2d):
            if pt2d is not None:
                x, y, conf = pt2d
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame_copy, (x, y), 4, color, -1)
                    keypoints_drawn += 1
                else:
                    debug_print(f"Frame {frame_idx}: Joint {j} at ({x}, {y}) out of bounds")

        # Draw skeleton connections
        for connection in limbs:
            if connection[0] >= len(pose_2d) or connection[1] >= len(pose_2d):
                continue

            pt1 = pose_2d[connection[0]]
            pt2 = pose_2d[connection[1]]

            if pt1 is not None and pt2 is not None:
                x1, y1, _ = pt1
                x2, y2, _ = pt2

                if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                    cv2.line(frame_copy, (x1, y1), (x2, y2), color, 2)
                    connections_drawn += 1

    debug_print(f"Frame {frame_idx}: Drew {keypoints_drawn} keypoints and {connections_drawn} connections")
    return frame_copy


def main():
    print("=" * 80)
    print("Webcam 3D Pose Estimation - Live Inference")
    print("=" * 80)

    # Check device availability
    if DEVICE == 'mps' and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        device = 'cpu'
    elif DEVICE == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'
    else:
        device = DEVICE

    print(f"Using device: {device}")

    # Load configuration
    update_config(CONFIG_FILE)
    config.DEVICE = device

    # Setup preprocessing
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    ori_image_size = config.DATASET.ORI_IMAGE_SIZE
    image_size = config.DATASET.IMAGE_SIZE
    resize_transform_matrix = get_resize_transform(ori_image_size, image_size)
    resize_transform_tensor = torch.as_tensor(
        resize_transform_matrix,
        dtype=torch.float,
        device=device
    )

    # Load models
    backbone, model = load_model(config, BACKBONE_FILE, MODEL_FILE, device)

    # Load camera calibration
    cameras = load_cameras(CAM_FILE, SEQ_NAME)

    if SEQ_NAME not in cameras:
        print(f"ERROR: Sequence '{SEQ_NAME}' not found in cameras!")
        print(f"Available sequences: {list(cameras.keys())}")
        return

    print(f"=> Camera structure verified for sequence: {SEQ_NAME}")
    print(f"=> Number of cameras: {len(cameras[SEQ_NAME])}")
    print(f"=> Camera indices: {list(cameras[SEQ_NAME].keys())}")

    # Ensure camera indices are INTEGERS
    cameras_fixed = {SEQ_NAME: {}}
    for cam_idx, cam_params in cameras[SEQ_NAME].items():
        idx = int(cam_idx) if isinstance(cam_idx, str) else cam_idx
        cameras_fixed[SEQ_NAME][idx] = cam_params

    cameras = cameras_fixed
    print(f"=> Fixed camera indices to integers: {list(cameras[SEQ_NAME].keys())}")

    # Setup metadata
    meta = {'seq': [SEQ_NAME]}

    # Create output directory
    if SAVE_OUTPUT:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Saving output to: {OUTPUT_DIR}")

    # Open webcam
    print(f"Opening webcam {WEBCAM_ID}...")
    cap = cv2.VideoCapture(WEBCAM_ID)

    if not cap.isOpened():
        print(f"Error: Could not open webcam {WEBCAM_ID}")
        return

    if FRAME_WIDTH is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    if FRAME_HEIGHT is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution: {actual_width}x{actual_height}")

    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    frame_count = 0

    print("\nStarting inference... Press 'q' to quit")
    print("-" * 80)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            target_width, target_height = config.DATASET.IMAGE_SIZE
            frame_resized = cv2.resize(frame, (target_width, target_height))

            frame_tensor = preprocess_frame(frame_resized, transform, resize_transform_matrix)
            inputs = frame_tensor.unsqueeze(0).unsqueeze(0).to(device)

            debug_print(f"Frame {frame_count}: Input tensor shape: {inputs.shape}")

            inference_start = time.time()
            with torch.no_grad():
                fused_poses, plane_poses, proposal_centers, input_heatmaps, _ = model(
                    backbone=backbone,
                    views=inputs,
                    meta=meta,
                    cameras=cameras,
                    resize_transform=resize_transform_tensor
                )
            inference_time = time.time() - inference_start

            debug_print(f"Frame {frame_count}: Inference took {inference_time * 1000:.1f}ms")

            frame_drawn = frame_resized.copy()

            # Draw 3D poses projected onto 2D frame
            if fused_poses is not None:
                debug_print(
                    f"Frame {frame_count}: fused_poses type={type(fused_poses)}, is None: {fused_poses is None}")

                if isinstance(fused_poses, torch.Tensor):
                    debug_print(
                        f"Frame {frame_count}: fused_poses shape={fused_poses.shape}, dtype={fused_poses.dtype}")
                    fused_poses_np = fused_poses.cpu().numpy()
                else:
                    debug_print(f"Frame {frame_count}: fused_poses is numpy array, shape={fused_poses.shape}")
                    fused_poses_np = fused_poses

                debug_print(f"Frame {frame_count}: fused_poses_np shape={fused_poses_np.shape}")

                if len(fused_poses_np) > 0:
                    poses_batch = fused_poses_np[0]  # Get first (only) batch
                    debug_print(f"Frame {frame_count}: poses_batch shape={poses_batch.shape}")

                    num_joints = poses_batch.shape[1] if len(poses_batch.shape) > 1 else 15
                    debug_print(f"Frame {frame_count}: num_joints={num_joints}")

                    cam_params = cameras[SEQ_NAME][0]
                    debug_print(
                        f"Frame {frame_count}: Camera params: fx={cam_params.get('fx')}, fy={cam_params.get('fy')}, cx={cam_params.get('cx')}, cy={cam_params.get('cy')}")

                    frame_drawn = draw_poses_on_frame_3d(
                        frame_drawn,
                        poses_batch,
                        cam_params,
                        num_joints=num_joints,
                        min_confidence=MIN_CONFIDENCE,
                        frame_idx=frame_count
                    )
                else:
                    debug_print(f"Frame {frame_count}: fused_poses_np is empty")
            else:
                debug_print(f"Frame {frame_count}: fused_poses is None", force=True)

            # Overlay heatmaps if enabled
            if SHOW_HEATMAPS and input_heatmaps is not None:
                debug_print(f"Frame {frame_count}: Drawing heatmaps")
                heatmaps_np = input_heatmaps[0].cpu().numpy()
                combined_heatmap = np.sum(heatmaps_np, axis=0)
                combined_heatmap = np.nan_to_num(combined_heatmap, nan=0.0, posinf=0.0, neginf=0.0)

                combined_heatmap -= combined_heatmap.min()
                if combined_heatmap.max() > 0:
                    combined_heatmap = combined_heatmap / combined_heatmap.max()
                combined_heatmap_uint8 = (combined_heatmap * 255).astype(np.uint8)

                colored_heatmap = cv2.applyColorMap(combined_heatmap_uint8, cv2.COLORMAP_JET)
                colored_heatmap = cv2.resize(colored_heatmap, (frame_resized.shape[1], frame_resized.shape[0]))

                alpha = 0.5
                frame_drawn = cv2.addWeighted(frame_drawn, 1 - alpha, colored_heatmap, alpha, 0)

            # Display FPS and inference time
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()

            if SHOW_FPS:
                cv2.putText(frame_drawn, f'FPS: {current_fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_drawn, f'Inference: {inference_time * 1000:.1f}ms', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if DISPLAY_SCALE != 1.0:
                display_width = int(frame_drawn.shape[1] * DISPLAY_SCALE)
                display_height = int(frame_drawn.shape[0] * DISPLAY_SCALE)
                frame_display = cv2.resize(frame_drawn, (display_width, display_height))
            else:
                frame_display = frame_drawn

            cv2.imshow('Pose Estimation', frame_display)

            if SAVE_OUTPUT:
                output_path = os.path.join(OUTPUT_DIR, f'frame_{frame_count:06d}.jpg')
                cv2.imwrite(output_path, frame_drawn)

            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\nQuitting...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames")


if __name__ == '__main__':
    main()