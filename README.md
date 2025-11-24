# Computer Vision Final Project: 3D Human Motion Capture

This project implements a full pipeline for recovering 3D human motion from monocular video. It detects 2D keypoints using a YOLO-based pose model, reconstructs motion in 3D space through voxelization and geometric reasoning, and exports the resulting animation to BVH (BioVision Hierarchy) format for use in 3D applications such as Blender or Unity.

---

## 📂 Project Structure

```
CV-final-project/
├── lib/                           # Core modules for 3D math, geometry, and utilities
├── scripts/                       # Setup utilities and batch-processing tools
├── FasterVoxelTest.py             # Standalone voxelization and visualization test
├── default_calibration.json       # Camera intrinsic and extrinsic parameters
├── export_bvh_humanoid_smooth.py  # 3D smoothing + BVH export logic
├── main.py                        # Main application entry point
├── make_custom_dataset.py         # Dataset creation and formatting utilities
├── yolo2d.py                      # Wrapper for YOLO-based 2D human detection
└── README.md                      # Documentation
```

---

## 📝 File Descriptions

| File                              | Description                                                                                                            |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **main.py**                       | Coordinates the full pipeline: video capture, 2D detection, voxel-based 3D reconstruction, and visualization.          |
| **yolo2d.py**                     | Interfaces with the Ultralytics YOLO model to obtain human detections and keypoints from video frames.                 |
| **FasterVoxelTest.py**            | Provides a focused environment for evaluating voxel reconstruction performance and visualization.                      |
| **export_bvh_humanoid_smooth.py** | Maps reconstructed motion to a humanoid skeleton and exports a BVH file. Includes smoothing filters to mitigate noise. |
| **default_calibration.json**      | Contains the camera matrix and distortion coefficients required for accurate 3D reconstruction.                        |
| **make_custom_dataset.py**        | Supports dataset generation for training or calibration tasks, including formatting and preprocessing steps.           |
| **generate_dummy_pose.py**        | Generates dummy `pose3d.npy` data for testing the BVH conversion script without running the full pipeline.             |

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/ThatE10/CV-final-project.git
cd CV-final-project
```

Install required packages (virtual environment recommended):

```bash
pip install ultralytics opencv-python matplotlib numpy scipy easydict torch
```

### Optional: VideoPose3D Dependencies
If you plan to extract 3D poses from video using VideoPose3D, you will also need `detectron2`. Please refer to the [Detectron2 installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

---

## 🚀 Usage

### 1. YOLOv8 2D Pose Extraction (Alternative to Detectron2)
To extract 2D keypoints from a video using YOLOv8:

1.  Open `yolo2d.py` and update the `VIDEO_PATH` variable to point to your `input.mp4`.
2.  Run the script:
    ```bash
    python yolo2d.py
    ```
3.  **Outputs**:
    *   `kps2d.npy`: 2D keypoints [Frames, 17, 2]
    *   `kps2d_conf.npy`: Confidence scores [Frames, 17]
    *   `input_with_kpts.mp4`: Visualization video

### 2. FasterVoxelPose (Standalone Voxelization Test)
To test the voxel reconstruction logic independently (e.g., with a webcam or configured source):

1.  Open `FasterVoxelTest.py` and configure the settings at the top (e.g., `WEBCAM_ID`, `video_file`, etc.).
2.  Run the script:
    ```bash
    python FasterVoxelTest.py
    ```

### 3. Extracting 3D Poses from Video (VideoPose3D)
To convert a video (`input.mp4`) into 3D pose data (`pose3d.npy`) using the included `VideoPose3D` library (requires `detectron2`):

1.  **Infer 2D Keypoints**:
    ```bash
    cd VideoPose3D/inference
    python infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir ../data/detections --image-ext mp4 ../../
    cd ../..
    ```
2.  **Prepare 2D Data**:
    ```bash
    cd VideoPose3D/data
    python prepare_data_2d_custom.py -i ../data/detections -o myvideos
    cd ../..
    ```
3.  **Infer 3D Poses**:
    ```bash
    cd VideoPose3D
    # Ensure 'pretrained_h36m_detectron_coco.bin' is in the checkpoint directory
    python run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --viz-export ../pose3d.npy --viz-subject input.mp4 --viz-action custom --viz-camera 0 --viz-video ../input.mp4 --viz-size 6
    cd ..
    ```

### 4. Convert 3D Poses to BVH
Once you have a `pose3d.npy` file (from VideoPose3D or other sources), you can convert it to a BVH animation:

1.  Run the conversion script:
    ```bash
    python export_bvh_humanoid_smooth.py
    ```
2.  **Output**: `pose_humanoid_smooth.bvh`

**Tip:** If you don't have `pose3d.npy`, you can generate dummy data for testing:
```bash
python generate_dummy_pose.py
```

---

## 🔧 Key Dependencies

* **Ultralytics YOLO** – For human detection and pose estimation.
* **OpenCV** – For video capture and image processing.
* **Matplotlib** – For visualization and diagnostic plotting.
* **NumPy** – For linear algebra, voxel math, and numerical operations.
