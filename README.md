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

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/ThatE10/CV-final-project.git
cd CV-final-project
```

Install required packages (virtual environment recommended):

```bash
pip install ultralytics opencv-python matplotlib numpy
```

---

## 🚀 Usage

Run the full motion-capture pipeline:

```bash
python main.py
```

Test voxelization independently:

```bash
python FasterVoxelTest.py
```

---

## 🔧 Key Dependencies

* **Ultralytics YOLO** – For human detection and pose estimation.
* **OpenCV** – For video capture and image processing.
* **Matplotlib** – For visualization and diagnostic plotting.
* **NumPy** – For linear algebra, voxel math, and numerical operations.
