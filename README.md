Computer Vision Final Project: 3D Human Motion Capture & VoxelizationA Computer Vision project designed to extract 2D keypoints from video input using YOLO, reconstruct them into 3D space using voxelization techniques, and export the resulting motion animation to BVH (BioVision Hierarchy) format for use in 3D software (Blender, Unity, etc.).📂 File StructureThe project is organized as follows:CV-final-project/
├── lib/                           # Core library modules for 3D math and processing
├── scripts/                       # Utility scripts for setup or batch processing
├── FasterVoxelTest.py             # Script to test and visualize the voxelization algorithm
├── default_calibration.json       # Intrinsic/Extrinsic camera calibration parameters
├── export_bvh_humanoid_smooth.py  # Logic for smoothing 3D points and exporting to .bvh
├── main.py                        # Main entry point for the application
├── make_custom_dataset.py         # Tool for generating or formatting custom datasets
├── yolo2d.py                      # Wrapper for YOLO model to handle 2D detection
└── README.md                      # Project documentation
📝 File DescriptionsFileDescriptionmain.pyThe primary script that orchestrates the pipeline. It likely captures video, runs detection, performs 3D reconstruction, and handles the display loop.yolo2d.pyHandles the interface with the Ultralytics YOLO model. It is responsible for detecting humans in frames and returning 2D bounding boxes or keypoints.FasterVoxelTest.pyA testing script focused on the voxelization performance. It converts 2D detections into a 3D voxel grid to estimate depth and volume.export_bvh_humanoid_smooth.pyContains the logic to map tracked 3D points onto a humanoid skeleton structure and save the motion data into the standard .bvh motion capture format. Includes smoothing filters to reduce jitter.default_calibration.jsonStores the camera matrix and distortion coefficients necessary for accurate 3D reconstruction from 2D images.make_custom_dataset.pyA utility script to help prepare training or validation data, possibly for fine-tuning the YOLO model or calibrating the 3D solver.🛠️ InstallationClone the repository:git clone [https://github.com/ThatE10/CV-final-project.git](https://github.com/ThatE10/CV-final-project.git)
cd CV-final-project
Install dependencies:It is recommended to use a virtual environment.pip install ultralytics opencv-python matplotlib numpy
🚀 UsageTo run the main motion capture pipeline:python main.py
To test the voxelization logic independently:python FasterVoxelTest.py
🔧 DependenciesUltralytics (YOLO) - For state-of-the-art object and pose detection.OpenCV - For video handling and image processing.Matplotlib - For plotting and visualization.NumPy - For matrix operations and 3D math.🤝 ContributingContributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
