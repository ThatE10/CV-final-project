import numpy as np

# Generate dummy pose data (T=100 frames, J=17 joints, C=3 coords)
# Using random data but ensuring it's not all zeros to avoid div by zero in normalization
poses = np.random.rand(100, 17, 3).astype(np.float64)
np.save("pose3d.npy", poses)
print("Created dummy pose3d.npy")
