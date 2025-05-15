import torch

import cv2
import numpy as np

# Check if CUDA is available and print the GPU device name
print("CUDA Available:", torch.cuda.is_available())
print("GPU Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Detected")

# Check if OpenCV is using CUDA
img = np.zeros((224, 224, 3), dtype=np.uint8)
img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

# Print the shape of the tensor
print("Torch Tensor shape:", img_tensor.shape)
print("Running on CUDA:", torch.cuda.is_available())