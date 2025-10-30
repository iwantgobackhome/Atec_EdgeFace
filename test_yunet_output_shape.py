#!/usr/bin/env python3
"""
Test YuNet NPU output shapes with actual image
YuNet NPU 실제 이미지로 출력 형식 확인
"""

import sys
import cv2
import numpy as np

try:
    from dx_engine import InferenceEngine
    print("✅ dx_engine imported successfully")
except ImportError as e:
    print(f"❌ Failed to import dx_engine: {e}")
    sys.exit(1)

print("=" * 60)
print("YuNet NPU Output Shape Analysis")
print("=" * 60)

model_path = "face_alignment/models/face_detection_yunet_2023mar.dxnn"

# Load model
print(f"\nLoading model: {model_path}")
ie = InferenceEngine(model_path)
print("✅ Model loaded")

# Find test image
import os
test_image_paths = [
    "/home/dxdemo/Atec_EdgeFace/npu_calibration/calibration_output/calibration_dataset/calib_0000.jpg",
    "npu_calibration/calibration_output/calibration_dataset/calib_0000.jpg",
]

test_image_path = None
for path in test_image_paths:
    if os.path.exists(path):
        test_image_path = path
        break

if test_image_path is None:
    print("❌ No test image found")
    sys.exit(1)

print(f"\nLoading test image: {test_image_path}")
img = cv2.imread(test_image_path)
print(f"Image shape: {img.shape}")

# Preprocess (320x320, RGB, CHW, batch)
resized = cv2.resize(img, (320, 320))
rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
chw_img = np.transpose(rgb_img, (2, 0, 1))
input_tensor = np.expand_dims(chw_img, axis=0).astype(np.uint8)

print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

# Run inference
print("\n🚀 Running inference...")
outputs = ie.run(input_tensor)

# Analyze each output in detail
print(f"\n📊 Number of outputs: {len(outputs)}")
print("\nDetailed output analysis:")
print("-" * 60)

for i, output in enumerate(outputs):
    print(f"\nOutput {i}:")
    print(f"  Shape: {output.shape}")
    print(f"  Dtype: {output.dtype}")
    print(f"  Min: {output.min():.4f}, Max: {output.max():.4f}")
    print(f"  Mean: {output.mean():.4f}, Std: {output.std():.4f}")

    # Check if this could be detection output
    flat = output.flatten()
    print(f"  Total elements: {flat.shape[0]}")

    # Print actual values
    if flat.shape[0] <= 100:
        print(f"  All values: {flat}")
    else:
        print(f"  First 20 values: {flat[:20]}")
        print(f"  Last 20 values: {flat[-20:]}")

    # Check for possible detection format
    if len(output.shape) == 2:
        print(f"  → 2D tensor: might be (N_detections, features)")
        if output.shape[1] == 15:
            print(f"    ✅ This looks like YuNet detection format!")
            print(f"    Number of detections: {output.shape[0]}")
    elif len(output.shape) == 3:
        print(f"  → 3D tensor: (batch={output.shape[0]}, H={output.shape[1]}, W={output.shape[2]})")
    elif len(output.shape) == 4:
        print(f"  → 4D tensor: (batch={output.shape[0]}, H={output.shape[1]}, W={output.shape[2]}, C={output.shape[3]})")

print("\n" + "=" * 60)
print("Analysis completed")
print("=" * 60)

# Recommendations
print("\n💡 Recommendations:")
print("1. Look for output with shape (N, 15) or similar")
print("2. Check which output has reasonable bbox/landmark values")
print("3. Values should be in image coordinate range (0-320)")
