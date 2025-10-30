#!/usr/bin/env python3
"""
Minimal YuNet output test - just print shapes
YuNet 출력 최소 테스트 - shape만 출력
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
print("YuNet NPU Minimal Output Test")
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
try:
    outputs = ie.run(input_tensor)
    print(f"✅ Inference completed!")
except Exception as e:
    print(f"❌ Inference failed: {e}")
    sys.exit(1)

# Just print basic info
print(f"\n📊 Number of outputs: {len(outputs)}")
print("\nOutput shapes:")
print("-" * 60)

try:
    for i in range(len(outputs)):
        out = outputs[i]
        print(f"Output {i}: shape = {out.shape}, dtype = {out.dtype}")
except Exception as e:
    print(f"Error printing shapes: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test completed")
print("=" * 60)
