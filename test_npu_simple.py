#!/usr/bin/env python3
"""
Simple NPU model test - minimal version for debugging
NPU 모델 간단 테스트 - 디버깅용 최소 버전
"""

import sys
import os

print("=" * 60)
print("Simple NPU Model Test")
print("=" * 60)

# Step 1: Check dx_engine availability
print("\n[Step 1] Checking dx_engine...")
try:
    from dx_engine import InferenceEngine
    print("✅ dx_engine imported successfully")
except ImportError as e:
    print(f"❌ Failed to import dx_engine: {e}")
    sys.exit(1)

# Step 2: Check model file
print("\n[Step 2] Checking model file...")
model_path = "face_alignment/models/face_detection_yunet_2023mar.dxnn"

if not os.path.exists(model_path):
    print(f"❌ Model file not found: {model_path}")
    print("   Current directory:", os.getcwd())
    print("   Files in face_alignment/models/:")
    if os.path.exists("face_alignment/models/"):
        for f in os.listdir("face_alignment/models/"):
            print(f"     - {f}")
    sys.exit(1)

file_size = os.path.getsize(model_path)
print(f"✅ Model file found: {model_path}")
print(f"   File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

# Step 3: Try to load model
print("\n[Step 3] Loading model...")
print(f"   Model path: {model_path}")

try:
    print("   Calling InferenceEngine(model_path)...")
    ie = InferenceEngine(model_path)
    print("✅ Model loaded successfully!")

    # Print model info
    print("\n[Step 4] Model information:")
    try:
        input_size = ie.input_size()
        print(f"   Input size: {input_size}")
    except Exception as e:
        print(f"   ⚠️ Could not get input size: {e}")

    try:
        output_dtype = ie.output_dtype()
        print(f"   Output dtype: {output_dtype}")
    except Exception as e:
        print(f"   ⚠️ Could not get output dtype: {e}")

    print("\n✅ All checks passed!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"   Error type: {type(e).__name__}")

    # Print stack trace
    import traceback
    print("\n   Stack trace:")
    traceback.print_exc()

    sys.exit(1)

print("\n" + "=" * 60)
print("Test completed successfully")
print("=" * 60)
