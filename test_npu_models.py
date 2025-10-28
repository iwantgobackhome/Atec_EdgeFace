#!/usr/bin/env python3
"""
Test script for DeepX NPU models (YuNet and EdgeFace)
NPU 모델 출력 형식 확인 및 테스트
"""

import sys
import cv2
import numpy as np
from PIL import Image

# Test if dx_engine is available
try:
    from dx_engine import InferenceEngine
    print("✅ dx_engine imported successfully")
except ImportError as e:
    print(f"❌ Failed to import dx_engine: {e}")
    print("Please install DeepX NPU SDK")
    sys.exit(1)


def test_yunet_npu():
    """Test YuNet NPU model"""
    print("\n" + "="*60)
    print("Testing YuNet NPU Model")
    print("="*60)

    model_path = "face_alignment/models/face_detection_yunet_2023mar.dxnn"

    # Check if model exists
    import os
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print(f"   Please copy the DXNN model file to this location")
        return

    try:
        # Load model
        print(f"Loading model: {model_path}")
        ie = InferenceEngine(model_path)

        # Print model info
        print(f"✅ Model loaded successfully")
        print(f"📊 Input size: {ie.input_size()}")
        print(f"📊 Output dtype: {ie.output_dtype()}")

        # Find a test image
        test_image_paths = [
            "npu_calibration/calibration_output/calibration_dataset/000001.jpg",
            "npu_calibration/calibration_output/calibration_dataset/000002.jpg",
            "test_images/test.jpg",
        ]

        test_image_path = None
        for path in test_image_paths:
            if os.path.exists(path):
                test_image_path = path
                break

        if test_image_path is None:
            print(f"❌ No test image found. Tried: {test_image_paths}")
            return

        print(f"\nLoading test image: {test_image_path}")
        img = cv2.imread(test_image_path)
        if img is None:
            print(f"❌ Failed to load image: {test_image_path}")
            return

        print(f"Image shape: {img.shape}")

        # Preprocess (320x320, RGB, CHW, batch)
        resized = cv2.resize(img, (320, 320))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw_img = np.transpose(rgb_img, (2, 0, 1))
        input_tensor = np.expand_dims(chw_img, axis=0).astype(np.uint8)

        print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

        # Run inference
        print("\n🚀 Running inference...")
        outputs = ie.Run(input_tensor)

        # Analyze outputs
        print(f"\n📊 Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"  Output {i}:")
            print(f"    - Shape: {output.shape}")
            print(f"    - Dtype: {output.dtype}")
            print(f"    - Min: {output.min():.4f}, Max: {output.max():.4f}")
            print(f"    - Mean: {output.mean():.4f}, Std: {output.std():.4f}")

            # Print first few values
            flat = output.flatten()
            print(f"    - Flattened shape: {flat.shape}")
            print(f"    - First 10 values: {flat[:10]}")

            # Check if this looks like YuNet output format
            if len(output.shape) == 2 and output.shape[1] == 15:
                print(f"    - ✅ Detected YuNet format: (N={output.shape[0]}, 15)")
                print(f"    - Confidence scores (column 14): {output[:, 14]}")
            elif len(output.shape) == 3 and output.shape[2] == 15:
                print(f"    - ✅ Detected YuNet format: (batch={output.shape[0]}, N={output.shape[1]}, 15)")

        # Try to decode using our decoder
        print("\n🔍 Testing decoder...")
        try:
            from face_alignment.yunet_npu import YuNetNPUDetector

            # Create a dummy detector just for decoding
            detector = YuNetNPUDetector(model_path)

            # Decode outputs
            faces = detector._decode_outputs(outputs, scale_x=1.0, scale_y=1.0)

            print(f"✅ Decoded {len(faces)} faces")
            for idx, face in enumerate(faces):
                print(f"  Face {idx}: confidence={face[14]:.3f}, bbox=[{face[0]:.1f}, {face[1]:.1f}, {face[2]:.1f}, {face[3]:.1f}]")
        except Exception as e:
            print(f"⚠️ Decoder test failed: {e}")

        print("\n✅ YuNet NPU test completed")

    except Exception as e:
        print(f"❌ Error testing YuNet NPU: {e}")
        import traceback
        traceback.print_exc()


def test_edgeface_npu():
    """Test EdgeFace NPU model"""
    print("\n" + "="*60)
    print("Testing EdgeFace NPU Model")
    print("="*60)

    model_path = "checkpoints/edgeface_xs_gamma_06.dxnn"

    # Check if model exists
    import os
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print(f"   Please copy the DXNN model file to this location")
        return

    try:
        # Load model
        print(f"Loading model: {model_path}")
        ie = InferenceEngine(model_path)

        # Print model info
        print(f"✅ Model loaded successfully")
        print(f"📊 Input size: {ie.input_size()}")
        print(f"📊 Output dtype: {ie.output_dtype()}")

        # Find a test aligned face
        test_image_paths = [
            "npu_calibration/edgeface_calibration_output/aligned_faces/000001.jpg",
            "npu_calibration/edgeface_calibration_output/aligned_faces/000002.jpg",
            "captured_references/test.jpg",
        ]

        test_image_path = None
        for path in test_image_paths:
            if os.path.exists(path):
                test_image_path = path
                break

        if test_image_path is None:
            print(f"❌ No test image found. Tried: {test_image_paths}")
            print(f"   You can use any 112x112 face image for testing")
            return

        print(f"\nLoading test image: {test_image_path}")
        img = cv2.imread(test_image_path)
        if img is None:
            print(f"❌ Failed to load image: {test_image_path}")
            return

        print(f"Image shape: {img.shape}")

        # Preprocess (112x112, RGB, normalize, CHW, batch)
        resized = cv2.resize(img, (112, 112))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        float_img = rgb_img.astype(np.float32) / 255.0
        normalized = (float_img - 0.5) / 0.5
        chw_img = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(chw_img, axis=0).astype(np.float32)

        print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        print(f"Input tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")

        # Run inference
        print("\n🚀 Running inference...")
        outputs = ie.Run(input_tensor)

        # Analyze outputs
        print(f"\n📊 Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"  Output {i}:")
            print(f"    - Shape: {output.shape}")
            print(f"    - Dtype: {output.dtype}")
            print(f"    - Min: {output.min():.4f}, Max: {output.max():.4f}")
            print(f"    - Mean: {output.mean():.4f}, Std: {output.std():.4f}")

            # Check if this is the embedding vector
            flat = output.flatten()
            print(f"    - Flattened shape: {flat.shape}")

            if len(flat) == 512:
                print(f"    - ✅ Detected EdgeFace embedding: 512-d vector")

            l2_norm = np.linalg.norm(flat)
            print(f"    - L2 norm: {l2_norm:.4f}")

            # L2 normalize
            if l2_norm > 0:
                normalized_emb = flat / l2_norm
                print(f"    - L2 normalized norm: {np.linalg.norm(normalized_emb):.4f} (should be ~1.0)")

            print(f"    - First 10 values: {flat[:10]}")

        # Try to use the recognizer
        print("\n🔍 Testing recognizer...")
        try:
            from edgeface_npu_recognizer import EdgeFaceNPURecognizer

            recognizer = EdgeFaceNPURecognizer(model_path)
            embedding = recognizer.extract_embedding(img)

            print(f"✅ Extracted embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
            print(f"   First 10 values: {embedding[:10]}")
        except Exception as e:
            print(f"⚠️ Recognizer test failed: {e}")

        print("\n✅ EdgeFace NPU test completed")

    except Exception as e:
        print(f"❌ Error testing EdgeFace NPU: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function"""
    print("DeepX NPU Model Testing")
    print("="*60)

    # Test YuNet NPU
    test_yunet_npu()

    # Test EdgeFace NPU
    test_edgeface_npu()

    print("\n" + "="*60)
    print("All tests completed")
    print("="*60)


if __name__ == "__main__":
    main()
