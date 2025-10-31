#!/usr/bin/env python3
"""
Test full NPU integration: YuNet NPU + EdgeFace NPU
YuNet NPU와 EdgeFace NPU 통합 테스트
"""

import sys
import cv2
import numpy as np
from PIL import Image

# Import the face recognition system
from face_recognition_system import FaceRecognitionSystem

def test_npu_full_system():
    """Test YuNet NPU detector + EdgeFace NPU recognizer"""

    print("=" * 80)
    print("Testing Full NPU Integration: YuNet NPU + EdgeFace NPU")
    print("=" * 80)

    # Initialize system with NPU
    print("\n1. Initializing Face Recognition System with NPU...")
    try:
        system = FaceRecognitionSystem(
            detector_method='yunet_npu',  # This will automatically enable EdgeFace NPU
            edgeface_model_path='checkpoints/edgeface_xs_gamma_06.dxnn',
            edgeface_model_name='edgeface_xs_gamma_06',
            device='npu',  # Explicit NPU device
            similarity_threshold=0.5
        )
        print("✅ System initialized successfully with NPU!")
        print(f"   - Detector: yunet_npu")
        print(f"   - EdgeFace: NPU mode = {system.use_npu}")
        print(f"   - Device: {system.device}")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test with a sample image
    print("\n2. Testing with sample image...")

    # Create a test image (or use webcam)
    print("   Opening webcam for testing...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("✅ Camera opened")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press SPACE to test detection + recognition")
    print("\nRunning live preview...")

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Failed to grab frame")
            break

        frame_count += 1

        # Process frame
        try:
            annotated_frame, detections = system.process_frame(frame)

            # Show NPU info on frame
            cv2.putText(annotated_frame, "NPU MODE: YuNet NPU + EdgeFace NPU", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display
            cv2.imshow('NPU Full System Test', annotated_frame)

            # Print detection info every 30 frames
            if frame_count % 30 == 0 and len(detections) > 0:
                print(f"\n[Frame {frame_count}] Detected {len(detections)} face(s):")
                for i, det in enumerate(detections):
                    print(f"  Face {i+1}: {det['person_id']} (similarity: {det['similarity']:.3f})")

        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            import traceback
            traceback.print_exc()
            cv2.imshow('NPU Full System Test', frame)

        # Handle key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            # Test detection + recognition on current frame
            print("\n" + "=" * 80)
            print("Testing detection + recognition on current frame...")
            print("=" * 80)

            try:
                # Convert to PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

                # Detect faces
                bboxes, landmarks = system.detector.detect_faces(pil_img)
                print(f"\n✅ Detected {len(bboxes)} face(s)")

                if len(bboxes) > 0:
                    for i, (bbox, lm) in enumerate(zip(bboxes, landmarks)):
                        print(f"\nFace {i+1}:")
                        print(f"  - BBox: {bbox[:4]}")
                        print(f"  - Confidence: {bbox[4]:.3f}")

                        # Align face
                        aligned_face = system.detector.align_face(pil_img, lm)

                        if aligned_face is not None:
                            # Convert to numpy BGR
                            face_np = np.array(aligned_face)
                            face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

                            # Extract embedding using NPU
                            embedding = system.recognizer.extract_embedding(face_np)
                            print(f"  - Embedding shape: {embedding.shape}")
                            print(f"  - Embedding norm: {np.linalg.norm(embedding):.6f}")
                            print(f"  - Embedding mean: {embedding.mean():.6f}")
                            print(f"  - Embedding std: {embedding.std():.6f}")

                            # Show aligned face
                            cv2.imshow(f'Aligned Face {i+1}', face_np)
                        else:
                            print(f"  - Failed to align face")

            except Exception as e:
                print(f"❌ Test failed: {e}")
                import traceback
                traceback.print_exc()

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test completed!")

if __name__ == '__main__':
    test_npu_full_system()
