#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Profiling Script for YuNet-EdgeFace Pipeline
병목 지점 측정 스크립트
"""

import cv2
import numpy as np
import time
from pathlib import Path
from PIL import Image
import sys

sys.path.insert(0, 'face_alignment')

def profile_detector(detector_name, use_npu=False):
    """Profile face detector performance"""
    print(f"\n{'='*60}")
    print(f"Profiling: {detector_name} (NPU: {use_npu})")
    print(f"{'='*60}")

    # Import detector
    from face_alignment.unified_detector import UnifiedFaceDetector

    device = 'npu' if use_npu else 'cpu'
    detector = UnifiedFaceDetector(detector_name, device=device)

    # Load test image
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Cannot capture test frame")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # Warmup
    print("Warming up...")
    for _ in range(5):
        detector.detect_faces(pil_img)

    # Profile detection
    print("\nProfiling face detection...")
    num_iterations = 30
    times = []

    for i in range(num_iterations):
        start = time.perf_counter()
        bboxes, landmarks = detector.detect_faces(pil_img)
        end = time.perf_counter()

        elapsed = (end - start) * 1000  # ms
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"  Iteration {i+1}/{num_iterations}: {elapsed:.2f} ms")

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1000.0 / avg_time

    print(f"\n📊 Detection Performance:")
    print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    print(f"  FPS: {fps:.1f}")

    if bboxes is not None and len(bboxes) > 0:
        print(f"  Faces detected: {len(bboxes)}")

        # Profile alignment
        print("\nProfiling face alignment...")
        align_times = []

        for i in range(num_iterations):
            start = time.perf_counter()
            aligned_face = detector.align(pil_img)
            end = time.perf_counter()

            elapsed = (end - start) * 1000  # ms
            align_times.append(elapsed)

        avg_align = np.mean(align_times)
        std_align = np.std(align_times)

        print(f"  Alignment average: {avg_align:.2f} ± {std_align:.2f} ms")

    return {
        'detector': detector_name,
        'use_npu': use_npu,
        'avg_detection_ms': avg_time,
        'std_detection_ms': std_time,
        'fps': fps,
        'avg_alignment_ms': avg_align if bboxes is not None and len(bboxes) > 0 else 0
    }


def profile_recognizer(model_path, use_npu=False):
    """Profile EdgeFace recognizer performance"""
    print(f"\n{'='*60}")
    print(f"Profiling: EdgeFace Recognizer (NPU: {use_npu})")
    print(f"{'='*60}")

    if use_npu:
        from edgeface_npu_recognizer import EdgeFaceNPURecognizer
        recognizer = EdgeFaceNPURecognizer(model_path, device='npu')
    else:
        from face_recognition_system import EdgeFaceRecognizer
        recognizer = EdgeFaceRecognizer(model_path, device='cpu')

    # Create dummy face image (112x112x3)
    face_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    # Warmup
    print("Warming up...")
    for _ in range(5):
        recognizer.extract_embedding(face_img)

    # Profile single embedding extraction
    print("\nProfiling single embedding extraction...")
    num_iterations = 50
    times = []

    for i in range(num_iterations):
        start = time.perf_counter()
        embedding = recognizer.extract_embedding(face_img)
        end = time.perf_counter()

        elapsed = (end - start) * 1000  # ms
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"  Iteration {i+1}/{num_iterations}: {elapsed:.2f} ms")

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"\n📊 Single Embedding Performance:")
    print(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")

    # Profile batch embedding extraction
    print("\nProfiling batch embedding extraction...")
    batch_sizes = [1, 2, 4, 8]
    batch_results = {}

    for batch_size in batch_sizes:
        face_batch = [face_img.copy() for _ in range(batch_size)]

        # Warmup
        for _ in range(3):
            recognizer.extract_embeddings_batch(face_batch)

        times = []
        for i in range(30):
            start = time.perf_counter()
            embeddings = recognizer.extract_embeddings_batch(face_batch)
            end = time.perf_counter()

            elapsed = (end - start) * 1000  # ms
            times.append(elapsed)

        avg_batch_time = np.mean(times)
        per_face_time = avg_batch_time / batch_size

        batch_results[batch_size] = {
            'total_ms': avg_batch_time,
            'per_face_ms': per_face_time
        }

        print(f"  Batch size {batch_size}: {avg_batch_time:.2f} ms total, {per_face_time:.2f} ms per face")

    return {
        'recognizer': 'EdgeFace',
        'use_npu': use_npu,
        'avg_single_ms': avg_time,
        'std_single_ms': std_time,
        'batch_results': batch_results
    }


def profile_end_to_end(detector_name, model_path, use_npu=False):
    """Profile end-to-end pipeline performance"""
    print(f"\n{'='*60}")
    print(f"Profiling: End-to-End Pipeline (NPU: {use_npu})")
    print(f"Detector: {detector_name}, Model: {model_path}")
    print(f"{'='*60}")

    from face_recognition_system import FaceRecognitionSystem

    system = FaceRecognitionSystem(
        detector_method=detector_name,
        edgeface_model_path=model_path,
        device='npu' if use_npu else 'cpu',
        use_npu=use_npu
    )

    # Capture test frames
    print("\nCapturing test frames...")
    cap = cv2.VideoCapture(0)
    frames = []
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    if not frames:
        print("Cannot capture test frames")
        return

    # Warmup
    print("Warming up...")
    for _ in range(5):
        system.process_frame(frames[0])

    # Profile end-to-end processing
    print("\nProfiling end-to-end processing...")
    times = {
        'total': [],
        'detection': [],
        'alignment': [],
        'embedding': [],
        'matching': []
    }

    for i, frame in enumerate(frames):
        start_total = time.perf_counter()

        # Detection
        start_det = time.perf_counter()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        bboxes, landmarks = system.detector.detect_faces(pil_img)
        end_det = time.perf_counter()
        times['detection'].append((end_det - start_det) * 1000)

        if bboxes is not None and len(bboxes) > 0:
            # Alignment
            start_align = time.perf_counter()
            aligned_faces = []
            for lm in landmarks:
                aligned_face = system.detector.align_face(pil_img, lm)
                if aligned_face is not None:
                    face_np = np.array(aligned_face)
                    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                    aligned_faces.append(face_np)
            end_align = time.perf_counter()
            times['alignment'].append((end_align - start_align) * 1000)

            # Embedding extraction
            if aligned_faces:
                start_emb = time.perf_counter()
                embeddings = system.recognizer.extract_embeddings_batch(aligned_faces)
                end_emb = time.perf_counter()
                times['embedding'].append((end_emb - start_emb) * 1000)

                # Matching
                start_match = time.perf_counter()
                for emb in embeddings:
                    person_id, similarity = system.ref_db.find_match(emb)
                end_match = time.perf_counter()
                times['matching'].append((end_match - start_match) * 1000)

        end_total = time.perf_counter()
        times['total'].append((end_total - start_total) * 1000)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(frames)} frames")

    print(f"\n📊 End-to-End Performance Breakdown:")
    for stage, stage_times in times.items():
        if stage_times:
            avg = np.mean(stage_times)
            std = np.std(stage_times)
            percentage = (avg / np.mean(times['total'])) * 100 if stage != 'total' else 100
            print(f"  {stage.capitalize():12s}: {avg:6.2f} ± {std:5.2f} ms ({percentage:5.1f}%)")

    total_avg = np.mean(times['total'])
    fps = 1000.0 / total_avg
    print(f"\n  Total FPS: {fps:.1f}")

    return {
        'detector': detector_name,
        'use_npu': use_npu,
        'avg_total_ms': np.mean(times['total']),
        'avg_detection_ms': np.mean(times['detection']) if times['detection'] else 0,
        'avg_alignment_ms': np.mean(times['alignment']) if times['alignment'] else 0,
        'avg_embedding_ms': np.mean(times['embedding']) if times['embedding'] else 0,
        'avg_matching_ms': np.mean(times['matching']) if times['matching'] else 0,
        'fps': fps
    }


def main():
    """Main profiling function"""
    print("\n" + "="*60)
    print("EdgeFace Performance Profiling Tool")
    print("="*60)

    results = {}

    # Profile YuNet CPU
    print("\n\n### Test 1: YuNet (CPU) ###")
    results['yunet_cpu'] = profile_end_to_end('yunet', 'checkpoints/edgeface_xs_gamma_06.pt', use_npu=False)

    # Profile YuNet NPU
    print("\n\n### Test 2: YuNet (NPU) + EdgeFace (NPU) ###")
    results['yunet_npu'] = profile_end_to_end('yunet_npu', 'checkpoints/edgeface_xs_gamma_06.dxnn', use_npu=True)

    # Summary
    print("\n\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)

    cpu_fps = results['yunet_cpu']['fps']
    npu_fps = results['yunet_npu']['fps']
    speedup = npu_fps / cpu_fps

    print(f"\nCPU Configuration:")
    print(f"  Total FPS: {cpu_fps:.1f}")
    print(f"  Detection: {results['yunet_cpu']['avg_detection_ms']:.2f} ms")
    print(f"  Alignment: {results['yunet_cpu']['avg_alignment_ms']:.2f} ms")
    print(f"  Embedding: {results['yunet_cpu']['avg_embedding_ms']:.2f} ms")
    print(f"  Matching:  {results['yunet_cpu']['avg_matching_ms']:.2f} ms")

    print(f"\nNPU Configuration:")
    print(f"  Total FPS: {npu_fps:.1f}")
    print(f"  Detection: {results['yunet_npu']['avg_detection_ms']:.2f} ms")
    print(f"  Alignment: {results['yunet_npu']['avg_alignment_ms']:.2f} ms")
    print(f"  Embedding: {results['yunet_npu']['avg_embedding_ms']:.2f} ms")
    print(f"  Matching:  {results['yunet_npu']['avg_matching_ms']:.2f} ms")

    print(f"\n🚀 Speedup: {speedup:.2f}x")

    # Identify bottleneck
    npu_times = {
        'Detection': results['yunet_npu']['avg_detection_ms'],
        'Alignment': results['yunet_npu']['avg_alignment_ms'],
        'Embedding': results['yunet_npu']['avg_embedding_ms'],
        'Matching': results['yunet_npu']['avg_matching_ms']
    }

    bottleneck = max(npu_times.items(), key=lambda x: x[1])
    print(f"\n⚠️  NPU Bottleneck: {bottleneck[0]} ({bottleneck[1]:.2f} ms)")

    # Optimization recommendations
    print(f"\n💡 Optimization Recommendations:")

    if results['yunet_npu']['avg_detection_ms'] > 50:
        print(f"  1. YuNet NPU detection is slow ({results['yunet_npu']['avg_detection_ms']:.1f}ms)")
        print(f"     - Reduce debug prints in yunet_npu.py")
        print(f"     - Optimize NMS implementation")
        print(f"     - Consider lower resolution input")

    if results['yunet_npu']['avg_embedding_ms'] > 20:
        print(f"  2. Embedding extraction is slow ({results['yunet_npu']['avg_embedding_ms']:.1f}ms)")
        print(f"     - Reduce memory copies in preprocessing")
        print(f"     - Implement true batch processing for NPU")

    if results['yunet_npu']['avg_alignment_ms'] > 10:
        print(f"  3. Face alignment is slow ({results['yunet_npu']['avg_alignment_ms']:.1f}ms)")
        print(f"     - Optimize warp_and_crop_face implementation")


if __name__ == '__main__':
    main()
