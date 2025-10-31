#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI-specific Performance Profiling
GUI 실제 성능 병목 측정
"""

import cv2
import numpy as np
import time
import sys
import tkinter as tk
from PIL import Image, ImageTk

sys.path.insert(0, 'face_alignment')

def profile_gui_pipeline(detector_name, model_path, use_npu=False, root=None):
    """Profile actual GUI pipeline with all overhead"""
    print(f"\n{'='*60}")
    print(f"Profiling GUI Pipeline: {detector_name} (NPU: {use_npu})")
    print(f"{'='*60}")

    from face_recognition_system import FaceRecognitionSystem

    # Create Tkinter root if not provided
    if root is None:
        root = tk.Tk()
        root.withdraw()  # Hide the window

    system = FaceRecognitionSystem(
        detector_method=detector_name,
        edgeface_model_path=model_path,
        device='npu' if use_npu else 'cpu',
        use_npu=use_npu
    )

    # Capture test frames
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
    for _ in range(3):
        _ = system.process_frame(frames[0])

    # Profile with GUI overhead simulation
    print("\nProfiling with GUI overhead...")

    times = {
        'process_frame': [],
        'cvtColor': [],
        'resize': [],
        'Image_fromarray': [],
        'ImageTk_PhotoImage': [],
        'frame_copy': [],
        'total_gui': []
    }

    for i, frame in enumerate(frames):
        start_total = time.perf_counter()

        # 1. Frame copy (line 408)
        start = time.perf_counter()
        current_frame = frame.copy()
        times['frame_copy'].append((time.perf_counter() - start) * 1000)

        # 2. Process frame (line 436)
        start = time.perf_counter()
        annotated_frame, detections = system.process_frame(frame)
        times['process_frame'].append((time.perf_counter() - start) * 1000)

        # 3. cvtColor BGR to RGB (line 450)
        start = time.perf_counter()
        display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        times['cvtColor'].append((time.perf_counter() - start) * 1000)

        # 4. Resize (line 452-458)
        h, w = display_frame.shape[:2]
        max_w, max_h = 800, 600
        if w > max_w or h > max_h:
            start = time.perf_counter()
            scale = min(max_w/w, max_h/h)
            new_w, new_h = int(w*scale), int(h*scale)
            display_frame = cv2.resize(display_frame, (new_w, new_h))
            times['resize'].append((time.perf_counter() - start) * 1000)
        else:
            times['resize'].append(0)

        # 5. Image.fromarray (line 461)
        start = time.perf_counter()
        img = Image.fromarray(display_frame)
        times['Image_fromarray'].append((time.perf_counter() - start) * 1000)

        # 6. ImageTk.PhotoImage (line 462) - This is SLOW!
        start = time.perf_counter()
        imgtk = ImageTk.PhotoImage(image=img)
        times['ImageTk_PhotoImage'].append((time.perf_counter() - start) * 1000)

        # Keep reference (line 465)
        _ = imgtk

        times['total_gui'].append((time.perf_counter() - start_total) * 1000)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(frames)} frames")

    # Calculate statistics
    print(f"\n📊 GUI Pipeline Performance Breakdown:")
    print(f"{'Stage':<25s} {'Avg (ms)':>10s} {'Std (ms)':>10s} {'%':>8s}")
    print("-" * 60)

    total_avg = np.mean(times['total_gui'])

    for stage, stage_times in times.items():
        if stage == 'total_gui':
            continue
        avg = np.mean(stage_times)
        std = np.std(stage_times)
        percentage = (avg / total_avg) * 100
        print(f"{stage:<25s} {avg:10.2f} {std:10.2f} {percentage:7.1f}%")

    print("-" * 60)
    print(f"{'TOTAL (GUI)':<25s} {total_avg:10.2f} {np.std(times['total_gui']):10.2f} {'100.0':>7s}%")

    fps = 1000.0 / total_avg
    print(f"\n  Actual GUI FPS: {fps:.1f}")

    # Compare with raw processing
    raw_processing_avg = np.mean(times['process_frame'])
    raw_fps = 1000.0 / raw_processing_avg
    overhead = total_avg - raw_processing_avg
    overhead_pct = (overhead / total_avg) * 100

    print(f"\n📈 Overhead Analysis:")
    print(f"  Raw process_frame: {raw_processing_avg:.2f} ms ({raw_fps:.1f} FPS)")
    print(f"  GUI overhead:      {overhead:.2f} ms ({overhead_pct:.1f}%)")
    print(f"  Total with GUI:    {total_avg:.2f} ms ({fps:.1f} FPS)")

    # Identify worst offenders
    gui_overhead_stages = {
        'ImageTk.PhotoImage': np.mean(times['ImageTk_PhotoImage']),
        'cvtColor': np.mean(times['cvtColor']),
        'resize': np.mean(times['resize']),
        'Image.fromarray': np.mean(times['Image_fromarray']),
        'frame_copy': np.mean(times['frame_copy'])
    }

    print(f"\n⚠️  GUI Overhead Breakdown:")
    for stage, time_ms in sorted(gui_overhead_stages.items(), key=lambda x: x[1], reverse=True):
        pct = (time_ms / overhead) * 100 if overhead > 0 else 0
        print(f"  {stage:<20s}: {time_ms:6.2f} ms ({pct:5.1f}% of overhead)")

    return {
        'detector': detector_name,
        'use_npu': use_npu,
        'total_avg_ms': total_avg,
        'raw_processing_ms': raw_processing_avg,
        'gui_overhead_ms': overhead,
        'gui_fps': fps,
        'raw_fps': raw_fps,
        'times': times
    }


def compare_gui_vs_raw():
    """Compare GUI performance vs raw processing"""
    print("\n" + "="*60)
    print("GUI vs Raw Processing Performance Comparison")
    print("="*60)

    # Create Tkinter root for both tests
    root = tk.Tk()
    root.withdraw()  # Hide the window

    # Test CPU
    print("\n\n### CPU Configuration ###")
    cpu_result = profile_gui_pipeline('yunet', 'checkpoints/edgeface_xs_gamma_06.pt', use_npu=False, root=root)

    # Test NPU
    print("\n\n### NPU Configuration ###")
    npu_result = profile_gui_pipeline('yunet_npu', 'checkpoints/edgeface_xs_gamma_06.dxnn', use_npu=True, root=root)

    # Summary
    print("\n\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)

    print(f"\nCPU:")
    print(f"  Raw processing:  {cpu_result['raw_processing_ms']:.2f} ms ({cpu_result['raw_fps']:.1f} FPS)")
    print(f"  GUI overhead:    {cpu_result['gui_overhead_ms']:.2f} ms")
    print(f"  Total GUI:       {cpu_result['total_avg_ms']:.2f} ms ({cpu_result['gui_fps']:.1f} FPS)")

    print(f"\nNPU:")
    print(f"  Raw processing:  {npu_result['raw_processing_ms']:.2f} ms ({npu_result['raw_fps']:.1f} FPS)")
    print(f"  GUI overhead:    {npu_result['gui_overhead_ms']:.2f} ms")
    print(f"  Total GUI:       {npu_result['total_avg_ms']:.2f} ms ({npu_result['gui_fps']:.1f} FPS)")

    print(f"\nSpeedup:")
    raw_speedup = cpu_result['raw_processing_ms'] / npu_result['raw_processing_ms']
    gui_speedup = cpu_result['total_avg_ms'] / npu_result['total_avg_ms']
    print(f"  Raw processing:  {raw_speedup:.2f}x")
    print(f"  With GUI:        {gui_speedup:.2f}x")

    if gui_speedup < raw_speedup:
        efficiency_loss = (1 - gui_speedup / raw_speedup) * 100
        print(f"\n⚠️  GUI overhead reduces NPU advantage by {efficiency_loss:.1f}%")
        print(f"   NPU is {raw_speedup:.2f}x faster, but GUI limits it to {gui_speedup:.2f}x")

    # Optimization recommendations
    print(f"\n💡 Optimization Recommendations:")

    # Check if ImageTk is the biggest overhead
    imagetk_time = np.mean(npu_result['times']['ImageTk_PhotoImage'])
    if imagetk_time > 5:
        print(f"  1. ImageTk.PhotoImage is slow ({imagetk_time:.1f}ms)")
        print(f"     - Consider using OpenCV's cv2.imshow instead")
        print(f"     - Or use faster image conversion methods")

    cvt_time = np.mean(npu_result['times']['cvtColor'])
    if cvt_time > 2:
        print(f"  2. cvtColor is slow ({cvt_time:.1f}ms)")
        print(f"     - Consider in-place color conversion")

    resize_time = np.mean(npu_result['times']['resize'])
    if resize_time > 2:
        print(f"  3. Resize is slow ({resize_time:.1f}ms)")
        print(f"     - Use faster interpolation (INTER_NEAREST)")

    if npu_result['gui_overhead_ms'] > npu_result['raw_processing_ms']:
        print(f"  4. GUI overhead ({npu_result['gui_overhead_ms']:.1f}ms) > processing ({npu_result['raw_processing_ms']:.1f}ms)")
        print(f"     - Move image conversion to separate thread")
        print(f"     - Use double buffering for display")


if __name__ == '__main__':
    compare_gui_vs_raw()
