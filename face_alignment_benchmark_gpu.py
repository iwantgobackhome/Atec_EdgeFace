#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face Alignment Methods Comprehensive Benchmark - GPU Version

This script compares multiple face alignment methods on LFW dataset using EdgeFace-XS(γ=0.6) model.
Optimized for GPU execution with MediaPipe excluded.

Methods Compared:
- MTCNN: Multi-task CNN (PyTorch)
- YuNet: OpenCV's fast face detection
- YOLO: YOLO-based face detection
- RetinaFace: ONNX-based RetinaFace (if available)
- RTMPose: Face keypoint detection (if available)

Evaluation Metrics:
- Accuracy: Face verification accuracy on LFW
- ROC AUC: Area under ROC curve
- EER: Equal Error Rate
- Speed: Processing time per image
- Success Rate: Percentage of successful alignments
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import time
import psutil
import gc

# Add face_alignment to path
sys.path.insert(0, 'face_alignment')

# GPU Setup
print("⚙️ GPU 환경 설정 중...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ CUDA GPU 감지됨: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        print("⚠️ CUDA GPU가 감지되지 않았습니다. CPU 모드로 실행됩니다.")
except Exception as e:
    device = 'cpu'
    print(f"⚠️ GPU 설정 중 오류: {e}")

# Import components
try:
    from face_alignment.unified_detector import UnifiedFaceDetector, benchmark_all_methods
    from lfw_evaluation_optimized import LFWEvaluatorOptimized as LFWEvaluator
    print("✅ All components imported successfully (OPTIMIZED version)")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required files are in place")
    sys.exit(1)

# Configuration
LFW_CONFIG = {
    'lfw_dir': '/mnt/c/Users/Admin/Downloads/lfw-deepfunneled/lfw-deepfunneled',
    'pairs_file': '/mnt/c/Users/Admin/Downloads/lfw-deepfunneled/pairs.csv',
    'edgeface_model_path': 'checkpoints/edgeface_xs_gamma_06.pt',
    'device': device,
    'max_pairs': 6000,  # Use full LFW dataset for comprehensive evaluation
    'batch_size': 64,   # Batch size for embedding extraction (GPU efficiency)
    'num_workers': 8    # Parallel workers for image loading
}

# Methods to exclude
EXCLUDED_METHODS = ['mediapipe', 'mediapipe_simple']

print("\n" + "="*60)
print("FACE ALIGNMENT BENCHMARK - GPU VERSION")
print("="*60)

print("\n📋 Configuration:")
print(f"   Device: {device.upper()}")
print(f"   LFW Directory: {LFW_CONFIG['lfw_dir']}")
print(f"   Pairs File: {LFW_CONFIG['pairs_file']}")
print(f"   EdgeFace Model: {LFW_CONFIG['edgeface_model_path']}")
print(f"   Max Pairs: {LFW_CONFIG['max_pairs']}")
print(f"   🚀 Batch Size: {LFW_CONFIG['batch_size']} (GPU efficiency)")
print(f"   ⚡ Parallel Workers: {LFW_CONFIG['num_workers']} (image loading)")

# Validate paths
print("\n🔍 경로 검증...")
for key in ['lfw_dir', 'pairs_file', 'edgeface_model_path']:
    path = LFW_CONFIG[key]
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"{status} {key}: {path}")
    if not exists:
        print(f"❌ Required file not found. Exiting.")
        sys.exit(1)

print("✅ All required files found")

# Check available methods
print("\n📋 Checking available methods...")
available_methods = UnifiedFaceDetector.list_available_methods()
dependency_status = UnifiedFaceDetector.check_dependencies()

# Filter out excluded methods
available_methods = [m for m in available_methods if m not in EXCLUDED_METHODS]

print("\n📋 Method Availability Status:")
print("=" * 40)

for method, available in dependency_status.items():
    if method in EXCLUDED_METHODS:
        status = "🚫"
        note = "Excluded (MediaPipe)"
    elif available and method in available_methods:
        status = "✅"
        note = "Available"
    else:
        status = "❌"
        note = "Not Available"

    print(f"{status} {method.upper()}: {note}")

print(f"\n🎯 Methods to evaluate: {available_methods}")

if not available_methods:
    print("❌ No methods available for evaluation. Exiting.")
    sys.exit(1)

# Initialize LFW Evaluator (OPTIMIZED)
print("\n🚀 Initializing LFW Evaluator (OPTIMIZED)...")
try:
    evaluator = LFWEvaluator(
        lfw_dir=LFW_CONFIG['lfw_dir'],
        pairs_file=LFW_CONFIG['pairs_file'],
        edgeface_model_path=LFW_CONFIG['edgeface_model_path'],
        device=LFW_CONFIG['device'],
        batch_size=LFW_CONFIG['batch_size'],
        num_workers=LFW_CONFIG['num_workers']
    )
    print("✅ LFW Evaluator initialized successfully (OPTIMIZED)")
except Exception as e:
    print(f"❌ Failed to initialize LFW Evaluator: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Run evaluation with memory tracking
print("\n🚀 Starting LFW evaluation...")
print(f"📊 Evaluating {len(available_methods)} methods on {LFW_CONFIG['max_pairs']} pairs")
print("=" * 60)

# Get initial memory state
process = psutil.Process()
initial_cpu_memory = process.memory_info().rss / 1024**2  # MB
if device == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    initial_gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB

start_time = time.time()

# Track memory per method
memory_results = {}

try:
    # Modified evaluation to track memory per method
    for method in available_methods:
        print(f"\n{'='*60}")
        print(f"Evaluating {method.upper()}...")
        print('='*60)

        # Aggressive memory cleanup before each method
        gc.collect()
        gc.collect()  # Call twice for thorough cleanup
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            method_start_gpu_mem = torch.cuda.memory_allocated() / 1024**2
        else:
            method_start_gpu_mem = 0

        method_start_cpu_mem = process.memory_info().rss / 1024**2

        # Evaluate single method
        method_results = evaluator.evaluate_all_methods([method], max_pairs=LFW_CONFIG['max_pairs'])

        # Aggressive memory cleanup after evaluation
        gc.collect()
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Track peak memory usage
        method_end_cpu_mem = process.memory_info().rss / 1024**2
        cpu_mem_used = method_end_cpu_mem - method_start_cpu_mem

        if device == 'cuda':
            gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024**2
            gpu_mem_used = gpu_mem_peak - method_start_gpu_mem
            memory_results[method] = {
                'cpu_memory_mb': cpu_mem_used,
                'gpu_memory_mb': gpu_mem_used,
                'gpu_memory_peak_mb': gpu_mem_peak
            }
        else:
            memory_results[method] = {
                'cpu_memory_mb': cpu_mem_used,
                'gpu_memory_mb': 0,
                'gpu_memory_peak_mb': 0
            }

        # Store results
        if method not in locals().get('lfw_results', {}):
            if 'lfw_results' not in locals():
                lfw_results = {}
            lfw_results.update(method_results)

        print(f"💾 Memory usage for {method.upper()}:")
        print(f"   CPU: {cpu_mem_used:.1f} MB")
        if device == 'cuda':
            print(f"   GPU: {gpu_mem_used:.1f} MB (Peak: {gpu_mem_peak:.1f} MB)")

    elapsed_time = time.time() - start_time
    print(f"\n✅ Evaluation completed in {elapsed_time:.1f} seconds")

except Exception as e:
    print(f"❌ LFW evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create comparison report
print("\n📊 Creating comparison report...")

report_data = []

for method, result in lfw_results.items():
    if 'error' in result:
        report_data.append({
            'Method': method.upper(),
            'Status': f"❌ {result['error']}",
            'Success Rate': 'N/A',
            'ROC AUC': 'N/A',
            'Accuracy': 'N/A',
            'EER': 'N/A',
            'Avg Time (s)': 'N/A',
            'CPU Memory (MB)': 'N/A',
            'GPU Memory (MB)': 'N/A'
        })
    else:
        # Safely handle potential NaN values
        success_rate = result.get('success_rate', 0)
        roc_auc = result.get('roc_auc', 0)
        accuracy = result.get('best_accuracy', 0)
        eer = result.get('eer', 1)
        processing_time = result.get('avg_processing_time', 0)

        # Check for NaN values and replace with defaults
        success_rate = success_rate if not np.isnan(success_rate) else 0
        roc_auc = roc_auc if not np.isnan(roc_auc) else 0
        accuracy = accuracy if not np.isnan(accuracy) else 0
        eer = eer if not np.isnan(eer) else 1
        processing_time = processing_time if not np.isnan(processing_time) else 0

        # Get memory usage for this method
        mem_info = memory_results.get(method, {})
        cpu_mem = mem_info.get('cpu_memory_mb', 0)
        gpu_mem = mem_info.get('gpu_memory_peak_mb', 0)

        # Only include methods with successful alignments
        if success_rate > 0:
            report_data.append({
                'Method': method.upper(),
                'Status': '✅ Success',
                'Success Rate': f"{success_rate:.1%}",
                'ROC AUC': f"{roc_auc:.4f}",
                'Accuracy': f"{accuracy:.1%}",
                'EER': f"{eer:.3f}",
                'Avg Time (s)': f"{processing_time:.3f}",
                'CPU Memory (MB)': f"{cpu_mem:.1f}",
                'GPU Memory (MB)': f"{gpu_mem:.1f}" if device == 'cuda' else 'N/A'
            })

report_df = pd.DataFrame(report_data)

print("\n📋 Face Alignment Methods Comparison Report (GPU)")
print("=" * 80)
print(report_df.to_string(index=False))

# Save report
output_dir = "benchmark_results"
os.makedirs(output_dir, exist_ok=True)

report_filename = f"{output_dir}/face_alignment_benchmark_report_gpu.csv"
report_df.to_csv(report_filename, index=False, encoding='utf-8-sig')
print(f"\n💾 Report saved to: {report_filename}")

# Visualization
print("\n📊 Creating visualizations...")

# Filter valid results (exclude methods with 0% success rate)
valid_results = {
    k: v for k, v in lfw_results.items()
    if 'error' not in v and v.get('success_rate', 0) > 0
}

if valid_results:
    methods = list(valid_results.keys())

    # Extract metrics with NaN handling
    accuracies = []
    roc_aucs = []
    success_rates = []
    processing_times = []
    eers = []
    cpu_memories = []
    gpu_memories = []

    for method in methods:
        result = valid_results[method]

        accuracy = result.get('best_accuracy', 0)
        roc_auc = result.get('roc_auc', 0)
        success_rate = result.get('success_rate', 0)
        processing_time = result.get('avg_processing_time', 0)
        eer = result.get('eer', 1)

        accuracies.append(accuracy if not np.isnan(accuracy) else 0)
        roc_aucs.append(roc_auc if not np.isnan(roc_auc) else 0)
        success_rates.append(success_rate if not np.isnan(success_rate) else 0)
        processing_times.append(processing_time if not np.isnan(processing_time) else 0)
        eers.append(eer if not np.isnan(eer) else 1)

        # Get memory info
        mem_info = memory_results.get(method, {})
        cpu_memories.append(mem_info.get('cpu_memory_mb', 0))
        gpu_memories.append(mem_info.get('gpu_memory_peak_mb', 0))

    # Create comprehensive dashboard with 6 subplots
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    fig.suptitle(f'Face Alignment Methods Benchmark ({device.upper()})',
                 fontsize=20, fontweight='bold')

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    # 1. Accuracy Comparison
    bars1 = ax1.bar([m.upper() for m in methods], [acc*100 for acc in accuracies],
                    alpha=0.8, color='lightcoral')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Face Verification Accuracy Comparison')
    ax1.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')

    # 2. ROC AUC vs Processing Time
    scatter = ax2.scatter(processing_times, roc_aucs, s=150, alpha=0.7,
                         c=range(len(methods)), cmap='viridis')
    ax2.set_xlabel('Processing Time (seconds)')
    ax2.set_ylabel('ROC AUC')
    ax2.set_title('Accuracy vs Speed Trade-off')
    ax2.grid(True, alpha=0.3)

    for i, method in enumerate(methods):
        ax2.annotate(method.upper(), (processing_times[i], roc_aucs[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    # 3. Success Rate Comparison
    bars3 = ax3.bar([m.upper() for m in methods], [sr*100 for sr in success_rates],
                    alpha=0.8, color='lightgreen')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Face Detection Success Rate')
    ax3.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)

    for bar, sr in zip(bars3, success_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{sr:.1%}', ha='center', va='bottom', fontweight='bold')

    # 4. Equal Error Rate
    bars4 = ax4.bar([m.upper() for m in methods], [eer*100 for eer in eers],
                    alpha=0.8, color='lightsalmon')
    ax4.set_ylabel('Equal Error Rate (%)')
    ax4.set_title('Equal Error Rate (Lower is Better)')
    ax4.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)

    for bar, eer in zip(bars4, eers):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{eer:.1%}', ha='center', va='bottom', fontweight='bold')

    # 5. CPU Memory Usage
    bars5 = ax5.bar([m.upper() for m in methods], cpu_memories,
                    alpha=0.8, color='skyblue')
    ax5.set_ylabel('CPU Memory (MB)')
    ax5.set_title('CPU Memory Usage')
    ax5.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)

    for bar, mem in zip(bars5, cpu_memories):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{mem:.0f}', ha='center', va='bottom', fontweight='bold')

    # 6. GPU Memory Usage
    if device == 'cuda':
        bars6 = ax6.bar([m.upper() for m in methods], gpu_memories,
                        alpha=0.8, color='mediumpurple')
        ax6.set_ylabel('GPU Memory (MB)')
        ax6.set_title('GPU Memory Usage (Peak)')
        ax6.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)

        for bar, mem in zip(bars6, gpu_memories):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{mem:.0f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'GPU not available', ha='center', va='center',
                transform=ax6.transAxes, fontsize=16)
        ax6.set_title('GPU Memory Usage')

    plt.tight_layout()

    plot_filename = f"{output_dir}/face_alignment_benchmark_gpu.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {plot_filename}")

    # Rankings
    print("\n🏆 METHOD RANKINGS")
    print("=" * 60)

    if len(methods) > 0:
        # Accuracy ranking
        if any(v > 0 for v in accuracies):
            print("\n📊 Accuracy Ranking:")
            print("-" * 30)
            ranking = sorted(zip(methods, accuracies), key=lambda x: x[1], reverse=True)
            for i, (method, score) in enumerate(ranking[:3], 1):
                print(f"{i}. {method.upper()}: {score:.1%}")

        # Speed ranking
        if any(v > 0 for v in processing_times):
            print("\n⚡ Speed Ranking (Fastest):")
            print("-" * 30)
            ranking = sorted(zip(methods, processing_times), key=lambda x: x[1])
            for i, (method, score) in enumerate(ranking[:3], 1):
                print(f"{i}. {method.upper()}: {score:.3f}s")

        # ROC AUC ranking
        if any(v > 0 for v in roc_aucs):
            print("\n📈 ROC AUC Ranking:")
            print("-" * 30)
            ranking = sorted(zip(methods, roc_aucs), key=lambda x: x[1], reverse=True)
            for i, (method, score) in enumerate(ranking[:3], 1):
                print(f"{i}. {method.upper()}: {score:.4f}")

        # Calculate balanced score
        print("\n⭐ BALANCED RECOMMENDATION:")
        print("-" * 30)

        if len(accuracies) > 1 and max(accuracies) > min(accuracies):
            balanced_scores = {}

            acc_range = max(accuracies) - min(accuracies)
            time_range = max(processing_times) - min(processing_times)
            success_range = max(success_rates) - min(success_rates)
            eer_range = max(eers) - min(eers)

            for method in methods:
                idx = methods.index(method)

                norm_acc = (accuracies[idx] - min(accuracies)) / acc_range if acc_range > 0 else 1.0
                norm_speed = 1 - ((processing_times[idx] - min(processing_times)) / time_range) if time_range > 0 else 1.0
                norm_success = (success_rates[idx] - min(success_rates)) / success_range if success_range > 0 else 1.0
                norm_eer = 1 - ((eers[idx] - min(eers)) / eer_range) if eer_range > 0 else 1.0

                balanced_score = (norm_acc * 0.4 + norm_speed * 0.25 + norm_success * 0.2 + norm_eer * 0.15)
                balanced_scores[method] = balanced_score

            if balanced_scores:
                best_balanced = max(balanced_scores.items(), key=lambda x: x[1])
                print(f"🎖️ BEST BALANCED CHOICE: {best_balanced[0].upper()}")
                print(f"   Score: {best_balanced[1]:.3f}/1.000")
                print(f"   Optimal balance of accuracy, speed, and reliability")

else:
    print("⚠️ No valid results to visualize")

print("\n" + "="*60)
print("✨ EVALUATION COMPLETED!")
print("="*60)
print(f"\n📁 Output files:")
print(f"   📄 {report_filename}")
if valid_results:
    print(f"   📊 {plot_filename}")
print(f"\n🚀 Device used: {device.upper()}")
print(f"🎯 Methods evaluated: {len(valid_results)}/{len(available_methods)}")
print(f"⏱️ Total time: {elapsed_time:.1f}s")
