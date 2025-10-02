#!/usr/bin/env python3
"""
Face Alignment Demo Script
Demonstrates face detection and alignment using multiple methods with visualization
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# Add face_alignment to path
sys.path.insert(0, 'face_alignment')

try:
    from face_alignment.unified_detector import UnifiedFaceDetector
    print("✅ Successfully imported UnifiedFaceDetector")
except ImportError as e:
    print(f"❌ Failed to import UnifiedFaceDetector: {e}")
    sys.exit(1)

def demo_face_alignment(image_path, output_dir="alignment_results"):
    """
    Demonstrate face alignment with available methods
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results
    """
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available methods
    available_methods = UnifiedFaceDetector.list_available_methods()
    print(f"📝 Available methods: {available_methods}")
    
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    print(f"📸 Original image size: {original_image.size}")
    
    # Create subplot layout
    n_methods = len(available_methods)
    cols = min(3, n_methods + 1)  # +1 for original
    rows = (n_methods + 1 + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Show original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Test each method
    for i, method in enumerate(available_methods):
        print(f"\n🔍 Testing {method.upper()}...")
        
        try:
            # Create detector
            detector = UnifiedFaceDetector(method, device='cpu')
            
            if not detector.available:
                print(f"❌ {method} not available")
                axes[i+1].text(0.5, 0.5, f"{method.upper()}\nNot Available", 
                              ha='center', va='center', fontsize=12)
                axes[i+1].set_xlim(0, 1)
                axes[i+1].set_ylim(0, 1)
                axes[i+1].axis('off')
                continue
            
            # Detect and align face
            aligned_face = detector.align(original_image)
            
            if aligned_face is not None:
                # Show aligned face
                axes[i+1].imshow(aligned_face)
                axes[i+1].set_title(f'{method.upper()}\n✅ Success', 
                                   fontsize=12, fontweight='bold', color='green')
                
                # Save individual result
                aligned_face.save(os.path.join(output_dir, f'{method}_aligned.jpg'))
                print(f"✅ {method} alignment successful")
            else:
                # Show failure
                axes[i+1].text(0.5, 0.5, f"{method.upper()}\n❌ Failed", 
                              ha='center', va='center', fontsize=12, color='red')
                axes[i+1].set_xlim(0, 1)
                axes[i+1].set_ylim(0, 1)
                print(f"❌ {method} alignment failed")
            
            axes[i+1].axis('off')
            
        except Exception as e:
            print(f"❌ Error with {method}: {e}")
            axes[i+1].text(0.5, 0.5, f"{method.upper()}\n❌ Error", 
                          ha='center', va='center', fontsize=12, color='red')
            axes[i+1].set_xlim(0, 1)
            axes[i+1].set_ylim(0, 1)
            axes[i+1].axis('off')
    
    # Hide unused subplots
    for j in range(n_methods + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'face_alignment_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Comparison saved to: {comparison_path}")
    plt.show()


def demo_multi_face_detection(image_path, output_dir="alignment_results"):
    """
    Demonstrate multi-face detection with bounding boxes
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results
    """
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available methods
    available_methods = UnifiedFaceDetector.list_available_methods()
    
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    
    # Create subplot for multi-face detection
    n_methods = len(available_methods)
    cols = min(2, n_methods)
    rows = (n_methods + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 8*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, method in enumerate(available_methods):
        print(f"\n🔍 Multi-face detection with {method.upper()}...")
        
        try:
            detector = UnifiedFaceDetector(method, device='cpu')
            
            if not detector.available:
                print(f"❌ {method} not available")
                continue
            
            # Detect multiple faces
            bboxes, aligned_faces = detector.align_multi(original_image, limit=5)
            
            # Show original image with bounding boxes
            axes[i].imshow(original_image)
            axes[i].set_title(f'{method.upper()}\nDetected {len(bboxes)} faces', 
                             fontsize=12, fontweight='bold')
            
            # Draw bounding boxes
            for j, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                rect = patches.Rectangle((x1, y1), width, height, 
                                       linewidth=2, edgecolor='red', 
                                       facecolor='none', alpha=0.8)
                axes[i].add_patch(rect)
                
                # Add face number
                axes[i].text(x1, y1-5, f'Face {j+1}', 
                           fontsize=10, color='red', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            
            axes[i].axis('off')
            
            print(f"✅ {method} detected {len(bboxes)} faces")
            
        except Exception as e:
            print(f"❌ Error with {method}: {e}")
    
    # Hide unused subplots
    for j in range(len(available_methods), len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    detection_path = os.path.join(output_dir, 'multi_face_detection.png')
    plt.savefig(detection_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Multi-face detection saved to: {detection_path}")
    plt.show()


def find_sample_images():
    """Find sample images from LFW dataset"""
    lfw_dir = "/mnt/c/Users/Admin/Downloads/lfw-deepfunneled/lfw-deepfunneled"
    sample_images = []
    
    if os.path.exists(lfw_dir):
        # Find some sample images
        for root, dirs, files in os.walk(lfw_dir):
            for file in files[:3]:  # Get first 3 images per directory
                if file.endswith(('.jpg', '.png')):
                    sample_images.append(os.path.join(root, file))
                    if len(sample_images) >= 5:  # Get 5 samples total
                        break
            if len(sample_images) >= 5:
                break
    
    return sample_images


if __name__ == "__main__":
    print("🚀 Face Alignment Demo")
    print("=" * 50)
    
    # Find sample images
    sample_images = find_sample_images()
    
    if sample_images:
        print(f"📸 Found {len(sample_images)} sample images")
        
        # Test with first sample image
        test_image = sample_images[0]
        print(f"🎯 Testing with: {os.path.basename(test_image)}")
        
        # Run single face alignment demo
        print("\n📋 Single Face Alignment Demo:")
        demo_face_alignment(test_image)
        
        # Run multi-face detection demo
        print("\n📋 Multi-Face Detection Demo:")
        demo_multi_face_detection(test_image)
        
        print("\n✅ Demo completed! Check the 'alignment_results' directory for outputs.")
        
    else:
        print("⚠️ No sample images found. Please provide an image path:")
        print("Example usage: python demo_face_alignment.py /path/to/your/image.jpg")
        
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
            if os.path.exists(image_path):
                demo_face_alignment(image_path)
                demo_multi_face_detection(image_path)
            else:
                print(f"❌ Image not found: {image_path}")