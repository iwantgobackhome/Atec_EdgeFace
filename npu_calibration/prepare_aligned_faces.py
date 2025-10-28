"""
Prepare Aligned Face Dataset for EdgeFace NPU Calibration

EdgeFace (recognition model)는 detection + alignment를 거친 얼굴 이미지를 입력받습니다.
따라서 calibration 데이터도 정렬된 얼굴 이미지여야 합니다.

이 스크립트는:
1. LFW 등의 데이터셋에서 이미지를 읽음
2. YuNet으로 얼굴 detection + alignment 수행
3. 정렬된 얼굴 이미지(112x112)를 calibration 데이터셋으로 저장
"""

import os
import sys
import cv2
import numpy as np
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import json

# Add face_alignment to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'face_alignment'))

try:
    from yunet import YuNetDetector
except ImportError:
    print("Error: Cannot import YuNetDetector")
    print("Make sure face_alignment/yunet.py exists")
    sys.exit(1)


def extract_aligned_face(image_path, detector, target_size=(112, 112)):
    """
    Extract aligned face from image using YuNet

    Args:
        image_path: Path to source image
        detector: YuNetDetector instance
        target_size: Output face size (width, height)

    Returns:
        aligned_face: PIL Image of aligned face (112x112) or None
        quality_score: Quality score of the detection
    """
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')

        # Detect and align face
        aligned_face = detector.align(img, return_landmarks=False)

        if aligned_face is None:
            return None, 0.0

        # Calculate quality score based on alignment confidence
        # YuNet detector stores detection confidence in the face data
        faces = detector.detect_faces(img)
        if len(faces) == 0:
            return None, 0.0

        # Use highest confidence face
        quality_score = max(face[-1] for face in faces) * 100  # Convert to 0-100 scale

        # Ensure correct size
        if aligned_face.size != target_size:
            aligned_face = aligned_face.resize(target_size, Image.LANCZOS)

        return aligned_face, quality_score

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, 0.0


def prepare_aligned_calibration_dataset(
    source_dir,
    output_dir,
    yunet_model_path,
    num_samples=100,
    target_size=(112, 112),
    quality_threshold=80,  # Higher threshold for face detection confidence
    device='cpu',
    save_analysis=True
):
    """
    Prepare aligned face calibration dataset for EdgeFace

    Args:
        source_dir: Source image directory (e.g., LFW dataset)
        output_dir: Output directory for aligned faces
        yunet_model_path: Path to YuNet ONNX model
        num_samples: Number of calibration samples
        target_size: Output face size (width, height)
        quality_threshold: Minimum detection confidence (0-100)
        device: Device for YuNet ('cpu' or 'cuda')
        save_analysis: Save quality analysis report

    Returns:
        stats: Preparation statistics
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("EdgeFace Aligned Face Calibration Dataset Preparation")
    print("="*60)

    # Initialize YuNet detector
    print(f"\nInitializing YuNet detector...")
    print(f"Model: {yunet_model_path}")
    print(f"Device: {device}")

    try:
        detector = YuNetDetector(
            model_path=yunet_model_path,
            device=device,
            crop_size=target_size
        )
    except Exception as e:
        print(f"Error: Failed to initialize YuNet detector: {e}")
        print("\nMake sure:")
        print(f"  1. YuNet model exists at: {yunet_model_path}")
        print(f"  2. face_alignment/yunet.py is available")
        sys.exit(1)

    # Find all source images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
    all_images = []
    for ext in extensions:
        all_images.extend(source_path.rglob(ext))

    print(f"\nFound {len(all_images)} total images in {source_dir}")

    if len(all_images) == 0:
        raise ValueError(f"No images found in {source_dir}")

    # Shuffle for diversity
    random.shuffle(all_images)

    # Extract aligned faces
    print(f"\nExtracting aligned faces...")
    print(f"Target: {num_samples} samples")
    print(f"Quality threshold: {quality_threshold}")

    aligned_faces = []
    failed_count = 0

    # Process images until we have enough good samples
    # Search through more images than needed (up to 10x) to ensure quality
    search_limit = min(len(all_images), num_samples * 10)

    for img_path in tqdm(all_images[:search_limit], desc="Processing images"):
        if len(aligned_faces) >= num_samples:
            break

        aligned_face, quality_score = extract_aligned_face(
            img_path,
            detector,
            target_size=target_size
        )

        if aligned_face is None:
            failed_count += 1
            continue

        if quality_score < quality_threshold:
            failed_count += 1
            continue

        aligned_faces.append({
            'image': aligned_face,
            'source': str(img_path),
            'quality': quality_score
        })

    print(f"\nExtraction complete:")
    print(f"  Successful: {len(aligned_faces)}")
    print(f"  Failed: {failed_count}")

    if len(aligned_faces) == 0:
        print(f"\nError: No faces detected with quality >= {quality_threshold}")
        print("Try:")
        print("  1. Lower --quality-threshold (e.g., 70)")
        print("  2. Use a larger source dataset")
        print("  3. Check if YuNet model is working correctly")
        sys.exit(1)

    if len(aligned_faces) < num_samples:
        print(f"\nWarning: Only found {len(aligned_faces)} faces (requested {num_samples})")
        print("Proceeding with available samples...")

    # Sort by quality and save
    aligned_faces.sort(key=lambda x: x['quality'], reverse=True)

    print(f"\nSaving aligned faces to: {output_dir}")

    analysis_data = []
    for idx, face_data in enumerate(tqdm(aligned_faces, desc="Saving faces")):
        # Save aligned face
        dst_filename = f"aligned_{idx:04d}.jpg"
        dst_path = output_path / dst_filename

        face_data['image'].save(dst_path, quality=95)

        # Record analysis
        analysis_data.append({
            'index': idx,
            'filename': dst_filename,
            'source': face_data['source'],
            'detection_confidence': float(face_data['quality'])
        })

    # Calculate statistics
    qualities = [f['quality'] for f in aligned_faces]
    stats = {
        'source_dir': str(source_dir),
        'output_dir': str(output_dir),
        'yunet_model': yunet_model_path,
        'total_processed': search_limit,
        'successful_detections': len(aligned_faces),
        'failed_detections': failed_count,
        'quality_range': (float(min(qualities)), float(max(qualities))),
        'avg_quality': float(np.mean(qualities)),
        'target_size': target_size
    }

    # Save analysis report
    if save_analysis:
        report_path = output_path / 'aligned_faces_analysis.json'
        with open(report_path, 'w') as f:
            json.dump({
                'statistics': stats,
                'images': analysis_data,
                'config': {
                    'num_samples': num_samples,
                    'target_size': list(target_size),
                    'quality_threshold': quality_threshold,
                    'device': device
                }
            }, f, indent=2)
        print(f"\nAnalysis report saved to: {report_path}")

    # Create README
    readme_path = output_path / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(f"""# Aligned Face Calibration Dataset for EdgeFace NPU

## Dataset Information
- **Total Aligned Faces**: {len(aligned_faces)}
- **Source**: {source_dir}
- **Face Size**: {target_size[0]}x{target_size[1]}
- **Detection Quality Range**: {stats['quality_range'][0]:.1f} - {stats['quality_range'][1]:.1f}
- **Average Detection Confidence**: {stats['avg_quality']:.1f}

## Processing Pipeline
1. Face detection using YuNet
2. Facial landmark detection (5 points)
3. Face alignment to {target_size[0]}x{target_size[1]}
4. Quality filtering (confidence >= {quality_threshold})

## Statistics
- Total images processed: {search_limit}
- Successful detections: {len(aligned_faces)}
- Failed detections: {failed_count}
- Success rate: {len(aligned_faces) / search_limit * 100:.1f}%

## Usage
These aligned face images are ready for EdgeFace NPU calibration.
Use with `calibration_config_edgeface.json`.

## Notes
- All faces are aligned using YuNet's 5-point landmark detection
- Images are sorted by detection confidence (highest first)
- Only high-confidence detections are included (>= {quality_threshold})
- Suitable for face recognition model calibration
""")

    print(f"\n{'='*60}")
    print(f"Preparation Complete!")
    print(f"{'='*60}")
    print(f"Aligned faces saved: {len(aligned_faces)}")
    print(f"Quality range: {stats['quality_range'][0]:.1f} - {stats['quality_range'][1]:.1f}")
    print(f"Average confidence: {stats['avg_quality']:.1f}")
    print(f"Output directory: {output_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Prepare aligned face dataset for EdgeFace NPU calibration'
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        required=True,
        help='Source directory with face images (e.g., LFW dataset)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./aligned_faces_calibration',
        help='Output directory for aligned faces'
    )
    parser.add_argument(
        '--yunet-model',
        type=str,
        default='./face_detection_yunet_2023mar.onnx',
        help='Path to YuNet ONNX model'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of aligned faces to extract (default: 100)'
    )
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=80,
        help='Minimum detection confidence 0-100 (default: 80)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for YuNet inference (default: cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Check if YuNet model exists
    if not os.path.exists(args.yunet_model):
        print(f"Error: YuNet model not found at: {args.yunet_model}")
        print("\nPlease download YuNet model:")
        print("  wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
        sys.exit(1)

    stats = prepare_aligned_calibration_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        yunet_model_path=args.yunet_model,
        num_samples=args.num_samples,
        target_size=(112, 112),  # EdgeFace standard size
        quality_threshold=args.quality_threshold,
        device=args.device,
        save_analysis=True
    )

    print("\nNext steps:")
    print(f"1. Review aligned faces in: {args.output_dir}")
    print("2. Generate EdgeFace calibration config:")
    print(f"   python generate_calibration_config.py \\")
    print(f"     --model-type edgeface \\")
    print(f"     --dataset-path {args.output_dir}")
    print("3. Use the config for EdgeFace NPU compilation")


if __name__ == '__main__':
    main()
