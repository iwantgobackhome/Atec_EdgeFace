"""
NPU Calibration Dataset Preparation Script
Prepares high-quality calibration images from LFW dataset for YuNet ONNX->NPU compilation

The calibration dataset is critical for quantization quality:
- Diverse faces: various ages, genders, ethnicities, lighting conditions
- High quality: sharp images with clear facial features
- Representative: covers real-world deployment scenarios
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


def analyze_image_quality(image_path):
    """
    Analyze image quality metrics to select best calibration samples

    Args:
        image_path: Path to image file

    Returns:
        quality_score: Higher is better (0-100)
        metrics: Dict with detailed metrics
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return 0, {}

        # Convert to grayscale for quality analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0 * 100, 100)

        # 2. Brightness (avoid too dark/bright images)
        mean_brightness = np.mean(gray)
        brightness_score = 100 - abs(mean_brightness - 128) / 128 * 100

        # 3. Contrast (standard deviation)
        contrast = np.std(gray)
        contrast_score = min(contrast / 64.0 * 100, 100)

        # 4. Resolution score
        h, w = img.shape[:2]
        resolution_score = min((h * w) / (250 * 250) * 100, 100)

        # 5. Color distribution (check if not too monochrome)
        color_std = np.mean([np.std(img[:, :, i]) for i in range(3)])
        color_score = min(color_std / 50.0 * 100, 100)

        # Weighted overall quality score
        quality_score = (
            sharpness_score * 0.35 +
            brightness_score * 0.20 +
            contrast_score * 0.20 +
            resolution_score * 0.15 +
            color_score * 0.10
        )

        metrics = {
            'sharpness': sharpness_score,
            'brightness': brightness_score,
            'contrast': contrast_score,
            'resolution': resolution_score,
            'color': color_score,
            'overall': quality_score,
            'image_size': (w, h)
        }

        return quality_score, metrics

    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return 0, {}


def select_diverse_samples(image_paths, num_samples=100, quality_threshold=40):
    """
    Select diverse, high-quality images for calibration

    Strategy:
    1. Filter out low-quality images
    2. Select images with diverse visual characteristics
    3. Ensure good distribution across dataset

    Args:
        image_paths: List of candidate image paths
        num_samples: Number of samples to select
        quality_threshold: Minimum quality score (0-100)

    Returns:
        selected_paths: List of selected image paths
        selection_info: Dict with selection statistics
    """
    print(f"Analyzing {len(image_paths)} candidate images...")

    # Analyze all images
    image_qualities = []
    for img_path in tqdm(image_paths, desc="Quality analysis"):
        score, metrics = analyze_image_quality(img_path)
        if score >= quality_threshold:
            image_qualities.append({
                'path': img_path,
                'score': score,
                'metrics': metrics
            })

    print(f"Found {len(image_qualities)} images above quality threshold ({quality_threshold})")

    if len(image_qualities) == 0:
        print(f"Warning: No images meet quality threshold. Lowering to 20...")
        return select_diverse_samples(image_paths, num_samples, quality_threshold=20)

    # Sort by quality
    image_qualities.sort(key=lambda x: x['score'], reverse=True)

    # Select top samples with diversity
    selected = []
    selected_indices = set()

    # Strategy 1: Take top quality images first (50% of samples)
    top_n = min(num_samples // 2, len(image_qualities))
    for i in range(top_n):
        selected.append(image_qualities[i])
        selected_indices.add(i)

    # Strategy 2: Sample uniformly across quality distribution (remaining samples)
    remaining = num_samples - len(selected)
    if remaining > 0 and len(image_qualities) > len(selected):
        step = len(image_qualities) / remaining
        for i in range(remaining):
            idx = int(i * step + len(selected))
            if idx < len(image_qualities) and idx not in selected_indices:
                selected.append(image_qualities[idx])
                selected_indices.add(idx)

    # If still need more samples, take any remaining high-quality ones
    if len(selected) < num_samples:
        for i, img_info in enumerate(image_qualities):
            if i not in selected_indices:
                selected.append(img_info)
                selected_indices.add(i)
                if len(selected) >= num_samples:
                    break

    selected_paths = [item['path'] for item in selected]

    selection_info = {
        'total_candidates': len(image_paths),
        'quality_filtered': len(image_qualities),
        'selected': len(selected_paths),
        'quality_range': (
            float(min(item['score'] for item in selected)),
            float(max(item['score'] for item in selected))
        ),
        'avg_quality': float(np.mean([item['score'] for item in selected]))
    }

    return selected_paths, selection_info


def prepare_calibration_dataset(
    source_dir,
    output_dir,
    num_samples=100,
    target_size=(320, 320),
    quality_threshold=40,
    save_analysis=True
):
    """
    Prepare calibration dataset from source images

    Args:
        source_dir: Directory containing source images (e.g., LFW dataset)
        output_dir: Output directory for calibration images
        num_samples: Number of calibration samples
        target_size: Target image size for preprocessing check
        quality_threshold: Minimum image quality score
        save_analysis: Save quality analysis report

    Returns:
        stats: Dictionary with preparation statistics
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
    all_images = []
    for ext in extensions:
        all_images.extend(source_path.rglob(ext))

    print(f"Found {len(all_images)} total images in {source_dir}")

    if len(all_images) == 0:
        raise ValueError(f"No images found in {source_dir}")

    # Randomly shuffle to ensure diversity across subdirectories
    random.shuffle(all_images)

    # Limit search space for efficiency (sample more than needed for quality filtering)
    search_pool = min(len(all_images), num_samples * 10)
    candidate_images = all_images[:search_pool]

    print(f"Searching in pool of {search_pool} images...")

    # Select diverse, high-quality samples
    selected_paths, selection_info = select_diverse_samples(
        candidate_images,
        num_samples=num_samples,
        quality_threshold=quality_threshold
    )

    print(f"\nSelected {len(selected_paths)} calibration images")
    print(f"Quality range: {selection_info['quality_range'][0]:.1f} - {selection_info['quality_range'][1]:.1f}")
    print(f"Average quality: {selection_info['avg_quality']:.1f}")

    # Copy selected images to output directory
    analysis_data = []
    for idx, src_path in enumerate(tqdm(selected_paths, desc="Copying images")):
        # Use zero-padded numbering for proper sorting
        dst_filename = f"calib_{idx:04d}{src_path.suffix}"
        dst_path = output_path / dst_filename

        # Copy image
        img = Image.open(src_path).convert('RGB')
        img.save(dst_path, quality=95)

        # Record analysis data
        score, metrics = analyze_image_quality(src_path)
        # Convert numpy types to Python types for JSON serialization
        metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in metrics.items()}
        analysis_data.append({
            'index': idx,
            'filename': dst_filename,
            'source': str(src_path),
            'quality_score': float(score),
            'metrics': metrics_json
        })

    # Save analysis report
    if save_analysis:
        report_path = output_path / 'calibration_analysis.json'
        with open(report_path, 'w') as f:
            json.dump({
                'selection_info': selection_info,
                'images': analysis_data,
                'config': {
                    'num_samples': num_samples,
                    'target_size': target_size,
                    'quality_threshold': quality_threshold
                }
            }, f, indent=2)
        print(f"\nAnalysis report saved to: {report_path}")

    # Create README
    readme_path = output_path / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(f"""# Calibration Dataset for YuNet NPU Compilation

## Dataset Information
- **Total Images**: {len(selected_paths)}
- **Source**: {source_dir}
- **Quality Range**: {selection_info['quality_range'][0]:.1f} - {selection_info['quality_range'][1]:.1f}
- **Average Quality**: {selection_info['avg_quality']:.1f}
- **Target Size**: {target_size}

## Selection Strategy
1. Quality filtering: Only images with quality score >= {quality_threshold}
2. Diversity sampling: Balanced selection across quality distribution
3. 50% top-quality + 50% diverse distribution

## Quality Metrics
- **Sharpness**: Laplacian variance (edge clarity)
- **Brightness**: Optimal exposure (not too dark/bright)
- **Contrast**: Standard deviation (detail richness)
- **Resolution**: Image size adequacy
- **Color**: Color distribution (not monochrome)

## Usage
These images are preprocessed according to YuNet input requirements for NPU calibration.
Use with `calibration_config.json` for NPU compilation.

## Notes
- Images are selected for maximum diversity in facial characteristics
- Quality analysis ensures sharp, well-exposed images
- Representative of real-world deployment scenarios
""")

    stats = {
        'source_dir': str(source_dir),
        'output_dir': str(output_dir),
        'total_images_found': len(all_images),
        'candidates_analyzed': search_pool,
        'images_selected': len(selected_paths),
        'selection_info': selection_info
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Prepare calibration dataset for YuNet NPU compilation'
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        required=True,
        help='Source directory containing images (e.g., LFW dataset path)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./calibration_dataset',
        help='Output directory for calibration images'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of calibration samples (default: 100)'
    )
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=40,
        help='Minimum quality threshold 0-100 (default: 40)'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        nargs=2,
        default=[320, 320],
        help='Target image size for YuNet (default: 320 320)'
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

    print("="*60)
    print("YuNet NPU Calibration Dataset Preparation")
    print("="*60)
    print(f"Source: {args.source_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Samples: {args.num_samples}")
    print(f"Quality threshold: {args.quality_threshold}")
    print(f"Target size: {args.target_size}")
    print("="*60)

    stats = prepare_calibration_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        target_size=tuple(args.target_size),
        quality_threshold=args.quality_threshold,
        save_analysis=True
    )

    print("\n" + "="*60)
    print("Preparation Complete!")
    print("="*60)
    print(f"Total images found: {stats['total_images_found']}")
    print(f"Candidates analyzed: {stats['candidates_analyzed']}")
    print(f"Images selected: {stats['images_selected']}")
    print(f"Output directory: {stats['output_dir']}")
    print("="*60)

    print("\nNext steps:")
    print("1. Review calibration images in:", args.output_dir)
    print("2. Generate calibration config: python generate_calibration_config.py")
    print("3. Compile YuNet to NPU format using the generated config")


if __name__ == '__main__':
    main()
