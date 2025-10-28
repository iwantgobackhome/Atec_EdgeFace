"""
Test and validate calibration dataset and configuration
Ensures preprocessing pipeline is correct before NPU compilation
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from PIL import Image


def load_config(config_path):
    """Load calibration configuration"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def apply_preprocessing(image, preprocessings):
    """
    Apply preprocessing steps from config to an image

    Args:
        image: Input image (numpy array, HWC, BGR format as loaded by cv2)
        preprocessings: List of preprocessing operations

    Returns:
        processed: Preprocessed image tensor
    """
    current = image.astype(np.float32)

    for step in preprocessings:
        operation = list(step.keys())[0]
        params = step[operation]

        if operation == "convertColor":
            form = params["form"]
            if form == "BGR2RGB":
                current = cv2.cvtColor(current.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)
            elif form == "RGB2BGR":
                current = cv2.cvtColor(current.astype(np.uint8), cv2.COLOR_RGB2BGR).astype(np.float32)

        elif operation == "resize":
            width = params["width"]
            height = params["height"]
            current = cv2.resize(current, (width, height))

        elif operation == "centercrop":
            width = params["width"]
            height = params["height"]
            h, w = current.shape[:2]
            start_x = (w - width) // 2
            start_y = (h - height) // 2
            current = current[start_y:start_y+height, start_x:start_x+width]

        elif operation == "div":
            divisor = params["x"]
            current = current / divisor

        elif operation == "normalize":
            mean = np.array(params["mean"], dtype=np.float32)
            std = np.array(params["std"], dtype=np.float32)
            current = (current - mean) / std

        elif operation == "transpose":
            axis = params["axis"]
            current = np.transpose(current, axis)

        elif operation == "expandDim":
            axis = params["axis"]
            current = np.expand_dims(current, axis=axis)

        else:
            print(f"Warning: Unknown preprocessing operation: {operation}")

    return current


def test_single_image(image_path, config):
    """
    Test preprocessing on a single image

    Args:
        image_path: Path to test image
        config: Calibration configuration dict

    Returns:
        processed: Preprocessed tensor
        stats: Processing statistics
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    original_shape = image.shape
    print(f"Original image shape: {original_shape}")

    # Apply preprocessing
    preprocessings = config["default_loader"]["preprocessings"]
    processed = apply_preprocessing(image, preprocessings)

    # Get expected shape
    input_name = list(config["inputs"].keys())[0]
    expected_shape = tuple(config["inputs"][input_name])

    stats = {
        'original_shape': original_shape,
        'processed_shape': processed.shape,
        'expected_shape': expected_shape,
        'dtype': processed.dtype,
        'value_range': (processed.min(), processed.max()),
        'mean': processed.mean(),
        'std': processed.std()
    }

    print(f"Processed shape: {processed.shape}")
    print(f"Expected shape: {expected_shape}")
    print(f"Data type: {processed.dtype}")
    print(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")
    print(f"Mean: {processed.mean():.3f}")
    print(f"Std: {processed.std():.3f}")

    # Validate
    if processed.shape != expected_shape:
        print(f"WARNING: Shape mismatch! Expected {expected_shape}, got {processed.shape}")
    else:
        print("✓ Shape validation passed")

    return processed, stats


def test_dataset(dataset_path, config, num_samples=5):
    """
    Test preprocessing on multiple dataset samples

    Args:
        dataset_path: Path to calibration dataset
        config: Calibration configuration
        num_samples: Number of samples to test

    Returns:
        all_stats: List of statistics for each sample
    """
    dataset_path = Path(dataset_path)

    # Find images
    extensions = config["default_loader"]["file_extensions"]
    images = []
    for ext in extensions:
        images.extend(dataset_path.glob(f"*.{ext}"))

    if len(images) == 0:
        raise ValueError(f"No images found in {dataset_path}")

    print(f"Found {len(images)} images in dataset")
    print(f"Testing {min(num_samples, len(images))} samples...\n")

    all_stats = []
    for i, img_path in enumerate(images[:num_samples]):
        print(f"\n--- Sample {i+1}: {img_path.name} ---")
        try:
            processed, stats = test_single_image(img_path, config)
            all_stats.append(stats)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()

    return all_stats


def visualize_preprocessing(image_path, config, output_path=None):
    """
    Visualize preprocessing steps

    Args:
        image_path: Path to test image
        config: Calibration configuration
        output_path: Path to save visualization (optional)
    """
    import matplotlib.pyplot as plt

    # Load original image
    original = cv2.imread(str(image_path))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Apply preprocessing
    preprocessings = config["default_loader"]["preprocessings"]
    processed = apply_preprocessing(original, preprocessings)

    # For visualization, reverse the preprocessing to show what the network "sees"
    # Remove batch dimension
    if len(processed.shape) == 4:
        processed = processed[0]

    # Transpose back to HWC if needed
    if processed.shape[0] == 3:  # CHW format
        vis_image = np.transpose(processed, (1, 2, 0))
    else:
        vis_image = processed

    # Denormalize if normalized
    vis_image_display = vis_image.copy()
    if vis_image.min() < 0:  # Likely normalized
        # Assume standard normalization
        vis_image_display = vis_image_display * 0.5 + 0.5  # Reverse [-1,1] to [0,1]
    if vis_image_display.max() <= 1.0:
        vis_image_display = vis_image_display * 255

    vis_image_display = np.clip(vis_image_display, 0, 255).astype(np.uint8)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(original_rgb)
    axes[0].set_title(f"Original\nShape: {original.shape}")
    axes[0].axis('off')

    axes[1].imshow(vis_image_display)
    axes[1].set_title(f"Preprocessed (visualization)\nShape: {processed.shape}")
    axes[1].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Test and validate NPU calibration dataset and configuration'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to calibration config JSON'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Path to calibration dataset (overrides config)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples to test (default: 5)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization of preprocessing'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./calibration_test_output',
        help='Output directory for visualizations'
    )

    args = parser.parse_args()

    print("="*60)
    print("Calibration Dataset and Config Validation")
    print("="*60)

    # Load configuration
    print(f"\nLoading config: {args.config}")
    config = load_config(args.config)

    # Get dataset path
    dataset_path = args.dataset_path or config["default_loader"]["dataset_path"]
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return

    print(f"Dataset path: {dataset_path}")
    print(f"\nConfiguration:")
    print(f"  Input name: {list(config['inputs'].keys())[0]}")
    print(f"  Input shape: {list(config['inputs'].values())[0]}")
    print(f"  Calibration samples: {config['calibration_num']}")
    print(f"  Calibration method: {config['calibration_method']}")

    print("\nPreprocessing pipeline:")
    for i, step in enumerate(config["default_loader"]["preprocessings"], 1):
        operation = list(step.keys())[0]
        params = step[operation]
        print(f"  {i}. {operation}: {params}")

    # Test dataset
    print("\n" + "="*60)
    print("Testing Dataset Samples")
    print("="*60)

    all_stats = test_dataset(dataset_path, config, args.num_samples)

    # Summary statistics
    if all_stats:
        print("\n" + "="*60)
        print("Summary Statistics")
        print("="*60)
        value_ranges = [s['value_range'] for s in all_stats]
        means = [s['mean'] for s in all_stats]
        stds = [s['std'] for s in all_stats]

        print(f"Value range across samples:")
        print(f"  Min: {min(r[0] for r in value_ranges):.3f}")
        print(f"  Max: {max(r[1] for r in value_ranges):.3f}")
        print(f"Mean across samples: {np.mean(means):.3f} ± {np.std(means):.3f}")
        print(f"Std across samples: {np.mean(stds):.3f} ± {np.std(stds):.3f}")

    # Create visualizations
    if args.visualize:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print("Creating Visualizations")
        print("="*60)

        # Find images
        images = list(dataset_path.glob("*.jpg"))
        images.extend(dataset_path.glob("*.png"))

        num_vis = min(3, len(images))
        for i, img_path in enumerate(images[:num_vis]):
            output_path = output_dir / f"preprocessing_vis_{i+1}.png"
            print(f"Visualizing: {img_path.name}")
            try:
                visualize_preprocessing(img_path, config, output_path)
            except Exception as e:
                print(f"Error creating visualization: {e}")

    print("\n" + "="*60)
    print("Validation Complete!")
    print("="*60)
    print("\nIf all tests passed, the calibration dataset and config are ready for NPU compilation.")


if __name__ == '__main__':
    main()
