"""
Generate NPU Calibration Configuration JSON for YuNet
Creates proper preprocessing pipeline matching YuNet's requirements
"""

import json
import argparse
from pathlib import Path


def generate_yunet_calibration_config(
    dataset_path="./calibration_dataset",
    output_path="./calibration_config.json",
    input_name="input",
    input_shape=(1, 3, 320, 320),
    calibration_num=100,
    calibration_method="ema"
):
    """
    Generate calibration configuration for YuNet NPU compilation

    YuNet preprocessing pipeline:
    - Input: BGR image (OpenCV default)
    - No normalization (uses raw pixel values 0-255)
    - Shape: (1, 3, H, W) where H, W are multiples of 32

    Args:
        dataset_path: Path to calibration dataset directory
        output_path: Output path for config JSON
        input_name: Name of input tensor (check ONNX model)
        input_shape: Input tensor shape [batch, channels, height, width]
        calibration_num: Number of calibration samples to use
        calibration_method: Calibration method (ema, minmax, kl, percentile)

    Returns:
        config: Dictionary with calibration configuration
    """

    # YuNet-specific preprocessing pipeline
    # YuNet expects BGR images with pixel values in range [0, 255]
    # No normalization or mean/std subtraction
    config = {
        "inputs": {
            input_name: list(input_shape)
        },
        "calibration_num": calibration_num,
        "calibration_method": calibration_method,
        "default_loader": {
            "dataset_path": dataset_path,
            "file_extensions": [
                "jpeg",
                "jpg",
                "png",
                "JPEG",
                "JPG",
                "PNG"
            ],
            "preprocessings": [
                # Step 1: Keep as BGR (YuNet expects BGR, not RGB)
                # No color conversion needed as OpenCV loads in BGR by default

                # Step 2: Resize to target size
                {
                    "resize": {
                        "width": input_shape[3],
                        "height": input_shape[2]
                    }
                },

                # Step 3: Transpose from HWC to CHW format
                {
                    "transpose": {
                        "axis": [2, 0, 1]  # HWC -> CHW
                    }
                },

                # Step 4: Add batch dimension
                {
                    "expandDim": {
                        "axis": 0
                    }
                }
            ]
        }
    }

    # Save configuration
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Calibration config saved to: {output_path}")
    print(f"\nConfiguration Summary:")
    print(f"  Input name: {input_name}")
    print(f"  Input shape: {input_shape}")
    print(f"  Calibration samples: {calibration_num}")
    print(f"  Calibration method: {calibration_method}")
    print(f"  Dataset path: {dataset_path}")
    print(f"\nPreprocessing pipeline:")
    for i, step in enumerate(config["default_loader"]["preprocessings"], 1):
        operation = list(step.keys())[0]
        params = step[operation]
        print(f"  {i}. {operation}: {params}")

    return config


def generate_edgeface_calibration_config(
    dataset_path="./calibration_dataset",
    output_path="./calibration_config_edgeface.json",
    input_name="input.1",
    input_shape=(1, 3, 112, 112),
    calibration_num=100,
    calibration_method="ema",
    use_arcface_preprocessing=True
):
    """
    Generate calibration configuration for EdgeFace/ArcFace models

    EdgeFace/ArcFace preprocessing:
    - Input: RGB image
    - Normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    - Shape: (1, 3, 112, 112)

    Args:
        dataset_path: Path to calibration dataset directory
        output_path: Output path for config JSON
        input_name: Name of input tensor
        input_shape: Input tensor shape
        calibration_num: Number of calibration samples
        calibration_method: Calibration method
        use_arcface_preprocessing: Use ArcFace-style preprocessing

    Returns:
        config: Dictionary with calibration configuration
    """

    preprocessings = []

    if use_arcface_preprocessing:
        # ArcFace/EdgeFace standard preprocessing
        preprocessings = [
            # Step 1: Convert BGR to RGB
            {
                "convertColor": {
                    "form": "BGR2RGB"
                }
            },
            # Step 2: Resize to target size
            {
                "resize": {
                    "width": input_shape[3],
                    "height": input_shape[2]
                }
            },
            # Step 3: Normalize to [0, 1]
            {
                "div": {
                    "x": 255.0
                }
            },
            # Step 4: Apply mean and std normalization
            {
                "normalize": {
                    "mean": [0.5, 0.5, 0.5],
                    "std": [0.5, 0.5, 0.5]
                }
            },
            # Step 5: Transpose HWC to CHW
            {
                "transpose": {
                    "axis": [2, 0, 1]
                }
            },
            # Step 6: Add batch dimension
            {
                "expandDim": {
                    "axis": 0
                }
            }
        ]
    else:
        # Basic preprocessing without normalization
        preprocessings = [
            {
                "convertColor": {
                    "form": "BGR2RGB"
                }
            },
            {
                "resize": {
                    "width": input_shape[3],
                    "height": input_shape[2]
                }
            },
            {
                "transpose": {
                    "axis": [2, 0, 1]
                }
            },
            {
                "expandDim": {
                    "axis": 0
                }
            }
        ]

    config = {
        "inputs": {
            input_name: list(input_shape)
        },
        "calibration_num": calibration_num,
        "calibration_method": calibration_method,
        "default_loader": {
            "dataset_path": dataset_path,
            "file_extensions": [
                "jpeg",
                "jpg",
                "png",
                "JPEG",
                "JPG",
                "PNG"
            ],
            "preprocessings": preprocessings
        }
    }

    # Save configuration
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"EdgeFace calibration config saved to: {output_path}")
    print(f"\nConfiguration Summary:")
    print(f"  Input name: {input_name}")
    print(f"  Input shape: {input_shape}")
    print(f"  Calibration samples: {calibration_num}")
    print(f"  Calibration method: {calibration_method}")
    print(f"  ArcFace preprocessing: {use_arcface_preprocessing}")

    return config


def main():
    parser = argparse.ArgumentParser(
        description='Generate NPU calibration configuration for YuNet or EdgeFace'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['yunet', 'edgeface'],
        default='yunet',
        help='Model type: yunet or edgeface'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='./calibration_dataset',
        help='Path to calibration dataset'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Output path for config JSON (auto-generated if not specified)'
    )
    parser.add_argument(
        '--input-name',
        type=str,
        default=None,
        help='Input tensor name (auto-detected from model type if not specified)'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        default=None,
        help='Input size [height width] (auto-detected from model type if not specified)'
    )
    parser.add_argument(
        '--calibration-num',
        type=int,
        default=100,
        help='Number of calibration samples (default: 100)'
    )
    parser.add_argument(
        '--calibration-method',
        type=str,
        choices=['ema', 'minmax', 'kl', 'percentile'],
        default='ema',
        help='Calibration method (default: ema)'
    )

    args = parser.parse_args()

    print("="*60)
    print("NPU Calibration Configuration Generator")
    print("="*60)

    # Set defaults based on model type
    if args.model_type == 'yunet':
        input_name = args.input_name or "input"
        input_size = args.input_size or [320, 320]
        output_path = args.output_path or "./calibration_config_yunet.json"

        config = generate_yunet_calibration_config(
            dataset_path=args.dataset_path,
            output_path=output_path,
            input_name=input_name,
            input_shape=(1, 3, input_size[0], input_size[1]),
            calibration_num=args.calibration_num,
            calibration_method=args.calibration_method
        )

    elif args.model_type == 'edgeface':
        input_name = args.input_name or "input.1"
        input_size = args.input_size or [112, 112]
        output_path = args.output_path or "./calibration_config_edgeface.json"

        config = generate_edgeface_calibration_config(
            dataset_path=args.dataset_path,
            output_path=output_path,
            input_name=input_name,
            input_shape=(1, 3, input_size[0], input_size[1]),
            calibration_num=args.calibration_num,
            calibration_method=args.calibration_method,
            use_arcface_preprocessing=True
        )

    print("\n" + "="*60)
    print("Configuration Generated Successfully!")
    print("="*60)
    print("\nNext steps:")
    print(f"1. Review the config file: {output_path}")
    print("2. Verify input tensor name matches your ONNX model")
    print("3. Use this config for NPU compilation")
    print("\nExample compilation command:")
    print(f"  <npu_compiler> --config {output_path} --model <model.onnx> --output <model.npu>")


if __name__ == '__main__':
    main()
