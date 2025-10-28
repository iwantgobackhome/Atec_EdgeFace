# Changelog

## 2024-10-28 - Initial Release

### Added
- **YuNet calibration pipeline**
  - `prepare_calibration_dataset.py`: Select high-quality images from LFW
  - `run_calibration_pipeline.sh`: Automated YuNet calibration
  - Quality metrics: sharpness, brightness, contrast, resolution, color

- **EdgeFace calibration pipeline** ⭐
  - `prepare_aligned_faces.py`: Extract aligned faces using YuNet
  - `run_edgeface_calibration.sh`: Automated EdgeFace calibration
  - Handles detection → alignment → calibration dataset

- **Config generation**
  - `generate_calibration_config.py`: Generate NPU calibration JSON
  - Support for YuNet (BGR, no normalization)
  - Support for EdgeFace (RGB, ArcFace normalization)
  - 4 calibration methods: ema, minmax, kl, percentile

- **Validation**
  - `test_calibration.py`: Verify preprocessing pipeline
  - Shape, dtype, value range validation
  - Visualization support

- **Documentation**
  - `README.md`: Complete guide
  - `README_EDGEFACE.md`: EdgeFace-specific guide
  - `QUICKSTART.md`: Quick start guide
  - `SUMMARY.md`: Overview and comparison

- **Examples**
  - `example_config_yunet.json`: YuNet config example
  - `example_config_edgeface.json`: EdgeFace config example

### Fixed (2024-10-28 16:30)
- Fixed JSON serialization error with numpy float32/int types
  - Convert numpy types to Python types in `prepare_aligned_faces.py`
  - Convert numpy types to Python types in `prepare_calibration_dataset.py`
  - All quality scores and metrics now properly serialized

### Notes
- EdgeFace requires aligned face images (112x112)
- YuNet model needed for EdgeFace calibration
- Recommended: 100 samples, EMA calibration method
