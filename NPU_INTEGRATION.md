# DeepX NPU Integration Guide

EdgeFace 얼굴 인식 시스템에 DeepX NPU 지원이 추가되었습니다.

## 개요

DeepX NPU를 사용하여 YuNet 얼굴 검출과 EdgeFace 얼굴 인식을 가속화할 수 있습니다.

### 지원하는 모델

1. **YuNet Face Detector** (NPU)
   - 모델 파일: `face_alignment/models/face_detection_yunet_2023mar.dxnn`
   - 입력 크기: 320x320
   - 기능: 얼굴 검출 및 랜드마크 추출 (5-point landmarks)

2. **EdgeFace Recognizer** (NPU)
   - 모델 파일: `checkpoints/edgeface_xs_gamma_06.dxnn`
   - 입력 크기: 112x112
   - 기능: 얼굴 임베딩 추출 (512-d vector)

## 파일 구조

```
EdgeFace/
├── face_alignment/
│   ├── yunet_npu.py                    # YuNet NPU detector wrapper
│   ├── unified_detector.py             # Updated with yunet_npu support
│   └── models/
│       └── face_detection_yunet_2023mar.dxnn  # YuNet NPU model
├── edgeface_npu_recognizer.py          # EdgeFace NPU recognizer
├── face_recognition_system.py          # Updated with NPU support
├── face_recognition_gui.py             # Updated GUI with NPU options
├── checkpoints/
│   ├── edgeface_xs_gamma_06.pt         # PyTorch model
│   └── edgeface_xs_gamma_06.dxnn       # NPU model
└── test_npu_models.py                  # NPU model testing script
```

## 사전 요구사항

### DeepX NPU SDK 설치

NPU를 사용하려면 DeepX NPU SDK가 설치되어 있어야 합니다.

```bash
# DeepX SDK 설치 (DeepX 문서 참조)
cd /your-dxrt-directory/python_package
pip install .
```

설치 확인:
```bash
python -c "from dx_engine import InferenceEngine; print('dx_engine available')"
```

### 모델 파일 준비

NPU 모델 파일(.dxnn)을 지정된 위치에 배치:

1. **YuNet NPU 모델**
   ```bash
   cp face_detection_yunet_2023mar.dxnn face_alignment/models/
   ```

2. **EdgeFace NPU 모델**
   ```bash
   cp edgeface_xs_gamma_06.dxnn checkpoints/
   ```

## 사용 방법

### 1. GUI에서 사용

GUI를 실행하고 NPU 옵션을 선택:

```bash
python face_recognition_gui.py
```

**설정 방법:**
1. **Face Detector** 드롭다운에서 `yunet_npu` 선택
2. **Device** 드롭다운에서 `npu` 선택
3. **Start Camera** 버튼 클릭

또는:
- **Device**를 `npu`로 설정하면 자동으로 `yunet`이 `yunet_npu`로 전환됩니다.

### 2. Python 코드에서 사용

```python
from face_recognition_system import FaceRecognitionSystem

# NPU 모드로 시스템 초기화
system = FaceRecognitionSystem(
    detector_method='yunet_npu',
    edgeface_model_path='checkpoints/edgeface_xs_gamma_06.dxnn',
    edgeface_model_name='edgeface_xs_gamma_06',
    device='npu',
    similarity_threshold=0.5,
    use_npu=True
)

# 카메라 실행
system.run_camera(camera_id=0)
```

### 3. NPU 모델 테스트

NPU 모델이 올바르게 작동하는지 확인:

```bash
python test_npu_models.py
```

이 스크립트는:
- YuNet NPU 모델의 입출력 형식 확인
- EdgeFace NPU 모델의 입출력 형식 확인
- 테스트 이미지로 추론 실행

## 구현 세부사항

### YuNet NPU Detector

**파일:** `face_alignment/yunet_npu.py`

**전처리:**
1. 이미지를 320x320으로 리사이즈
2. BGR → RGB 변환
3. HWC → CHW 전치
4. 배치 차원 추가 (1, 3, 320, 320)
5. uint8 타입으로 변환

**후처리:**
- YuNet 출력을 얼굴 검출 결과로 디코딩
- 각 얼굴: [x, y, w, h, x1, y1, ..., x5, y5, confidence]
- 좌표를 원본 이미지 크기로 스케일 변환

### EdgeFace NPU Recognizer

**파일:** `edgeface_npu_recognizer.py`

**전처리:**
1. 얼굴 이미지를 112x112로 리사이즈
2. BGR → RGB 변환
3. [0, 255] → [0, 1] 정규화 (÷ 255.0)
4. 표준화: (x - 0.5) / 0.5
5. HWC → CHW 전치
6. 배치 차원 추가 (1, 3, 112, 112)
7. float32 타입으로 변환

**후처리:**
- 출력 텐서를 512-d 벡터로 평탄화
- L2 정규화

## 성능 비교

| 모드 | Detector | Recognizer | 예상 속도 |
|------|----------|------------|-----------|
| CPU  | YuNet (OpenCV) | EdgeFace (PyTorch) | ~10-15 FPS |
| GPU  | YuNet (OpenCV) | EdgeFace (PyTorch CUDA) | ~30-50 FPS |
| NPU  | YuNet (DeepX) | EdgeFace (DeepX) | ~40-60 FPS* |

\* 실제 성능은 NPU 하드웨어에 따라 다름

## 문제 해결

### 1. dx_engine import 실패

```
ImportError: cannot import name 'InferenceEngine' from 'dx_engine'
```

**해결:** DeepX SDK를 설치하세요.
```bash
cd /your-dxrt-directory/python_package
pip install .
```

### 2. 모델 파일을 찾을 수 없음

```
RuntimeError: YuNet NPU detector 초기화 실패
```

**해결:** 모델 파일이 올바른 위치에 있는지 확인:
- `face_alignment/models/face_detection_yunet_2023mar.dxnn`
- `checkpoints/edgeface_xs_gamma_06.dxnn`

### 3. YuNet 출력 형식 확인

YuNet NPU의 출력 디코딩은 다음과 같은 형식을 가정합니다:

**예상 출력 형식:**
- **Shape**: `(N, 15)` 또는 `(1, N, 15)`
  - N: 검출된 얼굴 개수
  - 15: [x, y, w, h, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, confidence]

**디버깅 방법:**

임베디드 보드에서 실행:
```bash
python test_npu_models.py
```

출력 예시:
```
Testing YuNet NPU Model
============================================================
Loading model: face_alignment/models/face_detection_yunet_2023mar.dxnn
✅ Model loaded successfully
📊 Input size: [1, 3, 320, 320]
📊 Output dtype: ['FLOAT32']

Loading test image: test.jpg
Image shape: (480, 640, 3)
Input tensor shape: (1, 3, 320, 320), dtype: uint8

🚀 Running inference...

📊 Number of outputs: 1
  Output 0:
    - Shape: (5, 15)           # 5개 얼굴 검출
    - Dtype: float32
    - Min: -12.3456, Max: 456.7890
    - Mean: 123.4567, Std: 89.0123
    - Flattened shape: (75,)
    - First 10 values: [120.5, 85.3, 60.2, 75.8, 115.2, ...]
    - ✅ Detected YuNet format: (N=5, 15)
    - Confidence scores (column 14): [0.95, 0.88, 0.82, 0.76, 0.65]

🔍 Testing decoder...
✅ Decoded 5 faces
  Face 0: confidence=0.950, bbox=[120.5, 85.3, 60.2, 75.8]
  Face 1: confidence=0.880, bbox=[250.1, 120.4, 58.9, 72.3]
  ...
```

**만약 출력 형식이 다른 경우:**

출력이 다른 형태(예: 여러 텐서, 다른 차원)라면 `yunet_npu.py`의 `_decode_outputs()` 함수를 수정해야 합니다.

예시 수정 위치 (face_alignment/yunet_npu.py:124):
```python
def _decode_outputs(self, outputs, scale_x, scale_y):
    # 출력 형식에 맞게 수정
    # ...
```

### 4. 성능 및 메모리 이슈

**느린 추론 속도:**
- NPU 드라이버가 올바르게 설치되었는지 확인
- `test_npu_models.py`로 추론 시간 측정
- calibration 설정 확인

**메모리 부족:**
```python
# 배치 크기 조정 (현재는 항상 1)
# EdgeFace recognizer의 extract_embeddings_batch() 사용 시 주의
```

### 5. 임베디드 보드 배포 체크리스트

1. ✅ DeepX SDK 설치 확인
   ```bash
   python -c "from dx_engine import InferenceEngine; print('OK')"
   ```

2. ✅ 모델 파일 복사
   ```bash
   ls face_alignment/models/face_detection_yunet_2023mar.dxnn
   ls checkpoints/edgeface_xs_gamma_06.dxnn
   ```

3. ✅ 테스트 스크립트 실행
   ```bash
   python test_npu_models.py
   ```

4. ✅ GUI 테스트
   ```bash
   python face_recognition_gui.py
   # Device: npu 선택
   # Detector: yunet_npu 선택
   ```

5. ✅ 카메라 권한 확인
   ```bash
   v4l2-ctl --list-devices
   ls -l /dev/video*
   ```

## 모델 변환 (참고)

ONNX 모델을 DXNN으로 변환하는 방법 (DeepX 문서 참조):

```bash
# YuNet 변환 예시
dx-com -m face_detection_yunet_2023mar.onnx \
       -c calibration_config_yunet.json \
       -o face_detection_yunet_2023mar.dxnn

# EdgeFace 변환 예시
dx-com -m edgeface_xs_gamma_06.onnx \
       -c calibration_config_edgeface.json \
       -o edgeface_xs_gamma_06.dxnn
```

calibration config 파일은 `npu_calibration/` 디렉토리를 참조하세요.

## 추가 정보

- DeepX SDK 문서: `npu_calibration/deepX_document/`
- Python 예제: `npu_calibration/deepX_document/07_Python_Examples.md`
- YuNet calibration config: `npu_calibration/calibration_output/calibration_config_yunet.json`
- EdgeFace calibration config: `npu_calibration/edgeface_calibration_output/calibration_config_edgeface.json`

## 라이선스

이 통합 코드는 EdgeFace 프로젝트의 라이선스를 따릅니다.
DeepX NPU SDK는 별도의 라이선스가 적용됩니다.
