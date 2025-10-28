# NPU Calibration Pipeline for YuNet and EdgeFace

NPU 컴파일을 위한 고품질 calibration 데이터셋 준비 및 설정 파일 생성 도구입니다.

## 개요

ONNX 모델을 NPU로 컴파일할 때 quantization calibration이 성능에 큰 영향을 미칩니다. 이 도구는:
- LFW 등의 데이터셋에서 고품질 이미지를 자동 선택
- 모델별 적절한 전처리 파이프라인 설정
- NPU 컴파일러가 요구하는 JSON 형식의 calibration config 생성

## 파일 구성

```
npu_calibration/
├── prepare_calibration_dataset.py   # Calibration 이미지 선택 및 준비
├── generate_calibration_config.py   # NPU calibration config JSON 생성
├── test_calibration.py              # Calibration 데이터셋 및 설정 검증
└── README.md                         # 본 문서
```

## 사용 방법

### 1단계: Calibration 데이터셋 준비

LFW 데이터셋에서 고품질 이미지 100장을 자동 선택합니다.

```bash
python prepare_calibration_dataset.py \
  --source-dir /path/to/lfw \
  --output-dir ./calibration_dataset \
  --num-samples 100 \
  --quality-threshold 40
```

**파라미터:**
- `--source-dir`: 소스 이미지 디렉토리 (예: LFW 데이터셋)
- `--output-dir`: Calibration 이미지 저장 경로
- `--num-samples`: 선택할 이미지 수 (기본값: 100)
- `--quality-threshold`: 최소 품질 점수 0-100 (기본값: 40)
- `--seed`: 재현성을 위한 랜덤 시드 (기본값: 42)

**품질 평가 지표:**
- **Sharpness (35%)**: Laplacian variance로 측정되는 선명도
- **Brightness (20%)**: 적절한 노출 (너무 어둡거나 밝지 않음)
- **Contrast (20%)**: 표준편차로 측정되는 대비
- **Resolution (15%)**: 이미지 해상도 적절성
- **Color (10%)**: 색상 분포 (모노크롬이 아님)

**선택 전략:**
1. 품질 필터링: 품질 점수 >= threshold인 이미지만 선택
2. 다양성 확보:
   - 50%는 최고 품질 이미지
   - 50%는 품질 분포 전반에서 균등 샘플링
3. 결과: 다양한 얼굴 특성 (나이, 성별, 인종, 조명 조건)을 포함

**출력:**
- `calibration_dataset/`: 선택된 이미지들
- `calibration_dataset/calibration_analysis.json`: 품질 분석 리포트
- `calibration_dataset/README.md`: 데이터셋 정보

### 2단계: Calibration Config 생성

#### YuNet용 설정 생성

```bash
python generate_calibration_config.py \
  --model-type yunet \
  --dataset-path ./calibration_dataset \
  --output-path ./calibration_config_yunet.json \
  --calibration-num 100 \
  --calibration-method ema
```

#### EdgeFace용 설정 생성

```bash
python generate_calibration_config.py \
  --model-type edgeface \
  --dataset-path ./calibration_dataset \
  --output-path ./calibration_config_edgeface.json \
  --calibration-num 100 \
  --calibration-method ema
```

**파라미터:**
- `--model-type`: `yunet` 또는 `edgeface`
- `--dataset-path`: Calibration 이미지 경로
- `--output-path`: 출력 JSON 파일 경로
- `--calibration-num`: Calibration 샘플 수
- `--calibration-method`: Calibration 방법
  - `ema`: Exponential Moving Average (권장)
  - `minmax`: Min-Max calibration
  - `kl`: KL divergence
  - `percentile`: Percentile-based

**생성되는 설정 파일 예시 (YuNet):**
```json
{
  "inputs": {
    "input": [1, 3, 320, 320]
  },
  "calibration_num": 100,
  "calibration_method": "ema",
  "default_loader": {
    "dataset_path": "./calibration_dataset",
    "file_extensions": ["jpeg", "jpg", "png"],
    "preprocessings": [
      {"resize": {"width": 320, "height": 320}},
      {"transpose": {"axis": [2, 0, 1]}},
      {"expandDim": {"axis": 0}}
    ]
  }
}
```

### 3단계: 설정 검증 (선택사항)

Calibration이 올바르게 작동하는지 테스트합니다.

```bash
python test_calibration.py \
  --config ./calibration_config_yunet.json \
  --num-samples 5 \
  --visualize \
  --output-dir ./calibration_test_output
```

**파라미터:**
- `--config`: 테스트할 config JSON 파일
- `--dataset-path`: 데이터셋 경로 (config에서 자동 읽음)
- `--num-samples`: 테스트할 샘플 수
- `--visualize`: 전처리 시각화 생성
- `--output-dir`: 시각화 저장 경로

**출력:**
- 각 샘플의 전처리 결과 검증
- 텐서 shape, dtype, value range 확인
- 전처리 전후 비교 시각화

### 4단계: NPU 컴파일

생성된 config 파일을 사용하여 NPU 컴파일을 수행합니다.

```bash
# 예시 - 실제 NPU 컴파일러 명령어는 플랫폼에 따라 다름
<npu_compiler> \
  --model yunet.onnx \
  --output yunet.npu \
  --config calibration_config_yunet.json
```

## 전처리 파이프라인 상세

### YuNet 전처리
YuNet은 BGR 이미지를 입력으로 받으며, 정규화를 하지 않습니다.

1. **Resize**: 320x320 (또는 지정된 크기)
2. **Transpose**: HWC → CHW
3. **ExpandDim**: 배치 차원 추가

입력 형식: `[1, 3, 320, 320]`, BGR, 픽셀 값 범위 `[0, 255]`

### EdgeFace 전처리
EdgeFace/ArcFace는 RGB 이미지를 입력으로 받으며, 정규화를 수행합니다.

1. **Color Conversion**: BGR → RGB
2. **Resize**: 112x112
3. **Normalize**: 픽셀 값을 `[0, 1]`로 변환 (÷ 255)
4. **Standardize**: mean=0.5, std=0.5 적용
5. **Transpose**: HWC → CHW
6. **ExpandDim**: 배치 차원 추가

입력 형식: `[1, 3, 112, 112]`, RGB, 정규화된 값 범위 `[-1, 1]`

## Calibration 방법 선택 가이드

### EMA (Exponential Moving Average)
- **권장**: 대부분의 경우 최고의 성능
- 안정적이고 robust한 quantization
- 이상치에 덜 민감

### MinMax
- 가장 간단한 방법
- 빠르지만 이상치에 민감
- 작은 데이터셋에서 사용 가능

### KL Divergence
- 원본 분포와 양자화된 분포 간의 차이 최소화
- 계산 비용이 높음
- 큰 데이터셋에서 효과적

### Percentile
- 극단값 제외
- 안정적이지만 정보 손실 가능

## Calibration 샘플 수 선택

- **100장**: 일반적으로 충분 (권장)
- **50장**: 빠른 테스트용, 성능 저하 가능
- **200-500장**: 매우 높은 정확도 필요 시
- **더 많은 샘플 ≠ 항상 더 나은 성능**
  - 품질과 다양성이 양보다 중요
  - 이 도구는 자동으로 고품질 다양한 이미지 선택

## 성능 최적화 팁

1. **고품질 이미지 선택**
   - `--quality-threshold`를 40-60으로 설정
   - 너무 높으면 샘플이 부족할 수 있음

2. **다양성 확보**
   - LFW 같은 다양한 데이터셋 사용
   - 다양한 조명, 포즈, 인종 포함

3. **실제 배포 환경 반영**
   - 배포 환경과 유사한 이미지 사용
   - 예: 실내 조명이 주요 환경이면 실내 이미지 많이 포함

4. **Calibration 방법**
   - 기본적으로 EMA 사용
   - 성능 문제 시 다른 방법 실험

5. **검증 필수**
   - `test_calibration.py`로 전처리 파이프라인 확인
   - 컴파일 후 정확도 테스트 수행

## 트러블슈팅

### 이미지를 찾을 수 없음
```bash
# LFW 데이터셋 구조 확인
find /path/to/lfw -name "*.jpg" | head -10
```

### 품질 threshold가 너무 높음
```bash
# Threshold를 낮춰서 재시도
python prepare_calibration_dataset.py \
  --source-dir /path/to/lfw \
  --quality-threshold 20  # 40에서 20으로 낮춤
```

### Shape 불일치
```bash
# ONNX 모델의 입력 이름과 shape 확인
python -c "import onnx; model = onnx.load('yunet.onnx'); print(model.graph.input[0])"

# Config 생성 시 올바른 값 지정
python generate_calibration_config.py \
  --input-name "actual_input_name" \
  --input-size 320 320
```

## 예제: 전체 워크플로우

```bash
# 1. Calibration 데이터셋 준비
python prepare_calibration_dataset.py \
  --source-dir ~/datasets/lfw \
  --output-dir ./yunet_calibration \
  --num-samples 100

# 2. Config 생성
python generate_calibration_config.py \
  --model-type yunet \
  --dataset-path ./yunet_calibration \
  --output-path ./yunet_calib.json

# 3. 검증 (선택사항)
python test_calibration.py \
  --config ./yunet_calib.json \
  --visualize

# 4. NPU 컴파일
# (NPU 컴파일러 명령어는 플랫폼마다 다름)
```

## 참고사항

- **재현성**: `--seed` 파라미터로 동일한 이미지 선택 가능
- **디스크 공간**: 100장 기준 약 10-50MB
- **처리 시간**:
  - 데이터셋 준비: 수 분 (LFW 기준)
  - Config 생성: 즉시
  - 검증: 수 초

## 라이센스 및 참고자료

이 도구는 EdgeFace 프로젝트의 일부입니다.

참고:
- YuNet: https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet
- LFW Dataset: http://vis-www.cs.umass.edu/lfw/
- NPU Calibration: 각 NPU 플랫폼의 문서 참조
