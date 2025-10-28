# Quick Start Guide - NPU Calibration

YuNet과 EdgeFace를 NPU로 컴파일하기 위한 calibration 데이터셋 준비 빠른 시작 가이드입니다.

## 🚀 5분 안에 시작하기

### 방법 1: 자동 스크립트 사용 (가장 쉬움)

```bash
# YuNet 모델용 calibration
./run_calibration_pipeline.sh \
  --source-dir ~/datasets/lfw \
  --model-type yunet \
  --num-samples 100

# EdgeFace 모델용 calibration
./run_calibration_pipeline.sh \
  --source-dir ~/datasets/lfw \
  --model-type edgeface \
  --num-samples 100
```

완료! `calibration_output/` 디렉토리에 모든 결과가 생성됩니다.

### 방법 2: 단계별 실행

#### Step 1: Calibration 이미지 선택

```bash
python prepare_calibration_dataset.py \
  --source-dir ~/datasets/lfw \
  --output-dir ./calibration_dataset \
  --num-samples 100
```

#### Step 2: Config 파일 생성

**YuNet의 경우:**
```bash
python generate_calibration_config.py \
  --model-type yunet \
  --dataset-path ./calibration_dataset
```

**EdgeFace의 경우:**
```bash
python generate_calibration_config.py \
  --model-type edgeface \
  --dataset-path ./calibration_dataset
```

#### Step 3: 검증 (선택사항)

```bash
python test_calibration.py \
  --config ./calibration_config_yunet.json \
  --visualize
```

## 📁 출력 파일

### Calibration 데이터셋
```
calibration_dataset/
├── calib_0000.jpg          # 선택된 calibration 이미지들
├── calib_0001.jpg
├── ...
├── calib_0099.jpg
├── calibration_analysis.json  # 품질 분석 리포트
└── README.md               # 데이터셋 정보
```

### Config 파일
```
calibration_config_yunet.json     # YuNet용 NPU calibration 설정
calibration_config_edgeface.json  # EdgeFace용 NPU calibration 설정
```

## 🎯 주요 사용 사례

### 사례 1: YuNet을 NPU로 컴파일

```bash
# 1. Calibration 준비
./run_calibration_pipeline.sh -s ~/datasets/lfw -m yunet

# 2. ONNX 모델의 입력 이름 확인
python -c "import onnx; m = onnx.load('yunet.onnx'); print(m.graph.input[0].name)"

# 3. 필요하면 config의 input name 수정
# vim calibration_output/calibration_config_yunet.json

# 4. NPU 컴파일 (플랫폼에 따라 명령어 다름)
# <npu_compiler> --model yunet.onnx --config calibration_output/calibration_config_yunet.json
```

### 사례 2: EdgeFace를 NPU로 컴파일

```bash
# 1. Calibration 준비
./run_calibration_pipeline.sh -s ~/datasets/lfw -m edgeface -n 100

# 2. NPU 컴파일
# <npu_compiler> --model edgeface.onnx --config calibration_output/calibration_config_edgeface.json
```

### 사례 3: 고품질 calibration (더 많은 샘플)

```bash
./run_calibration_pipeline.sh \
  -s ~/datasets/lfw \
  -m yunet \
  -n 200 \
  -q 50  # 더 높은 품질 threshold
```

### 사례 4: 빠른 테스트 (적은 샘플)

```bash
./run_calibration_pipeline.sh \
  -s ~/datasets/lfw \
  -m yunet \
  -n 50 \
  -q 30 \
  --skip-validation  # 검증 스킵
```

## ⚙️ 고급 설정

### 다른 calibration 방법 시도

```bash
# MinMax (가장 빠름)
./run_calibration_pipeline.sh -s ~/datasets/lfw -m yunet -c minmax

# KL Divergence (높은 정확도)
./run_calibration_pipeline.sh -s ~/datasets/lfw -m yunet -c kl

# Percentile (안정적)
./run_calibration_pipeline.sh -s ~/datasets/lfw -m yunet -c percentile
```

### 커스텀 입력 크기

```bash
python generate_calibration_config.py \
  --model-type yunet \
  --input-size 640 640  # 640x640 입력 사용
```

### 커스텀 입력 이름

```bash
python generate_calibration_config.py \
  --model-type yunet \
  --input-name "images"  # ONNX 모델의 실제 입력 이름
```

## 🔍 문제 해결

### "No images found" 에러

```bash
# LFW 데이터셋 경로 확인
ls ~/datasets/lfw

# 이미지가 있는지 확인
find ~/datasets/lfw -name "*.jpg" | head -5
```

### "Not enough quality images" 경고

```bash
# Quality threshold를 낮춤
./run_calibration_pipeline.sh -s ~/datasets/lfw -q 20
```

### Shape mismatch 에러

```bash
# ONNX 모델의 입력 정보 확인
python -c "
import onnx
model = onnx.load('your_model.onnx')
input_tensor = model.graph.input[0]
print(f'Name: {input_tensor.name}')
print(f'Shape: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}')
"

# 올바른 값으로 config 생성
python generate_calibration_config.py \
  --input-name "실제_입력_이름" \
  --input-size 실제_높이 실제_너비
```

## 📊 성능 체크리스트

calibration 품질을 확인하려면:

- [ ] `calibration_analysis.json`에서 평균 품질 점수 확인
  - 60 이상: 우수
  - 40-60: 양호
  - 40 미만: 품질 향상 필요

- [ ] 선택된 이미지들이 다양한지 육안으로 확인

- [ ] `test_calibration.py`로 전처리 파이프라인 검증
  - Shape가 예상과 일치하는지
  - Value range가 올바른지

- [ ] NPU 컴파일 후 정확도 테스트
  - LFW evaluation 등으로 성능 측정
  - 원본 ONNX 모델과 비교

## 💡 팁

1. **처음 사용할 때는 검증 활성화**
   ```bash
   # --skip-validation을 사용하지 않음
   ./run_calibration_pipeline.sh -s ~/datasets/lfw -m yunet
   ```

2. **재현성이 중요하면 seed 고정**
   ```bash
   python prepare_calibration_dataset.py --seed 42
   ```

3. **디스크 공간 절약**
   ```bash
   # 샘플 수를 50-100으로 제한
   # 일반적으로 충분함
   ```

4. **배포 환경 반영**
   - 실제 사용할 환경과 유사한 이미지 사용
   - 예: 실내 카메라 → 실내 조명 이미지 많이 포함

## 📖 더 알아보기

- 상세 문서: [README.md](README.md)
- 전처리 파이프라인 설명: README.md의 "전처리 파이프라인 상세" 섹션
- Calibration 방법 비교: README.md의 "Calibration 방법 선택 가이드" 섹션

## 🆘 도움이 필요하신가요?

스크립트 도움말 보기:
```bash
./run_calibration_pipeline.sh --help
python prepare_calibration_dataset.py --help
python generate_calibration_config.py --help
python test_calibration.py --help
```

## 예제 출력

성공적으로 실행되면 다음과 같은 출력을 볼 수 있습니다:

```
========================================
NPU Calibration Pipeline
========================================
Source directory: /home/user/datasets/lfw
Output directory: ./calibration_output
Model type: yunet
Number of samples: 100
========================================

[Step 1/3] Preparing calibration dataset...
Found 13233 total images in /home/user/datasets/lfw
Searching in pool of 1000 images...
Selected 100 calibration images
Quality range: 45.2 - 87.3
Average quality: 62.1
✓ Dataset preparation completed

[Step 2/3] Generating calibration config...
Calibration config saved to: ./calibration_output/calibration_config_yunet.json
✓ Config generation completed

[Step 3/3] Validating calibration setup...
✓ Shape validation passed
✓ Validation completed

========================================
Calibration Pipeline Completed!
========================================
```
