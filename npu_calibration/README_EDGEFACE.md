# EdgeFace NPU Calibration - 중요 차이점

## ⚠️ EdgeFace는 다릅니다!

**Detection 모델 (YuNet)** vs **Recognition 모델 (EdgeFace)**

| 모델 | 입력 데이터 | Calibration 데이터 |
|------|------------|-------------------|
| **YuNet** | 일반 이미지 (임의 크기) | 일반 이미지 (LFW 원본) ✓ |
| **EdgeFace** | **정렬된 얼굴** (112x112) | **정렬된 얼굴** 필요! ⚠️ |

## 왜 다른가?

EdgeFace는 실제 사용 시 다음 파이프라인을 거칩니다:

```
원본 이미지 → YuNet (detection) → 얼굴 정렬 (112x112) → EdgeFace (recognition)
                                        ↑
                                  이 부분이 입력!
```

따라서 **EdgeFace calibration도 정렬된 얼굴 이미지를 사용**해야 합니다!

## EdgeFace Calibration 준비 방법

### 방법 1: 자동 스크립트 (권장)

```bash
# YuNet으로 얼굴 정렬 + EdgeFace calibration config 생성
./run_edgeface_calibration.sh \
  --source-dir ~/datasets/lfw \
  --yunet-model ./face_detection_yunet_2023mar.onnx \
  --num-samples 100
```

**필요한 것:**
- LFW 데이터셋 (또는 얼굴이 포함된 이미지들)
- YuNet ONNX 모델 (detection + alignment용)

**처리 과정:**
1. YuNet으로 얼굴 detection
2. 5-point landmark로 얼굴 정렬 (112x112)
3. 정렬된 얼굴 이미지 저장
4. EdgeFace용 calibration config 생성

### 방법 2: 단계별 실행

#### Step 1: 얼굴 정렬 및 추출

```bash
python prepare_aligned_faces.py \
  --source-dir ~/datasets/lfw \
  --output-dir ./aligned_faces \
  --yunet-model ./face_detection_yunet_2023mar.onnx \
  --num-samples 100 \
  --quality-threshold 80
```

**출력:**
- `aligned_faces/aligned_0000.jpg` ~ `aligned_0099.jpg`
- 모두 112x112 크기의 정렬된 얼굴 이미지

#### Step 2: EdgeFace calibration config 생성

```bash
python generate_calibration_config.py \
  --model-type edgeface \
  --dataset-path ./aligned_faces \
  --output-path ./calibration_config_edgeface.json
```

#### Step 3: 검증

```bash
python test_calibration.py \
  --config ./calibration_config_edgeface.json \
  --visualize
```

## YuNet 모델 다운로드

EdgeFace calibration을 위해 YuNet 모델이 필요합니다:

```bash
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## 전체 워크플로우 비교

### YuNet Calibration (일반 이미지)
```bash
# 일반 이미지로 바로 calibration
./run_calibration_pipeline.sh -s ~/datasets/lfw -m yunet
```

### EdgeFace Calibration (정렬된 얼굴)
```bash
# 1단계: YuNet으로 얼굴 정렬 필요!
./run_edgeface_calibration.sh -s ~/datasets/lfw -y yunet.onnx
```

## EdgeFace Preprocessing Pipeline

정렬된 얼굴 이미지 (112x112)에 적용되는 전처리:

```json
{
  "preprocessings": [
    {"convertColor": {"form": "BGR2RGB"}},      // OpenCV BGR → RGB
    {"resize": {"width": 112, "height": 112}},  // 이미 112x112지만 명시
    {"div": {"x": 255.0}},                      // [0,255] → [0,1]
    {"normalize": {                              // ArcFace 정규화
      "mean": [0.5, 0.5, 0.5],
      "std": [0.5, 0.5, 0.5]                    // 결과: [-1, 1]
    }},
    {"transpose": {"axis": [2, 0, 1]}},         // HWC → CHW
    {"expandDim": {"axis": 0}}                  // 배치 차원
  ]
}
```

## 주의사항

### ❌ 잘못된 방법
```bash
# EdgeFace에 일반 이미지 사용 - 안됨!
python prepare_calibration_dataset.py --source-dir ~/lfw
python generate_calibration_config.py --model-type edgeface
```

**문제:** EdgeFace는 정렬된 얼굴만 처리할 수 있습니다.

### ✅ 올바른 방법
```bash
# EdgeFace용: 먼저 얼굴 정렬
python prepare_aligned_faces.py --source-dir ~/lfw --yunet-model yunet.onnx
python generate_calibration_config.py --model-type edgeface --dataset-path ./aligned_faces
```

**올바름:** 정렬된 얼굴 이미지 사용

## 파라미터 조정

### Quality Threshold

```bash
# 기본값 (80): 높은 품질의 detection만
./run_edgeface_calibration.sh -s ~/lfw -y yunet.onnx -q 80

# 낮은 값 (70): 더 많은 샘플, 품질은 낮음
./run_edgeface_calibration.sh -s ~/lfw -y yunet.onnx -q 70

# 매우 높은 값 (90): 최고 품질, 샘플 부족 가능
./run_edgeface_calibration.sh -s ~/lfw -y yunet.onnx -q 90
```

**권장값:**
- LFW 같은 고품질 데이터셋: 80
- 실제 촬영 이미지 (품질 낮음): 70
- 스튜디오 사진 (품질 높음): 90

### 샘플 수

```bash
# 빠른 테스트
./run_edgeface_calibration.sh -s ~/lfw -y yunet.onnx -n 50

# 표준 (권장)
./run_edgeface_calibration.sh -s ~/lfw -y yunet.onnx -n 100

# 고품질
./run_edgeface_calibration.sh -s ~/lfw -y yunet.onnx -n 200
```

## 트러블슈팅

### "No faces detected"

**원인:** YuNet이 얼굴을 찾지 못함

**해결:**
```bash
# Quality threshold 낮추기
./run_edgeface_calibration.sh -s ~/lfw -y yunet.onnx -q 70

# 더 많은 소스 이미지 사용
# YuNet은 샘플 10배까지 탐색함 (100개 필요 → 1000개 탐색)
```

### "Not enough faces"

**원인:** 충분한 고품질 detection이 없음

**해결:**
```bash
# 1. Quality threshold 낮추기
-q 70

# 2. 더 큰 데이터셋 사용
--source-dir /path/to/larger/dataset

# 3. 필요 샘플 수 줄이기
-n 50
```

### YuNet 모델 에러

**원인:** YuNet 모델 파일이 없거나 손상됨

**해결:**
```bash
# 모델 다운로드
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# 경로 지정
./run_edgeface_calibration.sh -s ~/lfw -y ./face_detection_yunet_2023mar.onnx
```

## 출력 파일 구조

```
edgeface_calibration_output/
├── aligned_faces/                      # 정렬된 얼굴 이미지
│   ├── aligned_0000.jpg               # 112x112 정렬된 얼굴
│   ├── aligned_0001.jpg
│   ├── ...
│   ├── aligned_0099.jpg
│   ├── aligned_faces_analysis.json    # Detection 통계
│   └── README.md
├── calibration_config_edgeface.json   # NPU calibration config
└── validation_output/                  # 검증 결과 (옵션)
    └── preprocessing_vis_*.png
```

## 요약

1. **EdgeFace는 정렬된 얼굴 이미지 필요**
2. **YuNet으로 먼저 얼굴 정렬**
3. **자동 스크립트 사용 권장:** `./run_edgeface_calibration.sh`
4. **YuNet 모델 필요** (detection + alignment용)
5. **Quality threshold 조정** (70-90)

## 다음 단계

calibration config 생성 후:

```bash
# EdgeFace ONNX → NPU 컴파일
<npu_compiler> \
  --model edgeface.onnx \
  --config edgeface_calibration_output/calibration_config_edgeface.json \
  --output edgeface.npu
```
