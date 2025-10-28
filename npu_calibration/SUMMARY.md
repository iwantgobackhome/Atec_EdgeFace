# NPU Calibration 전체 요약

## 🎯 핵심 차이점

YuNet과 EdgeFace는 **입력 데이터가 다르므로** calibration 방법도 다릅니다!

| 모델 | 입력 | Calibration 데이터 | 스크립트 |
|------|------|-------------------|---------|
| **YuNet** (Detection) | 일반 이미지 | 일반 이미지 | `run_calibration_pipeline.sh` |
| **EdgeFace** (Recognition) | 정렬된 얼굴 (112x112) | 정렬된 얼굴 | `run_edgeface_calibration.sh` |

## 📁 파일 구성

```
npu_calibration/
├── 문서
│   ├── README.md                      # 전체 가이드
│   ├── README_EDGEFACE.md             # EdgeFace 특화 가이드 ⭐
│   ├── QUICKSTART.md                  # 빠른 시작
│   └── SUMMARY.md                     # 본 문서
│
├── YuNet용 (Detection 모델)
│   ├── prepare_calibration_dataset.py # 일반 이미지 선택
│   ├── run_calibration_pipeline.sh    # 전체 자동화
│   └── example_config_yunet.json      # Config 예제
│
├── EdgeFace용 (Recognition 모델)
│   ├── prepare_aligned_faces.py       # 얼굴 정렬 및 추출 ⭐
│   ├── run_edgeface_calibration.sh    # 전체 자동화 ⭐
│   └── example_config_edgeface.json   # Config 예제
│
└── 공통
    ├── generate_calibration_config.py # Config JSON 생성
    └── test_calibration.py            # 검증
```

## 🚀 빠른 사용법

### YuNet NPU Calibration

```bash
# 한 줄로 끝!
./run_calibration_pipeline.sh \
  --source-dir ~/datasets/lfw \
  --model-type yunet \
  --num-samples 100
```

**출력:**
- `calibration_output/calibration_dataset/`: 선택된 일반 이미지
- `calibration_output/calibration_config_yunet.json`: NPU config

### EdgeFace NPU Calibration

```bash
# YuNet 모델 다운로드 (한 번만)
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# 한 줄로 끝!
./run_edgeface_calibration.sh \
  --source-dir ~/datasets/lfw \
  --yunet-model ./face_detection_yunet_2023mar.onnx \
  --num-samples 100
```

**출력:**
- `edgeface_calibration_output/aligned_faces/`: 정렬된 얼굴 이미지 (112x112)
- `edgeface_calibration_output/calibration_config_edgeface.json`: NPU config

## 🔑 EdgeFace의 핵심 차이점

### 왜 다른가?

**실제 사용 파이프라인:**
```
입력 이미지 → [YuNet Detection] → 정렬된 얼굴 → [EdgeFace Recognition] → 특징 벡터
                                      ↑
                                 이게 EdgeFace 입력!
```

**따라서:**
- YuNet calibration: 일반 이미지 사용 ✓
- EdgeFace calibration: **정렬된 얼굴 필요!** ⚠️

### EdgeFace Calibration 프로세스

```
LFW 이미지 → [YuNet Detection] → 얼굴 정렬 (112x112) → Calibration 데이터셋
                                                              ↓
                                                    EdgeFace NPU Config
```

## 📊 Config 파일 비교

### YuNet Config

```json
{
  "inputs": {"input": [1, 3, 320, 320]},
  "calibration_num": 100,
  "calibration_method": "ema",
  "default_loader": {
    "preprocessings": [
      {"resize": {"width": 320, "height": 320}},  // BGR 유지
      {"transpose": {"axis": [2, 0, 1]}},
      {"expandDim": {"axis": 0}}
    ]
  }
}
```

**특징:**
- BGR 입력 (변환 없음)
- 정규화 없음 (픽셀 값 0-255)
- 크기만 조정

### EdgeFace Config

```json
{
  "inputs": {"input.1": [1, 3, 112, 112]},
  "calibration_num": 100,
  "calibration_method": "ema",
  "default_loader": {
    "preprocessings": [
      {"convertColor": {"form": "BGR2RGB"}},      // BGR → RGB
      {"resize": {"width": 112, "height": 112}},
      {"div": {"x": 255.0}},                      // [0,255] → [0,1]
      {"normalize": {                              // ArcFace 정규화
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5]
      }},
      {"transpose": {"axis": [2, 0, 1]}},
      {"expandDim": {"axis": 0}}
    ]
  }
}
```

**특징:**
- RGB 입력 (BGR2RGB 변환)
- ArcFace 정규화 (결과: [-1, 1])
- 정렬된 얼굴 이미지 필요

## 💡 Calibration 품질 최적화

### 샘플 수
- **50장**: 빠른 테스트
- **100장**: 표준 (권장) ⭐
- **200장**: 고품질 (시간 2배)

### Quality Threshold

**YuNet용 (일반 이미지 품질):**
- 40: 표준 (권장)
- 30: 더 많은 샘플, 품질 낮음
- 50: 고품질, 샘플 부족 가능

**EdgeFace용 (Detection confidence):**
- 80: 표준 (권장) ⭐
- 70: 더 많은 샘플
- 90: 최고 품질 detection만

### Calibration Method
- **ema**: 권장 (가장 안정적) ⭐
- **minmax**: 빠름, 이상치에 민감
- **kl**: 높은 정확도, 느림
- **percentile**: 안정적, 중간 성능

## 🔍 검증

```bash
# Calibration config 검증
python test_calibration.py \
  --config calibration_config_yunet.json \
  --visualize \
  --num-samples 5
```

**확인 사항:**
- ✓ Tensor shape 일치
- ✓ Value range 올바름
- ✓ Preprocessing 정상 작동

## 📖 상세 문서

- **처음 사용:** `QUICKSTART.md` 읽기
- **YuNet 상세:** `README.md` 읽기
- **EdgeFace 상세:** `README_EDGEFACE.md` 읽기 ⭐

## 🆘 트러블슈팅

### YuNet

**"No images found"**
```bash
# 경로 확인
ls ~/datasets/lfw
find ~/datasets/lfw -name "*.jpg" | head -5
```

**"Not enough quality images"**
```bash
# Quality threshold 낮추기
./run_calibration_pipeline.sh -s ~/lfw -q 30
```

### EdgeFace

**"No faces detected"**
```bash
# 1. YuNet 모델 확인
ls -l face_detection_yunet_2023mar.onnx

# 2. Quality threshold 낮추기
./run_edgeface_calibration.sh -s ~/lfw -y yunet.onnx -q 70
```

**"YuNet model not found"**
```bash
# 다운로드
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

## 📝 체크리스트

### YuNet NPU Calibration
- [ ] LFW 데이터셋 준비
- [ ] `run_calibration_pipeline.sh` 실행
- [ ] Config 파일 생성 확인
- [ ] Input tensor name 확인 (ONNX 모델과 일치)
- [ ] NPU 컴파일

### EdgeFace NPU Calibration
- [ ] LFW 데이터셋 준비
- [ ] YuNet ONNX 모델 다운로드
- [ ] `run_edgeface_calibration.sh` 실행
- [ ] 정렬된 얼굴 이미지 확인
- [ ] Config 파일 생성 확인
- [ ] Input tensor name 확인 (ONNX 모델과 일치)
- [ ] NPU 컴파일

## 🎓 핵심 요약

1. **YuNet**: 일반 이미지로 calibration ✓
2. **EdgeFace**: 정렬된 얼굴 필요 ⚠️
3. **자동 스크립트 사용 권장**
4. **100개 샘플이 일반적으로 최적**
5. **EMA calibration method 권장**
6. **반드시 검증 수행**

## 다음 단계

Calibration config 생성 후:

```bash
# YuNet NPU 컴파일
<npu_compiler> \
  --model yunet.onnx \
  --config calibration_output/calibration_config_yunet.json \
  --output yunet.npu

# EdgeFace NPU 컴파일
<npu_compiler> \
  --model edgeface.onnx \
  --config edgeface_calibration_output/calibration_config_edgeface.json \
  --output edgeface.npu
```

컴파일 후:
- LFW evaluation으로 정확도 확인
- 원본 ONNX 모델과 비교
- 성능이 낮으면 calibration 재시도 (다른 방법, 더 많은 샘플)
