# EdgeFace - Face Alignment Benchmark

EdgeFace 모델을 기반으로 한 얼굴 정렬(Face Alignment) 방법들의 성능 비교 벤치마크 시스템입니다.

## 📋 프로젝트 구조

### 🚀 메인 실행 파일

#### LFW 벤치마크
- **`face_alignment_benchmark_gpu.py`** - GPU 최적화된 LFW 벤치마크 메인 스크립트
  - 여러 face alignment 방법을 LFW 데이터셋에서 비교 평가
  - 메모리 사용량 추적 (CPU/GPU)
  - 6가지 메트릭 시각화 (정확도, 속도, 성공률, EER, 메모리)

- **`lfw_evaluation_optimized.py`** - 배치 처리 최적화 LFW 평가 모듈
  - GPU 효율성을 위한 배치 임베딩 추출
  - 병렬 이미지 로딩 (ThreadPoolExecutor)
  - EdgeFace 모델 기반 얼굴 검증

- **`lfw_evaluation.py`** - 기본 LFW 평가 모듈 (백업용)

### 🔍 Face Alignment 모듈 (`face_alignment/`)

#### 통합 인터페이스
- **`unified_detector.py`** - 모든 face detection 방법의 통합 인터페이스
  - 단일 API로 여러 detector 사용 가능
  - 자동 의존성 체크
  - 벤치마크 기능 내장

#### Face Detectors
- **`mtcnn.py`** - MTCNN (Multi-task CNN) detector
  - PyTorch 기반
  - GPU/CPU 지원
  - 5개 랜드마크 검출

- **`yunet.py`** - YuNet detector
  - OpenCV DNN 기반
  - CUDA 자동 fallback 지원
  - 빠른 속도

- **`yolo_detector.py`** - YOLO 기반 detectors
  - YOLOv5-Face: 얼굴 특화 ONNX 모델
  - YOLOv8: Ultralytics 범용 모델
  - 높은 정확도

- **`retinaface_onnx.py`** - RetinaFace detector
  - ONNX Runtime 기반
  - 높은 검출 정확도

- **`rtmpose_detector.py`** - RTMPose detector
  - 얼굴 키포인트 검출
  - ONNX Runtime 기반

- **`mediapipe_detector.py`** - MediaPipe detectors
  - MediaPipe Face Mesh: 468 랜드마크
  - MediaPipe Face Detection: 6 랜드마크
  - 빠르고 가벼움

#### MTCNN 의존성
- **`mtcnn_pytorch/`** - MTCNN PyTorch 구현체
  - `src/detector.py` - MTCNN 핵심 detector
  - `src/align_trans.py` - 얼굴 정렬 변환
  - `src/get_nets.py` - P-Net, R-Net, O-Net 모델
  - `src/box_utils.py` - Bounding box NMS

### 🎯 원본 EdgeFace 코드

#### 모델 관련
- **`backbones/`** - EdgeFace 모델 백본 아키텍처
  - `timmfr.py` - Timm 기반 feature extractor

#### 학습 관련
- **`train_v2.py`** - EdgeFace 모델 학습 스크립트
- **`train_v2_restart.py`** - 체크포인트에서 재시작
- **`dataset.py`** - 얼굴 인식 데이터셋 로더
- **`losses.py`** - 손실 함수 (ArcFace, CosFace 등)
- **`lr_scheduler.py`** - 학습률 스케줄러
- **`partial_fc_v2.py`** - Partial FC 구현

#### 평가 관련
- **`eval/`** - 모델 평가 모듈
  - `verification.py` - 얼굴 검증 평가
- **`eval_edgeface.py`** - EdgeFace 모델 평가
- **`eval_ijbc.py`** - IJB-C 벤치마크 평가
- **`onnx_ijbc.py`** - ONNX 모델 IJB-C 평가

#### 유틸리티
- **`utils/`** - 유틸리티 함수
  - `utils_config.py` - 설정 관리
  - `utils_logging.py` - 로깅
  - `utils_callbacks.py` - 학습 콜백
  - `utils_distributed_sampler.py` - 분산 학습

#### 기타
- **`configs/`** - 학습 설정 파일들
- **`inference.py`** - 모델 추론 예제
- **`hubconf.py`** - PyTorch Hub 설정
- **`torch2onnx.py`** - PyTorch → ONNX 변환
- **`onnx_helper.py`** - ONNX 헬퍼 함수
- **`flops.py`** - FLOPs 계산

### 📦 모델 및 데이터

- **`face_alignment/models/`** - Face detection 모델 파일
  - `face_detection_yunet_2023mar.onnx` - YuNet 모델
  - `yolov8n.pt` - YOLOv8 nano 모델
  - `yolov5_face.onnx` - YOLOv5-Face 모델
  - `yolov8n-face*.onnx/pt` - YOLOv8 얼굴 모델들

- **`face_alignment/mtcnn_pytorch/data/`** - MTCNN 사전학습 가중치
  - `pnet.npy`, `rnet.npy`, `onet.npy`

- **`checkpoints/`** - EdgeFace 모델 체크포인트
  - `edgeface_xs_gamma_06.pt` - EdgeFace-XS(γ=0.6) 모델

### 📊 결과 및 아카이브

- **`benchmark_results/`** - 벤치마크 결과 파일
  - CSV 리포트
  - 시각화 차트 (PNG)

- **`alignment_results/`** - 얼굴 정렬 결과 이미지
  - 각 detector별 정렬된 얼굴 샘플

- **`archives/`** - 아카이브 파일들
  - `notebooks/` - Jupyter 노트북 실험 코드
  - `analysis_scripts/` - 일회성 분석 스크립트

## 🚀 사용 방법

### 1. 환경 설정

```bash
# PyTorch 설치 (GPU 버전 권장)
# https://pytorch.org/get-started/locally/

# 필수 패키지 설치
pip install -r requirements.txt

# 선택적: MediaPipe, YOLOv8
pip install mediapipe ultralytics
```

### 2. LFW 벤치마크 실행

```bash
python face_alignment_benchmark_gpu.py
```

**설정 수정**: `face_alignment_benchmark_gpu.py` 파일에서
- `LFW_CONFIG['lfw_dir']` - LFW 데이터셋 경로
- `LFW_CONFIG['pairs_file']` - pairs.csv 파일 경로
- `LFW_CONFIG['edgeface_model_path']` - EdgeFace 모델 경로
- `LFW_CONFIG['batch_size']` - GPU 메모리에 맞게 조정
- `LFW_CONFIG['num_workers']` - CPU 코어 수에 맞게 조정

### 3. 특정 Detector 사용하기

```python
from face_alignment.unified_detector import UnifiedFaceDetector

# Detector 초기화
detector = UnifiedFaceDetector('mtcnn', device='cuda')

# 얼굴 검출
from PIL import Image
image = Image.open('face.jpg')
bboxes, landmarks = detector.detect_faces(image)

# 얼굴 정렬
aligned_face = detector.align(image)
```

**사용 가능한 methods**:
- `mtcnn` - MTCNN
- `yunet` - YuNet
- `yolov5_face` - YOLOv5-Face
- `yolov8` - YOLOv8
- `retinaface` - RetinaFace (ONNX)
- `rtmpose` - RTMPose (ONNX)
- `mediapipe` - MediaPipe Face Mesh
- `mediapipe_simple` - MediaPipe Face Detection

## 📈 벤치마크 메트릭

1. **Accuracy** - 얼굴 검증 정확도
2. **ROC AUC** - ROC 곡선 아래 면적
3. **EER** - Equal Error Rate (낮을수록 좋음)
4. **Success Rate** - 얼굴 검출 성공률
5. **Processing Time** - 이미지당 처리 시간
6. **Memory Usage** - CPU/GPU 메모리 사용량

## 📝 주요 특징

- ✅ **7가지 face detection 방법** 통합 지원
- ✅ **GPU 최적화** 배치 처리
- ✅ **자동 메모리 추적** (CPU/GPU)
- ✅ **병렬 이미지 로딩** (ThreadPoolExecutor)
- ✅ **자동 CUDA fallback** (YuNet, ONNX Runtime)
- ✅ **통합 인터페이스** (단일 API)
- ✅ **종합 벤치마크 리포트** (CSV + 시각화)

## 🔧 트러블슈팅

### CUDA 관련
- **TensorFlow CUDA 경고**: RTX 5090 등 최신 GPU는 JIT 컴파일 사용 (정상 동작)
- **OpenCV CUDA 없음**: YuNet이 자동으로 CPU로 fallback
- **ONNX Runtime CUDA 없음**: CPU 버전 사용 중 (onnxruntime-gpu 설치 필요)

### 메모리 관련
- `batch_size` 줄이기 (기본: 64)
- `num_workers` 조정 (기본: 8)
- GPU 메모리 부족 시 일부 detector 제외

### Detector 오류
- **MTCNN**: PyTorch GPU 설치 필요
- **YuNet**: OpenCV 설치 필요
- **YOLO**: ONNX Runtime 또는 Ultralytics 필요
- **MediaPipe**: mediapipe 패키지 설치 필요

## 📄 라이선스

원본 EdgeFace 프로젝트의 라이선스를 따릅니다.

## 🙏 감사

- EdgeFace 원본 프로젝트
- MTCNN PyTorch 구현
- OpenCV YuNet
- Ultralytics YOLOv8
- MediaPipe
