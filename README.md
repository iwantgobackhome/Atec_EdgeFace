# EdgeFace - Face Alignment Benchmark & Real-time Recognition System

EdgeFace 모델을 기반으로 한 얼굴 정렬(Face Alignment) 방법들의 성능 비교 벤치마크 시스템 및 실시간 얼굴 인식 시스템입니다.

## 📋 프로젝트 구조

### 🚀 메인 실행 파일

#### NPU Calibration 시스템 (NEW! 🔥)
- **`npu_calibration/`** - YuNet/EdgeFace ONNX → NPU 컴파일용 calibration 시스템
  - **자동화된 고품질 calibration 데이터셋 생성**
  - **YuNet용**: 일반 이미지 선택 및 품질 분석
  - **EdgeFace용**: YuNet으로 얼굴 정렬 후 calibration 데이터 생성
  - **NPU calibration config JSON 자동 생성**
  - 4가지 calibration 방법 지원 (EMA, MinMax, KL, Percentile)
  - 전처리 파이프라인 검증 도구 포함
  - [상세 가이드 보기](npu_calibration/README.md)

#### ONNX 변환 및 평가 (NEW! 🔥)
- **`edgeface_to_onnx_with_lfw_evaluation.ipynb`** - EdgeFace PyTorch → ONNX 변환 및 LFW 평가
  - PyTorch 모델을 ONNX 형식으로 변환
  - NPU calibration config와 일치하는 입력 이름 사용 (input.1)
  - ONNX 모델 검증 및 출력 비교
  - LFW 데이터셋으로 PyTorch vs ONNX 성능 비교
  - ROC 커브, 정확도, EER, 처리 속도 비교 분석
  - 유사도 분포 시각화
  - 한국어 설명 포함

#### 실시간 얼굴 인식 시스템 (NEW! NPU 지원 🚀)
- **`face_recognition_gui.py`** - GUI 기반 실시간 얼굴 인식 시스템 (권장)
  - Tkinter 기반 직관적인 GUI 인터페이스
  - 실시간 카메라 피드 및 얼굴 detection/recognition
  - 참조 이미지 추가/삭제 관리
  - Detection 모델 선택 (MTCNN, YuNet, **YuNet NPU**, YOLO 등)
  - **DeepX NPU 가속 지원** - CPU/GPU/NPU 선택 가능
  - FPS, 인물 ID, 유사도 실시간 표시
  - 📖 [NPU 빠른 시작 가이드](QUICKSTART_NPU.md) | [상세 NPU 통합 문서](NPU_INTEGRATION.md)

#### NPU 통합 모듈 (NEW! 🚀)
- **`face_alignment/yunet_npu.py`** - YuNet NPU 얼굴 검출기
  - DeepX NPU SDK 기반 YuNet detector
  - 640x640 입력, 5-point 랜드마크 검출
  - ✅ Multi-scale FPN 출력 디코딩 완료
  - ✅ Anchor-based bbox/landmark decoding 구현

- **`edgeface_npu_recognizer.py`** - EdgeFace NPU 인식기
  - DeepX NPU SDK 기반 EdgeFace recognizer
  - 112x112 입력, 512차원 임베딩 출력
  - ✅ L2 정규화된 임베딩 출력 확인 완료

- **`test_npu_inference.py`** - NPU 모델 통합 테스트
  - YuNet과 EdgeFace NPU 모델 동시 테스트
  - 출력 shape, 통계, L2 norm 분석
  - 임베디드 보드에서 실행 권장

- **`face_recognition_system.py`** - 커맨드라인 기반 얼굴 인식 시스템
  - EdgeFace 기반 얼굴 임베딩 추출
  - 코사인 유사도 기반 얼굴 검증
  - 참조 이미지 데이터베이스 관리
  - 실시간 카메라 처리

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

- **`yunet_npu.py`** - YuNet NPU detector (NEW! 🚀)
  - DeepX NPU SDK 기반
  - 임베디드 보드 최적화
  - 640x640 입력, 5-point 랜드마크
  - ✅ 출력 디코딩 완료 (anchor-based detection)

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

## 🚀 빠른 시작 (Quick Start)

### 1. 가상환경 생성 및 활성화

Python 3.10 권장 (3.8 이상 지원)

```bash
# Conda 사용 시
conda create -n edgeface python=3.10
conda activate edgeface

# venv 사용 시
python -m venv edgeface_env
source edgeface_env/bin/activate  # Linux/Mac
# edgeface_env\Scripts\activate  # Windows
```

### 2. PyTorch 설치 (GPU 버전 권장)

**본인의 CUDA 버전에 맞게 설치**하세요. GPU 벤치마크 실행을 위해서는 CUDA 지원 버전 필수입니다.

```bash
# (https://pytorch.org/get-started/locally/ 참고)
# CUDA 12.8 예시
pip3 install torch torchvision

# CUDA 12.6 예시
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CUDA 12.1 예시
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 예시
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU만 사용하는 경우 (권장하지 않음)
pip install torch torchvision torchaudio
```

**설치 확인**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 3. 필수 패키지 설치

```bash
pip install -r requirements.txt
```

**requirements.txt에 포함된 주요 패키지**:
- `numpy`, `opencv-python`, `Pillow` - 이미지 처리
- `onnxruntime-gpu` - ONNX 모델 실행 (GPU 지원, YOLO detector 필수)
- `ultralytics` - YOLOv8 detector (필수)
- `pandas`, `matplotlib`, `seaborn` - 데이터 분석 및 시각화
- `scikit-learn` - 평가 메트릭
- `tqdm`, `psutil` - 진행 표시 및 메모리 모니터링

### 4. 선택적 패키지 설치

```bash
# MediaPipe detector를 사용하려는 경우
pip install mediapipe
```

### 5. LFW 데이터셋 및 모델 준비

벤치마크 실행 전에 필요한 파일들을 준비하세요:

1. **LFW 데이터셋 다운로드**: [LFW-deepfunneled](http://vis-www.cs.umass.edu/lfw/)
2. **EdgeFace 모델**: `checkpoints/edgeface_xs_gamma_06.pt`
3. **Face detection 모델들**: `face_alignment/models/` 디렉토리

### 6. LFW 벤치마크 실행

```bash
python face_alignment_benchmark_gpu.py
```

**설정 수정**: `face_alignment_benchmark_gpu.py` 파일에서
- `LFW_CONFIG['lfw_dir']` - LFW 데이터셋 경로
- `LFW_CONFIG['pairs_file']` - pairs.csv 파일 경로
- `LFW_CONFIG['edgeface_model_path']` - EdgeFace 모델 경로
- `LFW_CONFIG['batch_size']` - GPU 메모리에 맞게 조정
- `LFW_CONFIG['num_workers']` - CPU 코어 수에 맞게 조정

### 6. 실시간 얼굴 인식 시스템 실행 (NEW!)

#### 🎥 GUI 버전 (권장)

EdgeFace 기반 실시간 얼굴 인식 시스템을 GUI로 편리하게 사용할 수 있습니다.

```bash
# GUI 기반 실시간 얼굴 인식 시스템 실행
python face_recognition_gui.py
```

**📖 GUI 사용법**:

1. **시스템 설정**
   - **Detector 선택**: MTCNN, YuNet, **YuNet NPU**, YOLO 등 선택
     - YOLOv8 권장 (GPU)
     - YuNet NPU 권장 (임베디드 보드)
   - **Device 선택**:
     - **CUDA (GPU)** - PC/워크스테이션
     - **CPU** - GPU 없는 환경
     - **NPU** - DeepX NPU 탑재 임베디드 보드 (🚀 최적화)
   - **Similarity Threshold 설정**: 인식 임계값 조정 (0.0~1.0, 기본: 0.5)

   **💡 NPU 사용 시**:
   - Device를 "NPU"로 선택하면 자동으로 YuNet NPU로 전환
   - EdgeFace도 자동으로 NPU 모델 사용 (.dxnn 파일)
   - 📖 [NPU 빠른 시작](QUICKSTART_NPU.md) | [NPU 통합 문서](NPU_INTEGRATION.md)

2. **카메라 시작**
   - "▶ Start Camera" 버튼 클릭
   - 실시간으로 얼굴 detection 및 recognition 시작
   - FPS, 인물 ID, 유사도가 화면에 표시됨

3. **참조 이미지 등록 (3가지 방법)**

   **방법 1: 📸 다각도 캡처 (권장 - 가장 정확!)**
   - "📸 Capture Multi-Angle" 버튼 클릭
   - 팝업 창에 이름 입력 후 확인
   - 화면 지시에 따라 얼굴을 천천히 움직이기:
     - ✅ **정면** (Front) - 카메라를 똑바로 보기
     - ✅ **좌** (Left) - 얼굴을 왼쪽으로 천천히 돌리기
     - ✅ **우** (Right) - 얼굴을 오른쪽으로 천천히 돌리기
     - ✅ **위** (Up) - 얼굴을 위로 올리기
     - ✅ **하** (Down) - 얼굴을 아래로 내리기
   - 각 각도가 자동으로 감지되면 자동 캡처됨
   - 진행 상황이 화면에 실시간 표시됨 (예: 3/5 완료)
   - 5가지 각도 모두 캡처 완료 시 자동으로 등록
   - 캡처된 이미지는 `captured_references/{이름}/` 폴더에 각도별로 저장

   **방법 2: 📁 파일에서 불러오기**
   - "➕ Add from File" 버튼 클릭
   - 이미지 파일 선택 (JPG, PNG 등)
   - 팝업 창에 이름 입력 후 확인
   - 자동으로 얼굴 검출 및 등록 (정면 각도로 저장)

4. **참조 이미지 관리**
   - 등록된 인물 목록에서 선택 시 캡처된 각도 정보 표시
   - "➖ Remove Person" - 선택한 인물 전체 삭제
   - "🗑️ Manage Angles" - 특정 각도만 선택 삭제 가능

5. **카메라 종료**
   - "⏹ Stop Camera" 버튼 클릭

**💡 사용 팁**:
- **GPU 사용 권장**: 처리 속도가 크게 향상됩니다 (CUDA 선택)
- **다각도 캡처 활용**: 5가지 각도를 모두 캡처하면 인식 정확도가 크게 향상됩니다
- **Threshold 조정**: 높을수록 엄격하게 인식 (0.5~0.7 권장)
- **조명 주의**: 밝고 균일한 조명에서 캡처하면 인식률 향상
- **다중 인물 인식**: 2명 이상이 동시에 화면에 있어도 안정적으로 인식됩니다
- **안정적인 추적**: IoU 기반 추적으로 프레임 간 identity 깜빡임 방지

#### ⌨️ 커맨드라인 버전

```bash
# 1. 파일에서 참조 이미지 추가
python face_recognition_system.py --add-ref path/to/person.jpg "Person Name"

# 2. 실시간 카메라 인식 실행
python face_recognition_system.py --detector mtcnn --device cuda

# 3. 실행 중 키보드 단축키 사용
#    - 'c' 키: 카메라에서 직접 캡처하여 참조 이미지 등록
#    - 'q' 키: 프로그램 종료

# 4. 옵션 커스터마이징
python face_recognition_system.py \
  --detector yolov8 \
  --device cuda \
  --threshold 0.6 \
  --camera 0
```

**주요 옵션**:
- `--detector`: 얼굴 detection 방법 (mtcnn, yunet, yolov5_face, yolov8)
- `--device`: 실행 디바이스 (cuda, cpu)
- `--threshold`: 유사도 임계값 (기본: 0.5)
- `--camera`: 카메라 ID (기본: 0, 웹캠)
- `--add-ref IMAGE ID`: 파일에서 참조 이미지 추가

**실행 중 키보드 단축키**:
- `q`: 프로그램 종료
- `c`: 현재 프레임을 캡처하여 참조 이미지로 등록

**📦 참조 데이터베이스**:
- 참조 이미지는 `reference_db.pkl` 파일에 저장됨
- GUI 또는 커맨드라인에서 추가/삭제 가능
- 얼굴 임베딩은 EdgeFace 모델로 추출하여 저장
- 캡처된 이미지는 `captured_references/` 폴더에 자동 저장

**🎯 인식 프로세스**:
1. 카메라에서 프레임 읽기
2. Face detector로 얼굴 검출 및 랜드마크 추출 (동시에 여러 얼굴)
3. 얼굴 정렬 (Face alignment) - 모든 검출된 얼굴
4. **배치 처리**: EdgeFace 모델로 512차원 임베딩 동시 추출 (GPU 최적화)
5. 참조 DB와 코사인 유사도 계산 (모든 각도 비교)
6. **Greedy Assignment**: 각 얼굴에 최적의 인물 매칭
7. **Face Tracking**: IoU 기반으로 이전 프레임과 연결하여 일관성 유지
8. Threshold 이상이면 인물 ID 표시

**⚡ 성능 최적화**:
- **배치 임베딩 추출**: 여러 얼굴을 한 번에 처리하여 ~2배 빠름
- **Face Tracking**: 10프레임까지 추적하여 깜빡임 제거
- **우선순위 매칭**: 추적된 얼굴은 threshold 0.8배로 낮춰 안정성 향상
- **다각도 DB**: 5가지 각도 저장으로 다양한 포즈에서 높은 인식률

### 7. NPU 모델 테스트 (임베디드 보드)

DeepX NPU 탑재 임베디드 보드에서 NPU 모델을 테스트할 수 있습니다.

```bash
# NPU 통합 테스트 (EdgeFace + YuNet)
python test_npu_inference.py
```

**테스트 내용**:
- EdgeFace NPU: 512차원 임베딩 출력 검증
- YuNet NPU: 얼굴 검출 출력 형식 분석
- 각 모델의 출력 shape, 통계, L2 norm 확인

**현재 상태**:
- ✅ EdgeFace NPU: 정상 동작 확인 (L2 norm = 1.0)
- ✅ YuNet NPU: 출력 디코딩 완료, 정상 동작 확인

### 8. LFW 벤치마크 실행

```bash
python face_alignment_benchmark_gpu.py
```

**설정 수정**: `face_alignment_benchmark_gpu.py` 파일에서
- `LFW_CONFIG['lfw_dir']` - LFW 데이터셋 경로
- `LFW_CONFIG['pairs_file']` - pairs.csv 파일 경로
- `LFW_CONFIG['edgeface_model_path']` - EdgeFace 모델 경로
- `LFW_CONFIG['batch_size']` - GPU 메모리에 맞게 조정
- `LFW_CONFIG['num_workers']` - CPU 코어 수에 맞게 조정

### 9. 특정 Detector 사용하기 (Python API)

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
- `yunet` - YuNet (OpenCV DNN)
- `yunet_npu` - YuNet NPU (DeepX NPU) ✅ 사용 가능
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

### NPU Calibration 시스템 (NEW! 🔥)
- ✅ **ONNX → NPU 컴파일을 위한 완전 자동화된 calibration 시스템**
- ✅ **YuNet (Detection)**: 일반 이미지로 calibration
  - 5가지 품질 지표로 자동 이미지 선택 (선명도, 밝기, 대비, 해상도, 색상)
  - 다양성 보장 샘플링 전략
- ✅ **EdgeFace (Recognition)**: 정렬된 얼굴로 calibration
  - YuNet으로 자동 얼굴 detection + alignment
  - 112x112 정렬된 얼굴 이미지 생성
  - ArcFace 전처리 파이프라인 적용
- ✅ **NPU calibration config JSON 자동 생성**
  - 모델별 최적화된 전처리 설정
  - BGR/RGB 변환, 정규화 자동 처리
- ✅ **4가지 calibration 방법**: EMA (권장), MinMax, KL, Percentile
- ✅ **검증 도구**: 전처리 파이프라인 테스트 및 시각화
- ✅ **완전 자동화 스크립트**: 한 줄 명령어로 전체 파이프라인 실행
- 📖 [NPU Calibration 가이드](npu_calibration/README.md)

### ONNX 변환 및 평가 (NEW! 🔥)
- ✅ **EdgeFace PyTorch → ONNX 변환**
  - ONNX opset 11 사용
  - NPU calibration config와 일치하는 입력 이름 (input.1)
  - 동적 배치 크기 지원
  - 모델 검증 및 출력 비교 (코사인 유사도, 차이 계산)
- ✅ **LFW 데이터셋 성능 평가**
  - PyTorch 모델 vs ONNX 모델 비교
  - ROC AUC, 정확도, EER, 처리 속도 측정
  - 유사도 분포 비교 시각화
- ✅ **Jupyter Notebook 형식**
  - 단계별 상세한 한국어 설명
  - 인터랙티브 실행 및 시각화
  - 성능 비교 리포트 자동 생성

### 실시간 얼굴 인식 시스템
- ✅ **GUI 기반 사용자 친화적 인터페이스** (Tkinter)
- ✅ **실시간 얼굴 detection 및 recognition**
- ✅ **참조 이미지 관리** (추가/삭제)
  - 📸 **카메라 직접 캡처**: 실시간으로 얼굴 촬영하여 등록
  - 🔄 **다각도 얼굴 캡처**: 정면/좌/우/위/아래 5가지 각도 자동 캡처 (NEW!)
  - 📁 **파일에서 불러오기**: 기존 이미지 파일 선택
- ✅ **다중 detector 선택** (MTCNN, YuNet, YOLO 등)
- ✅ **EdgeFace 기반 얼굴 임베딩 추출**
- ✅ **코사인 유사도 기반 검증**
- ✅ **배치 임베딩 추출**: 여러 얼굴을 동시에 처리하여 성능 향상 (NEW!)
- ✅ **얼굴 추적 (Face Tracking)**: IoU 기반 프레임 간 일관성 유지 (NEW!)
- ✅ **안정적인 다중 인물 인식**: 2명 이상도 깜빡임 없이 안정적 인식 (NEW!)
- ✅ **FPS, 인물 ID, 유사도 실시간 표시**
- ✅ **GPU 가속 지원**
- ✅ **캡처된 이미지 자동 저장** (captured_references/)

### LFW 벤치마크 시스템
- ✅ **7가지 face detection 방법** 통합 지원
- ✅ **GPU 최적화** 배치 처리
- ✅ **자동 메모리 추적** (CPU/GPU)
- ✅ **병렬 이미지 로딩** (ThreadPoolExecutor)
- ✅ **자동 CUDA fallback** (YuNet, ONNX Runtime)
- ✅ **통합 인터페이스** (단일 API)
- ✅ **종합 벤치마크 리포트** (CSV + 시각화)

## 🔧 YuNet NPU 상세 가이드

### YuNet NPU 출력 구조

YuNet ONNX 모델을 DeepX NPU로 컴파일하면 **출력 구조가 변경**됩니다:

#### 원본 ONNX 모델 출력 (12개)
- Feature Pyramid Network (FPN) 기반 3-scale detection
- 각 scale당 4가지 출력: cls, obj, bbox, kps
- 출력 이름: `cls_8`, `cls_16`, `cls_32`, `obj_8`, `obj_16`, `obj_32`, `bbox_8`, `bbox_16`, `bbox_32`, `kps_8`, `kps_16`, `kps_32`
- **Post-processing 포함**: NMS, bbox decoding 등이 ONNX 모델 내부에 구현됨

#### NPU 컴파일된 모델 출력 (13개)
- DeepX NPU 컴파일러가 **출력 순서를 재배열**
- **Post-processing 제거**: NMS, bbox decoding이 제거되어 raw feature map 출력
- 13개 텐서 출력 (spatial feature map 1개 + FPN outputs 12개)

```python
# NPU 출력 매핑 예시 (user's compiled version)
Output 0:  (1, 1, 80, 80)    # Spatial feature map (unused)
Output 1:  (1, 6400, 1)      # cls_8 (stride 8, 80x80 = 6400 anchors)
Output 2:  (1, 1600, 1)      # cls_16 (stride 16, 40x40 = 1600 anchors)
Output 3:  (1, 6400, 4)      # bbox_8
Output 4:  (1, 1600, 4)      # bbox_16
Output 5:  (1, 1600, 1)      # obj_16
Output 6:  (1, 400, 10)      # kps_32 (stride 32, 20x20 = 400 anchors)
Output 7:  (1, 6400, 10)     # kps_8
Output 8:  (1, 1600, 10)     # kps_16
Output 9:  (1, 400, 1)       # obj_32
Output 10: (1, 400, 4)       # bbox_32
Output 11: (1, 6400, 1)      # obj_8
Output 12: (1, 400, 1)       # cls_32
```

**⚠️ 중요**: NPU 컴파일러 버전이나 설정에 따라 출력 순서가 달라질 수 있습니다!

### YuNet NPU 디코딩 구현

NPU 모델의 raw 출력을 bbox/landmark로 변환하는 과정:

#### 1. Score 계산
```python
# YuNet은 cls × obj를 final confidence로 사용
score = cls_score * obj_score
```

#### 2. Anchor 기반 Bbox 디코딩
```python
# Anchor 위치 계산 (grid 좌상단 기준, 0.5 offset 없음)
feat_size = input_size // stride  # 640/32 = 20 for stride 32
anchor_y = idx // feat_size
anchor_x = idx % feat_size

# Center 디코딩 (offset scaling)
cx = (anchor_x + bbox[0] * 0.5) * stride
cy = (anchor_y + bbox[1] * 0.5) * stride

# Size 디코딩 (linear scaling with prior)
prior_size = stride * 3  # 또는 stride * 4
w = bbox[2] * prior_size
h = bbox[3] * prior_size

# Top-left corner 형식으로 변환
x = cx - w / 2
y = cy - h / 2
```

#### 3. Landmark 디코딩
```python
# Landmarks도 anchor 기준 offset
for i in range(5):  # 5 keypoints
    lm_x = (anchor_x + lms[i*2] * 1.0) * stride
    lm_y = (anchor_y + lms[i*2 + 1] * 1.0) * stride
```

#### 4. Multi-scale Detection
```python
# 3개 scale의 detection을 모두 수집
detections_stride8 = process_scale(..., stride=8)   # 80x80 feature map
detections_stride16 = process_scale(..., stride=16) # 40x40 feature map
detections_stride32 = process_scale(..., stride=32) # 20x20 feature map

# 모든 detection 합치기
all_detections = detections_stride8 + detections_stride16 + detections_stride32
```

#### 5. NMS (Non-Maximum Suppression)
```python
# 중복 detection 제거
final_detections = apply_nms(all_detections, iou_threshold=0.3)
```

### Bbox/Landmark 조정 방법

Detection 결과가 정확하지 않을 때 조정하는 방법:

#### 파일 위치
`face_alignment/yunet_npu.py`의 `_process_scale()` 메서드 (라인 ~295-325)

#### 조정 파라미터

1. **Bbox 크기 조정** (`prior_size`)
   ```python
   prior_size = stride * 3  # 이 값을 조정
   # 크게: stride * 4, stride * 5
   # 작게: stride * 2, stride * 1.5
   ```

2. **Bbox width/height 비율 조정**
   ```python
   # 정사각형이 아닌 얼굴 비율 적용
   prior_w = stride * 3    # width
   prior_h = stride * 4    # height (더 길게)
   w = bbox[2] * prior_w
   h = bbox[3] * prior_h
   ```

3. **Bbox center offset 조정**
   ```python
   # 위치 shift 조정 (현재 0.5)
   cx = (anchor_x + bbox[0] * 0.5) * stride  # 0.5를 0.3~1.0 사이로 조정
   cy = (anchor_y + bbox[1] * 0.5) * stride

   # 우하단으로 치우치면: 계수를 줄임 (0.3, 0.4)
   # 좌상단으로 치우치면: 계수를 늘림 (0.7, 0.8)
   ```

4. **Landmark offset 조정**
   ```python
   # 현재 1.0 계수 사용
   lm_x = (anchor_x + lms[i*2] * 1.0) * stride  # 1.0을 조정
   lm_y = (anchor_y + lms[i*2 + 1] * 1.0) * stride

   # Landmark가 bbox center와 함께 움직이지 않으면 bbox center 기준으로:
   lm_x = cx + lms[i*2] * w  # bbox 크기 기준
   lm_y = cy + lms[i*2 + 1] * h
   ```

5. **Anchor 0.5 offset**
   ```python
   # 현재: anchor는 grid 좌상단 (0.5 offset 없음)
   anchor_y = (idx // feat_size)
   anchor_x = (idx % feat_size)

   # Grid center를 사용하려면:
   anchor_y = (idx // feat_size) + 0.5
   anchor_x = (idx % feat_size) + 0.5
   ```

#### Debug 출력 활용

코드 실행 시 다음과 같은 debug 정보가 출력됩니다:

```
[DEBUG] Scale 3 (stride 32): score range [0.000000, 0.810580], mean=0.015924
[DEBUG]   Stride 32: 7/400 above threshold 0.6
[DEBUG]     First valid score: 0.7965272068977356
[DEBUG]     First valid bbox raw: [0.9806061 1.3796387 1.3396912 1.7766724]
[DEBUG]     First valid landmark raw: [-0.03892517  0.46404266  1.4368286 ...]
```

이 값들을 보면서:
- `bbox raw` 값이 크면 (>2.0) → offset scaling 줄이기
- `bbox[2]`, `bbox[3]` (width/height scale) → 1.0~2.0 사이면 linear, >3.0이면 exponential 고려
- Landmark 값의 범위 → bbox와 비슷한 스케일이면 같은 방식 적용

### NPU 컴파일 시 주의사항

1. **출력 순서 확인 필수**
   - 다른 버전으로 컴파일하면 출력 순서가 다를 수 있음
   - `test_npu_inference.py` 실행해서 shape 확인
   - `_decode_outputs()` 메서드에서 output 매핑 수정

2. **Post-processing 제거됨**
   - ONNX 모델의 NMS, bbox decoding이 제거됨
   - Python에서 직접 구현 필요

3. **Calibration 데이터 중요**
   - YuNet은 다양한 얼굴 크기/각도 이미지로 calibration
   - `npu_calibration/` 시스템 활용 권장

4. **입력 전처리**
   - YuNet: RGB, HWC format, uint8
   - 640x640 입력 크기

## 🔧 트러블슈팅

### WSL 카메라 연결 설정

WSL2에서 USB 카메라를 사용하려면 Windows의 PowerShell에서 USB 장치를 WSL로 연결해야 합니다.

**1. PowerShell에서 USB 장치 확인** (관리자 권한 필요)
```powershell
usbipd list
```

**2. 카메라 연결**
```powershell
# busid는 위에서 확인한 카메라의 BUSID로 대체
usbipd attach --wsl --busid <busid>
```

**3. 연결 확인**
```powershell
usbipd list
# "Attached" 상태인지 확인
```

**4. WSL에서 카메라 인식 확인**
```bash
# USB 장치 목록 확인
lsusb

# 비디오 장치 확인
ll /dev/video*
```

카메라가 정상적으로 연결되면 `/dev/video0` 등의 장치가 표시됩니다.

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
- **YOLO**: Ultralytics 및 ONNX Runtime GPU 필수 (`pip install ultralytics onnxruntime-gpu`)
- **MediaPipe**: mediapipe 패키지 설치 필요

### 패키지 설치 순서 요약

완전히 새로운 환경에서 시작하는 경우:

```bash
# 1. 가상환경 생성 및 활성화
conda create -n edgeface python=3.10
conda activate edgeface

# 2. PyTorch 설치 (CUDA 버전에 맞게)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. requirements.txt 설치
pip install -r requirements.txt

# 4. (선택) MediaPipe
pip install mediapipe

# 5. 설치 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('Ultralytics OK')"
```

## 📄 라이선스

원본 EdgeFace 프로젝트의 라이선스를 따릅니다.

## 🙏 감사

- EdgeFace 원본 프로젝트 (https://github.com/otroshi/edgeface)
- OpenCV YuNet
- Ultralytics YOLOv8
- MediaPipe
