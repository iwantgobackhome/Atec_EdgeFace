# EdgeFace - Face Alignment Benchmark & Real-time Recognition System

EdgeFace 모델을 기반으로 한 얼굴 정렬(Face Alignment) 방법들의 성능 비교 벤치마크 시스템 및 실시간 얼굴 인식 시스템입니다.

## 📋 프로젝트 구조

### 🚀 메인 실행 파일

#### 실시간 얼굴 인식 시스템 (NEW!)
- **`face_recognition_gui.py`** - GUI 기반 실시간 얼굴 인식 시스템 (권장)
  - Tkinter 기반 직관적인 GUI 인터페이스
  - 실시간 카메라 피드 및 얼굴 detection/recognition
  - 참조 이미지 추가/삭제 관리
  - Detection 모델 선택 (MTCNN, YuNet, YOLO 등)
  - FPS, 인물 ID, 유사도 실시간 표시

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
   - **Detector 선택**: MTCNN, YuNet, YOLO 등 선택 (YOLOv8 권장)
   - **Device 선택**: CUDA (GPU) 또는 CPU
   - **Similarity Threshold 설정**: 인식 임계값 조정 (0.0~1.0, 기본: 0.5)

2. **카메라 시작**
   - "▶ Start Camera" 버튼 클릭
   - 실시간으로 얼굴 detection 및 recognition 시작
   - FPS, 인물 ID, 유사도가 화면에 표시됨

3. **참조 이미지 등록 (2가지 방법)**

   **방법 1: 📸 카메라로 직접 캡처 (권장)**
   - 카메라 앞에서 얼굴을 정면으로 위치
   - "📸 Capture from Camera" 버튼 클릭
   - 팝업 창에 이름 입력 후 확인
   - 자동으로 얼굴 검출, 정렬, 임베딩 추출하여 저장
   - 캡처된 이미지는 `captured_references/` 폴더에 자동 저장

   **방법 2: 📁 파일에서 불러오기**
   - "➕ Add from File" 버튼 클릭
   - 이미지 파일 선택 (JPG, PNG 등)
   - 팝업 창에 이름 입력 후 확인
   - 자동으로 얼굴 검출 및 등록

4. **참조 이미지 관리**
   - 등록된 인물 목록 확인
   - 선택한 인물 삭제 가능 ("🗑️ Delete Selected" 버튼)

5. **카메라 종료**
   - "⏹ Stop Camera" 버튼 클릭

**💡 사용 팁**:
- GPU를 사용하면 처리 속도가 크게 향상됩니다 (CUDA 선택)
- Threshold가 높을수록 엄격하게 인식 (0.5~0.7 권장)
- 카메라 캡처로 등록 시 조명이 좋은 환경에서 정면 얼굴 권장
- 여러 각도/조명에서 캡처하면 인식 정확도 향상

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
2. Face detector로 얼굴 검출 및 랜드마크 추출
3. 얼굴 정렬 (Face alignment)
4. EdgeFace 모델로 512차원 임베딩 추출
5. 참조 DB와 코사인 유사도 계산
6. Threshold 이상이면 인물 ID 표시

### 7. LFW 벤치마크 실행

```bash
python face_alignment_benchmark_gpu.py
```

**설정 수정**: `face_alignment_benchmark_gpu.py` 파일에서
- `LFW_CONFIG['lfw_dir']` - LFW 데이터셋 경로
- `LFW_CONFIG['pairs_file']` - pairs.csv 파일 경로
- `LFW_CONFIG['edgeface_model_path']` - EdgeFace 모델 경로
- `LFW_CONFIG['batch_size']` - GPU 메모리에 맞게 조정
- `LFW_CONFIG['num_workers']` - CPU 코어 수에 맞게 조정

### 8. 특정 Detector 사용하기 (Python API)

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

### 실시간 얼굴 인식 시스템
- ✅ **GUI 기반 사용자 친화적 인터페이스** (Tkinter)
- ✅ **실시간 얼굴 detection 및 recognition**
- ✅ **참조 이미지 관리** (추가/삭제)
  - 📸 **카메라 직접 캡처**: 실시간으로 얼굴 촬영하여 등록
  - 📁 **파일에서 불러오기**: 기존 이미지 파일 선택
- ✅ **다중 detector 선택** (MTCNN, YuNet, YOLO 등)
- ✅ **EdgeFace 기반 얼굴 임베딩 추출**
- ✅ **코사인 유사도 기반 검증**
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
