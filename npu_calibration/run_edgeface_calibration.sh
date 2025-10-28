#!/bin/bash

# EdgeFace NPU Calibration Pipeline
# EdgeFace는 정렬된 얼굴 이미지를 입력받으므로,
# YuNet으로 detection+alignment를 먼저 수행해야 합니다.

set -e  # Exit on error

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 기본 설정
LFW_DIR=""
OUTPUT_DIR="./edgeface_calibration_output"
YUNET_MODEL="../face_alignment/models/face_detection_yunet_2023mar.onnx"
NUM_SAMPLES=100
QUALITY_THRESHOLD=80 
CALIBRATION_METHOD="ema"
DEVICE="cpu"
SKIP_VALIDATION=false

# 사용법 출력
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --source-dir DIR       LFW 데이터셋 경로 (필수)"
    echo "  -o, --output-dir DIR       출력 디렉토리 (기본: ./edgeface_calibration_output)"
    echo "  -y, --yunet-model PATH     YuNet 모델 경로 (기본: ../face_alignment/models/face_detection_yunet_2023mar.onnx)"
    echo "  -n, --num-samples NUM      Calibration 샘플 수 (기본: 100)"
    echo "  -q, --quality NUM          Detection confidence threshold 0-100 (기본: 80)"
    echo "  -c, --calib-method METHOD  Calibration 방법: ema|minmax|kl|percentile (기본: ema)"
    echo "  -d, --device DEVICE        YuNet 실행 장치: cpu|cuda (기본: cpu)"
    echo "  --skip-validation          검증 단계 스킵"
    echo "  -h, --help                 이 도움말 출력"
    echo ""
    echo "Example:"
    echo "  $0 -s ~/datasets/lfw -y ./yunet.onnx -n 100"
    echo ""
    echo "Note: YuNet 모델이 필요합니다. 다운로드:"
    echo "  wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    exit 1
}

# 파라미터 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--source-dir)
            LFW_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -y|--yunet-model)
            YUNET_MODEL="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -q|--quality)
            QUALITY_THRESHOLD="$2"
            shift 2
            ;;
        -c|--calib-method)
            CALIBRATION_METHOD="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            ;;
    esac
done

# 필수 파라미터 체크
if [ -z "$LFW_DIR" ]; then
    echo -e "${RED}Error: --source-dir is required${NC}"
    usage
fi

if [ ! -d "$LFW_DIR" ]; then
    echo -e "${RED}Error: Source directory does not exist: $LFW_DIR${NC}"
    exit 1
fi

if [ ! -f "$YUNET_MODEL" ]; then
    echo -e "${RED}Error: YuNet model not found: $YUNET_MODEL${NC}"
    echo ""
    echo "Download YuNet model:"
    echo "  wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    exit 1
fi

# 설정 출력
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}EdgeFace NPU Calibration Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Source directory: ${GREEN}$LFW_DIR${NC}"
echo -e "Output directory: ${GREEN}$OUTPUT_DIR${NC}"
echo -e "YuNet model: ${GREEN}$YUNET_MODEL${NC}"
echo -e "Number of samples: ${GREEN}$NUM_SAMPLES${NC}"
echo -e "Detection quality threshold: ${GREEN}$QUALITY_THRESHOLD${NC}"
echo -e "Calibration method: ${GREEN}$CALIBRATION_METHOD${NC}"
echo -e "Device: ${GREEN}$DEVICE${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

ALIGNED_FACES_DIR="$OUTPUT_DIR/aligned_faces"
CONFIG_FILE="$OUTPUT_DIR/calibration_config_edgeface.json"
TEST_OUTPUT_DIR="$OUTPUT_DIR/validation_output"

# Step 1: 얼굴 detection + alignment
echo -e "${YELLOW}[Step 1/3] Extracting and aligning faces with YuNet...${NC}"
echo "This step detects faces and aligns them to 112x112 for EdgeFace"

python prepare_aligned_faces.py \
    --source-dir "$LFW_DIR" \
    --output-dir "$ALIGNED_FACES_DIR" \
    --yunet-model "$YUNET_MODEL" \
    --num-samples "$NUM_SAMPLES" \
    --quality-threshold "$QUALITY_THRESHOLD" \
    --device "$DEVICE" \
    --seed 42

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Face alignment completed${NC}\n"
else
    echo -e "${RED}✗ Face alignment failed${NC}"
    exit 1
fi

# Step 2: EdgeFace calibration config 생성
echo -e "${YELLOW}[Step 2/3] Generating EdgeFace calibration config...${NC}"

python generate_calibration_config.py \
    --model-type edgeface \
    --dataset-path "$ALIGNED_FACES_DIR" \
    --output-path "$CONFIG_FILE" \
    --calibration-num "$NUM_SAMPLES" \
    --calibration-method "$CALIBRATION_METHOD"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Config generation completed${NC}\n"
else
    echo -e "${RED}✗ Config generation failed${NC}"
    exit 1
fi

# Step 3: 검증 (선택사항)
if [ "$SKIP_VALIDATION" = false ]; then
    echo -e "${YELLOW}[Step 3/3] Validating calibration setup...${NC}"
    python test_calibration.py \
        --config "$CONFIG_FILE" \
        --num-samples 5 \
        --visualize \
        --output-dir "$TEST_OUTPUT_DIR"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Validation completed${NC}\n"
    else
        echo -e "${RED}✗ Validation failed (continuing anyway)${NC}\n"
    fi
else
    echo -e "${YELLOW}[Step 3/3] Skipping validation${NC}\n"
fi

# 완료 메시지
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}EdgeFace Calibration Pipeline Completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Output files:"
echo -e "  • Aligned faces: ${BLUE}$ALIGNED_FACES_DIR${NC}"
echo -e "  • Config file:   ${BLUE}$CONFIG_FILE${NC}"
if [ "$SKIP_VALIDATION" = false ]; then
    echo -e "  • Validation:    ${BLUE}$TEST_OUTPUT_DIR${NC}"
fi
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Review aligned faces: ls $ALIGNED_FACES_DIR"
echo -e "  2. Check calibration config: cat $CONFIG_FILE"
echo -e "  3. Verify input tensor name matches your EdgeFace ONNX model"
echo -e "  4. Compile EdgeFace to NPU:"
echo -e "     ${BLUE}<npu_compiler> --model edgeface.onnx --config $CONFIG_FILE --output edgeface.npu${NC}"
echo ""
echo -e "${YELLOW}Important Notes:${NC}"
echo -e "  • Aligned faces are 112x112 RGB images"
echo -e "  • Preprocessing includes ArcFace normalization (mean=0.5, std=0.5)"
echo -e "  • All faces passed YuNet detection with confidence >= $QUALITY_THRESHOLD"
echo ""
