#!/bin/bash

# NPU Calibration Pipeline Runner
# 전체 calibration 프로세스를 자동으로 실행하는 스크립트

set -e  # Exit on error

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 기본 설정
LFW_DIR=""
OUTPUT_DIR="./calibration_output"
NUM_SAMPLES=100
QUALITY_THRESHOLD=40
MODEL_TYPE="yunet"
CALIBRATION_METHOD="ema"
SKIP_VALIDATION=false

# 사용법 출력
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --source-dir DIR       LFW 데이터셋 경로 (필수)"
    echo "  -o, --output-dir DIR       출력 디렉토리 (기본: ./calibration_output)"
    echo "  -n, --num-samples NUM      Calibration 샘플 수 (기본: 100)"
    echo "  -q, --quality NUM          품질 threshold 0-100 (기본: 40)"
    echo "  -m, --model-type TYPE      모델 타입: yunet|edgeface (기본: yunet)"
    echo "  -c, --calib-method METHOD  Calibration 방법: ema|minmax|kl|percentile (기본: ema)"
    echo "  --skip-validation          검증 단계 스킵"
    echo "  -h, --help                 이 도움말 출력"
    echo ""
    echo "Example:"
    echo "  $0 -s ~/datasets/lfw -m yunet -n 100"
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
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -q|--quality)
            QUALITY_THRESHOLD="$2"
            shift 2
            ;;
        -m|--model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        -c|--calib-method)
            CALIBRATION_METHOD="$2"
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

# 설정 출력
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}NPU Calibration Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Source directory: ${GREEN}$LFW_DIR${NC}"
echo -e "Output directory: ${GREEN}$OUTPUT_DIR${NC}"
echo -e "Model type: ${GREEN}$MODEL_TYPE${NC}"
echo -e "Number of samples: ${GREEN}$NUM_SAMPLES${NC}"
echo -e "Quality threshold: ${GREEN}$QUALITY_THRESHOLD${NC}"
echo -e "Calibration method: ${GREEN}$CALIBRATION_METHOD${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 출력 디렉토리 생성
mkdir -p "$OUTPUT_DIR"

DATASET_DIR="$OUTPUT_DIR/calibration_dataset"
CONFIG_FILE="$OUTPUT_DIR/calibration_config_${MODEL_TYPE}.json"
TEST_OUTPUT_DIR="$OUTPUT_DIR/validation_output"

# Step 1: Calibration 데이터셋 준비
echo -e "${YELLOW}[Step 1/3] Preparing calibration dataset...${NC}"
python prepare_calibration_dataset.py \
    --source-dir "$LFW_DIR" \
    --output-dir "$DATASET_DIR" \
    --num-samples "$NUM_SAMPLES" \
    --quality-threshold "$QUALITY_THRESHOLD" \
    --seed 42

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dataset preparation completed${NC}\n"
else
    echo -e "${RED}✗ Dataset preparation failed${NC}"
    exit 1
fi

# Step 2: Calibration config 생성
echo -e "${YELLOW}[Step 2/3] Generating calibration config...${NC}"
python generate_calibration_config.py \
    --model-type "$MODEL_TYPE" \
    --dataset-path "$DATASET_DIR" \
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
        echo -e "${RED}✗ Validation failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}[Step 3/3] Skipping validation${NC}\n"
fi

# 완료 메시지
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Calibration Pipeline Completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Output files:"
echo -e "  • Dataset: ${BLUE}$DATASET_DIR${NC}"
echo -e "  • Config:  ${BLUE}$CONFIG_FILE${NC}"
if [ "$SKIP_VALIDATION" = false ]; then
    echo -e "  • Validation: ${BLUE}$TEST_OUTPUT_DIR${NC}"
fi
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Review the calibration config: cat $CONFIG_FILE"
echo -e "  2. Verify input tensor name matches your ONNX model"
echo -e "  3. Use the config for NPU compilation:"
echo -e "     ${BLUE}<npu_compiler> --model model.onnx --config $CONFIG_FILE --output model.npu${NC}"
echo ""
