#!/bin/bash

# Script to run the model with configurable parameters


# Default parameters
#MODEL="cGAN"
#NUM_CLASSES=8
#PAR_PATH="./experiments/log/20250820-105330/Generation.yaml"
#GEN_ROOT="./data/fake_figure/"
#REAL_ROOT="./data/ODIR/ODIR-5K/Training Images"
#CLASSIFIER_PATH="./experiments/log/20250819-224014/checkpoint_77.pth"
#MODEL_PATH="./experiments/log/20250820-105330/model_epoch_184.pth"
#DEVICE="cuda:4"
#LATENT_DIM=100

MODEL="ddim"
NUM_CLASSES=8
PAR_PATH="./experiments/log/20250820-105330/Generation.yaml" #This par aims to get data_path and you can change this to any other Generation.yaml if its dataset is 'ODIR' .
GEN_ROOT="./data/fake_figure/"
REAL_ROOT="./data/ODIR/ODIR-5K/Training Images"
CLASSIFIER_PATH="/home/xukaijie/ODIR_project/checkpoints/resnet_cifar_10_ODIR.pth"
MODEL_PATH="/home/xukaijie/ODIR_project/checkpoints/ddim.pt"
DEVICE="cuda:3"
LATENT_DIM=100


# Function to display usage
usage() {
    echo "Usage: $0 --model [cGAN|ddim|vqvae] [OPTIONS]"
    echo ""
    echo "Required parameters:"
    echo "  --model MODEL_NAME      Model name (cGAN, ddim, or vqvae)"
    echo ""
    echo "Optional parameters:"
    echo "  --num_classes NUM       Number of classes (default: 8)"
    echo "  --par_path PATH         Parameter configuration file path"
    echo "  --gen_root PATH         Generated data save path"
    echo "  --real_root PATH        Real data directory path"
    echo "  --classifier_path PATH  Classifier model path"
    echo "  --model_path PATH       Main model path"
    echo "  --device DEVICE         Device specification (e.g., cuda:0, cpu)"
    echo "  --latent_dim DIM        Latent dimension size (default: 100)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --model cGAN --device cuda:4 --num_classes 8"
    echo "  $0 --model ddim --gen_root ./results/ddim/ --latent_dim 128"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            if [[ ! "$MODEL" =~ ^(cGAN|ddim|vqvae)$ ]]; then
                echo "Error: Invalid model name '$MODEL'. Must be one of: cGAN, ddim, vqvae"
                exit 1
            fi
            shift 2
            ;;
        --num_classes)
            NUM_CLASSES="$2"
            if ! [[ "$NUM_CLASSES" =~ ^[0-9]+$ ]] || [ "$NUM_CLASSES" -le 0 ]; then
                echo "Error: num_classes must be a positive integer"
                exit 1
            fi
            shift 2
            ;;
        --par_path)
            PAR_PATH="$2"
            shift 2
            ;;
        --gen_root)
            GEN_ROOT="$2"
            shift 2
            ;;
        --real_root)
            REAL_ROOT="$2"
            shift 2
            ;;
        --classifier_path)
            CLASSIFIER_PATH="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --latent_dim)
            LATENT_DIM="$2"
            if ! [[ "$LATENT_DIM" =~ ^[0-9]+$ ]] || [ "$LATENT_DIM" -le 0 ]; then
                echo "Error: latent_dim must be a positive integer"
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$MODEL" ]; then
    echo "Error: --model parameter is required"
    usage
    exit 1
fi

# Auto-detect device if not specified
if [ -z "$DEVICE" ]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        DEVICE="cuda:0"
        echo "Auto-detected GPU device: $DEVICE"
    else
        DEVICE="cpu"
        echo "Auto-detected device: $DEVICE (no GPU found)"
    fi
fi

# Create output directory if it doesn't exist
mkdir -p "$GEN_ROOT"

echo "=========================================="
echo "Running Model: $MODEL"
echo "=========================================="
echo "Configuration:"
echo "  Number of classes: $NUM_CLASSES"
echo "  Parameter file: $PAR_PATH"
echo "  Generated data path: $GEN_ROOT"
echo "  Real data path: $REAL_ROOT"
echo "  Classifier path: $CLASSIFIER_PATH"
echo "  Model path: $MODEL_PATH"
echo "  Device: $DEVICE"
echo "  Latent dimension: $LATENT_DIM"
echo "=========================================="

# Run the Python script
python Generation_evaluation.py \
    --model "$MODEL" \
    --num_classes "$NUM_CLASSES" \
    --par_path "$PAR_PATH" \
    --gen_root "$GEN_ROOT" \
    --real_root "$REAL_ROOT" \
    --classifier_path "$CLASSIFIER_PATH" \
    --model_path "$MODEL_PATH" \
    --device "$DEVICE" \
    --latent_dim "$LATENT_DIM"

# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Model execution completed successfully!"
    echo "Results saved to: $GEN_ROOT"
else
    echo "❌ Model execution failed!"
    exit 1
fi