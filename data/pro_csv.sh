#!/bin/bash

# Default configuration parameters
DEFAULT_CSV_PATH="./ODIR/full_df.csv"
DEFAULT_OUTPUT_PATH="./ODIR/csv"
DEFAULT_DATA_FOLDER="./ODIR/ODIR-5K/Training Images"
DEFAULT_FOLDER_NUM=5

# Initialize variables with default values
CSV_PATH="$DEFAULT_CSV_PATH"
OUTPUT_PATH="$DEFAULT_OUTPUT_PATH"
DATA_FOLDER="$DEFAULT_DATA_FOLDER"
FOLDER_NUM="$DEFAULT_FOLDER_NUM"

# Usage help message
usage() {
    echo "Usage: $0 [csv_path] [output_path] [data_folder] [folder_num]"
    echo "  Default values:"
    echo "  CSV_PATH: $DEFAULT_CSV_PATH"
    echo "  OUTPUT_PATH: $DEFAULT_OUTPUT_PATH"
    echo "  DATA_FOLDER: $DEFAULT_DATA_FOLDER"
    echo "  FOLDER_NUM: $DEFAULT_FOLDER_NUM"
}

# Parse command line arguments
if [ ! -z "$1" ]; then
    CSV_PATH="$1"
fi
if [ ! -z "$2" ]; then
    OUTPUT_PATH="$2"
fi
if [ ! -z "$3" ]; then
    DATA_FOLDER="$3"
fi
if [ ! -z "$4" ]; then
    FOLDER_NUM="$4"
fi

# Validate parameters
if [ ! -f "$CSV_PATH" ]; then
    echo "Error: CSV file not found at $CSV_PATH"
    usage
    exit 1
fi

if [ ! -d "$DATA_FOLDER" ]; then
    echo "Error: Data folder not found at $DATA_FOLDER"
    usage
    exit 1
fi

# Display configuration
echo ">>> Configuration:"
echo ">>>   CSV file:    $CSV_PATH"
echo ">>>   Output dir:  $OUTPUT_PATH"
echo ">>>   Data folder: $DATA_FOLDER"
echo ">>>   Fold number: $FOLDER_NUM"

# Run Python script with parameters
echo ">>> Starting data processing..."
python pro_csv.py \
    --csv_path "$CSV_PATH" \
    --output_path "$OUTPUT_PATH" \
    --data_folder "$DATA_FOLDER" \
    --folder_num "$FOLDER_NUM"

# Check execution status
if [ $? -eq 0 ]; then
    echo ">>> Data processing completed successfully"
else
    echo ">>> Error occurred during data processing"
    exit 1
fi