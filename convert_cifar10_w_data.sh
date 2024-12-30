#!/bin/bash
ROOT_DIR="datasets/cifar10_w"

# xxx/yyy 
find "$ROOT_DIR" -type d -mindepth 2 -maxdepth 2 | while read DATASET_DIR; do
    PARENT_DIR=$(dirname "$DATASET_DIR")
    DATASET_NAME=$(basename "$DATASET_DIR")
    OUTPUT_PATH="$PARENT_DIR/${DATASET_NAME//-/_}.pt"

    echo "Converting $DATASET_DIR to $OUTPUT_PATH..."
    python3 convert_cifar10_w_data.py "$DATASET_DIR" "$OUTPUT_PATH" || { echo "Failed to convert $DATASET_DIR"; continue; }

    echo "Removing $DATASET_DIR..."
    rm -rf "$DATASET_DIR"
done
echo "All datasets have been converted and cleaned up."