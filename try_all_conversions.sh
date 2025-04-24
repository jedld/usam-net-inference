#!/bin/bash

CHECKPOINT="stereo_cnn_stereo_cnn_sa_baseline.checkpoint"
OUTPUT_DIR="model_outputs"

mkdir -p $OUTPUT_DIR

echo "========================"
echo "1. Inspecting checkpoint"
echo "========================"
python inspect_model.py $CHECKPOINT --mode inspect

echo ""
echo "============================"
echo "2. Trying fixed model"
echo "============================"
python convert_fixed_model.py $CHECKPOINT --output $OUTPUT_DIR/fixed_model.pt

echo ""
echo "============================"
echo "3. Trying with different sizes"
echo "============================"
for height in 224 240 320 384 480; do
  for width in 320 384 416 512 640; do
    echo ""
    echo "Trying size: ${height}x${width}"
    python convert_fixed_model.py $CHECKPOINT --output $OUTPUT_DIR/fixed_model_${height}x${width}.pt --height $height --width $width
  done
done

echo ""
echo "============================"
echo "4. Trying debug_model_shape"
echo "============================"
python debug_model_shape.py $CHECKPOINT

echo ""
echo "Success! Check $OUTPUT_DIR for all output models." 