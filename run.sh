#!/bin/bash
trap "exit" INT

LOG_DIR="$HOME/log/Learn_DL"
DATA_DIR="$HOME/DATA"
NUM_WORKERS=4

python ./fast_rcnn_main.py \
  --run_name "detection-fast_rcnn-voc" --exp_name "resnet-rpn" \
  --data_dir $DATA_DIR --log_dir $LOG_DIR --num_workers $NUM_WORKERS \
  --batch 16 \
  "$@"


trap - INT
echo "END."