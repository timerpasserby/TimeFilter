#!/bin/bash
# 这个脚本用于批量启动 radar 数据上的 TimeFilter 预测任务，并将日志写入 logs/ 目录。

set -euo pipefail

PYTHON_PATH="/opt/homebrew/Caskroom/miniforge/base/envs/tslib/bin/python"
DATA_PATH="./dataset/radar"
DATA_FILE="sim_radar_hourly_displacement.csv"
COORDS_FILE="./dataset/radar/sim_nodes_static.csv"
LOG_DIR="./logs"
LOG_FILE="$LOG_DIR/run_models_$(date +%Y%m%d_%H%M%S).log"
MODEL_NAME="TimeFilter"
SEQ_LEN=96
LABEL_LEN=48
ENC_IN=1000
DEC_IN=1000
C_OUT=1000
TASK_NAME="${TASK_NAME:-long_term_forecast}"
USE_CSP_ADAPTER="${USE_CSP_ADAPTER:-0}"
CSP_DEBUG="${CSP_DEBUG:-0}"
PHYSICAL_MASK_RADIUS="${PHYSICAL_MASK_RADIUS:-120.0}"
TOP_P="${TOP_P:-0.5}"

mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR/matplotlib"
export MPLCONFIGDIR="$LOG_DIR/matplotlib"

{
echo "Logging to $LOG_FILE"
echo "Running TimeFilter on $DATA_FILE..."

# 参考 PEMS04 的多预测窗口训练方式，依次运行 2 组预测长度。
for pred_len in 12 24 
do
  echo "Starting pred_len=$pred_len"
  "$PYTHON_PATH" -u run.py \
    --task_name "$TASK_NAME" \
    --is_training 1 \
    --root_path "$DATA_PATH" \
    --data_path "$DATA_FILE" \
    --model_id "radar_${SEQ_LEN}_${pred_len}" \
    --model "$MODEL_NAME" \
    --data custom \
    --features M \
    --seq_len "$SEQ_LEN" \
    --label_len "$LABEL_LEN" \
    --pred_len "$pred_len" \
    --e_layers 1 \
    --n_heads 4 \
    --enc_in "$ENC_IN" \
    --dec_in "$DEC_IN" \
    --c_out "$C_OUT" \
    --d_model 32 \
    --d_ff 64 \
    --dropout 0.1 \
    --patch_len 96 \
    --top_p "$TOP_P" \
    --learning_rate 0.0005 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 1 \
    --train_epochs 1 \
    --num_workers 0 \
    --use_norm 0 \
    --use_csp_adapter "$USE_CSP_ADAPTER" \
    --coords_path "$COORDS_FILE" \
    --physical_mask_radius "$PHYSICAL_MASK_RADIUS" \
    --csp_debug "$CSP_DEBUG"
done
} 2>&1 | tee -a "$LOG_FILE"
