#!/bin/bash
# 这个脚本用于批量启动 radar 数据上的 TimeFilter 预测任务，并将日志写入 logs/ 目录。

set -euo pipefail

# 自动解析当前可用的 Python 解释器，允许外部通过 PYTHON_PATH 覆盖。
resolve_python_path() {
  if [ -n "${PYTHON_PATH:-}" ]; then
    echo "$PYTHON_PATH"
    return 0
  fi

  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    echo "${CONDA_PREFIX}/bin/python"
    return 0
  fi

  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    echo "${VIRTUAL_ENV}/bin/python"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  echo "未找到可用的 Python 解释器，请通过环境变量 PYTHON_PATH 显式指定。" >&2
  exit 1
}

PYTHON_PATH="$(resolve_python_path)"
DATA_PATH="./dataset/radar"
DATA_FILE="sim_radar_hourly_displacement.csv"
COORDS_FILE="./dataset/radar/sim_nodes_static.csv"
WEATHER_FILE="./dataset/radar/sim_weather.csv"
BLAST_FILE="./dataset/radar/sim_blast_logs.csv"
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
USE_WEATHER_MODULE="${USE_WEATHER_MODULE:-0}"
WEATHER_ABLATION_MODE="${WEATHER_ABLATION_MODE:-causal_attn}"
USE_BLAST_MODULE="${USE_BLAST_MODULE:-0}"
BLAST_MODE="${BLAST_MODE:-main}"
BLAST_MAX_EVENTS="${BLAST_MAX_EVENTS:-64}"
CSP_DEBUG="${CSP_DEBUG:-0}"
EXO_DEBUG="${EXO_DEBUG:-0}"
PHYSICAL_MASK_RADIUS="${PHYSICAL_MASK_RADIUS:-120.0}"
TOP_P="${TOP_P:-0.5}"
PRED_LENS="${PRED_LENS:-12 24 48 96}"

mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR/matplotlib"
export MPLCONFIGDIR="$LOG_DIR/matplotlib"

{
echo "Logging to $LOG_FILE"
echo "Running TimeFilter on $DATA_FILE..."
echo "Python path: $PYTHON_PATH"

# 参考 PEMS04 的多预测窗口训练方式，依次运行 4 组预测长度。
for pred_len in $PRED_LENS
do
  echo "Starting pred_len=$pred_len"
  echo "use_csp_adapter=$USE_CSP_ADAPTER use_weather_module=$USE_WEATHER_MODULE weather_mode=$WEATHER_ABLATION_MODE"
  echo "use_blast_module=$USE_BLAST_MODULE blast_mode=$BLAST_MODE"

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
    --csp_debug "$CSP_DEBUG" \
    --use_weather_module "$USE_WEATHER_MODULE" \
    --weather_path "$WEATHER_FILE" \
    --weather_ablation_mode "$WEATHER_ABLATION_MODE" \
    --use_blast_module "$USE_BLAST_MODULE" \
    --blast_path "$BLAST_FILE" \
    --blast_max_events "$BLAST_MAX_EVENTS" \
    --blast_mode "$BLAST_MODE" \
    --exo_debug "$EXO_DEBUG"
done
} 2>&1 | tee -a "$LOG_FILE"
