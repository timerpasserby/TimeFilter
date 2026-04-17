#!/bin/bash
# 这个脚本用于统一管理当前 radar 数据上的正式训练消融命令和模块级验证命令。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_PATH="/opt/homebrew/Caskroom/miniforge/base/envs/tslib/bin/python"
DATA_PATH="./dataset/radar"
DATA_FILE="sim_radar_hourly_displacement.csv"
COORDS_FILE="./dataset/radar/sim_nodes_static.csv"
LOG_DIR="./logs/ablation_pipeline"
MODE="${1:-help}"

SEQ_LEN="${SEQ_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-48}"
PRED_LENS="${PRED_LENS:-12 24 48 96}"
ENC_IN="${ENC_IN:-1000}"
DEC_IN="${DEC_IN:-1000}"
C_OUT="${C_OUT:-1000}"
D_MODEL="${D_MODEL:-32}"
D_FF="${D_FF:-64}"
N_HEADS="${N_HEADS:-4}"
E_LAYERS="${E_LAYERS:-1}"
PATCH_LEN="${PATCH_LEN:-96}"
DROPOUT="${DROPOUT:-0.1}"
TOP_P="${TOP_P:-0.5}"
LEARNING_RATE="${LEARNING_RATE:-0.0005}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PHYSICAL_MASK_RADIUS="${PHYSICAL_MASK_RADIUS:-120.0}"
CSP_DEBUG="${CSP_DEBUG:-0}"

mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR/matplotlib"
export MPLCONFIGDIR="$LOG_DIR/matplotlib"


# 打印脚本使用说明。
print_help() {
  cat <<'EOF'
用法：
  bash scripts/radar_ablation_pipeline.sh <mode>

可用 mode：
  baseline_long    运行长预测 baseline 消融，pred_len 默认 12 24 48 96
  csp_long         运行长预测 CSP-TimeFilter 消融，pred_len 默认 12 24 48 96
  baseline_short   运行短预测 baseline 消融，pred_len 默认 12 24 48 96
  csp_short        运行短预测 CSP-TimeFilter 消融，pred_len 默认 12 24 48 96
  train_all        顺序运行以上 4 组正式训练消融
  weather_checks   运行天气模块单测和最小 demo
  blast_checks     运行爆破模块单测和最小 demo
  module_checks    顺序运行天气和爆破模块检查
  help             打印帮助

说明：
  1. 目前正式接入训练入口的只有 baseline 和 CSPAdapter 两组主干消融。
  2. weather / blast 目前放在 module_checks 中，只做模块级验证，不做正式训练实验。
  3. 如果要后台运行，直接使用：
       nohup bash scripts/radar_ablation_pipeline.sh train_all &
EOF
}


# 根据任务名和是否启用 CSP 生成实验标签。
build_case_tag() {
  local task_name="$1"
  local use_csp="$2"
  local pred_len="$3"
  local prefix="baseline"
  if [ "$use_csp" = "1" ]; then
    prefix="csp"
  fi
  if [ "$task_name" = "long_term_forecast" ]; then
    echo "${prefix}_long_pl${pred_len}"
  else
    echo "${prefix}_short_pl${pred_len}"
  fi
}


# 执行单个正式训练实验。
run_train_case() {
  local task_name="$1"
  local use_csp="$2"
  local pred_len="$3"
  local case_tag
  local log_file

  case_tag="$(build_case_tag "$task_name" "$use_csp" "$pred_len")"
  log_file="$LOG_DIR/${case_tag}.log"

  {
    echo "============================================================"
    echo "开始运行正式训练实验: ${case_tag}"
    echo "task_name=${task_name} use_csp_adapter=${use_csp} pred_len=${pred_len}"
    echo "日志文件: ${log_file}"
    echo "============================================================"

    "$PYTHON_PATH" -u run.py \
      --task_name "$task_name" \
      --is_training 1 \
      --root_path "$DATA_PATH" \
      --data_path "$DATA_FILE" \
      --model_id "$case_tag" \
      --model TimeFilter \
      --data custom \
      --features M \
      --seq_len "$SEQ_LEN" \
      --label_len "$LABEL_LEN" \
      --pred_len "$pred_len" \
      --e_layers "$E_LAYERS" \
      --n_heads "$N_HEADS" \
      --enc_in "$ENC_IN" \
      --dec_in "$DEC_IN" \
      --c_out "$C_OUT" \
      --d_model "$D_MODEL" \
      --d_ff "$D_FF" \
      --dropout "$DROPOUT" \
      --patch_len "$PATCH_LEN" \
      --top_p "$TOP_P" \
      --learning_rate "$LEARNING_RATE" \
      --des Exp \
      --itr 1 \
      --batch_size "$BATCH_SIZE" \
      --train_epochs "$TRAIN_EPOCHS" \
      --num_workers "$NUM_WORKERS" \
      --use_norm 0 \
      --use_csp_adapter "$use_csp" \
      --coords_path "$COORDS_FILE" \
      --physical_mask_radius "$PHYSICAL_MASK_RADIUS" \
      --csp_debug "$CSP_DEBUG"
  } 2>&1 | tee -a "$log_file"
}


# 顺序运行某一组正式训练消融。
run_train_group() {
  local task_name="$1"
  local use_csp="$2"
  local pred_len

  for pred_len in $PRED_LENS
  do
    run_train_case "$task_name" "$use_csp" "$pred_len"
  done
}


# 运行天气模块检查。
run_weather_checks() {
  local log_file="$LOG_DIR/weather_checks.log"
  {
    echo "============================================================"
    echo "开始运行天气模块检查"
    echo "日志文件: ${log_file}"
    echo "============================================================"
    "$PYTHON_PATH" -m unittest discover -s tests -p "test_weather_*.py"
    "$PYTHON_PATH" scripts/weather_injection_demo.py
  } 2>&1 | tee -a "$log_file"
}


# 运行爆破模块检查。
run_blast_checks() {
  local log_file="$LOG_DIR/blast_checks.log"
  {
    echo "============================================================"
    echo "开始运行爆破模块检查"
    echo "日志文件: ${log_file}"
    echo "============================================================"
    "$PYTHON_PATH" -m unittest discover -s tests -p "test_blast_*.py"
    "$PYTHON_PATH" scripts/blast_injection_demo.py
  } 2>&1 | tee -a "$log_file"
}


# 按模式路由到对应 pipeline。
main() {
  case "$MODE" in
    baseline_long)
      run_train_group long_term_forecast 0
      ;;
    csp_long)
      run_train_group long_term_forecast 1
      ;;
    baseline_short)
      run_train_group short_term_forecast 0
      ;;
    csp_short)
      run_train_group short_term_forecast 1
      ;;
    train_all)
      run_train_group long_term_forecast 0
      run_train_group long_term_forecast 1
      run_train_group short_term_forecast 0
      run_train_group short_term_forecast 1
      ;;
    weather_checks)
      run_weather_checks
      ;;
    blast_checks)
      run_blast_checks
      ;;
    module_checks)
      run_weather_checks
      run_blast_checks
      ;;
    help|--help|-h)
      print_help
      ;;
    *)
      echo "不支持的 mode: $MODE"
      print_help
      exit 1
      ;;
  esac
}


main
