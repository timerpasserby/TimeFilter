#!/bin/bash
# 这个脚本用于统一管理 radar 数据上的正式训练消融实验，以及天气/爆破模块的独立检查命令。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

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
DATA_PATH="${DATA_PATH:-./dataset/radar}"
DATA_FILE="${DATA_FILE:-sim_radar_hourly_displacement.csv}"
COORDS_FILE="${COORDS_FILE:-./dataset/radar/sim_nodes_static.csv}"
WEATHER_FILE="${WEATHER_FILE:-./dataset/radar/sim_weather.csv}"
BLAST_FILE="${BLAST_FILE:-./dataset/radar/sim_blast_logs.csv}"
LOG_DIR="${LOG_DIR:-./logs/ablation_pipeline}"
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
EXO_DEBUG="${EXO_DEBUG:-0}"

WEATHER_DIM="${WEATHER_DIM:-3}"
WEATHER_HIDDEN_DIM="${WEATHER_HIDDEN_DIM:-0}"
WEATHER_KERNEL_SIZE="${WEATHER_KERNEL_SIZE:-3}"
WEATHER_DILATIONS="${WEATHER_DILATIONS:-1 2}"
BLAST_MAX_EVENTS="${BLAST_MAX_EVENTS:-64}"
BLAST_HIDDEN_DIM="${BLAST_HIDDEN_DIM:-0}"
BLAST_INIT_SIGMA_B="${BLAST_INIT_SIGMA_B:-120.0}"
BLAST_INIT_GAMMA_B="${BLAST_INIT_GAMMA_B:-0.1}"

mkdir -p "$LOG_DIR"
mkdir -p "$LOG_DIR/matplotlib"
export MPLCONFIGDIR="$LOG_DIR/matplotlib"


# 打印脚本使用说明。
print_help() {
  cat <<'EOF'
用法：
  bash scripts/radar_ablation_pipeline.sh <mode>

主模型实验：
  baseline_long                   baseline，长预测
  csp_long                        CSP-TimeFilter，长预测
  csp_weather_long                CSP + Weather(Causal-Attn)，长预测
  csp_weather_blast_long          CSP + Weather + Blast(main)，长预测
  baseline_short                  baseline，短预测
  csp_short                       CSP-TimeFilter，短预测
  csp_weather_short               CSP + Weather(Causal-Attn)，短预测
  csp_weather_blast_short         CSP + Weather + Blast(main)，短预测

消融实验：
  csp_weather_vanilla_long        CSP + Weather(Vanilla-Attn)，长预测
  csp_weather_concat_long         CSP + Weather(Concat-Fusion)，长预测
  csp_weather_blast_wogate_long   CSP + Weather + Blast(w/o Gate)，长预测
  csp_weather_blast_gru_long      CSP + Weather + Blast(GRU-blast)，长预测
  csp_weather_vanilla_short       CSP + Weather(Vanilla-Attn)，短预测
  csp_weather_concat_short        CSP + Weather(Concat-Fusion)，短预测
  csp_weather_blast_wogate_short  CSP + Weather + Blast(w/o Gate)，短预测
  csp_weather_blast_gru_short     CSP + Weather + Blast(GRU-blast)，短预测

批量运行：
  train_main_long                 运行 baseline/csp/csp+weather/csp+weather+blast 四组长预测
  train_ablation_long             运行天气与爆破消融四组长预测
  train_all_long                  运行长预测主模型与消融全部实验
  train_all_short                 运行短预测主模型与消融全部实验
  train_all                       顺序运行长预测与短预测全部实验

模块检查：
  weather_checks                  运行天气模块单测和最小 demo
  blast_checks                    运行爆破模块单测和最小 demo
  module_checks                   顺序运行天气和爆破模块检查
  help                            打印帮助

说明：
  1. 当前所有正式训练命令都已经打通到主干，可直接生成 results/ 与 test_results/。
  2. 默认 pred_len 为 12 24 48 96，可通过环境变量 PRED_LENS 覆盖。
  3. 如果要后台运行，直接使用：
       nohup bash scripts/radar_ablation_pipeline.sh train_all >> logs/ablation_pipeline/nohup.log 2>&1 &
EOF
}


# 根据实验开关生成统一实验标签。
build_case_tag() {
  local task_name="$1"
  local use_csp="$2"
  local use_weather="$3"
  local weather_mode="$4"
  local use_blast="$5"
  local blast_mode="$6"
  local pred_len="$7"
  local parts=()

  if [ "$use_csp" = "1" ]; then
    parts+=("csp")
  else
    parts+=("baseline")
  fi

  if [ "$use_weather" = "1" ]; then
    case "$weather_mode" in
      causal_attn)
        parts+=("weather")
        ;;
      vanilla_attn)
        parts+=("weather-vanilla")
        ;;
      concat_fusion)
        parts+=("weather-concat")
        ;;
      *)
        parts+=("weather-${weather_mode}")
        ;;
    esac
  fi

  if [ "$use_blast" = "1" ]; then
    case "$blast_mode" in
      main)
        parts+=("blast")
        ;;
      wo_gate)
        parts+=("blast-wogate")
        ;;
      gru_blast)
        parts+=("blast-gru")
        ;;
      *)
        parts+=("blast-${blast_mode}")
        ;;
    esac
  fi

  if [ "$task_name" = "long_term_forecast" ]; then
    parts+=("long")
  else
    parts+=("short")
  fi
  parts+=("pl${pred_len}")

  local joined=""
  local item
  for item in "${parts[@]}"; do
    if [ -z "$joined" ]; then
      joined="$item"
    else
      joined="${joined}_${item}"
    fi
  done
  echo "$joined"
}


# 执行单个正式训练实验。
run_train_case() {
  local task_name="$1"
  local use_csp="$2"
  local use_weather="$3"
  local weather_mode="$4"
  local use_blast="$5"
  local blast_mode="$6"
  local pred_len="$7"
  local case_tag
  local log_file

  case_tag="$(build_case_tag "$task_name" "$use_csp" "$use_weather" "$weather_mode" "$use_blast" "$blast_mode" "$pred_len")"
  log_file="$LOG_DIR/${case_tag}.log"

  {
    echo "============================================================"
    echo "开始运行正式训练实验: ${case_tag}"
    echo "task_name=${task_name} use_csp_adapter=${use_csp} use_weather_module=${use_weather} weather_mode=${weather_mode}"
    echo "use_blast_module=${use_blast} blast_mode=${blast_mode} pred_len=${pred_len}"
    echo "Python path=${PYTHON_PATH}"
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
      --inverse \
      --use_csp_adapter "$use_csp" \
      --coords_path "$COORDS_FILE" \
      --physical_mask_radius "$PHYSICAL_MASK_RADIUS" \
      --csp_debug "$CSP_DEBUG" \
      --use_weather_module "$use_weather" \
      --weather_path "$WEATHER_FILE" \
      --weather_dim "$WEATHER_DIM" \
      --weather_hidden_dim "$WEATHER_HIDDEN_DIM" \
      --weather_kernel_size "$WEATHER_KERNEL_SIZE" \
      --weather_dilations $WEATHER_DILATIONS \
      --weather_ablation_mode "$weather_mode" \
      --use_blast_module "$use_blast" \
      --blast_path "$BLAST_FILE" \
      --blast_max_events "$BLAST_MAX_EVENTS" \
      --blast_hidden_dim "$BLAST_HIDDEN_DIM" \
      --blast_mode "$blast_mode" \
      --blast_init_sigma_b "$BLAST_INIT_SIGMA_B" \
      --blast_init_gamma_b "$BLAST_INIT_GAMMA_B" \
      --exo_debug "$EXO_DEBUG"
  } 2>&1 | tee -a "$log_file"
}


# 顺序运行某一组正式训练消融。
run_train_group() {
  local task_name="$1"
  local use_csp="$2"
  local use_weather="$3"
  local weather_mode="$4"
  local use_blast="$5"
  local blast_mode="$6"
  local pred_len

  for pred_len in $PRED_LENS
  do
    run_train_case "$task_name" "$use_csp" "$use_weather" "$weather_mode" "$use_blast" "$blast_mode" "$pred_len"
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


# 运行长预测主模型四组实验。
run_main_long_suite() {
  run_train_group long_term_forecast 0 0 causal_attn 0 main
  run_train_group long_term_forecast 1 0 causal_attn 0 main
  run_train_group long_term_forecast 1 1 causal_attn 0 main
  run_train_group long_term_forecast 1 1 causal_attn 1 main
}


# 运行长预测消融四组实验。
run_ablation_long_suite() {
  run_train_group long_term_forecast 1 1 vanilla_attn 0 main
  run_train_group long_term_forecast 1 1 concat_fusion 0 main
  run_train_group long_term_forecast 1 1 causal_attn 1 wo_gate
  run_train_group long_term_forecast 1 1 causal_attn 1 gru_blast
}


# 运行短预测主模型四组实验。
run_main_short_suite() {
  run_train_group short_term_forecast 0 0 causal_attn 0 main
  run_train_group short_term_forecast 1 0 causal_attn 0 main
  run_train_group short_term_forecast 1 1 causal_attn 0 main
  run_train_group short_term_forecast 1 1 causal_attn 1 main
}


# 运行短预测消融四组实验。
run_ablation_short_suite() {
  run_train_group short_term_forecast 1 1 vanilla_attn 0 main
  run_train_group short_term_forecast 1 1 concat_fusion 0 main
  run_train_group short_term_forecast 1 1 causal_attn 1 wo_gate
  run_train_group short_term_forecast 1 1 causal_attn 1 gru_blast
}


# 按模式路由到对应 pipeline。
main() {
  case "$MODE" in
    baseline_long)
      run_train_group long_term_forecast 0 0 causal_attn 0 main
      ;;
    csp_long)
      run_train_group long_term_forecast 1 0 causal_attn 0 main
      ;;
    csp_weather_long)
      run_train_group long_term_forecast 1 1 causal_attn 0 main
      ;;
    csp_weather_vanilla_long)
      run_train_group long_term_forecast 1 1 vanilla_attn 0 main
      ;;
    csp_weather_concat_long)
      run_train_group long_term_forecast 1 1 concat_fusion 0 main
      ;;
    csp_weather_blast_long)
      run_train_group long_term_forecast 1 1 causal_attn 1 main
      ;;
    csp_weather_blast_wogate_long)
      run_train_group long_term_forecast 1 1 causal_attn 1 wo_gate
      ;;
    csp_weather_blast_gru_long)
      run_train_group long_term_forecast 1 1 causal_attn 1 gru_blast
      ;;
    baseline_short)
      run_train_group short_term_forecast 0 0 causal_attn 0 main
      ;;
    csp_short)
      run_train_group short_term_forecast 1 0 causal_attn 0 main
      ;;
    csp_weather_short)
      run_train_group short_term_forecast 1 1 causal_attn 0 main
      ;;
    csp_weather_vanilla_short)
      run_train_group short_term_forecast 1 1 vanilla_attn 0 main
      ;;
    csp_weather_concat_short)
      run_train_group short_term_forecast 1 1 concat_fusion 0 main
      ;;
    csp_weather_blast_short)
      run_train_group short_term_forecast 1 1 causal_attn 1 main
      ;;
    csp_weather_blast_wogate_short)
      run_train_group short_term_forecast 1 1 causal_attn 1 wo_gate
      ;;
    csp_weather_blast_gru_short)
      run_train_group short_term_forecast 1 1 causal_attn 1 gru_blast
      ;;
    train_main_long)
      run_main_long_suite
      ;;
    train_ablation_long)
      run_ablation_long_suite
      ;;
    train_all_long)
      run_main_long_suite
      run_ablation_long_suite
      ;;
    train_all_short)
      run_main_short_suite
      run_ablation_short_suite
      ;;
    train_all)
      run_main_long_suite
      run_ablation_long_suite
      run_main_short_suite
      run_ablation_short_suite
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
