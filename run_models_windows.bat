@echo off
REM 这个脚本用于在 Windows 上批量启动 radar 数据上的 TimeFilter 预测任务，并将日志写入 logs 目录。

setlocal enabledelayedexpansion

if not defined PYTHON_PATH set "PYTHON_PATH=D:\Anaconda\envs\TLib\python.exe"
if not exist "%PYTHON_PATH%" (
  echo Python 路径不存在: %PYTHON_PATH%
  exit /b 1
)

set "DATA_PATH=./dataset/radar"
set "DATA_FILE=sim_radar_hourly_displacement.csv"
set "COORDS_FILE=./dataset/radar/sim_nodes_static.csv"
set "LOG_DIR=.\logs"
set "MODEL_NAME=TimeFilter"
set "SEQ_LEN=96"
set "LABEL_LEN=48"
set "ENC_IN=1000"
set "DEC_IN=1000"
set "C_OUT=1000"

if not defined TASK_NAME set "TASK_NAME=long_term_forecast"
if not defined USE_CSP_ADAPTER set "USE_CSP_ADAPTER=0"
if not defined CSP_DEBUG set "CSP_DEBUG=0"
if not defined PHYSICAL_MASK_RADIUS set "PHYSICAL_MASK_RADIUS=120.0"
if not defined TOP_P set "TOP_P=0.5"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%LOG_DIR%\matplotlib" mkdir "%LOG_DIR%\matplotlib"
set "MPLCONFIGDIR=%LOG_DIR%\matplotlib"

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_TS=%%I"
set "LOG_FILE=%LOG_DIR%\run_models_%RUN_TS%.log"

echo Logging to %LOG_FILE%
echo Running TimeFilter on %DATA_FILE%...
echo Python path: %PYTHON_PATH%

>> "%LOG_FILE%" echo Logging to %LOG_FILE%
>> "%LOG_FILE%" echo Running TimeFilter on %DATA_FILE%...
>> "%LOG_FILE%" echo Python path: %PYTHON_PATH%

REM 参考 PEMS04 的多预测窗口训练方式，依次运行 2 组预测长度。
for %%P in (12 24) do (
  echo ============================================================
  echo Starting pred_len=%%P
  >> "%LOG_FILE%" echo ============================================================
  >> "%LOG_FILE%" echo Starting pred_len=%%P

  "%PYTHON_PATH%" -u run.py ^
    --task_name "%TASK_NAME%" ^
    --is_training 1 ^
    --root_path "%DATA_PATH%" ^
    --data_path "%DATA_FILE%" ^
    --model_id "radar_%SEQ_LEN%_%%P" ^
    --model "%MODEL_NAME%" ^
    --data custom ^
    --features M ^
    --seq_len %SEQ_LEN% ^
    --label_len %LABEL_LEN% ^
    --pred_len %%P ^
    --e_layers 1 ^
    --n_heads 4 ^
    --enc_in %ENC_IN% ^
    --dec_in %DEC_IN% ^
    --c_out %C_OUT% ^
    --d_model 32 ^
    --d_ff 64 ^
    --dropout 0.1 ^
    --patch_len 96 ^
    --top_p "%TOP_P%" ^
    --learning_rate 0.0005 ^
    --des Exp ^
    --itr 1 ^
    --batch_size 1 ^
    --train_epochs 1 ^
    --num_workers 0 ^
    --use_norm 0 ^
    --use_csp_adapter "%USE_CSP_ADAPTER%" ^
    --coords_path "%COORDS_FILE%" ^
    --physical_mask_radius "%PHYSICAL_MASK_RADIUS%" ^
    --csp_debug "%CSP_DEBUG%" >> "%LOG_FILE%" 2>&1

  if errorlevel 1 (
    echo 运行失败，详情请查看日志: %LOG_FILE%
    exit /b 1
  )
)

echo 全部任务已完成，日志保存在 %LOG_FILE%
exit /b 0
