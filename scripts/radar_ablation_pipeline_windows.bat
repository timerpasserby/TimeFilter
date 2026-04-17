@echo off
REM 这个脚本用于在 Windows 上统一管理 radar 数据上的正式训练消融命令和模块级验证命令。

setlocal enabledelayedexpansion

if not defined PYTHON_PATH set "PYTHON_PATH=D:\Anaconda\envs\TLib\python.exe"
if not exist "%PYTHON_PATH%" (
  echo Python 路径不存在: %PYTHON_PATH%
  exit /b 1
)

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

set "DATA_PATH=./dataset/radar"
set "DATA_FILE=sim_radar_hourly_displacement.csv"
set "COORDS_FILE=./dataset/radar/sim_nodes_static.csv"
set "LOG_DIR=.\logs\ablation_pipeline"
set "MODE=%~1"

if "%MODE%"=="" set "MODE=help"
if not defined SEQ_LEN set "SEQ_LEN=96"
if not defined LABEL_LEN set "LABEL_LEN=48"
if not defined PRED_LENS set "PRED_LENS=12 24 48 96"
if not defined ENC_IN set "ENC_IN=1000"
if not defined DEC_IN set "DEC_IN=1000"
if not defined C_OUT set "C_OUT=1000"
if not defined D_MODEL set "D_MODEL=32"
if not defined D_FF set "D_FF=64"
if not defined N_HEADS set "N_HEADS=4"
if not defined E_LAYERS set "E_LAYERS=1"
if not defined PATCH_LEN set "PATCH_LEN=96"
if not defined DROPOUT set "DROPOUT=0.1"
if not defined TOP_P set "TOP_P=0.5"
if not defined LEARNING_RATE set "LEARNING_RATE=0.0005"
if not defined TRAIN_EPOCHS set "TRAIN_EPOCHS=1"
if not defined BATCH_SIZE set "BATCH_SIZE=1"
if not defined NUM_WORKERS set "NUM_WORKERS=0"
if not defined PHYSICAL_MASK_RADIUS set "PHYSICAL_MASK_RADIUS=120.0"
if not defined CSP_DEBUG set "CSP_DEBUG=0"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%LOG_DIR%\matplotlib" mkdir "%LOG_DIR%\matplotlib"
set "MPLCONFIGDIR=%LOG_DIR%\matplotlib"

if /i "%MODE%"=="baseline_long" (
  call :run_train_group long_term_forecast 0
  exit /b %errorlevel%
)
if /i "%MODE%"=="csp_long" (
  call :run_train_group long_term_forecast 1
  exit /b %errorlevel%
)
if /i "%MODE%"=="baseline_short" (
  call :run_train_group short_term_forecast 0
  exit /b %errorlevel%
)
if /i "%MODE%"=="csp_short" (
  call :run_train_group short_term_forecast 1
  exit /b %errorlevel%
)
if /i "%MODE%"=="train_all" (
  call :run_train_group long_term_forecast 0 || exit /b 1
  call :run_train_group long_term_forecast 1 || exit /b 1
  call :run_train_group short_term_forecast 0 || exit /b 1
  call :run_train_group short_term_forecast 1 || exit /b 1
  exit /b 0
)
if /i "%MODE%"=="weather_checks" (
  call :run_weather_checks
  exit /b %errorlevel%
)
if /i "%MODE%"=="blast_checks" (
  call :run_blast_checks
  exit /b %errorlevel%
)
if /i "%MODE%"=="module_checks" (
  call :run_weather_checks || exit /b 1
  call :run_blast_checks || exit /b 1
  exit /b 0
)
if /i "%MODE%"=="help" goto :print_help
if /i "%MODE%"=="--help" goto :print_help
if /i "%MODE%"=="-h" goto :print_help

echo 不支持的 mode: %MODE%
goto :print_help

:print_help
echo 用法：
echo   scripts\radar_ablation_pipeline_windows.bat ^<mode^>
echo.
echo 可用 mode：
echo   baseline_long    运行长预测 baseline 消融，pred_len 默认 12 24 48 96
echo   csp_long         运行长预测 CSP-TimeFilter 消融，pred_len 默认 12 24 48 96
echo   baseline_short   运行短预测 baseline 消融，pred_len 默认 12 24 48 96
echo   csp_short        运行短预测 CSP-TimeFilter 消融，pred_len 默认 12 24 48 96
echo   train_all        顺序运行以上 4 组正式训练消融
echo   weather_checks   运行天气模块单测和最小 demo
echo   blast_checks     运行爆破模块单测和最小 demo
echo   module_checks    顺序运行天气和爆破模块检查
echo   help             打印帮助
echo.
echo 说明：
echo   1. 当前正式接入训练入口的只有 baseline 和 CSPAdapter 两组主干消融。
echo   2. weather / blast 目前只做模块级验证，不做正式训练实验。
echo   3. Python 默认固定为 %PYTHON_PATH%
exit /b 0

:run_train_group
set "TASK_NAME_ARG=%~1"
set "USE_CSP_ARG=%~2"
for %%P in (%PRED_LENS%) do (
  call :run_train_case "%TASK_NAME_ARG%" "%USE_CSP_ARG%" "%%P" || exit /b 1
)
exit /b 0

:run_train_case
set "TASK_NAME_ARG=%~1"
set "USE_CSP_ARG=%~2"
set "PRED_LEN_ARG=%~3"

if "%USE_CSP_ARG%"=="1" (
  set "PREFIX=csp"
) else (
  set "PREFIX=baseline"
)

if "%TASK_NAME_ARG%"=="long_term_forecast" (
  set "CASE_TAG=!PREFIX!_long_pl%PRED_LEN_ARG%"
) else (
  set "CASE_TAG=!PREFIX!_short_pl%PRED_LEN_ARG%"
)

set "LOG_FILE=%LOG_DIR%\!CASE_TAG!.log"

echo ============================================================
echo 开始运行正式训练实验: !CASE_TAG!
echo task_name=%TASK_NAME_ARG% use_csp_adapter=%USE_CSP_ARG% pred_len=%PRED_LEN_ARG%
echo Python path=%PYTHON_PATH%
echo 日志文件: !LOG_FILE!

>> "!LOG_FILE!" echo ============================================================
>> "!LOG_FILE!" echo 开始运行正式训练实验: !CASE_TAG!
>> "!LOG_FILE!" echo task_name=%TASK_NAME_ARG% use_csp_adapter=%USE_CSP_ARG% pred_len=%PRED_LEN_ARG%
>> "!LOG_FILE!" echo Python path=%PYTHON_PATH%
>> "!LOG_FILE!" echo 日志文件: !LOG_FILE!

"%PYTHON_PATH%" -u run.py ^
  --task_name "%TASK_NAME_ARG%" ^
  --is_training 1 ^
  --root_path "%DATA_PATH%" ^
  --data_path "%DATA_FILE%" ^
  --model_id "!CASE_TAG!" ^
  --model TimeFilter ^
  --data custom ^
  --features M ^
  --seq_len %SEQ_LEN% ^
  --label_len %LABEL_LEN% ^
  --pred_len %PRED_LEN_ARG% ^
  --e_layers %E_LAYERS% ^
  --n_heads %N_HEADS% ^
  --enc_in %ENC_IN% ^
  --dec_in %DEC_IN% ^
  --c_out %C_OUT% ^
  --d_model %D_MODEL% ^
  --d_ff %D_FF% ^
  --dropout %DROPOUT% ^
  --patch_len %PATCH_LEN% ^
  --top_p %TOP_P% ^
  --learning_rate %LEARNING_RATE% ^
  --des Exp ^
  --itr 1 ^
  --batch_size %BATCH_SIZE% ^
  --train_epochs %TRAIN_EPOCHS% ^
  --num_workers %NUM_WORKERS% ^
  --use_norm 0 ^
  --use_csp_adapter %USE_CSP_ARG% ^
  --coords_path "%COORDS_FILE%" ^
  --physical_mask_radius %PHYSICAL_MASK_RADIUS% ^
  --csp_debug %CSP_DEBUG% >> "!LOG_FILE!" 2>&1

if errorlevel 1 (
  echo 运行失败，详情请查看日志: !LOG_FILE!
  exit /b 1
)
exit /b 0

:run_weather_checks
set "LOG_FILE=%LOG_DIR%\weather_checks_windows.log"
echo ============================================================
echo 开始运行天气模块检查
echo 日志文件: %LOG_FILE%
>> "%LOG_FILE%" echo ============================================================
>> "%LOG_FILE%" echo 开始运行天气模块检查
>> "%LOG_FILE%" echo 日志文件: %LOG_FILE%

"%PYTHON_PATH%" -m unittest discover -s tests -p test_weather_*.py >> "%LOG_FILE%" 2>&1
if errorlevel 1 exit /b 1
"%PYTHON_PATH%" scripts/weather_injection_demo.py >> "%LOG_FILE%" 2>&1
if errorlevel 1 exit /b 1
exit /b 0

:run_blast_checks
set "LOG_FILE=%LOG_DIR%\blast_checks_windows.log"
echo ============================================================
echo 开始运行爆破模块检查
echo 日志文件: %LOG_FILE%
>> "%LOG_FILE%" echo ============================================================
>> "%LOG_FILE%" echo 开始运行爆破模块检查
>> "%LOG_FILE%" echo 日志文件: %LOG_FILE%

"%PYTHON_PATH%" -m unittest discover -s tests -p test_blast_*.py >> "%LOG_FILE%" 2>&1
if errorlevel 1 exit /b 1
"%PYTHON_PATH%" scripts/blast_injection_demo.py >> "%LOG_FILE%" 2>&1
if errorlevel 1 exit /b 1
exit /b 0
