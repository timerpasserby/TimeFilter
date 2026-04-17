# TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting


## 📰 News

🚩 2025-05-01: TimeFilter has been accepted as ICML 2025 Poster.

🚩 2025-01-22: Initial upload to arXiv [PDF](https://arxiv.org/pdf/2501.13041).

## 🌟 Overview

TimeFilter is a cutting-edge solution for time series forecasting, incorporating three main components: the **Spatial-Temporal Construction** Module, the **Patch-Specific Filtration** Module, and the **Adaptive Graph Learning** Module.

![](./assets/pipline.png)

## 🛠 Prerequisites

Ensure you are using Python 3.10.16 and install the necessary dependencies by running:

```
pip install -r requirements.txt
```

## 📊 Prepare Datastes

Begin by downloading the required datasets. All datasets are conveniently available at [iTransformer](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link). Create a separate folder named `./data` and neatly organize all the csv files as shown below:
```
data
└── electricity.csv
└── ETTh1.csv
└── ETTh2.csv
└── ETTm1.csv
└── ETTm2.csv
└── traffic.csv
└── weather.csv
└── solar_AL.txt
└── PEMS03.npz
└── PEMS04.npz
└── PEMS07.npz
└── PEMS08.npz
```

## 💻 Training

All scripts are located in `./scripts`. For instance, to train a model using the ETTh1 dataset with an input length of 96, simply run:

```shell
bash ./scripts/ETTh1.sh
```

After training:

- Your trained model will be safely stored in `./checkpoints`.
- Numerical results in .npy format can be found in `./results`.
- A comprehensive summary of quantitative metrics is accessible in `./result_long_term_forecast.txt`.
- The long-term forecasting experiment now also saves `input.npy`, `pred.npy`, `true.npy`, `metrics.npy`, and optional DTW statistics when enabled.

### Radar custom run

The radar task now reads its data directly from `dataset/radar/`:

- sequence file: `dataset/radar/sim_radar_hourly_displacement.csv`
- coordinate file: `dataset/radar/sim_nodes_static.csv`

`run_models.sh` is wired for this radar dataset. It uses `features=M`, which means multivariate input and multivariate output, so the script does not need to pass `target`.
At the moment, `run_models.sh` loops through `pred_len=12 24`.
For non-M4 custom data, `short_term_forecast` reuses the same supervised forecasting loop as the long-term task, so the radar dataset can be launched with either task name.

For Windows, the repository now also provides batch-script entrypoints with a fixed interpreter path:

- `run_models_windows.bat`
- `scripts/radar_ablation_pipeline_windows.bat`

Their default Python path is:

```text
D:\Anaconda\envs\TLib\python.exe
```

If you want one entrypoint for the currently available ablation pipelines, use:

```shell
bash scripts/radar_ablation_pipeline.sh help
```

This script groups:

- formal train-and-test ablations that already connect to `run.py`: `baseline / CSP-TimeFilter`
- module-level checks that are not yet wired into the full training graph: `weather / blast`
- default `pred_len` coverage `12 24 48 96`

To launch it in the background:

```shell
nohup bash run_models.sh &
```

The full training output is written to `logs/run_models_*.log`.

You can also switch the task type or enable CSPAdapter from the shell:

```shell
TASK_NAME=short_term_forecast USE_CSP_ADAPTER=1 CSP_DEBUG=1 TOP_P=0.5 nohup bash run_models.sh &
```

To launch the organized ablation pipeline in the background:

```shell
nohup bash scripts/radar_ablation_pipeline.sh train_all &
```

Windows examples:

```bat
run_models_windows.bat
scripts\radar_ablation_pipeline_windows.bat help
scripts\radar_ablation_pipeline_windows.bat train_all
```

### CSPAdapter smoke test

The repository now includes a minimal script-level verification for the unified spatial module:

```shell
/opt/homebrew/Caskroom/miniforge/base/envs/tslib/bin/python scripts/csp_adapter_smoke.py --steps 3 --debug
```

This script checks:

- dummy shape compatibility
- baseline compatibility against the original committed model when `use_csp_adapter=False`
- single-batch forward/backward for `use_csp_adapter=False/True`
- a short real-data smoke training run on `dataset/radar`
- key debug shapes, mask ratios, adjacency statistics, and forward/backward timing

The short-term forecasting experiment has also been aligned with the reference M4 workflow while keeping compatibility with the current TimeFilter direct-forecast interface.

### Physics-Constrained Weather Injection

The repository now includes an independent weather module for slow-varying exogenous effects:

- main block: `models/weather/weather_injection_block.py`
- causal weather encoder: `models/weather/causal_conv.py`
- causal weather cross-attention: `models/weather/causal_cross_attention.py`
- module guide: `models/weather/README.md`

This block is designed to sit after the CSP-TimeFilter backbone output `H_main` and does not modify the backbone itself.

Run the weather module tests with:

```shell
/opt/homebrew/Caskroom/miniforge/base/envs/tslib/bin/python -m unittest discover -s tests -p "test_weather_*.py"
```

Run the minimal weather forward demo with:

```shell
/opt/homebrew/Caskroom/miniforge/base/envs/tslib/bin/python scripts/weather_injection_demo.py
```

### Physics-Informed Blast Injection

The repository now includes an independent blast module for transient disturbance modeling:

- main block: `models/blast/blast_injection_block.py`
- analytic blast encoder: `models/blast/blast_analytic_encoder.py`
- step-response gate and bypass residual: `models/blast/step_response_gate.py`
- module guide: `models/blast/README.md`

This block is designed to sit after the weather-enhanced hidden state `H_exo` and keeps the CSP-TimeFilter backbone unchanged.

Run the blast module tests with:

```shell
/opt/homebrew/Caskroom/miniforge/base/envs/tslib/bin/python -m unittest discover -s tests -p "test_blast_*.py"
```

Run the minimal blast forward demo with:

```shell
/opt/homebrew/Caskroom/miniforge/base/envs/tslib/bin/python scripts/blast_injection_demo.py
```

## 📚 Citation
If you find this repo useful, please consider citing our paper as follows:
```bibtex
@inproceedings{
hu2025timefilter,
title={TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting},
author={Yifan Hu and Guibin Zhang and Peiyuan Liu and Disen Lan and Naiqi Li and Dawei Cheng and Tao Dai and Shu-Tao Xia and Shirui Pan},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=490VcNtjh7}
}
```

## 🙏 Acknowledgement
Special thanks to the following repositories for their invaluable code and datasets:

- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [DUET](https://github.com/decisionintelligence/DUET)
- [TimeSeriesCCM](https://github.com/Graph-and-Geometric-Learning/TimeSeriesCCM)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [iTransformer](https://github.com/thuml/iTransformer)

## 📩 Contact
If you have any questions, please contact [huyf0122@gmail.com](huyf0122@gmail.com) or submit an issue.
