# Architecture Overview

## Core Flow

`run_models.sh` starts the radar experiment, `run.py` parses arguments, `exp/exp_long_term_forecasting.py` runs training and testing, and `data_provider/` prepares the sequences that are fed into `models/TimeFilter.py`.
When `use_csp_adapter=True`, `models/csp_adapter.py` injects the spatial prior after patch embedding and passes a physical radius mask into the graph-learning backbone.

## Module Responsibilities

- `run_models.sh`
  - Launches the radar experiment.
  - Reads the radar sequence table from `dataset/radar/sim_radar_hourly_displacement.csv`.
  - Reads node coordinates from `dataset/radar/sim_nodes_static.csv`.
  - Loops through multiple prediction lengths for the radar task.
  - Writes logs to `logs/run_models_*.log`.
  - Runs in the background when started with `nohup`.
  - Accepts environment overrides such as `TASK_NAME`, `USE_CSP_ADAPTER`, `CSP_DEBUG`, `PHYSICAL_MASK_RADIUS`, and `TOP_P`.

- `run.py`
  - Defines the command-line interface.
  - Chooses the experiment class based on `task_name`.
  - Routes non-M4 `short_term_forecast` runs to the general supervised forecasting flow.
  - Sets the model and training loop configuration.

- `data_provider/data_factory.py`
  - Selects the dataset implementation for each dataset name.
  - Creates the PyTorch `DataLoader` objects used by training and testing.

- `data_provider/data_loader.py`
  - Reads and splits the raw data.
  - Builds training, validation, and test sequences.
  - For `custom` data, allows `features=M` to use all monitoring points directly, even when `target` is not present.
  - Accepts common time-column aliases such as `report_time` and normalizes them to `date`.

- `exp/exp_basic.py`
  - Creates the model and selects the device.

- `exp/exp_long_term_forecasting.py`
  - Implements the long-term forecasting training loop.
  - Handles validation, checkpoint saving, and testing.

- `models/TimeFilter.py`
  - Defines the TimeFilter model.
  - Consumes the masks prepared by the experiment class.
  - Loads coordinate files when CSPAdapter is enabled.
  - Injects spatial prompts after patch embedding and forwards the physical mask into the backbone.
  - Exposes compact debug information through `return_debug` and `csp_debug`.

- `models/csp_adapter.py`
  - Implements the unified spatial module.
  - Encodes centered 3D coordinates with RFF + MLP.
  - Injects patch-level spatial prompts.
  - Builds the physical radius mask used to suppress long-distance false edges.

- `layers/TimeFilter_layers.py`
  - Keeps the original graph-learning pipeline.
  - Applies the optional physical mask during adjacency construction and routing.
  - Records adjacency and routing statistics for debugging.

- `scripts/csp_adapter_smoke.py`
  - Runs the minimal verification suite for baseline compatibility and CSP training stability.
  - Prints shapes, mask ratios, adjacency statistics, and forward/backward timing.

- `utils/tools.py`
  - Provides early stopping, learning-rate scheduling, and simple visualization helpers.

## Key Decisions

- Radar runs use `features=M` so the model predicts all monitored points together.
- The custom dataset loader no longer requires `target` when `features=M`.
- Radar custom data now comes from `dataset/radar/`, not `data/`.
- The radar coordinate prior is sourced from `dataset/radar/sim_nodes_static.csv`.
- Radar batch runs now iterate over `pred_len=12,24,48,96` in one script.
- `use_csp_adapter=False` keeps the committed baseline path unchanged.
- `use_csp_adapter=True` adds coordinate encoding, prompt injection, and physical radius masking without rewriting the TimeFilter backbone.
- CPU execution is used on this machine to avoid GPU/MPS compatibility issues.
- Logs are kept on disk so long-running runs can be checked after the shell exits.
