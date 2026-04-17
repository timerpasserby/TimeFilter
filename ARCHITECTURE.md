# Architecture Overview

## Core Flow

`run_models.sh` starts the radar experiment, `run.py` parses arguments, `exp/exp_long_term_forecasting.py` runs training and testing, and `data_provider/` prepares the sequences that are fed into `models/TimeFilter.py`.

## Module Responsibilities

- `run_models.sh`
  - Launches the radar experiment.
  - Loops through multiple prediction lengths for the radar task.
  - Writes logs to `logs/run_models_*.log`.
  - Runs in the background when started with `nohup`.

- `run.py`
  - Defines the command-line interface.
  - Chooses the experiment class based on `task_name`.
  - Sets the model and training loop configuration.

- `data_provider/data_factory.py`
  - Selects the dataset implementation for each dataset name.
  - Creates the PyTorch `DataLoader` objects used by training and testing.

- `data_provider/data_loader.py`
  - Reads and splits the raw data.
  - Builds training, validation, and test sequences.
  - For `custom` data, allows `features=M` to use all monitoring points directly, even when `target` is not present.

- `exp/exp_basic.py`
  - Creates the model and selects the device.

- `exp/exp_long_term_forecasting.py`
  - Implements the long-term forecasting training loop.
  - Handles validation, checkpoint saving, and testing.

- `models/TimeFilter.py`
  - Defines the TimeFilter model.
  - Consumes the masks prepared by the experiment class.

- `utils/tools.py`
  - Provides early stopping, learning-rate scheduling, and simple visualization helpers.

## Key Decisions

- Radar runs use `features=M` so the model predicts all monitored points together.
- The custom dataset loader no longer requires `target` when `features=M`.
- Radar batch runs now iterate over `pred_len=12,24,48,96` in one script.
- CPU execution is used on this machine to avoid GPU/MPS compatibility issues.
- Logs are kept on disk so long-running runs can be checked after the shell exits.
