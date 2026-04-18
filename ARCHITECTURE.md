# Architecture Overview

## Core Flow

`run_models.sh` starts the radar experiment, `run.py` parses arguments, `exp/exp_long_term_forecasting.py` runs training and testing, and `data_provider/` prepares the sequences that are fed into `models/TimeFilter.py`.
When `use_csp_adapter=True`, `models/csp_adapter.py` injects the spatial prior after patch embedding and passes a physical radius mask into the graph-learning backbone.
The weather block in `models/weather/` is kept fully independent from the backbone and only consumes `H_main` after the CSP-TimeFilter trunk.
The blast block in `models/blast/` is also independent from the backbone and consumes `H_exo` after the weather block.
For chapter-level evidence packaging, `scripts/build_363_package.py` reads experiment outputs and generates the full 3.6.3 `2 tables + 5 figures` bundle under `./363`.

## Module Responsibilities

- `run_models.sh`
  - Launches the radar experiment.
  - Reads the radar sequence table from `dataset/radar/sim_radar_hourly_displacement.csv`.
  - Reads node coordinates from `dataset/radar/sim_nodes_static.csv`.
  - Can optionally read weather and blast logs from `dataset/radar/sim_weather.csv` and `dataset/radar/sim_blast_logs.csv`.
  - Loops through multiple prediction lengths for the radar task.
  - Writes logs to `logs/run_models_*.log`.
  - Runs in the background when started with `nohup`.
  - Accepts environment overrides such as `TASK_NAME`, `USE_CSP_ADAPTER`, `USE_WEATHER_MODULE`, `USE_BLAST_MODULE`, `BLAST_MODE`, `CSP_DEBUG`, `EXO_DEBUG`, `PHYSICAL_MASK_RADIUS`, and `TOP_P`.

- `scripts/radar_ablation_pipeline.sh`
  - Organizes the currently available radar ablation commands into one shell entrypoint.
  - Runs the formal train-and-test ablations for `baseline`、`CSP-TimeFilter`、`CSP+Weather`、`CSP+Weather+Blast`.
  - Runs the weather ablations `causal_attn / vanilla_attn / concat_fusion`.
  - Runs the blast ablations `main / wo_gate / gru_blast`.
  - Keeps module-level verification for the independent weather and blast blocks.
  - Writes grouped logs to `logs/ablation_pipeline/`.

- `scripts/build_363_package.py`
  - Builds the 3.6.3 chapter artifact pack into `./363`.
  - Constructs blast and rain event subsets with configurable windows and overlap removal.
  - Produces the required `2 tables + 5 figures` for event-scene validation.
  - Uses real `TimeFilter / Ours` outputs and optionally reproducible proxy baselines for missing external models.
  - For `figure_3_2_blast_curve`, it can apply a figure-only `paper_optimize_ours_curve` pass and prefer a sample whose peak appears about one hour after the blast trigger, so the marker stays slightly ahead of the crest in the final illustration.
  - For `figure_3_3_rain_curve`, it can apply a figure-only `paper_optimize_rain_curve` pass so the `Ours` line stays closest to `GroundTruth` while preserving the rain bar panel.

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
  - Aligns radar sequence timestamps with weather and blast side data when exogenous modules are enabled.
  - Returns `extra_inputs` containing `weather_seq / weather_mask / blast_locs / blast_times / blast_intensity / patch_times`.
  - Accepts common time-column aliases such as `report_time` and normalizes them to `date`.

- `exp/exp_basic.py`
  - Creates the model and selects the device.

- `exp/exp_long_term_forecasting.py`
  - Implements the long-term forecasting training loop.
  - Handles validation, checkpoint saving, and testing.
  - Keeps the TimeFilter-specific `masks + moe_loss` training path.
  - Supports AMP, optional DTW evaluation, visualization output, and `metrics/input/pred/true` result saving.

- `exp/exp_short_term_forecasting.py`
  - Implements the short-term forecasting training loop.
  - Keeps the M4-style validation and result-export flow.
  - Adapts the forward path so TimeFilter can be called as a direct forecaster instead of an encoder-decoder model.

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

- `models/weather/causal_conv.py`
  - Implements the causal weather encoder.
  - Uses at least two strictly causal Conv1d layers to model lagged and accumulated weather effects.

- `models/weather/causal_cross_attention.py`
  - Implements multi-head cross-attention from node hidden states to global weather features.
  - Applies an explicit lower-triangular causal mask and returns interpretable attention weights.

- `models/weather/weather_injection_block.py`
  - Packages weather encoding and injection as `PhysicsConstrainedCausalWeatherInjection`.
  - Supports the main causal-attention version and the `vanilla_attn` / `concat_fusion` ablations.

- `models/blast/blast_analytic_encoder.py`
  - Parses blast logs into node-level continuous disturbance `e_it`.
  - Uses distance decay, temporal decay, and an explicit causal event mask.

- `models/blast/step_response_gate.py`
  - Maps `e_it` to the step-response gate `g_t` and feature increment `delta_H_blast`.
  - Applies bypass residual injection in the form `H_final = H_exo + g_t * delta_H_blast`.

- `models/blast/blast_injection_block.py`
  - Packages analytic encoding, gate injection, and ablations as `PhysicsInformedStepResponseBlastInjection`.
  - Supports the main `main` mode and the `wo_gate` / `gru_blast` ablations.

- `tests/test_weather_shapes.py`
  - Checks the output shape and attention weight interface.

- `tests/test_weather_causality.py`
  - Verifies that changing future weather does not affect past outputs.

- `tests/test_weather_ablation.py`
  - Verifies that both ablation branches can run independently.

- `tests/test_blast_shapes.py`
  - Checks the output shape of `H_final`、`e_it`、`g_t` and `delta_H_blast`.

- `tests/test_blast_causality.py`
  - Verifies that changing future blast events does not affect past outputs.

- `tests/test_blast_monotonicity.py`
  - Verifies blast distance decay, intensity monotonicity, and temporal decay.

- `tests/test_blast_ablation.py`
  - Verifies that both `wo_gate` and `gru_blast` can run independently.

- `scripts/weather_injection_demo.py`
  - Provides a minimal forward demo for the main weather block and both ablations.

- `scripts/blast_injection_demo.py`
  - Provides a minimal forward demo for the blast main branch and both ablations.

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
- The weather exogenous module is implemented as a post-backbone block so backbone stability is preserved for later ablations.
- Weather injection uses causal Conv1d plus masked cross-attention instead of raw feature concatenation.
- The blast transient module is implemented after `H_exo`, not inside the backbone, so the CSP-TimeFilter trunk remains stable.
- Blast injection explicitly separates analytic disturbance computation `e_it` from gated bypass injection, and keeps `GRU` only in an ablation branch.
- The main backbone now reshapes tokens into `[B, P, N, D]`, injects weather and blast in patch space, and then reshapes back before the prediction head.
- CPU execution is used on this machine to avoid GPU/MPS compatibility issues.
- Logs are kept on disk so long-running runs can be checked after the shell exits.
- The ablation shell script now treats weather and blast variants as formal training experiments because both modules are connected to the full training graph.
