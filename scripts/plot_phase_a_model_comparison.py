#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


FEATURE_GROUPS = [
    "A1 (内部)",
    "A2 (内部+天气)",
    "A3 (内部+爆破)",
    "A4 (内部+爆破+天气)",
]

MODELS = [
    "原始 TimeFilter",
    "GNN",
    "DLinear",
    "PatchTST",
    "TFT",
    "TimesNet",
]

METRICS = ["MSE", "RMSE"]

CJK_FONT_CANDIDATES = [
    "Songti SC",
    "PingFang HK",
    "Hiragino Sans GB",
    "STHeiti",
    "Arial Unicode MS",
]

MODEL_COLORS: Dict[str, str] = {
    "原始 TimeFilter": "#2F5D73",
    "GNN": "#B7C8D6",
    "DLinear": "#C8D6C0",
    "PatchTST": "#E8DCC9",
    "TFT": "#D9C7C7",
    "TimesNet": "#D6D6D6",
}

TIMEFILTER_EDGE = "#1F3E4D"
SECOND_BEST_EDGE = "#8C6A43"
DEFAULT_EDGE = "#F7F7F7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready Phase A model comparison figure."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_model_comparison"),
        help="Directory for exported PNG/PDF figures.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG output dpi.")
    return parser.parse_args()


def setup_plot_style() -> None:
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    cjk_font = next(
        (name for name in CJK_FONT_CANDIDATES if name in available_fonts),
        "DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": [cjk_font, "Times New Roman", "DejaVu Serif"],
            "font.serif": [cjk_font, "Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "grid.color": "#D0D0D0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "axes.facecolor": "#FAFAFA",
        }
    )


def build_dataframe() -> pd.DataFrame:
    raw_records: List[Dict[str, object]] = [
        {"feature_group": "A1 (内部)", "model": "原始 TimeFilter", "MSE": 0.1041, "RMSE": 0.3226},
        {"feature_group": "A1 (内部)", "model": "GNN", "MSE": 0.1535, "RMSE": 0.3918},
        {"feature_group": "A1 (内部)", "model": "DLinear", "MSE": 0.1527, "RMSE": 0.3908},
        {"feature_group": "A1 (内部)", "model": "PatchTST", "MSE": 0.1418, "RMSE": 0.3766},
        {"feature_group": "A1 (内部)", "model": "TFT", "MSE": 0.1500, "RMSE": 0.3873},
        {"feature_group": "A1 (内部)", "model": "TimesNet", "MSE": 0.1512, "RMSE": 0.3888},
        {"feature_group": "A2 (内部+天气)", "model": "原始 TimeFilter", "MSE": 0.1025, "RMSE": 0.3202},
        {"feature_group": "A2 (内部+天气)", "model": "GNN", "MSE": 0.1667, "RMSE": 0.4083},
        {"feature_group": "A2 (内部+天气)", "model": "DLinear", "MSE": 0.1563, "RMSE": 0.3953},
        {"feature_group": "A2 (内部+天气)", "model": "PatchTST", "MSE": 0.1439, "RMSE": 0.3793},
        {"feature_group": "A2 (内部+天气)", "model": "TFT", "MSE": 0.1493, "RMSE": 0.3864},
        {"feature_group": "A2 (内部+天气)", "model": "TimesNet", "MSE": 0.1634, "RMSE": 0.4042},
        {"feature_group": "A3 (内部+爆破)", "model": "原始 TimeFilter", "MSE": 0.1002, "RMSE": 0.3165},
        {"feature_group": "A3 (内部+爆破)", "model": "GNN", "MSE": 0.1457, "RMSE": 0.3817},
        {"feature_group": "A3 (内部+爆破)", "model": "DLinear", "MSE": 0.1581, "RMSE": 0.3976},
        {"feature_group": "A3 (内部+爆破)", "model": "PatchTST", "MSE": 0.1411, "RMSE": 0.3756},
        {"feature_group": "A3 (内部+爆破)", "model": "TFT", "MSE": 0.1434, "RMSE": 0.3787},
        {"feature_group": "A3 (内部+爆破)", "model": "TimesNet", "MSE": 0.1514, "RMSE": 0.3891},
        {"feature_group": "A4 (内部+爆破+天气)", "model": "原始 TimeFilter", "MSE": 0.1007, "RMSE": 0.3173},
        {"feature_group": "A4 (内部+爆破+天气)", "model": "GNN", "MSE": 0.1717, "RMSE": 0.4144},
        {"feature_group": "A4 (内部+爆破+天气)", "model": "DLinear", "MSE": 0.1562, "RMSE": 0.3952},
        {"feature_group": "A4 (内部+爆破+天气)", "model": "PatchTST", "MSE": 0.1436, "RMSE": 0.3789},
        {"feature_group": "A4 (内部+爆破+天气)", "model": "TFT", "MSE": 0.1431, "RMSE": 0.3783},
        {"feature_group": "A4 (内部+爆破+天气)", "model": "TimesNet", "MSE": 0.1613, "RMSE": 0.4016},
    ]

    wide_df = pd.DataFrame(raw_records)
    long_df = wide_df.melt(
        id_vars=["feature_group", "model"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    )
    long_df["feature_group"] = pd.Categorical(
        long_df["feature_group"],
        categories=FEATURE_GROUPS,
        ordered=True,
    )
    long_df["model"] = pd.Categorical(
        long_df["model"],
        categories=MODELS,
        ordered=True,
    )
    long_df = long_df.sort_values(["metric", "feature_group", "model"]).reset_index(drop=True)
    long_df["rank_in_group"] = (
        long_df.groupby(["feature_group", "metric"], observed=False)["value"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    return long_df


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_metric_panel(ax: plt.Axes, data: pd.DataFrame, metric: str, show_y_labels: bool) -> None:
    panel_df = data.loc[data["metric"] == metric].copy()

    bar_height = 0.11
    group_gap = 0.58
    centers = np.arange(len(FEATURE_GROUPS)) * (len(MODELS) * bar_height + group_gap)
    offsets = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2.0) * bar_height

    metric_max = float(panel_df["value"].max())
    x_padding = max(metric_max * 0.18, 0.018)

    for group_index, feature_group in enumerate(FEATURE_GROUPS):
        group_data = (
            panel_df.loc[panel_df["feature_group"] == feature_group]
            .sort_values("model")
            .reset_index(drop=True)
        )
        base_y = centers[group_index]

        for model_index, row in group_data.iterrows():
            model = str(row["model"])
            value = float(row["value"])
            rank = int(row["rank_in_group"])
            y_pos = base_y + offsets[model_index]

            edgecolor = DEFAULT_EDGE
            linewidth = 0.7
            zorder = 3

            if model == "原始 TimeFilter":
                edgecolor = TIMEFILTER_EDGE
                linewidth = 1.5
                zorder = 4
            elif rank == 2:
                edgecolor = SECOND_BEST_EDGE
                linewidth = 1.2

            ax.barh(
                y=y_pos,
                width=value,
                height=bar_height * 0.88,
                color=MODEL_COLORS[model],
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=0.98,
                zorder=zorder,
            )

            if model == "原始 TimeFilter" or rank == 2:
                label_color = TIMEFILTER_EDGE if model == "原始 TimeFilter" else SECOND_BEST_EDGE
                ax.text(
                    value + x_padding * 0.12,
                    y_pos,
                    f"{value:.4f}",
                    va="center",
                    ha="left",
                    fontsize=9,
                    color=label_color,
                )

    panel_tag = "a" if metric == "MSE" else "b"
    ax.set_title(f"({panel_tag}) {metric}")
    ax.set_xlabel(metric)
    ax.set_yticks(centers)
    if show_y_labels:
        ax.set_yticklabels(FEATURE_GROUPS)
        ax.set_ylabel("特征组")
    else:
        ax.set_yticklabels(FEATURE_GROUPS)
        ax.tick_params(axis="y", length=0, labelleft=False)
    ax.set_xlim(0.0, metric_max + x_padding)
    ax.grid(axis="x", alpha=0.9)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", pad=8)
    ax.invert_yaxis()


def build_legend_handles() -> List[Patch]:
    handles = []
    for model in MODELS:
        edgecolor = TIMEFILTER_EDGE if model == "原始 TimeFilter" else "#FFFFFF"
        linewidth = 1.3 if model == "原始 TimeFilter" else 0.8
        handles.append(
            Patch(
                facecolor=MODEL_COLORS[model],
                edgecolor=edgecolor,
                linewidth=linewidth,
                label=model,
            )
        )
    return handles


def main() -> None:
    args = parse_args()
    setup_plot_style()
    data = build_dataframe()

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6), sharey=True)
    plot_metric_panel(axes[0], data, metric="MSE", show_y_labels=True)
    plot_metric_panel(axes[1], data, metric="RMSE", show_y_labels=False)

    fig.legend(
        handles=build_legend_handles(),
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.2,
        handlelength=1.6,
    )
    fig.subplots_adjust(top=0.82, bottom=0.10, left=0.12, right=0.98, wspace=0.10)

    save_figure(fig, args.output_dir, "phase_a_model_comparison", args.dpi)


if __name__ == "__main__":
    main()
