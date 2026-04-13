#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle


@dataclass(frozen=True)
class PlotConfig:
    csv_path: Path
    output_dir: Path
    patch_width: int = 24
    patch_height: int = 24
    time_block_hours: int = 24
    dpi: int = 300
    chunksize: int = 500_000


@dataclass(frozen=True)
class ZoneRuleConfig:
    background_area_ids: Tuple[int, ...]
    toe_y_max_ratio: float = 0.22
    crest_y_min_ratio: float = 0.74
    bench_y_min_ratio: float = 0.46
    bench_y_max_ratio: float = 0.58
    bench_x_min_ratio: float = 0.22
    bench_x_max_ratio: float = 0.92


ZONE_STYLES: Dict[str, Dict[str, str]] = {
    "stable_background": {"label": "稳定背景区", "color": "#D9D9D9"},
    "toe": {"label": "坡脚区", "color": "#F4A261"},
    "middle_slope": {"label": "坡中区", "color": "#2A9D8F"},
    "platform": {"label": "平台区", "color": "#E9C46A"},
    "crest": {"label": "坡顶区", "color": "#457B9D"},
}

PATCH_FACE_COLORS = ["#CFE8F3", "#F7D9C4", "#D7F0E4", "#FAEDCB"]
CJK_FONT_CANDIDATES = [
    "Songti SC",
    "PingFang HK",
    "Hiragino Sans GB",
    "STHeiti",
    "Arial Unicode MS",
]


def parse_args() -> PlotConfig:
    parser = argparse.ArgumentParser(
        description="Generate thesis-style spatial and spatiotemporal figures "
        "for radar-based slope monitoring data."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("/Users/dc/Z研究生/eedsProject/device_10001_sorted.csv"),
        help="Input CSV file path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/thesis_spatiotemporal_figures"),
        help="Directory for exported PNG/PDF figures.",
    )
    parser.add_argument("--patch-width", type=int, default=10, help="Patch width.")
    parser.add_argument("--patch-height", type=int, default=10, help="Patch height.")
    parser.add_argument(
        "--time-block-hours",
        type=int,
        default=24,
        help="Number of hourly steps per temporal block.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG output dpi.")
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="Chunk size for large-file aggregation.",
    )
    args = parser.parse_args()
    return PlotConfig(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        patch_width=args.patch_width,
        patch_height=args.patch_height,
        time_block_hours=args.time_block_hours,
        dpi=args.dpi,
        chunksize=args.chunksize,
    )


def setup_plot_style() -> None:
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    cjk_font = next((name for name in CJK_FONT_CANDIDATES if name in available_fonts), "DejaVu Sans")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": [cjk_font, "Times New Roman", "DejaVu Serif"],
            "font.serif": [cjk_font, "Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "grid.color": "#D0D0D0",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_unique_points(csv_path: Path) -> pd.DataFrame:
    dtypes = {"grid_x": "int16", "grid_y": "int16", "area_id": "int16"}
    points = pd.read_csv(csv_path, usecols=["grid_x", "grid_y", "area_id"], dtype=dtypes)
    points = points.drop_duplicates(subset=["grid_x", "grid_y"]).reset_index(drop=True)
    return points


def load_unique_times(csv_path: Path, chunksize: int) -> pd.DatetimeIndex:
    unique_times = set()
    for chunk in pd.read_csv(csv_path, usecols=["report_time"], chunksize=chunksize):
        unique_times.update(chunk["report_time"].dropna().unique().tolist())
    times = pd.to_datetime(sorted(unique_times))
    return pd.DatetimeIndex(times)


def detect_background_area_ids(points: pd.DataFrame, share_threshold: float = 0.01) -> Tuple[int, ...]:
    if "area_id" not in points.columns or points["area_id"].nunique() <= 1:
        return ()
    shares = points["area_id"].value_counts(normalize=True)
    background_ids = tuple(int(idx) for idx, share in shares.items() if share <= share_threshold)
    return background_ids


def assign_engineering_zone(points: pd.DataFrame, zone_rules: ZoneRuleConfig) -> pd.DataFrame:
    frame = points.copy()
    x_min, x_max = frame["grid_x"].min(), frame["grid_x"].max()
    y_min, y_max = frame["grid_y"].min(), frame["grid_y"].max()
    x_range = max(x_max - x_min, 1)
    y_range = max(y_max - y_min, 1)

    x_norm = (frame["grid_x"] - x_min) / x_range
    y_norm = (frame["grid_y"] - y_min) / y_range

    zone = np.full(len(frame), "middle_slope", dtype=object)

    if zone_rules.background_area_ids:
        background_mask = frame["area_id"].isin(zone_rules.background_area_ids).to_numpy()
        zone[background_mask] = "stable_background"

    platform_mask = (
        (y_norm >= zone_rules.bench_y_min_ratio)
        & (y_norm <= zone_rules.bench_y_max_ratio)
        & (x_norm >= zone_rules.bench_x_min_ratio)
        & (x_norm <= zone_rules.bench_x_max_ratio)
    )
    zone[platform_mask & (zone != "stable_background")] = "platform"

    crest_mask = y_norm >= zone_rules.crest_y_min_ratio
    zone[crest_mask & (zone != "stable_background")] = "crest"

    toe_mask = y_norm <= zone_rules.toe_y_max_ratio
    zone[toe_mask & (zone == "middle_slope")] = "toe"

    frame["zone"] = zone
    return frame


def prepare_patch_summary(points: pd.DataFrame, config: PlotConfig) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    frame = points.copy()
    x_min, x_max = int(frame["grid_x"].min()), int(frame["grid_x"].max())
    y_min, y_max = int(frame["grid_y"].min()), int(frame["grid_y"].max())

    frame["patch_col"] = ((frame["grid_x"] - x_min) // config.patch_width).astype(int)
    frame["patch_row"] = ((frame["grid_y"] - y_min) // config.patch_height).astype(int)

    patch_summary = (
        frame.groupby(["patch_row", "patch_col"], as_index=False)
        .agg(
            n_points=("grid_x", "size"),
            dominant_zone=("zone", lambda s: s.mode().iat[0]),
        )
        .reset_index(drop=True)
    )

    patch_summary["fill_ratio"] = patch_summary["n_points"] / (
        config.patch_width * config.patch_height
    )

    total_cols = math.ceil((x_max - x_min + 1) / config.patch_width)
    total_rows = math.ceil((y_max - y_min + 1) / config.patch_height)

    patch_summary["center_dist"] = np.hypot(
        patch_summary["patch_col"] - (total_cols - 1) / 2.0,
        patch_summary["patch_row"] - (total_rows - 1) / 2.0,
    )
    max_n_points = max(int(patch_summary["n_points"].max()), 1)
    max_center_dist = max(float(patch_summary["center_dist"].max()), 1.0)
    patch_summary["score"] = (
        0.70 * patch_summary["fill_ratio"]
        + 0.30 * (patch_summary["n_points"] / max_n_points)
        - 0.10 * (patch_summary["center_dist"] / max_center_dist)
    )

    candidate_patches = patch_summary.loc[
        patch_summary["dominant_zone"] != "stable_background"
    ].copy()
    if candidate_patches.empty:
        candidate_patches = patch_summary.copy()

    selected_patch = candidate_patches.sort_values(
        ["score", "n_points"], ascending=[False, False]
    ).iloc[0]

    selected_patch_info = {
        "patch_row": int(selected_patch["patch_row"]),
        "patch_col": int(selected_patch["patch_col"]),
        "x_min": x_min + int(selected_patch["patch_col"]) * config.patch_width,
        "x_max": min(
            x_min + (int(selected_patch["patch_col"]) + 1) * config.patch_width - 1,
            x_max,
        ),
        "y_min": y_min + int(selected_patch["patch_row"]) * config.patch_height,
        "y_max": min(
            y_min + (int(selected_patch["patch_row"]) + 1) * config.patch_height - 1,
            y_max,
        ),
        "total_cols": total_cols,
        "total_rows": total_rows,
        "global_x_min": x_min,
        "global_x_max": x_max,
        "global_y_min": y_min,
        "global_y_max": y_max,
    }
    return frame, patch_summary, selected_patch_info


def select_time_block(times: pd.DatetimeIndex, hours_per_block: int) -> Dict[str, object]:
    if len(times) == 0:
        raise ValueError("No report_time values found in the input file.")

    block_len = max(int(hours_per_block), 1)
    n_blocks = math.ceil(len(times) / block_len)
    selected_idx = n_blocks // 2
    start_idx = selected_idx * block_len
    end_idx = min((selected_idx + 1) * block_len, len(times))
    if start_idx >= len(times):
        start_idx = max(len(times) - block_len, 0)
        end_idx = len(times)

    return {
        "index": selected_idx + 1,
        "n_blocks": n_blocks,
        "start": times[start_idx],
        "end": times[end_idx - 1],
        "length": end_idx - start_idx,
    }


def load_patch_block_statistics(
    config: PlotConfig, selected_patch: Dict[str, float], time_block: Dict[str, object]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    usecols = [
        "grid_x",
        "grid_y",
        "report_time",
        "deformation",
        "speed",
        "acceleration",
    ]
    dtypes = {
        "grid_x": "int16",
        "grid_y": "int16",
        "deformation": "float32",
        "speed": "float32",
        "acceleration": "float32",
    }

    filtered_chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        config.csv_path,
        usecols=usecols,
        dtype=dtypes,
        parse_dates=["report_time"],
        chunksize=config.chunksize,
    ):
        mask = (
            chunk["grid_x"].between(selected_patch["x_min"], selected_patch["x_max"])
            & chunk["grid_y"].between(selected_patch["y_min"], selected_patch["y_max"])
            & chunk["report_time"].between(time_block["start"], time_block["end"])
        )
        if mask.any():
            filtered_chunks.append(chunk.loc[mask].copy())

    if not filtered_chunks:
        raise ValueError("No samples found inside the selected patch and temporal block.")

    patch_block = pd.concat(filtered_chunks, ignore_index=True)
    point_summary = (
        patch_block.groupby(["grid_x", "grid_y"], as_index=False)
        .agg(
            deformation=("deformation", "mean"),
            speed=("speed", "mean"),
            acceleration=("acceleration", "mean"),
            valid_steps=("report_time", "nunique"),
        )
        .sort_values(["grid_x", "grid_y"])
        .reset_index(drop=True)
    )

    expected_points = (
        (selected_patch["x_max"] - selected_patch["x_min"] + 1)
        * (selected_patch["y_max"] - selected_patch["y_min"] + 1)
    )
    expected_steps = max(int(time_block["length"]), 1)

    stats = {
        "n_unique_points": int(point_summary.shape[0]),
        "valid_point_ratio": float(point_summary.shape[0] / expected_points),
        "temporal_coverage_ratio": float(
            point_summary["valid_steps"].mean() / expected_steps
        ),
        "deformation_mean": float(patch_block["deformation"].mean()),
        "deformation_std": float(patch_block["deformation"].std()),
        "speed_mean": float(patch_block["speed"].mean()),
        "speed_std": float(patch_block["speed"].std()),
        "acceleration_mean": float(patch_block["acceleration"].mean()),
        "acceleration_std": float(patch_block["acceleration"].std()),
    }
    return point_summary, stats


def style_spatial_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, pad=10)
    ax.set_xlabel("grid_x（投影横坐标）")
    ax.set_ylabel("grid_y（投影纵坐标）")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.5)
    ax.set_facecolor("#FCFCFC")


def plot_raw_points(points: pd.DataFrame, config: PlotConfig) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ax.scatter(
        points["grid_x"],
        points["grid_y"],
        s=3.5,
        c="#355070",
        alpha=0.72,
        linewidths=0,
    )
    style_spatial_axes(ax, "原始二维投影监测点分布")
    save_figure(fig, config.output_dir, "fig01_raw_projected_points", config.dpi)


def build_zone_raster(points: pd.DataFrame) -> Tuple[np.ma.MaskedArray, List[str], List[str], Tuple[float, float, float, float]]:
    x_min, x_max = int(points["grid_x"].min()), int(points["grid_x"].max())
    y_min, y_max = int(points["grid_y"].min()), int(points["grid_y"].max())
    zone_keys = list(ZONE_STYLES.keys())
    zone_to_idx = {key: idx for idx, key in enumerate(zone_keys)}

    raster = np.full((y_max - y_min + 1, x_max - x_min + 1), np.nan)
    raster[
        points["grid_y"].to_numpy() - y_min,
        points["grid_x"].to_numpy() - x_min,
    ] = points["zone"].map(zone_to_idx).to_numpy()

    masked = np.ma.masked_invalid(raster)
    colors = [ZONE_STYLES[key]["color"] for key in zone_keys]
    extent = (x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5)
    return masked, zone_keys, colors, extent


def plot_engineering_zones(points: pd.DataFrame, config: PlotConfig) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    zone_raster, zone_keys, colors, extent = build_zone_raster(points)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(zone_keys) + 0.5, 1), cmap.N)
    ax.imshow(
        zone_raster,
        origin="lower",
        interpolation="nearest",
        extent=extent,
        cmap=cmap,
        norm=norm,
        alpha=0.55,
    )
    ax.scatter(
        points["grid_x"],
        points["grid_y"],
        s=1.8,
        c="#222222",
        alpha=0.40,
        linewidths=0,
    )
    style_spatial_axes(ax, "边坡工程分区示意图")
    handles = [
        Patch(
            facecolor=ZONE_STYLES[key]["color"],
            edgecolor="none",
            alpha=0.70,
            label=ZONE_STYLES[key]["label"],
        )
        for key in zone_keys
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95)
    save_figure(fig, config.output_dir, "fig02_engineering_zonation", config.dpi)


def plot_patch_partition(
    points: pd.DataFrame,
    patch_summary: pd.DataFrame,
    selected_patch: Dict[str, float],
    config: PlotConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    ax.scatter(
        points["grid_x"],
        points["grid_y"],
        s=1.6,
        c="#555555",
        alpha=0.38,
        linewidths=0,
    )

    for _, row in patch_summary.iterrows():
        x0 = selected_patch["global_x_min"] + int(row["patch_col"]) * config.patch_width
        y0 = selected_patch["global_y_min"] + int(row["patch_row"]) * config.patch_height
        width = min(config.patch_width, selected_patch["global_x_max"] - x0 + 1)
        height = min(config.patch_height, selected_patch["global_y_max"] - y0 + 1)
        rect = Rectangle(
            (x0 - 0.5, y0 - 0.5),
            width,
            height,
            facecolor=PATCH_FACE_COLORS[(int(row["patch_row"]) + int(row["patch_col"])) % len(PATCH_FACE_COLORS)],
            edgecolor="#355070",
            linewidth=0.9,
            alpha=0.20,
        )
        ax.add_patch(rect)

    selected_rect = Rectangle(
        (selected_patch["x_min"] - 0.5, selected_patch["y_min"] - 0.5),
        selected_patch["x_max"] - selected_patch["x_min"] + 1,
        selected_patch["y_max"] - selected_patch["y_min"] + 1,
        facecolor="none",
        edgecolor="#C1121F",
        linewidth=2.0,
    )
    ax.add_patch(selected_rect)
    ax.text(
        selected_patch["x_min"] + 1.0,
        selected_patch["y_max"] + 3.0,
        "代表性 patch",
        color="#C1121F",
        fontsize=10,
        ha="left",
        va="bottom",
    )

    style_spatial_axes(ax, "规则网格 patch 划分结果")
    handles = [
        Patch(facecolor="#CFE8F3", edgecolor="#355070", alpha=0.25, label="非空 patch"),
        Line2D([0], [0], color="#C1121F", linewidth=2.0, label="代表性 patch"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95)
    save_figure(fig, config.output_dir, "fig03_regular_patch_partition", config.dpi)


def plot_representative_patch(
    points: pd.DataFrame,
    point_summary: pd.DataFrame,
    patch_stats: Dict[str, float],
    selected_patch: Dict[str, float],
    time_block: Dict[str, object],
    config: PlotConfig,
) -> None:
    patch_w = selected_patch["x_max"] - selected_patch["x_min"] + 1
    patch_h = selected_patch["y_max"] - selected_patch["y_min"] + 1
    margin_x = patch_w * 0.8
    margin_y = patch_h * 0.7

    context_points = points.loc[
        points["grid_x"].between(selected_patch["x_min"] - margin_x, selected_patch["x_max"] + margin_x)
        & points["grid_y"].between(selected_patch["y_min"] - margin_y, selected_patch["y_max"] + margin_y)
    ].copy()

    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    ax.scatter(
        context_points["grid_x"],
        context_points["grid_y"],
        s=10,
        c="#C7C7C7",
        alpha=0.6,
        linewidths=0,
        label="周边监测点",
    )

    sc = ax.scatter(
        point_summary["grid_x"],
        point_summary["grid_y"],
        c=point_summary["deformation"],
        cmap="viridis",
        s=28,
        edgecolors="white",
        linewidths=0.25,
        zorder=3,
    )

    patch_rect = Rectangle(
        (selected_patch["x_min"] - 0.5, selected_patch["y_min"] - 0.5),
        patch_w,
        patch_h,
        facecolor="none",
        edgecolor="#1D3557",
        linewidth=2.2,
        zorder=4,
    )
    ax.add_patch(patch_rect)

    vector_x = selected_patch["x_max"] + patch_w * 0.95
    vector_y = selected_patch["y_min"] + patch_h * 0.05
    box_w = patch_w * 0.95
    box_h = patch_h * 0.13

    feature_items = [
        (r"$\mu_d$", patch_stats["deformation_mean"]),
        (r"$\sigma_d$", patch_stats["deformation_std"]),
        (r"$\mu_v$", patch_stats["speed_mean"]),
        (r"$\sigma_v$", patch_stats["speed_std"]),
        (r"$\mu_a$", patch_stats["acceleration_mean"]),
        (r"$\sigma_a$", patch_stats["acceleration_std"]),
        (r"$r_{valid}$", patch_stats["valid_point_ratio"]),
    ]

    ax.text(
        vector_x,
        vector_y + box_h * (len(feature_items) + 1.1),
        r"patch统计特征",
        fontsize=11,
        color="#111111",
        ha="left",
        va="bottom",
    )

    for idx, (symbol, value) in enumerate(feature_items):
        y = vector_y + box_h * (len(feature_items) - idx)
        rect = Rectangle(
            (vector_x, y),
            box_w,
            box_h * 0.78,
            facecolor="#E9F1F7" if idx % 2 == 0 else "#F6F6F6",
            edgecolor="#4A4A4A",
            linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(
            vector_x + box_w * 0.08,
            y + box_h * 0.40,
            f"{symbol} = {value:.4f}",
            fontsize=10,
            ha="left",
            va="center",
            color="#222222",
        )

    arrow = FancyArrowPatch(
        posA=(selected_patch["x_max"] + 0.8, selected_patch["y_min"] + patch_h / 2.0),
        posB=(vector_x - patch_w * 0.10, vector_y + box_h * 4.1),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.5,
        color="#1D3557",
    )
    ax.add_patch(arrow)

    # block_text = (
    #     "选定时间块\n"
    #     f"{pd.Timestamp(time_block['start']).strftime('%Y-%m-%d %H:%M')} 至\n"
    #     f"{pd.Timestamp(time_block['end']).strftime('%Y-%m-%d %H:%M')}\n"
    #     f"时间覆盖率 = {patch_stats['temporal_coverage_ratio']:.2%}"
    # )
    # ax.text(
    #     vector_x,
    #     selected_patch["y_min"] - patch_h * 0.35,
    #     block_text,
    #     fontsize=10,
    #     ha="left",
    #     va="top",
    #     color="#333333",
    # )

    ax.set_xlim(selected_patch["x_min"] - margin_x, vector_x + box_w * 1.45)
    ax.set_ylim(selected_patch["y_min"] - margin_y * 0.5, selected_patch["y_max"] + margin_y)
    ax.set_title("")  # 设置为空标题
    ax.set_xlabel("grid_x（投影横坐标）")
    ax.set_ylabel("grid_y（投影纵坐标）")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.90)
    cbar.set_label("所选时间块内平均位移")
    save_figure(fig, config.output_dir, "fig04_representative_patch_features", config.dpi)


def draw_tensor_stack(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    depth: float,
    layers: int,
    facecolor: str,
    edgecolor: str = "#264653",
) -> None:
    for idx in range(layers):
        dx = idx * depth
        dy = idx * depth * 0.65
        rect = Rectangle(
            (x + dx, y + dy),
            width,
            height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.0,
            alpha=0.18 + idx * 0.06,
        )
        ax.add_patch(rect)


def plot_spatiotemporal_blocking(
    patch_summary: pd.DataFrame,
    selected_patch: Dict[str, float],
    time_block: Dict[str, object],
    config: PlotConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)

    ax.text(6, 36, "空间分块", fontsize=13, color="#1F2933", ha="left")
    grid_x0, grid_y0 = 8, 14
    cell_w, cell_h = 5.0, 5.0
    demo_cols, demo_rows = min(selected_patch["total_cols"], 6), min(selected_patch["total_rows"], 4)
    sel_c = min(selected_patch["patch_col"], demo_cols - 1)
    sel_r = min(selected_patch["patch_row"], demo_rows - 1)

    for row in range(demo_rows):
        for col in range(demo_cols):
            face = PATCH_FACE_COLORS[(row + col) % len(PATCH_FACE_COLORS)]
            if row == sel_r and col == sel_c:
                face = "#E76F51"
            rect = Rectangle(
                (grid_x0 + col * cell_w, grid_y0 + row * cell_h),
                cell_w - 0.2,
                cell_h - 0.2,
                facecolor=face,
                edgecolor="#355070",
                linewidth=1.0,
                alpha=0.70,
            )
            ax.add_patch(rect)

    ax.text(
        grid_x0,
        grid_y0 - 4.5,
        f"patch尺寸 = {config.patch_width} × {config.patch_height}",
        fontsize=10,
        color="#334E68",
        ha="left",
    )

    ax.text(6, 57, "时间分块", fontsize=13, color="#1F2933", ha="left")
    time_x0, time_y0 = 8, 49
    for idx in range(6):
        color = "#A8DADC" if idx != 2 else "#E63946"
        rect = Rectangle(
            (time_x0 + idx * 7.8, time_y0),
            6.6,
            5.0,
            facecolor=color,
            edgecolor="#355070",
            linewidth=1.0,
            alpha=0.82,
        )
        ax.add_patch(rect)
        label = r"$\tau_{" + str(idx + 1) + "}$"
        ax.text(time_x0 + idx * 7.8 + 3.3, time_y0 + 2.5, label, ha="center", va="center", fontsize=11)

    ax.text(
        time_x0,
        time_y0 - 4.8,
        f"{time_block['n_blocks']} 个时间块，Δt = {config.time_block_hours} h",
        fontsize=10,
        color="#334E68",
        ha="left",
    )

    op_rect = Rectangle((47, 28), 16, 14, facecolor="#F1FAEE", edgecolor="#1D3557", linewidth=1.2)
    ax.add_patch(op_rect)
    ax.text(
        55,
        35,
        "空间 patch\n×\n时间块",
        fontsize=12,
        ha="center",
        va="center",
        color="#1D3557",
    )

    arrow_a = FancyArrowPatch((36, 24), (47, 31), arrowstyle="-|>", mutation_scale=15, linewidth=1.6, color="#1D3557")
    arrow_b = FancyArrowPatch((33, 49), (47, 39), arrowstyle="-|>", mutation_scale=15, linewidth=1.6, color="#1D3557")
    ax.add_patch(arrow_a)
    ax.add_patch(arrow_b)

    ax.text(69, 58, "时空张量表示", fontsize=13, color="#1F2933", ha="left")
    ax.text(69, 52.5, r"$H^{patch}$", fontsize=18, color="#111111", ha="left")
    ax.text(
        69,
        48.8,
        r"$\in\mathbb{R}^{N_t \times R_p \times P_s \times d_{model}}$",
        fontsize=13,
        color="#111111",
        ha="left",
    )
    draw_tensor_stack(ax, 71, 22, 16, 20, depth=1.5, layers=5, facecolor="#4EA8DE")

    ax.annotate("", xy=(90, 24), xytext=(90, 44), arrowprops=dict(arrowstyle="<->", linewidth=1.2, color="#1D3557"))
    ax.text(91.5, 34, r"$N_t$", fontsize=12, va="center", ha="left")

    ax.annotate("", xy=(71, 18.2), xytext=(87, 18.2), arrowprops=dict(arrowstyle="<->", linewidth=1.2, color="#1D3557"))
    ax.text(79, 16.1, r"$R_p \times P_s$", fontsize=12, va="top", ha="center")

    ax.annotate("", xy=(87, 44), xytext=(93, 48), arrowprops=dict(arrowstyle="<->", linewidth=1.2, color="#1D3557"))
    ax.text(94, 49, r"$d_{model}$", fontsize=12, va="bottom", ha="left")

    ax.text(
        69,
        7.0,
        "$N_t$ 表示时间块数，\n$R_p$ 表示patch行数，$P_s$ 表示patch列数",
        fontsize=10,
        color="#334E68",
        ha="left",
        va="top",
    )

    save_figure(fig, config.output_dir, "fig05_spatiotemporal_blocking", config.dpi)


def plot_tensor_structure(
    points: pd.DataFrame,
    times: pd.DatetimeIndex,
    time_block: Dict[str, object],
    config: PlotConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)

    ax.text(10, 52, "原始监测序列", fontsize=13, ha="left", color="#1F2933")
    for idx, alpha in enumerate([0.28, 0.42, 0.58]):
        rect = Rectangle(
            (10 + idx * 1.1, 16 + idx * 1.1),
            18,
            24,
            facecolor="#A8DADC",
            edgecolor="#355070",
            linewidth=1.0,
            alpha=alpha,
        )
        ax.add_patch(rect)
    ax.text(19.5, 12.5, r"$X\in\mathbb{R}^{L\times N\times C_{in}}$", fontsize=14, ha="center", color="#111111")
    ax.text(
        10,
        6.5,
        f"$L$ = {len(times)} 个小时步\n"
        f"$N$ = {points.shape[0]} 个空间监测点\n"
        r"$C_{in}=3$：deformation、speed、acceleration",
        fontsize=10,
        ha="left",
        va="bottom",
        color="#334E68",
    )

    arrow = FancyArrowPatch((33, 28), (48, 28), arrowstyle="-|>", mutation_scale=16, linewidth=1.7, color="#1D3557")
    ax.add_patch(arrow)

    mid_rect = Rectangle((49, 18), 18, 20, facecolor="#F1FAEE", edgecolor="#1D3557", linewidth=1.2)
    ax.add_patch(mid_rect)
    ax.text(
        58,
        28,
        "patch统计特征\n+ 线性/神经投影",
        fontsize=12,
        ha="center",
        va="center",
        color="#1D3557",
    )

    arrow2 = FancyArrowPatch((67, 28), (76, 28), arrowstyle="-|>", mutation_scale=16, linewidth=1.7, color="#1D3557")
    ax.add_patch(arrow2)

    draw_tensor_stack(ax, 77, 18, 12, 16, depth=1.4, layers=5, facecolor="#4EA8DE")
    ax.text(
        73,
        12.5,
        r"$H^{patch}\in\mathbb{R}^{N_t\times R_p\times P_s\times d_{model}}$",
        fontsize=14,
        ha="left",
        color="#111111",
    )
    ax.text(
        73,
        6.5,
        f"$N_t$ = {time_block['n_blocks']} 个时间块\n"
        r"$R_p\times P_s$：空间patch网格"
        "\n"
        r"$d_{model}$：嵌入后的patch特征维度",
        fontsize=10,
        ha="left",
        va="bottom",
        color="#334E68",
    )

    save_figure(fig, config.output_dir, "fig06_input_tensor_structure", config.dpi)


def main() -> None:
    config = parse_args()
    setup_plot_style()

    points = load_unique_points(config.csv_path)
    background_ids = detect_background_area_ids(points)
    zone_rules = ZoneRuleConfig(background_area_ids=background_ids)
    points = assign_engineering_zone(points, zone_rules)

    points_with_patch, patch_summary, selected_patch = prepare_patch_summary(points, config)
    times = load_unique_times(config.csv_path, config.chunksize)
    time_block = select_time_block(times, config.time_block_hours)
    point_summary, patch_stats = load_patch_block_statistics(config, selected_patch, time_block)

    plot_raw_points(points_with_patch, config)
    plot_engineering_zones(points_with_patch, config)
    plot_patch_partition(points_with_patch, patch_summary, selected_patch, config)
    plot_representative_patch(points_with_patch, point_summary, patch_stats, selected_patch, time_block, config)
    plot_spatiotemporal_blocking(patch_summary, selected_patch, time_block, config)
    plot_tensor_structure(points_with_patch, times, time_block, config)

    print(f"图片已保存至: {config.output_dir.resolve()}")
    print(
        "代表性 patch 范围: "
        f"x=[{selected_patch['x_min']}, {selected_patch['x_max']}], "
        f"y=[{selected_patch['y_min']}, {selected_patch['y_max']}]"
    )
    print(
        "选定时间块: "
        f"{pd.Timestamp(time_block['start']).strftime('%Y-%m-%d %H:%M')} -> "
        f"{pd.Timestamp(time_block['end']).strftime('%Y-%m-%d %H:%M')}"
    )
    print(f"自动识别的背景区 area_id: {list(background_ids)}")


if __name__ == "__main__":
    main()
