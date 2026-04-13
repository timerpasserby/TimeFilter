#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PlotConfig:
    monitor_csv: Path
    blast_csv: Optional[Path]
    boundary_csv: Optional[Path]
    output_dir: Path
    default_sigma: float = 38.0
    grid_res_x: int = 280
    grid_res_y: int = 220
    dpi: int = 300


CJK_FONT_CANDIDATES = [
    "Songti SC",
    "PingFang HK",
    "Hiragino Sans GB",
    "STHeiti",
    "Arial Unicode MS",
]


def parse_args() -> PlotConfig:
    parser = argparse.ArgumentParser(description="Generate thesis-ready blast spatial decay figures.")
    parser.add_argument(
        "--monitor-csv",
        type=Path,
        default=Path("/Users/dc/Z研究生/eedsProject/device_10001_sorted.csv"),
        help="Monitoring-point CSV with grid_x and grid_y.",
    )
    parser.add_argument(
        "--blast-csv",
        type=Path,
        default=None,
        help="Optional blast-event CSV with columns: blast_id, x_b, y_b, Q, sigma(optional).",
    )
    parser.add_argument(
        "--boundary-csv",
        type=Path,
        default=None,
        help="Optional engineering boundary CSV with columns x, y.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/blast_spatial_decay_figures"),
        help="Output directory for PNG/PDF figures.",
    )
    parser.add_argument("--sigma", type=float, default=38.0, help="Default Gaussian sigma.")
    parser.add_argument("--grid-res-x", type=int, default=280, help="Grid resolution along x.")
    parser.add_argument("--grid-res-y", type=int, default=220, help="Grid resolution along y.")
    parser.add_argument("--dpi", type=int, default=300, help="Output dpi.")
    args = parser.parse_args()
    return PlotConfig(
        monitor_csv=args.monitor_csv,
        blast_csv=args.blast_csv,
        boundary_csv=args.boundary_csv,
        output_dir=args.output_dir,
        default_sigma=args.sigma,
        grid_res_x=args.grid_res_x,
        grid_res_y=args.grid_res_y,
        dpi=args.dpi,
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
            "axes.titlesize": 16,
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


def load_monitoring_points(monitor_csv: Path) -> pd.DataFrame:
    points = pd.read_csv(monitor_csv, usecols=["grid_x", "grid_y"])
    points = points.drop_duplicates(subset=["grid_x", "grid_y"]).copy()
    points = points.rename(columns={"grid_x": "x", "grid_y": "y"})
    points[["x", "y"]] = points[["x", "y"]].astype("float32")
    points = points.sort_values(["x", "y"]).reset_index(drop=True)
    return points


def load_boundary(boundary_csv: Optional[Path]) -> Optional[pd.DataFrame]:
    if boundary_csv is None:
        return None
    boundary = pd.read_csv(boundary_csv)
    expected = {"x", "y"}
    if not expected.issubset(boundary.columns):
        raise ValueError("Boundary CSV must contain columns: x, y")
    return boundary.loc[:, ["x", "y"]].copy()


def create_example_blasts(points: pd.DataFrame, sigma: float) -> pd.DataFrame:
    targets = [
        ("B1", 0.25, 0.22, 120.0),
        ("B2", 0.52, 0.50, 180.0),
        
    ]
    coords = points[["x", "y"]].to_numpy()
    tree = cKDTree(coords)
    used_indices = set()
    records = []

    x_min, x_max = float(points["x"].min()), float(points["x"].max())
    y_min, y_max = float(points["y"].min()), float(points["y"].max())

    for blast_id, x_ratio, y_ratio, charge in targets:
        target = np.array([x_min + (x_max - x_min) * x_ratio, y_min + (y_max - y_min) * y_ratio])
        _, neighbor_ids = tree.query(target, k=25)
        neighbor_ids = np.atleast_1d(neighbor_ids)
        chosen = None
        for idx in neighbor_ids:
            idx = int(idx)
            if idx not in used_indices:
                used_indices.add(idx)
                chosen = idx
                break
        if chosen is None:
            chosen = int(neighbor_ids[0])
        records.append(
            {
                "blast_id": blast_id,
                "x_b": float(coords[chosen, 0]),
                "y_b": float(coords[chosen, 1]),
                "Q": float(charge),
                "sigma": float(sigma),
                "data_source": "爆破事件",
            }
        )
    return pd.DataFrame.from_records(records)


def load_or_create_blasts(blast_csv: Optional[Path], points: pd.DataFrame, sigma: float) -> pd.DataFrame:
    if blast_csv is not None and blast_csv.exists():
        blasts = pd.read_csv(blast_csv)
        required = {"blast_id", "x_b", "y_b", "Q"}
        if not required.issubset(blasts.columns):
            raise ValueError("Blast CSV must contain columns: blast_id, x_b, y_b, Q")
        if "sigma" not in blasts.columns:
            blasts["sigma"] = sigma
        blasts["data_source"] = "真实爆破事件"
    else:
        blasts = create_example_blasts(points, sigma)
    return blasts.loc[:, ["blast_id", "x_b", "y_b", "Q", "sigma", "data_source"]].reset_index(drop=True).copy()


def gaussian_decay(distance: np.ndarray, charge: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return charge * np.exp(-(distance ** 2) / (2.0 * sigma ** 2))


def compute_point_impacts(points: pd.DataFrame, blasts: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    point_xy = points[["x", "y"]].to_numpy()
    blast_xy = blasts[["x_b", "y_b"]].to_numpy()
    charge = blasts["Q"].to_numpy()[None, :]
    sigma = blasts["sigma"].to_numpy()[None, :]

    distance = np.sqrt(((point_xy[:, None, :] - blast_xy[None, :, :]) ** 2).sum(axis=2))
    contributions = gaussian_decay(distance, charge, sigma)
    impact_df = points.copy()
    impact_df["impact_total"] = contributions.sum(axis=1)
    impact_df["distance_to_nearest_blast"] = distance.min(axis=1)

    for idx, blast_id in enumerate(blasts["blast_id"].tolist()):
        impact_df[f"impact_{blast_id}"] = contributions[:, idx]
    return impact_df, contributions


def estimate_mask_threshold(points: pd.DataFrame) -> float:
    coords = points[["x", "y"]].to_numpy()
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nearest = distances[:, 1]
    threshold = max(6.0, float(np.quantile(nearest, 0.95) * 4.5))
    return threshold


def build_regular_grid(points: pd.DataFrame, config: PlotConfig) -> Tuple[np.ndarray, np.ndarray]:
    x_min, x_max = float(points["x"].min()), float(points["x"].max())
    y_min, y_max = float(points["y"].min()), float(points["y"].max())
    grid_x = np.linspace(x_min, x_max, config.grid_res_x)
    grid_y = np.linspace(y_min, y_max, config.grid_res_y)
    return np.meshgrid(grid_x, grid_y)


def compute_grid_field(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    blasts: pd.DataFrame,
    points: pd.DataFrame,
    threshold: float,
) -> np.ma.MaskedArray:
    field = np.zeros_like(grid_x, dtype=float)
    for row in blasts.itertuples(index=False):
        dist_sq = (grid_x - row.x_b) ** 2 + (grid_y - row.y_b) ** 2
        field += row.Q * np.exp(-dist_sq / (2.0 * row.sigma ** 2))

    tree = cKDTree(points[["x", "y"]].to_numpy())
    grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    nearest_dist, _ = tree.query(grid_coords, k=1)
    mask = nearest_dist.reshape(grid_x.shape) > threshold
    return np.ma.array(field, mask=mask)


def draw_boundary(ax: plt.Axes, boundary: Optional[pd.DataFrame]) -> None:
    if boundary is None:
        return
    ax.plot(boundary["x"], boundary["y"], color="#1D3557", linewidth=1.4, linestyle="-", alpha=0.95, zorder=4)


def draw_blasts(ax: plt.Axes, blasts: pd.DataFrame, annotate: bool = True, highlight_main: Optional[str] = None) -> None:
    for row in blasts.itertuples(index=False):
        is_main = highlight_main is not None and row.blast_id == highlight_main
        ax.scatter(
            row.x_b,
            row.y_b,
            s=220 if is_main else 180,
            marker="*",
            color="#C1121F" if is_main or highlight_main is None else "#E76F51",
            edgecolors="white",
            linewidths=0.9,
            zorder=6,
        )
        if annotate:
            ax.text(
                row.x_b + 4.0,
                row.y_b + 4.0,
                str(row.blast_id),
                fontsize=10,
                color="#7F0000",
                weight="bold",
                ha="left",
                va="bottom",
                zorder=7,
            )


def style_spatial_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, pad=10)
    ax.set_xlabel("坡面二维投影 X 坐标")
    ax.set_ylabel("坡面二维投影 Y 坐标")
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#FCFCFC")
    ax.grid(True, alpha=0.45)


def plot_spatial_field(
    impact_df: pd.DataFrame,
    blasts: pd.DataFrame,
    field: np.ma.MaskedArray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    boundary: Optional[pd.DataFrame],
    config: PlotConfig,
) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    vmax = float(max(np.nanmax(field.filled(np.nan)), impact_df["impact_total"].max()))
    levels = np.linspace(0.0, vmax, 16)
    norm = Normalize(vmin=0.0, vmax=vmax)

    contour = ax.contourf(
        grid_x,
        grid_y,
        field,
        levels=levels,
        cmap="YlOrRd",
        norm=norm,
        alpha=0.88,
        antialiased=True,
    )
    ax.contour(
        grid_x,
        grid_y,
        field,
        levels=levels[::3],
        colors="#8C510A",
        linewidths=0.55,
        alpha=0.50,
    )
    ax.scatter(
        impact_df["x"],
        impact_df["y"],
        c=impact_df["impact_total"],
        cmap="YlOrRd",
        norm=norm,
        s=12,
        edgecolors="white",
        linewidths=0.12,
        alpha=0.97,
        zorder=5,
    )
    draw_boundary(ax, boundary)
    draw_blasts(ax, blasts, annotate=True)

    style_spatial_axis(ax, "")
    handles = [
        # Line2D([0], [0], marker="o", color="none", markerfacecolor="#63FDAD", markeredgecolor="white", markersize=7, label="监测点"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#C1121F", markeredgecolor="white", markersize=14, label="爆破点"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.96)

    note = (
        f"$\\sigma$ = {blasts['sigma'].iloc[0]:.1f}\n"
        f"爆破事件数 = {len(blasts)}\n"
        f"{blasts['data_source'].iloc[0]}"
    )
    ax.text(
        0.985,
        0.02,
        note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#BBBBBB", alpha=0.92),
    )

    cbar = fig.colorbar(contour, ax=ax, pad=0.02, shrink=0.92)
    cbar.set_label("爆破影响强度 $s_i$")
    save_figure(fig, config.output_dir, "fig01_blast_spatial_decay_field", config.dpi)


def plot_decay_curves(blasts: pd.DataFrame, points: pd.DataFrame, config: PlotConfig) -> None:
    x_span = float(points["x"].max() - points["x"].min())
    y_span = float(points["y"].max() - points["y"].min())
    max_distance = float(np.hypot(x_span, y_span) * 0.75)
    distance = np.linspace(0.0, max_distance, 500)

    q_values = sorted(blasts["Q"].unique().tolist())
    sigma_ref = float(blasts["sigma"].median())
    sigma_values = [max(18.0, sigma_ref * 0.65), sigma_ref, sigma_ref * 1.45]
    q_ref = float(np.median(q_values))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=False)
    ax1, ax2 = axes

    colors_left = ["#457B9D", "#E76F51", "#2A9D8F", "#7B2CBF"]
    for idx, q in enumerate(q_values):
        curve = gaussian_decay(distance, np.full_like(distance, q), np.full_like(distance, sigma_ref))
        ax1.plot(distance, curve, linewidth=2.0, color=colors_left[idx % len(colors_left)], label=f"$Q={q:.0f}$")
    ax1.axvline(sigma_ref, color="#7A7A7A", linestyle="--", linewidth=1.1, alpha=0.8)
    ax1.text(sigma_ref + 2.0, ax1.get_ylim()[1] * 0.78 if ax1.get_ylim()[1] > 0 else 1, r"$d=\sigma$", fontsize=9.5, color="#555555")
    ax1.set_title("不同炸药量下的高斯衰减曲线")
    ax1.set_xlabel("爆破点至监测点距离 / m")
    ax1.set_ylabel("爆破影响强度")
    ax1.grid(True, alpha=0.45)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.96)

    colors_right = ["#D62828", "#1D3557", "#2A9D8F"]
    for idx, sigma_value in enumerate(sigma_values):
        curve = gaussian_decay(distance, np.full_like(distance, q_ref), np.full_like(distance, sigma_value))
        ax2.plot(distance, curve, linewidth=2.0, color=colors_right[idx], label=fr"$\sigma={sigma_value:.0f}$")
    ax2.set_title("不同影响范围参数下的衰减对比")
    ax2.set_xlabel("爆破点至监测点距离 / m")
    ax2.set_ylabel("爆破影响强度")
    ax2.grid(True, alpha=0.45)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.96)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, config.output_dir, "fig02_blast_decay_curves", config.dpi)


def plot_superposition_comparison(
    impact_df: pd.DataFrame,
    contributions: np.ndarray,
    blasts: pd.DataFrame,
    points: pd.DataFrame,
    boundary: Optional[pd.DataFrame],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    total_field: np.ma.MaskedArray,
    threshold: float,
    config: PlotConfig,
) -> None:
    main_idx = int(blasts["Q"].astype(float).idxmax())
    main_blast = blasts.loc[main_idx]
    single_blast_field = compute_grid_field(grid_x, grid_y, blasts.loc[[main_idx]], points, threshold)
    single_point_impact = contributions[:, main_idx]

    vmax = float(
        max(
            np.nanmax(total_field.filled(np.nan)),
            np.nanmax(single_blast_field.filled(np.nan)),
            impact_df["impact_total"].max(),
        )
    )
    levels = np.linspace(0.0, vmax, 16)
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), sharex=True, sharey=True)
    panels = [
        (axes[0], single_blast_field, single_point_impact, "单次主爆破事件作用场", str(main_blast["blast_id"])),
        (axes[1], total_field, impact_df["impact_total"].to_numpy(), "多爆破事件叠加作用场", None),
    ]

    for ax, field, point_impact, title, highlight in panels:
        ax.contourf(grid_x, grid_y, field, levels=levels, cmap="YlOrRd", norm=norm, alpha=0.88, antialiased=True)
        ax.contour(grid_x, grid_y, field, levels=levels[::3], colors="#8C510A", linewidths=0.50, alpha=0.45)
        ax.scatter(
            points["x"],
            points["y"],
            c=point_impact,
            cmap="YlOrRd",
            norm=norm,
            s=11,
            edgecolors="white",
            linewidths=0.10,
            alpha=0.96,
            zorder=5,
        )
        draw_boundary(ax, boundary)
        draw_blasts(ax, blasts, annotate=False, highlight_main=highlight)
        style_spatial_axis(ax, title)

    handles = [
        # Line2D([0], [0], marker="o", color="none", markerfacecolor="#63FDAD", markeredgecolor="white", markersize=7, label="监测点"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#C1121F", markeredgecolor="white", markersize=14, label="爆破点"),
    ]
    axes[0].legend(handles=handles, loc="upper left", frameon=True, framealpha=0.96)
    sm = ScalarMappable(norm=norm, cmap="YlOrRd")
    sm.set_array([])
    # 将颜色条定位到最右端
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.02, shrink=0.92, aspect=20, anchor=(1.0, 0.5))
    cbar.set_label("爆破影响强度")
    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.10, top=0.86, wspace=0.09)
    save_figure(fig, config.output_dir, "fig03_blast_superposition_comparison", config.dpi)


def main() -> None:
    config = parse_args()
    setup_plot_style()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    points = load_monitoring_points(config.monitor_csv)
    blasts = load_or_create_blasts(config.blast_csv, points, config.default_sigma)
    boundary = load_boundary(config.boundary_csv)

    impact_df, contributions = compute_point_impacts(points, blasts)
    threshold = estimate_mask_threshold(points)
    grid_x, grid_y = build_regular_grid(points, config)
    total_field = compute_grid_field(grid_x, grid_y, blasts, points, threshold)

    plot_spatial_field(impact_df, blasts, total_field, grid_x, grid_y, boundary, config)
    plot_decay_curves(blasts, points, config)
    plot_superposition_comparison(
        impact_df=impact_df,
        contributions=contributions,
        blasts=blasts,
        points=points,
        boundary=boundary,
        grid_x=grid_x,
        grid_y=grid_y,
        total_field=total_field,
        threshold=threshold,
        config=config,
    )

    impact_df.to_csv(config.output_dir / "monitoring_point_blast_impact.csv", index=False)
    blasts.to_csv(config.output_dir / "blast_events_used.csv", index=False)

    print(f"图片已保存至: {config.output_dir.resolve()}")
    print(f"监测点数量: {len(points)}")
    print(f"爆破事件数量: {len(blasts)}")
    print(f"使用的爆破数据来源: {blasts['data_source'].iloc[0]}")
    print(f"默认 sigma: {config.default_sigma:.1f}")


if __name__ == "__main__":
    main()
