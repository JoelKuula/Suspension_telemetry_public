from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

from .analysis_service import (
    VELOCITY_AXIS_LINEAR,
    VELOCITY_AXIS_SIGNED_LOG,
    channel_series,
    inverse_transform_velocity_axis_values,
    normalize_velocity_axis_mode,
    transform_velocity_axis_values,
    velocity_axis_label,
    velocity_axis_range_for_data,
    velocity_axis_ticks,
)


def finite_range(values: np.ndarray, low_percentile: float, high_percentile: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite values available")
    low = float(np.percentile(finite, low_percentile))
    high = float(np.percentile(finite, high_percentile))
    if low == high:
        pad = 1.0 if low == 0.0 else abs(low) * 0.05
        low -= pad
        high += pad
    return low, high


def build_color_norm(histogram: np.ndarray, color_scale: str):
    import matplotlib.colors as colors

    positive = histogram[histogram > 0]
    vmax = float(np.max(histogram)) if histogram.size and np.max(histogram) > 0 else 1.0
    if color_scale == "linear":
        return colors.Normalize(vmin=0.0, vmax=vmax)
    if color_scale == "sqrt":
        return colors.PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
    if positive.size == 0:
        return colors.Normalize(vmin=0.0, vmax=vmax)
    return colors.LogNorm(vmin=float(np.min(positive)), vmax=vmax)


def compute_occupancy_grid(
    derived_df,
    channel: str,
    travel_bins: int,
    velocity_bins: int,
    velocity_source: str = "filtered",
    series_mode: str = "native",
    color_scale: str = "sqrt",
    velocity_axis_mode: str = VELOCITY_AXIS_LINEAR,
    travel_percentile_low: float = 0.0,
    travel_percentile_high: float = 100.0,
    velocity_percentile: float = 99.5,
    travel_min: float | None = None,
    travel_max: float | None = None,
    velocity_min: float | None = None,
    velocity_max: float | None = None,
) -> dict[str, Any]:
    series = channel_series(derived_df, channel, velocity_source=velocity_source, series_mode=series_mode)
    travel = series["travel"]
    velocity = series["velocity"]
    weights_s = series["dt_s"]
    axis_mode = normalize_velocity_axis_mode(velocity_axis_mode)

    auto_travel_min, auto_travel_max = finite_range(travel, travel_percentile_low, travel_percentile_high)
    finite_velocity = velocity[np.isfinite(velocity)]
    if finite_velocity.size and axis_mode == VELOCITY_AXIS_SIGNED_LOG:
        auto_velocity_abs = float(np.max(np.abs(finite_velocity)))
    else:
        auto_velocity_abs = float(np.percentile(np.abs(finite_velocity), velocity_percentile)) if finite_velocity.size else 1.0
    if auto_velocity_abs <= 0:
        auto_velocity_abs = 1.0

    if series_mode == "stroke_percent" and travel_min is None and travel_max is None:
        travel_min = 0.0
        travel_max = 100.0
    else:
        travel_min = auto_travel_min if travel_min is None else float(travel_min)
        travel_max = auto_travel_max if travel_max is None else float(travel_max)
    velocity_min = -auto_velocity_abs if velocity_min is None else float(velocity_min)
    velocity_max = auto_velocity_abs if velocity_max is None else float(velocity_max)
    display_velocity = transform_velocity_axis_values(velocity, axis_mode)
    display_velocity_min, display_velocity_max = velocity_axis_range_for_data(velocity_min, velocity_max, axis_mode)

    finite = (
        np.isfinite(travel)
        & np.isfinite(velocity)
        & np.isfinite(display_velocity)
        & np.isfinite(weights_s)
        & (weights_s > 0)
        & (travel >= travel_min)
        & (travel <= travel_max)
        & (velocity >= velocity_min)
        & (velocity <= velocity_max)
    )
    if not np.any(finite):
        raise ValueError("all rows were clipped away by the current axis limits")

    histogram, travel_edges, velocity_edges = np.histogram2d(
        travel[finite],
        display_velocity[finite],
        bins=[travel_bins, velocity_bins],
        range=[(travel_min, travel_max), (display_velocity_min, display_velocity_max)],
        weights=weights_s[finite],
    )
    return {
        "channel": channel,
        "histogram": histogram,
        "travel_edges": travel_edges,
        "velocity_edges": velocity_edges,
        "travel_label": series["travel_label"],
        "travel_units": series["travel_units"],
        "velocity_label": velocity_axis_label(series["velocity_label"], axis_mode),
        "velocity_units": series["velocity_units"],
        "velocity_source": velocity_source,
        "velocity_axis_mode": axis_mode,
        "velocity_axis_ticks": velocity_axis_ticks(velocity_min, velocity_max, axis_mode),
        "color_scale": color_scale,
        "total_occupancy_s": float(np.sum(weights_s[finite])),
        "sample_count": int(np.sum(finite)),
        "travel_range": (travel_min, travel_max),
        "velocity_range": (display_velocity_min, display_velocity_max),
        "velocity_data_range": (velocity_min, velocity_max),
    }


def save_occupancy_grid_csv(path: Path, occupancy: dict[str, Any]) -> None:
    histogram = occupancy["histogram"]
    travel_edges = occupancy["travel_edges"]
    velocity_edges = occupancy["velocity_edges"]
    travel_centers = 0.5 * (travel_edges[:-1] + travel_edges[1:])
    velocity_axis_centers = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])
    velocity_centers = inverse_transform_velocity_axis_values(
        velocity_axis_centers,
        occupancy.get("velocity_axis_mode", VELOCITY_AXIS_LINEAR),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["velocity_center", "travel_center", "occupancy_s"])
        for travel_index, travel_center in enumerate(travel_centers):
            for velocity_index, velocity_center in enumerate(velocity_centers):
                writer.writerow(
                    [
                        f"{velocity_center:.9g}",
                        f"{travel_center:.9g}",
                        f"{histogram[travel_index, velocity_index]:.9g}",
                    ]
                )


def save_occupancy_heatmap_png(path: Path, occupancy: dict[str, Any], dpi: int = 180) -> None:
    matplotlib_dir = path.resolve().parents[2] / ".codex_tmp" / "matplotlib" if len(path.resolve().parents) >= 3 else Path(".codex_tmp/matplotlib")
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    histogram = occupancy["histogram"]
    travel_edges = occupancy["travel_edges"]
    velocity_edges = occupancy["velocity_edges"]
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 7))
    mesh = axis.pcolormesh(
        velocity_edges,
        travel_edges,
        histogram,
        shading="auto",
        cmap="hot",
        norm=build_color_norm(histogram, occupancy["color_scale"]),
    )
    axis.set_xlabel(f"{occupancy.get('velocity_label', 'Velocity')} [{occupancy['velocity_units']}]")
    axis.set_ylabel(f"{occupancy.get('travel_label', 'Travel')} [{occupancy['travel_units']}]")
    axis.set_title(f"{occupancy['channel'].capitalize()} velocity-travel occupancy")
    axis_ticks = occupancy.get("velocity_axis_ticks") or []
    if axis_ticks:
        axis.set_xticks([position for position, _label in axis_ticks])
        axis.set_xticklabels([label for _position, label in axis_ticks])
    figure.colorbar(mesh, ax=axis, label="Occupancy [s]")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
