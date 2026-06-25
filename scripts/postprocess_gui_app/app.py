from __future__ import annotations

import copy
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MPLCONFIGDIR = REPO_ROOT / ".codex_tmp" / "matplotlib"
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))

import numpy as np
import polars as pl
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .backend.analysis_service import (
    build_derived_analog,
    channel_series,
    compute_balance_analysis,
    compute_channel_metrics,
    compute_travel_histogram,
    compute_velocity_histogram,
    normalize_velocity_axis_mode,
)
from .backend.braking_analysis import build_fork_braking_analysis, normalize_braking_config
from .backend.context_analysis import (
    ACCEL_AXIS_OPTIONS,
    DEFAULT_CONTEXT_CONFIG,
    GYRO_AXIS_OPTIONS,
    RESPONSE_OPTIONS,
    SOURCE_OPTIONS,
    WHEEL_SPEED_SOURCE,
    build_breakdown_analysis,
    build_context_dataset,
    compute_breakdown_heatmaps,
    compute_context_heatmap,
    normalize_breakdown_config,
    summarize_breakdown_selection,
    summarize_context_bins,
)
from .backend.occupancy_service import (
    compute_occupancy_grid,
    save_occupancy_grid_csv,
    save_occupancy_heatmap_png,
)
from .backend.session_metadata_service import (
    SessionDatabaseEntry,
    auto_assign_set_labels,
    save_session_metadata,
    scan_session_database,
)
from .backend.session_config import DEFAULT_BRAKING_CONFIG, DEFAULT_BREAKDOWN_CONFIG, analysis_dir, save_session_config
from .backend.session_service import SessionBundle, open_bin_session, open_export_session, rebuild_session_analysis


pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")

COMPARE_MIN_SESSIONS = 2
COMPARE_MAX_SESSIONS = 10
COMPARE_SESSION_COLORS = (
    "#4e79a7",
    "#e15759",
    "#59a14f",
    "#f28e2b",
    "#b07aa1",
    "#76b7b2",
    "#edc948",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
)


def _reset_plot_log_mode(plot_item: pg.PlotItem) -> None:
    try:
        plot_item.setLogMode(False, False)
    except TypeError:
        plot_item.setLogMode(x=False, y=False)


def _apply_velocity_axis_ticks(plot_item: pg.PlotItem, axis_ticks: list[tuple[float, str]] | None) -> None:
    _reset_plot_log_mode(plot_item)
    axis = plot_item.getAxis("bottom")
    if axis_ticks:
        axis.setTicks([axis_ticks])
    else:
        axis.setTicks(None)


CALIBRATION_CONFIG_KEYS = (
    "invert",
    "mm_per_count",
    "full_scale_mm",
    "sensor_full_scale_mm",
    "manual_reference_count",
    "velocity_filter_window",
)


def format_metric_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return ""
        return f"{float(value):.2f}"
    return str(value)


def format_duration_clock(seconds: Any) -> str:
    if seconds is None:
        return ""
    try:
        total_seconds = float(seconds)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(total_seconds):
        return ""
    rounded_seconds = int(round(max(total_seconds, 0.0)))
    minutes, seconds_part = divmod(rounded_seconds, 60)
    hours, minutes_part = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}:{minutes_part:02d}:{seconds_part:02d}"
    return f"{minutes}:{seconds_part:02d}"


def format_metrics_table_value(metric: dict[str, Any]) -> str:
    value = metric.get("value")
    if metric.get("metric") == "Session duration" and metric.get("units") == "s":
        clock_value = format_duration_clock(value)
        seconds_value = format_metric_value(value)
        if clock_value and seconds_value:
            return f"{seconds_value} ({clock_value})"
        return seconds_value or clock_value
    return format_metric_value(value)


def parse_optional_float(text: str, field_name: str = "value") -> float | None:
    stripped = text.strip()
    if not stripped:
        return None
    if "/" in stripped:
        parts = stripped.split("/")
        if len(parts) != 2:
            raise ValueError(f"{field_name} must be a number or a simple ratio like 315/4035")
        numerator = float(parts[0].strip())
        denominator = float(parts[1].strip())
        if denominator == 0.0:
            raise ValueError(f"{field_name} denominator must not be zero")
        return numerator / denominator
    try:
        return float(stripped)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number or a simple ratio like 315/4035") from exc


def parse_float_list(text: str, field_name: str) -> list[float]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"{field_name} must contain at least one comma-separated number")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain comma-separated numbers") from exc


def open_path_in_file_manager(path: Path) -> bool:
    if not path.exists():
        return False
    resolved = path.resolve()
    startfile = getattr(os, "startfile", None)
    if startfile is not None:
        try:
            startfile(str(resolved))
            return True
        except OSError:
            pass
    return QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(resolved)))


def format_rtc_start(summary: dict[str, Any]) -> str:
    header = summary.get("header", {})
    start_epoch = header.get("start_epoch")
    rtc_valid = bool(header.get("rtc_valid"))
    if not rtc_valid or not start_epoch:
        return "no"
    try:
        timestamp = datetime.fromtimestamp(int(start_epoch), tz=timezone.utc)
    except (OverflowError, OSError, TypeError, ValueError):
        return str(start_epoch)
    return f"{timestamp.isoformat()} ({int(start_epoch)})"


def build_summary_text(bundle: SessionBundle) -> str:
    summary = bundle.summary
    metadata = bundle.session_metadata or {}
    lines = [
        f"Session: {metadata.get('session_id') or bundle.export_dir.name}",
        f"Date: {metadata.get('date') or '-'}",
        f"Track: {metadata.get('track') or '-'}",
        f"Set: {metadata.get('set_label') or bundle.export_dir.name}",
        f"Comment: {metadata.get('comment') or '-'}",
        "",
        f"Source: {bundle.source_path or summary.get('source_path', '')}",
        f"Export: {bundle.export_dir}",
        f"RTC start: {format_rtc_start(summary)}",
        f"Duration: {format_duration_clock(summary.get('timing', {}).get('duration_us', 0) / 1_000_000.0 if summary.get('timing', {}).get('duration_us') is not None else None)}",
        f"Analog rows: {summary.get('counts', {}).get('analog_rows', 0)}",
        f"Wheel rows: {summary.get('counts', {}).get('wheel_rows', 0)}",
        f"Status rows: {summary.get('counts', {}).get('stat_rows', 0)}",
        "",
        "Record counts:",
    ]
    for tag, count in sorted(summary.get("record_counts", {}).items()):
        lines.append(f"  {tag}: {count}")
    lines.extend(
        [
            "",
            "Derived config:",
            json.dumps(bundle.session_config, indent=2),
        ]
    )
    return "\n".join(lines)


def apply_calibration_template(target_config: dict[str, Any], template_config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(target_config)
    for channel in ("front", "rear"):
        target_channel = copy.deepcopy(config.get(channel, {}))
        template_channel = template_config.get(channel, {})
        for key in CALIBRATION_CONFIG_KEYS:
            if key in template_channel:
                target_channel[key] = copy.deepcopy(template_channel[key])
        config[channel] = target_channel
    return config


def frame_series(frame: pl.DataFrame, column: str, fill_value: float = np.nan) -> np.ndarray:
    if column not in frame.columns:
        return np.asarray([], dtype=np.float64)
    values = frame.get_column(column).fill_null(fill_value).to_numpy()
    return np.asarray(values, dtype=np.float64)


def finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0 or max_points <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if x.size <= max_points:
        return x, y
    indices = np.linspace(0, x.size - 1, num=max_points, dtype=np.int64)
    return x[indices], y[indices]


def downsample_xy_preserve_extrema(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0 or max_points <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if x.size <= max_points:
        return x, y

    bucket_count = max(1, max_points // 2)
    edges = np.linspace(0, x.size, num=bucket_count + 1, dtype=np.int64)
    selected_indices: set[int] = {0, x.size - 1}
    for bucket_index in range(bucket_count):
        start = int(edges[bucket_index])
        end = int(edges[bucket_index + 1])
        if end <= start:
            continue
        segment = y[start:end]
        selected_indices.add(start + int(np.argmin(segment)))
        selected_indices.add(start + int(np.argmax(segment)))

    ordered = np.asarray(sorted(selected_indices), dtype=np.int64)
    if ordered.size > max_points:
        sample_positions = np.linspace(0, ordered.size - 1, num=max_points, dtype=np.int64)
        ordered = ordered[sample_positions]
    return x[ordered], y[ordered]


def downsample_shared_x_preserve_extrema(
    x: np.ndarray,
    ys: list[np.ndarray],
    max_points: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if x.size == 0 or max_points <= 0 or not ys:
        return np.asarray([], dtype=np.float64), [np.asarray([], dtype=np.float64) for _ in ys]
    if any(series.size != x.size for series in ys):
        raise ValueError("all y series must match the x shape")
    if x.size <= max_points:
        return x, ys

    per_bucket_series = max(1, 2 * len(ys))
    bucket_count = max(1, max_points // per_bucket_series)
    edges = np.linspace(0, x.size, num=bucket_count + 1, dtype=np.int64)
    selected_indices: set[int] = {0, x.size - 1}
    for bucket_index in range(bucket_count):
        start = int(edges[bucket_index])
        end = int(edges[bucket_index + 1])
        if end <= start:
            continue
        for series in ys:
            segment = series[start:end]
            selected_indices.add(start + int(np.argmin(segment)))
            selected_indices.add(start + int(np.argmax(segment)))

    ordered = np.asarray(sorted(selected_indices), dtype=np.int64)
    if ordered.size > max_points:
        sample_positions = np.linspace(0, ordered.size - 1, num=max_points, dtype=np.int64)
        ordered = ordered[sample_positions]
    return x[ordered], [series[ordered] for series in ys]


def matplotlib_colormap(name: str) -> pg.ColorMap:
    return pg.colormap.getFromMatplotlib(name)


def combo_current_data(combo: QtWidgets.QComboBox) -> str:
    data = combo.currentData()
    if data is None:
        return combo.currentText()
    return str(data)


def set_combo_to_data(combo: QtWidgets.QComboBox, value: str) -> None:
    index = combo.findData(value)
    if index < 0:
        index = combo.findText(value)
    if index >= 0:
        combo.setCurrentIndex(index)


def bundle_session_label(bundle: SessionBundle) -> str:
    metadata = bundle.session_metadata or {}
    parts = [str(metadata.get("session_id") or bundle.export_dir.name)]
    if metadata.get("date"):
        parts.append(str(metadata["date"]))
    if metadata.get("track"):
        parts.append(str(metadata["track"]))
    if metadata.get("set_label") and str(metadata["set_label"]) != bundle.export_dir.name:
        parts.append(str(metadata["set_label"]))
    if metadata.get("comment"):
        parts.append(str(metadata["comment"]))
    return " | ".join(parts)


def bundle_session_id(bundle: SessionBundle) -> str:
    metadata = bundle.session_metadata or {}
    return str(metadata.get("session_id") or bundle.export_dir.name)


def bundle_compare_label(bundle: SessionBundle) -> str:
    metadata = bundle.session_metadata or {}
    session_id = bundle_session_id(bundle)
    set_label = str(metadata.get("set_label") or "").strip()
    parts = [
        str(metadata.get("date") or "").strip(),
        str(metadata.get("track") or "").strip(),
    ]
    if set_label and set_label != session_id:
        parts.append(set_label)
    else:
        parts.append(session_id)
    comment = str(metadata.get("comment") or "").strip()
    if comment:
        parts.append(comment)
    return " | ".join(part for part in parts if part)


def format_delta_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric):
        return ""
    formatted = format_metric_value(numeric)
    return formatted if numeric < 0 else f"+{formatted}"


RIDING_METRIC_ROWS: tuple[tuple[str, str | None, str | None, str | None], ...] = (
    ("Session duration", "Session duration", None, None),
    ("Minimum", "Minimum travel", None, "Peak rebound velocity"),
    ("Maximum", "Maximum travel", "Peak compression velocity", None),
    ("Mean", "Mean position", "Mean compression velocity", "Mean rebound velocity"),
    ("Median", "Median position", "Median compression velocity", "Median rebound velocity"),
    ("Mode", "Mode position", "Mode compression velocity", "Mode rebound velocity"),
    ("Geometric SD", "Position geometric SD", "Compression velocity geometric SD", "Rebound velocity geometric SD"),
    *(
        (
            f"P{band_start}-{band_start + 10}",
            f"Position P{band_start}-{band_start + 10}",
            f"Compression velocity P{band_start}-{band_start + 10}",
            f"Rebound velocity P{band_start}-{band_start + 10}",
        )
        for band_start in range(0, 100, 10)
    ),
)


def build_metrics_table_widget() -> QtWidgets.QTableWidget:
    table = QtWidgets.QTableWidget(0, 5)
    table.setHorizontalHeaderLabels(["Metric", "Front", "Front units", "Rear", "Rear units"])
    table.horizontalHeader().setStretchLastSection(True)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    return table


def bundle_session_start_time_s(bundle: SessionBundle) -> float | None:
    for row in bundle.status_rows:
        if str(row.get("name", "")) != "SESSION_START":
            continue
        try:
            return float(row.get("host_time_s", ""))
        except (TypeError, ValueError):
            continue

    first_host_timestamp_us = bundle.summary.get("timing", {}).get("first_host_timestamp_us")
    if first_host_timestamp_us is not None:
        try:
            return float(first_host_timestamp_us) / 1_000_000.0
        except (TypeError, ValueError):
            pass

    candidates: list[float] = []
    for frame in (bundle.analog_df, bundle.wheel_df, bundle.imu_frame_df):
        if "host_time_s" not in frame.columns:
            continue
        values = frame_series(frame, "host_time_s")
        finite = values[np.isfinite(values)]
        if finite.size:
            candidates.append(float(np.min(finite)))
    if candidates:
        return min(candidates)
    return None


class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(object)
    error = QtCore.Signal(str)


class BackgroundTask(QtCore.QRunnable):
    def __init__(self, fn: Callable[[], object]) -> None:
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self.fn()
        except Exception as exc:  # pragma: no cover - background thread path
            detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self.signals.error.emit(detail)
            return
        self.signals.finished.emit(result)


class SignalsPlotWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.showGrid(x=True, y=True, alpha=0.2)
        self.plot_item.setLabel("bottom", "Time", units="s")
        self.plot_item.showAxis("right")

        self.velocity_view = pg.ViewBox()
        self.plot_item.scene().addItem(self.velocity_view)
        self.plot_item.getAxis("right").linkToView(self.velocity_view)
        self.velocity_view.setXLink(self.plot_item)
        self.plot_item.vb.sigResized.connect(self._update_views)

        self.travel_curve = self.plot_item.plot(
            pen=pg.mkPen("#0b84a5", width=2),
            autoDownsample=True,
            clipToView=True,
        )
        self.velocity_curve = pg.PlotCurveItem(
            pen=pg.mkPen("#f6c85f", width=1.5),
            autoDownsample=True,
            clipToView=True,
        )
        self.velocity_view.addItem(self.velocity_curve)

    def _update_views(self) -> None:
        self.velocity_view.setGeometry(self.plot_item.vb.sceneBoundingRect())
        self.velocity_view.linkedViewChanged(self.plot_item.vb, self.velocity_view.XAxis)

    def set_series(self, channel: str, series: dict[str, Any]) -> None:
        self.travel_curve.setData(series["time_s"], series["travel"])
        self.velocity_curve.setData(series["time_s"], series["velocity"])
        self.plot_item.setTitle(f"{channel.capitalize()} signals")
        self.plot_item.setLabel("left", "Travel", units=series["travel_units"])
        self.plot_item.getAxis("right").setLabel("Velocity", units=series["velocity_units"])
        self._update_views()

    def clear(self) -> None:
        self.travel_curve.setData([], [])
        self.velocity_curve.setData([], [])


class _OccupancyPane(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.getPlotItem().showGrid(x=True, y=True, alpha=0.2)
        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)
        layout.addWidget(self.plot_widget, 1)

        self.lut_widget = pg.HistogramLUTWidget()
        self.lut_widget.setImageItem(self.image_item)
        self.lut_widget.gradient.setColorMap(matplotlib_colormap("hot"))
        self.lut_widget.setMinimumWidth(120)
        layout.addWidget(self.lut_widget)

    def set_occupancy(self, occupancy: dict[str, Any]) -> None:
        histogram = occupancy["histogram"]
        display = OccupancyPlotWidget.display_image(histogram, occupancy["color_scale"])
        self.image_item.setImage(display)

        velocity_min, velocity_max = occupancy["velocity_range"]
        travel_min, travel_max = occupancy["travel_range"]
        self.image_item.setRect(QtCore.QRectF(
            velocity_min,
            travel_min,
            velocity_max - velocity_min,
            travel_max - travel_min,
        ))
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setTitle(f"{occupancy['channel'].capitalize()} position/velocity heatmap")
        plot_item.setLabel("bottom", occupancy.get("velocity_label", "Velocity"), units=occupancy["velocity_units"])
        plot_item.setLabel("left", occupancy.get("travel_label", "Travel"), units=occupancy["travel_units"])
        _apply_velocity_axis_ticks(plot_item, occupancy.get("velocity_axis_ticks"))
        plot_item.enableAutoRange()
        if occupancy["travel_units"] == "%":
            plot_item.setYRange(0.0, 100.0, padding=0.02)

    def clear(self) -> None:
        self.image_item.setImage(np.zeros((1, 1), dtype=float))


class OccupancyPlotWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.front_pane = _OccupancyPane()
        self.rear_pane = _OccupancyPane()
        layout.addWidget(self.front_pane, 1)
        layout.addWidget(self.rear_pane, 1)

    @staticmethod
    def display_image(histogram: np.ndarray, color_scale: str) -> np.ndarray:
        if color_scale == "linear":
            return histogram
        if color_scale == "sqrt":
            return np.sqrt(histogram)
        positive = histogram[histogram > 0]
        if positive.size == 0:
            return histogram
        clipped = np.clip(histogram, float(np.min(positive)), None)
        return np.log10(clipped)

    def set_occupancies(self, front_occupancy: dict[str, Any], rear_occupancy: dict[str, Any]) -> None:
        self.front_pane.set_occupancy(front_occupancy)
        self.rear_pane.set_occupancy(rear_occupancy)

    def clear(self) -> None:
        self.front_pane.clear()
        self.rear_pane.clear()


class _ChannelHistogramsPane(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.travel_plot = pg.PlotWidget()
        self.travel_plot.getPlotItem().showGrid(x=True, y=True, alpha=0.2)
        self.velocity_plot = pg.PlotWidget()
        self.velocity_plot.getPlotItem().showGrid(x=True, y=True, alpha=0.2)
        self.velocity_legend = self.velocity_plot.addLegend()
        layout.addWidget(self.travel_plot, 1)
        layout.addWidget(self.velocity_plot, 1)

    @staticmethod
    def _add_legend_entry(plot_widget: pg.PlotWidget, label: str, color: str) -> None:
        plot_widget.plot([], [], pen=pg.mkPen(color, width=2), name=label)

    def set_histograms(self, channel: str, travel_hist: dict[str, Any], velocity_hist: dict[str, Any]) -> None:
        self.travel_plot.clear()
        self.velocity_plot.clear()
        self.velocity_legend.clear()

        travel_edges = travel_hist["edges"]
        travel_centers = 0.5 * (travel_edges[:-1] + travel_edges[1:])
        travel_widths = np.diff(travel_edges)
        self.travel_plot.addItem(
            pg.BarGraphItem(
                x=travel_centers,
                height=travel_hist["histogram"],
                width=travel_widths,
                brush=pg.mkBrush("#0b84a5"),
                pen=pg.mkPen("#0b84a5"),
            )
        )
        self.travel_plot.setTitle(f"{channel.capitalize()} position distribution")
        self.travel_plot.setLabel("bottom", travel_hist.get("label", "Travel"), units=travel_hist["units"])
        _apply_velocity_axis_ticks(self.travel_plot.getPlotItem(), None)
        self.travel_plot.setLabel(
            "left",
            travel_hist.get("occupancy_label", "Time in bin"),
            units=travel_hist.get("occupancy_units", "s"),
        )
        if travel_hist["units"] == "%":
            self.travel_plot.setXRange(0.0, 100.0, padding=0.02)

        velocity_edges = velocity_hist["edges"]
        velocity_centers = 0.5 * (velocity_edges[:-1] + velocity_edges[1:])
        velocity_width = float(velocity_edges[1] - velocity_edges[0]) if velocity_edges.size > 1 else 1.0
        self.velocity_plot.addItem(
            pg.BarGraphItem(
                x=velocity_centers,
                height=velocity_hist["positive"],
                width=velocity_width * 0.85,
                brush=pg.mkBrush(QtGui.QColor(89, 161, 79, 140)),
                pen=pg.mkPen("#59a14f", width=1),
            )
        )
        self.velocity_plot.addItem(
            pg.BarGraphItem(
                x=velocity_centers,
                height=velocity_hist["negative"],
                width=velocity_width * 0.85,
                brush=pg.mkBrush(QtGui.QColor(225, 87, 89, 140)),
                pen=pg.mkPen("#e15759", width=1),
            )
        )
        self._add_legend_entry(self.velocity_plot, "Compression (+)", "#59a14f")
        self._add_legend_entry(self.velocity_plot, "Rebound (-)", "#e15759")
        self.velocity_plot.addItem(
            pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#808080", style=QtCore.Qt.PenStyle.DashLine))
        )
        self.velocity_plot.setTitle(f"{channel.capitalize()} velocity distribution")
        self.velocity_plot.setLabel("bottom", velocity_hist.get("label", "Velocity"), units=velocity_hist["units"])
        _apply_velocity_axis_ticks(self.velocity_plot.getPlotItem(), velocity_hist.get("velocity_axis_ticks"))
        self.velocity_plot.setLabel(
            "left",
            velocity_hist.get("occupancy_label", "Time in bin"),
            units=velocity_hist.get("occupancy_units", "s"),
        )

    def clear(self) -> None:
        self.travel_plot.clear()
        self.velocity_plot.clear()


class HistogramsWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.front_pane = _ChannelHistogramsPane()
        self.rear_pane = _ChannelHistogramsPane()
        layout.addWidget(self.front_pane, 1)
        layout.addWidget(self.rear_pane, 1)

    def set_histograms(
        self,
        front_travel_hist: dict[str, Any],
        front_velocity_hist: dict[str, Any],
        rear_travel_hist: dict[str, Any],
        rear_velocity_hist: dict[str, Any],
    ) -> None:
        self.front_pane.set_histograms("front", front_travel_hist, front_velocity_hist)
        self.rear_pane.set_histograms("rear", rear_travel_hist, rear_velocity_hist)

    def clear(self) -> None:
        self.front_pane.clear()
        self.rear_pane.clear()


class CompareWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.summary_label = QtWidgets.QLabel(
            f"Select {COMPARE_MIN_SESSIONS}-{COMPARE_MAX_SESSIONS} rows in Data base and click Compare Selected."
        )
        self.summary_label.setWordWrap(True)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        layout.addWidget(scroll_area, 1)

        self.content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area.setWidget(self.content_widget)
        content_layout.addWidget(self.summary_label)

        self.plots_widget = QtWidgets.QWidget()
        plots_layout = QtWidgets.QGridLayout(self.plots_widget)
        plots_layout.setContentsMargins(0, 0, 0, 0)

        self.front_travel_plot = self._build_plot_widget()
        self.front_travel_legend = self.front_travel_plot.addLegend()
        self.rear_travel_plot = self._build_plot_widget()
        self.rear_travel_legend = self.rear_travel_plot.addLegend()
        self.front_velocity_plot = self._build_plot_widget()
        self.front_velocity_legend = self.front_velocity_plot.addLegend()
        self.rear_velocity_plot = self._build_plot_widget()
        self.rear_velocity_legend = self.rear_velocity_plot.addLegend()

        for plot_widget in (
            self.front_travel_plot,
            self.rear_travel_plot,
            self.front_velocity_plot,
            self.rear_velocity_plot,
        ):
            plot_widget.setMinimumHeight(260)

        plots_layout.addWidget(self.front_travel_plot, 0, 0)
        plots_layout.addWidget(self.rear_travel_plot, 0, 1)
        plots_layout.addWidget(self.front_velocity_plot, 1, 0)
        plots_layout.addWidget(self.rear_velocity_plot, 1, 1)
        content_layout.addWidget(self.plots_widget)

        tables_widget = QtWidgets.QWidget()
        tables_layout = QtWidgets.QHBoxLayout(tables_widget)
        tables_layout.setContentsMargins(0, 0, 0, 0)
        self.front_metrics_group = QtWidgets.QGroupBox("Front riding metrics")
        front_metrics_layout = QtWidgets.QVBoxLayout(self.front_metrics_group)
        self.front_metrics_table = self._build_metrics_table()
        front_metrics_layout.addWidget(self.front_metrics_table)
        self.rear_metrics_group = QtWidgets.QGroupBox("Rear riding metrics")
        rear_metrics_layout = QtWidgets.QVBoxLayout(self.rear_metrics_group)
        self.rear_metrics_table = self._build_metrics_table()
        rear_metrics_layout.addWidget(self.rear_metrics_table)
        tables_layout.addWidget(self.front_metrics_group, 1)
        tables_layout.addWidget(self.rear_metrics_group, 1)
        content_layout.addWidget(tables_widget)

    @staticmethod
    def _build_plot_widget() -> pg.PlotWidget:
        plot_widget = pg.PlotWidget()
        plot_widget.getPlotItem().showGrid(x=True, y=True, alpha=0.2)
        return plot_widget

    @staticmethod
    def _build_metrics_table() -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Metric", "Session", "Units"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return table

    @staticmethod
    def _clear_plot(plot_widget: pg.PlotWidget, legend: pg.LegendItem) -> None:
        plot_widget.clear()
        legend.clear()

    @staticmethod
    def _shared_velocity_range(bundles: list[SessionBundle], channel: str) -> tuple[float, float]:
        minima: list[float] = []
        maxima: list[float] = []
        for bundle in bundles:
            series = channel_series(bundle.derived_df, channel, series_mode="stroke_percent")
            velocity = series["velocity"]
            finite_velocity = velocity[np.isfinite(velocity)]
            if finite_velocity.size == 0:
                continue
            minima.append(float(np.min(finite_velocity)))
            maxima.append(float(np.max(finite_velocity)))

        if not minima or not maxima:
            return (-1.0, 1.0)

        limit = max(abs(min(minima)), abs(max(maxima)))
        if not np.isfinite(limit) or limit <= 0.0:
            limit = 1.0
        return (-limit, limit)

    @staticmethod
    def _set_overlay_histogram(
        plot_widget: pg.PlotWidget,
        legend: pg.LegendItem,
        title: str,
        label: str,
        units: str,
        histograms: list[tuple[str, dict[str, Any]]],
        *,
        x_range: tuple[float, float] | None = None,
        axis_ticks: list[tuple[float, str]] | None = None,
        zero_line: bool = False,
    ) -> None:
        plot_widget.clear()
        legend.clear()

        plot_item = plot_widget.getPlotItem()
        plot_item.setTitle(title)
        plot_item.setLabel("bottom", label, units=units)
        _apply_velocity_axis_ticks(plot_item, axis_ticks)
        occupancy_label = "Time in bin"
        occupancy_units = "s"
        if histograms:
            occupancy_label = histograms[0][1].get("occupancy_label", occupancy_label)
            occupancy_units = histograms[0][1].get("occupancy_units", occupancy_units)
        plot_item.setLabel("left", occupancy_label, units=occupancy_units)

        for index, (session_label, histogram) in enumerate(histograms):
            edges = histogram["edges"]
            centers = 0.5 * (edges[:-1] + edges[1:])
            plot_widget.plot(
                centers,
                histogram["histogram"],
                pen=pg.mkPen(COMPARE_SESSION_COLORS[index % len(COMPARE_SESSION_COLORS)], width=2),
                name=session_label,
            )

        if zero_line:
            plot_widget.addItem(pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#808080", style=QtCore.Qt.PenStyle.DashLine)))
        if x_range is not None:
            plot_widget.setXRange(x_range[0], x_range[1], padding=0.02)

    @staticmethod
    def _set_overlay_bar_histogram(
        plot_widget: pg.PlotWidget,
        legend: pg.LegendItem,
        title: str,
        label: str,
        units: str,
        histograms: list[tuple[str, dict[str, Any]]],
        *,
        x_range: tuple[float, float] | None = None,
        axis_ticks: list[tuple[float, str]] | None = None,
        zero_line: bool = False,
    ) -> None:
        plot_widget.clear()
        legend.clear()

        plot_item = plot_widget.getPlotItem()
        plot_item.setTitle(title)
        plot_item.setLabel("bottom", label, units=units)
        _apply_velocity_axis_ticks(plot_item, axis_ticks)
        occupancy_label = "Time in bin"
        occupancy_units = "s"
        if histograms:
            occupancy_label = histograms[0][1].get("occupancy_label", occupancy_label)
            occupancy_units = histograms[0][1].get("occupancy_units", occupancy_units)
        plot_item.setLabel("left", occupancy_label, units=occupancy_units)

        if histograms:
            base_edges = histograms[0][1]["edges"]
            centers = 0.5 * (base_edges[:-1] + base_edges[1:])
            base_width = float(base_edges[1] - base_edges[0]) if base_edges.size > 1 else 1.0
            session_count = len(histograms)
            bar_width = base_width * 0.8 / max(session_count, 1)

            for index, (session_label, histogram) in enumerate(histograms):
                offset = (index - (session_count - 1) / 2.0) * bar_width
                color = COMPARE_SESSION_COLORS[index % len(COMPARE_SESSION_COLORS)]
                plot_widget.addItem(
                    pg.BarGraphItem(
                        x=centers + offset,
                        height=histogram["histogram"],
                        width=bar_width * 0.95,
                        brush=pg.mkBrush(QtGui.QColor(color)),
                        pen=pg.mkPen(color, width=1),
                    )
                )
                plot_widget.plot([], [], pen=pg.mkPen(color, width=2), name=session_label)

        if zero_line:
            plot_widget.addItem(pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#808080", style=QtCore.Qt.PenStyle.DashLine)))
        if x_range is not None:
            plot_widget.setXRange(x_range[0], x_range[1], padding=0.02)

    @staticmethod
    def _metric_value(metric_map: dict[str, dict[str, Any]], metric_name: str) -> dict[str, Any]:
        return metric_map.get(metric_name, {"value": None, "units": ""})

    @staticmethod
    def _visible_riding_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            metric
            for metric in metrics
            if metric.get("category") == "riding" and metric.get("metrics_tab_visible", True)
        ]

    @staticmethod
    def _resize_metrics_table_to_contents(table: QtWidgets.QTableWidget) -> None:
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        height = table.horizontalHeader().height() + 2 * table.frameWidth() + 4
        for row_index in range(table.rowCount()):
            height += table.rowHeight(row_index)
        height += table.horizontalScrollBar().sizeHint().height()
        table.setMinimumHeight(height)
        table.setMaximumHeight(height)

    def set_comparison(
        self,
        bundles: list[SessionBundle],
        travel_bins: int,
        velocity_bins: int,
        velocity_axis_mode: str = "linear",
    ) -> None:
        if len(bundles) < COMPARE_MIN_SESSIONS:
            self.clear()
            return

        bundles = bundles[:COMPARE_MAX_SESSIONS]
        velocity_axis_mode = normalize_velocity_axis_mode(velocity_axis_mode)
        session_ids = [bundle_session_id(bundle) for bundle in bundles]
        session_labels = [bundle_compare_label(bundle) for bundle in bundles]

        self.summary_label.setText(
            "Compare:\n"
            + "\n".join(f"  {bundle_session_label(bundle)}" for bundle in bundles)
            + "\n"
            "Travel and velocity plots use relative time (%) on normalized %stroke axes."
        )

        travel_hist_range = (0.0, 100.0)
        front_velocity_range = self._shared_velocity_range(bundles, "front")
        rear_velocity_range = self._shared_velocity_range(bundles, "rear")

        front_travel_histograms = [
            (
                session_id,
                compute_travel_histogram(
                    bundle.derived_df,
                    "front",
                    bins=travel_bins,
                    series_mode="stroke_percent",
                    histogram_range=travel_hist_range,
                    relative_occupancy=True,
                ),
            )
            for session_id, bundle in zip(session_labels, bundles)
        ]
        rear_travel_histograms = [
            (
                session_id,
                compute_travel_histogram(
                    bundle.derived_df,
                    "rear",
                    bins=travel_bins,
                    series_mode="stroke_percent",
                    histogram_range=travel_hist_range,
                    relative_occupancy=True,
                ),
            )
            for session_id, bundle in zip(session_labels, bundles)
        ]
        front_velocity_histograms = []
        rear_velocity_histograms = []
        for session_label, bundle in zip(session_labels, bundles):
            front_velocity_hist = compute_velocity_histogram(
                bundle.derived_df,
                "front",
                bins=velocity_bins,
                series_mode="stroke_percent",
                histogram_range=front_velocity_range,
                relative_occupancy=True,
                velocity_axis_mode=velocity_axis_mode,
            )
            front_velocity_histograms.append((session_label, {**front_velocity_hist, "histogram": front_velocity_hist["all"]}))
            rear_velocity_hist = compute_velocity_histogram(
                bundle.derived_df,
                "rear",
                bins=velocity_bins,
                series_mode="stroke_percent",
                histogram_range=rear_velocity_range,
                relative_occupancy=True,
                velocity_axis_mode=velocity_axis_mode,
            )
            rear_velocity_histograms.append((session_label, {**rear_velocity_hist, "histogram": rear_velocity_hist["all"]}))

        self._set_overlay_histogram(
            self.front_travel_plot,
            self.front_travel_legend,
            "Front position distribution",
            front_travel_histograms[0][1]["label"],
            front_travel_histograms[0][1]["units"],
            front_travel_histograms,
            x_range=travel_hist_range,
        )
        self._set_overlay_histogram(
            self.rear_travel_plot,
            self.rear_travel_legend,
            "Rear position distribution",
            rear_travel_histograms[0][1]["label"],
            rear_travel_histograms[0][1]["units"],
            rear_travel_histograms,
            x_range=travel_hist_range,
        )
        self._set_overlay_bar_histogram(
            self.front_velocity_plot,
            self.front_velocity_legend,
            "Front velocity distribution",
            front_velocity_histograms[0][1]["label"],
            front_velocity_histograms[0][1]["units"],
            front_velocity_histograms,
            x_range=front_velocity_histograms[0][1]["velocity_range"],
            axis_ticks=front_velocity_histograms[0][1].get("velocity_axis_ticks"),
            zero_line=True,
        )
        self._set_overlay_bar_histogram(
            self.rear_velocity_plot,
            self.rear_velocity_legend,
            "Rear velocity distribution",
            rear_velocity_histograms[0][1]["label"],
            rear_velocity_histograms[0][1]["units"],
            rear_velocity_histograms,
            x_range=rear_velocity_histograms[0][1]["velocity_range"],
            axis_ticks=rear_velocity_histograms[0][1].get("velocity_axis_ticks"),
            zero_line=True,
        )

        front_metric_maps: list[dict[str, dict[str, Any]]] = []
        rear_metric_maps: list[dict[str, dict[str, Any]]] = []
        visible_front_metric_names: list[str] = []
        for bundle_index, bundle in enumerate(bundles):
            analog_resolution_bits = int(bundle.summary["header"]["analog_resolution_bits"])
            front_metrics = self._visible_riding_metrics(
                compute_channel_metrics(
                    bundle.derived_df,
                    "front",
                    analog_resolution_bits,
                    series_mode="stroke_percent",
                )
            )
            rear_metrics = self._visible_riding_metrics(
                compute_channel_metrics(
                    bundle.derived_df,
                    "rear",
                    analog_resolution_bits,
                    series_mode="stroke_percent",
                )
            )
            if bundle_index == 0:
                visible_front_metric_names = [str(metric["metric"]) for metric in front_metrics]
            front_metric_maps.append({str(metric["metric"]): metric for metric in front_metrics})
            rear_metric_maps.append({str(metric["metric"]): metric for metric in rear_metrics})

        self._populate_compare_metric_table(
            self.front_metrics_table,
            visible_front_metric_names,
            front_metric_maps,
            session_labels,
        )
        self._populate_compare_metric_table(
            self.rear_metrics_table,
            visible_front_metric_names,
            rear_metric_maps,
            session_labels,
        )

    def _populate_compare_metric_table(
        self,
        table: QtWidgets.QTableWidget,
        metric_names: list[str],
        metric_maps: list[dict[str, dict[str, Any]]],
        session_ids: list[str],
    ) -> None:
        table.setColumnCount(2 + len(session_ids))
        table.setHorizontalHeaderLabels(["Metric", *session_ids, "Units"])
        table.setRowCount(len(metric_names))
        for row_index, metric_name in enumerate(metric_names):
            row_metrics = [self._metric_value(metric_map, metric_name) for metric_map in metric_maps]
            units = next((str(metric.get("units", "")) for metric in row_metrics if metric.get("units")), "")
            values = [
                metric_name,
                *[format_metrics_table_value(metric) for metric in row_metrics],
                units,
            ]
            for column, value in enumerate(values):
                table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(value)))
        self._resize_metrics_table_to_contents(table)

    def clear(self) -> None:
        self.summary_label.setText(
            f"Select {COMPARE_MIN_SESSIONS}-{COMPARE_MAX_SESSIONS} rows in Data base and click Compare Selected."
        )
        self._clear_plot(self.front_travel_plot, self.front_travel_legend)
        self._clear_plot(self.rear_travel_plot, self.rear_travel_legend)
        self._clear_plot(self.front_velocity_plot, self.front_velocity_legend)
        self._clear_plot(self.rear_velocity_plot, self.rear_velocity_legend)
        for table in (self.front_metrics_table, self.rear_metrics_table):
            table.setColumnCount(3)
            table.setHorizontalHeaderLabels(["Metric", "Session", "Units"])
            table.setRowCount(0)
            self._resize_metrics_table_to_contents(table)


class WheelPlotWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.summary_label = QtWidgets.QLabel("No wheel pulse rows")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.speed_plot = pg.PlotWidget()
        self.speed_plot.getPlotItem().showGrid(x=True, y=True, alpha=0.2)
        self.speed_curve = self.speed_plot.plot(
            pen=pg.mkPen("#4e79a7", width=2),
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush("#4e79a7"),
            symbolPen=pg.mkPen("#4e79a7"),
        )
        layout.addWidget(self.speed_plot, 1)

        self.period_plot = pg.PlotWidget()
        self.period_plot.getPlotItem().showGrid(x=True, y=True, alpha=0.2)
        self.period_curve = self.period_plot.plot(
            pen=pg.mkPen("#f28e2b", width=2),
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush("#f28e2b"),
            symbolPen=pg.mkPen("#f28e2b"),
        )
        layout.addWidget(self.period_plot, 1)

    def set_wheel_data(self, wheel_df: pl.DataFrame, start_time_s: float | None = None) -> None:
        time_s = frame_series(wheel_df, "host_time_s")
        speed_kph = frame_series(wheel_df, "speed_kph")
        period_s = frame_series(wheel_df, "period_s")
        pulse_count = frame_series(wheel_df, "pulse_count")

        speed_time, speed_values = finite_xy(time_s, speed_kph)
        period_time, period_values = finite_xy(time_s, period_s)
        finite_pulse_times = time_s[np.isfinite(time_s)]
        if speed_time.size and finite_pulse_times.size and start_time_s is not None and np.isfinite(start_time_s):
            first_pulse_time = float(finite_pulse_times[0])
            zero_start_time = float(start_time_s)
            if zero_start_time < first_pulse_time:
                speed_time = np.concatenate(
                    (
                        np.asarray([zero_start_time, first_pulse_time], dtype=np.float64),
                        speed_time,
                    )
                )
                speed_values = np.concatenate(
                    (
                        np.asarray([0.0, 0.0], dtype=np.float64),
                        speed_values,
                    )
                )
        finite_pulses = pulse_count[np.isfinite(pulse_count)]
        pulse_total = int(np.max(finite_pulses)) if finite_pulses.size else 0

        if pulse_total == 0:
            self.summary_label.setText("No wheel pulse rows in this session.")
        else:
            speed_stat_values = speed_values[np.isfinite(speed_values)]
            max_speed = float(np.max(speed_stat_values)) if speed_stat_values.size else None
            mean_speed = float(np.mean(speed_stat_values)) if speed_stat_values.size else None
            if max_speed is not None and mean_speed is not None:
                self.summary_label.setText(
                    f"Max. Speed {max_speed:.2f} km/h | Mean Speed {mean_speed:.2f} km/h"
                )
            else:
                self.summary_label.setText("No valid wheel speed rows in this session.")

        self.speed_curve.setData(speed_time, speed_values)
        self.period_curve.setData(period_time, period_values)

        speed_item = self.speed_plot.getPlotItem()
        speed_item.setTitle("Front wheel speed")
        speed_item.setLabel("bottom", "Time", units="s")
        speed_item.setLabel("left", "Speed", units="km/h")

        period_item = self.period_plot.getPlotItem()
        period_item.setTitle("Wheel pulse period")
        period_item.setLabel("bottom", "Time", units="s")
        period_item.setLabel("left", "Period", units="s")

    def clear(self) -> None:
        self.summary_label.setText("No wheel pulse rows")
        self.speed_curve.setData([], [])
        self.period_curve.setData([], [])


class ImuPlotWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.summary_label = QtWidgets.QLabel("No IMU frame rows")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.accel_plot = pg.PlotWidget()
        accel_item = self.accel_plot.getPlotItem()
        accel_item.showGrid(x=True, y=True, alpha=0.2)
        accel_item.addLegend()
        self.accel_curves = {
            "x": accel_item.plot(pen=pg.mkPen("#e15759", width=1.5), name="Accel X"),
            "y": accel_item.plot(pen=pg.mkPen("#59a14f", width=1.5), name="Accel Y"),
            "z": accel_item.plot(pen=pg.mkPen("#4e79a7", width=1.5), name="Accel Z"),
        }
        layout.addWidget(self.accel_plot, 1)

        self.gyro_plot = pg.PlotWidget()
        gyro_item = self.gyro_plot.getPlotItem()
        gyro_item.showGrid(x=True, y=True, alpha=0.2)
        gyro_item.addLegend()
        self.gyro_curves = {
            "x": gyro_item.plot(pen=pg.mkPen("#e15759", width=1.5), name="Gyro X"),
            "y": gyro_item.plot(pen=pg.mkPen("#59a14f", width=1.5), name="Gyro Y"),
            "z": gyro_item.plot(pen=pg.mkPen("#4e79a7", width=1.5), name="Gyro Z"),
        }
        layout.addWidget(self.gyro_plot, 1)

    def set_imu_data(self, imu_frame_df: pl.DataFrame) -> None:
        time_s = frame_series(imu_frame_df, "estimated_host_time_s")
        if time_s.size == 0 or not np.isfinite(time_s).any():
            time_s = frame_series(imu_frame_df, "burst_host_time_s")

        accel_x = frame_series(imu_frame_df, "accel_x_g")
        accel_y = frame_series(imu_frame_df, "accel_y_g")
        accel_z = frame_series(imu_frame_df, "accel_z_g")
        gyro_x = frame_series(imu_frame_df, "gyro_x_dps")
        gyro_y = frame_series(imu_frame_df, "gyro_y_dps")
        gyro_z = frame_series(imu_frame_df, "gyro_z_dps")

        accel_mask = np.isfinite(time_s) & np.isfinite(accel_x) & np.isfinite(accel_y) & np.isfinite(accel_z)
        gyro_mask = np.isfinite(time_s) & np.isfinite(gyro_x) & np.isfinite(gyro_y) & np.isfinite(gyro_z)
        accel_count = int(np.count_nonzero(accel_mask))
        gyro_count = int(np.count_nonzero(gyro_mask))

        if accel_count == 0 and gyro_count == 0:
            self.summary_label.setText("No IMU accel or gyro frame rows in this session.")
        else:
            summary_parts = [f"Accel samples: {accel_count}", f"Gyro samples: {gyro_count}"]
            if accel_count:
                accel_mag = np.sqrt(accel_x[accel_mask] ** 2 + accel_y[accel_mask] ** 2 + accel_z[accel_mask] ** 2)
                summary_parts.append(f"Peak |accel|: {float(np.max(accel_mag)):.2f} g")
            if gyro_count:
                gyro_mag = np.sqrt(gyro_x[gyro_mask] ** 2 + gyro_y[gyro_mask] ** 2 + gyro_z[gyro_mask] ** 2)
                summary_parts.append(f"Peak |gyro|: {float(np.max(gyro_mag)):.1f} dps")
            self.summary_label.setText(" | ".join(summary_parts))

        accel_series = {
            "x": accel_x,
            "y": accel_y,
            "z": accel_z,
        }
        for axis, values in accel_series.items():
            x, y = finite_xy(time_s, values)
            self.accel_curves[axis].setData(x, y)

        gyro_series = {
            "x": gyro_x,
            "y": gyro_y,
            "z": gyro_z,
        }
        for axis, values in gyro_series.items():
            x, y = finite_xy(time_s, values)
            self.gyro_curves[axis].setData(x, y)

        accel_item = self.accel_plot.getPlotItem()
        accel_item.setTitle("IMU acceleration")
        accel_item.setLabel("bottom", "Time", units="s")
        accel_item.setLabel("left", "Accel", units="g")

        gyro_item = self.gyro_plot.getPlotItem()
        gyro_item.setTitle("IMU angular rate")
        gyro_item.setLabel("bottom", "Time", units="s")
        gyro_item.setLabel("left", "Gyro", units="dps")

    def clear(self) -> None:
        self.summary_label.setText("No IMU frame rows")
        for curve in self.accel_curves.values():
            curve.setData([], [])
        for curve in self.gyro_curves.values():
            curve.setData([], [])


class BlankAnalysisWidget(QtWidgets.QWidget):
    settings_changed = QtCore.Signal()

    def clear(self) -> None:
        pass

    def load_settings(self, config: dict[str, Any]) -> None:
        _ = config

    def current_settings(self) -> dict[str, Any]:
        return copy.deepcopy(DEFAULT_BREAKDOWN_CONFIG)

    def set_balance_data(self, balance: dict[str, Any]) -> None:
        _ = balance

    def set_breakdown_analysis(self, analysis: dict[str, Any]) -> None:
        _ = analysis


class BalancePlotWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.summary_label = QtWidgets.QLabel("No balance data")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.trace_plot = pg.PlotWidget()
        trace_item = self.trace_plot.getPlotItem()
        trace_item.showGrid(x=True, y=True, alpha=0.2)
        trace_item.addLegend()
        self.front_trace = trace_item.plot(
            pen=pg.mkPen("#e15759", width=2),
            name="Front %",
            autoDownsample=True,
            clipToView=True,
        )
        self.rear_trace = trace_item.plot(
            pen=pg.mkPen("#4e79a7", width=2),
            name="Rear %",
            autoDownsample=True,
            clipToView=True,
        )
        layout.addWidget(self.trace_plot, 1)

        self.scatter_plot = pg.PlotWidget()
        scatter_item = self.scatter_plot.getPlotItem()
        scatter_item.showGrid(x=True, y=True, alpha=0.2)
        self.balance_scatter = pg.ScatterPlotItem(
            size=4,
            brush=pg.mkBrush(78, 121, 167, 110),
            pen=pg.mkPen(78, 121, 167, 140),
        )
        self.balance_line = pg.PlotCurveItem(
            pen=pg.mkPen("#9c755f", width=1.5, style=QtCore.Qt.PenStyle.DashLine)
        )
        scatter_item.addItem(self.balance_scatter)
        scatter_item.addItem(self.balance_line)
        layout.addWidget(self.scatter_plot, 1)

    def set_balance_data(self, balance: dict[str, Any]) -> None:
        time_s = np.asarray(balance["time_s"], dtype=np.float64)
        front_percent = np.asarray(balance["front_percent"], dtype=np.float64)
        rear_percent = np.asarray(balance["rear_percent"], dtype=np.float64)

        trace_mask = np.isfinite(time_s) & np.isfinite(front_percent) & np.isfinite(rear_percent)
        trace_time, trace_series = downsample_shared_x_preserve_extrema(
            time_s[trace_mask],
            [front_percent[trace_mask], rear_percent[trace_mask]],
            6000,
        )
        front_trace, rear_trace = trace_series
        self.front_trace.setData(trace_time, front_trace)
        self.rear_trace.setData(trace_time, rear_trace)

        active_mask = np.asarray(balance["active_mask"], dtype=bool)
        scatter_mask = active_mask & np.isfinite(front_percent) & np.isfinite(rear_percent)
        scatter_x, scatter_y = downsample_xy(rear_percent[scatter_mask], front_percent[scatter_mask], 4000)
        self.balance_scatter.setData(scatter_x, scatter_y)

        diagonal_limit = 100.0
        if np.count_nonzero(scatter_mask):
            diagonal_limit = max(
                100.0,
                float(np.nanmax(front_percent[scatter_mask])),
                float(np.nanmax(rear_percent[scatter_mask])),
            )
        self.balance_line.setData([0.0, diagonal_limit], [0.0, diagonal_limit])

        trace_item = self.trace_plot.getPlotItem()
        trace_item.setTitle("Front / rear normalized travel")
        trace_item.setLabel("bottom", "Time", units="s")
        trace_item.setLabel("left", "Used stroke", units="%")

        scatter_item = self.scatter_plot.getPlotItem()
        scatter_item.setTitle("Front vs rear normalized travel")
        scatter_item.setLabel("bottom", "Rear used stroke", units="%")
        scatter_item.setLabel("left", "Front used stroke", units="%")

        if not np.any(trace_mask):
            self.summary_label.setText("No front/rear balance data available for this session.")
            return

        direction = "front deeper" if (balance.get("mean_balance_pct") or 0.0) >= 0.0 else "rear deeper"
        summary_parts = [
            "Normalization: each channel scaled to 0..100% of its own used stroke.",
            f"Scatter active threshold: combined travel >= {int(balance['active_threshold_pct'])}% used stroke.",
            f"Mean balance: {format_metric_value(balance.get('mean_balance_pct'))} p.p. ({direction})",
            f"RMS balance: {format_metric_value(balance.get('rms_balance_pct'))} p.p.",
            f"Front share when active: {format_metric_value(balance.get('mean_front_share_pct'))} %",
            f"Correlation: {format_metric_value(balance.get('correlation'))}",
            f"Front-biased time: {format_metric_value(balance.get('front_bias_time_pct'))} %",
            f"Rear-biased time: {format_metric_value(balance.get('rear_bias_time_pct'))} %",
            f"Within +/-{int(balance['neutral_band_pct'])} p.p.: {format_metric_value(balance.get('neutral_time_pct'))} %",
            f"Active balance time: {format_metric_value(balance.get('active_time_s'))} s",
        ]

        front_used = balance.get("front_used_stroke")
        rear_used = balance.get("rear_used_stroke")
        if front_used is not None and rear_used is not None:
            summary_parts.insert(
                1,
                (
                    f"Used stroke: front {format_metric_value(front_used)} {balance.get('front_travel_units', '')}"
                    f" | rear {format_metric_value(rear_used)} {balance.get('rear_travel_units', '')}"
                ),
            )
        self.summary_label.setText(" | ".join(summary_parts))

    def clear(self) -> None:
        self.summary_label.setText("No balance data")
        self.front_trace.setData([], [])
        self.rear_trace.setData([], [])
        self.balance_scatter.setData([], [])
        self.balance_line.setData([], [])


class BrakingWidget(QtWidgets.QWidget):
    settings_changed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = copy.deepcopy(DEFAULT_BRAKING_CONFIG)
        self._updating_controls = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        layout.addWidget(self.scroll_area, 1)

        self.content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setWidget(self.content_widget)

        self.method_label = QtWidgets.QLabel(
            "Wheel-speed deceleration analysis using the selected absolute or percentile decel threshold. "
            "Loaded front-fork stroke is compared against same-speed coasting, and the results are further "
            "analyzed across speed bins."
        )
        self.method_label.setWordWrap(True)
        content_layout.addWidget(self.method_label)

        controls_group = QtWidgets.QGroupBox("Braking controls")
        controls_layout = QtWidgets.QHBoxLayout(controls_group)
        self.threshold_type_combo = QtWidgets.QComboBox()
        self.threshold_type_combo.addItem("Percentile", "percentile")
        self.threshold_type_combo.addItem("Absolute", "absolute")
        self.threshold_label_combo = QtWidgets.QComboBox()
        controls_layout.addWidget(QtWidgets.QLabel("Threshold type"))
        controls_layout.addWidget(self.threshold_type_combo)
        controls_layout.addWidget(QtWidgets.QLabel("Selected threshold"))
        controls_layout.addWidget(self.threshold_label_combo)
        controls_layout.addStretch(1)
        content_layout.addWidget(controls_group)

        self.warning_label = QtWidgets.QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #8a5a00; background: #fff4ce; padding: 6px;")
        self.warning_label.hide()
        content_layout.addWidget(self.warning_label)

        self.summary_label = QtWidgets.QLabel("No braking analysis")
        self.summary_label.setWordWrap(True)
        content_layout.addWidget(self.summary_label)

        self.key_table = QtWidgets.QTableWidget(0, 6)
        self.key_table.setHorizontalHeaderLabels(["Section", "Metric", "Value", "Unit", "Status", "Meaning"])
        self.key_table.horizontalHeader().setStretchLastSection(True)
        self.key_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._prepare_full_height_table(self.key_table)
        content_layout.addWidget(self.key_table)

        self.speed_bin_table = QtWidgets.QTableWidget(0, 12)
        self.speed_bin_table.setHorizontalHeaderLabels(
            [
                "Speed bin",
                "Loaded time",
                "Events",
                "Topout %",
                "Median %",
                "P90 %",
                "Coast med %",
                "Dive %",
                "Deep80 %",
                "Rough",
                "P75 pack %",
                "Flag",
            ]
        )
        self.speed_bin_table.horizontalHeader().setStretchLastSection(True)
        self.speed_bin_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._prepare_full_height_table(self.speed_bin_table)
        content_layout.addWidget(self.speed_bin_table)

        self.event_table = QtWidgets.QTableWidget(0, 10)
        self.event_table.setHorizontalHeaderLabels(
            [
                "ID",
                "Speed bin",
                "Start [s]",
                "Duration [s]",
                "Entry speed",
                "Exit speed",
                "Peak decel",
                "Max stroke",
                "P95 stroke",
                "Pack-down",
            ]
        )
        self.event_table.horizontalHeader().setStretchLastSection(True)
        self.event_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._prepare_full_height_table(self.event_table)
        content_layout.addWidget(self.event_table)

        self.flag_table = QtWidgets.QTableWidget(0, 3)
        self.flag_table.setHorizontalHeaderLabels(["Level", "Topic", "Message"])
        self.flag_table.horizontalHeader().setStretchLastSection(True)
        self.flag_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._prepare_full_height_table(self.flag_table)
        content_layout.addWidget(self.flag_table)
        content_layout.addStretch(1)

        self.threshold_type_combo.currentIndexChanged.connect(self._on_threshold_type_changed)
        self.threshold_label_combo.currentIndexChanged.connect(self._emit_settings_changed)
        self.load_settings(copy.deepcopy(DEFAULT_BRAKING_CONFIG))
        self.clear()

    @staticmethod
    def _prepare_full_height_table(table: QtWidgets.QTableWidget) -> None:
        table.verticalHeader().setVisible(False)
        table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    @staticmethod
    def _resize_table_to_contents(table: QtWidgets.QTableWidget) -> None:
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        height = table.horizontalHeader().height() + 2 * table.frameWidth() + 4
        for row_index in range(table.rowCount()):
            height += table.rowHeight(row_index)
        height += table.horizontalScrollBar().sizeHint().height()
        table.setMinimumHeight(height)
        table.setMaximumHeight(height)

    def _threshold_labels(self, threshold_type: str) -> list[str]:
        config = normalize_braking_config(self._settings)
        if threshold_type == "absolute":
            return [f"{float(value):.2f} ms2" for value in config["absolute_decel_thresholds_mps2"]]
        return [f"P{float(value):g}" for value in config["decel_percentiles"]]

    def _set_threshold_label_options(self, threshold_type: str, selected_label: str) -> None:
        self.threshold_label_combo.blockSignals(True)
        try:
            self.threshold_label_combo.clear()
            labels = self._threshold_labels(threshold_type)
            if selected_label and selected_label not in labels:
                labels.append(selected_label)
            for label in labels:
                self.threshold_label_combo.addItem(label)
            index = self.threshold_label_combo.findText(selected_label)
            self.threshold_label_combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self.threshold_label_combo.blockSignals(False)

    def _on_threshold_type_changed(self) -> None:
        threshold_type = str(self.threshold_type_combo.currentData() or "percentile")
        selected_label = "3.00 ms2" if threshold_type == "absolute" else "P80"
        if self._settings.get("selected_threshold_type") == threshold_type:
            selected_label = str(self._settings.get("selected_threshold_label", selected_label))
        self._set_threshold_label_options(threshold_type, selected_label)
        self._emit_settings_changed()

    def _emit_settings_changed(self) -> None:
        if self._updating_controls:
            return
        self.settings_changed.emit()

    def load_settings(self, config: dict[str, Any]) -> None:
        normalized = normalize_braking_config(config)
        self._settings = copy.deepcopy(normalized)
        self._updating_controls = True
        try:
            threshold_type = str(normalized.get("selected_threshold_type", normalized["threshold_mode"]))
            index = self.threshold_type_combo.findData(threshold_type)
            self.threshold_type_combo.setCurrentIndex(max(index, 0))
            self._set_threshold_label_options(threshold_type, str(normalized.get("selected_threshold_label", "P80")))
        finally:
            self._updating_controls = False

    def current_settings(self) -> dict[str, Any]:
        settings = copy.deepcopy(self._settings)
        threshold_type = str(self.threshold_type_combo.currentData() or "percentile")
        settings["threshold_mode"] = threshold_type
        settings["selected_threshold_type"] = threshold_type
        settings["selected_threshold_label"] = self.threshold_label_combo.currentText()
        return normalize_braking_config(settings)

    def set_braking_analysis(self, analysis: dict[str, Any]) -> None:
        meta = analysis.get("meta", {})
        warning = meta.get("warning")
        if warning:
            self.warning_label.setText(str(warning))
            self.warning_label.show()
        else:
            self.warning_label.hide()
        self.summary_label.setText(str(analysis.get("summary_text", "")))
        self._populate_key_table(analysis.get("key_metrics", []))
        self._populate_speed_bin_table(analysis.get("speed_bin_rows", []))
        self._populate_event_table(analysis.get("event_rows", []), meta.get("selected", {}))
        self._populate_flag_table(analysis.get("flag_rows", []))

    def _populate_key_table(self, rows: list[dict[str, Any]]) -> None:
        self.key_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row.get("section", ""),
                row.get("metric", ""),
                format_metric_value(row.get("value")),
                row.get("unit", ""),
                row.get("status", ""),
                row.get("meaning", ""),
            ]
            for column, value in enumerate(values):
                self.key_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(value)))
        self._resize_table_to_contents(self.key_table)

    def _populate_speed_bin_table(self, rows: list[dict[str, Any]]) -> None:
        self.speed_bin_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row.get("speed_range_kph", ""),
                format_metric_value(row.get("loaded_time_s")),
                format_metric_value(row.get("events")),
                format_metric_value(row.get("topout_pct")),
                format_metric_value(row.get("loaded_median_stroke_pct")),
                format_metric_value(row.get("loaded_p90_stroke_pct")),
                format_metric_value(row.get("coasting_median_stroke_pct")),
                format_metric_value(row.get("braking_dive_index_pct")),
                format_metric_value(row.get("deep80_pct")),
                format_metric_value(row.get("rough_braking_ratio")),
                format_metric_value(row.get("pack_down_p75_pct")),
                row.get("flag", ""),
            ]
            for column, value in enumerate(values):
                self.speed_bin_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(value)))
        self._resize_table_to_contents(self.speed_bin_table)

    def _populate_event_table(self, rows: list[dict[str, Any]], selected: dict[str, Any]) -> None:
        threshold_type = selected.get("threshold_type")
        threshold_label = selected.get("threshold_label")
        selected_rows = [
            row for row in rows
            if row.get("threshold_type") == threshold_type and row.get("threshold_label") == threshold_label
        ]
        self.event_table.setRowCount(len(selected_rows))
        for row_index, row in enumerate(selected_rows):
            values = [
                row.get("global_event_index", ""),
                row.get("speed_bin", ""),
                format_metric_value(row.get("start_time_s")),
                format_metric_value(row.get("duration_s")),
                format_metric_value(row.get("entry_speed_kph")),
                format_metric_value(row.get("exit_speed_kph")),
                format_metric_value(row.get("peak_decel_mps2")),
                format_metric_value(row.get("max_stroke_pct")),
                format_metric_value(row.get("p95_stroke_pct")),
                format_metric_value(row.get("pack_down_pct")),
            ]
            for column, value in enumerate(values):
                self.event_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(value)))
        self._resize_table_to_contents(self.event_table)

    def _populate_flag_table(self, rows: list[dict[str, Any]]) -> None:
        self.flag_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [row.get("level", ""), row.get("topic", ""), row.get("message", "")]
            for column, value in enumerate(values):
                self.flag_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(value)))
        self._resize_table_to_contents(self.flag_table)

    def clear(self) -> None:
        self.summary_label.setText("No braking analysis")
        self.warning_label.hide()
        self.key_table.setRowCount(0)
        self.speed_bin_table.setRowCount(0)
        self.event_table.setRowCount(0)
        self.flag_table.setRowCount(0)
        for table in (self.key_table, self.speed_bin_table, self.event_table, self.flag_table):
            self._resize_table_to_contents(table)


class ContextAnalysisWidget(QtWidgets.QWidget):
    settings_changed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        controls_group = QtWidgets.QGroupBox("Context controls")
        controls_form = QtWidgets.QFormLayout(controls_group)

        self.source_combo = QtWidgets.QComboBox()
        for value, label in SOURCE_OPTIONS:
            self.source_combo.addItem(label, value)
        self.response_combo = QtWidgets.QComboBox()
        for value, label in RESPONSE_OPTIONS:
            self.response_combo.addItem(label, value)

        self.source_bins_spin = QtWidgets.QSpinBox()
        self.source_bins_spin.setRange(4, 100)
        self.response_bins_spin = QtWidgets.QSpinBox()
        self.response_bins_spin.setRange(4, 200)

        self.longitudinal_axis_combo = QtWidgets.QComboBox()
        for value, label in ACCEL_AXIS_OPTIONS:
            self.longitudinal_axis_combo.addItem(label, value)
        self.longitudinal_invert = QtWidgets.QCheckBox("Invert")

        self.vertical_axis_combo = QtWidgets.QComboBox()
        for value, label in ACCEL_AXIS_OPTIONS:
            self.vertical_axis_combo.addItem(label, value)
        self.vertical_invert = QtWidgets.QCheckBox("Invert")

        self.pitch_axis_combo = QtWidgets.QComboBox()
        for value, label in GYRO_AXIS_OPTIONS:
            self.pitch_axis_combo.addItem(label, value)
        self.pitch_invert = QtWidgets.QCheckBox("Invert")

        longitudinal_row = QtWidgets.QHBoxLayout()
        longitudinal_row.setContentsMargins(0, 0, 0, 0)
        longitudinal_row.addWidget(self.longitudinal_axis_combo, 1)
        longitudinal_row.addWidget(self.longitudinal_invert)
        longitudinal_widget = QtWidgets.QWidget()
        longitudinal_widget.setLayout(longitudinal_row)

        vertical_row = QtWidgets.QHBoxLayout()
        vertical_row.setContentsMargins(0, 0, 0, 0)
        vertical_row.addWidget(self.vertical_axis_combo, 1)
        vertical_row.addWidget(self.vertical_invert)
        vertical_widget = QtWidgets.QWidget()
        vertical_widget.setLayout(vertical_row)

        pitch_row = QtWidgets.QHBoxLayout()
        pitch_row.setContentsMargins(0, 0, 0, 0)
        pitch_row.addWidget(self.pitch_axis_combo, 1)
        pitch_row.addWidget(self.pitch_invert)
        pitch_widget = QtWidgets.QWidget()
        pitch_widget.setLayout(pitch_row)

        controls_form.addRow("Source", self.source_combo)
        controls_form.addRow("Response", self.response_combo)
        controls_form.addRow("Context bins", self.source_bins_spin)
        controls_form.addRow("Response bins", self.response_bins_spin)
        controls_form.addRow("Longitudinal accel", longitudinal_widget)
        controls_form.addRow("Vertical accel", vertical_widget)
        controls_form.addRow("Pitch rate", pitch_widget)
        layout.addWidget(controls_group)

        self.warning_label = QtWidgets.QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #8a5a00; background: #fff4ce; padding: 6px;")
        self.warning_label.hide()
        layout.addWidget(self.warning_label)

        self.time_plot = pg.PlotWidget()
        time_item = self.time_plot.getPlotItem()
        time_item.showGrid(x=True, y=True, alpha=0.2)
        time_item.setLabel("bottom", "Time", units="s")
        time_item.showAxis("right")
        self.context_view = pg.ViewBox()
        time_item.scene().addItem(self.context_view)
        time_item.getAxis("right").linkToView(self.context_view)
        self.context_view.setXLink(time_item)
        time_item.vb.sigResized.connect(self._update_time_plot_views)
        self.response_curve = time_item.plot(
            pen=pg.mkPen("#0b84a5", width=2),
            autoDownsample=True,
            clipToView=True,
        )
        self.source_curve = pg.PlotCurveItem(
            pen=pg.mkPen("#e15759", width=1.5),
            autoDownsample=True,
            clipToView=True,
        )
        self.context_view.addItem(self.source_curve)
        layout.addWidget(self.time_plot, 1)

        heatmap_row = QtWidgets.QHBoxLayout()
        self.heatmap_plot = pg.PlotWidget()
        self.heatmap_plot.getPlotItem().showGrid(x=True, y=True, alpha=0.2)
        self.heatmap_image = pg.ImageItem()
        self.heatmap_plot.addItem(self.heatmap_image)
        heatmap_row.addWidget(self.heatmap_plot, 1)

        self.heatmap_lut = pg.HistogramLUTWidget()
        self.heatmap_lut.setImageItem(self.heatmap_image)
        self.heatmap_lut.setMinimumWidth(120)
        heatmap_row.addWidget(self.heatmap_lut)
        heatmap_widget = QtWidgets.QWidget()
        heatmap_widget.setLayout(heatmap_row)
        layout.addWidget(heatmap_widget, 1)

        self.summary_table = QtWidgets.QTableWidget(0, 7)
        self.summary_table.setHorizontalHeaderLabels(
            ["Context bin", "Time [s]", "Mean", "RMS", "P10", "P50", "P90"]
        )
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.summary_table, 1)

        for widget in (
            self.source_combo,
            self.response_combo,
            self.source_bins_spin,
            self.response_bins_spin,
            self.longitudinal_axis_combo,
            self.vertical_axis_combo,
            self.pitch_axis_combo,
        ):
            if isinstance(widget, QtWidgets.QComboBox):
                widget.currentIndexChanged.connect(lambda _=None: self.settings_changed.emit())
            else:
                widget.valueChanged.connect(lambda _=None: self.settings_changed.emit())
        for checkbox in (
            self.longitudinal_invert,
            self.vertical_invert,
            self.pitch_invert,
        ):
            checkbox.toggled.connect(lambda _=None: self.settings_changed.emit())

        self._apply_defaults()
        self.clear()

    def _apply_defaults(self) -> None:
        set_combo_to_data(self.source_combo, str(DEFAULT_CONTEXT_CONFIG["source"]))
        set_combo_to_data(self.response_combo, str(DEFAULT_CONTEXT_CONFIG["response"]))
        self.source_bins_spin.setValue(int(DEFAULT_CONTEXT_CONFIG["source_bins"]))
        self.response_bins_spin.setValue(int(DEFAULT_CONTEXT_CONFIG["response_bins"]))
        set_combo_to_data(self.longitudinal_axis_combo, str(DEFAULT_CONTEXT_CONFIG["longitudinal_axis"]))
        set_combo_to_data(self.vertical_axis_combo, str(DEFAULT_CONTEXT_CONFIG["vertical_axis"]))
        set_combo_to_data(self.pitch_axis_combo, str(DEFAULT_CONTEXT_CONFIG["pitch_axis"]))
        self.longitudinal_invert.setChecked(bool(DEFAULT_CONTEXT_CONFIG["longitudinal_invert"]))
        self.vertical_invert.setChecked(bool(DEFAULT_CONTEXT_CONFIG["vertical_invert"]))
        self.pitch_invert.setChecked(bool(DEFAULT_CONTEXT_CONFIG["pitch_invert"]))

    def _update_time_plot_views(self) -> None:
        plot_item = self.time_plot.getPlotItem()
        self.context_view.setGeometry(plot_item.vb.sceneBoundingRect())
        self.context_view.linkedViewChanged(plot_item.vb, self.context_view.XAxis)

    def current_settings(self) -> dict[str, Any]:
        return {
            "source": combo_current_data(self.source_combo),
            "response": combo_current_data(self.response_combo),
            "source_bins": self.source_bins_spin.value(),
            "response_bins": self.response_bins_spin.value(),
            "longitudinal_axis": combo_current_data(self.longitudinal_axis_combo),
            "longitudinal_invert": self.longitudinal_invert.isChecked(),
            "vertical_axis": combo_current_data(self.vertical_axis_combo),
            "vertical_invert": self.vertical_invert.isChecked(),
            "pitch_axis": combo_current_data(self.pitch_axis_combo),
            "pitch_invert": self.pitch_invert.isChecked(),
        }

    def set_context_analysis(
        self,
        context_df: pl.DataFrame,
        meta: dict[str, Any],
        source: str,
        response: str,
        heatmap: dict[str, Any],
        summary_rows: list[dict[str, Any]],
        summary_enabled: bool,
        warning_text: str | None = None,
    ) -> None:
        source_info = meta["sources"][source]
        response_info = meta["responses"][response]
        time_s = frame_series(context_df, "time_s")
        source_values = frame_series(context_df, source_info["column"])
        response_values = frame_series(context_df, response_info["column"])

        x_response, y_response = finite_xy(time_s, response_values)
        x_source, y_source = finite_xy(time_s, source_values)
        self.response_curve.setData(x_response, y_response)
        self.source_curve.setData(x_source, y_source)

        time_item = self.time_plot.getPlotItem()
        time_item.setTitle(f"{response_info['label']} and {source_info['label']} over time")
        time_item.setLabel("left", response_info["label"], units=response_info["units"])
        time_item.getAxis("right").setLabel(source_info["label"], units=source_info["units"])
        self._update_time_plot_views()

        histogram = np.asarray(heatmap["histogram"], dtype=np.float64)
        self.heatmap_image.setImage(np.sqrt(histogram) if heatmap.get("has_data") else histogram)
        source_min, source_max = heatmap["source_range"]
        response_min, response_max = heatmap["response_range"]
        self.heatmap_image.setRect(
            QtCore.QRectF(
                source_min,
                response_min,
                source_max - source_min,
                response_max - response_min,
            )
        )
        heatmap_item = self.heatmap_plot.getPlotItem()
        heatmap_item.setTitle(f"{source_info['label']} vs {response_info['label']} time heatmap")
        heatmap_item.setLabel("bottom", source_info["label"], units=source_info["units"])
        heatmap_item.setLabel("left", response_info["label"], units=response_info["units"])
        heatmap_item.enableAutoRange()

        self.summary_table.setRowCount(len(summary_rows))
        self.summary_table.setEnabled(summary_enabled)
        for row_index, row in enumerate(summary_rows):
            values = [
                (
                    f"{format_metric_value(row.get('source_min'))} to "
                    f"{format_metric_value(row.get('source_max'))} {source_info['units']}"
                ).strip(),
                format_metric_value(row.get("occupancy_s")),
                format_metric_value(row.get("mean")),
                format_metric_value(row.get("rms")),
                format_metric_value(row.get("p10")),
                format_metric_value(row.get("p50")),
                format_metric_value(row.get("p90")),
            ]
            for column, value in enumerate(values):
                self.summary_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(value)))
        self.summary_table.resizeColumnsToContents()

        if warning_text:
            self.warning_label.setText(warning_text)
            self.warning_label.show()
        else:
            self.warning_label.hide()

    def clear(self) -> None:
        self.response_curve.setData([], [])
        self.source_curve.setData([], [])
        self.heatmap_image.setImage(np.zeros((1, 1), dtype=float))
        self.heatmap_image.setRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))
        self.summary_table.setRowCount(0)
        self.summary_table.setEnabled(True)
        self.warning_label.hide()


class BreakdownHeatmapPane(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.getPlotItem().showGrid(x=True, y=True, alpha=0.2)
        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)
        self.image_item.setLookupTable(matplotlib_colormap("hot").getLookupTable(0.0, 1.0, 256))
        layout.addWidget(self.plot_widget)

    def set_heatmap(self, title: str, heatmap: dict[str, Any]) -> None:
        histogram = np.asarray(heatmap["histogram"], dtype=np.float64)
        if heatmap.get("has_data"):
            positive = histogram[histogram > 0.0]
            if positive.size:
                display = np.log10(np.clip(histogram, float(np.min(positive)), None))
            else:
                display = histogram
        else:
            display = np.zeros((1, 1), dtype=np.float64)
        self.image_item.setImage(display)

        source_min, source_max = heatmap["source_range"]
        response_min, response_max = heatmap["response_range"]
        self.image_item.setRect(
            QtCore.QRectF(
                source_min,
                response_min,
                source_max - source_min,
                response_max - response_min,
            )
        )
        plot_item = self.plot_widget.getPlotItem()
        plot_item.setTitle(title)
        plot_item.setLabel("bottom", heatmap.get("source_label", "Speed"), units=heatmap.get("source_units", ""))
        plot_item.setLabel("left", heatmap.get("response_label", "Response"), units=heatmap.get("response_units", ""))
        plot_item.enableAutoRange()

    def clear(self) -> None:
        self.image_item.setImage(np.zeros((1, 1), dtype=np.float64))
        self.image_item.setRect(QtCore.QRectF(0.0, 0.0, 1.0, 1.0))


class BreakdownWidget(QtWidgets.QWidget):
    settings_changed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._analysis: dict[str, Any] | None = None
        self._selected_state_id: str | None = None
        self._selected_segment_id: str | None = None
        self._settings = copy.deepcopy(DEFAULT_BREAKDOWN_CONFIG)
        self._updating_controls = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        controls_group = QtWidgets.QGroupBox("Breakdown controls")
        controls_grid = QtWidgets.QGridLayout(controls_group)

        self.speed_bands_edit = QtWidgets.QLineEdit()
        self.motion_smoothing_spin = QtWidgets.QSpinBox()
        self.motion_smoothing_spin.setRange(50, 5000)
        self.braking_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.braking_threshold_spin.setRange(0.05, 20.0)
        self.braking_threshold_spin.setDecimals(2)
        self.braking_threshold_spin.setSingleStep(0.1)
        self.accel_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.accel_threshold_spin.setRange(0.05, 20.0)
        self.accel_threshold_spin.setDecimals(2)
        self.accel_threshold_spin.setSingleStep(0.1)
        self.min_segment_spin = QtWidgets.QSpinBox()
        self.min_segment_spin.setRange(100, 10000)
        self.repeated_speed_spin = QtWidgets.QDoubleSpinBox()
        self.repeated_speed_spin.setRange(0.0, 200.0)
        self.repeated_speed_spin.setDecimals(1)
        self.repeated_speed_spin.setSingleStep(1.0)
        self.repeated_activity_spin = QtWidgets.QDoubleSpinBox()
        self.repeated_activity_spin.setRange(0.1, 10.0)
        self.repeated_activity_spin.setDecimals(2)
        self.repeated_activity_spin.setSingleStep(0.1)
        self.repeated_duration_spin = QtWidgets.QSpinBox()
        self.repeated_duration_spin.setRange(100, 10000)
        self.detail_source_combo = QtWidgets.QComboBox()
        self.detail_source_combo.addItems(["Selected state", "Manual window"])

        controls_grid.addWidget(QtWidgets.QLabel("Speed bands [km/h]"), 0, 0)
        controls_grid.addWidget(self.speed_bands_edit, 0, 1)
        controls_grid.addWidget(QtWidgets.QLabel("Motion smoothing [ms]"), 0, 2)
        controls_grid.addWidget(self.motion_smoothing_spin, 0, 3)
        controls_grid.addWidget(QtWidgets.QLabel("Braking threshold [m/s²]"), 1, 0)
        controls_grid.addWidget(self.braking_threshold_spin, 1, 1)
        controls_grid.addWidget(QtWidgets.QLabel("Accel threshold [m/s²]"), 1, 2)
        controls_grid.addWidget(self.accel_threshold_spin, 1, 3)
        controls_grid.addWidget(QtWidgets.QLabel("Min segment [ms]"), 2, 0)
        controls_grid.addWidget(self.min_segment_spin, 2, 1)
        controls_grid.addWidget(QtWidgets.QLabel("Repeated-hit min speed [km/h]"), 2, 2)
        controls_grid.addWidget(self.repeated_speed_spin, 2, 3)
        controls_grid.addWidget(QtWidgets.QLabel("Repeated-hit activity"), 3, 0)
        controls_grid.addWidget(self.repeated_activity_spin, 3, 1)
        controls_grid.addWidget(QtWidgets.QLabel("Repeated-hit min duration [ms]"), 3, 2)
        controls_grid.addWidget(self.repeated_duration_spin, 3, 3)
        controls_grid.addWidget(QtWidgets.QLabel("Heatmap detail source"), 4, 0)
        controls_grid.addWidget(self.detail_source_combo, 4, 1)
        layout.addWidget(controls_group)

        self.warning_label = QtWidgets.QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #8a5a00; background: #fff4ce; padding: 6px;")
        self.warning_label.hide()
        layout.addWidget(self.warning_label)

        self.finding_table = QtWidgets.QTableWidget(0, 3)
        self.finding_table.setHorizontalHeaderLabels(["Severity", "Finding", "Detail"])
        self.finding_table.horizontalHeader().setStretchLastSection(True)
        self.finding_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(self.finding_table, 1)

        self.timeline_plot = pg.PlotWidget()
        timeline_item = self.timeline_plot.getPlotItem()
        timeline_item.showGrid(x=True, y=True, alpha=0.2)
        timeline_item.setLabel("bottom", "Time", units="s")
        timeline_item.showAxis("right")
        self.speed_view = pg.ViewBox()
        timeline_item.scene().addItem(self.speed_view)
        timeline_item.getAxis("right").linkToView(self.speed_view)
        self.speed_view.setXLink(timeline_item)
        timeline_item.vb.sigResized.connect(self._update_timeline_views)
        self.front_timeline = timeline_item.plot(
            pen=pg.mkPen("#e15759", width=2),
            name="Front travel",
            autoDownsample=True,
            clipToView=True,
        )
        self.rear_timeline = timeline_item.plot(
            pen=pg.mkPen("#4e79a7", width=2),
            name="Rear travel",
            autoDownsample=True,
            clipToView=True,
        )
        self.speed_timeline = pg.PlotCurveItem(
            pen=pg.mkPen("#222222", width=1.5, style=QtCore.Qt.PenStyle.DashLine),
            autoDownsample=True,
            clipToView=True,
        )
        self.speed_view.addItem(self.speed_timeline)
        self.manual_region = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Vertical, brush=(246, 200, 95, 40))
        self.selected_segment_region = pg.LinearRegionItem(
            values=(0.0, 0.0),
            orientation=pg.LinearRegionItem.Vertical,
            brush=(14, 168, 168, 35),
            movable=False,
        )
        self.timeline_plot.addItem(self.selected_segment_region)
        self.timeline_plot.addItem(self.manual_region)
        self.selected_segment_region.hide()

        lower_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(lower_splitter, 3)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.addWidget(QtWidgets.QLabel("Riding-state summary"))
        self.group_table = QtWidgets.QTableWidget(0, 15)
        self.state_table = self.group_table
        self.group_table.setHorizontalHeaderLabels(
            [
                "Speed band",
                "Motion",
                "Activity",
                "Time [s]",
                "Session [%]",
                "Front med",
                "Rear med",
                "Front P95",
                "Rear P95",
                "Front max",
                "Rear max",
                "Balance",
                "Near-bottom",
                "Bottom-out",
                "Flags",
            ]
        )
        self.group_table.horizontalHeader().setStretchLastSection(True)
        self.group_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        left_layout.addWidget(self.group_table, 1)
        left_layout.addWidget(QtWidgets.QLabel("Event list"))
        self.segment_table = QtWidgets.QTableWidget(0, 10)
        self.event_table = self.segment_table
        self.segment_table.setHorizontalHeaderLabels(
            [
                "ID",
                "Type",
                "Channel",
                "Timestamp [s]",
                "Duration [s]",
                "Speed",
                "Motion",
                "Activity",
                "Front peak",
                "Rear peak",
            ]
        )
        self.segment_table.horizontalHeader().setStretchLastSection(True)
        self.segment_table.setSortingEnabled(True)
        left_layout.addWidget(self.segment_table, 2)
        lower_splitter.addWidget(left_panel)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        self.auto_label = QtWidgets.QLabel("Selected state: none")
        self.auto_label.setWordWrap(True)
        self.manual_label = QtWidgets.QLabel("Manual window: none")
        self.manual_label.setWordWrap(True)
        self.detail_label = QtWidgets.QLabel("Detail source: Selected state")
        self.detail_label.setWordWrap(True)
        right_layout.addWidget(self.auto_label)
        right_layout.addWidget(self.manual_label)
        right_layout.addWidget(self.detail_label)

        self.comparison_table = QtWidgets.QTableWidget(0, 3)
        self.detail_table = self.comparison_table
        self.comparison_table.setHorizontalHeaderLabels(["Metric", "Value", "Units"])
        self.comparison_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.comparison_table, 1)

        heatmap_grid = QtWidgets.QGridLayout()
        self.front_travel_heatmap = BreakdownHeatmapPane()
        self.rear_travel_heatmap = BreakdownHeatmapPane()
        self.front_velocity_heatmap = BreakdownHeatmapPane()
        self.rear_velocity_heatmap = BreakdownHeatmapPane()
        heatmap_grid.addWidget(self.front_travel_heatmap, 0, 0)
        heatmap_grid.addWidget(self.rear_travel_heatmap, 0, 1)
        heatmap_grid.addWidget(self.front_velocity_heatmap, 1, 0)
        heatmap_grid.addWidget(self.rear_velocity_heatmap, 1, 1)
        heatmap_widget = QtWidgets.QWidget()
        heatmap_widget.setLayout(heatmap_grid)
        right_layout.addWidget(heatmap_widget, 2)
        lower_splitter.addWidget(right_panel)
        lower_splitter.setStretchFactor(0, 0)
        lower_splitter.setStretchFactor(1, 1)
        layout.addWidget(self.timeline_plot, 1)

        for widget in (
            self.speed_bands_edit,
            self.motion_smoothing_spin,
            self.braking_threshold_spin,
            self.accel_threshold_spin,
            self.min_segment_spin,
            self.repeated_speed_spin,
            self.repeated_activity_spin,
            self.repeated_duration_spin,
        ):
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.editingFinished.connect(self._emit_settings_changed)
            else:
                widget.valueChanged.connect(lambda _=None: self._emit_settings_changed())
        self.detail_source_combo.currentTextChanged.connect(lambda _=None: self._update_detail_views())
        self.group_table.itemSelectionChanged.connect(self._on_state_selection_changed)
        self.segment_table.itemSelectionChanged.connect(self._on_segment_selection_changed)
        self.manual_region.sigRegionChangeFinished.connect(self._on_manual_region_changed)

        self.load_settings(copy.deepcopy(DEFAULT_BREAKDOWN_CONFIG))
        self.clear()

    def _emit_settings_changed(self) -> None:
        if self._updating_controls:
            return
        self.settings_changed.emit()

    def _update_timeline_views(self) -> None:
        plot_item = self.timeline_plot.getPlotItem()
        self.speed_view.setGeometry(plot_item.vb.sceneBoundingRect())
        self.speed_view.linkedViewChanged(plot_item.vb, self.speed_view.XAxis)

    def load_settings(self, config: dict[str, Any]) -> None:
        normalized = normalize_breakdown_config(config)
        self._settings = copy.deepcopy(normalized)
        self._updating_controls = True
        try:
            self.speed_bands_edit.setText(", ".join(f"{value:g}" for value in normalized["speed_band_edges_kph"]))
            self.motion_smoothing_spin.setValue(int(normalized["motion_smoothing_ms"]))
            self.braking_threshold_spin.setValue(float(normalized["braking_threshold_mps2"]))
            self.accel_threshold_spin.setValue(float(normalized["accel_threshold_mps2"]))
            self.min_segment_spin.setValue(int(normalized["min_segment_ms"]))
            self.repeated_speed_spin.setValue(float(normalized["repeated_hit_min_speed_kph"]))
            self.repeated_activity_spin.setValue(float(normalized["repeated_hit_activity_threshold"]))
            self.repeated_duration_spin.setValue(int(normalized["repeated_hit_min_duration_ms"]))
        finally:
            self._updating_controls = False

    def current_settings(self) -> dict[str, Any]:
        settings = copy.deepcopy(self._settings)
        settings.update(
            {
                "speed_band_edges_kph": parse_float_list(self.speed_bands_edit.text(), "Breakdown speed bands"),
                "motion_smoothing_ms": self.motion_smoothing_spin.value(),
                "braking_threshold_mps2": self.braking_threshold_spin.value(),
                "accel_threshold_mps2": self.accel_threshold_spin.value(),
                "min_segment_ms": self.min_segment_spin.value(),
                "repeated_hit_min_speed_kph": self.repeated_speed_spin.value(),
                "repeated_hit_activity_threshold": self.repeated_activity_spin.value(),
                "repeated_hit_min_duration_ms": self.repeated_duration_spin.value(),
            }
        )
        return normalize_breakdown_config(
            settings
        )

    def set_breakdown_analysis(self, analysis: dict[str, Any]) -> None:
        self._analysis = analysis
        context_df = analysis["context_df"]
        meta = analysis["meta"]
        time_s = frame_series(context_df, "time_s")
        front_travel = frame_series(context_df, "front_travel_resp")
        rear_travel = frame_series(context_df, "rear_travel_resp")
        speed_kph = frame_series(context_df, "speed_kph_ctx")

        timeline_time, front_values = finite_xy(time_s, front_travel)
        _, rear_values = finite_xy(time_s, rear_travel)
        speed_time, speed_values = finite_xy(time_s, speed_kph)
        self.front_timeline.setData(timeline_time, front_values)
        self.rear_timeline.setData(timeline_time, rear_values)
        self.speed_timeline.setData(speed_time, speed_values)

        plot_item = self.timeline_plot.getPlotItem()
        plot_item.setTitle("Speed-conditioned suspension timeline")
        plot_item.setLabel("left", "Travel", units=meta["responses"]["front_travel"]["units"])
        plot_item.getAxis("right").setLabel("Speed", units="km/h")
        self._update_timeline_views()

        state_rows = analysis.get("state_rows", analysis.get("group_rows", []))
        event_rows = analysis.get("event_rows", analysis.get("segment_rows", []))
        self._populate_finding_table(analysis.get("finding_rows", []))
        self._populate_group_table(state_rows, bool(meta["wheel_speed"].get("usable", False)))
        self._populate_segment_table(event_rows, True)

        warnings = [
            f"{row.get('name')}: {row.get('detail')}"
            for row in analysis.get("quality_rows", [])
            if row.get("status") == "warning"
        ]
        if meta["wheel_speed"].get("warning") and not warnings:
            warnings.append(str(meta["wheel_speed"]["warning"]))
        if warnings:
            self.warning_label.setText(" | ".join(str(warning) for warning in warnings))
            self.warning_label.show()
        else:
            self.warning_label.hide()

        if time_s.size:
            start_time = float(np.nanmin(time_s))
            end_time = float(np.nanmax(time_s))
            if state_rows:
                first = state_rows[0]
                self._selected_state_id = str(first["state_id"])
                self._set_region(first["start_time_s"], first["end_time_s"])
                self._set_selected_segment_region(first["start_time_s"], first["end_time_s"], "state")
            else:
                self._selected_state_id = None
                self._selected_segment_id = None
                self._set_region(start_time, end_time)
                self.selected_segment_region.hide()
        self.detail_source_combo.setEnabled(bool(state_rows))
        if state_rows:
            self.detail_source_combo.setCurrentText("Selected state")
        else:
            self.detail_source_combo.setCurrentText("Manual window")
        self._update_detail_views()

    def _set_region(self, start_time_s: float, end_time_s: float) -> None:
        self.manual_region.blockSignals(True)
        self.manual_region.setRegion((float(start_time_s), float(end_time_s)))
        self.manual_region.blockSignals(False)

    def _set_selected_segment_region(self, start_time_s: float, end_time_s: float, segment_type: str) -> None:
        brush = (14, 168, 168, 35) if segment_type in {"speed_motion", "state"} else (148, 103, 189, 40)
        self.selected_segment_region.setBrush(pg.mkBrush(*brush))
        self.selected_segment_region.setRegion((float(start_time_s), float(end_time_s)))
        self.selected_segment_region.show()

    def _populate_finding_table(self, rows: list[dict[str, Any]]) -> None:
        self.finding_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                format_metric_value(row.get("severity")),
                row.get("title", ""),
                row.get("detail", ""),
            ]
            for column, value in enumerate(values):
                self.finding_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(value)))
        self.finding_table.resizeColumnsToContents()

    def _populate_group_table(self, rows: list[dict[str, Any]], enabled: bool) -> None:
        self.group_table.setSortingEnabled(False)
        self.group_table.setRowCount(len(rows))
        self.group_table.setEnabled(enabled)
        for row_index, row in enumerate(rows):
            values = [
                row.get("speed_band", ""),
                row.get("motion_state", ""),
                row.get("activity_state", ""),
                format_metric_value(row.get("time_in_state_s", row.get("duration_s"))),
                format_metric_value(row.get("session_pct")),
                format_metric_value(row.get("front_median_stroke_pct")),
                format_metric_value(row.get("rear_median_stroke_pct")),
                format_metric_value(row.get("front_p95_stroke_pct")),
                format_metric_value(row.get("rear_p95_stroke_pct")),
                format_metric_value(row.get("front_max_stroke_pct")),
                format_metric_value(row.get("rear_max_stroke_pct")),
                format_metric_value(row.get("mean_balance_pct")),
                str((row.get("front_near_full_event_count") or 0) + (row.get("rear_near_full_event_count") or 0)),
                str((row.get("front_bottom_out_event_count") or 0) + (row.get("rear_bottom_out_event_count") or 0)),
                ", ".join(str(flag) for flag in row.get("diagnostic_flags", [])),
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, row.get("state_id"))
                self.group_table.setItem(row_index, column, item)
        self.group_table.setSortingEnabled(True)
        self.group_table.resizeColumnsToContents()
        if rows and enabled:
            self.group_table.selectRow(0)

    def _populate_segment_table(self, rows: list[dict[str, Any]], enabled: bool) -> None:
        self.segment_table.setSortingEnabled(False)
        self.segment_table.setRowCount(len(rows))
        self.segment_table.setEnabled(enabled)
        for row_index, row in enumerate(rows):
            values = [
                row.get("event_id", row.get("segment_id", "")),
                row.get("event_type", row.get("type", "")),
                row.get("channel", ""),
                format_metric_value(row.get("timestamp_s", row.get("start_time_s"))),
                format_metric_value(row.get("duration_s")),
                format_metric_value(row.get("speed_kph", row.get("mean_speed_kph"))),
                row.get("motion_state", ""),
                row.get("activity_state", ""),
                format_metric_value(row.get("front_peak_stroke_pct", row.get("front_max_stroke_pct"))),
                format_metric_value(row.get("rear_peak_stroke_pct", row.get("rear_max_stroke_pct"))),
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, row.get("event_id", row.get("segment_id")))
                self.segment_table.setItem(row_index, column, item)
        self.segment_table.setSortingEnabled(True)
        self.segment_table.resizeColumnsToContents()
        if rows and enabled:
            self.segment_table.selectRow(0)

    def _selected_segment_row(self) -> dict[str, Any] | None:
        if self._analysis is None or self._selected_segment_id is None:
            return None
        for row in self._analysis.get("event_rows", self._analysis.get("segment_rows", [])):
            if str(row.get("event_id", row.get("segment_id"))) == self._selected_segment_id:
                return row
        return None

    def _selected_state_row(self) -> dict[str, Any] | None:
        if self._analysis is None or self._selected_state_id is None:
            return None
        for row in self._analysis.get("state_rows", self._analysis.get("group_rows", [])):
            if str(row.get("state_id")) == self._selected_state_id:
                return row
        return None

    def _state_selection_mask(self, row: dict[str, Any]) -> np.ndarray:
        if self._analysis is None:
            return np.zeros(0, dtype=bool)
        context_df = self._analysis["context_df"]
        speed_band_id = frame_series(context_df, "speed_band_id", -1).astype(np.int64)
        motion_state = np.asarray(context_df.get_column("motion_state").to_list(), dtype=object)
        activity_state = np.asarray(context_df.get_column("activity_state").to_list(), dtype=object)
        return (
            (speed_band_id == int(row.get("speed_band_id", -1)))
            & (motion_state == str(row.get("motion_state", "")))
            & (activity_state == str(row.get("activity_state", "")))
        )

    def _manual_selection_mask(self) -> np.ndarray:
        if self._analysis is None:
            return np.zeros(0, dtype=bool)
        start_time_s, end_time_s = self.manual_region.getRegion()
        return self._analysis["context_df"].get_column("time_s").is_between(
            min(start_time_s, end_time_s),
            max(start_time_s, end_time_s),
            closed="both",
        ).to_numpy()

    def _mask_from_time_range(self, start_time_s: float, end_time_s: float) -> np.ndarray:
        if self._analysis is None:
            return np.zeros(0, dtype=bool)
        return self._analysis["context_df"].get_column("time_s").is_between(
            min(float(start_time_s), float(end_time_s)),
            max(float(start_time_s), float(end_time_s)),
            closed="both",
        ).to_numpy()

    def _detail_text(self, detail: dict[str, Any]) -> str:
        if not detail.get("has_data"):
            return f"{detail.get('label', 'Selection')}: no data"
        return (
            f"{detail.get('label')}: {format_metric_value(detail.get('duration_s'))} s, "
            f"entry {format_metric_value(detail.get('entry_speed_kph'))} km/h, "
            f"exit {format_metric_value(detail.get('exit_speed_kph'))} km/h, "
            f"front used {format_metric_value(detail.get('front_used_stroke'))} {detail.get('front_travel_units', '')}, "
            f"rear used {format_metric_value(detail.get('rear_used_stroke'))} {detail.get('rear_travel_units', '')}"
        )

    def _comparison_rows(self, automatic: dict[str, Any], manual: dict[str, Any]) -> list[tuple[str, Any, Any, str]]:
        return [
            ("Duration", automatic.get("duration_s"), manual.get("duration_s"), "s"),
            ("Mean speed", automatic.get("mean_speed_kph"), manual.get("mean_speed_kph"), "km/h"),
            ("Entry speed", automatic.get("entry_speed_kph"), manual.get("entry_speed_kph"), "km/h"),
            ("Exit speed", automatic.get("exit_speed_kph"), manual.get("exit_speed_kph"), "km/h"),
            ("Speed delta", automatic.get("speed_delta_kph"), manual.get("speed_delta_kph"), "km/h"),
            ("Peak decel", automatic.get("peak_decel_mps2"), manual.get("peak_decel_mps2"), "m/s²"),
            ("Peak accel", automatic.get("peak_accel_mps2"), manual.get("peak_accel_mps2"), "m/s²"),
            ("Front used stroke", automatic.get("front_used_stroke"), manual.get("front_used_stroke"), automatic.get("front_travel_units", "")),
            ("Rear used stroke", automatic.get("rear_used_stroke"), manual.get("rear_used_stroke"), automatic.get("rear_travel_units", "")),
            ("Front RMS vel", automatic.get("front_rms_velocity"), manual.get("front_rms_velocity"), automatic.get("velocity_units", "")),
            ("Rear RMS vel", automatic.get("rear_rms_velocity"), manual.get("rear_rms_velocity"), automatic.get("velocity_units", "")),
            ("Front top 10%", automatic.get("front_top_10_pct"), manual.get("front_top_10_pct"), "%"),
            ("Rear top 10%", automatic.get("rear_top_10_pct"), manual.get("rear_top_10_pct"), "%"),
            ("Mean balance", automatic.get("mean_balance_pct"), manual.get("mean_balance_pct"), "p.p."),
            ("Front share", automatic.get("mean_front_share_pct"), manual.get("mean_front_share_pct"), "%"),
            ("Repeated-hit active", automatic.get("repeated_hit_active_duration_s"), manual.get("repeated_hit_active_duration_s"), "s"),
            ("Oscillations", automatic.get("oscillation_count"), manual.get("oscillation_count"), "count"),
            ("Cycle rate", automatic.get("mean_cycle_rate_hz"), manual.get("mean_cycle_rate_hz"), "Hz"),
            ("Front near-full", automatic.get("front_near_full_event_count"), manual.get("front_near_full_event_count"), "events"),
            ("Rear near-full", automatic.get("rear_near_full_event_count"), manual.get("rear_near_full_event_count"), "events"),
            ("Front bottom-out", automatic.get("front_bottom_out_event_count"), manual.get("front_bottom_out_event_count"), "events"),
            ("Rear bottom-out", automatic.get("rear_bottom_out_event_count"), manual.get("rear_bottom_out_event_count"), "events"),
        ]

    def _detail_rows(self, detail: dict[str, Any]) -> list[tuple[str, Any, str]]:
        return [
            ("Duration", detail.get("duration_s"), "s"),
            ("Mean speed", detail.get("mean_speed_kph"), "km/h"),
            ("Entry speed", detail.get("entry_speed_kph"), "km/h"),
            ("Exit speed", detail.get("exit_speed_kph"), "km/h"),
            ("Peak decel", detail.get("peak_decel_mps2"), "m/s^2"),
            ("Peak accel", detail.get("peak_accel_mps2"), "m/s^2"),
            ("Front median stroke", detail.get("front_median_stroke_pct"), "%"),
            ("Rear median stroke", detail.get("rear_median_stroke_pct"), "%"),
            ("Front P90 stroke", detail.get("front_p90_stroke_pct"), "%"),
            ("Rear P90 stroke", detail.get("rear_p90_stroke_pct"), "%"),
            ("Front P95 stroke", detail.get("front_p95_stroke_pct"), "%"),
            ("Rear P95 stroke", detail.get("rear_p95_stroke_pct"), "%"),
            ("Front max stroke", detail.get("front_max_stroke_pct"), "%"),
            ("Rear max stroke", detail.get("rear_max_stroke_pct"), "%"),
            ("Front RMS vel", detail.get("front_rms_velocity"), detail.get("velocity_units", "")),
            ("Rear RMS vel", detail.get("rear_rms_velocity"), detail.get("velocity_units", "")),
            ("Front peak compression", detail.get("front_peak_compression_velocity"), detail.get("velocity_units", "")),
            ("Rear peak compression", detail.get("rear_peak_compression_velocity"), detail.get("velocity_units", "")),
            ("Front peak rebound", detail.get("front_peak_rebound_velocity"), detail.get("velocity_units", "")),
            ("Rear peak rebound", detail.get("rear_peak_rebound_velocity"), detail.get("velocity_units", "")),
            ("Front time above 70", detail.get("front_time_above_70_pct"), "%"),
            ("Rear time above 70", detail.get("rear_time_above_70_pct"), "%"),
            ("Front time above 85", detail.get("front_time_above_85_pct"), "%"),
            ("Rear time above 85", detail.get("rear_time_above_85_pct"), "%"),
            ("Front time above 95", detail.get("front_time_above_95_pct"), "%"),
            ("Rear time above 95", detail.get("rear_time_above_95_pct"), "%"),
            ("Mean balance", detail.get("mean_balance_pct"), "p.p."),
            ("Front share", detail.get("mean_front_share_pct"), "%"),
            ("Repeated-hit active", detail.get("repeated_hit_active_duration_s"), "s"),
            ("Front near-bottom", detail.get("front_near_full_event_count"), "events"),
            ("Rear near-bottom", detail.get("rear_near_full_event_count"), "events"),
            ("Front bottom-out", detail.get("front_bottom_out_event_count"), "events"),
            ("Rear bottom-out", detail.get("rear_bottom_out_event_count"), "events"),
        ]

    def _set_detail_heatmaps(self, heatmaps: dict[str, dict[str, Any]]) -> None:
        self.front_travel_heatmap.set_heatmap("Speed vs front travel", heatmaps["front_travel"])
        self.rear_travel_heatmap.set_heatmap("Speed vs rear travel", heatmaps["rear_travel"])
        self.front_velocity_heatmap.set_heatmap("Speed vs front velocity", heatmaps["front_velocity"])
        self.rear_velocity_heatmap.set_heatmap("Speed vs rear velocity", heatmaps["rear_velocity"])

    def _update_detail_views(self) -> None:
        if self._analysis is None:
            return
        state_row = self._selected_state_row()
        if state_row is None:
            state_detail = {"label": "Selected state", "has_data": False}
            state_mask = np.zeros(self._analysis["context_df"].height, dtype=bool)
            self.selected_segment_region.hide()
        else:
            state_mask = self._state_selection_mask(state_row)
            state_detail = summarize_breakdown_selection(
                self._analysis["context_df"],
                self._analysis["meta"],
                state_mask,
                str(state_row["label"]),
            )
            if state_detail.get("has_data"):
                self._set_selected_segment_region(
                    float(state_detail["start_time_s"]),
                    float(state_detail["end_time_s"]),
                    "state",
                )

        manual_mask = self._manual_selection_mask()
        manual_detail = summarize_breakdown_selection(
            self._analysis["context_df"],
            self._analysis["meta"],
            manual_mask,
            "Manual window",
        )

        self.auto_label.setText(self._detail_text(state_detail).replace("Selected state", "Selected state"))
        self.manual_label.setText(self._detail_text(manual_detail))

        use_state = self.detail_source_combo.currentText() == "Selected state" and state_detail.get("has_data")
        detail = state_detail if use_state else manual_detail
        detail_rows = self._detail_rows(detail)
        self.comparison_table.setRowCount(len(detail_rows))
        for row_index, (metric, value, units) in enumerate(detail_rows):
            values = [
                metric,
                format_metric_value(value),
                units,
            ]
            for column, value in enumerate(values):
                self.comparison_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(value)))
        self.comparison_table.resizeColumnsToContents()

        detail_label = detail.get("label")
        detail_mask = state_mask if use_state else manual_mask
        self.detail_label.setText(f"Detail source: {detail_label}")
        self._set_detail_heatmaps(
            compute_breakdown_heatmaps(
                self._analysis["context_df"],
                self._analysis["meta"],
                detail_mask,
            )
        )

    def _on_segment_selection_changed(self) -> None:
        selected_rows = self.segment_table.selectionModel().selectedRows(0)
        if not selected_rows:
            self._selected_segment_id = None
            return
        item = self.segment_table.item(selected_rows[0].row(), 0)
        if item is None:
            self._selected_segment_id = None
            return
        self._selected_segment_id = str(item.data(QtCore.Qt.ItemDataRole.UserRole))
        row = self._selected_segment_row()
        if row is not None and row.get("start_time_s") is not None and row.get("end_time_s") is not None:
            self._set_selected_segment_region(
                float(row["start_time_s"]),
                float(row["end_time_s"]),
                str(row.get("event_type", row.get("type", "event"))),
            )

    def _on_state_selection_changed(self) -> None:
        selected_rows = self.group_table.selectionModel().selectedRows(0)
        if not selected_rows:
            self._selected_state_id = None
            self._update_detail_views()
            return
        item = self.group_table.item(selected_rows[0].row(), 0)
        if item is None:
            self._selected_state_id = None
            self._update_detail_views()
            return
        self._selected_state_id = str(item.data(QtCore.Qt.ItemDataRole.UserRole))
        self._update_detail_views()

    def _on_manual_region_changed(self) -> None:
        self._update_detail_views()

    def clear(self) -> None:
        self._analysis = None
        self._selected_state_id = None
        self._selected_segment_id = None
        self.front_timeline.setData([], [])
        self.rear_timeline.setData([], [])
        self.speed_timeline.setData([], [])
        self.finding_table.setRowCount(0)
        self.group_table.setRowCount(0)
        self.segment_table.setRowCount(0)
        self.comparison_table.setRowCount(0)
        self.auto_label.setText("Selected state: none")
        self.manual_label.setText("Manual window: none")
        self.detail_label.setText("Detail source: Selected state")
        self.front_travel_heatmap.clear()
        self.rear_travel_heatmap.clear()
        self.front_velocity_heatmap.clear()
        self.rear_velocity_heatmap.clear()
        self.warning_label.hide()
        self.selected_segment_region.hide()


class BreakdownSummaryWidget(QtWidgets.QWidget):
    settings_changed = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.warning_label = QtWidgets.QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #8a5a00; background: #fff4ce; padding: 6px;")
        self.warning_label.hide()
        layout.addWidget(self.warning_label)

        self.summary_label = QtWidgets.QLabel(
            "Time is summarized over five wheel-speed bands derived from session percentiles."
        )
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.band_table = QtWidgets.QTableWidget(0, 6)
        self.band_table.setHorizontalHeaderLabels(
            ["Band", "Percentiles", "Speed range [km/h]", "Time [s]", "Time [%]", "Mean speed [km/h]"]
        )
        self.band_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.band_table, 1)

    def load_settings(self, config: dict[str, Any]) -> None:
        _ = config

    def current_settings(self) -> dict[str, Any]:
        return copy.deepcopy(DEFAULT_BREAKDOWN_CONFIG)

    def set_breakdown_analysis(self, analysis: dict[str, Any]) -> None:
        meta = analysis["meta"]
        rows = analysis.get("band_rows", [])
        self.band_table.setRowCount(len(rows))
        self.band_table.setEnabled(bool(meta["wheel_speed"].get("usable", False)))
        for row_index, row in enumerate(rows):
            values = [
                str(row.get("band_index", "")),
                str(row.get("percentile_label", "")),
                f"{format_metric_value(row.get('speed_min_kph'))} - {format_metric_value(row.get('speed_max_kph'))}",
                format_metric_value(row.get("occupancy_s")),
                format_metric_value(row.get("occupancy_pct")),
                format_metric_value(row.get("mean_speed_kph")),
            ]
            for column, value in enumerate(values):
                self.band_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(value)))
        self.band_table.resizeColumnsToContents()

        warning = meta["wheel_speed"].get("warning")
        if warning:
            self.warning_label.setText(str(warning))
            self.warning_label.show()
        else:
            self.warning_label.hide()

    def clear(self) -> None:
        self.band_table.setRowCount(0)
        self.warning_label.hide()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Suspension Telemetry Post-Processing Workbench")
        self.resize(1600, 980)

        self.thread_pool = QtCore.QThreadPool(self)
        self.current_bundle: SessionBundle | None = None
        self.compare_bundles: list[SessionBundle] = []
        self.current_occupancy: dict[str, Any] | None = None
        self._loading_controls = False
        self._loading_database = False
        self._active_tasks: list[BackgroundTask] = []
        self._preview_derived_df = None
        self.session_database_entries: list[SessionDatabaseEntry] = []
        self._database_row_entries: dict[int, SessionDatabaseEntry] = {}
        self._database_group_child_rows: dict[tuple[str, str], list[int]] = {}
        self._collapsed_database_groups: set[tuple[str, str]] = set()
        self._breakdown_refresh_timer = QtCore.QTimer(self)
        self._breakdown_refresh_timer.setSingleShot(True)
        self._breakdown_refresh_timer.setInterval(100)
        self._breakdown_refresh_timer.timeout.connect(self._refresh_breakdown_tab)

        self._build_ui()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, 1)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([420, 920, 320])

        self.status_label = QtWidgets.QLabel("Ready")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximumWidth(180)
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self._set_metadata_controls_enabled(False)
        self.apply_all_button.setEnabled(False)
        self.refresh_session_database()

    def _build_left_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        open_bin_button = QtWidgets.QPushButton("Open BIN(s)...")
        open_bin_button.clicked.connect(self.open_bin_dialog)
        layout.addWidget(open_bin_button)

        open_export_button = QtWidgets.QPushButton("Load Export Folder...")
        open_export_button.clicked.connect(self.open_export_dialog)
        layout.addWidget(open_export_button)

        self.show_export_button = QtWidgets.QPushButton("Show Current Export Folder")
        self.show_export_button.setEnabled(False)
        self.show_export_button.clicked.connect(self.show_current_export_folder)
        layout.addWidget(self.show_export_button)

        refresh_database_button = QtWidgets.QPushButton("Refresh Data Base")
        refresh_database_button.clicked.connect(self.refresh_session_database)
        layout.addWidget(refresh_database_button)

        self.compare_button = QtWidgets.QPushButton("Compare Selected")
        self.compare_button.setEnabled(False)
        self.compare_button.clicked.connect(self.load_selected_comparison)
        layout.addWidget(self.compare_button)

        self.delete_bin_button = QtWidgets.QPushButton("Delete Selected BIN...")
        self.delete_bin_button.setEnabled(False)
        self.delete_bin_button.clicked.connect(self.delete_selected_bin_files)
        layout.addWidget(self.delete_bin_button)

        self.clear_compare_button = QtWidgets.QPushButton("Clear Compare")
        self.clear_compare_button.setEnabled(False)
        self.clear_compare_button.clicked.connect(self.clear_compare)
        layout.addWidget(self.clear_compare_button)

        layout.addSpacing(12)
        self.current_session_label = QtWidgets.QLabel("Currently open: -")
        self.current_session_label.setWordWrap(True)
        layout.addWidget(self.current_session_label)

        layout.addSpacing(12)
        self.database_label = QtWidgets.QLabel("Data base")
        layout.addWidget(self.database_label)
        self.database_table = QtWidgets.QTableWidget(0, 3)
        self.database_table.setHorizontalHeaderLabels(["BIN / Group", "Set", "Comment"])
        self.database_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.database_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.database_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.database_table.verticalHeader().setVisible(False)
        self.database_table.horizontalHeader().setStretchLastSection(True)
        self.database_table.itemSelectionChanged.connect(self.on_database_selection_changed)
        self.database_table.itemClicked.connect(self.on_database_item_clicked)
        layout.addWidget(self.database_table, 1)
        return panel

    def _build_center_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self.session_summary_text = QtWidgets.QPlainTextEdit()
        self.session_summary_text.setReadOnly(True)
        self.status_table = QtWidgets.QTableWidget(0, 5)
        self.status_table.setHorizontalHeaderLabels(["t [s]", "Code", "Name", "Flags", "Values"])
        self.status_table.horizontalHeader().setStretchLastSection(True)
        session_tab = QtWidgets.QWidget()
        session_layout = QtWidgets.QVBoxLayout(session_tab)
        metadata_group = QtWidgets.QGroupBox("Session metadata")
        metadata_form = QtWidgets.QFormLayout(metadata_group)
        self.metadata_date_edit = QtWidgets.QLineEdit()
        self.metadata_track_edit = QtWidgets.QLineEdit()
        self.metadata_set_edit = QtWidgets.QLineEdit()
        self.metadata_comment_edit = QtWidgets.QLineEdit()
        self.save_metadata_button = QtWidgets.QPushButton("Save Session Metadata")
        self.save_metadata_button.clicked.connect(self.save_metadata_for_current_session)
        metadata_form.addRow("Date", self.metadata_date_edit)
        metadata_form.addRow("Track", self.metadata_track_edit)
        metadata_form.addRow("Set", self.metadata_set_edit)
        metadata_form.addRow("Comment", self.metadata_comment_edit)
        metadata_form.addRow("", self.save_metadata_button)
        session_layout.addWidget(metadata_group)
        session_layout.addWidget(self.session_summary_text, 1)
        session_layout.addWidget(self.status_table, 1)
        self.tabs.addTab(session_tab, "Session")

        self.signals_widget = SignalsPlotWidget()
        self.balance_widget = BlankAnalysisWidget()

        self.wheel_widget = WheelPlotWidget()

        self.imu_widget = ImuPlotWidget()

        self.histograms_widget = HistogramsWidget()
        self.tabs.addTab(self.histograms_widget, "Histograms")

        self.occupancy_widget = OccupancyPlotWidget()
        self.tabs.addTab(self.occupancy_widget, "Position/Velocity Heatmap")

        self.tabs.addTab(self.balance_widget, "Balance")
        self.braking_widget = BrakingWidget()
        self.braking_widget.settings_changed.connect(self.on_braking_settings_changed)
        self.tabs.addTab(self.braking_widget, "Braking")
        self.breakdown_widget = BlankAnalysisWidget()
        self.tabs.addTab(self.breakdown_widget, "Breakdown")
        self.tabs.addTab(self.wheel_widget, "Speed")

        self.metrics_table = build_metrics_table_widget()
        self.hardware_metrics_table = build_metrics_table_widget()
        metrics_tab = QtWidgets.QWidget()
        metrics_layout = QtWidgets.QVBoxLayout(metrics_tab)

        riding_group = QtWidgets.QGroupBox("Riding metrics")
        riding_layout = QtWidgets.QVBoxLayout(riding_group)
        riding_layout.addWidget(self.metrics_table)
        metrics_layout.addWidget(riding_group, 2)

        hardware_group = QtWidgets.QGroupBox("Hardware metrics")
        hardware_layout = QtWidgets.QVBoxLayout(hardware_group)
        hardware_layout.addWidget(self.hardware_metrics_table)
        metrics_layout.addWidget(hardware_group, 1)
        self.tabs.addTab(metrics_tab, "Metrics")

        self.compare_widget = CompareWidget()
        self.tabs.addTab(self.compare_widget, "Compare")
        return panel

    def _build_right_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        self.channel_controls: dict[str, dict[str, QtWidgets.QWidget]] = {}
        for channel in ("front", "rear"):
            group = QtWidgets.QGroupBox(f"{channel.capitalize()} calibration")
            form = QtWidgets.QFormLayout(group)

            invert = QtWidgets.QCheckBox("Invert sign")
            manual_reference = QtWidgets.QLineEdit()
            manual_reference.setPlaceholderText("raw ADC count")
            manual_reference.setToolTip(
                "Manual raw ADC anchor used as the travel-zero reference. Leave empty to show calibrated raw counts unanchored."
            )
            velocity_filter_window = QtWidgets.QSpinBox()
            velocity_filter_window.setRange(1, 999)
            velocity_filter_window.setSingleStep(2)
            velocity_filter_window.setToolTip(
                "Odd-sized moving-average window used before differentiating travel into velocity."
            )
            calibration_summary = QtWidgets.QLabel()
            calibration_summary.setWordWrap(True)

            form.addRow("", invert)
            form.addRow("Manual anchor", manual_reference)
            form.addRow("Velocity window", velocity_filter_window)
            form.addRow("Stroke model", calibration_summary)

            self.channel_controls[channel] = {
                "invert": invert,
                "manual_reference": manual_reference,
                "velocity_filter_window": velocity_filter_window,
                "calibration_summary": calibration_summary,
            }
            layout.addWidget(group)

        plot_group = QtWidgets.QGroupBox("Plot controls")
        plot_form = QtWidgets.QFormLayout(plot_group)
        self.selected_channel_combo = QtWidgets.QComboBox()
        self.selected_channel_combo.addItems(["front", "rear"])
        self.selected_channel_combo.currentTextChanged.connect(self.on_view_settings_changed)
        self.travel_bins_spin = QtWidgets.QSpinBox()
        self.travel_bins_spin.setRange(10, 400)
        self.travel_bins_spin.valueChanged.connect(self.on_view_settings_changed)
        self.velocity_bins_spin = QtWidgets.QSpinBox()
        self.velocity_bins_spin.setRange(10, 400)
        self.velocity_bins_spin.valueChanged.connect(self.on_view_settings_changed)
        self.color_scale_combo = QtWidgets.QComboBox()
        self.color_scale_combo.addItems(["sqrt", "log", "linear"])
        self.color_scale_combo.currentTextChanged.connect(self.on_view_settings_changed)
        self.velocity_axis_combo = QtWidgets.QComboBox()
        self.velocity_axis_combo.addItem("Linear", "linear")
        self.velocity_axis_combo.addItem("Signed log", "signed_log")
        self.velocity_axis_combo.currentTextChanged.connect(self.on_view_settings_changed)
        plot_form.addRow("Save/export channel", self.selected_channel_combo)
        plot_form.addRow("Travel bins", self.travel_bins_spin)
        plot_form.addRow("Velocity bins", self.velocity_bins_spin)
        plot_form.addRow("Color scale", self.color_scale_combo)
        plot_form.addRow("Velocity axis", self.velocity_axis_combo)
        layout.addWidget(plot_group)

        self.apply_button = QtWidgets.QPushButton("Apply Calibration")
        self.apply_button.clicked.connect(self.apply_calibration)
        layout.addWidget(self.apply_button)

        self.apply_all_button = QtWidgets.QPushButton("Apply Calibration to All")
        self.apply_all_button.setEnabled(False)
        self.apply_all_button.clicked.connect(self.apply_calibration_to_all)
        layout.addWidget(self.apply_all_button)

        self.refresh_button = QtWidgets.QPushButton("Refresh Views")
        self.refresh_button.clicked.connect(self.refresh_views_from_controls)
        layout.addWidget(self.refresh_button)

        self.save_png_button = QtWidgets.QPushButton("Save Heatmap PNG...")
        self.save_png_button.clicked.connect(self.save_occupancy_png)
        layout.addWidget(self.save_png_button)

        self.save_grid_button = QtWidgets.QPushButton("Save Heatmap Grid CSV...")
        self.save_grid_button.clicked.connect(self.save_occupancy_grid)
        layout.addWidget(self.save_grid_button)

        layout.addStretch(1)
        return panel

    def set_busy(self, busy: bool, message: str) -> None:
        self.status_label.setText(message)
        if busy:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)

    def _current_velocity_axis_mode(self) -> str:
        return normalize_velocity_axis_mode(str(self.velocity_axis_combo.currentData() or "linear"))

    def run_task(self, message: str, fn: Callable[[], object], callback: Callable[[object], None]) -> None:
        self.set_busy(True, message)
        task = BackgroundTask(fn)
        task.setAutoDelete(False)
        self._active_tasks.append(task)
        task.signals.finished.connect(lambda result, task=task: self._handle_task_finished(message, result, callback, task))
        task.signals.error.connect(lambda detail, task=task: self._handle_task_error(detail, task))
        self.thread_pool.start(task)

    def _release_task(self, task: BackgroundTask) -> None:
        try:
            self._active_tasks.remove(task)
        except ValueError:
            pass

    def _handle_task_finished(
        self,
        message: str,
        result: object,
        callback: Callable[[object], None],
        task: BackgroundTask,
    ) -> None:
        self._release_task(task)
        self.set_busy(False, f"{message} complete")
        callback(result)

    def _handle_task_error(self, detail: str, task: BackgroundTask) -> None:
        self._release_task(task)
        self.set_busy(False, "Task failed")
        QtWidgets.QMessageBox.critical(self, "Operation failed", detail)

    def open_bin_dialog(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open BIN session(s)",
            "",
            "BIN files (*.BIN *.bin);;All files (*.*)",
        )
        if not paths:
            return
        bin_paths = [Path(path) for path in paths]
        self.run_task(
            self._bin_import_message(bin_paths),
            lambda: self._open_bin_sessions(bin_paths),
            self._on_bin_import_loaded,
        )

    @staticmethod
    def _bin_import_message(bin_paths: list[Path]) -> str:
        if len(bin_paths) == 1:
            return f"Opening {bin_paths[0].name}"
        return f"Opening {len(bin_paths)} BIN sessions"

    @staticmethod
    def _session_start_epoch(bundle: SessionBundle) -> int:
        start_epoch = bundle.summary.get("header", {}).get("start_epoch")
        try:
            return int(start_epoch) if start_epoch not in (None, "") else 0
        except (TypeError, ValueError):
            return 0

    def _open_bin_sessions(self, bin_paths: list[Path]) -> list[SessionBundle]:
        bundles = [open_bin_session(bin_path) for bin_path in bin_paths]
        updated_metadata = auto_assign_set_labels()
        for bundle in bundles:
            metadata = updated_metadata.get(bundle.export_dir.resolve())
            if metadata is not None:
                bundle.session_metadata = metadata
        bundles.sort(key=lambda bundle: (self._session_start_epoch(bundle), bundle.export_dir.name))
        return bundles

    def _on_bin_import_loaded(self, result: object) -> None:
        bundles = list(result) if isinstance(result, list) else []
        if not bundles or not all(isinstance(bundle, SessionBundle) for bundle in bundles):
            QtWidgets.QMessageBox.warning(
                self,
                "BIN import failed",
                "The BIN import task did not return any session bundles.",
            )
            return
        self._on_bundle_loaded(bundles[-1])
        if len(bundles) > 1:
            self.status_label.setText(f"Imported {len(bundles)} BIN sessions")

    def open_export_dialog(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open export folder", "")
        if not path:
            return
        export_dir = Path(path)
        self.run_task(
            f"Opening {export_dir.name}",
            lambda: open_export_session(export_dir),
            self._on_bundle_loaded,
        )

    def _on_bundle_loaded(self, result: object) -> None:
        bundle = result
        assert isinstance(bundle, SessionBundle)
        self.current_bundle = bundle
        self._sync_compare_bundle(bundle)
        self._preview_derived_df = None
        self.show_export_button.setEnabled(True)
        self.apply_all_button.setEnabled(True)
        self.populate_controls_from_bundle(bundle)
        self.populate_metadata_from_bundle(bundle)
        self.refresh_session_database()
        self.refresh_views()

    def populate_controls_from_bundle(self, bundle: SessionBundle) -> None:
        self._loading_controls = True
        try:
            for channel in ("front", "rear"):
                config = bundle.session_config[channel]
                controls = self.channel_controls[channel]
                controls["invert"].setChecked(bool(config.get("invert", False)))
                manual_reference = config.get("manual_reference_count")
                controls["manual_reference"].setText(
                    "" if manual_reference in (None, "") else f"{float(manual_reference):g}"
                )
                controls["velocity_filter_window"].setValue(int(config.get("velocity_filter_window", 3)))
                controls["calibration_summary"].setText(self._describe_channel_calibration(channel, config))

            plot_defaults = bundle.session_config.get("plot_defaults", {})
            self.selected_channel_combo.setCurrentText(str(plot_defaults.get("selected_channel", "front")))
            self.travel_bins_spin.setValue(int(plot_defaults.get("occupancy_travel_bins", 400)))
            self.velocity_bins_spin.setValue(int(plot_defaults.get("occupancy_velocity_bins", 400)))
            self.color_scale_combo.setCurrentText(str(plot_defaults.get("occupancy_color_scale", "sqrt")))
            velocity_axis_index = self.velocity_axis_combo.findData(
                normalize_velocity_axis_mode(str(plot_defaults.get("velocity_axis_mode", "linear")))
            )
            self.velocity_axis_combo.setCurrentIndex(max(velocity_axis_index, 0))
            self.braking_widget.load_settings(bundle.session_config.get("braking", copy.deepcopy(DEFAULT_BRAKING_CONFIG)))
            self.breakdown_widget.load_settings(bundle.session_config.get("breakdown", copy.deepcopy(DEFAULT_BREAKDOWN_CONFIG)))
        finally:
            self._loading_controls = False

    def _set_metadata_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.metadata_date_edit,
            self.metadata_track_edit,
            self.metadata_set_edit,
            self.metadata_comment_edit,
            self.save_metadata_button,
        ):
            widget.setEnabled(enabled)

    def populate_metadata_from_bundle(self, bundle: SessionBundle) -> None:
        metadata = bundle.session_metadata or {}
        self.metadata_date_edit.setText(str(metadata.get("date", "")))
        self.metadata_track_edit.setText(str(metadata.get("track", "")))
        self.metadata_set_edit.setText(str(metadata.get("set_label", "")))
        self.metadata_comment_edit.setText(str(metadata.get("comment", "")))
        self._set_metadata_controls_enabled(True)

    def _current_metadata(self) -> dict[str, Any]:
        if self.current_bundle is None:
            raise RuntimeError("no session loaded")
        metadata = copy.deepcopy(self.current_bundle.session_metadata)
        metadata["date"] = self.metadata_date_edit.text().strip()
        metadata["track"] = self.metadata_track_edit.text().strip()
        metadata["set_label"] = self.metadata_set_edit.text().strip()
        metadata["set_label_auto"] = False
        metadata["comment"] = self.metadata_comment_edit.text().strip()
        return metadata

    def save_metadata_for_current_session(self) -> None:
        if self.current_bundle is None:
            return
        metadata = self._current_metadata()
        save_session_metadata(self.current_bundle.export_dir, metadata)
        self.current_bundle.session_metadata = metadata
        self._sync_compare_bundle(self.current_bundle)
        self._update_session_overview(self.current_bundle)
        self.refresh_session_database()
        self._refresh_compare_tab()
        self.status_label.setText(f"Saved metadata for {self.current_bundle.export_dir.name}")

    def _describe_channel_calibration(self, channel: str, config: dict[str, Any]) -> str:
        full_scale_mm = config.get("full_scale_mm")
        sensor_full_scale_mm = config.get("sensor_full_scale_mm")
        manual_reference = config.get("manual_reference_count")
        if manual_reference not in (None, ""):
            reference_text = f"Manual raw anchor {format_metric_value(float(manual_reference))} counts."
        else:
            reference_text = "No manual raw anchor is set; calibrated raw counts are shown unanchored."
        if full_scale_mm not in (None, "") and sensor_full_scale_mm not in (None, ""):
            return (
                f"{reference_text} Stroke is scaled to "
                f"{format_metric_value(full_scale_mm)} mm full travel using a "
                f"{format_metric_value(sensor_full_scale_mm)} mm sensor stroke."
            )
        return f"{reference_text} Stroke is scaled to the used range seen in this session."

    def gather_session_config(self) -> dict[str, Any]:
        if self.current_bundle is None:
            raise RuntimeError("no session loaded")
        config = copy.deepcopy(self.current_bundle.session_config)
        for channel in ("front", "rear"):
            controls = self.channel_controls[channel]
            channel_config = copy.deepcopy(config.get(channel, {}))
            channel_config["invert"] = controls["invert"].isChecked()
            channel_config["travel_reference"] = "manual"
            channel_config["manual_reference_count"] = parse_optional_float(
                controls["manual_reference"].text(),
                f"{channel} manual anchor",
            )
            for obsolete_key in ("reference_method", "reference_percentile", "reference_window_samples"):
                channel_config.pop(obsolete_key, None)
            channel_config["velocity_filter_window"] = controls["velocity_filter_window"].value()
            config[channel] = channel_config
        config["plot_defaults"] = {
            "selected_channel": self.selected_channel_combo.currentText(),
            "occupancy_travel_bins": self.travel_bins_spin.value(),
            "occupancy_velocity_bins": self.velocity_bins_spin.value(),
            "occupancy_color_scale": self.color_scale_combo.currentText(),
            "velocity_axis_mode": self._current_velocity_axis_mode(),
        }
        config["braking"] = self.braking_widget.current_settings()
        return config

    def on_view_settings_changed(self) -> None:
        if self._loading_controls or self.current_bundle is None:
            return
        self.current_bundle.session_config["plot_defaults"] = {
            "selected_channel": self.selected_channel_combo.currentText(),
            "occupancy_travel_bins": self.travel_bins_spin.value(),
            "occupancy_velocity_bins": self.velocity_bins_spin.value(),
            "occupancy_color_scale": self.color_scale_combo.currentText(),
            "velocity_axis_mode": self._current_velocity_axis_mode(),
        }
        save_session_config(self.current_bundle.export_dir, self.current_bundle.session_config)
        self._refresh_plot_views()

    def on_braking_settings_changed(self) -> None:
        if self._loading_controls or self.current_bundle is None:
            return
        self.current_bundle.session_config["braking"] = self.braking_widget.current_settings()
        save_session_config(self.current_bundle.export_dir, self.current_bundle.session_config)
        self._refresh_braking_tab()

    def on_breakdown_settings_changed(self) -> None:
        if self.current_bundle is None:
            return
        try:
            breakdown_config = self.breakdown_widget.current_settings()
        except ValueError as exc:
            self.status_label.setText("Invalid breakdown setting")
            QtWidgets.QMessageBox.warning(self, "Invalid breakdown value", str(exc))
            return
        self.current_bundle.session_config["breakdown"] = breakdown_config
        save_session_config(self.current_bundle.export_dir, self.current_bundle.session_config)
        self._breakdown_refresh_timer.start()

    def apply_calibration(self) -> None:
        if self.current_bundle is None:
            return
        try:
            config = self.gather_session_config()
        except ValueError as exc:
            self.status_label.setText("Invalid calibration input")
            QtWidgets.QMessageBox.warning(self, "Invalid calibration value", str(exc))
            return
        export_dir = self.current_bundle.export_dir
        self.run_task(
            "Rebuilding derived analysis",
            lambda: rebuild_session_analysis(export_dir, config),
            self._on_bundle_loaded,
        )

    def apply_calibration_to_all(self) -> None:
        if self.current_bundle is None:
            return
        try:
            template_config = self.gather_session_config()
        except ValueError as exc:
            self.status_label.setText("Invalid calibration input")
            QtWidgets.QMessageBox.warning(self, "Invalid calibration value", str(exc))
            return

        entries = list(self.session_database_entries)
        if not entries:
            QtWidgets.QMessageBox.information(
                self,
                "Apply Calibration to All",
                "No processed sets are available in the Data base.",
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Apply calibration to all?",
            "Apply the current front and rear calibration controls to "
            f"all {len(entries)} processed set(s) in the Data base?\n\n"
            "Session metadata, source paths, plot settings, and comments will be left unchanged.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        self.run_task(
            f"Applying calibration to {len(entries)} sets",
            lambda: self._rebuild_all_sessions_with_calibration(template_config, entries),
            self._on_all_calibrations_applied,
        )

    def _rebuild_all_sessions_with_calibration(
        self,
        template_config: dict[str, Any],
        entries: list[SessionDatabaseEntry],
    ) -> dict[str, object]:
        bundles: list[SessionBundle] = []
        errors: list[str] = []
        for entry in entries:
            try:
                bundle = open_export_session(entry.export_dir)
                config = apply_calibration_template(bundle.session_config, template_config)
                bundles.append(rebuild_session_analysis(entry.export_dir, config))
            except Exception as exc:  # noqa: BLE001 - surface per-session failures to the GUI.
                errors.append(f"{entry.session_id}: {exc}")
        return {"bundles": bundles, "errors": errors}

    def _on_all_calibrations_applied(self, result: object) -> None:
        if not isinstance(result, dict):
            QtWidgets.QMessageBox.warning(
                self,
                "Apply Calibration to All failed",
                "The calibration task returned an unexpected result.",
            )
            return

        bundles = [bundle for bundle in result.get("bundles", []) if isinstance(bundle, SessionBundle)]
        errors = [str(error) for error in result.get("errors", [])]
        current_export_dir = None if self.current_bundle is None else self.current_bundle.export_dir.resolve()
        replacement_bundle = None
        for bundle in bundles:
            self._sync_compare_bundle(bundle)
            if current_export_dir is not None and bundle.export_dir.resolve() == current_export_dir:
                replacement_bundle = bundle

        if replacement_bundle is not None:
            self.current_bundle = replacement_bundle
            self._preview_derived_df = None
            self.populate_controls_from_bundle(replacement_bundle)
            self.populate_metadata_from_bundle(replacement_bundle)
            self.refresh_views()

        self.refresh_session_database()
        if errors:
            QtWidgets.QMessageBox.warning(
                self,
                "Apply Calibration to All incomplete",
                "Some processed sets could not be rebuilt:\n\n" + "\n".join(errors[:8]),
            )
        self.status_label.setText(f"Applied calibration to {len(bundles)} processed set(s)")

    def refresh_views_from_controls(self) -> None:
        if self.current_bundle is None:
            self.refresh_views()
            return
        try:
            config = self.gather_session_config()
        except ValueError as exc:
            self.status_label.setText("Invalid calibration input")
            QtWidgets.QMessageBox.warning(self, "Invalid calibration value", str(exc))
            return

        preview_df, _ = build_derived_analog(
            export_dir=self.current_bundle.export_dir,
            config=copy.deepcopy(config),
            analog_df=self.current_bundle.analog_df,
            force=True,
            persist=False,
        )
        self._preview_derived_df = preview_df
        self.status_label.setText("Views refreshed from current controls (not saved)")
        self._refresh_analysis_views()

    def show_current_export_folder(self) -> None:
        if self.current_bundle is None:
            return
        export_dir = self.current_bundle.export_dir
        if open_path_in_file_manager(export_dir):
            self.status_label.setText(f"Opened {export_dir}")
            return
        QtWidgets.QMessageBox.warning(
            self,
            "Open export folder failed",
            f"Could not open {export_dir}",
        )

    def selected_channel(self) -> str:
        return self.selected_channel_combo.currentText()

    def _active_derived_df(self):
        if self._preview_derived_df is not None:
            return self._preview_derived_df
        if self.current_bundle is None:
            return None
        return self.current_bundle.derived_df

    def _sync_compare_bundle(self, updated_bundle: SessionBundle) -> None:
        if not self.compare_bundles:
            return
        updated_export_dir = updated_bundle.export_dir.resolve()
        for index, bundle in enumerate(self.compare_bundles):
            if bundle.export_dir.resolve() == updated_export_dir:
                self.compare_bundles[index] = updated_bundle

    @staticmethod
    def _database_group_key(entry: SessionDatabaseEntry) -> tuple[str, str]:
        metadata = entry.metadata
        date_key = str(metadata.get("date", "")).strip() or "undated"
        track_key = str(metadata.get("track", "")).strip()
        return date_key, track_key

    @staticmethod
    def _database_group_label(group_key: tuple[str, str], set_count: int) -> str:
        date_key, track_key = group_key
        track_text = track_key or "No track"
        set_suffix = "set" if set_count == 1 else "sets"
        return f"{date_key} | {track_text} ({set_count} {set_suffix})"

    def refresh_session_database(self) -> None:
        selected_export_dirs: set[Path] = set()
        selection_model = self.database_table.selectionModel()
        if selection_model is not None:
            for index in selection_model.selectedRows():
                row_index = index.row()
                entry = self._database_row_entries.get(row_index)
                if entry is not None:
                    selected_export_dirs.add(entry.export_dir.resolve())
        if not selected_export_dirs and self.current_bundle is not None:
            selected_export_dirs.add(self.current_bundle.export_dir.resolve())

        entries = scan_session_database()
        self.session_database_entries = entries
        groups: dict[tuple[str, str], list[SessionDatabaseEntry]] = {}
        group_order: list[tuple[str, str]] = []
        date_keys: set[str] = set()
        for entry in entries:
            group_key = self._database_group_key(entry)
            if group_key not in groups:
                groups[group_key] = []
                group_order.append(group_key)
            groups[group_key].append(entry)
            date_keys.add(group_key[0])
        group_index_by_key = {group_key: index for index, group_key in enumerate(group_order)}
        day_count = len(date_keys)
        day_suffix = "day" if day_count == 1 else "days"
        group_count = len(group_order)
        group_suffix = "group" if group_count == 1 else "groups"
        self.database_label.setText(
            f"Data base ({len(entries)} processed sets, {day_count} {day_suffix}, {group_count} {group_suffix})"
        )

        self._loading_database = True
        try:
            self._database_row_entries = {}
            self._database_group_child_rows = {}
            self.database_table.setRowCount(len(entries) + len(group_order))
            group_colors = [
                QtGui.QColor("#f6f8fb"),
                QtGui.QColor("#fff8ef"),
                QtGui.QColor("#f4fbf6"),
            ]
            row_index = 0
            for group_key in group_order:
                group_entries = groups[group_key]
                date_key, track_key = group_key
                group_index = group_index_by_key.get(group_key, 0)
                group_color = group_colors[group_index % len(group_colors)]
                collapsed = group_key in self._collapsed_database_groups
                indicator = ">" if collapsed else "v"
                group_values = [
                    f"{indicator} {self._database_group_label(group_key, len(group_entries))}",
                    f"{len(group_entries)} set(s)",
                    "",
                ]
                for column, value in enumerate(group_values):
                    item = QtWidgets.QTableWidgetItem(str(value))
                    item.setBackground(group_color)
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, group_key)
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    self.database_table.setItem(row_index, column, item)
                row_index += 1
                child_rows: list[int] = []
                for entry in group_entries:
                    metadata = entry.metadata
                    child_values = [
                        entry.session_id,
                        metadata.get("set_label", "") or entry.session_id,
                        metadata.get("comment", ""),
                    ]
                    self._database_row_entries[row_index] = entry
                    child_rows.append(row_index)
                    for column, value in enumerate(child_values):
                        item = QtWidgets.QTableWidgetItem(str(value))
                        item.setBackground(group_color)
                        self.database_table.setItem(row_index, column, item)
                    self.database_table.setRowHidden(row_index, collapsed)
                    row_index += 1
                self._database_group_child_rows[group_key] = child_rows
            self.database_table.clearSelection()
            selection_model = self.database_table.selectionModel()
            if selection_model is not None:
                select_flags = (
                    QtCore.QItemSelectionModel.SelectionFlag.Select
                    | QtCore.QItemSelectionModel.SelectionFlag.Rows
                )
                for row_index, entry in self._database_row_entries.items():
                    if entry.export_dir.resolve() in selected_export_dirs:
                        selection_model.select(self.database_table.model().index(row_index, 0), select_flags)
            self.database_table.resizeColumnsToContents()
        finally:
            self._loading_database = False
        self._update_database_actions()

    def _selected_database_row_indexes(self) -> list[int]:
        selection_model = self.database_table.selectionModel()
        if selection_model is None:
            return []
        rows = sorted(index.row() for index in selection_model.selectedRows())
        return [row for row in rows if row in self._database_row_entries]

    def _selected_database_entries(self) -> list[SessionDatabaseEntry]:
        return [self._database_row_entries[row] for row in self._selected_database_row_indexes()]

    def _update_database_actions(self) -> None:
        selected_count = len(self._selected_database_row_indexes())
        self.compare_button.setEnabled(COMPARE_MIN_SESSIONS <= selected_count <= COMPARE_MAX_SESSIONS)
        self.delete_bin_button.setEnabled(
            any(self._deletable_bin_path(entry) is not None for entry in self._selected_database_entries())
        )
        self.clear_compare_button.setEnabled(bool(self.compare_bundles))

    def _deletable_bin_path(self, entry: SessionDatabaseEntry) -> Path | None:
        source_path = entry.source_path
        if source_path is None:
            return None
        bin_path = source_path if source_path.is_absolute() else Path.cwd() / source_path
        if bin_path.suffix.lower() != ".bin" or not bin_path.exists() or not bin_path.is_file():
            return None
        return bin_path

    def delete_selected_bin_files(self) -> None:
        entries = self._selected_database_entries()
        candidates: list[Path] = []
        for entry in entries:
            bin_path = self._deletable_bin_path(entry)
            if bin_path is not None and bin_path not in candidates:
                candidates.append(bin_path)

        if not candidates:
            QtWidgets.QMessageBox.information(
                self,
                "Delete BIN",
                "No selected database rows have an existing .BIN source file.",
            )
            self._update_database_actions()
            return

        display_paths = "\n".join(str(path) for path in candidates[:8])
        if len(candidates) > 8:
            display_paths += f"\n... and {len(candidates) - 8} more"
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete selected BIN files?",
            "Delete these .BIN source files?\n\n"
            f"{display_paths}\n\n"
            "Processed export folders will be left in place.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        deleted_count = 0
        failures: list[str] = []
        for bin_path in candidates:
            try:
                bin_path.unlink()
                deleted_count += 1
            except OSError as exc:
                failures.append(f"{bin_path}: {exc}")

        self.refresh_session_database()
        if failures:
            QtWidgets.QMessageBox.warning(
                self,
                "Delete BIN incomplete",
                "Some .BIN files could not be deleted:\n\n" + "\n".join(failures[:8]),
            )
        self.status_label.setText(f"Deleted {deleted_count} .BIN source file(s)")

    def on_database_item_clicked(self, item: QtWidgets.QTableWidgetItem) -> None:
        group_key = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(group_key, tuple):
            return
        if group_key in self._collapsed_database_groups:
            self._collapsed_database_groups.remove(group_key)
            collapsed = False
        else:
            self._collapsed_database_groups.add(group_key)
            collapsed = True

        for row_index in self._database_group_child_rows.get(group_key, []):
            self.database_table.setRowHidden(row_index, collapsed)
        group_row = item.row()
        header_item = self.database_table.item(group_row, 0)
        if header_item is not None:
            indicator = ">" if collapsed else "v"
            header_item.setText(f"{indicator} {self._database_group_label(group_key, len(self._database_group_child_rows.get(group_key, [])))}")
        self.database_table.clearSelection()
        self._update_database_actions()

    def on_database_selection_changed(self) -> None:
        if self._loading_database:
            return
        self._update_database_actions()
        selected_rows = self._selected_database_row_indexes()
        if len(selected_rows) != 1:
            return
        row_index = selected_rows[0]
        entry = self._database_row_entries.get(row_index)
        if entry is None:
            return
        if self.current_bundle is not None and entry.export_dir.resolve() == self.current_bundle.export_dir.resolve():
            return
        self.run_task(
            f"Opening {entry.session_id}",
            lambda export_dir=entry.export_dir: open_export_session(export_dir),
            self._on_bundle_loaded,
        )

    def load_selected_comparison(self) -> None:
        entries = self._selected_database_entries()
        if not (COMPARE_MIN_SESSIONS <= len(entries) <= COMPARE_MAX_SESSIONS):
            QtWidgets.QMessageBox.information(
                self,
                "Compare Selected",
                f"Select {COMPARE_MIN_SESSIONS}-{COMPARE_MAX_SESSIONS} processed sets in Data base to compare.",
            )
            return
        compare_label = ", ".join(entry.session_id for entry in entries)
        self.run_task(
            f"Comparing {compare_label}",
            lambda entries=entries: [open_export_session(entry.export_dir) for entry in entries],
            self._on_compare_bundles_loaded,
        )

    def _on_compare_bundles_loaded(self, result: object) -> None:
        bundles = list(result) if isinstance(result, list) else []
        if (
            not (COMPARE_MIN_SESSIONS <= len(bundles) <= COMPARE_MAX_SESSIONS)
            or not all(isinstance(bundle, SessionBundle) for bundle in bundles)
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Compare load failed",
                f"The compare task must return {COMPARE_MIN_SESSIONS}-{COMPARE_MAX_SESSIONS} session bundles.",
            )
            return
        self.compare_bundles = [bundle for bundle in bundles if isinstance(bundle, SessionBundle)]
        self._refresh_compare_tab()
        compare_index = self.tabs.indexOf(self.compare_widget)
        if compare_index >= 0:
            self.tabs.setCurrentIndex(compare_index)
        self.status_label.setText(
            "Compare loaded: " + ", ".join(bundle_session_id(bundle) for bundle in self.compare_bundles)
        )

    def clear_compare(self) -> None:
        self.compare_bundles = []
        self.compare_widget.clear()
        self._update_database_actions()
        self.status_label.setText("Compare cleared")

    def _refresh_compare_tab(self) -> None:
        if len(self.compare_bundles) < COMPARE_MIN_SESSIONS:
            self.compare_widget.clear()
            self._update_database_actions()
            return
        self.compare_widget.set_comparison(
            self.compare_bundles,
            travel_bins=self.travel_bins_spin.value(),
            velocity_bins=self.velocity_bins_spin.value(),
            velocity_axis_mode=self._current_velocity_axis_mode(),
        )
        self._update_database_actions()

    def _update_session_overview(self, bundle: SessionBundle) -> None:
        self.session_summary_text.setPlainText(build_summary_text(bundle))
        self.current_session_label.setText(f"Currently open: {bundle_session_label(bundle)}")

    def refresh_views(self) -> None:
        if self.current_bundle is None:
            self._breakdown_refresh_timer.stop()
            self.session_summary_text.clear()
            self.current_session_label.setText("Currently open: -")
            self.metadata_date_edit.clear()
            self.metadata_track_edit.clear()
            self.metadata_set_edit.clear()
            self.metadata_comment_edit.clear()
            self._set_metadata_controls_enabled(False)
            self.signals_widget.clear()
            self.balance_widget.clear()
            self.braking_widget.clear()
            self.breakdown_widget.clear()
            self.wheel_widget.clear()
            self.imu_widget.clear()
            self.occupancy_widget.clear()
            self.metrics_table.setRowCount(0)
            self.hardware_metrics_table.setRowCount(0)
            self.show_export_button.setEnabled(False)
            self.apply_all_button.setEnabled(False)
            return

        bundle = self.current_bundle
        self._breakdown_refresh_timer.stop()
        self._update_session_overview(bundle)

        self._populate_status_table(bundle.status_rows)
        self._refresh_analysis_views()
        self._refresh_wheel_tab()
        self._refresh_imu_tab()

    def _refresh_plot_views(self) -> None:
        self._refresh_occupancy_tab()
        self._refresh_histograms_tab()
        self._refresh_compare_tab()

    def _refresh_analysis_views(self) -> None:
        self._refresh_balance_tab()
        self._refresh_braking_tab()
        self._refresh_breakdown_tab()
        self._refresh_occupancy_tab()
        self._refresh_histograms_tab()
        self._refresh_metrics_tab()
        self._refresh_compare_tab()

    def _populate_status_table(self, rows: list[dict[str, str]]) -> None:
        self.status_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row.get("host_time_s", ""),
                row.get("code", ""),
                row.get("name", ""),
                row.get("flags", ""),
                f"{row.get('value0', '')}, {row.get('value1', '')}",
            ]
            for column, value in enumerate(values):
                self.status_table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(value)))
        self.status_table.resizeColumnsToContents()

    def _refresh_signals_tab(self) -> None:
        derived_df = self._active_derived_df()
        if derived_df is None:
            return
        channel = self.selected_channel()
        series = channel_series(derived_df, channel)
        self.signals_widget.set_series(channel, series)

    def _refresh_balance_tab(self) -> None:
        self.balance_widget.clear()

    def _refresh_braking_tab(self) -> None:
        derived_df = self._active_derived_df()
        if self.current_bundle is None or derived_df is None:
            self.braking_widget.clear()
            return
        analysis = build_fork_braking_analysis(
            derived_df=derived_df,
            wheel_df=self.current_bundle.wheel_df,
            config=self.current_bundle.session_config.get("braking", copy.deepcopy(DEFAULT_BRAKING_CONFIG)),
        )
        self.braking_widget.set_braking_analysis(analysis)

    def _refresh_breakdown_tab(self) -> None:
        self.breakdown_widget.clear()

    def _refresh_wheel_tab(self) -> None:
        if self.current_bundle is None:
            return
        self.wheel_widget.set_wheel_data(
            self.current_bundle.wheel_df,
            start_time_s=bundle_session_start_time_s(self.current_bundle),
        )

    def _refresh_imu_tab(self) -> None:
        if self.current_bundle is None:
            return
        self.imu_widget.set_imu_data(self.current_bundle.imu_frame_df)

    def _refresh_occupancy_tab(self) -> None:
        derived_df = self._active_derived_df()
        if derived_df is None:
            return
        front_occupancy = compute_occupancy_grid(
            derived_df=derived_df,
            channel="front",
            travel_bins=self.travel_bins_spin.value(),
            velocity_bins=self.velocity_bins_spin.value(),
            series_mode="stroke_percent",
            color_scale=self.color_scale_combo.currentText(),
            velocity_axis_mode=self._current_velocity_axis_mode(),
        )
        rear_occupancy = compute_occupancy_grid(
            derived_df=derived_df,
            channel="rear",
            travel_bins=self.travel_bins_spin.value(),
            velocity_bins=self.velocity_bins_spin.value(),
            series_mode="stroke_percent",
            color_scale=self.color_scale_combo.currentText(),
            velocity_axis_mode=self._current_velocity_axis_mode(),
        )
        self.current_occupancy = {
            "front": front_occupancy,
            "rear": rear_occupancy,
        }
        self.occupancy_widget.set_occupancies(front_occupancy, rear_occupancy)

    def _refresh_histograms_tab(self) -> None:
        derived_df = self._active_derived_df()
        if derived_df is None:
            return
        front_travel_hist = compute_travel_histogram(
            derived_df,
            "front",
            series_mode="stroke_percent",
            relative_occupancy=True,
        )
        front_velocity_hist = compute_velocity_histogram(
            derived_df,
            "front",
            series_mode="stroke_percent",
            relative_occupancy=True,
            velocity_axis_mode=self._current_velocity_axis_mode(),
        )
        rear_travel_hist = compute_travel_histogram(
            derived_df,
            "rear",
            series_mode="stroke_percent",
            relative_occupancy=True,
        )
        rear_velocity_hist = compute_velocity_histogram(
            derived_df,
            "rear",
            series_mode="stroke_percent",
            relative_occupancy=True,
            velocity_axis_mode=self._current_velocity_axis_mode(),
        )
        self.histograms_widget.set_histograms(
            front_travel_hist,
            front_velocity_hist,
            rear_travel_hist,
            rear_velocity_hist,
        )

    def _refresh_metrics_tab(self) -> None:
        derived_df = self._active_derived_df()
        if self.current_bundle is None or derived_df is None:
            return
        analog_resolution_bits = int(self.current_bundle.summary["header"]["analog_resolution_bits"])
        front_metrics = compute_channel_metrics(derived_df, "front", analog_resolution_bits, series_mode="stroke_percent")
        rear_metrics = compute_channel_metrics(derived_df, "rear", analog_resolution_bits, series_mode="stroke_percent")
        self._populate_metrics_table(self.metrics_table, front_metrics, rear_metrics, category="riding")
        self._populate_metrics_table(self.hardware_metrics_table, front_metrics, rear_metrics, category="hardware")

    @staticmethod
    def _populate_metrics_table(
        table: QtWidgets.QTableWidget,
        front_metrics: list[dict[str, Any]],
        rear_metrics: list[dict[str, Any]],
        *,
        category: str,
    ) -> None:
        if category == "riding":
            MainWindow._populate_riding_metrics_table(table, front_metrics, rear_metrics)
            return

        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Metric", "Front", "Front units", "Rear", "Rear units"])
        front_rows = [
            metric
            for metric in front_metrics
            if metric.get("category") == category and metric.get("metrics_tab_visible", True)
        ]
        rear_metric_map = {
            str(metric["metric"]): metric
            for metric in rear_metrics
            if metric.get("category") == category and metric.get("metrics_tab_visible", True)
        }

        table.setRowCount(len(front_rows))
        for row_index, front_metric in enumerate(front_rows):
            rear_metric = rear_metric_map.get(str(front_metric["metric"]), {"value": None, "units": ""})
            values = [
                front_metric["metric"],
                format_metrics_table_value(front_metric),
                front_metric["units"],
                format_metrics_table_value(rear_metric),
                rear_metric["units"],
            ]
            for column, value in enumerate(values):
                table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()

    @staticmethod
    def _visible_metric_map(metrics: list[dict[str, Any]], category: str) -> dict[str, dict[str, Any]]:
        return {
            str(metric["metric"]): metric
            for metric in metrics
            if metric.get("category") == category and metric.get("metrics_tab_visible", True)
        }

    @staticmethod
    def _metric_or_empty(metric_map: dict[str, dict[str, Any]], metric_name: str | None) -> dict[str, Any]:
        if metric_name is None:
            return {"value": None, "units": ""}
        return metric_map.get(metric_name, {"value": None, "units": ""})

    @staticmethod
    def _populate_riding_metrics_table(
        table: QtWidgets.QTableWidget,
        front_metrics: list[dict[str, Any]],
        rear_metrics: list[dict[str, Any]],
    ) -> None:
        table.setColumnCount(13)
        table.setHorizontalHeaderLabels(
            [
                "Metric",
                "Front position",
                "Units",
                "Front compression",
                "Units",
                "Front rebound",
                "Units",
                "Rear position",
                "Units",
                "Rear compression",
                "Units",
                "Rear rebound",
                "Units",
            ]
        )
        front_metric_map = MainWindow._visible_metric_map(front_metrics, "riding")
        rear_metric_map = MainWindow._visible_metric_map(rear_metrics, "riding")

        table.setRowCount(len(RIDING_METRIC_ROWS))
        for row_index, (row_label, position_metric_name, compression_metric_name, rebound_metric_name) in enumerate(RIDING_METRIC_ROWS):
            front_position = MainWindow._metric_or_empty(front_metric_map, position_metric_name)
            front_compression = MainWindow._metric_or_empty(front_metric_map, compression_metric_name)
            front_rebound = MainWindow._metric_or_empty(front_metric_map, rebound_metric_name)
            rear_position = MainWindow._metric_or_empty(rear_metric_map, position_metric_name)
            rear_compression = MainWindow._metric_or_empty(rear_metric_map, compression_metric_name)
            rear_rebound = MainWindow._metric_or_empty(rear_metric_map, rebound_metric_name)
            values = [
                row_label,
                format_metrics_table_value(front_position),
                front_position.get("units", ""),
                format_metrics_table_value(front_compression),
                front_compression.get("units", ""),
                format_metrics_table_value(front_rebound),
                front_rebound.get("units", ""),
                format_metrics_table_value(rear_position),
                rear_position.get("units", ""),
                format_metrics_table_value(rear_compression),
                rear_compression.get("units", ""),
                format_metrics_table_value(rear_rebound),
                rear_rebound.get("units", ""),
            ]
            for column, value in enumerate(values):
                table.setItem(row_index, column, QtWidgets.QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()

    def save_occupancy_png(self) -> None:
        if self.current_bundle is None or self.current_occupancy is None:
            return
        channel = self.selected_channel()
        default_path = analysis_dir(self.current_bundle.export_dir) / "plots" / f"{self.current_bundle.export_dir.name}_{channel}_heatmap.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save heatmap PNG",
            str(default_path),
            "PNG files (*.png)",
        )
        if not path:
            return
        save_occupancy_heatmap_png(Path(path), self.current_occupancy[channel])
        self.status_label.setText(f"Saved {path}")

    def save_occupancy_grid(self) -> None:
        if self.current_bundle is None or self.current_occupancy is None:
            return
        channel = self.selected_channel()
        default_path = analysis_dir(self.current_bundle.export_dir) / "plots" / f"{self.current_bundle.export_dir.name}_{channel}_heatmap.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save heatmap grid CSV",
            str(default_path),
            "CSV files (*.csv)",
        )
        if not path:
            return
        save_occupancy_grid_csv(Path(path), self.current_occupancy[channel])
        self.status_label.setText(f"Saved {path}")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Suspension Post-Processing Workbench")
    window = MainWindow()
    window.show()
    return app.exec()
