from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .analysis_service import channel_series, compute_channel_metrics, compute_velocity_histogram
from .session_config import analysis_dir
from .session_service import SessionBundle, open_bin_session, open_export_session


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MPLCONFIGDIR = REPO_ROOT / ".codex_tmp" / "matplotlib"
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))

QUICKLOOK_DIRNAME = "quicklook"
NEAR_FULL_STROKE_THRESHOLD_PCT = 95.0
BOTTOM_OUT_THRESHOLD_PCT = 99.0
NOMINAL_STATUS_NAMES = {
    "BOOT",
    "RTC_VALID",
    "SD_READY",
    "IMU_READY",
    "SESSION_START",
    "SESSION_STOP",
    "CLEAN_CLOSE",
}


@dataclass
class QuicklookResult:
    input_path: Path
    bundle: SessionBundle
    output_dir: Path
    summary: dict[str, Any]
    summary_json_path: Path
    summary_text_path: Path
    artifact_paths: dict[str, Path]
    summary_text: str


def quicklook_dir(export_dir: Path) -> Path:
    return analysis_dir(export_dir) / QUICKLOOK_DIRNAME


def load_session_for_quicklook(input_path: Path) -> SessionBundle:
    if not input_path.exists():
        raise FileNotFoundError(f"quicklook input does not exist: {input_path}")
    if input_path.is_dir():
        return open_export_session(input_path)
    if input_path.is_file() and input_path.suffix.lower() == ".bin":
        return open_bin_session(input_path)
    raise ValueError("quicklook input must be a .BIN file or an exports/<session>/ directory")


def _import_matplotlib_pyplot():
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_value(value: Any, precision: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{precision}f}"
    return str(value)


def _count_segments(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    mask = np.asarray(mask, dtype=bool)
    return int(mask[0]) + int(np.count_nonzero(mask[1:] & ~mask[:-1]))


def _longest_segment_duration(mask: np.ndarray, dt_s: np.ndarray) -> float:
    if mask.size == 0 or dt_s.size == 0:
        return 0.0
    longest = 0.0
    running = 0.0
    for active, sample_dt in zip(mask, dt_s, strict=False):
        if active:
            running += float(sample_dt)
            if running > longest:
                longest = running
        else:
            running = 0.0
    return longest


def _threshold_event_summary(
    travel: np.ndarray,
    dt_s: np.ndarray,
    threshold_pct: float,
) -> dict[str, Any]:
    finite = np.isfinite(travel) & np.isfinite(dt_s) & (dt_s > 0.0)
    total_time_s = float(np.sum(dt_s[finite])) if np.any(finite) else 0.0
    if not np.any(finite):
        return {
            "threshold_pct": float(threshold_pct),
            "threshold_value": None,
            "event_count": 0,
            "time_s": 0.0,
            "time_pct": 0.0,
            "longest_event_s": 0.0,
        }

    finite_travel = travel[finite]
    minimum = float(np.min(finite_travel))
    maximum = float(np.max(finite_travel))
    used_stroke = maximum - minimum
    if used_stroke <= 0.0:
        return {
            "threshold_pct": float(threshold_pct),
            "threshold_value": None,
            "event_count": 0,
            "time_s": 0.0,
            "time_pct": 0.0,
            "longest_event_s": 0.0,
        }

    threshold_value = minimum + used_stroke * (float(threshold_pct) / 100.0)
    active = finite & (travel >= threshold_value)
    active_time_s = float(np.sum(dt_s[active])) if np.any(active) else 0.0
    time_pct = 0.0 if total_time_s <= 0.0 else 100.0 * active_time_s / total_time_s
    return {
        "threshold_pct": float(threshold_pct),
        "threshold_value": threshold_value,
        "event_count": _count_segments(active),
        "time_s": active_time_s,
        "time_pct": time_pct,
        "longest_event_s": _longest_segment_duration(active, dt_s),
    }


def _status_summary(bundle: SessionBundle) -> dict[str, Any]:
    counts = Counter(row.get("name", "") for row in bundle.status_rows if row.get("name"))
    highlighted_names = sorted(name for name in counts if name not in NOMINAL_STATUS_NAMES)
    highlighted_rows = []
    for name in highlighted_names:
        event_times = [
            value
            for value in (_safe_float(row.get("host_time_s")) for row in bundle.status_rows if row.get("name") == name)
            if value is not None
        ]
        highlighted_rows.append(
            {
                "name": name,
                "count": int(counts[name]),
                "first_time_s": None if not event_times else float(min(event_times)),
                "last_time_s": None if not event_times else float(max(event_times)),
            }
        )
    return {
        "row_count": len(bundle.status_rows),
        "status_counts": {name: int(count) for name, count in sorted(counts.items())},
        "highlighted_statuses": highlighted_rows,
        "highlight_count": int(sum(row["count"] for row in highlighted_rows)),
        "clean_close_recorded": bool(counts.get("CLEAN_CLOSE", 0)),
    }


def _channel_summary(bundle: SessionBundle, channel: str, analog_resolution_bits: int) -> dict[str, Any]:
    metrics = compute_channel_metrics(bundle.derived_df, channel, analog_resolution_bits)
    metric_map = {metric["metric"]: metric for metric in metrics}
    series = channel_series(bundle.derived_df, channel)
    travel_for_events = series["filtered_travel"] if np.any(np.isfinite(series["filtered_travel"])) else series["travel"]
    near_full = _threshold_event_summary(travel_for_events, series["dt_s"], NEAR_FULL_STROKE_THRESHOLD_PCT)
    bottom_out = _threshold_event_summary(travel_for_events, series["dt_s"], BOTTOM_OUT_THRESHOLD_PCT)
    return {
        "travel_units": series["travel_units"],
        "velocity_units": series["velocity_units"],
        "used_stroke": _safe_float(metric_map["Used stroke"]["value"]),
        "minimum_travel": _safe_float(metric_map["Minimum travel"]["value"]),
        "maximum_travel": _safe_float(metric_map["Maximum travel"]["value"]),
        "peak_compression_velocity": _safe_float(metric_map["Peak compression velocity"]["value"]),
        "peak_rebound_velocity": _safe_float(metric_map["Peak rebound velocity"]["value"]),
        "rms_velocity": _safe_float(metric_map["RMS velocity"]["value"]),
        "time_in_bottom_10_pct": _safe_float(metric_map["Time in bottom 10%"]["value"]),
        "time_in_top_10_pct": _safe_float(metric_map["Time in top 10%"]["value"]),
        "adc_rail_near_low_samples": int(metric_map["ADC rail-near low"]["value"]),
        "adc_rail_near_high_samples": int(metric_map["ADC rail-near high"]["value"]),
        "near_full_stroke": near_full,
        "bottom_out": bottom_out,
    }


def build_quicklook_summary(bundle: SessionBundle, input_path: Path, output_dir: Path) -> dict[str, Any]:
    analog_resolution_bits = int(bundle.summary["header"]["analog_resolution_bits"])
    status = _status_summary(bundle)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "source_path": None if bundle.source_path is None else str(bundle.source_path),
        "export_dir": str(bundle.export_dir),
        "output_dir": str(output_dir),
        "session": {
            "duration_s": _safe_float(bundle.summary.get("timing", {}).get("duration_us")) / 1_000_000.0
            if bundle.summary.get("timing", {}).get("duration_us") is not None
            else None,
            "analog_rows": int(bundle.analog_df.height),
            "wheel_rows": int(bundle.wheel_df.height),
            "imu_frame_rows": int(bundle.imu_frame_df.height),
            "status_rows": len(bundle.status_rows),
            "record_counts": dict(bundle.summary.get("record_counts", {})),
        },
        "channels": {
            "front": _channel_summary(bundle, "front", analog_resolution_bits),
            "rear": _channel_summary(bundle, "rear", analog_resolution_bits),
        },
        "status": status,
    }
    return summary


def build_quicklook_text(summary: dict[str, Any]) -> str:
    lines = [
        "Trackside Quicklook",
        f"Input: {summary.get('input_path', '-')}",
        f"Source: {summary.get('source_path', '-')}",
        f"Export: {summary.get('export_dir', '-')}",
        f"Output: {summary.get('output_dir', '-')}",
        "",
        "Session",
        f"  Duration: {_format_value(summary['session'].get('duration_s'), 3)} s",
        f"  Analog rows: {summary['session'].get('analog_rows', 0)}",
        f"  Wheel rows: {summary['session'].get('wheel_rows', 0)}",
        f"  IMU frame rows: {summary['session'].get('imu_frame_rows', 0)}",
        f"  Status rows: {summary['session'].get('status_rows', 0)}",
        "",
        "Logger health",
        f"  Clean close recorded: {'yes' if summary['status'].get('clean_close_recorded') else 'no'}",
    ]
    highlighted = summary["status"].get("highlighted_statuses", [])
    if highlighted:
        for row in highlighted:
            first_time = _format_value(row.get("first_time_s"), 3)
            last_time = _format_value(row.get("last_time_s"), 3)
            lines.append(f"  {row['name']}: {row['count']} (first {first_time} s, last {last_time} s)")
    else:
        lines.append("  No highlighted status rows")

    for channel in ("front", "rear"):
        channel_summary = summary["channels"][channel]
        near_full = channel_summary["near_full_stroke"]
        bottom_out = channel_summary["bottom_out"]
        lines.extend(
            [
                "",
                f"{channel.capitalize()} channel",
                f"  Used stroke: {_format_value(channel_summary.get('used_stroke'))} {channel_summary['travel_units']}",
                (
                    f"  Peak compression velocity: {_format_value(channel_summary.get('peak_compression_velocity'))} "
                    f"{channel_summary['velocity_units']}"
                ),
                (
                    f"  Peak rebound velocity: {_format_value(channel_summary.get('peak_rebound_velocity'))} "
                    f"{channel_summary['velocity_units']}"
                ),
                f"  RMS velocity: {_format_value(channel_summary.get('rms_velocity'))} {channel_summary['velocity_units']}",
                (
                    f"  Near-full stroke >= {near_full['threshold_pct']:.0f}%: "
                    f"{near_full['event_count']} events, {_format_value(near_full['time_s'], 3)} s, "
                    f"{_format_value(near_full['time_pct'], 2)}% session"
                ),
                (
                    f"  Bottom-out >= {bottom_out['threshold_pct']:.0f}%: "
                    f"{bottom_out['event_count']} events, {_format_value(bottom_out['time_s'], 3)} s, "
                    f"{_format_value(bottom_out['time_pct'], 2)}% session"
                ),
                (
                    f"  ADC rail-near low/high: {channel_summary['adc_rail_near_low_samples']} / "
                    f"{channel_summary['adc_rail_near_high_samples']} samples"
                ),
            ]
        )

    return "\n".join(lines) + "\n"


def _save_travel_overview_png(path: Path, bundle: SessionBundle, summary: dict[str, Any], dpi: int = 180) -> None:
    plt = _import_matplotlib_pyplot()
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = {"front": "#0b84a5", "rear": "#f28e2b"}

    for axis, channel in zip(axes, ("front", "rear"), strict=False):
        series = channel_series(bundle.derived_df, channel)
        channel_summary = summary["channels"][channel]
        axis.plot(series["time_s"], series["travel"], color=colors[channel], alpha=0.30, linewidth=0.9, label="Travel")
        axis.plot(
            series["time_s"],
            series["filtered_travel"],
            color=colors[channel],
            linewidth=1.4,
            label="Filtered travel",
        )

        near_full_value = channel_summary["near_full_stroke"].get("threshold_value")
        if near_full_value is not None:
            axis.axhline(float(near_full_value), color="#7f7f7f", linestyle="--", linewidth=1.0, label="Near-full")
        bottom_out_value = channel_summary["bottom_out"].get("threshold_value")
        if bottom_out_value is not None:
            axis.axhline(float(bottom_out_value), color="#c44536", linestyle=":", linewidth=1.0, label="Bottom-out")

        axis.set_title(
            f"{channel.capitalize()} travel overview | used stroke "
            f"{_format_value(channel_summary.get('used_stroke'))} {channel_summary['travel_units']}"
        )
        axis.set_ylabel(f"Travel [{series['travel_units']}]")
        axis.grid(True, alpha=0.2)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _save_velocity_histograms_png(path: Path, bundle: SessionBundle, dpi: int = 180) -> None:
    plt = _import_matplotlib_pyplot()
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    colors = {
        "all": "#4e79a7",
        "positive": "#59a14f",
        "negative": "#e15759",
    }

    for axis, channel in zip(axes, ("front", "rear"), strict=False):
        histogram = compute_velocity_histogram(bundle.derived_df, channel)
        edges = histogram["edges"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        axis.plot(centers, histogram["all"], color=colors["all"], linewidth=1.8, label="All")
        axis.plot(centers, histogram["positive"], color=colors["positive"], linewidth=1.5, label="Compression (+)")
        axis.plot(centers, histogram["negative"], color=colors["negative"], linewidth=1.5, label="Rebound (-)")
        axis.set_title(f"{channel.capitalize()} velocity histogram")
        axis.set_ylabel("Occupancy [s]")
        axis.set_xlabel(f"Velocity [{histogram['units']}]")
        axis.grid(True, alpha=0.2)
        axis.legend(loc="upper right")

    figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def run_quicklook(input_path: Path, output_dir: Path | None = None) -> QuicklookResult:
    resolved_input = input_path.resolve()
    bundle = load_session_for_quicklook(resolved_input)
    resolved_output_dir = (quicklook_dir(bundle.export_dir) if output_dir is None else output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_quicklook_summary(bundle, resolved_input, resolved_output_dir)
    artifact_paths = {
        "travel_overview_png": resolved_output_dir / "travel_overview.png",
        "velocity_histograms_png": resolved_output_dir / "velocity_histograms.png",
    }
    _save_travel_overview_png(artifact_paths["travel_overview_png"], bundle, summary)
    _save_velocity_histograms_png(artifact_paths["velocity_histograms_png"], bundle)

    summary["artifacts"] = {name: str(path) for name, path in artifact_paths.items()}
    summary_text = build_quicklook_text(summary)
    summary_json_path = resolved_output_dir / "quicklook_summary.json"
    summary_text_path = resolved_output_dir / "quicklook_summary.txt"
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    summary_text_path.write_text(summary_text, encoding="utf-8")

    return QuicklookResult(
        input_path=resolved_input,
        bundle=bundle,
        output_dir=resolved_output_dir,
        summary=summary,
        summary_json_path=summary_json_path,
        summary_text_path=summary_text_path,
        artifact_paths=artifact_paths,
        summary_text=summary_text,
    )
