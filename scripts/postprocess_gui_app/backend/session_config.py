from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 10
ANALYSIS_DIRNAME = "analysis"
CONFIG_FILENAME = "session_config.json"
DEFAULT_FRONT_FULL_SCALE_MM = 300.0
DEFAULT_FRONT_SENSOR_FULL_SCALE_MM = 635.0
LEGACY_OCCUPANCY_TRAVEL_BINS = 100
LEGACY_OCCUPANCY_VELOCITY_BINS = 120
DEFAULT_BREAKDOWN_CONFIG = {
    "speed_band_edges_kph": [10.0, 25.0, 45.0],
    "motion_smoothing_ms": 500,
    "braking_threshold_mps2": 1.0,
    "accel_threshold_mps2": 1.0,
    "min_segment_ms": 750,
    "stroke_warning_thresholds_pct": [70.0, 85.0, 95.0],
    "near_bottom_threshold_pct": 95.0,
    "bottom_out_threshold_pct": 99.0,
    "active_activity_threshold": 0.75,
    "repeated_hit_min_speed_kph": 25.0,
    "repeated_hit_activity_threshold": 1.6,
    "repeated_hit_min_duration_ms": 400,
    "high_speed_impact_min_speed_kph": 25.0,
    "high_compression_velocity_percentile": 99.0,
    "high_compression_velocity_min_pct_s": 300.0,
    "pack_down_window_ms": 600,
    "pack_down_shift_threshold_pct": 8.0,
}
DEFAULT_BRAKING_CONFIG = {
    "threshold_mode": "percentile",
    "selected_threshold_type": "percentile",
    "selected_threshold_label": "P80",
    "decel_percentiles": [40.0, 60.0, 80.0],
    "absolute_decel_thresholds_mps2": [2.0, 3.0, 4.0, 5.0],
    "speed_bin_mode": "percentile",
    "speed_percentile_edges": [0.0, 25.0, 50.0, 75.0, 100.0],
    "speed_edges_abs_kph": [0.0, 20.0, 35.0, 50.0, 120.0],
    "valid_speed_min_kph": 2.0,
    "valid_speed_max_kph": 120.0,
    "speed_source_min_kph": 0.5,
    "speed_source_max_kph": 120.0,
    "speed_smoothing_window_s": 0.20,
    "valid_abs_accel_max_mps2": 20.0,
    "coasting_accel_abs_max_mps2": 0.30,
    "valid_stroke_min_pct": -5.0,
    "valid_stroke_max_pct": 120.0,
    "topout_threshold_pct": 2.0,
    "min_braking_event_duration_s": 0.25,
    "merge_event_gap_s": 0.10,
    "min_samples_per_metric_bin": 500,
    "min_events_per_metric_bin": 5,
    "topout_caution_pct": 5.0,
    "topout_high_pct": 10.0,
    "rough_caution_ratio": 1.2,
    "rough_high_ratio": 1.5,
    "dive_low_pct": 4.0,
    "dive_high_pct": 10.0,
    "packdown_caution_pct": 3.0,
    "packdown_high_pct": 6.0,
    "deep80_caution_pct": 1.0,
    "deep90_caution_pct": 0.2,
    "bottom_margin_caution_pct": 10.0,
    "bottom_margin_high_pct": 5.0,
}


def analysis_dir(export_dir: Path) -> Path:
    return export_dir / ANALYSIS_DIRNAME


def config_path(export_dir: Path) -> Path:
    return analysis_dir(export_dir) / CONFIG_FILENAME


def _default_channel_config(
    default_zero_count: int,
    *,
    full_scale_mm: float | None = None,
    sensor_full_scale_mm: float | None = None,
) -> dict[str, Any]:
    return {
        "zero_count": int(default_zero_count),
        "invert": False,
        "mm_per_count": None,
        "full_scale_mm": full_scale_mm,
        "sensor_full_scale_mm": sensor_full_scale_mm,
        "travel_reference": "manual",
        "manual_reference_count": None,
        "velocity_filter_window": 3,
    }


def default_session_config(
    source_path: Path | None,
    export_dir: Path,
    front_zero_count: int,
    rear_zero_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_path": None if source_path is None else str(source_path),
        "export_dir": str(export_dir),
        "front": _default_channel_config(
            front_zero_count,
            full_scale_mm=DEFAULT_FRONT_FULL_SCALE_MM,
            sensor_full_scale_mm=DEFAULT_FRONT_SENSOR_FULL_SCALE_MM,
        ),
        "rear": _default_channel_config(rear_zero_count),
        "plot_defaults": {
            "selected_channel": "front",
            "occupancy_travel_bins": 400,
            "occupancy_velocity_bins": 400,
            "occupancy_color_scale": "sqrt",
            "velocity_axis_mode": "linear",
        },
        "braking": copy.deepcopy(DEFAULT_BRAKING_CONFIG),
        "breakdown": copy.deepcopy(DEFAULT_BREAKDOWN_CONFIG),
        "last_derived_at": None,
        "last_derived_signature": None,
    }


def merge_defaults(defaults: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(defaults)
    if not overrides:
        return merged

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_defaults(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_front_channel_config(front: dict[str, Any]) -> None:
    # Front fork stroke is fixed by the bike geometry, not by the observed data span.
    front["full_scale_mm"] = DEFAULT_FRONT_FULL_SCALE_MM
    front["sensor_full_scale_mm"] = DEFAULT_FRONT_SENSOR_FULL_SCALE_MM


def migrate_session_config(config: dict[str, Any]) -> dict[str, Any]:
    schema_version = int(config.get("schema_version", 0) or 0)

    front = config.setdefault("front", {})
    rear = config.setdefault("rear", {})
    plot_defaults = config.setdefault("plot_defaults", {})
    braking = config.setdefault("braking", {})
    breakdown = config.setdefault("breakdown", {})
    _normalize_front_channel_config(front)
    if schema_version < SCHEMA_VERSION:
        for channel_config in (front, rear):
            if channel_config.get("velocity_filter_window") in (None, 21):
                channel_config["velocity_filter_window"] = 3
            manual_reference_count = channel_config.get("manual_reference_count")
            if manual_reference_count in ("", None):
                channel_config["manual_reference_count"] = None
            else:
                try:
                    channel_config["manual_reference_count"] = float(manual_reference_count)
                except (TypeError, ValueError):
                    channel_config["manual_reference_count"] = None

        if plot_defaults.get("occupancy_travel_bins") in (None, LEGACY_OCCUPANCY_TRAVEL_BINS):
            plot_defaults["occupancy_travel_bins"] = 400
        if plot_defaults.get("occupancy_velocity_bins") in (None, LEGACY_OCCUPANCY_VELOCITY_BINS):
            plot_defaults["occupancy_velocity_bins"] = 400

    for channel_config in (front, rear):
        channel_config["travel_reference"] = "manual"
        for obsolete_key in ("reference_method", "reference_percentile", "reference_window_samples"):
            channel_config.pop(obsolete_key, None)

    if schema_version < SCHEMA_VERSION and plot_defaults.get("occupancy_color_scale") in (None, "", "log"):
        plot_defaults["occupancy_color_scale"] = "sqrt"
    if plot_defaults.get("occupancy_color_scale") not in {"linear", "sqrt", "log"}:
        plot_defaults["occupancy_color_scale"] = "sqrt"
    if plot_defaults.get("velocity_axis_mode") not in {"linear", "signed_log"}:
        plot_defaults["velocity_axis_mode"] = "linear"

    config["braking"] = merge_defaults(DEFAULT_BRAKING_CONFIG, braking)
    config["breakdown"] = merge_defaults(DEFAULT_BREAKDOWN_CONFIG, breakdown)
    config["schema_version"] = SCHEMA_VERSION
    return config


def load_or_create_session_config(
    export_dir: Path,
    source_path: Path | None,
    front_zero_count: int,
    rear_zero_count: int,
) -> dict[str, Any]:
    defaults = default_session_config(
        source_path=source_path,
        export_dir=export_dir,
        front_zero_count=front_zero_count,
        rear_zero_count=rear_zero_count,
    )
    path = config_path(export_dir)
    if not path.exists():
        return defaults

    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    merged = migrate_session_config(merge_defaults(defaults, loaded))
    if source_path is not None:
        merged["source_path"] = str(source_path)
    merged["export_dir"] = str(export_dir)
    return merged


def save_session_config(export_dir: Path, config: dict[str, Any]) -> Path:
    target_dir = analysis_dir(export_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = config_path(export_dir)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    return path


def channel_mm_per_count(channel_config: dict[str, Any], analog_resolution_bits: int) -> float | None:
    mm_per_count = channel_config.get("mm_per_count")
    if mm_per_count not in (None, ""):
        return float(mm_per_count)

    sensor_full_scale_mm = channel_config.get("sensor_full_scale_mm")
    if sensor_full_scale_mm not in (None, ""):
        adc_max_count = (1 << analog_resolution_bits) - 1
        if adc_max_count <= 0:
            return None
        return float(sensor_full_scale_mm) / float(adc_max_count)

    full_scale_mm = channel_config.get("full_scale_mm")
    if full_scale_mm in (None, ""):
        return None

    adc_max_count = (1 << analog_resolution_bits) - 1
    if adc_max_count <= 0:
        return None
    return float(full_scale_mm) / float(adc_max_count)


def build_analysis_signature(export_dir: Path, config: dict[str, Any]) -> str:
    analog_path = export_dir / "analog.csv"
    summary_path = export_dir / "summary.json"
    analog_stat = analog_path.stat() if analog_path.exists() else None
    summary_stat = summary_path.stat() if summary_path.exists() else None

    payload = {
        "schema_version": SCHEMA_VERSION,
        "front": config.get("front", {}),
        "rear": config.get("rear", {}),
        "plot_defaults": config.get("plot_defaults", {}),
        "analog": None
        if analog_stat is None
        else {"size": analog_stat.st_size, "mtime_ns": analog_stat.st_mtime_ns},
        "summary": None
        if summary_stat is None
        else {"size": summary_stat.st_size, "mtime_ns": summary_stat.st_mtime_ns},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
