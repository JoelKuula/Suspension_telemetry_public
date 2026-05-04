from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .session_config import (
    DEFAULT_REFERENCE_METHOD,
    DEFAULT_REFERENCE_PERCENTILE,
    DEFAULT_REFERENCE_WINDOW_SAMPLES,
    analysis_dir,
    build_analysis_signature,
    channel_mm_per_count,
)


DERIVED_ANALOG_FILENAME = "derived_analog.parquet"
BALANCE_ACTIVE_THRESHOLD_PCT = 10.0
BALANCE_NEUTRAL_BAND_PCT = 10.0
REFERENCE_LOW_PERCENTILE = 1.0
REFERENCE_HIGH_PERCENTILE = 99.0
OCCUPANCY_MODE_BINS = 100
OCCUPANCY_MODE_SMOOTH_SIGMA_BINS = 2.0
VELOCITY_AXIS_LINEAR = "linear"
VELOCITY_AXIS_SIGNED_LOG = "signed_log"
VALID_VELOCITY_AXIS_MODES = {VELOCITY_AXIS_LINEAR, VELOCITY_AXIS_SIGNED_LOG}


def normalize_velocity_axis_mode(axis_mode: str | None) -> str:
    return axis_mode if axis_mode in VALID_VELOCITY_AXIS_MODES else VELOCITY_AXIS_LINEAR


def transform_velocity_axis_values(values: np.ndarray, axis_mode: str | None) -> np.ndarray:
    mode = normalize_velocity_axis_mode(axis_mode)
    values = np.asarray(values, dtype=np.float64)
    if mode == VELOCITY_AXIS_LINEAR:
        return values.copy()

    transformed = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    transformed[finite] = np.sign(values[finite]) * np.log10(1.0 + np.abs(values[finite]))
    return transformed


def inverse_transform_velocity_axis_values(values: np.ndarray, axis_mode: str | None) -> np.ndarray:
    mode = normalize_velocity_axis_mode(axis_mode)
    values = np.asarray(values, dtype=np.float64)
    if mode == VELOCITY_AXIS_LINEAR:
        return values.copy()

    original = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    original[finite] = np.sign(values[finite]) * (np.power(10.0, np.abs(values[finite])) - 1.0)
    return original


def velocity_axis_range_for_data(velocity_min: float, velocity_max: float, axis_mode: str | None) -> tuple[float, float]:
    mode = normalize_velocity_axis_mode(axis_mode)
    if mode == VELOCITY_AXIS_LINEAR:
        low = float(velocity_min)
        high = float(velocity_max)
    else:
        transformed = transform_velocity_axis_values(np.asarray([velocity_min, velocity_max]), mode)
        finite = transformed[np.isfinite(transformed)]
        if finite.size == 0:
            low, high = -1.0, 1.0
        else:
            low = float(np.min(finite))
            high = float(np.max(finite))

    if low == high:
        pad = 1.0 if low == 0.0 else abs(low) * 0.05
        low -= pad
        high += pad
    return low, high


def velocity_axis_label(base_label: str, axis_mode: str | None) -> str:
    mode = normalize_velocity_axis_mode(axis_mode)
    if mode == VELOCITY_AXIS_SIGNED_LOG:
        return f"{base_label} magnitude (Rebound | Compression)"
    return base_label


def _format_velocity_tick_magnitude(value: float) -> str:
    value = abs(float(value))
    if value >= 1000.0:
        scaled = value / 1000.0
        if scaled >= 10.0:
            return f"{scaled:.0f}k"
        return f"{scaled:g}k"
    if value >= 1.0:
        return f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:g}"
    return f"{value:g}"


def _velocity_tick_magnitudes(max_abs: float) -> list[float]:
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        return []
    max_exponent = int(np.floor(np.log10(max_abs)))
    min_exponent = min(0, max_exponent)
    return [
        float(10.0**exponent)
        for exponent in range(min_exponent, max_exponent + 1)
        if float(10.0**exponent) <= max_abs * 1.000001
    ]


def velocity_axis_ticks(
    velocity_min: float,
    velocity_max: float,
    axis_mode: str | None,
) -> list[tuple[float, str]]:
    mode = normalize_velocity_axis_mode(axis_mode)
    if mode == VELOCITY_AXIS_LINEAR:
        return []

    max_abs = max(abs(float(velocity_min)), abs(float(velocity_max)))
    magnitudes = _velocity_tick_magnitudes(max_abs)
    ticks: list[tuple[float, str]] = []
    for magnitude in reversed(magnitudes):
        if velocity_min <= -magnitude <= velocity_max:
            position = float(transform_velocity_axis_values(np.asarray([-magnitude]), mode)[0])
            ticks.append((position, f"R {_format_velocity_tick_magnitude(magnitude)}"))
    if velocity_min <= 0.0 <= velocity_max:
        ticks.append((0.0, "0"))
    for magnitude in magnitudes:
        if velocity_min <= magnitude <= velocity_max:
            position = float(transform_velocity_axis_values(np.asarray([magnitude]), mode)[0])
            ticks.append((position, f"C {_format_velocity_tick_magnitude(magnitude)}"))
    return ticks


def derived_analog_path(export_dir: Path) -> Path:
    return analysis_dir(export_dir) / DERIVED_ANALOG_FILENAME


def load_export_analog(export_dir: Path) -> pl.DataFrame:
    analog_path = export_dir / "analog.csv"
    if not analog_path.exists():
        raise FileNotFoundError(f"missing analog export: {analog_path}")
    return pl.read_csv(analog_path, null_values=[""], infer_schema_length=1000)


def estimate_default_zero_count(raw_values: np.ndarray) -> int:
    if raw_values.size == 0:
        return 0
    window = raw_values[: min(raw_values.size, 250)]
    return int(round(float(np.median(window))))


def _finite_percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, percentile))


def _finite_extreme(values: np.ndarray, which: str) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    if which == "min":
        return float(np.min(finite))
    if which == "max":
        return float(np.max(finite))
    raise ValueError(f"unsupported extreme selector: {which}")


def _finite_values(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _robust_percentile_reference(values: np.ndarray, which: str, percentile: float) -> float | None:
    finite = _finite_values(values)
    if finite.size == 0:
        return None
    clipped_percentile = min(max(float(percentile), 0.0001), 50.0)
    if which == "min":
        return float(np.percentile(finite, clipped_percentile))
    if which == "max":
        return float(np.percentile(finite, 100.0 - clipped_percentile))
    raise ValueError(f"unsupported robust reference selector: {which}")


def _rolling_window_view(values: np.ndarray, window: int) -> np.ndarray | None:
    finite = _finite_values(values)
    if finite.size == 0:
        return None
    window = max(1, int(window))
    if finite.size < window:
        return None
    shape = (finite.size - window + 1, window)
    strides = (finite.strides[0], finite.strides[0])
    return np.lib.stride_tricks.as_strided(finite, shape=shape, strides=strides)


def _sustained_reference(values: np.ndarray, which: str, window: int) -> float | None:
    windows = _rolling_window_view(values, window)
    if windows is None:
        return _finite_extreme(values, which)
    if which == "min":
        return float(np.min(np.max(windows, axis=1)))
    if which == "max":
        return float(np.max(np.min(windows, axis=1)))
    raise ValueError(f"unsupported sustained reference selector: {which}")


def _robust_reference(
    values: np.ndarray,
    which: str,
    *,
    method: str,
    percentile: float,
    window_samples: int,
) -> tuple[float | None, str]:
    normalized_method = method if method in {"percentile", "sustained"} else DEFAULT_REFERENCE_METHOD
    if normalized_method == "sustained":
        return _sustained_reference(values, which, window_samples), f"sustained-{which}-n{max(1, int(window_samples))}"
    return (
        _robust_percentile_reference(values, which, percentile),
        f"p{min(max(float(percentile), 0.0001), 50.0):g}-{which}",
    )


def _ensure_odd_window(window: int) -> int:
    if window < 1:
        return 1
    if window % 2 == 0:
        return window + 1
    return window


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = _ensure_odd_window(window)
    if window <= 1 or values.size == 0:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def choose_auto_reference(position_counts: np.ndarray) -> str:
    if position_counts.size == 0:
        return "absolute"
    initial_window = position_counts[: min(position_counts.size, 250)]
    initial_level = float(np.median(initial_window))
    span_min = _finite_percentile(position_counts, REFERENCE_LOW_PERCENTILE)
    span_max = _finite_percentile(position_counts, REFERENCE_HIGH_PERCENTILE)
    if span_min is None or span_max is None:
        return "absolute"
    if abs(initial_level - span_max) <= abs(initial_level - span_min):
        return "max"
    return "min"


def apply_travel_reference(
    position_counts: np.ndarray,
    reference_mode: str,
    *,
    reference_method: str = DEFAULT_REFERENCE_METHOD,
    reference_percentile: float = DEFAULT_REFERENCE_PERCENTILE,
    reference_window_samples: int = DEFAULT_REFERENCE_WINDOW_SAMPLES,
    manual_reference_count: float | None = None,
) -> tuple[np.ndarray, str]:
    if position_counts.size == 0:
        return position_counts.copy(), "absolute"

    resolved_mode = reference_mode
    if reference_mode == "auto":
        resolved_mode = choose_auto_reference(position_counts)
    elif reference_mode == "robust_min":
        resolved_mode = "min"
    elif reference_mode == "robust_max":
        resolved_mode = "max"

    lower_reference = _finite_extreme(position_counts, "min")
    upper_reference = _finite_extreme(position_counts, "max")
    if lower_reference is None or upper_reference is None:
        return position_counts.copy(), "absolute"

    if resolved_mode == "absolute":
        return position_counts.copy(), "absolute"

    if resolved_mode in {"min", "max"} and manual_reference_count is not None:
        anchor = float(manual_reference_count)
        if resolved_mode == "min":
            return position_counts - anchor, "relative-from-manual-min"
        return anchor - position_counts, "relative-from-manual-max"

    robust_requested = reference_mode in {"auto", "robust_min", "robust_max"}
    if resolved_mode == "min":
        if robust_requested:
            anchor, label = _robust_reference(
                position_counts,
                "min",
                method=reference_method,
                percentile=reference_percentile,
                window_samples=reference_window_samples,
            )
            if anchor is None:
                anchor = lower_reference
                label = "absolute-min"
            return position_counts - anchor, f"relative-from-robust-{label}"
        return position_counts - lower_reference, "relative-from-absolute-min"
    if resolved_mode == "max":
        if robust_requested:
            anchor, label = _robust_reference(
                position_counts,
                "max",
                method=reference_method,
                percentile=reference_percentile,
                window_samples=reference_window_samples,
            )
            if anchor is None:
                anchor = upper_reference
                label = "absolute-max"
            return anchor - position_counts, f"relative-from-robust-{label}"
        return upper_reference - position_counts, "relative-from-absolute-max"
    raise ValueError(f"unsupported travel reference: {reference_mode}")


def _series_to_numpy(frame: pl.DataFrame, column: str, fill_value: float = np.nan) -> np.ndarray:
    values = frame.get_column(column).fill_null(fill_value).to_numpy()
    return np.asarray(values, dtype=np.float64)


def _monotonic_time_seconds(time_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if time_us.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    time_s = np.asarray(time_us, dtype=np.float64) / 1_000_000.0
    diffs = np.diff(time_s)
    positive = diffs[diffs > 0]
    default_dt = float(np.median(positive)) if positive.size else 0.001

    monotonic = time_s.copy()
    for index in range(1, monotonic.size):
        if monotonic[index] <= monotonic[index - 1]:
            monotonic[index] = monotonic[index - 1] + default_dt

    dt_s = np.empty(monotonic.size, dtype=np.float64)
    dt_s[0] = default_dt
    if monotonic.size > 1:
        dt_s[1:] = np.diff(monotonic)
    return monotonic, dt_s


def _channel_arrays(
    raw_values: np.ndarray,
    time_s: np.ndarray,
    analog_resolution_bits: int,
    channel_config: dict[str, Any],
) -> dict[str, Any]:
    sign = -1.0 if channel_config.get("invert") else 1.0
    zero_count = float(channel_config.get("zero_count", 0))
    reference_mode = str(channel_config.get("travel_reference", "auto"))
    reference_method = str(channel_config.get("reference_method", DEFAULT_REFERENCE_METHOD))
    reference_percentile = float(channel_config.get("reference_percentile", DEFAULT_REFERENCE_PERCENTILE))
    reference_window_samples = int(
        channel_config.get("reference_window_samples", DEFAULT_REFERENCE_WINDOW_SAMPLES)
    )
    velocity_filter_window = _ensure_odd_window(int(channel_config.get("velocity_filter_window", 3)))
    mm_per_count = channel_mm_per_count(channel_config, analog_resolution_bits)

    calibrated_counts = sign * (raw_values - zero_count)
    manual_raw_reference = _optional_float(channel_config.get("manual_reference_count"))
    manual_reference_count = None
    if manual_raw_reference is not None:
        manual_reference_count = sign * (manual_raw_reference - zero_count)
    travel_counts, reference_used = apply_travel_reference(
        calibrated_counts,
        reference_mode,
        reference_method=reference_method,
        reference_percentile=reference_percentile,
        reference_window_samples=reference_window_samples,
        manual_reference_count=manual_reference_count,
    )
    filtered_counts = moving_average(travel_counts, velocity_filter_window)
    velocity_counts = np.gradient(filtered_counts, time_s) if filtered_counts.size else np.asarray([], dtype=np.float64)

    arrays: dict[str, Any] = {
        "calibrated_counts": calibrated_counts,
        "travel_counts": travel_counts,
        "filtered_travel_counts": filtered_counts,
        "velocity_counts_per_s_filtered": velocity_counts,
        "travel_reference_used": reference_used,
        "velocity_filter_window_used": velocity_filter_window,
        "mm_per_count_used": mm_per_count,
    }
    if mm_per_count is not None:
        arrays["travel_mm"] = travel_counts * mm_per_count
        arrays["filtered_travel_mm"] = filtered_counts * mm_per_count
        arrays["velocity_mm_per_s_filtered"] = velocity_counts * mm_per_count

    full_scale_mm = channel_config.get("full_scale_mm")
    if full_scale_mm not in (None, "", 0, 0.0) and arrays.get("travel_mm") is not None:
        stroke_scale = float(full_scale_mm)
        arrays["travel_stroke_pct"] = 100.0 * arrays["travel_mm"] / stroke_scale
        arrays["filtered_travel_stroke_pct"] = 100.0 * arrays["filtered_travel_mm"] / stroke_scale
        arrays["velocity_stroke_pct_per_s_filtered"] = 100.0 * arrays["velocity_mm_per_s_filtered"] / stroke_scale
        arrays["stroke_percent_source"] = "configured-full-scale"
    else:
        stroke_minimum, stroke_span = _percent_scale_params(travel_counts)
        arrays["travel_stroke_pct"] = _apply_percent_scale(travel_counts, stroke_minimum, stroke_span)
        arrays["filtered_travel_stroke_pct"] = _apply_percent_scale(filtered_counts, stroke_minimum, stroke_span)
        velocity_stroke_pct = np.full(velocity_counts.shape, np.nan, dtype=np.float64)
        finite_velocity = np.isfinite(velocity_counts)
        if stroke_span == 0.0:
            velocity_stroke_pct[finite_velocity] = 0.0
        elif stroke_span is not None and stroke_span > 0.0:
            velocity_stroke_pct[finite_velocity] = 100.0 * velocity_counts[finite_velocity] / stroke_span
        arrays["velocity_stroke_pct_per_s_filtered"] = velocity_stroke_pct
        arrays["stroke_percent_source"] = "used-stroke"
    return arrays


def build_derived_analog(
    export_dir: Path,
    config: dict[str, Any],
    analog_df: pl.DataFrame | None = None,
    force: bool = False,
    persist: bool = True,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    derived_path = derived_analog_path(export_dir)
    signature = build_analysis_signature(export_dir, config)
    if persist and not force and derived_path.exists() and config.get("last_derived_signature") == signature:
        return pl.read_parquet(derived_path), config

    if analog_df is None:
        analog_df = load_export_analog(export_dir)

    time_us = _series_to_numpy(analog_df, "host_timestamp_us", 0.0)
    time_s, dt_s = _monotonic_time_seconds(time_us)
    analog_resolution_bits = 12
    summary_path = export_dir / "summary.json"
    if summary_path.exists():
        import json

        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        analog_resolution_bits = int(summary["header"]["analog_resolution_bits"])
    else:
        analog_resolution_bits = int(config.get("analog_resolution_bits", 12))

    front_arrays = _channel_arrays(
        raw_values=_series_to_numpy(analog_df, "front_raw", 0.0),
        time_s=time_s,
        analog_resolution_bits=analog_resolution_bits,
        channel_config=config["front"],
    )
    rear_arrays = _channel_arrays(
        raw_values=_series_to_numpy(analog_df, "rear_raw", 0.0),
        time_s=time_s,
        analog_resolution_bits=analog_resolution_bits,
        channel_config=config["rear"],
    )

    derived_df = analog_df.with_columns(
        [
            pl.Series("dt_s", dt_s),
            pl.Series("front_calibrated_counts", front_arrays["calibrated_counts"]),
            pl.Series("front_travel_counts", front_arrays["travel_counts"]),
            pl.Series("front_filtered_travel_counts", front_arrays["filtered_travel_counts"]),
            pl.Series("front_velocity_counts_per_s_filtered", front_arrays["velocity_counts_per_s_filtered"]),
            pl.Series("front_travel_stroke_pct", front_arrays["travel_stroke_pct"]),
            pl.Series("front_filtered_travel_stroke_pct", front_arrays["filtered_travel_stroke_pct"]),
            pl.Series("front_velocity_stroke_pct_per_s_filtered", front_arrays["velocity_stroke_pct_per_s_filtered"]),
            pl.Series("front_travel_reference_used", [front_arrays["travel_reference_used"]] * analog_df.height),
            pl.Series("front_stroke_percent_source", [front_arrays["stroke_percent_source"]] * analog_df.height),
            pl.Series("rear_calibrated_counts", rear_arrays["calibrated_counts"]),
            pl.Series("rear_travel_counts", rear_arrays["travel_counts"]),
            pl.Series("rear_filtered_travel_counts", rear_arrays["filtered_travel_counts"]),
            pl.Series("rear_velocity_counts_per_s_filtered", rear_arrays["velocity_counts_per_s_filtered"]),
            pl.Series("rear_travel_stroke_pct", rear_arrays["travel_stroke_pct"]),
            pl.Series("rear_filtered_travel_stroke_pct", rear_arrays["filtered_travel_stroke_pct"]),
            pl.Series("rear_velocity_stroke_pct_per_s_filtered", rear_arrays["velocity_stroke_pct_per_s_filtered"]),
            pl.Series("rear_travel_reference_used", [rear_arrays["travel_reference_used"]] * analog_df.height),
            pl.Series("rear_stroke_percent_source", [rear_arrays["stroke_percent_source"]] * analog_df.height),
        ]
    )

    if front_arrays.get("travel_mm") is not None:
        derived_df = derived_df.with_columns(
            [
                pl.Series("front_travel_mm", front_arrays["travel_mm"]),
                pl.Series("front_filtered_travel_mm", front_arrays["filtered_travel_mm"]),
                pl.Series("front_velocity_mm_per_s_filtered", front_arrays["velocity_mm_per_s_filtered"]),
            ]
        )
    if rear_arrays.get("travel_mm") is not None:
        derived_df = derived_df.with_columns(
            [
                pl.Series("rear_travel_mm", rear_arrays["travel_mm"]),
                pl.Series("rear_filtered_travel_mm", rear_arrays["filtered_travel_mm"]),
                pl.Series("rear_velocity_mm_per_s_filtered", rear_arrays["velocity_mm_per_s_filtered"]),
            ]
        )

    config["analog_resolution_bits"] = analog_resolution_bits
    if persist:
        derived_path.parent.mkdir(parents=True, exist_ok=True)
        derived_df.write_parquet(derived_path)
        config["last_derived_at"] = datetime.now(timezone.utc).isoformat()
        config["last_derived_signature"] = signature
    return derived_df, config


def channel_series(
    derived_df: pl.DataFrame,
    channel: str,
    velocity_source: str = "filtered",
    series_mode: str = "native",
) -> dict[str, Any]:
    if series_mode == "stroke_percent":
        stroke_travel_column = f"{channel}_travel_stroke_pct"
        stroke_filtered_travel_column = f"{channel}_filtered_travel_stroke_pct"
        stroke_velocity_column = f"{channel}_velocity_stroke_pct_per_s_filtered"
        if stroke_travel_column in derived_df.columns:
            return {
                "time_s": _series_to_numpy(derived_df, "host_timestamp_us", 0.0) / 1_000_000.0,
                "dt_s": _series_to_numpy(derived_df, "dt_s", 0.001),
                "raw": _series_to_numpy(derived_df, f"{channel}_raw", 0.0),
                "travel": _series_to_numpy(derived_df, stroke_travel_column, np.nan),
                "filtered_travel": _series_to_numpy(derived_df, stroke_filtered_travel_column, np.nan),
                "velocity": _series_to_numpy(derived_df, stroke_velocity_column, np.nan),
                "travel_units": "%",
                "velocity_units": "%/s",
                "travel_label": "Stroke",
                "velocity_label": "Stroke rate",
                "travel_column": stroke_travel_column,
                "velocity_column": stroke_velocity_column,
            }

    mm_travel_column = f"{channel}_travel_mm"
    mm_velocity_column = f"{channel}_velocity_mm_per_s_filtered"
    mm_velocity_csv_column = f"{channel}_velocity_mm_per_s"
    mm_filtered_travel_column = f"{channel}_filtered_travel_mm"
    counts_travel_column = f"{channel}_travel_counts"
    counts_velocity_column = f"{channel}_velocity_counts_per_s_filtered"
    counts_velocity_csv_column = f"{channel}_velocity_counts_per_s"
    counts_filtered_travel_column = f"{channel}_filtered_travel_counts"

    if mm_travel_column in derived_df.columns:
        travel_column = mm_travel_column
        filtered_travel_column = mm_filtered_travel_column
        velocity_column = mm_velocity_column
        velocity_csv_column = mm_velocity_csv_column
        travel_units = "mm"
        velocity_units = "mm/s"
        travel_label = "Travel"
        velocity_label = "Velocity"
    else:
        travel_column = counts_travel_column
        filtered_travel_column = counts_filtered_travel_column
        velocity_column = counts_velocity_column
        velocity_csv_column = counts_velocity_csv_column
        travel_units = "counts"
        velocity_units = "counts/s"
        travel_label = "Travel"
        velocity_label = "Velocity"

    if velocity_source == "csv" and velocity_csv_column in derived_df.columns:
        velocity_column = velocity_csv_column

    return {
        "time_s": _series_to_numpy(derived_df, "host_timestamp_us", 0.0) / 1_000_000.0,
        "dt_s": _series_to_numpy(derived_df, "dt_s", 0.001),
        "raw": _series_to_numpy(derived_df, f"{channel}_raw", 0.0),
        "travel": _series_to_numpy(derived_df, travel_column, 0.0),
        "filtered_travel": _series_to_numpy(derived_df, filtered_travel_column, 0.0),
        "velocity": _series_to_numpy(derived_df, velocity_column, 0.0),
        "travel_units": travel_units,
        "velocity_units": velocity_units,
        "travel_label": travel_label,
        "velocity_label": velocity_label,
        "travel_column": travel_column,
        "velocity_column": velocity_column,
    }


def compute_channel_metrics(
    derived_df: pl.DataFrame,
    channel: str,
    analog_resolution_bits: int,
    series_mode: str = "native",
) -> list[dict[str, Any]]:
    series = channel_series(derived_df, channel, series_mode=series_mode)
    travel = series["travel"]
    velocity = series["velocity"]
    dt_s = series["dt_s"]
    raw = series["raw"]
    travel_units = series["travel_units"]
    velocity_units = series["velocity_units"]

    finite_travel = travel[np.isfinite(travel)]
    finite_velocity = velocity[np.isfinite(velocity)]
    finite_dt = dt_s[np.isfinite(dt_s) & (dt_s > 0)]
    session_duration = float(np.sum(finite_dt)) if finite_dt.size else 0.0
    sample_count = int(travel.size)

    min_travel = float(np.min(finite_travel)) if finite_travel.size else None
    max_travel = float(np.max(finite_travel)) if finite_travel.size else None
    used_stroke = None if min_travel is None or max_travel is None else max_travel - min_travel

    rms_velocity = float(np.sqrt(np.mean(np.square(finite_velocity)))) if finite_velocity.size else None
    peak_compression = float(np.max(finite_velocity)) if finite_velocity.size else None
    peak_rebound = float(np.min(finite_velocity)) if finite_velocity.size else None

    time_top = None
    time_bottom = None
    if finite_travel.size and finite_dt.size:
        threshold_low = float(np.min(finite_travel)) + 0.1 * float(np.max(finite_travel) - np.min(finite_travel))
        threshold_high = float(np.max(finite_travel)) - 0.1 * float(np.max(finite_travel) - np.min(finite_travel))
        time_bottom = 100.0 * float(np.sum(dt_s[travel <= threshold_low])) / session_duration if session_duration else None
        time_top = 100.0 * float(np.sum(dt_s[travel >= threshold_high])) / session_duration if session_duration else None

    adc_max_count = (1 << analog_resolution_bits) - 1
    rail_margin = max(1, int(round(adc_max_count * 0.02)))
    rail_low = int(np.sum(raw <= rail_margin))
    rail_high = int(np.sum(raw >= adc_max_count - rail_margin))
    raw_span = float(np.max(raw) - np.min(raw)) if raw.size else None
    adc_span_used_pct = None if raw_span is None or adc_max_count <= 0 else 100.0 * raw_span / float(adc_max_count)
    implied_mm_per_count = None
    if raw_span not in (None, 0.0) and travel_units == "mm" and used_stroke is not None:
        implied_mm_per_count = used_stroke / raw_span

    occupancy_mean = None
    occupancy_median = None
    occupancy_mode = None
    occupancy_geometric_sd = None
    travel_band_metrics: list[dict[str, Any]] = []
    if session_duration > 0.0:
        finite_mask = np.isfinite(travel) & np.isfinite(dt_s) & (dt_s > 0.0)
        if np.any(finite_mask):
            normalized_travel, _ = _occupancy_distribution_values(travel, finite_mask, travel_units)

            occupancy_values = normalized_travel[finite_mask]
            occupancy_weights = dt_s[finite_mask]
            occupancy_mean = _weighted_mean(occupancy_values, occupancy_weights)
            occupancy_median = _weighted_quantile(occupancy_values, occupancy_weights, 0.5)
            occupancy_mode = _weighted_smoothed_histogram_mode(
                occupancy_values,
                occupancy_weights,
                bins=OCCUPANCY_MODE_BINS,
                histogram_range=(0.0, 100.0),
            )
            occupancy_geometric_sd = _weighted_geometric_std(occupancy_values, occupancy_weights)

            for band_start in range(0, 100, 10):
                band_end = band_start + 10
                if band_end < 100:
                    band_mask = finite_mask & (normalized_travel >= float(band_start)) & (normalized_travel < float(band_end))
                else:
                    band_mask = finite_mask & (normalized_travel >= float(band_start)) & (normalized_travel <= float(band_end))
                occupancy_pct = 100.0 * float(np.sum(dt_s[band_mask])) / session_duration
                travel_band_metrics.append(
                    {
                        "metric": f"Occupancy P{band_start}-{band_end}",
                        "value": occupancy_pct,
                        "units": "%",
                        "category": "riding",
                    }
                )
        else:
            for band_start in range(0, 100, 10):
                band_end = band_start + 10
                travel_band_metrics.append(
                    {
                        "metric": f"Occupancy P{band_start}-{band_end}",
                        "value": None,
                        "units": "%",
                        "category": "riding",
                    }
                )
    else:
        for band_start in range(0, 100, 10):
            band_end = band_start + 10
            travel_band_metrics.append(
                {
                    "metric": f"Occupancy P{band_start}-{band_end}",
                    "value": None,
                    "units": "%",
                    "category": "riding",
                }
            )

    metrics = [
        {"metric": "Session duration", "value": session_duration, "units": "s", "category": "riding"},
        {"metric": "Analog sample count", "value": sample_count, "units": "samples", "category": "hardware"},
        {
            "metric": "Median analog dt",
            "value": None if finite_dt.size == 0 else float(np.median(finite_dt) * 1_000_000.0),
            "units": "us",
            "category": "hardware",
        },
        {"metric": "Minimum travel", "value": min_travel, "units": travel_units, "category": "riding"},
        {"metric": "Maximum travel", "value": max_travel, "units": travel_units, "category": "riding"},
        {"metric": "Used stroke", "value": used_stroke, "units": travel_units, "category": "riding", "metrics_tab_visible": False},
        {"metric": "Mean occupancy", "value": occupancy_mean, "units": "%", "category": "riding"},
        {"metric": "Median occupancy", "value": occupancy_median, "units": "%", "category": "riding"},
        {"metric": "Mode occupancy", "value": occupancy_mode, "units": "%", "category": "riding"},
        {"metric": "Occupancy geometric SD", "value": occupancy_geometric_sd, "units": "x", "category": "riding"},
        *travel_band_metrics,
        {"metric": "Peak compression velocity", "value": peak_compression, "units": velocity_units, "category": "riding"},
        {"metric": "Peak rebound velocity", "value": peak_rebound, "units": velocity_units, "category": "riding"},
        {"metric": "RMS velocity", "value": rms_velocity, "units": velocity_units, "category": "riding"},
        {"metric": "Time in bottom 10%", "value": time_bottom, "units": "%", "category": "riding", "metrics_tab_visible": False},
        {"metric": "Time in top 10%", "value": time_top, "units": "%", "category": "riding", "metrics_tab_visible": False},
        {"metric": "Raw count span", "value": raw_span, "units": "counts", "category": "hardware"},
        {"metric": "ADC span used", "value": adc_span_used_pct, "units": "%", "category": "hardware"},
        {"metric": "Implied mm / count", "value": implied_mm_per_count, "units": "mm/count", "category": "hardware"},
        {"metric": "ADC rail-near low", "value": rail_low, "units": "samples", "category": "hardware"},
        {"metric": "ADC rail-near high", "value": rail_high, "units": "samples", "category": "hardware"},
    ]
    return metrics


def compute_travel_histogram(
    derived_df: pl.DataFrame,
    channel: str,
    bins: int = 80,
    series_mode: str = "native",
    histogram_range: tuple[float, float] | None = None,
    relative_occupancy: bool = False,
) -> dict[str, Any]:
    series = channel_series(derived_df, channel, series_mode=series_mode)
    travel = series["travel"]
    weights = series["dt_s"]
    finite = np.isfinite(travel) & np.isfinite(weights) & (weights > 0)
    if histogram_range is None and series_mode == "stroke_percent":
        histogram_range = (0.0, 100.0)
    hist, edges = np.histogram(travel[finite], bins=bins, range=histogram_range, weights=weights[finite])
    session_duration_s = float(np.sum(weights[np.isfinite(weights) & (weights > 0)]))
    occupancy_label = "Occupancy"
    occupancy_units = "s"
    if relative_occupancy:
        occupancy_label = "Relative occupancy"
        occupancy_units = "%"
        if session_duration_s > 0.0:
            hist = 100.0 * hist / session_duration_s
    return {
        "histogram": hist,
        "edges": edges,
        "label": series["travel_label"],
        "units": series["travel_units"],
        "occupancy_label": occupancy_label,
        "occupancy_units": occupancy_units,
        "session_duration_s": session_duration_s,
    }


def compute_velocity_histogram(
    derived_df: pl.DataFrame,
    channel: str,
    bins: int = 100,
    series_mode: str = "native",
    histogram_range: tuple[float, float] | None = None,
    relative_occupancy: bool = False,
    velocity_axis_mode: str = VELOCITY_AXIS_LINEAR,
) -> dict[str, Any]:
    series = channel_series(derived_df, channel, series_mode=series_mode)
    axis_mode = normalize_velocity_axis_mode(velocity_axis_mode)
    velocity = series["velocity"]
    weights = series["dt_s"]
    finite = np.isfinite(velocity) & np.isfinite(weights) & (weights > 0)
    display_velocity = transform_velocity_axis_values(velocity, axis_mode)
    finite = finite & np.isfinite(display_velocity)

    data_range = histogram_range
    axis_range: tuple[float, float] | None = None
    if data_range is not None:
        axis_range = velocity_axis_range_for_data(data_range[0], data_range[1], axis_mode)
    elif axis_mode == VELOCITY_AXIS_SIGNED_LOG:
        finite_velocity = velocity[finite]
        if finite_velocity.size:
            max_abs = float(np.max(np.abs(finite_velocity)))
            if not np.isfinite(max_abs) or max_abs <= 0.0:
                max_abs = 1.0
            data_range = (-max_abs, max_abs)
            axis_range = velocity_axis_range_for_data(data_range[0], data_range[1], axis_mode)

    hist_all, edges = np.histogram(display_velocity[finite], bins=bins, range=axis_range, weights=weights[finite])
    positive_mask = finite & (velocity >= 0)
    negative_mask = finite & (velocity < 0)
    hist_positive, _ = np.histogram(display_velocity[positive_mask], bins=edges, weights=weights[positive_mask])
    hist_negative, _ = np.histogram(display_velocity[negative_mask], bins=edges, weights=weights[negative_mask])
    session_duration_s = float(np.sum(weights[np.isfinite(weights) & (weights > 0)]))
    occupancy_label = "Occupancy"
    occupancy_units = "s"
    if relative_occupancy:
        occupancy_label = "Relative occupancy"
        occupancy_units = "%"
        if session_duration_s > 0.0:
            scale = 100.0 / session_duration_s
            hist_all = hist_all * scale
            hist_positive = hist_positive * scale
            hist_negative = hist_negative * scale
    if data_range is None:
        finite_velocity = velocity[finite]
        if finite_velocity.size:
            data_range = (float(np.min(finite_velocity)), float(np.max(finite_velocity)))
        else:
            data_range = (-1.0, 1.0)
    display_range = (float(edges[0]), float(edges[-1]))
    return {
        "edges": edges,
        "all": hist_all,
        "positive": hist_positive,
        "negative": hist_negative,
        "label": velocity_axis_label(series["velocity_label"], axis_mode),
        "units": series["velocity_units"],
        "occupancy_label": occupancy_label,
        "occupancy_units": occupancy_units,
        "session_duration_s": session_duration_s,
        "velocity_axis_mode": axis_mode,
        "velocity_axis_ticks": velocity_axis_ticks(data_range[0], data_range[1], axis_mode),
        "velocity_range": display_range,
        "velocity_data_range": data_range,
    }


def _percent_scale_params(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return None, None

    finite_values = values[finite]
    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))
    return minimum, maximum - minimum


def _apply_percent_scale(values: np.ndarray, minimum: float | None, span: float | None) -> np.ndarray:
    normalized = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite) or minimum is None or span is None:
        return normalized
    if span <= 0.0:
        normalized[finite] = 0.0
        return normalized

    finite_values = values[finite]
    normalized[finite] = 100.0 * (finite_values - minimum) / span
    return normalized


def _normalize_travel_percent(travel: np.ndarray) -> tuple[np.ndarray, float | None]:
    minimum, used_stroke = _percent_scale_params(travel)
    normalized = _apply_percent_scale(travel, minimum, used_stroke)
    finite = np.isfinite(travel)
    if not np.any(finite):
        return normalized, None
    return normalized, used_stroke


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    if values.size == 0 or weights.size == 0:
        return None
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return None
    return float(np.sum(values * weights) / total_weight)


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float | None:
    if values.size == 0 or weights.size == 0:
        return None
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        return None
    return float(np.sqrt(np.sum(np.square(values) * weights) / total_weight))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float | None:
    if values.size == 0 or weights.size == 0:
        return None
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return None
    sorted_indices = np.argsort(values[finite], kind="mergesort")
    sorted_values = values[finite][sorted_indices]
    sorted_weights = weights[finite][sorted_indices]
    total_weight = float(np.sum(sorted_weights))
    if total_weight <= 0.0:
        return None
    cumulative = np.cumsum(sorted_weights)
    target = float(np.clip(quantile, 0.0, 1.0)) * total_weight
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(index, sorted_values.size - 1)
    return float(sorted_values[index])


def _gaussian_kernel(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0.0:
        return np.asarray([1.0], dtype=np.float64)
    radius = max(1, int(np.ceil(4.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(x / sigma_bins))
    total = float(np.sum(kernel))
    if total <= 0.0:
        return np.asarray([1.0], dtype=np.float64)
    return kernel / total


def _weighted_smoothed_histogram_mode(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int,
    histogram_range: tuple[float, float],
    sigma_bins: float = OCCUPANCY_MODE_SMOOTH_SIGMA_BINS,
) -> float | None:
    if values.size == 0 or weights.size == 0:
        return None
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return None
    hist, edges = np.histogram(
        values[finite],
        bins=bins,
        range=histogram_range,
        weights=weights[finite],
    )
    if hist.size == 0 or float(np.sum(hist)) <= 0.0:
        return None

    kernel = _gaussian_kernel(sigma_bins)
    radius = (kernel.size - 1) // 2
    smoothed = np.convolve(np.pad(hist, (radius, radius), mode="constant"), kernel, mode="valid")
    mode_index = int(np.argmax(smoothed))
    centers = 0.5 * (edges[:-1] + edges[1:])
    if 0 < mode_index < smoothed.size - 1:
        left = float(smoothed[mode_index - 1])
        center = float(smoothed[mode_index])
        right = float(smoothed[mode_index + 1])
        denominator = left - 2.0 * center + right
        if denominator != 0.0:
            offset = 0.5 * (left - right) / denominator
            offset = float(np.clip(offset, -1.0, 1.0))
            bin_width = float(edges[1] - edges[0])
            return float(centers[mode_index] + offset * bin_width)
    return float(centers[mode_index])


def _occupancy_distribution_values(
    travel: np.ndarray,
    finite_mask: np.ndarray,
    travel_units: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    occupancy_values = np.full(travel.shape, np.nan, dtype=np.float64)
    if travel_units == "%":
        occupancy_values[finite_mask] = np.clip(travel[finite_mask], 0.0, 100.0)
        return occupancy_values, 100.0

    finite_band_travel = travel[finite_mask]
    band_min = float(np.min(finite_band_travel))
    band_max = float(np.max(finite_band_travel))
    band_span = band_max - band_min
    if band_span <= 0.0:
        occupancy_values[finite_mask] = 0.0
    else:
        occupancy_values[finite_mask] = 100.0 * (travel[finite_mask] - band_min) / band_span
    return occupancy_values, band_span


def _weighted_geometric_std(values: np.ndarray, weights: np.ndarray) -> float | None:
    if values.size == 0 or weights.size == 0:
        return None
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0) & (values > 0.0)
    if not np.any(finite):
        return None
    finite_values = values[finite]
    finite_weights = weights[finite]
    total_weight = float(np.sum(finite_weights))
    if total_weight <= 0.0:
        return None
    log_values = np.log(finite_values)
    mean_log = float(np.sum(log_values * finite_weights) / total_weight)
    variance_log = float(np.sum(np.square(log_values - mean_log) * finite_weights) / total_weight)
    return float(np.exp(np.sqrt(max(variance_log, 0.0))))


def compute_balance_analysis(derived_df: pl.DataFrame) -> dict[str, Any]:
    front_series = channel_series(derived_df, "front")
    rear_series = channel_series(derived_df, "rear")

    time_s = front_series["time_s"]
    dt_s = front_series["dt_s"]
    front_travel = front_series["travel"]
    rear_travel = rear_series["travel"]

    front_percent, front_used_stroke = _normalize_travel_percent(front_travel)
    rear_percent, rear_used_stroke = _normalize_travel_percent(rear_travel)

    finite = (
        np.isfinite(time_s)
        & np.isfinite(dt_s)
        & (dt_s > 0.0)
        & np.isfinite(front_percent)
        & np.isfinite(rear_percent)
    )
    balance_pct = front_percent - rear_percent
    combined_percent = front_percent + rear_percent
    active = finite & (combined_percent >= BALANCE_ACTIVE_THRESHOLD_PCT)

    front_share_pct = np.full(front_percent.shape, np.nan, dtype=np.float64)
    front_share_pct[active] = 100.0 * front_percent[active] / combined_percent[active]

    finite_weights = dt_s[finite]
    active_weights = dt_s[active]
    mean_balance_pct = _weighted_mean(balance_pct[finite], finite_weights)
    rms_balance_pct = _weighted_rms(balance_pct[finite], finite_weights)
    mean_front_share_pct = _weighted_mean(front_share_pct[active], active_weights)

    active_time_s = float(np.sum(active_weights)) if active_weights.size else 0.0
    front_bias = active & (balance_pct > BALANCE_NEUTRAL_BAND_PCT)
    rear_bias = active & (balance_pct < -BALANCE_NEUTRAL_BAND_PCT)
    neutral_bias = active & ~(front_bias | rear_bias)

    def _time_pct(mask: np.ndarray) -> float | None:
        if active_time_s <= 0.0:
            return None
        return 100.0 * float(np.sum(dt_s[mask])) / active_time_s

    correlation = None
    if np.count_nonzero(finite) >= 2:
        front_finite = front_percent[finite]
        rear_finite = rear_percent[finite]
        if float(np.std(front_finite)) > 0.0 and float(np.std(rear_finite)) > 0.0:
            correlation = float(np.corrcoef(front_finite, rear_finite)[0, 1])

    return {
        "time_s": time_s,
        "dt_s": dt_s,
        "front_percent": front_percent,
        "rear_percent": rear_percent,
        "balance_pct": balance_pct,
        "front_share_pct": front_share_pct,
        "active_mask": active,
        "neutral_band_pct": BALANCE_NEUTRAL_BAND_PCT,
        "active_threshold_pct": BALANCE_ACTIVE_THRESHOLD_PCT,
        "front_used_stroke": front_used_stroke,
        "rear_used_stroke": rear_used_stroke,
        "front_travel_units": front_series["travel_units"],
        "rear_travel_units": rear_series["travel_units"],
        "mean_balance_pct": mean_balance_pct,
        "rms_balance_pct": rms_balance_pct,
        "mean_front_share_pct": mean_front_share_pct,
        "correlation": correlation,
        "active_time_s": active_time_s,
        "front_bias_time_pct": _time_pct(front_bias),
        "rear_bias_time_pct": _time_pct(rear_bias),
        "neutral_time_pct": _time_pct(neutral_bias),
    }
