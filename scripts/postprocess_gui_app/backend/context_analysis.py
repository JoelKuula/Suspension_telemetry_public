from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .analysis_service import channel_series, compute_balance_analysis
from .session_config import DEFAULT_BREAKDOWN_CONFIG


WHEEL_SPEED_SOURCE = "wheel_speed"
LONGITUDINAL_ACCEL_SOURCE = "longitudinal_accel"
VERTICAL_ACCEL_SOURCE = "vertical_accel"
PITCH_RATE_SOURCE = "pitch_rate"
VERTICAL_ROUGHNESS_SOURCE = "vertical_roughness"

FRONT_TRAVEL_RESPONSE = "front_travel"
REAR_TRAVEL_RESPONSE = "rear_travel"
FRONT_VELOCITY_RESPONSE = "front_velocity"
REAR_VELOCITY_RESPONSE = "rear_velocity"
BALANCE_RESPONSE = "front_rear_balance"
FRONT_SHARE_RESPONSE = "front_share"

ACCEL_AXIS_OPTIONS = (
    ("accel_x_g", "Accel X"),
    ("accel_y_g", "Accel Y"),
    ("accel_z_g", "Accel Z"),
)

GYRO_AXIS_OPTIONS = (
    ("gyro_x_dps", "Gyro X"),
    ("gyro_y_dps", "Gyro Y"),
    ("gyro_z_dps", "Gyro Z"),
)

SOURCE_OPTIONS = (
    (WHEEL_SPEED_SOURCE, "Wheel speed"),
    (LONGITUDINAL_ACCEL_SOURCE, "Longitudinal accel"),
    (VERTICAL_ACCEL_SOURCE, "Vertical accel"),
    (PITCH_RATE_SOURCE, "Pitch rate"),
    (VERTICAL_ROUGHNESS_SOURCE, "Vertical roughness"),
)

RESPONSE_OPTIONS = (
    (FRONT_TRAVEL_RESPONSE, "Front travel"),
    (REAR_TRAVEL_RESPONSE, "Rear travel"),
    (FRONT_VELOCITY_RESPONSE, "Front velocity"),
    (REAR_VELOCITY_RESPONSE, "Rear velocity"),
    (BALANCE_RESPONSE, "Front/rear balance"),
    (FRONT_SHARE_RESPONSE, "Front share"),
)

DEFAULT_CONTEXT_CONFIG = {
    "source": LONGITUDINAL_ACCEL_SOURCE,
    "response": FRONT_VELOCITY_RESPONSE,
    "source_bins": 12,
    "response_bins": 24,
    "longitudinal_axis": "accel_x_g",
    "longitudinal_invert": False,
    "vertical_axis": "accel_z_g",
    "vertical_invert": False,
    "pitch_axis": "gyro_y_dps",
    "pitch_invert": False,
}

WHEEL_SPEED_MIN_VALID_INTERVALS = 3
WHEEL_SPEED_MIN_COVERAGE_PCT = 10.0
WHEEL_SPEED_TIMEOUT_MULTIPLIER = 1.5
VERTICAL_ROUGHNESS_WINDOW_S = 0.1
BREAKDOWN_NEAR_FULL_STROKE_THRESHOLD_PCT = 95.0
BREAKDOWN_BOTTOM_OUT_THRESHOLD_PCT = 99.0
MOTION_UNKNOWN = -1
MOTION_BRAKING = 0
MOTION_COASTING = 1
MOTION_ACCELERATING = 2
MOTION_LABELS = {
    MOTION_UNKNOWN: "unknown",
    MOTION_BRAKING: "braking",
    MOTION_COASTING: "coasting",
    MOTION_ACCELERATING: "accelerating",
}
MOTION_ORDER = {
    "braking": 0,
    "coasting": 1,
    "accelerating": 2,
    "unknown": 3,
}
ACTIVITY_SMOOTH = "smooth_low_activity"
ACTIVITY_ACTIVE = "active_roughness"
ACTIVITY_REPEATED = "repeated_hit_section"
ACTIVITY_IMPACT = "high_speed_impact_candidate"
ACTIVITY_UNKNOWN = "unknown"
ACTIVITY_ORDER = {
    ACTIVITY_SMOOTH: 0,
    ACTIVITY_ACTIVE: 1,
    ACTIVITY_REPEATED: 2,
    ACTIVITY_IMPACT: 3,
    ACTIVITY_UNKNOWN: 4,
}


def _series_to_numpy(frame: pl.DataFrame, column: str, fill_value: float = np.nan) -> np.ndarray:
    if column not in frame.columns:
        return np.asarray([], dtype=np.float64)
    values = frame.get_column(column).fill_null(fill_value).to_numpy()
    return np.asarray(values, dtype=np.float64)


def _effective_imu_time_s(imu_frame_df: pl.DataFrame) -> np.ndarray:
    estimated = _series_to_numpy(imu_frame_df, "estimated_host_time_s", np.nan)
    burst = _series_to_numpy(imu_frame_df, "burst_host_time_s", np.nan)
    if estimated.size == 0 and burst.size == 0:
        return np.asarray([], dtype=np.float64)
    if estimated.size == 0:
        return burst
    if burst.size == 0:
        return estimated
    return np.where(np.isfinite(estimated), estimated, burst)


def _collapse_duplicate_times(time_s: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if time_s.size == 0:
        return time_s, values
    order = np.argsort(time_s, kind="stable")
    sorted_time = time_s[order]
    sorted_values = values[order]
    unique_time, inverse = np.unique(sorted_time, return_inverse=True)
    if unique_time.size == sorted_time.size:
        return sorted_time, sorted_values
    sums = np.bincount(inverse, weights=sorted_values)
    counts = np.bincount(inverse)
    return unique_time, sums / counts


def _interpolate_imu_series(analog_time_s: np.ndarray, imu_frame_df: pl.DataFrame, column: str) -> np.ndarray:
    imu_time_s = _effective_imu_time_s(imu_frame_df)
    values = _series_to_numpy(imu_frame_df, column, np.nan)
    if analog_time_s.size == 0 or imu_time_s.size == 0 or values.size == 0:
        return np.full(analog_time_s.shape, np.nan, dtype=np.float64)

    finite = np.isfinite(imu_time_s) & np.isfinite(values)
    if not np.any(finite):
        return np.full(analog_time_s.shape, np.nan, dtype=np.float64)

    time_s, collapsed_values = _collapse_duplicate_times(imu_time_s[finite], values[finite])
    if time_s.size == 1:
        result = np.full(analog_time_s.shape, np.nan, dtype=np.float64)
        result[np.isclose(analog_time_s, time_s[0])] = collapsed_values[0]
        return result

    return np.interp(
        analog_time_s,
        time_s,
        collapsed_values,
        left=np.nan,
        right=np.nan,
    )


def _moving_average_nan(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64)
    finite = np.isfinite(values)
    weighted_values = np.where(finite, values, 0.0)
    weights = finite.astype(np.float64)
    padded_values = np.pad(weighted_values, (window // 2, window // 2), mode="edge")
    padded_weights = np.pad(weights, (window // 2, window // 2), mode="edge")
    summed_values = np.convolve(padded_values, kernel, mode="valid")
    summed_weights = np.convolve(padded_weights, kernel, mode="valid")
    result = np.full(values.shape, np.nan, dtype=np.float64)
    nonzero = summed_weights > 0.0
    result[nonzero] = summed_values[nonzero] / summed_weights[nonzero]
    return result


def _roughness_window_samples(time_s: np.ndarray) -> int:
    if time_s.size < 2:
        return 1
    diffs = np.diff(time_s)
    positive = diffs[diffs > 0.0]
    if positive.size == 0:
        return 1
    window = max(1, int(round(VERTICAL_ROUGHNESS_WINDOW_S / float(np.median(positive)))))
    if window % 2 == 0:
        window += 1
    return window


def _compute_vertical_roughness(vertical_accel: np.ndarray, analog_time_s: np.ndarray) -> np.ndarray:
    if vertical_accel.size == 0 or analog_time_s.size == 0:
        return np.asarray([], dtype=np.float64)
    window = _roughness_window_samples(analog_time_s)
    trend = _moving_average_nan(vertical_accel, window)
    detrended = vertical_accel - trend
    return np.sqrt(_moving_average_nan(np.square(detrended), window))


def _float_array(frame: pl.DataFrame, column: str) -> np.ndarray:
    return _series_to_numpy(frame, column, np.nan)


def _build_speed_context(
    analog_time_s: np.ndarray,
    dt_s: np.ndarray,
    wheel_df: pl.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    speed = np.full(analog_time_s.shape, np.nan, dtype=np.float64)
    measured_speed = np.zeros(analog_time_s.shape, dtype=bool)
    pulse_time_s = _float_array(wheel_df, "host_time_s")
    speed_kph = _float_array(wheel_df, "speed_kph")
    period_s = _float_array(wheel_df, "period_s")
    valid_intervals = 0

    finite_pulse_times = pulse_time_s[np.isfinite(pulse_time_s)]
    if finite_pulse_times.size:
        first_pulse_time = float(finite_pulse_times[0])
        if first_pulse_time > 0.0:
            speed[analog_time_s < first_pulse_time] = 0.0

    if pulse_time_s.size:
        for index, pulse_time in enumerate(pulse_time_s):
            if not np.isfinite(pulse_time):
                continue
            current_speed = speed_kph[index] if index < speed_kph.size else np.nan
            current_period = period_s[index] if index < period_s.size else np.nan
            if not (np.isfinite(current_speed) and np.isfinite(current_period) and current_period > 0.0):
                continue
            valid_intervals += 1
            next_pulse_time = np.inf
            if index + 1 < pulse_time_s.size and np.isfinite(pulse_time_s[index + 1]):
                next_pulse_time = pulse_time_s[index + 1]
            timeout_time = pulse_time + WHEEL_SPEED_TIMEOUT_MULTIPLIER * current_period
            end_time = min(next_pulse_time, timeout_time)
            if end_time <= pulse_time:
                continue
            mask = (analog_time_s >= pulse_time) & (analog_time_s < end_time)
            speed[mask] = current_speed
            measured_speed[mask] = True

    measured_intervals = measured_speed & np.isfinite(dt_s) & (dt_s > 0.0)
    coverage_s = float(np.sum(dt_s[measured_intervals])) if measured_intervals.size else 0.0
    total_s = float(np.sum(dt_s[np.isfinite(dt_s) & (dt_s > 0.0)])) if dt_s.size else 0.0
    coverage_pct = 100.0 * coverage_s / total_s if total_s > 0.0 else 0.0
    usable = valid_intervals >= WHEEL_SPEED_MIN_VALID_INTERVALS and coverage_pct >= WHEEL_SPEED_MIN_COVERAGE_PCT

    warning = None
    if not usable:
        warning = (
            "Wheel speed coverage is weak for this session. "
            f"Valid intervals: {valid_intervals}, held-speed coverage: {coverage_pct:.1f}%."
        )

    return speed, {
        "valid_intervals": valid_intervals,
        "coverage_s": coverage_s,
        "coverage_pct": coverage_pct,
        "usable": usable,
        "warning": warning,
    }


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
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    total_weight = float(np.sum(sorted_weights))
    if total_weight <= 0.0:
        return None
    cumulative = np.cumsum(sorted_weights)
    target = quantile * total_weight
    return float(np.interp(target, cumulative, sorted_values))


def build_context_dataset(
    derived_df: pl.DataFrame,
    wheel_df: pl.DataFrame,
    imu_frame_df: pl.DataFrame,
    context_config: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    analog_time_s = _series_to_numpy(derived_df, "host_timestamp_us", 0.0) / 1_000_000.0
    dt_s = _series_to_numpy(derived_df, "dt_s", np.nan)

    front_series = channel_series(derived_df, "front")
    rear_series = channel_series(derived_df, "rear")
    balance = compute_balance_analysis(derived_df)

    speed_kph_ctx, wheel_meta = _build_speed_context(analog_time_s, dt_s, wheel_df)

    longitudinal = _interpolate_imu_series(analog_time_s, imu_frame_df, str(context_config["longitudinal_axis"]))
    if context_config.get("longitudinal_invert"):
        longitudinal = -longitudinal

    vertical = _interpolate_imu_series(analog_time_s, imu_frame_df, str(context_config["vertical_axis"]))
    if context_config.get("vertical_invert"):
        vertical = -vertical

    pitch_rate = _interpolate_imu_series(analog_time_s, imu_frame_df, str(context_config["pitch_axis"]))
    if context_config.get("pitch_invert"):
        pitch_rate = -pitch_rate

    vertical_roughness = _compute_vertical_roughness(vertical, analog_time_s)

    context_df = pl.DataFrame(
        [
            pl.Series("time_s", analog_time_s),
            pl.Series("dt_s", dt_s),
            pl.Series("speed_kph_ctx", speed_kph_ctx),
            pl.Series("longitudinal_accel_g_ctx", longitudinal),
            pl.Series("vertical_accel_g_ctx", vertical),
            pl.Series("pitch_rate_dps_ctx", pitch_rate),
            pl.Series("vertical_roughness_g_ctx", vertical_roughness),
            pl.Series("front_travel_resp", front_series["travel"]),
            pl.Series("rear_travel_resp", rear_series["travel"]),
            pl.Series("front_velocity_resp", front_series["velocity"]),
            pl.Series("rear_velocity_resp", rear_series["velocity"]),
            pl.Series("front_rear_balance_resp", np.asarray(balance["balance_pct"], dtype=np.float64)),
            pl.Series("front_share_resp", np.asarray(balance["front_share_pct"], dtype=np.float64)),
        ]
    )

    meta = {
        "sources": {
            WHEEL_SPEED_SOURCE: {
                "label": "Wheel speed",
                "units": "km/h",
                "column": "speed_kph_ctx",
                "usable": wheel_meta["usable"],
                "warning": wheel_meta["warning"],
            },
            LONGITUDINAL_ACCEL_SOURCE: {
                "label": "Longitudinal accel",
                "units": "g",
                "column": "longitudinal_accel_g_ctx",
            },
            VERTICAL_ACCEL_SOURCE: {
                "label": "Vertical accel",
                "units": "g",
                "column": "vertical_accel_g_ctx",
            },
            PITCH_RATE_SOURCE: {
                "label": "Pitch rate",
                "units": "dps",
                "column": "pitch_rate_dps_ctx",
            },
            VERTICAL_ROUGHNESS_SOURCE: {
                "label": "Vertical roughness",
                "units": "g",
                "column": "vertical_roughness_g_ctx",
            },
        },
        "responses": {
            FRONT_TRAVEL_RESPONSE: {
                "label": "Front travel",
                "units": front_series["travel_units"],
                "column": "front_travel_resp",
            },
            REAR_TRAVEL_RESPONSE: {
                "label": "Rear travel",
                "units": rear_series["travel_units"],
                "column": "rear_travel_resp",
            },
            FRONT_VELOCITY_RESPONSE: {
                "label": "Front velocity",
                "units": front_series["velocity_units"],
                "column": "front_velocity_resp",
            },
            REAR_VELOCITY_RESPONSE: {
                "label": "Rear velocity",
                "units": rear_series["velocity_units"],
                "column": "rear_velocity_resp",
            },
            BALANCE_RESPONSE: {
                "label": "Front/rear balance",
                "units": "p.p.",
                "column": "front_rear_balance_resp",
            },
            FRONT_SHARE_RESPONSE: {
                "label": "Front share",
                "units": "%",
                "column": "front_share_resp",
            },
        },
        "wheel_speed": wheel_meta,
    }
    return context_df, meta


def compute_context_heatmap(
    context_df: pl.DataFrame,
    meta: dict[str, Any],
    source: str,
    response: str,
    source_bins: int,
    response_bins: int,
) -> dict[str, Any]:
    source_info = meta["sources"][source]
    response_info = meta["responses"][response]
    source_values = _series_to_numpy(context_df, source_info["column"], np.nan)
    response_values = _series_to_numpy(context_df, response_info["column"], np.nan)
    dt_s = _series_to_numpy(context_df, "dt_s", np.nan)

    finite = (
        np.isfinite(source_values)
        & np.isfinite(response_values)
        & np.isfinite(dt_s)
        & (dt_s > 0.0)
    )
    if not np.any(finite):
        return {
            "histogram": np.zeros((1, 1), dtype=np.float64),
            "source_range": (0.0, 1.0),
            "response_range": (0.0, 1.0),
            "source_units": source_info["units"],
            "response_units": response_info["units"],
            "source_label": source_info["label"],
            "response_label": response_info["label"],
            "has_data": False,
        }

    source_finite = source_values[finite]
    response_finite = response_values[finite]
    weights = dt_s[finite]
    source_min = float(np.min(source_finite))
    source_max = float(np.max(source_finite))
    response_min = float(np.min(response_finite))
    response_max = float(np.max(response_finite))

    if source_min == source_max:
        source_min -= 0.5
        source_max += 0.5
    if response_min == response_max:
        response_min -= 0.5
        response_max += 0.5

    histogram, _, _ = np.histogram2d(
        source_finite,
        response_finite,
        bins=(source_bins, response_bins),
        range=((source_min, source_max), (response_min, response_max)),
        weights=weights,
    )
    return {
        "histogram": histogram.T,
        "source_range": (source_min, source_max),
        "response_range": (response_min, response_max),
        "source_units": source_info["units"],
        "response_units": response_info["units"],
        "source_label": source_info["label"],
        "response_label": response_info["label"],
        "has_data": True,
    }


def summarize_context_bins(
    context_df: pl.DataFrame,
    meta: dict[str, Any],
    source: str,
    response: str,
    bin_count: int,
) -> list[dict[str, Any]]:
    source_info = meta["sources"][source]
    response_info = meta["responses"][response]
    source_values = _series_to_numpy(context_df, source_info["column"], np.nan)
    response_values = _series_to_numpy(context_df, response_info["column"], np.nan)
    dt_s = _series_to_numpy(context_df, "dt_s", np.nan)

    finite = (
        np.isfinite(source_values)
        & np.isfinite(response_values)
        & np.isfinite(dt_s)
        & (dt_s > 0.0)
    )
    if not np.any(finite):
        return []

    source_finite = source_values[finite]
    response_finite = response_values[finite]
    weights = dt_s[finite]
    source_min = float(np.min(source_finite))
    source_max = float(np.max(source_finite))

    if source_min == source_max:
        values = response_finite
        row = {
            "source_min": source_min,
            "source_max": source_max,
            "source_center": source_min,
            "occupancy_s": float(np.sum(weights)),
            "mean": _weighted_mean(values, weights),
            "rms": _weighted_rms(values, weights),
            "p10": _weighted_quantile(values, weights, 0.10),
            "p50": _weighted_quantile(values, weights, 0.50),
            "p90": _weighted_quantile(values, weights, 0.90),
        }
        return [row]

    edges = np.linspace(source_min, source_max, max(1, int(bin_count)) + 1)
    rows: list[dict[str, Any]] = []
    for index in range(edges.size - 1):
        low = edges[index]
        high = edges[index + 1]
        if index == edges.size - 2:
            mask = finite & (source_values >= low) & (source_values <= high)
        else:
            mask = finite & (source_values >= low) & (source_values < high)
        if not np.any(mask):
            continue
        bin_values = response_values[mask]
        bin_weights = dt_s[mask]
        rows.append(
            {
                "source_min": float(low),
                "source_max": float(high),
                "source_center": float(0.5 * (low + high)),
                "occupancy_s": float(np.sum(bin_weights)),
                "mean": _weighted_mean(bin_values, bin_weights),
                "rms": _weighted_rms(bin_values, bin_weights),
                "p10": _weighted_quantile(bin_values, bin_weights, 0.10),
                "p50": _weighted_quantile(bin_values, bin_weights, 0.50),
                "p90": _weighted_quantile(bin_values, bin_weights, 0.90),
            }
        )
    return rows


def normalize_breakdown_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_BREAKDOWN_CONFIG)
    if config:
        merged.update(config)

    raw_edges = merged.get("speed_band_edges_kph", DEFAULT_BREAKDOWN_CONFIG["speed_band_edges_kph"])
    if not isinstance(raw_edges, list):
        raw_edges = list(DEFAULT_BREAKDOWN_CONFIG["speed_band_edges_kph"])
    edges = sorted({float(value) for value in raw_edges if float(value) > 0.0})
    if not edges:
        edges = list(DEFAULT_BREAKDOWN_CONFIG["speed_band_edges_kph"])
    merged["speed_band_edges_kph"] = edges
    merged["motion_smoothing_ms"] = max(50, int(merged.get("motion_smoothing_ms", 500)))
    merged["braking_threshold_mps2"] = max(0.05, float(merged.get("braking_threshold_mps2", 1.0)))
    merged["accel_threshold_mps2"] = max(0.05, float(merged.get("accel_threshold_mps2", 1.0)))
    merged["min_segment_ms"] = max(100, int(merged.get("min_segment_ms", 750)))
    raw_stroke_thresholds = merged.get(
        "stroke_warning_thresholds_pct",
        DEFAULT_BREAKDOWN_CONFIG["stroke_warning_thresholds_pct"],
    )
    if not isinstance(raw_stroke_thresholds, list):
        raw_stroke_thresholds = list(DEFAULT_BREAKDOWN_CONFIG["stroke_warning_thresholds_pct"])
    stroke_thresholds = sorted(
        {
            min(max(float(value), 0.0), 100.0)
            for value in raw_stroke_thresholds
            if np.isfinite(float(value))
        }
    )
    if not stroke_thresholds:
        stroke_thresholds = list(DEFAULT_BREAKDOWN_CONFIG["stroke_warning_thresholds_pct"])
    merged["stroke_warning_thresholds_pct"] = stroke_thresholds
    merged["near_bottom_threshold_pct"] = min(
        max(float(merged.get("near_bottom_threshold_pct", 95.0)), 0.0),
        100.0,
    )
    merged["bottom_out_threshold_pct"] = min(
        max(float(merged.get("bottom_out_threshold_pct", 99.0)), 0.0),
        100.0,
    )
    merged["active_activity_threshold"] = max(0.05, float(merged.get("active_activity_threshold", 0.75)))
    merged["repeated_hit_min_speed_kph"] = max(0.0, float(merged.get("repeated_hit_min_speed_kph", 25.0)))
    merged["repeated_hit_activity_threshold"] = max(
        0.1, float(merged.get("repeated_hit_activity_threshold", 1.6))
    )
    merged["repeated_hit_min_duration_ms"] = max(
        100, int(merged.get("repeated_hit_min_duration_ms", 400))
    )
    merged["high_speed_impact_min_speed_kph"] = max(
        0.0, float(merged.get("high_speed_impact_min_speed_kph", 25.0))
    )
    merged["high_compression_velocity_percentile"] = min(
        max(float(merged.get("high_compression_velocity_percentile", 99.0)), 50.0),
        100.0,
    )
    merged["high_compression_velocity_min_pct_s"] = max(
        0.0, float(merged.get("high_compression_velocity_min_pct_s", 300.0))
    )
    merged["pack_down_window_ms"] = max(100, int(merged.get("pack_down_window_ms", 600)))
    merged["pack_down_shift_threshold_pct"] = max(
        0.1, float(merged.get("pack_down_shift_threshold_pct", 8.0))
    )
    return merged


def _window_samples(time_s: np.ndarray, duration_ms: int) -> int:
    if time_s.size < 2:
        return 1
    diffs = np.diff(time_s)
    positive = diffs[diffs > 0.0]
    if positive.size == 0:
        return 1
    window = max(1, int(round((float(duration_ms) / 1000.0) / float(np.median(positive)))))
    if window % 2 == 0:
        window += 1
    return window


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    if window <= 1:
        return np.asarray(values, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64)
    padded = np.pad(np.asarray(values, dtype=np.float64), (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _interpolate_series(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(time_s) & np.isfinite(values)
    if np.count_nonzero(finite) == 0:
        return result
    if np.count_nonzero(finite) == 1:
        result[finite] = values[finite]
        return result
    finite_time = time_s[finite]
    finite_values = values[finite]
    interpolated = np.interp(time_s, finite_time, finite_values)
    start = float(finite_time[0])
    end = float(finite_time[-1])
    valid = (time_s >= start) & (time_s <= end)
    result[valid] = interpolated[valid]
    return result


def _gradient_nan(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    gradient = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(time_s)
    if np.count_nonzero(finite) < 2:
        return gradient
    finite_values = values[finite]
    finite_time = time_s[finite]
    gradient[finite] = np.gradient(finite_values, finite_time)
    return gradient


def _merge_short_state_runs(codes: np.ndarray, valid: np.ndarray, min_samples: int) -> np.ndarray:
    if codes.size == 0 or min_samples <= 1:
        return codes
    result = codes.copy()
    index = 0
    while index < result.size:
        if not valid[index]:
            index += 1
            continue
        end = index + 1
        while end < result.size and valid[end] and result[end] == result[index]:
            end += 1
        if end - index < min_samples:
            left = result[index - 1] if index > 0 and valid[index - 1] else None
            right = result[end] if end < result.size and valid[end] else None
            if left is not None and right is not None and left == right:
                replacement = left
            elif left is not None:
                replacement = left
            elif right is not None:
                replacement = right
            else:
                replacement = MOTION_COASTING
            result[index:end] = replacement
        index = end
    return result


def _sign_change_mask(values: np.ndarray) -> np.ndarray:
    mask = np.zeros(values.shape, dtype=bool)
    if values.size < 2:
        return mask
    finite = np.isfinite(values)
    sign_change = finite[1:] & finite[:-1] & ((values[1:] * values[:-1]) < 0.0)
    mask[1:] = sign_change
    return mask


def _threshold_event_summary(
    travel: np.ndarray,
    dt_s: np.ndarray,
    threshold_pct: float,
    *,
    absolute_threshold: bool = False,
) -> dict[str, Any]:
    finite = np.isfinite(travel) & np.isfinite(dt_s) & (dt_s > 0.0)
    if not np.any(finite):
        return {"event_count": 0, "time_s": 0.0}
    if absolute_threshold:
        threshold_value = float(threshold_pct)
    else:
        finite_travel = travel[finite]
        minimum = float(np.min(finite_travel))
        maximum = float(np.max(finite_travel))
        used_stroke = maximum - minimum
        if used_stroke <= 0.0:
            return {"event_count": 0, "time_s": 0.0}
        threshold_value = minimum + used_stroke * (float(threshold_pct) / 100.0)
    active = finite & (travel >= threshold_value)
    event_count = 0
    if active.size:
        event_count = int(active[0]) + int(np.count_nonzero(active[1:] & ~active[:-1]))
    return {
        "event_count": event_count,
        "time_s": float(np.sum(dt_s[active])) if np.any(active) else 0.0,
    }


def _speed_band_label(edges: list[float], band_id: int) -> str:
    lower_bounds = [0.0] + list(edges)
    upper_bounds = list(edges) + [None]
    if band_id < 0 or band_id >= len(lower_bounds):
        return "unknown"
    lower = lower_bounds[band_id]
    upper = upper_bounds[band_id]
    if upper is None:
        return f"{int(lower)}+ km/h"
    return f"{int(lower)}-{int(upper)} km/h"


def _segment_ids_from_runs(
    speed_band_id: np.ndarray,
    motion_codes: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    segment_ids = np.full(speed_band_id.shape, -1, dtype=np.int64)
    if speed_band_id.size == 0:
        return segment_ids
    current_id = 0
    index = 0
    while index < speed_band_id.size:
        if not valid[index]:
            index += 1
            continue
        end = index + 1
        while (
            end < speed_band_id.size
            and valid[end]
            and speed_band_id[end] == speed_band_id[index]
            and motion_codes[end] == motion_codes[index]
        ):
            end += 1
        segment_ids[index:end] = current_id
        current_id += 1
        index = end
    return segment_ids


def _keep_long_true_runs(mask: np.ndarray, min_samples: int) -> np.ndarray:
    result = np.zeros(mask.shape, dtype=bool)
    if mask.size == 0:
        return result
    index = 0
    while index < mask.size:
        if not mask[index]:
            index += 1
            continue
        end = index + 1
        while end < mask.size and mask[end]:
            end += 1
        if end - index >= min_samples:
            result[index:end] = True
        index = end
    return result


def _stroke_percent_source(derived_df: pl.DataFrame, channel: str) -> str:
    source_column = f"{channel}_stroke_percent_source"
    if source_column in derived_df.columns and derived_df.height:
        values = [str(value) for value in derived_df.get_column(source_column).drop_nulls().unique().to_list()]
        if values:
            return values[0]
    if f"{channel}_travel_stroke_pct" in derived_df.columns:
        return "unknown"
    return "unavailable"


def build_breakdown_dataset(
    derived_df: pl.DataFrame,
    wheel_df: pl.DataFrame,
    imu_frame_df: pl.DataFrame,
    breakdown_config: dict[str, Any] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    config = normalize_breakdown_config(breakdown_config)
    context_df, meta = build_context_dataset(
        derived_df=derived_df,
        wheel_df=wheel_df,
        imu_frame_df=imu_frame_df,
        context_config=DEFAULT_CONTEXT_CONFIG,
    )
    front_series = channel_series(derived_df, "front", series_mode="stroke_percent")
    rear_series = channel_series(derived_df, "rear", series_mode="stroke_percent")
    context_df = context_df.with_columns(
        [
            pl.Series("front_travel_resp", front_series["travel"]),
            pl.Series("rear_travel_resp", rear_series["travel"]),
            pl.Series("front_velocity_resp", front_series["velocity"]),
            pl.Series("rear_velocity_resp", rear_series["velocity"]),
        ]
    )
    meta["responses"][FRONT_TRAVEL_RESPONSE]["units"] = front_series["travel_units"]
    meta["responses"][FRONT_TRAVEL_RESPONSE]["column"] = "front_travel_resp"
    meta["responses"][REAR_TRAVEL_RESPONSE]["units"] = rear_series["travel_units"]
    meta["responses"][REAR_TRAVEL_RESPONSE]["column"] = "rear_travel_resp"
    meta["responses"][FRONT_VELOCITY_RESPONSE]["units"] = front_series["velocity_units"]
    meta["responses"][FRONT_VELOCITY_RESPONSE]["column"] = "front_velocity_resp"
    meta["responses"][REAR_VELOCITY_RESPONSE]["units"] = rear_series["velocity_units"]
    meta["responses"][REAR_VELOCITY_RESPONSE]["column"] = "rear_velocity_resp"
    meta["stroke_percent"] = {
        "front": {
            "usable": front_series["travel_units"] == "%",
            "source": _stroke_percent_source(derived_df, "front"),
        },
        "rear": {
            "usable": rear_series["travel_units"] == "%",
            "source": _stroke_percent_source(derived_df, "rear"),
        },
    }

    time_s = _series_to_numpy(context_df, "time_s", np.nan)
    dt_s = _series_to_numpy(context_df, "dt_s", np.nan)
    speed_kph = _series_to_numpy(context_df, "speed_kph_ctx", np.nan)
    speed_mps = speed_kph / 3.6
    interpolated_speed_mps = _interpolate_series(time_s, speed_mps)
    raw_accel = _gradient_nan(interpolated_speed_mps, time_s)
    motion_window = _window_samples(time_s, config["motion_smoothing_ms"])
    smooth_accel = _moving_average_nan(raw_accel, motion_window)
    finite_speed = np.isfinite(speed_kph)

    motion_codes = np.full(speed_kph.shape, MOTION_UNKNOWN, dtype=np.int64)
    motion_valid = finite_speed & np.isfinite(smooth_accel)
    motion_codes[motion_valid] = MOTION_COASTING
    motion_codes[motion_valid & (smooth_accel <= -config["braking_threshold_mps2"])] = MOTION_BRAKING
    motion_codes[motion_valid & (smooth_accel >= config["accel_threshold_mps2"])] = MOTION_ACCELERATING
    motion_codes = _merge_short_state_runs(
        motion_codes,
        motion_valid,
        _window_samples(time_s, config["min_segment_ms"]),
    )

    speed_band_id = np.full(speed_kph.shape, -1, dtype=np.int64)
    speed_band_id[finite_speed] = np.digitize(speed_kph[finite_speed], config["speed_band_edges_kph"], right=False)
    speed_band_label = np.asarray(
        [_speed_band_label(config["speed_band_edges_kph"], int(value)) for value in speed_band_id],
        dtype=object,
    )
    motion_state = np.asarray([MOTION_LABELS[int(code)] for code in motion_codes], dtype=object)

    front_velocity = _series_to_numpy(context_df, "front_velocity_resp", np.nan)
    rear_velocity = _series_to_numpy(context_df, "rear_velocity_resp", np.nan)
    front_scale = float(np.percentile(np.abs(front_velocity[np.isfinite(front_velocity)]), 90.0)) if np.isfinite(front_velocity).any() else 1.0
    rear_scale = float(np.percentile(np.abs(rear_velocity[np.isfinite(rear_velocity)]), 90.0)) if np.isfinite(rear_velocity).any() else 1.0
    if front_scale <= 0.0:
        front_scale = 1.0
    if rear_scale <= 0.0:
        rear_scale = 1.0
    velocity_energy = np.sqrt(np.square(np.nan_to_num(front_velocity / front_scale)) + np.square(np.nan_to_num(rear_velocity / rear_scale)))
    sign_changes = _sign_change_mask(front_velocity) | _sign_change_mask(rear_velocity)
    repeated_window = _window_samples(time_s, config["repeated_hit_min_duration_ms"])
    oscillation_density = _rolling_sum(sign_changes.astype(np.float64), repeated_window) / max(1.0, repeated_window / 4.0)
    suspension_activity_score = velocity_energy * (1.0 + 0.5 * np.clip(oscillation_density, 0.0, 2.0))
    repeated_raw = (
        finite_speed
        & (speed_kph >= config["repeated_hit_min_speed_kph"])
        & np.isfinite(suspension_activity_score)
        & (suspension_activity_score >= config["repeated_hit_activity_threshold"])
        & (oscillation_density >= 0.25)
    )
    repeated_hit_proxy = _keep_long_true_runs(repeated_raw, repeated_window)

    positive_compression = np.concatenate(
        [
            front_velocity[np.isfinite(front_velocity) & (front_velocity > 0.0)],
            rear_velocity[np.isfinite(rear_velocity) & (rear_velocity > 0.0)],
        ]
    )
    if positive_compression.size:
        high_compression_threshold = max(
            float(np.percentile(positive_compression, config["high_compression_velocity_percentile"])),
            float(config["high_compression_velocity_min_pct_s"]),
        )
    else:
        high_compression_threshold = float(config["high_compression_velocity_min_pct_s"])
    high_compression_candidate = (
        finite_speed
        & (speed_kph >= config["high_speed_impact_min_speed_kph"])
        & (
            (np.isfinite(front_velocity) & (front_velocity >= high_compression_threshold))
            | (np.isfinite(rear_velocity) & (rear_velocity >= high_compression_threshold))
        )
    )
    activity_state = np.full(speed_kph.shape, ACTIVITY_UNKNOWN, dtype=object)
    finite_activity = np.isfinite(suspension_activity_score)
    activity_state[finite_activity] = ACTIVITY_SMOOTH
    activity_state[
        finite_activity & (suspension_activity_score >= config["active_activity_threshold"])
    ] = ACTIVITY_ACTIVE
    activity_state[repeated_hit_proxy] = ACTIVITY_REPEATED
    activity_state[high_compression_candidate] = ACTIVITY_IMPACT

    segment_valid = finite_speed & np.isfinite(smooth_accel) & (speed_band_id >= 0) & (motion_codes != MOTION_UNKNOWN)
    segment_ids = _segment_ids_from_runs(speed_band_id, motion_codes, segment_valid)

    context_df = context_df.with_columns(
        [
            pl.Series("speed_mps_ctx", speed_mps),
            pl.Series("speed_accel_mps2_ctx", smooth_accel),
            pl.Series("speed_band_id", speed_band_id),
            pl.Series("speed_band_label", speed_band_label),
            pl.Series("motion_state", motion_state),
            pl.Series("suspension_activity_score", suspension_activity_score),
            pl.Series("repeated_hit_proxy", repeated_hit_proxy),
            pl.Series("high_compression_velocity_candidate", high_compression_candidate),
            pl.Series("activity_state", activity_state),
            pl.Series("segment_id", segment_ids),
        ]
    )
    meta["breakdown_config"] = config
    meta["speed_bands"] = {
        "edges_kph": list(config["speed_band_edges_kph"]),
        "labels": [_speed_band_label(config["speed_band_edges_kph"], index) for index in range(len(config["speed_band_edges_kph"]) + 1)],
    }
    meta["breakdown_thresholds"] = {
        "high_compression_velocity_pct_s": high_compression_threshold,
        "activity_front_velocity_scale": front_scale,
        "activity_rear_velocity_scale": rear_scale,
    }
    return context_df, meta


def _selection_mask_from_time(context_df: pl.DataFrame, start_time_s: float, end_time_s: float) -> np.ndarray:
    time_s = _series_to_numpy(context_df, "time_s", np.nan)
    low = min(float(start_time_s), float(end_time_s))
    high = max(float(start_time_s), float(end_time_s))
    return np.isfinite(time_s) & (time_s >= low) & (time_s <= high)


def summarize_breakdown_selection(
    context_df: pl.DataFrame,
    meta: dict[str, Any],
    selection_mask: np.ndarray,
    label: str,
) -> dict[str, Any]:
    dt_s = _series_to_numpy(context_df, "dt_s", np.nan)
    time_s = _series_to_numpy(context_df, "time_s", np.nan)
    valid = selection_mask & np.isfinite(dt_s) & (dt_s > 0.0)
    if not np.any(valid):
        return {
            "label": label,
            "has_data": False,
            "duration_s": 0.0,
            "start_time_s": None,
            "end_time_s": None,
        }

    duration_s = float(np.sum(dt_s[valid]))
    speed_kph = _series_to_numpy(context_df, "speed_kph_ctx", np.nan)
    speed_accel = _series_to_numpy(context_df, "speed_accel_mps2_ctx", np.nan)
    front_travel = _series_to_numpy(context_df, "front_travel_resp", np.nan)
    rear_travel = _series_to_numpy(context_df, "rear_travel_resp", np.nan)
    front_velocity = _series_to_numpy(context_df, "front_velocity_resp", np.nan)
    rear_velocity = _series_to_numpy(context_df, "rear_velocity_resp", np.nan)
    balance = _series_to_numpy(context_df, "front_rear_balance_resp", np.nan)
    front_share = _series_to_numpy(context_df, "front_share_resp", np.nan)
    repeated_hit = np.asarray(
        context_df.get_column("repeated_hit_proxy").fill_null(False).to_numpy(),
        dtype=bool,
    )
    config = normalize_breakdown_config(meta.get("breakdown_config"))

    front_valid = valid & np.isfinite(front_travel)
    rear_valid = valid & np.isfinite(rear_travel)
    front_velocity_valid = valid & np.isfinite(front_velocity)
    rear_velocity_valid = valid & np.isfinite(rear_velocity)
    speed_valid = valid & np.isfinite(speed_kph)
    accel_valid = valid & np.isfinite(speed_accel)
    balance_valid = valid & np.isfinite(balance)
    share_valid = valid & np.isfinite(front_share)

    def _used_stroke(values: np.ndarray, mask: np.ndarray) -> float | None:
        if not np.any(mask):
            return None
        finite_values = values[mask]
        return float(np.max(finite_values) - np.min(finite_values))

    def _top_10_pct(values: np.ndarray, mask: np.ndarray) -> float | None:
        if not np.any(mask):
            return None
        finite_values = values[mask]
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
        span = maximum - minimum
        if span <= 0.0:
            return 0.0
        threshold = maximum - 0.1 * span
        time_in = float(np.sum(dt_s[mask & (values >= threshold)]))
        return 100.0 * time_in / duration_s if duration_s > 0.0 else None

    def _stroke_quantile(values: np.ndarray, mask: np.ndarray, quantile: float) -> float | None:
        if not np.any(mask):
            return None
        return _weighted_quantile(values[mask], dt_s[mask], quantile)

    def _stroke_max(values: np.ndarray, mask: np.ndarray) -> float | None:
        if not np.any(mask):
            return None
        return float(np.max(values[mask]))

    def _time_above(values: np.ndarray, mask: np.ndarray, threshold_pct: float) -> float | None:
        if duration_s <= 0.0 or not np.any(mask):
            return None
        return 100.0 * float(np.sum(dt_s[mask & (values >= threshold_pct)])) / duration_s

    sign_changes = _sign_change_mask(front_velocity) | _sign_change_mask(rear_velocity)
    oscillation_count = int(np.count_nonzero(sign_changes & valid))
    mean_cycle_rate_hz = None if duration_s <= 0.0 else float(oscillation_count / max(duration_s, 1e-9) / 2.0)

    front_units = meta["responses"][FRONT_TRAVEL_RESPONSE]["units"]
    rear_units = meta["responses"][REAR_TRAVEL_RESPONSE]["units"]
    front_percent = front_units == "%"
    rear_percent = rear_units == "%"
    front_near_full = _threshold_event_summary(
        front_travel[valid],
        dt_s[valid],
        float(config["near_bottom_threshold_pct"]),
        absolute_threshold=front_percent,
    )
    front_bottom = _threshold_event_summary(
        front_travel[valid],
        dt_s[valid],
        float(config["bottom_out_threshold_pct"]),
        absolute_threshold=front_percent,
    )
    rear_near_full = _threshold_event_summary(
        rear_travel[valid],
        dt_s[valid],
        float(config["near_bottom_threshold_pct"]),
        absolute_threshold=rear_percent,
    )
    rear_bottom = _threshold_event_summary(
        rear_travel[valid],
        dt_s[valid],
        float(config["bottom_out_threshold_pct"]),
        absolute_threshold=rear_percent,
    )
    front_time_above = {
        int(threshold): _time_above(front_travel, front_valid, float(threshold)) if front_percent else None
        for threshold in config["stroke_warning_thresholds_pct"]
    }
    rear_time_above = {
        int(threshold): _time_above(rear_travel, rear_valid, float(threshold)) if rear_percent else None
        for threshold in config["stroke_warning_thresholds_pct"]
    }

    speed_points = speed_kph[speed_valid]
    result = {
        "label": label,
        "has_data": True,
        "duration_s": duration_s,
        "sample_count": int(np.count_nonzero(valid)),
        "start_time_s": float(np.min(time_s[valid])),
        "end_time_s": float(np.max(time_s[valid])),
        "entry_speed_kph": None if speed_points.size == 0 else float(speed_points[0]),
        "exit_speed_kph": None if speed_points.size == 0 else float(speed_points[-1]),
        "mean_speed_kph": _weighted_mean(speed_kph[speed_valid], dt_s[speed_valid]),
        "speed_delta_kph": None
        if speed_points.size == 0
        else float(speed_points[-1] - speed_points[0]),
        "peak_decel_mps2": None if not np.any(accel_valid) else float(np.min(speed_accel[accel_valid])),
        "peak_accel_mps2": None if not np.any(accel_valid) else float(np.max(speed_accel[accel_valid])),
        "front_used_stroke": _used_stroke(front_travel, front_valid),
        "rear_used_stroke": _used_stroke(rear_travel, rear_valid),
        "front_median_stroke_pct": _stroke_quantile(front_travel, front_valid, 0.50) if front_percent else None,
        "rear_median_stroke_pct": _stroke_quantile(rear_travel, rear_valid, 0.50) if rear_percent else None,
        "front_p90_stroke_pct": _stroke_quantile(front_travel, front_valid, 0.90) if front_percent else None,
        "rear_p90_stroke_pct": _stroke_quantile(rear_travel, rear_valid, 0.90) if rear_percent else None,
        "front_p95_stroke_pct": _stroke_quantile(front_travel, front_valid, 0.95) if front_percent else None,
        "rear_p95_stroke_pct": _stroke_quantile(rear_travel, rear_valid, 0.95) if rear_percent else None,
        "front_max_stroke_pct": _stroke_max(front_travel, front_valid) if front_percent else None,
        "rear_max_stroke_pct": _stroke_max(rear_travel, rear_valid) if rear_percent else None,
        "front_time_above_70_pct": front_time_above.get(70),
        "rear_time_above_70_pct": rear_time_above.get(70),
        "front_time_above_85_pct": front_time_above.get(85),
        "rear_time_above_85_pct": rear_time_above.get(85),
        "front_time_above_95_pct": front_time_above.get(95),
        "rear_time_above_95_pct": rear_time_above.get(95),
        "front_peak_compression_velocity": None
        if not np.any(front_velocity_valid)
        else float(np.max(front_velocity[front_velocity_valid])),
        "front_peak_rebound_velocity": None
        if not np.any(front_velocity_valid)
        else float(np.min(front_velocity[front_velocity_valid])),
        "rear_peak_compression_velocity": None
        if not np.any(rear_velocity_valid)
        else float(np.max(rear_velocity[rear_velocity_valid])),
        "rear_peak_rebound_velocity": None
        if not np.any(rear_velocity_valid)
        else float(np.min(rear_velocity[rear_velocity_valid])),
        "front_rms_velocity": _weighted_rms(front_velocity[front_velocity_valid], dt_s[front_velocity_valid]),
        "rear_rms_velocity": _weighted_rms(rear_velocity[rear_velocity_valid], dt_s[rear_velocity_valid]),
        "front_top_10_pct": _top_10_pct(front_travel, front_valid),
        "rear_top_10_pct": _top_10_pct(rear_travel, rear_valid),
        "mean_balance_pct": _weighted_mean(balance[balance_valid], dt_s[balance_valid]),
        "mean_front_share_pct": _weighted_mean(front_share[share_valid], dt_s[share_valid]),
        "front_bias_time_pct": None
        if duration_s <= 0.0
        else 100.0 * float(np.sum(dt_s[valid & (balance > 10.0)])) / duration_s,
        "rear_bias_time_pct": None
        if duration_s <= 0.0
        else 100.0 * float(np.sum(dt_s[valid & (balance < -10.0)])) / duration_s,
        "repeated_hit_active_duration_s": float(np.sum(dt_s[valid & repeated_hit])),
        "oscillation_count": oscillation_count,
        "mean_cycle_rate_hz": mean_cycle_rate_hz,
        "front_near_full_event_count": int(front_near_full["event_count"]),
        "front_bottom_out_event_count": int(front_bottom["event_count"]),
        "rear_near_full_event_count": int(rear_near_full["event_count"]),
        "rear_bottom_out_event_count": int(rear_bottom["event_count"]),
        "front_travel_units": front_units,
        "rear_travel_units": rear_units,
        "velocity_units": meta["responses"][FRONT_VELOCITY_RESPONSE]["units"],
    }
    return result


def compute_breakdown_heatmaps(
    context_df: pl.DataFrame,
    meta: dict[str, Any],
    selection_mask: np.ndarray,
    speed_bins: int = 48,
    response_bins: int = 48,
) -> dict[str, dict[str, Any]]:
    filtered_df = context_df.filter(pl.Series("selected", selection_mask))
    responses = {
        "front_travel": FRONT_TRAVEL_RESPONSE,
        "rear_travel": REAR_TRAVEL_RESPONSE,
        "front_velocity": FRONT_VELOCITY_RESPONSE,
        "rear_velocity": REAR_VELOCITY_RESPONSE,
    }
    heatmaps: dict[str, dict[str, Any]] = {}
    for key, response in responses.items():
        heatmaps[key] = compute_context_heatmap(
            context_df=filtered_df,
            meta=meta,
            source=WHEEL_SPEED_SOURCE,
            response=response,
            source_bins=speed_bins,
            response_bins=response_bins,
        )
    return heatmaps


def _diagnostic_flags(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    high_thresholds = config["stroke_warning_thresholds_pct"]
    high_stroke_threshold = float(high_thresholds[1] if len(high_thresholds) > 1 else high_thresholds[-1])
    motion_state = str(row.get("motion_state", ""))
    front_p95 = row.get("front_p95_stroke_pct")
    rear_p95 = row.get("rear_p95_stroke_pct")
    mean_balance = row.get("mean_balance_pct")
    if (row.get("front_bottom_out_event_count") or 0) > 0 or (row.get("rear_bottom_out_event_count") or 0) > 0:
        flags.append("bottom-out")
    if (row.get("front_near_full_event_count") or 0) > 0 or (row.get("rear_near_full_event_count") or 0) > 0:
        flags.append("near-bottom")
    if motion_state == "braking" and front_p95 is not None and float(front_p95) >= high_stroke_threshold:
        flags.append("braking fork usage high")
    if motion_state == "accelerating" and rear_p95 is not None and float(rear_p95) >= high_stroke_threshold:
        flags.append("accel rear usage high")
    if mean_balance is not None and abs(float(mean_balance)) >= 15.0:
        flags.append("front-biased" if float(mean_balance) > 0.0 else "rear-biased")
    if str(row.get("activity_state")) == ACTIVITY_REPEATED:
        flags.append("repeated hits")
    if str(row.get("activity_state")) == ACTIVITY_IMPACT:
        flags.append("impact candidate")
    return flags


def summarize_breakdown_state_rows(context_df: pl.DataFrame, meta: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(meta.get("wheel_speed", {}).get("usable", False)):
        return []
    config = normalize_breakdown_config(meta.get("breakdown_config"))
    dt_s = _series_to_numpy(context_df, "dt_s", np.nan)
    speed_kph = _series_to_numpy(context_df, "speed_kph_ctx", np.nan)
    speed_band_id = _series_to_numpy(context_df, "speed_band_id", -1).astype(np.int64)
    speed_band_label = np.asarray(context_df.get_column("speed_band_label").to_list(), dtype=object)
    motion_state = np.asarray(context_df.get_column("motion_state").to_list(), dtype=object)
    if "activity_state" in context_df.columns:
        activity_state = np.asarray(context_df.get_column("activity_state").to_list(), dtype=object)
    else:
        activity_state = np.full(speed_band_id.shape, ACTIVITY_UNKNOWN, dtype=object)

    valid = (
        np.isfinite(dt_s)
        & (dt_s > 0.0)
        & np.isfinite(speed_kph)
        & (speed_band_id >= 0)
        & (motion_state != "unknown")
        & (activity_state != ACTIVITY_UNKNOWN)
    )
    if not np.any(valid):
        return []

    total_session_s = float(np.sum(dt_s[np.isfinite(dt_s) & (dt_s > 0.0)]))
    rows: list[dict[str, Any]] = []
    unique_keys = sorted(
        {
            (
                int(speed_band_id[index]),
                str(speed_band_label[index]),
                str(motion_state[index]),
                str(activity_state[index]),
            )
            for index in np.where(valid)[0]
        },
        key=lambda item: (item[0], MOTION_ORDER.get(item[2], 99), ACTIVITY_ORDER.get(item[3], 99)),
    )
    for row_index, (band_id, band_label, motion, activity) in enumerate(unique_keys):
        mask = valid & (speed_band_id == band_id) & (motion_state == motion) & (activity_state == activity)
        row = summarize_breakdown_selection(context_df, meta, mask, f"{band_label} / {motion} / {activity}")
        if not row.get("has_data"):
            continue
        row.update(
            {
                "state_id": f"S{row_index + 1:03d}",
                "speed_band_id": band_id,
                "speed_band": band_label,
                "motion_state": motion,
                "activity_state": activity,
                "time_in_state_s": row["duration_s"],
                "session_pct": None
                if total_session_s <= 0.0
                else 100.0 * float(row["duration_s"]) / total_session_s,
            }
        )
        row["diagnostic_flags"] = _diagnostic_flags(row, config)
        rows.append(row)
    return rows


def summarize_breakdown_groups(context_df: pl.DataFrame, meta: dict[str, Any]) -> list[dict[str, Any]]:
    return summarize_breakdown_state_rows(context_df, meta)


def _iter_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    index = 0
    while index < mask.size:
        if not mask[index]:
            index += 1
            continue
        end = index + 1
        while end < mask.size and mask[end]:
            end += 1
        runs.append((index, end))
        index = end
    return runs


def _event_row_from_run(
    context_df: pl.DataFrame,
    meta: dict[str, Any],
    arrays: dict[str, Any],
    start_index: int,
    end_index: int,
    *,
    event_type: str,
    channel: str,
    label: str,
) -> dict[str, Any]:
    mask = np.zeros(context_df.height, dtype=bool)
    mask[start_index:end_index] = True
    dt_s = arrays["dt_s"]
    time_s = arrays["time_s"]
    speed_kph = arrays["speed_kph"]
    speed_accel = arrays["speed_accel"]
    front_travel = arrays["front_travel"]
    rear_travel = arrays["rear_travel"]
    front_velocity = arrays["front_velocity"]
    rear_velocity = arrays["rear_velocity"]
    motion_state = arrays["motion_state"]
    activity_state = arrays["activity_state"]
    roughness = arrays["roughness"]
    longitudinal = arrays["longitudinal"]
    valid_time = mask & np.isfinite(time_s) & np.isfinite(dt_s) & (dt_s > 0.0)
    indexes = np.where(valid_time & np.isfinite(time_s))[0]
    mid_index = start_index + max(0, (end_index - start_index - 1) // 2)
    speed_mask = mask & np.isfinite(speed_kph) & np.isfinite(dt_s) & (dt_s > 0.0)
    accel_mask = mask & np.isfinite(speed_accel)
    roughness_mask = mask & np.isfinite(roughness)
    longitudinal_mask = mask & np.isfinite(longitudinal)

    def _max_value(values: np.ndarray) -> float | None:
        finite = mask & np.isfinite(values)
        return None if not np.any(finite) else float(np.max(values[finite]))

    def _min_value(values: np.ndarray) -> float | None:
        finite = mask & np.isfinite(values)
        return None if not np.any(finite) else float(np.min(values[finite]))

    duration_s = float(np.sum(dt_s[valid_time])) if np.any(valid_time) else 0.0
    start_time_s = float(time_s[indexes[0]]) if indexes.size else None
    end_time_s = float(time_s[indexes[-1]]) if indexes.size else None
    row = {
        "event_id": "",
        "event_type": event_type,
        "type": event_type,
        "channel": channel,
        "label": label,
        "has_data": bool(np.any(valid_time)),
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "duration_s": duration_s,
        "speed_kph": _weighted_mean(speed_kph[speed_mask], dt_s[speed_mask]),
        "mean_speed_kph": _weighted_mean(speed_kph[speed_mask], dt_s[speed_mask]),
        "peak_decel_mps2": None if not np.any(accel_mask) else float(np.min(speed_accel[accel_mask])),
        "peak_accel_mps2": None if not np.any(accel_mask) else float(np.max(speed_accel[accel_mask])),
        "motion_state": str(motion_state[mid_index]) if motion_state.size > mid_index else "unknown",
        "activity_state": str(activity_state[mid_index]) if activity_state.size > mid_index else ACTIVITY_UNKNOWN,
        "front_peak_stroke_pct": _max_value(front_travel),
        "rear_peak_stroke_pct": _max_value(rear_travel),
        "front_max_stroke_pct": _max_value(front_travel),
        "rear_max_stroke_pct": _max_value(rear_travel),
        "front_peak_compression_velocity": _max_value(front_velocity),
        "rear_peak_compression_velocity": _max_value(rear_velocity),
        "front_peak_rebound_velocity": _min_value(front_velocity),
        "rear_peak_rebound_velocity": _min_value(rear_velocity),
        "peak_vertical_roughness_g": None
        if not np.any(roughness_mask)
        else float(np.nanmax(np.abs(roughness[roughness_mask]))),
        "peak_longitudinal_accel_g": None
        if not np.any(longitudinal_mask)
        else float(np.nanmax(np.abs(longitudinal[longitudinal_mask]))),
    }
    if indexes.size:
        row["timestamp_s"] = float(time_s[indexes[0]])
    else:
        row["timestamp_s"] = start_time_s
    return row


def _pack_down_mask(values: np.ndarray, context_df: pl.DataFrame, config: dict[str, Any]) -> np.ndarray:
    dt_s = _series_to_numpy(context_df, "dt_s", np.nan)
    time_s = _series_to_numpy(context_df, "time_s", np.nan)
    activity = _series_to_numpy(context_df, "suspension_activity_score", np.nan)
    repeated = (
        np.asarray(context_df.get_column("repeated_hit_proxy").fill_null(False).to_numpy(), dtype=bool)
        if "repeated_hit_proxy" in context_df.columns
        else np.zeros(values.shape, dtype=bool)
    )
    finite = np.isfinite(values) & np.isfinite(dt_s) & (dt_s > 0.0)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=bool)
    baseline = _weighted_quantile(values[finite], dt_s[finite], 0.50)
    if baseline is None:
        return np.zeros(values.shape, dtype=bool)
    smoothed = _moving_average_nan(values, _window_samples(time_s, int(config["pack_down_window_ms"])))
    active = repeated | (np.isfinite(activity) & (activity >= float(config["active_activity_threshold"])))
    mask = finite & active & np.isfinite(smoothed) & (
        smoothed >= float(baseline) + float(config["pack_down_shift_threshold_pct"])
    )
    return _keep_long_true_runs(mask, _window_samples(time_s, int(config["pack_down_window_ms"])))


def detect_breakdown_events(context_df: pl.DataFrame, meta: dict[str, Any]) -> list[dict[str, Any]]:
    if context_df.is_empty():
        return []
    config = normalize_breakdown_config(meta.get("breakdown_config"))
    rows: list[dict[str, Any]] = []
    dt_s = _series_to_numpy(context_df, "dt_s", np.nan)
    valid_time = np.isfinite(dt_s) & (dt_s > 0.0)
    front_travel = _series_to_numpy(context_df, "front_travel_resp", np.nan)
    rear_travel = _series_to_numpy(context_df, "rear_travel_resp", np.nan)
    front_velocity = _series_to_numpy(context_df, "front_velocity_resp", np.nan)
    rear_velocity = _series_to_numpy(context_df, "rear_velocity_resp", np.nan)
    event_arrays = {
        "dt_s": dt_s,
        "time_s": _series_to_numpy(context_df, "time_s", np.nan),
        "speed_kph": _series_to_numpy(context_df, "speed_kph_ctx", np.nan),
        "speed_accel": _series_to_numpy(context_df, "speed_accel_mps2_ctx", np.nan),
        "front_travel": front_travel,
        "rear_travel": rear_travel,
        "front_velocity": front_velocity,
        "rear_velocity": rear_velocity,
        "motion_state": np.asarray(context_df.get_column("motion_state").to_list(), dtype=object) if "motion_state" in context_df.columns else np.asarray([], dtype=object),
        "activity_state": np.asarray(context_df.get_column("activity_state").to_list(), dtype=object) if "activity_state" in context_df.columns else np.asarray([], dtype=object),
        "roughness": _series_to_numpy(context_df, "vertical_roughness_g_ctx", np.nan),
        "longitudinal": _series_to_numpy(context_df, "longitudinal_accel_g_ctx", np.nan),
    }
    front_pct = meta["responses"][FRONT_TRAVEL_RESPONSE]["units"] == "%"
    rear_pct = meta["responses"][REAR_TRAVEL_RESPONSE]["units"] == "%"
    high_velocity_threshold = float(
        meta.get("breakdown_thresholds", {}).get(
            "high_compression_velocity_pct_s",
            config["high_compression_velocity_min_pct_s"],
        )
    )

    event_specs: list[tuple[str, str, str, np.ndarray]] = []
    if front_pct:
        event_specs.extend(
            [
                (
                    "near_bottom",
                    "front",
                    "Front near-bottom",
                    valid_time & np.isfinite(front_travel) & (front_travel >= float(config["near_bottom_threshold_pct"])),
                ),
                (
                    "bottom_out",
                    "front",
                    "Front bottom-out",
                    valid_time & np.isfinite(front_travel) & (front_travel >= float(config["bottom_out_threshold_pct"])),
                ),
            ]
        )
    if rear_pct:
        event_specs.extend(
            [
                (
                    "near_bottom",
                    "rear",
                    "Rear near-bottom",
                    valid_time & np.isfinite(rear_travel) & (rear_travel >= float(config["near_bottom_threshold_pct"])),
                ),
                (
                    "bottom_out",
                    "rear",
                    "Rear bottom-out",
                    valid_time & np.isfinite(rear_travel) & (rear_travel >= float(config["bottom_out_threshold_pct"])),
                ),
            ]
        )
    event_specs.extend(
        [
            (
                "high_compression_velocity",
                "front",
                "Front high compression velocity",
                valid_time & np.isfinite(front_velocity) & (front_velocity >= high_velocity_threshold),
            ),
            (
                "high_compression_velocity",
                "rear",
                "Rear high compression velocity",
                valid_time & np.isfinite(rear_velocity) & (rear_velocity >= high_velocity_threshold),
            ),
            (
                "potential_pack_down",
                "front",
                "Front potential pack-down",
                _pack_down_mask(front_travel, context_df, config),
            ),
            (
                "potential_pack_down",
                "rear",
                "Rear potential pack-down",
                _pack_down_mask(rear_travel, context_df, config),
            ),
        ]
    )

    for event_type, channel, label, mask in event_specs:
        for start, end in _iter_true_runs(mask):
            rows.append(
                _event_row_from_run(
                    context_df,
                    meta,
                    event_arrays,
                    start,
                    end,
                    event_type=event_type,
                    channel=channel,
                    label=label,
                )
            )

    repeated = (
        np.asarray(context_df.get_column("repeated_hit_proxy").fill_null(False).to_numpy(), dtype=bool)
        if "repeated_hit_proxy" in context_df.columns
        else np.zeros(context_df.height, dtype=bool)
    )
    motion_state = np.asarray(context_df.get_column("motion_state").to_list(), dtype=object) if "motion_state" in context_df.columns else np.asarray([], dtype=object)
    for start, end in _iter_true_runs(repeated):
        mid = start + max(0, (end - start - 1) // 2)
        event_type = "braking_bump_section" if motion_state.size > mid and str(motion_state[mid]) == "braking" else "repeated_hit_section"
        rows.append(
            _event_row_from_run(
                context_df,
                meta,
                event_arrays,
                start,
                end,
                event_type=event_type,
                channel="both",
                label="Braking-bump section" if event_type == "braking_bump_section" else "Repeated-hit section",
            )
        )

    rows.sort(key=lambda row: (float(row.get("start_time_s") or 0.0), str(row.get("event_type", ""))))
    for index, row in enumerate(rows, start=1):
        row["event_id"] = f"E{index:03d}"
        row["segment_id"] = row["event_id"]
    return rows


def build_breakdown_quality_rows(context_df: pl.DataFrame, meta: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wheel_meta = meta.get("wheel_speed", {})
    rows.append(
        {
            "name": "Wheel speed coverage",
            "status": "ok" if wheel_meta.get("usable") else "warning",
            "detail": (
                f"{float(wheel_meta.get('coverage_pct', 0.0)):.1f}% held-speed coverage, "
                f"{int(wheel_meta.get('valid_intervals', 0))} valid intervals"
            ),
        }
    )
    imu_columns = ("longitudinal_accel_g_ctx", "vertical_accel_g_ctx", "pitch_rate_dps_ctx")
    imu_has_data = any(
        column in context_df.columns and np.isfinite(_series_to_numpy(context_df, column, np.nan)).any()
        for column in imu_columns
    )
    rows.append(
        {
            "name": "IMU context",
            "status": "ok" if imu_has_data else "info",
            "detail": "available" if imu_has_data else "unavailable; suspension activity is used for roughness state",
        }
    )
    for channel in ("front", "rear"):
        stroke_info = meta.get("stroke_percent", {}).get(channel, {})
        source = str(stroke_info.get("source", "unknown"))
        if source == "configured-full-scale":
            status = "ok"
        elif source in {"used-stroke", "unknown"}:
            status = "warning"
        else:
            status = "warning"
        rows.append(
            {
                "name": f"{channel.capitalize()} stroke basis",
                "status": status,
                "detail": source,
            }
        )
    return rows


def build_breakdown_findings(
    state_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    config = normalize_breakdown_config(meta.get("breakdown_config"))
    findings: list[dict[str, Any]] = []
    for quality in quality_rows:
        if quality.get("status") == "warning":
            severity = 95 if quality.get("name") == "Wheel speed coverage" else 55
            title = "Wheel speed coverage too weak for reliable state classification" if quality.get("name") == "Wheel speed coverage" else f"{quality.get('name')} needs review"
            findings.append(
                {
                    "severity": severity,
                    "title": title,
                    "detail": str(quality.get("detail", "")),
                    "category": "quality",
                }
            )

    bottom_events = [row for row in event_rows if row.get("event_type") == "bottom_out"]
    if bottom_events:
        findings.append(
            {
                "severity": 90,
                "title": "Bottom-out events detected",
                "detail": f"{len(bottom_events)} bottom-out event(s) across front/rear channels",
                "category": "event",
            }
        )

    braking_rows = [row for row in state_rows if row.get("motion_state") == "braking"]
    braking_front = [
        row for row in braking_rows
        if row.get("front_p95_stroke_pct") is not None and float(row["front_p95_stroke_pct"]) >= 85.0
    ]
    if braking_front:
        worst = max(braking_front, key=lambda row: float(row.get("front_p95_stroke_pct") or 0.0))
        findings.append(
            {
                "severity": 82,
                "title": "Braking fork usage high",
                "detail": f"{worst.get('speed_band')} braking front P95 {float(worst['front_p95_stroke_pct']):.1f}%",
                "category": "state",
            }
        )

    rear_pack = [row for row in event_rows if row.get("event_type") == "potential_pack_down" and row.get("channel") == "rear"]
    if rear_pack:
        findings.append(
            {
                "severity": 76,
                "title": "Rear pack-down suspected",
                "detail": f"{len(rear_pack)} rear pack-down candidate section(s)",
                "category": "event",
            }
        )

    high_comp = [row for row in event_rows if row.get("event_type") == "high_compression_velocity"]
    rear_high = [row for row in high_comp if row.get("channel") == "rear"]
    front_high = [row for row in high_comp if row.get("channel") == "front"]
    if rear_high and len(rear_high) > len(front_high):
        findings.append(
            {
                "severity": 70,
                "title": "High-speed compression events are rear-biased",
                "detail": f"{len(rear_high)} rear event(s) vs {len(front_high)} front event(s)",
                "category": "event",
            }
        )

    coasting_rows = [row for row in state_rows if row.get("motion_state") == "coasting" and row.get("mean_balance_pct") is not None]
    braking_balance = [row for row in braking_rows if row.get("mean_balance_pct") is not None]
    if coasting_rows and braking_balance:
        coasting_mean = float(np.mean([float(row["mean_balance_pct"]) for row in coasting_rows]))
        braking_mean = float(np.mean([float(row["mean_balance_pct"]) for row in braking_balance]))
        shift = braking_mean - coasting_mean
        if abs(shift) >= float(config.get("pack_down_shift_threshold_pct", 8.0)):
            findings.append(
                {
                    "severity": 66,
                    "title": "Front/rear balance shifts strongly under braking",
                    "detail": f"Mean balance shifts {shift:.1f} percentage points from coasting to braking",
                    "category": "state",
                }
            )

    enough_state_time = [row for row in state_rows if float(row.get("duration_s") or 0.0) >= 1.0]
    low_front = [row for row in enough_state_time if row.get("front_p95_stroke_pct") is not None and float(row["front_p95_stroke_pct"]) < 70.0]
    low_rear = [row for row in enough_state_time if row.get("rear_p95_stroke_pct") is not None and float(row["rear_p95_stroke_pct"]) < 70.0]
    if enough_state_time and len(low_front) == len(enough_state_time) and len(low_rear) == len(enough_state_time):
        findings.append(
            {
                "severity": 42,
                "title": "Travel use is low across classified states",
                "detail": "Front and rear P95 stroke stay below 70% in all longer states",
                "category": "state",
            }
        )

    findings.sort(key=lambda row: int(row.get("severity", 0)), reverse=True)
    return findings[:8]


def summarize_speed_percentile_bands(
    context_df: pl.DataFrame,
    meta: dict[str, Any],
    band_count: int = 5,
) -> list[dict[str, Any]]:
    if band_count <= 0 or not bool(meta.get("wheel_speed", {}).get("usable", False)):
        return []

    dt_s = _series_to_numpy(context_df, "dt_s", np.nan)
    speed_kph = _series_to_numpy(context_df, "speed_kph_ctx", np.nan)
    valid = np.isfinite(dt_s) & (dt_s > 0.0) & np.isfinite(speed_kph)
    if not np.any(valid):
        return []

    valid_speed = speed_kph[valid]
    valid_dt = dt_s[valid]
    total_occupancy_s = float(np.sum(valid_dt))
    if total_occupancy_s <= 0.0:
        return []

    quantiles = np.linspace(0.0, 1.0, num=band_count + 1)
    edges: list[float] = []
    for quantile in quantiles:
        edge = _weighted_quantile(valid_speed, valid_dt, float(quantile))
        if edge is None:
            edge = float(np.min(valid_speed)) if quantile <= 0.0 else float(np.max(valid_speed))
        edges.append(float(edge))
    edges[0] = float(np.min(valid_speed))
    edges[-1] = float(np.max(valid_speed))

    rows: list[dict[str, Any]] = []
    for band_index in range(band_count):
        lower = float(edges[band_index])
        upper = float(edges[band_index + 1])
        if band_index == band_count - 1:
            mask = valid & (speed_kph >= lower) & (speed_kph <= upper)
        else:
            mask = valid & (speed_kph >= lower) & (speed_kph < upper)
        occupancy_s = float(np.sum(dt_s[mask])) if np.any(mask) else 0.0
        occupancy_pct = 100.0 * occupancy_s / total_occupancy_s if total_occupancy_s > 0.0 else 0.0
        rows.append(
            {
                "band_index": band_index + 1,
                "percentile_label": f"P{int(quantiles[band_index] * 100)}-{int(quantiles[band_index + 1] * 100)}",
                "speed_min_kph": lower,
                "speed_max_kph": upper,
                "occupancy_s": occupancy_s,
                "occupancy_pct": occupancy_pct,
                "mean_speed_kph": _weighted_mean(speed_kph[mask], dt_s[mask]),
            }
        )
    return rows


def detect_breakdown_segments(context_df: pl.DataFrame, meta: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(meta.get("wheel_speed", {}).get("usable", False)):
        return []
    config = normalize_breakdown_config(meta.get("breakdown_config"))
    dt_s = _series_to_numpy(context_df, "dt_s", np.nan)
    time_s = _series_to_numpy(context_df, "time_s", np.nan)
    speed_band_id = _series_to_numpy(context_df, "speed_band_id", -1).astype(np.int64)
    speed_band_label = np.asarray(context_df.get_column("speed_band_label").to_list(), dtype=object)
    motion_state = np.asarray(context_df.get_column("motion_state").to_list(), dtype=object)
    segment_id = _series_to_numpy(context_df, "segment_id", -1).astype(np.int64)
    repeated_hit = np.asarray(
        context_df.get_column("repeated_hit_proxy").fill_null(False).to_numpy(),
        dtype=bool,
    )

    rows: list[dict[str, Any]] = []
    base_valid = segment_id >= 0
    min_segment_s = float(config["min_segment_ms"]) / 1000.0
    for current_id in sorted({int(value) for value in segment_id[base_valid]}):
        mask = segment_id == current_id
        duration_s = float(np.sum(dt_s[mask & np.isfinite(dt_s) & (dt_s > 0.0)]))
        if duration_s < min_segment_s:
            continue
        summary = summarize_breakdown_selection(context_df, meta, mask, f"Segment A{current_id + 1:03d}")
        summary.update(
            {
                "segment_id": f"A{current_id + 1:03d}",
                "type": "speed_motion",
                "speed_band": str(speed_band_label[np.where(mask)[0][0]]),
                "motion_state": str(motion_state[np.where(mask)[0][0]]),
            }
        )
        rows.append(summary)

    index = 0
    repeated_min_s = float(config["repeated_hit_min_duration_ms"]) / 1000.0
    repeated_counter = 1
    while index < repeated_hit.size:
        if not repeated_hit[index]:
            index += 1
            continue
        end = index + 1
        while end < repeated_hit.size and repeated_hit[end]:
            end += 1
        mask = np.zeros(repeated_hit.shape, dtype=bool)
        mask[index:end] = True
        duration_s = float(np.sum(dt_s[mask & np.isfinite(dt_s) & (dt_s > 0.0)]))
        if duration_s >= repeated_min_s:
            mid = index + (end - index) // 2
            summary = summarize_breakdown_selection(context_df, meta, mask, f"Repeated hits R{repeated_counter:03d}")
            summary.update(
                {
                    "segment_id": f"R{repeated_counter:03d}",
                    "type": "repeated_hits",
                    "speed_band": str(speed_band_label[mid]),
                    "motion_state": str(motion_state[mid]),
                }
            )
            rows.append(summary)
            repeated_counter += 1
        index = end

    rows.sort(key=lambda row: (0 if row["type"] == "speed_motion" else 1, row.get("start_time_s") or 0.0))
    return rows


def build_breakdown_analysis(
    derived_df: pl.DataFrame,
    wheel_df: pl.DataFrame,
    imu_frame_df: pl.DataFrame,
    breakdown_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context_df, meta = build_breakdown_dataset(
        derived_df=derived_df,
        wheel_df=wheel_df,
        imu_frame_df=imu_frame_df,
        breakdown_config=breakdown_config,
    )
    state_rows = summarize_breakdown_state_rows(context_df, meta)
    event_rows = detect_breakdown_events(context_df, meta)
    quality_rows = build_breakdown_quality_rows(context_df, meta)
    finding_rows = build_breakdown_findings(state_rows, event_rows, quality_rows, meta)
    return {
        "context_df": context_df,
        "meta": meta,
        "band_rows": summarize_speed_percentile_bands(context_df, meta, band_count=5),
        "group_rows": state_rows,
        "state_rows": state_rows,
        "segment_rows": event_rows,
        "event_rows": event_rows,
        "quality_rows": quality_rows,
        "finding_rows": finding_rows,
    }
