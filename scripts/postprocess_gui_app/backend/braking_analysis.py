from __future__ import annotations

import copy
from typing import Any

import numpy as np
import polars as pl

from .session_config import DEFAULT_BRAKING_CONFIG, merge_defaults


def normalize_braking_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = merge_defaults(DEFAULT_BRAKING_CONFIG, config or {})

    threshold_mode = str(merged.get("threshold_mode", "percentile")).lower()
    if threshold_mode not in {"percentile", "absolute", "both"}:
        threshold_mode = "percentile"
    merged["threshold_mode"] = threshold_mode

    selected_type = str(merged.get("selected_threshold_type", threshold_mode)).lower()
    if selected_type not in {"percentile", "absolute"}:
        selected_type = "percentile" if threshold_mode != "absolute" else "absolute"
    merged["selected_threshold_type"] = selected_type

    speed_bin_mode = str(merged.get("speed_bin_mode", "percentile")).lower()
    if speed_bin_mode not in {"percentile", "absolute"}:
        speed_bin_mode = "percentile"
    merged["speed_bin_mode"] = speed_bin_mode

    merged["selected_threshold_label"] = str(merged.get("selected_threshold_label", "P80"))
    merged["decel_percentiles"] = _float_list(merged.get("decel_percentiles"), [40.0, 60.0, 80.0])
    merged["absolute_decel_thresholds_mps2"] = _float_list(
        merged.get("absolute_decel_thresholds_mps2"),
        [2.0, 3.0, 4.0, 5.0],
    )
    merged["speed_percentile_edges"] = _float_list(
        merged.get("speed_percentile_edges"),
        [0.0, 25.0, 50.0, 75.0, 100.0],
    )
    if len(merged["speed_percentile_edges"]) < 2:
        merged["speed_percentile_edges"] = [0.0, 100.0]
    merged["speed_edges_abs_kph"] = _float_list(
        merged.get("speed_edges_abs_kph"),
        [0.0, 20.0, 35.0, 50.0, 120.0],
    )
    if len(merged["speed_edges_abs_kph"]) < 2:
        merged["speed_edges_abs_kph"] = [0.0, 120.0]

    for key, minimum in (
        ("valid_speed_min_kph", 0.0),
        ("valid_speed_max_kph", 1.0),
        ("speed_source_min_kph", 0.0),
        ("speed_source_max_kph", 1.0),
        ("speed_smoothing_window_s", 0.0),
        ("valid_abs_accel_max_mps2", 0.1),
        ("coasting_accel_abs_max_mps2", 0.0),
        ("min_braking_event_duration_s", 0.0),
        ("merge_event_gap_s", 0.0),
        ("topout_threshold_pct", -100.0),
    ):
        merged[key] = max(minimum, _safe_float(merged.get(key), float(DEFAULT_BRAKING_CONFIG[key])))

    for key in ("min_samples_per_metric_bin", "min_events_per_metric_bin"):
        merged[key] = max(1, int(round(_safe_float(merged.get(key), float(DEFAULT_BRAKING_CONFIG[key])))))

    for key in (
        "valid_stroke_min_pct",
        "valid_stroke_max_pct",
        "topout_caution_pct",
        "topout_high_pct",
        "rough_caution_ratio",
        "rough_high_ratio",
        "dive_low_pct",
        "dive_high_pct",
        "packdown_caution_pct",
        "packdown_high_pct",
        "deep80_caution_pct",
        "deep90_caution_pct",
        "bottom_margin_caution_pct",
        "bottom_margin_high_pct",
    ):
        merged[key] = _safe_float(merged.get(key), float(DEFAULT_BRAKING_CONFIG[key]))

    if merged["valid_speed_max_kph"] <= merged["valid_speed_min_kph"]:
        merged["valid_speed_max_kph"] = merged["valid_speed_min_kph"] + 1.0
    if merged["speed_source_max_kph"] <= merged["speed_source_min_kph"]:
        merged["speed_source_max_kph"] = merged["speed_source_min_kph"] + 1.0
    if merged["valid_stroke_max_pct"] <= merged["valid_stroke_min_pct"]:
        merged["valid_stroke_max_pct"] = merged["valid_stroke_min_pct"] + 1.0
    return merged


def build_fork_braking_analysis(
    derived_df: pl.DataFrame,
    wheel_df: pl.DataFrame,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = normalize_braking_config(config)
    if derived_df.is_empty():
        return _empty_analysis(cfg, "No analog samples available.")

    time_s = _time_seconds(derived_df)
    dt_s = _series_to_numpy(derived_df, "dt_s", np.nan)
    if dt_s.size != time_s.size or not np.isfinite(dt_s).any():
        dt_s = _dt_from_time(time_s)

    stroke_column = "front_filtered_travel_stroke_pct"
    if stroke_column not in derived_df.columns:
        stroke_column = "front_travel_stroke_pct"
    if stroke_column not in derived_df.columns:
        return _empty_analysis(cfg, "Front stroke percent is unavailable.")
    stroke_pct = _series_to_numpy(derived_df, stroke_column, np.nan)

    if "front_velocity_mm_per_s_filtered" in derived_df.columns:
        velocity = _series_to_numpy(derived_df, "front_velocity_mm_per_s_filtered", np.nan)
        velocity_units = "mm/s"
    elif "front_velocity_stroke_pct_per_s_filtered" in derived_df.columns:
        velocity = _series_to_numpy(derived_df, "front_velocity_stroke_pct_per_s_filtered", np.nan)
        velocity_units = "%/s"
    else:
        velocity = np.full(stroke_pct.shape, np.nan, dtype=np.float64)
        velocity_units = ""

    speed_kph, speed_meta = _interpolate_speed_to_analog(wheel_df, time_s, cfg)
    if speed_meta["valid_count"] < 2:
        return _empty_analysis(cfg, "Wheel speed has fewer than two valid samples.", speed_meta=speed_meta)

    speed_smooth = _moving_average_nan(speed_kph, _window_samples(time_s, cfg["speed_smoothing_window_s"]))
    accel_mps2 = _time_gradient(speed_smooth / 3.6, time_s)
    decel_mps2 = -accel_mps2

    valid_base = (
        np.isfinite(time_s)
        & np.isfinite(dt_s)
        & (dt_s > 0.0)
        & np.isfinite(speed_kph)
        & np.isfinite(speed_smooth)
        & np.isfinite(accel_mps2)
        & np.isfinite(decel_mps2)
        & np.isfinite(stroke_pct)
        & (speed_kph >= cfg["valid_speed_min_kph"])
        & (speed_kph <= cfg["valid_speed_max_kph"])
        & (np.abs(accel_mps2) <= cfg["valid_abs_accel_max_mps2"])
        & (stroke_pct >= cfg["valid_stroke_min_pct"])
        & (stroke_pct <= cfg["valid_stroke_max_pct"])
    )
    if not np.any(valid_base):
        return _empty_analysis(cfg, "No valid braking-analysis samples after speed/stroke filtering.", speed_meta=speed_meta)

    try:
        speed_edges, speed_labels = _make_speed_bins(speed_kph, valid_base, cfg)
        thresholds = _make_decel_thresholds(decel_mps2, valid_base, cfg)
    except ValueError as exc:
        return _empty_analysis(cfg, str(exc), speed_meta=speed_meta)

    speed_bin_index = _assign_speed_bins(speed_kph, speed_edges)
    topout = stroke_pct <= cfg["topout_threshold_pct"]
    not_topout = ~topout
    coasting = valid_base & (np.abs(accel_mps2) <= cfg["coasting_accel_abs_max_mps2"]) & not_topout

    metric_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    event_counter = 0
    for threshold in thresholds:
        for bin_zero_index, speed_label in enumerate(speed_labels):
            bin_index = bin_zero_index + 1
            in_speed_bin = valid_base & (speed_bin_index == bin_index)
            braking = in_speed_bin & (decel_mps2 >= threshold["decel_threshold_mps2"])
            loaded = braking & not_topout
            coasting_mask = coasting & (speed_bin_index == bin_index)

            row_events, event_stats = _compute_braking_events(
                braking,
                time_s,
                dt_s,
                stroke_pct,
                velocity,
                speed_kph,
                decel_mps2,
                topout,
                threshold,
                bin_index,
                speed_label,
                speed_edges[bin_zero_index],
                speed_edges[bin_zero_index + 1],
                cfg,
            )
            for event in row_events:
                event_counter += 1
                event["global_event_index"] = event_counter
                event_rows.append(event)

            metric_rows.append(
                _build_metric_row(
                    threshold,
                    bin_index,
                    speed_label,
                    speed_edges[bin_zero_index],
                    speed_edges[bin_zero_index + 1],
                    braking,
                    loaded,
                    coasting_mask,
                    topout,
                    stroke_pct,
                    velocity,
                    dt_s,
                    event_stats,
                    cfg,
                )
            )

    selected_rows, selected = _select_metric_rows(metric_rows, cfg)
    selected_events = [
        row
        for row in event_rows
        if row["threshold_type"] == selected["threshold_type"]
        and row["threshold_label"] == selected["threshold_label"]
    ]
    summary = _build_one_page_summary(selected_rows, selected_events, selected, cfg, velocity_units)
    meta = {
        "has_data": bool(selected_rows),
        "warning": None if selected_rows else "No metric rows matched the selected threshold.",
        "config": cfg,
        "speed_source": speed_meta,
        "speed_edges_kph": [float(value) for value in speed_edges],
        "speed_bin_labels": [str(value) for value in speed_labels],
        "thresholds": copy.deepcopy(thresholds),
        "selected": selected,
        "stroke_column": stroke_column,
        "velocity_units": velocity_units,
    }
    if not selected_rows:
        meta["warning"] = "No metric rows matched the selected threshold."

    return {
        "metric_rows": metric_rows,
        "event_rows": event_rows,
        "key_metrics": summary["key_metrics"],
        "speed_bin_rows": summary["speed_bin_rows"],
        "flag_rows": summary["flag_rows"],
        "summary_text": summary["summary_text"],
        "meta": meta,
    }


def _empty_analysis(
    config: dict[str, Any],
    warning: str,
    *,
    speed_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "metric_rows": [],
        "event_rows": [],
        "key_metrics": [],
        "speed_bin_rows": [],
        "flag_rows": [{"level": "INFO", "topic": "No braking analysis", "message": warning}],
        "summary_text": warning,
        "meta": {
            "has_data": False,
            "warning": warning,
            "config": normalize_braking_config(config),
            "speed_source": speed_meta or {"source_count": 0, "valid_count": 0},
            "speed_edges_kph": [],
            "speed_bin_labels": [],
            "thresholds": [],
            "selected": {},
        },
    }


def _build_metric_row(
    threshold: dict[str, Any],
    speed_bin_index: int,
    speed_bin: str,
    speed_min: float,
    speed_max: float,
    braking: np.ndarray,
    loaded: np.ndarray,
    coasting: np.ndarray,
    topout: np.ndarray,
    stroke_pct: np.ndarray,
    velocity: np.ndarray,
    dt_s: np.ndarray,
    event_stats: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    loaded_stroke = stroke_pct[loaded]
    loaded_weights = dt_s[loaded]
    coasting_stroke = stroke_pct[coasting]
    coasting_weights = dt_s[coasting]
    loaded_velocity = velocity[loaded]
    loaded_velocity_weights = dt_s[loaded]
    coasting_velocity = velocity[coasting]
    coasting_velocity_weights = dt_s[coasting]

    loaded_median = _weighted_quantile(loaded_stroke, loaded_weights, 0.50)
    coasting_median = _weighted_quantile(coasting_stroke, coasting_weights, 0.50)
    loaded_p75 = _weighted_quantile(loaded_stroke, loaded_weights, 0.75)
    coasting_p75 = _weighted_quantile(coasting_stroke, coasting_weights, 0.75)
    loaded_p90 = _weighted_quantile(loaded_stroke, loaded_weights, 0.90)
    coasting_p90 = _weighted_quantile(coasting_stroke, coasting_weights, 0.90)
    loaded_p99 = _weighted_quantile(loaded_stroke, loaded_weights, 0.99)

    row = {
        "threshold_type": threshold["threshold_type"],
        "threshold_label": threshold["threshold_label"],
        "decel_percentile": threshold["decel_percentile"],
        "decel_threshold_mps2": threshold["decel_threshold_mps2"],
        "speed_bin_index": speed_bin_index,
        "speed_bin": speed_bin,
        "speed_min_kph": float(speed_min),
        "speed_max_kph": float(speed_max),
        "sample_count": int(np.count_nonzero(braking)),
        "loaded_sample_count": int(np.count_nonzero(loaded)),
        "coasting_sample_count": int(np.count_nonzero(coasting)),
        "time_s": _sum_time(dt_s, braking),
        "loaded_time_s": _sum_time(dt_s, loaded),
        "coasting_time_s": _sum_time(dt_s, coasting),
        "topout_contamination_pct": _time_fraction_pct(topout & braking, braking, dt_s),
        "loaded_braking_median_stroke_pct": loaded_median,
        "loaded_braking_p75_stroke_pct": loaded_p75,
        "loaded_braking_p90_stroke_pct": loaded_p90,
        "loaded_braking_p95_stroke_pct": _weighted_quantile(loaded_stroke, loaded_weights, 0.95),
        "loaded_braking_p99_stroke_pct": loaded_p99,
        "coasting_median_stroke_pct": coasting_median,
        "coasting_p75_stroke_pct": coasting_p75,
        "coasting_p90_stroke_pct": coasting_p90,
        "braking_dive_index_pct": _subtract_optional(loaded_median, coasting_median),
        "p75_stroke_offset_pct": _subtract_optional(loaded_p75, coasting_p75),
        "p90_stroke_offset_pct": _subtract_optional(loaded_p90, coasting_p90),
        "deep70_frac_pct": _time_fraction_pct(stroke_pct >= 70.0, loaded, dt_s),
        "deep80_frac_pct": _time_fraction_pct(stroke_pct >= 80.0, loaded, dt_s),
        "deep90_frac_pct": _time_fraction_pct(stroke_pct >= 90.0, loaded, dt_s),
        "deep95_frac_pct": _time_fraction_pct(stroke_pct >= 95.0, loaded, dt_s),
        "bottom_margin_p99_pct": None if loaded_p99 is None else 100.0 - loaded_p99,
        "rms_fork_velocity_braking": _weighted_rms(loaded_velocity, loaded_velocity_weights),
        "rms_fork_velocity_coasting": _weighted_rms(coasting_velocity, coasting_velocity_weights),
        "rough_braking_ratio": None,
        "p95_compression_velocity": _weighted_quantile(loaded_velocity[loaded_velocity > 0.0], loaded_velocity_weights[loaded_velocity > 0.0], 0.95),
        "p95_rebound_velocity": _weighted_quantile(np.abs(loaded_velocity[loaded_velocity < 0.0]), loaded_velocity_weights[loaded_velocity < 0.0], 0.95),
        "event_count": int(event_stats["event_count"]),
        "median_pack_down_pct": event_stats["median_pack_down_pct"],
        "p75_pack_down_pct": event_stats["p75_pack_down_pct"],
        "event_median_max_stroke_pct": event_stats["event_median_max_stroke_pct"],
        "event_p90_max_stroke_pct": event_stats["event_p90_max_stroke_pct"],
        "event_median_p95_stroke_pct": event_stats["event_median_p95_stroke_pct"],
    }
    row["rough_braking_ratio"] = _safe_divide(
        row["rms_fork_velocity_braking"],
        row["rms_fork_velocity_coasting"],
    )
    row["enough_samples"] = row["sample_count"] >= cfg["min_samples_per_metric_bin"]
    row["enough_loaded_samples"] = row["loaded_sample_count"] >= cfg["min_samples_per_metric_bin"]
    row["enough_coasting_samples"] = row["coasting_sample_count"] >= cfg["min_samples_per_metric_bin"]
    row["enough_events"] = row["event_count"] >= cfg["min_events_per_metric_bin"]
    row["warning"] = _metric_warning(row, cfg)
    return row


def _compute_braking_events(
    braking: np.ndarray,
    time_s: np.ndarray,
    dt_s: np.ndarray,
    stroke_pct: np.ndarray,
    velocity: np.ndarray,
    speed_kph: np.ndarray,
    decel_mps2: np.ndarray,
    topout: np.ndarray,
    threshold: dict[str, Any],
    speed_bin_index: int,
    speed_bin: str,
    speed_min: float,
    speed_max: float,
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    median_dt = _median_dt(time_s, dt_s)
    min_samples = max(1, int(np.ceil(cfg["min_braking_event_duration_s"] / median_dt)))
    gap_samples = max(0, int(round(cfg["merge_event_gap_s"] / median_dt)))
    merged = _merge_short_false_gaps(braking, gap_samples)
    runs = [
        (start, end)
        for start, end in _logical_runs(merged)
        if end - start + 1 >= min_samples
    ]
    rows: list[dict[str, Any]] = []
    packdowns: list[float] = []
    max_strokes: list[float] = []
    p95_strokes: list[float] = []

    for start, end in runs:
        index = np.arange(start, end + 1)
        loaded_index = index[~topout[index] & np.isfinite(stroke_pct[index])]
        event_stroke = stroke_pct[loaded_index]
        event_velocity = velocity[loaded_index]
        event_weights = dt_s[loaded_index]

        packdown = None
        if loaded_index.size >= 6:
            third = max(1, loaded_index.size // 3)
            first = stroke_pct[loaded_index[:third]]
            last = stroke_pct[loaded_index[-third:]]
            first_median = _nan_percentile(first, 50.0)
            last_median = _nan_percentile(last, 50.0)
            packdown = _subtract_optional(last_median, first_median)
        if packdown is not None:
            packdowns.append(packdown)

        max_stroke = _nan_max(event_stroke)
        p95_stroke = _weighted_quantile(event_stroke, event_weights, 0.95)
        if max_stroke is not None:
            max_strokes.append(max_stroke)
        if p95_stroke is not None:
            p95_strokes.append(p95_stroke)

        rows.append(
            {
                "global_event_index": 0,
                "threshold_type": threshold["threshold_type"],
                "threshold_label": threshold["threshold_label"],
                "decel_percentile": threshold["decel_percentile"],
                "decel_threshold_mps2": threshold["decel_threshold_mps2"],
                "speed_bin_index": speed_bin_index,
                "speed_bin": speed_bin,
                "speed_min_kph": float(speed_min),
                "speed_max_kph": float(speed_max),
                "event_start_index": int(start),
                "event_end_index": int(end),
                "start_time_s": float(time_s[start]),
                "end_time_s": float(time_s[end]),
                "duration_s": _sum_time(dt_s, _index_mask(dt_s.size, start, end)),
                "entry_speed_kph": _finite_or_none(speed_kph[start]),
                "exit_speed_kph": _finite_or_none(speed_kph[end]),
                "peak_decel_mps2": _nan_max(decel_mps2[index]),
                "mean_decel_mps2": _nan_mean(decel_mps2[index]),
                "topout_fraction_pct": _time_fraction_pct(topout, _index_mask(dt_s.size, start, end), dt_s),
                "median_stroke_pct": _weighted_quantile(event_stroke, event_weights, 0.50),
                "max_stroke_pct": max_stroke,
                "p95_stroke_pct": p95_stroke,
                "rms_fork_velocity": _weighted_rms(event_velocity, event_weights),
                "pack_down_pct": packdown,
            }
        )

    return rows, {
        "event_count": len(rows),
        "median_pack_down_pct": _nan_percentile(np.asarray(packdowns, dtype=np.float64), 50.0),
        "p75_pack_down_pct": _nan_percentile(np.asarray(packdowns, dtype=np.float64), 75.0),
        "event_median_max_stroke_pct": _nan_percentile(np.asarray(max_strokes, dtype=np.float64), 50.0),
        "event_p90_max_stroke_pct": _nan_percentile(np.asarray(max_strokes, dtype=np.float64), 90.0),
        "event_median_p95_stroke_pct": _nan_percentile(np.asarray(p95_strokes, dtype=np.float64), 50.0),
    }


def _build_one_page_summary(
    selected_rows: list[dict[str, Any]],
    selected_events: list[dict[str, Any]],
    selected: dict[str, Any],
    cfg: dict[str, Any],
    velocity_units: str,
) -> dict[str, Any]:
    if not selected_rows:
        return {"key_metrics": [], "speed_bin_rows": [], "flag_rows": [], "summary_text": "No selected braking rows."}

    weights = np.asarray([_none_to_nan(row.get("loaded_time_s")) for row in selected_rows], dtype=np.float64)
    if not np.isfinite(weights).any() or np.nansum(weights) <= 0.0:
        weights = np.asarray([_none_to_nan(row.get("loaded_sample_count")) for row in selected_rows], dtype=np.float64)

    def agg_mean(key: str) -> float | None:
        return _weighted_mean(np.asarray([_none_to_nan(row.get(key)) for row in selected_rows]), weights)

    def agg_max(key: str) -> float | None:
        return _finite_max([row.get(key) for row in selected_rows])

    def agg_min(key: str) -> float | None:
        return _finite_min([row.get(key) for row in selected_rows])

    total_loaded_time = _finite_sum([row.get("loaded_time_s") for row in selected_rows])
    total_event_count = int(round(_finite_sum([row.get("event_count") for row in selected_rows])))
    speed_min = agg_min("speed_min_kph")
    speed_max = agg_max("speed_max_kph")
    event_summary = _event_summary(selected_events)

    aggregate = {
        "total_loaded_time_s": total_loaded_time,
        "total_event_count": total_event_count,
        "speed_min_kph": speed_min,
        "speed_max_kph": speed_max,
        "topout_weighted_pct": agg_mean("topout_contamination_pct"),
        "topout_max_pct": agg_max("topout_contamination_pct"),
        "loaded_median_weighted_pct": agg_mean("loaded_braking_median_stroke_pct"),
        "loaded_p90_weighted_pct": agg_mean("loaded_braking_p90_stroke_pct"),
        "loaded_p95_weighted_pct": agg_mean("loaded_braking_p95_stroke_pct"),
        "braking_dive_weighted_pct": agg_mean("braking_dive_index_pct"),
        "deep80_weighted_pct": agg_mean("deep80_frac_pct"),
        "deep90_weighted_pct": agg_mean("deep90_frac_pct"),
        "bottom_margin_p99_min_pct": agg_min("bottom_margin_p99_pct"),
        "rough_ratio_weighted": agg_mean("rough_braking_ratio"),
        "rough_ratio_max": agg_max("rough_braking_ratio"),
        "rms_fork_velocity_braking_weighted": agg_mean("rms_fork_velocity_braking"),
    }

    threshold_desc = f"{selected['threshold_type']} {selected['threshold_label']}, decel threshold {_fmt(selected['decel_threshold_mps2'], 2)} m/s2"
    key_metrics = _key_metric_rows(aggregate, event_summary, threshold_desc, cfg, velocity_units)
    speed_rows = [_speed_bin_summary_row(row, cfg) for row in selected_rows]
    flag_rows = _flag_rows(aggregate, event_summary, cfg)
    summary_text = _summary_text(aggregate, event_summary, speed_rows, flag_rows, threshold_desc)
    return {
        "key_metrics": key_metrics,
        "speed_bin_rows": speed_rows,
        "flag_rows": flag_rows,
        "summary_text": summary_text,
    }


def _key_metric_rows(
    agg: dict[str, Any],
    event_summary: dict[str, Any],
    threshold_desc: str,
    cfg: dict[str, Any],
    velocity_units: str,
) -> list[dict[str, Any]]:
    return [
        _key_row("Setup", "Selected threshold", threshold_desc, "", "INFO", "Braking subset used for this summary."),
        _key_row("Coverage", "Speed range", f"{_fmt(agg['speed_min_kph'], 1)}-{_fmt(agg['speed_max_kph'], 1)}", "km/h", "INFO", "Speed span covered by selected bins."),
        _key_row("Coverage", "Loaded braking time", agg["total_loaded_time_s"], "s", "INFO", "Time in selected braking samples after topout filtering."),
        _key_row("Coverage", "Braking events", event_summary["event_count"], "events", _status_event_count(event_summary["event_count"], cfg), "Number of braking events in the selected subset."),
        _key_row("Fork support", "Loaded median stroke", agg["loaded_median_weighted_pct"], "% stroke", "INFO", "Typical fork position during loaded braking."),
        _key_row("Fork support", "Loaded P90 stroke", agg["loaded_p90_weighted_pct"], "% stroke", "INFO", "Upper-end fork position during loaded braking."),
        _key_row("Fork support", "Braking dive index", agg["braking_dive_weighted_pct"], "% stroke", _status_dive(agg["braking_dive_weighted_pct"], cfg), "Loaded braking median minus same-speed coasting median."),
        _key_row("Stroke reserve", "Deep80 fraction", agg["deep80_weighted_pct"], "% samples", _status_high_worse(agg["deep80_weighted_pct"], cfg["deep80_caution_pct"], max(cfg["deep80_caution_pct"] * 3.0, 5.0)), "Fraction of loaded braking samples above 80% fork stroke."),
        _key_row("Stroke reserve", "Deep90 fraction", agg["deep90_weighted_pct"], "% samples", _status_high_worse(agg["deep90_weighted_pct"], cfg["deep90_caution_pct"], max(cfg["deep90_caution_pct"] * 3.0, 1.0)), "Fraction of loaded braking samples above 90% fork stroke."),
        _key_row("Stroke reserve", "P99 bottom margin", agg["bottom_margin_p99_min_pct"], "% stroke left", _status_low_worse(agg["bottom_margin_p99_min_pct"], cfg["bottom_margin_caution_pct"], cfg["bottom_margin_high_pct"]), "Smallest speed-bin P99 margin to full stroke."),
        _key_row("Data quality", "Topout contamination", f"{_fmt(agg['topout_weighted_pct'], 1)} avg / {_fmt(agg['topout_max_pct'], 1)} max", "% samples", _status_high_worse(agg["topout_max_pct"], cfg["topout_caution_pct"], cfg["topout_high_pct"]), "Near-extension samples inside the braking subset."),
        _key_row("Dynamic behavior", "Rough braking ratio", f"{_fmt(agg['rough_ratio_weighted'], 2)} avg / {_fmt(agg['rough_ratio_max'], 2)} max", "ratio", _status_high_worse(agg["rough_ratio_max"], cfg["rough_caution_ratio"], cfg["rough_high_ratio"]), "Fork velocity RMS during braking divided by same-speed coasting."),
        _key_row("Dynamic behavior", "RMS fork velocity", agg["rms_fork_velocity_braking_weighted"], velocity_units, "INFO", "Average fork activity during loaded braking."),
        _key_row("Dynamic behavior", "Pack-down", f"{_fmt_signed(event_summary['median_pack_down_pct'], 1)} median / {_fmt_signed(event_summary['p75_pack_down_pct'], 1)} P75", "% stroke", _status_packdown(event_summary["p75_pack_down_pct"], cfg), "Event end-stroke minus event start-stroke; positive means riding deeper."),
        _key_row("Interpretation", "Main read", _main_read(agg, event_summary, cfg), "", "TEXT", "Short evidence summary for display."),
    ]


def _speed_bin_summary_row(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    flag = _speed_bin_flag(row, cfg)
    return {
        "speed_bin": row["speed_bin"],
        "speed_range_kph": f"{_fmt(row['speed_min_kph'], 1)}-{_fmt(row['speed_max_kph'], 1)}",
        "loaded_time_s": row["loaded_time_s"],
        "events": row["event_count"],
        "topout_pct": row["topout_contamination_pct"],
        "loaded_median_stroke_pct": row["loaded_braking_median_stroke_pct"],
        "loaded_p90_stroke_pct": row["loaded_braking_p90_stroke_pct"],
        "coasting_median_stroke_pct": row["coasting_median_stroke_pct"],
        "braking_dive_index_pct": row["braking_dive_index_pct"],
        "deep80_pct": row["deep80_frac_pct"],
        "deep90_pct": row["deep90_frac_pct"],
        "bottom_margin_p99_pct": row["bottom_margin_p99_pct"],
        "rough_braking_ratio": row["rough_braking_ratio"],
        "pack_down_p75_pct": row["p75_pack_down_pct"],
        "flag": flag,
    }


def _flag_rows(agg: dict[str, Any], event_summary: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if event_summary["event_count"] < cfg["min_events_per_metric_bin"]:
        rows.append({"level": "HIGH", "topic": "Low event count", "message": "Too few braking events for a reliable tuning conclusion."})
    topout_max = agg["topout_max_pct"]
    if _isfinite(topout_max) and topout_max >= cfg["topout_high_pct"]:
        rows.append({"level": "HIGH", "topic": "Topout contamination", "message": "High topout fraction in at least one speed bin; some decel samples may be unweighted or speed artifacts."})
    elif _isfinite(topout_max) and topout_max >= cfg["topout_caution_pct"]:
        rows.append({"level": "WATCH", "topic": "Topout contamination", "message": "Moderate topout fraction; rely on loaded-braking metrics rather than raw braking samples."})
    dive = agg["braking_dive_weighted_pct"]
    if _isfinite(dive) and dive >= cfg["dive_high_pct"]:
        rows.append({"level": "WATCH", "topic": "Brake dive", "message": "Brake dive index is high relative to same-speed coasting."})
    elif _isfinite(dive) and dive < cfg["dive_low_pct"]:
        rows.append({"level": "INFO", "topic": "Low brake dive", "message": "Fork ride height changes little during braking; compare with comfort and grip."})
    if _isfinite(agg["deep90_weighted_pct"]) and agg["deep90_weighted_pct"] >= cfg["deep90_caution_pct"]:
        rows.append({"level": "HIGH", "topic": "Near-bottom use", "message": "Near-bottom use appears in the selected braking condition."})
    elif _isfinite(agg["deep80_weighted_pct"]) and agg["deep80_weighted_pct"] >= cfg["deep80_caution_pct"]:
        rows.append({"level": "WATCH", "topic": "Deep stroke use", "message": "Deep stroke use appears in the selected braking condition."})
    rough = agg["rough_ratio_max"]
    if _isfinite(rough) and rough >= cfg["rough_high_ratio"]:
        rows.append({"level": "HIGH", "topic": "Rough braking", "message": "Fork activity during braking is much higher than same-speed coasting in at least one bin."})
    elif _isfinite(rough) and rough >= cfg["rough_caution_ratio"]:
        rows.append({"level": "WATCH", "topic": "Rough braking", "message": "Fork activity during braking is elevated in at least one speed bin."})
    pack_p75 = event_summary["p75_pack_down_pct"]
    if _isfinite(pack_p75) and pack_p75 >= cfg["packdown_high_pct"]:
        rows.append({"level": "WATCH", "topic": "Pack-down", "message": "Upper-quartile pack-down is positive; some braking events ride deeper toward the end."})
    elif _isfinite(pack_p75) and pack_p75 >= cfg["packdown_caution_pct"]:
        rows.append({"level": "INFO", "topic": "Mild pack-down", "message": "Some events show mild positive pack-down."})
    if not rows:
        rows.append({"level": "INFO", "topic": "Baseline", "message": "No braking-analysis warning thresholds were crossed."})
    return rows


def _summary_text(
    agg: dict[str, Any],
    event_summary: dict[str, Any],
    speed_rows: list[dict[str, Any]],
    flag_rows: list[dict[str, str]],
    threshold_desc: str,
) -> str:
    lines = [
        f"Selected: {threshold_desc}",
        f"Coverage: {_fmt(agg['total_loaded_time_s'], 1)} s loaded braking, {event_summary['event_count']} events, {_fmt(agg['speed_min_kph'], 1)}-{_fmt(agg['speed_max_kph'], 1)} km/h",
    ]
    return "\n".join(lines)


def _event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "event_count": len(events),
        "median_duration_s": _nan_percentile(_values(events, "duration_s"), 50.0),
        "p90_duration_s": _nan_percentile(_values(events, "duration_s"), 90.0),
        "median_entry_speed_kph": _nan_percentile(_values(events, "entry_speed_kph"), 50.0),
        "p90_entry_speed_kph": _nan_percentile(_values(events, "entry_speed_kph"), 90.0),
        "median_peak_decel_mps2": _nan_percentile(_values(events, "peak_decel_mps2"), 50.0),
        "p90_peak_decel_mps2": _nan_percentile(_values(events, "peak_decel_mps2"), 90.0),
        "median_max_stroke_pct": _nan_percentile(_values(events, "max_stroke_pct"), 50.0),
        "p90_max_stroke_pct": _nan_percentile(_values(events, "max_stroke_pct"), 90.0),
        "median_p95_stroke_pct": _nan_percentile(_values(events, "p95_stroke_pct"), 50.0),
        "p90_p95_stroke_pct": _nan_percentile(_values(events, "p95_stroke_pct"), 90.0),
        "median_topout_frac_pct": _nan_percentile(_values(events, "topout_fraction_pct"), 50.0),
        "p90_topout_frac_pct": _nan_percentile(_values(events, "topout_fraction_pct"), 90.0),
        "median_rms_fork_velocity": _nan_percentile(_values(events, "rms_fork_velocity"), 50.0),
        "p90_rms_fork_velocity": _nan_percentile(_values(events, "rms_fork_velocity"), 90.0),
        "median_pack_down_pct": _nan_percentile(_values(events, "pack_down_pct"), 50.0),
        "p75_pack_down_pct": _nan_percentile(_values(events, "pack_down_pct"), 75.0),
        "p90_pack_down_pct": _nan_percentile(_values(events, "pack_down_pct"), 90.0),
    }


def _select_metric_rows(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_type = str(cfg["selected_threshold_type"]).lower()
    selected_label = str(cfg["selected_threshold_label"])
    selected_rows = [
        row
        for row in rows
        if row["threshold_type"] == selected_type and row["threshold_label"] == selected_label
    ]
    if not selected_rows:
        candidates = [row for row in rows if row["threshold_type"] == selected_type]
        if not candidates:
            candidates = rows
        if candidates:
            best = max(candidates, key=lambda row: _none_to_nan(row.get("decel_threshold_mps2")))
            selected_rows = [
                row
                for row in rows
                if row["threshold_type"] == best["threshold_type"] and row["threshold_label"] == best["threshold_label"]
            ]
    if selected_rows:
        first = selected_rows[0]
        selected = {
            "threshold_type": first["threshold_type"],
            "threshold_label": first["threshold_label"],
            "decel_percentile": first["decel_percentile"],
            "decel_threshold_mps2": float(np.nanmedian([row["decel_threshold_mps2"] for row in selected_rows])),
        }
    else:
        selected = {
            "threshold_type": selected_type,
            "threshold_label": selected_label,
            "decel_percentile": None,
            "decel_threshold_mps2": None,
        }
    return selected_rows, selected


def _make_speed_bins(speed_kph: np.ndarray, valid: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    values = speed_kph[valid & np.isfinite(speed_kph)]
    if values.size == 0:
        raise ValueError("No valid speed samples available for speed bins.")
    if cfg["speed_bin_mode"] == "percentile":
        pct_edges = np.asarray(cfg["speed_percentile_edges"], dtype=np.float64)
        edges = np.percentile(values, pct_edges)
        edges = _make_strictly_increasing(edges)
        labels = [
            f"P{pct_edges[index]:g}-P{pct_edges[index + 1]:g}: {edges[index]:.1f}-{edges[index + 1]:.1f} kph"
            for index in range(edges.size - 1)
        ]
        return edges, labels

    edges = _make_strictly_increasing(np.asarray(cfg["speed_edges_abs_kph"], dtype=np.float64))
    labels = [
        f"{edges[index]:.1f}-{edges[index + 1]:.1f} kph"
        for index in range(edges.size - 1)
    ]
    return edges, labels


def _make_decel_thresholds(decel_mps2: np.ndarray, valid: np.ndarray, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    valid_decel = valid & np.isfinite(decel_mps2) & (decel_mps2 > 0.0)
    if not np.any(valid_decel):
        raise ValueError("No valid positive deceleration samples available.")
    rows: list[dict[str, Any]] = []
    if cfg["threshold_mode"] in {"percentile", "both"}:
        for percentile in cfg["decel_percentiles"]:
            value = float(np.percentile(decel_mps2[valid_decel], percentile))
            rows.append(
                {
                    "threshold_type": "percentile",
                    "threshold_label": f"P{percentile:g}",
                    "decel_percentile": float(percentile),
                    "decel_threshold_mps2": value,
                }
            )
    if cfg["threshold_mode"] in {"absolute", "both"}:
        for threshold in cfg["absolute_decel_thresholds_mps2"]:
            rows.append(
                {
                    "threshold_type": "absolute",
                    "threshold_label": f"{threshold:.2f} ms2",
                    "decel_percentile": None,
                    "decel_threshold_mps2": float(threshold),
                }
            )
    if not rows:
        raise ValueError("No deceleration thresholds configured.")
    return rows


def _interpolate_speed_to_analog(wheel_df: pl.DataFrame, time_s: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    speed = np.full(time_s.shape, np.nan, dtype=np.float64)
    if wheel_df.is_empty() or "speed_kph" not in wheel_df.columns:
        return speed, {"source_count": 0, "valid_count": 0}
    if "host_time_s" in wheel_df.columns:
        t_speed = _series_to_numpy(wheel_df, "host_time_s", np.nan)
    elif "host_timestamp_us" in wheel_df.columns:
        t_speed = _series_to_numpy(wheel_df, "host_timestamp_us", np.nan) / 1_000_000.0
    else:
        return speed, {"source_count": wheel_df.height, "valid_count": 0}
    v_speed = _series_to_numpy(wheel_df, "speed_kph", np.nan)
    valid = (
        np.isfinite(t_speed)
        & np.isfinite(v_speed)
        & (v_speed >= cfg["speed_source_min_kph"])
        & (v_speed <= cfg["speed_source_max_kph"])
    )
    if np.count_nonzero(valid) < 2:
        return speed, {"source_count": int(v_speed.size), "valid_count": int(np.count_nonzero(valid))}
    t_valid = t_speed[valid]
    v_valid = v_speed[valid]
    order = np.argsort(t_valid, kind="mergesort")
    t_valid = t_valid[order]
    v_valid = v_valid[order]
    unique_t, inverse = np.unique(t_valid, return_inverse=True)
    sums = np.bincount(inverse, weights=v_valid)
    counts = np.bincount(inverse)
    unique_v = sums / counts
    if unique_t.size < 2:
        return speed, {"source_count": int(v_speed.size), "valid_count": int(unique_t.size)}
    speed = np.interp(time_s, unique_t, unique_v, left=np.nan, right=np.nan)
    return speed, {"source_count": int(v_speed.size), "valid_count": int(unique_t.size)}


def _time_seconds(frame: pl.DataFrame) -> np.ndarray:
    if "host_time_s" in frame.columns:
        return _series_to_numpy(frame, "host_time_s", np.nan)
    return _series_to_numpy(frame, "host_timestamp_us", np.nan) / 1_000_000.0


def _series_to_numpy(frame: pl.DataFrame, column: str, fill_value: float = np.nan) -> np.ndarray:
    if column not in frame.columns:
        return np.asarray([], dtype=np.float64)
    return np.asarray(frame.get_column(column).fill_null(fill_value).to_numpy(), dtype=np.float64)


def _dt_from_time(time_s: np.ndarray) -> np.ndarray:
    if time_s.size == 0:
        return np.asarray([], dtype=np.float64)
    positive = np.diff(time_s)
    positive = positive[np.isfinite(positive) & (positive > 0.0)]
    default_dt = float(np.median(positive)) if positive.size else 0.001
    dt_s = np.empty(time_s.shape, dtype=np.float64)
    dt_s[0] = default_dt
    if time_s.size > 1:
        diffs = np.diff(time_s)
        dt_s[1:] = np.where(np.isfinite(diffs) & (diffs > 0.0), diffs, default_dt)
    return dt_s


def _time_gradient(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(time_s)
    for start, end in _logical_runs(valid):
        if end - start + 1 < 3:
            continue
        idx = slice(start, end + 1)
        local_time = time_s[idx]
        local_values = values[idx]
        if np.any(np.diff(local_time) <= 0.0):
            dt = _median_dt(local_time, np.full(local_time.shape, np.nan))
            result[idx] = np.gradient(local_values, dt)
        else:
            result[idx] = np.gradient(local_values, local_time)
    return result


def _moving_average_nan(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64)
    finite = np.isfinite(values)
    weighted_values = np.where(finite, values, 0.0)
    weights = finite.astype(np.float64)
    pad = window // 2
    summed_values = np.convolve(np.pad(weighted_values, (pad, pad), mode="edge"), kernel, mode="valid")
    summed_weights = np.convolve(np.pad(weights, (pad, pad), mode="edge"), kernel, mode="valid")
    result = np.full(values.shape, np.nan, dtype=np.float64)
    nonzero = summed_weights > 0.0
    result[nonzero] = summed_values[nonzero] / summed_weights[nonzero]
    return result


def _window_samples(time_s: np.ndarray, window_s: float) -> int:
    dt = _median_dt(time_s, np.full(time_s.shape, np.nan))
    samples = max(1, int(round(float(window_s) / dt)))
    return samples + 1 if samples % 2 == 0 else samples


def _median_dt(time_s: np.ndarray, dt_s: np.ndarray) -> float:
    dt = dt_s[np.isfinite(dt_s) & (dt_s > 0.0)] if dt_s.size else np.asarray([], dtype=np.float64)
    if dt.size:
        return float(np.median(dt))
    diffs = np.diff(time_s)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    return float(np.median(diffs)) if diffs.size else 0.001


def _assign_speed_bins(speed: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bin_index = np.full(speed.shape, -1, dtype=np.int64)
    finite = np.isfinite(speed)
    bin_index[finite] = np.searchsorted(edges[1:-1], speed[finite], side="right") + 1
    bin_index[finite & ((speed < edges[0]) | (speed > edges[-1]))] = -1
    return bin_index


def _make_strictly_increasing(values: np.ndarray) -> np.ndarray:
    edges = np.asarray(values, dtype=np.float64).copy()
    if edges.size < 2:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    for index in range(1, edges.size):
        if not edges[index] > edges[index - 1]:
            edges[index] = edges[index - 1] + max(abs(edges[index - 1]) * 1e-9, 1e-6)
    return edges


def _logical_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    bool_mask = np.asarray(mask, dtype=bool)
    if bool_mask.size == 0 or not np.any(bool_mask):
        return []
    padded = np.concatenate(([False], bool_mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _merge_short_false_gaps(mask: np.ndarray, gap_samples: int) -> np.ndarray:
    merged = np.asarray(mask, dtype=bool).copy()
    if gap_samples <= 0 or not np.any(merged):
        return merged
    for start, end in _logical_runs(~merged):
        if start == 0 or end == merged.size - 1:
            continue
        if end - start + 1 <= gap_samples:
            merged[start : end + 1] = True
    return merged


def _index_mask(size: int, start: int, end: int) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    mask[start : end + 1] = True
    return mask


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return None
    total = float(np.sum(weights[finite]))
    if total <= 0.0:
        return None
    return float(np.sum(values[finite] * weights[finite]) / total)


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float | None:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return None
    total = float(np.sum(weights[finite]))
    if total <= 0.0:
        return None
    return float(np.sqrt(np.sum(np.square(values[finite]) * weights[finite]) / total))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float | None:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return None
    local_values = values[finite]
    local_weights = weights[finite]
    order = np.argsort(local_values, kind="mergesort")
    sorted_values = local_values[order]
    sorted_weights = local_weights[order]
    total = float(np.sum(sorted_weights))
    if total <= 0.0:
        return None
    target = float(np.clip(quantile, 0.0, 1.0)) * total
    index = int(np.searchsorted(np.cumsum(sorted_weights), target, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def _nan_percentile(values: np.ndarray, percentile: float) -> float | None:
    local = values[np.isfinite(values)]
    if local.size == 0:
        return None
    return float(np.percentile(local, percentile))


def _nan_max(values: np.ndarray) -> float | None:
    local = values[np.isfinite(values)]
    if local.size == 0:
        return None
    return float(np.max(local))


def _nan_mean(values: np.ndarray) -> float | None:
    local = values[np.isfinite(values)]
    if local.size == 0:
        return None
    return float(np.mean(local))


def _sum_time(dt_s: np.ndarray, mask: np.ndarray) -> float:
    local = dt_s[mask & np.isfinite(dt_s) & (dt_s > 0.0)]
    return float(np.sum(local)) if local.size else 0.0


def _time_fraction_pct(numerator_mask: np.ndarray, denominator_mask: np.ndarray, dt_s: np.ndarray) -> float | None:
    denominator = _sum_time(dt_s, denominator_mask)
    if denominator <= 0.0:
        return None
    numerator = _sum_time(dt_s, numerator_mask & denominator_mask)
    return 100.0 * numerator / denominator


def _metric_warning(row: dict[str, Any], cfg: dict[str, Any]) -> str:
    parts: list[str] = []
    if not row["enough_samples"]:
        parts.append("low total samples")
    if not row["enough_loaded_samples"]:
        parts.append("low loaded samples")
    if not row["enough_coasting_samples"]:
        parts.append("low coasting baseline")
    if not row["enough_events"]:
        parts.append("low event count")
    topout = row.get("topout_contamination_pct")
    if _isfinite(topout) and float(topout) >= cfg["topout_high_pct"]:
        parts.append("high topout contamination")
    return "; ".join(parts)


def _speed_bin_flag(row: dict[str, Any], cfg: dict[str, Any]) -> str:
    parts: list[str] = []
    if row.get("event_count", 0) < cfg["min_events_per_metric_bin"]:
        parts.append("few events")
    topout = row.get("topout_contamination_pct")
    if _isfinite(topout) and topout >= cfg["topout_high_pct"]:
        parts.append("topout high")
    elif _isfinite(topout) and topout >= cfg["topout_caution_pct"]:
        parts.append("topout watch")
    dive = row.get("braking_dive_index_pct")
    if _isfinite(dive) and dive >= cfg["dive_high_pct"]:
        parts.append("dive high")
    deep90 = row.get("deep90_frac_pct")
    deep80 = row.get("deep80_frac_pct")
    if _isfinite(deep90) and deep90 >= cfg["deep90_caution_pct"]:
        parts.append("near-bottom")
    elif _isfinite(deep80) and deep80 >= cfg["deep80_caution_pct"]:
        parts.append("deep stroke")
    margin = row.get("bottom_margin_p99_pct")
    if _isfinite(margin) and margin <= cfg["bottom_margin_high_pct"]:
        parts.append("low margin")
    rough = row.get("rough_braking_ratio")
    if _isfinite(rough) and rough >= cfg["rough_high_ratio"]:
        parts.append("rough high")
    elif _isfinite(rough) and rough >= cfg["rough_caution_ratio"]:
        parts.append("rough watch")
    pack = row.get("p75_pack_down_pct")
    if _isfinite(pack) and pack >= cfg["packdown_high_pct"]:
        parts.append("packdown watch")
    return ", ".join(parts) if parts else "baseline"


def _key_row(section: str, metric: str, value: Any, unit: str, status: str, meaning: str) -> dict[str, Any]:
    return {
        "section": section,
        "metric": metric,
        "value": value,
        "unit": unit,
        "status": status,
        "meaning": meaning,
    }


def _status_event_count(value: int, cfg: dict[str, Any]) -> str:
    return "WATCH" if value < cfg["min_events_per_metric_bin"] else "OK"


def _status_high_worse(value: Any, caution: float, high: float) -> str:
    if not _isfinite(value):
        return "NO DATA"
    if value >= high:
        return "HIGH"
    if value >= caution:
        return "WATCH"
    return "OK"


def _status_low_worse(value: Any, caution: float, high: float) -> str:
    if not _isfinite(value):
        return "NO DATA"
    if value <= high:
        return "HIGH"
    if value <= caution:
        return "WATCH"
    return "OK"


def _status_dive(value: Any, cfg: dict[str, Any]) -> str:
    if not _isfinite(value):
        return "NO DATA"
    if value < cfg["dive_low_pct"]:
        return "LOW"
    if value > cfg["dive_high_pct"]:
        return "HIGH"
    return "MODERATE"


def _status_packdown(value: Any, cfg: dict[str, Any]) -> str:
    if not _isfinite(value):
        return "NO DATA"
    if value >= cfg["packdown_high_pct"]:
        return "WATCH"
    if value >= cfg["packdown_caution_pct"]:
        return "INFO"
    return "OK"


def _main_read(agg: dict[str, Any], event_summary: dict[str, Any], cfg: dict[str, Any]) -> str:
    parts: list[str] = []
    if _isfinite(agg["deep90_weighted_pct"]) and _isfinite(agg["bottom_margin_p99_min_pct"]):
        if agg["deep90_weighted_pct"] < cfg["deep90_caution_pct"] and agg["bottom_margin_p99_min_pct"] > cfg["bottom_margin_caution_pct"]:
            parts.append("good stroke reserve")
        elif agg["bottom_margin_p99_min_pct"] <= cfg["bottom_margin_high_pct"]:
            parts.append("small bottom margin")
        else:
            parts.append("some deep stroke use")
    dive = agg["braking_dive_weighted_pct"]
    if _isfinite(dive):
        if dive < cfg["dive_low_pct"]:
            parts.append("low brake dive")
        elif dive > cfg["dive_high_pct"]:
            parts.append("high brake dive")
        else:
            parts.append("moderate brake dive")
    topout = agg["topout_max_pct"]
    if _isfinite(topout):
        if topout >= cfg["topout_high_pct"]:
            parts.append("topout contamination high")
        elif topout >= cfg["topout_caution_pct"]:
            parts.append("some topout contamination")
        else:
            parts.append("topout clean")
    rough = agg["rough_ratio_max"]
    if _isfinite(rough):
        if rough >= cfg["rough_high_ratio"]:
            parts.append("rough braking high")
        elif rough >= cfg["rough_caution_ratio"]:
            parts.append("rough braking elevated")
        else:
            parts.append("roughness controlled")
    if _isfinite(event_summary["p75_pack_down_pct"]) and event_summary["p75_pack_down_pct"] >= cfg["packdown_high_pct"]:
        parts.append("some events pack down")
    return "; ".join(parts) if parts else "not enough braking evidence"


def _values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_none_to_nan(row.get(key)) for row in rows], dtype=np.float64)


def _finite_sum(values: list[Any]) -> float:
    local = np.asarray([_none_to_nan(value) for value in values], dtype=np.float64)
    local = local[np.isfinite(local)]
    return float(np.sum(local)) if local.size else 0.0


def _finite_min(values: list[Any]) -> float | None:
    local = np.asarray([_none_to_nan(value) for value in values], dtype=np.float64)
    local = local[np.isfinite(local)]
    return float(np.min(local)) if local.size else None


def _finite_max(values: list[Any]) -> float | None:
    local = np.asarray([_none_to_nan(value) for value in values], dtype=np.float64)
    local = local[np.isfinite(local)]
    return float(np.max(local)) if local.size else None


def _subtract_optional(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _safe_divide(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or not np.isfinite(left) or not np.isfinite(right) or abs(right) < 1e-12:
        return None
    return float(left / right)


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _float_list(value: Any, default: list[float]) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return list(default)
    result: list[float] = []
    for item in value:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            result.append(number)
    return result if result else list(default)


def _none_to_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _finite_or_none(value: Any) -> float | None:
    result = _none_to_nan(value)
    return float(result) if np.isfinite(result) else None


def _isfinite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _fmt(value: Any, digits: int) -> str:
    if not _isfinite(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _fmt_signed(value: Any, digits: int) -> str:
    if not _isfinite(value):
        return "n/a"
    return f"{float(value):+.{digits}f}"
