from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl

from scripts.postprocess_gui_app.backend.analysis_service import (
    apply_travel_reference,
    build_derived_analog,
    channel_series,
    compute_balance_analysis,
    compute_channel_metrics,
    compute_travel_histogram,
    compute_velocity_histogram,
)
from scripts.postprocess_gui_app.backend.context_analysis import (
    BALANCE_RESPONSE,
    DEFAULT_CONTEXT_CONFIG,
    FRONT_SHARE_RESPONSE,
    FRONT_VELOCITY_RESPONSE,
    LONGITUDINAL_ACCEL_SOURCE,
    WHEEL_SPEED_SOURCE,
    build_breakdown_analysis,
    build_context_dataset,
    compute_context_heatmap,
    summarize_context_bins,
)
from scripts.postprocess_gui_app.backend import session_service
from scripts.postprocess_gui_app.backend.export_service import export_bin_to_directory, export_session
from scripts.postprocess_gui_app.backend.occupancy_service import compute_occupancy_grid
from scripts.postprocess_gui_app.backend.quicklook_service import run_quicklook
from scripts.postprocess_gui_app.backend.session_config import migrate_session_config
from scripts.postprocess_gui_app.backend.session_metadata_service import (
    auto_assign_set_labels,
    load_session_metadata,
    save_session_metadata,
    scan_session_database,
)
from scripts.postprocess_gui_app.backend.session_service import (
    bin_export_is_current,
    open_export_session,
    sanitize_wheel_speed_frame,
)
from scripts.log_parser import STAT_STRUCT, decode_stat


class PostprocessBackendTests(unittest.TestCase):
    def _workspace_run_dir(self, prefix: str) -> Path:
        base = Path(".codex_tmp") / "test_tmp" / f"{prefix}_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _bin_fixture_path(self, name: str) -> Path:
        candidates = (Path(f"{name}.BIN"), Path("data") / f"{name}.BIN")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"missing BIN fixture for {name}")

    def _copy_export(self, name: str) -> Path:
        temp_dir = self._workspace_run_dir(name.lower())
        dst = temp_dir / name
        export_bin_to_directory(bin_path=self._bin_fixture_path(name), output_dir=dst)
        return dst

    def _synthetic_derived_df(self, time_s: np.ndarray) -> pl.DataFrame:
        time_s = np.asarray(time_s, dtype=np.float64)
        dt_s = np.full(time_s.shape, time_s[1] - time_s[0] if time_s.size > 1 else 0.02, dtype=np.float64)
        return pl.DataFrame(
            {
                "host_timestamp_us": np.round(time_s * 1_000_000.0).astype(np.int64),
                "dt_s": dt_s,
                "front_raw": np.linspace(1000.0, 1020.0, time_s.size),
                "rear_raw": np.linspace(1100.0, 1080.0, time_s.size),
                "front_travel_counts": np.linspace(10.0, 30.0, time_s.size),
                "front_filtered_travel_counts": np.linspace(10.0, 30.0, time_s.size),
                "rear_travel_counts": np.linspace(8.0, 24.0, time_s.size),
                "rear_filtered_travel_counts": np.linspace(8.0, 24.0, time_s.size),
                "front_velocity_counts_per_s_filtered": np.linspace(-5.0, 5.0, time_s.size),
                "front_velocity_counts_per_s": np.linspace(-5.0, 5.0, time_s.size),
                "rear_velocity_counts_per_s_filtered": np.linspace(4.0, -4.0, time_s.size),
                "rear_velocity_counts_per_s": np.linspace(4.0, -4.0, time_s.size),
            }
        )

    def _write_small_summary(self, export_dir: Path, session_id: str, start_epoch: int) -> dict[str, object]:
        summary = {
            "source_path": f"data\\{session_id}.BIN",
            "header": {
                "start_epoch": start_epoch,
                "analog_resolution_bits": 12,
            },
            "timing": {"duration_us": 12_500_000},
            "counts": {"analog_rows": 4, "wheel_rows": 0, "stat_rows": 1},
            "record_counts": {"ANLG": 4, "STAT": 1},
        }
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    def test_open_export_session_builds_derived_data(self) -> None:
        export_dir = self._copy_export("LOG00016")
        bundle = open_export_session(export_dir)
        self.assertGreater(bundle.derived_df.height, 0)
        self.assertIn("front_velocity_counts_per_s_filtered", bundle.derived_df.columns)
        self.assertIn("front_travel_stroke_pct", bundle.derived_df.columns)
        self.assertIn("rear_velocity_counts_per_s_filtered", bundle.derived_df.columns)
        self.assertIn("speed_kph", bundle.wheel_df.columns)
        self.assertIn("accel_x_g", bundle.imu_frame_df.columns)
        self.assertGreater(bundle.imu_frame_df.height, 0)
        self.assertGreater(bundle.session_config["front"]["zero_count"], 0)
        self.assertEqual(bundle.session_config["schema_version"], 8)
        self.assertEqual(bundle.session_config["front"]["full_scale_mm"], 300.0)
        self.assertEqual(bundle.session_config["front"]["sensor_full_scale_mm"], 635.0)
        self.assertEqual(bundle.session_config["front"]["reference_method"], "percentile")
        self.assertEqual(bundle.session_config["front"]["reference_percentile"], 0.001)
        self.assertEqual(bundle.session_config["front"]["velocity_filter_window"], 3)
        self.assertEqual(bundle.session_config["rear"]["velocity_filter_window"], 3)
        self.assertEqual(bundle.session_config["plot_defaults"]["occupancy_travel_bins"], 400)
        self.assertEqual(bundle.session_config["plot_defaults"]["occupancy_velocity_bins"], 400)
        self.assertEqual(bundle.session_config["plot_defaults"]["occupancy_color_scale"], "sqrt")
        self.assertEqual(bundle.session_config["plot_defaults"]["velocity_axis_mode"], "linear")

    def test_sanitize_wheel_speed_frame_rejects_implausible_spikes_and_preserves_raw(self) -> None:
        wheel_df = pl.DataFrame(
            {
                "host_time_s": [1.0, 2.0, 3.0, 4.0],
                "speed_m_s": [10.0 / 3.6, 1200.0 / 3.6, -1.0 / 3.6, None],
                "speed_kph": [10.0, 1200.0, -1.0, None],
                "speed_mph": [6.21371, 745.6452, -0.621371, None],
            }
        )

        filtered = sanitize_wheel_speed_frame(wheel_df, max_speed_kph=200.0)

        speed = filtered.get_column("speed_kph").to_numpy()
        raw_speed = filtered.get_column("speed_kph_raw").to_numpy()
        rejected = filtered.get_column("speed_sanity_rejected").to_numpy()
        self.assertAlmostEqual(float(speed[0]), 10.0, places=6)
        self.assertTrue(np.isnan(speed[1]))
        self.assertTrue(np.isnan(speed[2]))
        self.assertTrue(np.isnan(speed[3]))
        self.assertAlmostEqual(float(raw_speed[1]), 1200.0, places=6)
        self.assertEqual(rejected.tolist(), [False, True, True, False])

        filtered_again = sanitize_wheel_speed_frame(filtered, max_speed_kph=200.0)
        self.assertAlmostEqual(float(filtered_again.get_column("speed_kph_raw")[1]), 1200.0, places=6)
        self.assertTrue(bool(filtered_again.get_column("speed_sanity_rejected")[1]))

    def test_migrate_session_config_locks_front_full_scale_to_bike_stroke(self) -> None:
        migrated = migrate_session_config(
            {
                "schema_version": 4,
                "front": {
                    "zero_count": 1234,
                    "invert": False,
                    "mm_per_count": None,
                    "full_scale_mm": 615.0,
                    "sensor_full_scale_mm": None,
                    "travel_reference": "auto",
                    "velocity_filter_window": 3,
                },
                "rear": {
                    "zero_count": 1000,
                    "invert": False,
                    "mm_per_count": None,
                    "full_scale_mm": None,
                    "sensor_full_scale_mm": None,
                    "travel_reference": "auto",
                    "velocity_filter_window": 3,
                },
                "plot_defaults": {
                    "selected_channel": "front",
                    "occupancy_travel_bins": 400,
                    "occupancy_velocity_bins": 400,
                    "occupancy_color_scale": "sqrt",
                },
            }
        )

        self.assertEqual(migrated["schema_version"], 8)
        self.assertEqual(migrated["front"]["full_scale_mm"], 300.0)
        self.assertEqual(migrated["front"]["sensor_full_scale_mm"], 635.0)
        self.assertEqual(migrated["front"]["reference_method"], "percentile")
        self.assertEqual(migrated["front"]["reference_percentile"], 0.001)
        self.assertEqual(migrated["front"]["reference_window_samples"], 3)
        self.assertIsNone(migrated["front"]["manual_reference_count"])
        self.assertEqual(migrated["plot_defaults"]["velocity_axis_mode"], "linear")
        self.assertEqual(migrated["breakdown"]["near_bottom_threshold_pct"], 95.0)
        self.assertEqual(migrated["breakdown"]["bottom_out_threshold_pct"], 99.0)
        self.assertIn("high_compression_velocity_min_pct_s", migrated["breakdown"])

    def test_apply_travel_reference_min_uses_absolute_extreme(self) -> None:
        position_counts = np.asarray([0.0, 190.0, 191.0, 192.0, 193.0], dtype=np.float64)

        travel, reference_used = apply_travel_reference(position_counts, "min")

        self.assertEqual(reference_used, "relative-from-absolute-min")
        self.assertAlmostEqual(float(np.min(travel)), 0.0, places=6)
        self.assertAlmostEqual(float(np.max(travel)), 193.0, places=6)

    def test_apply_travel_reference_robust_percentile_ignores_single_low_outlier(self) -> None:
        position_counts = np.asarray([0.0] + [100.0] * 100 + [120.0], dtype=np.float64)

        travel, reference_used = apply_travel_reference(
            position_counts,
            "robust_min",
            reference_method="percentile",
            reference_percentile=1.0,
        )

        self.assertEqual(reference_used, "relative-from-robust-p1-min")
        self.assertLess(float(np.min(travel)), 0.0)
        self.assertAlmostEqual(float(travel[1]), 0.0, places=6)

    def test_apply_travel_reference_manual_anchor_overrides_robust_reference(self) -> None:
        position_counts = np.asarray([90.0, 100.0, 110.0], dtype=np.float64)

        travel, reference_used = apply_travel_reference(
            position_counts,
            "robust_min",
            manual_reference_count=95.0,
        )

        self.assertEqual(reference_used, "relative-from-manual-min")
        self.assertEqual(travel.tolist(), [-5.0, 5.0, 15.0])

    def test_apply_travel_reference_sustained_min_requires_window(self) -> None:
        position_counts = np.asarray([0.0, 100.0, 101.0, 102.0, 103.0], dtype=np.float64)

        travel, reference_used = apply_travel_reference(
            position_counts,
            "robust_min",
            reference_method="sustained",
            reference_window_samples=3,
        )

        self.assertEqual(reference_used, "relative-from-robust-sustained-min-n3")
        self.assertAlmostEqual(float(travel[1]), -1.0, places=6)

    def test_session_metadata_defaults_and_scan_database(self) -> None:
        root_dir = self._workspace_run_dir("session_metadata")
        exports_dir = root_dir / "exports"
        first_export = exports_dir / "LOG00016"
        second_export = exports_dir / "LOG00036"
        first_summary = self._write_small_summary(first_export, "LOG00016", 1_776_470_400)
        second_summary = self._write_small_summary(second_export, "LOG00036", 1_776_556_800)

        defaults = load_session_metadata(first_export, first_summary, Path("data/LOG00016.BIN"))
        self.assertEqual(defaults["set_label"], "LOG00016")
        self.assertTrue(defaults["set_label_auto"])
        self.assertEqual(defaults["date"], "2026-04-18")

        customized = dict(defaults)
        customized["track"] = "Test Track"
        customized["comment"] = "dry"
        save_session_metadata(first_export, customized)

        entries = scan_session_database(exports_dir)

        self.assertEqual([entry.session_id for entry in entries], ["LOG00036", "LOG00016"])
        first_entry = entries[1]
        self.assertEqual(first_entry.metadata["track"], "Test Track")
        self.assertEqual(first_entry.metadata["comment"], "dry")
        self.assertEqual(entries[0].metadata["set_label"], "LOG00036")

    def test_auto_assign_set_labels_numbers_sessions_by_timestamp_per_date(self) -> None:
        root_dir = self._workspace_run_dir("auto_set_labels")
        exports_dir = root_dir / "exports"
        first_export = exports_dir / "LOG00010"
        second_export = exports_dir / "LOG00020"
        third_export = exports_dir / "LOG00030"
        first_summary = self._write_small_summary(first_export, "LOG00010", 1_776_470_400)
        second_summary = self._write_small_summary(second_export, "LOG00020", 1_776_470_500)
        third_summary = self._write_small_summary(third_export, "LOG00030", 1_776_470_600)

        custom = load_session_metadata(second_export, second_summary, Path("data/LOG00020.BIN"))
        custom["set_label"] = "Race setup"
        custom["set_label_auto"] = False
        save_session_metadata(second_export, custom)

        updated = auto_assign_set_labels(exports_dir)
        entries = scan_session_database(exports_dir)
        metadata_by_id = {entry.session_id: entry.metadata for entry in entries}

        self.assertEqual(metadata_by_id["LOG00010"]["set_label"], "Set 1")
        self.assertTrue(metadata_by_id["LOG00010"]["set_label_auto"])
        self.assertEqual(metadata_by_id["LOG00020"]["set_label"], "Race setup")
        self.assertFalse(metadata_by_id["LOG00020"]["set_label_auto"])
        self.assertEqual(metadata_by_id["LOG00030"]["set_label"], "Set 3")
        self.assertIn(first_export.resolve(), updated)
        self.assertIn(third_export.resolve(), updated)
        self.assertNotIn(second_export.resolve(), updated)

    def test_build_derived_with_mm_scaling_adds_mm_columns(self) -> None:
        export_dir = self._copy_export("LOG00016")
        bundle = open_export_session(export_dir)
        config = copy.deepcopy(bundle.session_config)
        config["front"]["mm_per_count"] = 0.1
        config["rear"]["mm_per_count"] = 0.2
        derived_df, _ = build_derived_analog(export_dir, config, analog_df=bundle.analog_df, force=True)

        self.assertIn("front_travel_mm", derived_df.columns)
        self.assertIn("rear_travel_mm", derived_df.columns)
        first_counts = float(derived_df.get_column("front_travel_counts")[0])
        first_mm = float(derived_df.get_column("front_travel_mm")[0])
        self.assertAlmostEqual(first_mm, first_counts * 0.1, places=6)

    def test_channel_series_supports_stroke_percent_mode(self) -> None:
        export_dir = self._copy_export("LOG00016")
        bundle = open_export_session(export_dir)

        front_percent = channel_series(bundle.derived_df, "front", series_mode="stroke_percent")
        rear_percent = channel_series(bundle.derived_df, "rear", series_mode="stroke_percent")

        self.assertEqual(front_percent["travel_units"], "%")
        self.assertEqual(front_percent["velocity_units"], "%/s")
        self.assertEqual(front_percent["travel_label"], "Stroke")
        self.assertEqual(rear_percent["travel_units"], "%")
        self.assertGreater(np.nanmax(front_percent["travel"]), 0.0)

    def test_compute_occupancy_grid_preserves_total_weight_with_explicit_ranges(self) -> None:
        export_dir = self._copy_export("LOG00016")
        bundle = open_export_session(export_dir)
        series = channel_series(bundle.derived_df, "front")
        finite = (
            np.isfinite(series["travel"])
            & np.isfinite(series["velocity"])
            & np.isfinite(series["dt_s"])
            & (series["dt_s"] > 0)
        )
        occupancy = compute_occupancy_grid(
            derived_df=bundle.derived_df,
            channel="front",
            travel_bins=64,
            velocity_bins=80,
            travel_min=float(np.min(series["travel"][finite])),
            travel_max=float(np.max(series["travel"][finite])),
            velocity_min=float(np.min(series["velocity"][finite])),
            velocity_max=float(np.max(series["velocity"][finite])),
        )

        self.assertEqual(occupancy["histogram"].shape, (64, 80))
        self.assertAlmostEqual(occupancy["total_occupancy_s"], float(np.sum(series["dt_s"][finite])), places=6)

    def test_compute_occupancy_grid_signed_log_keeps_velocity_tails(self) -> None:
        derived_df = pl.DataFrame(
            {
                "host_timestamp_us": [0, 1_000_000, 2_000_000, 3_000_000],
                "dt_s": [1.0, 1.0, 1.0, 1.0],
                "front_raw": [1200.0, 1400.0, 1600.0, 1500.0],
                "front_travel_stroke_pct": [5.0, 35.0, 60.0, 95.0],
                "front_filtered_travel_stroke_pct": [5.0, 35.0, 60.0, 95.0],
                "front_velocity_stroke_pct_per_s_filtered": [-2500.0, -25.0, 40.0, 3000.0],
            }
        )

        occupancy = compute_occupancy_grid(
            derived_df=derived_df,
            channel="front",
            travel_bins=20,
            velocity_bins=20,
            series_mode="stroke_percent",
            velocity_axis_mode="signed_log",
        )

        self.assertEqual(occupancy["velocity_axis_mode"], "signed_log")
        self.assertEqual(occupancy["velocity_data_range"], (-3000.0, 3000.0))
        self.assertGreater(occupancy["velocity_range"][1], 3.0)
        self.assertAlmostEqual(float(np.sum(occupancy["histogram"])), 4.0, places=6)
        tick_labels = [label for _position, label in occupancy["velocity_axis_ticks"]]
        self.assertIn("R 1k", tick_labels)
        self.assertIn("C 1k", tick_labels)

    def test_compute_metrics_contains_expected_rows(self) -> None:
        export_dir = self._copy_export("LOG00016")
        bundle = open_export_session(export_dir)
        metrics = compute_channel_metrics(bundle.derived_df, "front", int(bundle.summary["header"]["analog_resolution_bits"]))
        metric_names = {metric["metric"] for metric in metrics}
        self.assertIn("Used stroke", metric_names)
        self.assertIn("Peak compression velocity", metric_names)
        self.assertIn("ADC rail-near low", metric_names)
        self.assertIn("Raw count span", metric_names)
        self.assertIn("ADC span used", metric_names)

    def test_compute_metrics_adds_visible_occupancy_deciles_and_categories(self) -> None:
        derived_df = pl.DataFrame(
            {
                "host_timestamp_us": [0, 1_000_000, 2_000_000, 3_000_000, 4_000_000],
                "dt_s": [1.0, 1.0, 1.0, 1.0, 1.0],
                "front_raw": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
                "rear_raw": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0],
                "front_travel_counts": [0.0, 10.0, 10.0, 10.0, 40.0],
                "front_filtered_travel_counts": [0.0, 10.0, 10.0, 10.0, 40.0],
                "front_velocity_counts_per_s_filtered": [0.0, 10.0, 10.0, 10.0, 10.0],
                "rear_travel_counts": [0.0, 10.0, 10.0, 10.0, 40.0],
                "rear_filtered_travel_counts": [0.0, 10.0, 10.0, 10.0, 40.0],
                "rear_velocity_counts_per_s_filtered": [0.0, 10.0, 10.0, 10.0, 10.0],
            }
        )

        metrics = compute_channel_metrics(derived_df, "front", 12)
        visible_metrics = [metric for metric in metrics if metric.get("metrics_tab_visible", True)]
        visible_names = [str(metric["metric"]) for metric in visible_metrics]
        riding_metrics = [metric for metric in visible_metrics if metric.get("category") == "riding"]
        hardware_metrics = [metric for metric in visible_metrics if metric.get("category") == "hardware"]
        occupancy_metrics = [metric for metric in riding_metrics if str(metric["metric"]).startswith("Occupancy P")]

        self.assertEqual(len(occupancy_metrics), 10)
        self.assertIn("Occupancy P0-10", visible_names)
        self.assertIn("Occupancy P90-100", visible_names)
        self.assertIn("Mean occupancy", visible_names)
        self.assertIn("Median occupancy", visible_names)
        self.assertIn("Mode occupancy", visible_names)
        self.assertIn("Occupancy geometric SD", visible_names)
        self.assertNotIn("Used stroke", visible_names)
        self.assertNotIn("Time in bottom 10%", visible_names)
        self.assertNotIn("Time in top 10%", visible_names)
        self.assertEqual(str(riding_metrics[0]["metric"]), "Session duration")
        self.assertEqual(str(hardware_metrics[0]["metric"]), "Analog sample count")
        self.assertEqual(
            visible_names.index("Occupancy P0-10"),
            visible_names.index("Occupancy geometric SD") + 1,
        )
        self.assertGreater(visible_names.index("Peak compression velocity"), visible_names.index("Occupancy P90-100"))
        self.assertAlmostEqual(sum(float(metric["value"]) for metric in occupancy_metrics if metric["value"] is not None), 100.0, places=6)
        riding_metric_map = {str(metric["metric"]): metric for metric in riding_metrics}
        self.assertAlmostEqual(float(riding_metric_map["Mean occupancy"]["value"]), 35.0, places=6)
        self.assertAlmostEqual(float(riding_metric_map["Median occupancy"]["value"]), 25.0, places=6)
        self.assertAlmostEqual(float(riding_metric_map["Mode occupancy"]["value"]), 25.5, places=6)
        self.assertGreater(float(riding_metric_map["Occupancy geometric SD"]["value"]), 1.0)

    def test_compute_metrics_supports_stroke_percent_series_mode(self) -> None:
        derived_df = pl.DataFrame(
            {
                "host_timestamp_us": [0, 1_000_000, 2_000_000, 3_000_000],
                "dt_s": [1.0, 1.0, 1.0, 1.0],
                "front_raw": [1200.0, 1400.0, 1600.0, 1500.0],
                "rear_raw": [1300.0, 1450.0, 1550.0, 1650.0],
                "front_travel_stroke_pct": [10.0, 35.0, 60.0, 25.0],
                "front_filtered_travel_stroke_pct": [10.0, 35.0, 60.0, 25.0],
                "front_velocity_stroke_pct_per_s_filtered": [0.0, 22.0, -18.0, 8.0],
                "rear_travel_stroke_pct": [12.0, 28.0, 44.0, 18.0],
                "rear_filtered_travel_stroke_pct": [12.0, 28.0, 44.0, 18.0],
                "rear_velocity_stroke_pct_per_s_filtered": [0.0, 15.0, -12.0, 6.0],
            }
        )

        metrics = compute_channel_metrics(derived_df, "front", 12, series_mode="stroke_percent")
        metric_map = {metric["metric"]: metric for metric in metrics}

        self.assertEqual(metric_map["Used stroke"]["units"], "%")
        self.assertEqual(metric_map["Peak compression velocity"]["units"], "%/s")
        self.assertAlmostEqual(float(metric_map["Used stroke"]["value"]), 50.0, places=6)
        self.assertAlmostEqual(float(metric_map["Mean occupancy"]["value"]), 32.5, places=6)
        self.assertAlmostEqual(float(metric_map["Median occupancy"]["value"]), 25.0, places=6)
        self.assertAlmostEqual(float(metric_map["Occupancy P10-20"]["value"]), 25.0, places=6)
        self.assertAlmostEqual(float(metric_map["Occupancy P20-30"]["value"]), 25.0, places=6)
        self.assertAlmostEqual(float(metric_map["Occupancy P30-40"]["value"]), 25.0, places=6)
        self.assertAlmostEqual(float(metric_map["Occupancy P60-70"]["value"]), 25.0, places=6)

    def test_stroke_percent_occupancy_metrics_clip_outside_tails(self) -> None:
        derived_df = pl.DataFrame(
            {
                "host_timestamp_us": [0, 1_000_000, 2_000_000, 3_000_000],
                "dt_s": [1.0, 1.0, 1.0, 1.0],
                "front_raw": [1200.0, 1400.0, 1600.0, 1500.0],
                "front_travel_stroke_pct": [-5.0, 5.0, 50.0, 105.0],
                "front_filtered_travel_stroke_pct": [-5.0, 5.0, 50.0, 105.0],
                "front_velocity_stroke_pct_per_s_filtered": [0.0, 22.0, -18.0, 8.0],
            }
        )

        metrics = compute_channel_metrics(derived_df, "front", 12, series_mode="stroke_percent")
        metric_map = {metric["metric"]: metric for metric in metrics}
        occupancy_metrics = [metric for metric in metrics if str(metric["metric"]).startswith("Occupancy P")]

        self.assertAlmostEqual(sum(float(metric["value"]) for metric in occupancy_metrics), 100.0, places=6)
        self.assertAlmostEqual(float(metric_map["Mean occupancy"]["value"]), 38.75, places=6)
        self.assertAlmostEqual(float(metric_map["Occupancy P0-10"]["value"]), 50.0, places=6)
        self.assertAlmostEqual(float(metric_map["Occupancy P50-60"]["value"]), 25.0, places=6)
        self.assertAlmostEqual(float(metric_map["Occupancy P90-100"]["value"]), 25.0, places=6)

    def test_mode_occupancy_uses_smoothed_distribution_not_noisy_single_bin(self) -> None:
        values = [12.0] * 4 + [30.0] * 3 + [31.0] * 3 + [32.0] * 3
        derived_df = pl.DataFrame(
            {
                "host_timestamp_us": [index * 1_000_000 for index in range(len(values))],
                "dt_s": [1.0] * len(values),
                "front_raw": [1200.0] * len(values),
                "front_travel_stroke_pct": values,
                "front_filtered_travel_stroke_pct": values,
                "front_velocity_stroke_pct_per_s_filtered": [0.0] * len(values),
            }
        )

        metrics = compute_channel_metrics(derived_df, "front", 12, series_mode="stroke_percent")
        metric_map = {metric["metric"]: metric for metric in metrics}

        self.assertGreater(float(metric_map["Mode occupancy"]["value"]), 29.0)
        self.assertLess(float(metric_map["Mode occupancy"]["value"]), 33.0)

    def test_histograms_support_relative_occupancy_percent(self) -> None:
        derived_df = pl.DataFrame(
            {
                "host_timestamp_us": [0, 1_000_000, 2_000_000, 3_000_000],
                "dt_s": [1.0, 1.0, 1.0, 1.0],
                "front_raw": [1200.0, 1400.0, 1600.0, 1500.0],
                "rear_raw": [1300.0, 1450.0, 1550.0, 1650.0],
                "front_travel_stroke_pct": [10.0, 35.0, 60.0, 25.0],
                "front_filtered_travel_stroke_pct": [10.0, 35.0, 60.0, 25.0],
                "front_velocity_stroke_pct_per_s_filtered": [0.0, 22.0, -18.0, 8.0],
                "rear_travel_stroke_pct": [12.0, 28.0, 44.0, 18.0],
                "rear_filtered_travel_stroke_pct": [12.0, 28.0, 44.0, 18.0],
                "rear_velocity_stroke_pct_per_s_filtered": [0.0, 15.0, -12.0, 6.0],
            }
        )

        travel_hist = compute_travel_histogram(
            derived_df,
            "front",
            series_mode="stroke_percent",
            relative_occupancy=True,
        )
        velocity_hist = compute_velocity_histogram(
            derived_df,
            "front",
            series_mode="stroke_percent",
            relative_occupancy=True,
        )

        self.assertEqual(travel_hist["occupancy_label"], "Relative occupancy")
        self.assertEqual(travel_hist["occupancy_units"], "%")
        self.assertAlmostEqual(float(np.sum(travel_hist["histogram"])), 100.0, places=6)
        self.assertEqual(velocity_hist["occupancy_label"], "Relative occupancy")
        self.assertEqual(velocity_hist["occupancy_units"], "%")
        self.assertAlmostEqual(float(np.sum(velocity_hist["all"])), 100.0, places=6)
        self.assertAlmostEqual(float(np.sum(velocity_hist["positive"] + velocity_hist["negative"])), 100.0, places=6)

    def test_velocity_histogram_signed_log_uses_rebound_compression_axis(self) -> None:
        derived_df = pl.DataFrame(
            {
                "host_timestamp_us": [0, 1_000_000, 2_000_000, 3_000_000],
                "dt_s": [1.0, 1.0, 1.0, 1.0],
                "front_raw": [1200.0, 1400.0, 1600.0, 1500.0],
                "front_travel_stroke_pct": [10.0, 35.0, 60.0, 25.0],
                "front_filtered_travel_stroke_pct": [10.0, 35.0, 60.0, 25.0],
                "front_velocity_stroke_pct_per_s_filtered": [-1200.0, -80.0, 40.0, 1600.0],
            }
        )

        velocity_hist = compute_velocity_histogram(
            derived_df,
            "front",
            series_mode="stroke_percent",
            relative_occupancy=True,
            velocity_axis_mode="signed_log",
        )

        self.assertEqual(velocity_hist["velocity_axis_mode"], "signed_log")
        self.assertIn("Rebound | Compression", velocity_hist["label"])
        self.assertAlmostEqual(float(np.sum(velocity_hist["all"])), 100.0, places=6)
        tick_labels = [label for _position, label in velocity_hist["velocity_axis_ticks"]]
        self.assertIn("R 1k", tick_labels)
        self.assertIn("C 1k", tick_labels)

    def test_compute_balance_analysis_returns_normalized_metrics(self) -> None:
        export_dir = self._copy_export("LOG00016")
        bundle = open_export_session(export_dir)
        balance = compute_balance_analysis(bundle.derived_df)

        self.assertEqual(balance["front_percent"].shape[0], bundle.derived_df.height)
        self.assertEqual(balance["rear_percent"].shape[0], bundle.derived_df.height)
        self.assertIsNotNone(balance["front_used_stroke"])
        self.assertIsNotNone(balance["rear_used_stroke"])
        self.assertGreaterEqual(balance["active_time_s"], 0.0)
        self.assertEqual(balance["front_travel_units"], channel_series(bundle.derived_df, "front")["travel_units"])
        self.assertEqual(balance["rear_travel_units"], channel_series(bundle.derived_df, "rear")["travel_units"])

    def test_context_dataset_holds_wheel_speed_with_timeout_gaps(self) -> None:
        time_s = np.arange(0.0, 9.5, 0.5)
        derived_df = self._synthetic_derived_df(time_s)
        wheel_df = pl.DataFrame(
            {
                "host_time_s": [1.0, 2.0, 5.0, 7.0, 8.0],
                "period_s": [None, 1.0, 2.0, 1.0, 1.0],
                "speed_kph": [None, 10.0, 20.0, 25.0, 30.0],
            }
        )

        context_df, meta = build_context_dataset(
            derived_df=derived_df,
            wheel_df=wheel_df,
            imu_frame_df=pl.DataFrame(),
            context_config=DEFAULT_CONTEXT_CONFIG,
        )

        speed = context_df.get_column("speed_kph_ctx").to_numpy()
        self.assertAlmostEqual(float(speed[0]), 0.0, places=6)
        self.assertAlmostEqual(float(speed[1]), 0.0, places=6)
        self.assertTrue(np.isnan(speed[3]))
        self.assertAlmostEqual(float(speed[4]), 10.0, places=6)
        self.assertTrue(np.isnan(speed[8]))
        self.assertAlmostEqual(float(speed[10]), 20.0, places=6)
        self.assertAlmostEqual(float(speed[14]), 25.0, places=6)
        self.assertAlmostEqual(float(speed[16]), 30.0, places=6)
        self.assertGreater(meta["wheel_speed"]["coverage_pct"], 10.0)
        self.assertTrue(meta["sources"][WHEEL_SPEED_SOURCE]["usable"])

    def test_context_dataset_interpolates_imu_and_builds_roughness(self) -> None:
        time_s = np.arange(0.0, 0.34, 0.02)
        derived_df = self._synthetic_derived_df(time_s)
        imu_frame_df = pl.DataFrame(
            {
                "estimated_host_time_s": [None, 0.08, 0.24, None, 0.32],
                "burst_host_time_s": [0.0, 0.08, 0.24, 0.28, 0.32],
                "accel_x_g": [0.0, 1.0, -1.0, 0.5, 0.0],
                "accel_y_g": [0.0, 0.0, 0.0, 0.0, 0.0],
                "accel_z_g": [0.0, 1.0, 0.0, -1.0, 0.0],
                "gyro_x_dps": [0.0, 0.0, 0.0, 0.0, 0.0],
                "gyro_y_dps": [0.0, 10.0, 20.0, 30.0, 40.0],
                "gyro_z_dps": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )

        context_df, _ = build_context_dataset(
            derived_df=derived_df,
            wheel_df=pl.DataFrame(),
            imu_frame_df=imu_frame_df,
            context_config=DEFAULT_CONTEXT_CONFIG,
        )

        longitudinal = context_df.get_column("longitudinal_accel_g_ctx").to_numpy()
        vertical = context_df.get_column("vertical_accel_g_ctx").to_numpy()
        pitch = context_df.get_column("pitch_rate_dps_ctx").to_numpy()
        roughness = context_df.get_column("vertical_roughness_g_ctx").to_numpy()

        self.assertAlmostEqual(float(longitudinal[0]), 0.0, places=6)
        self.assertAlmostEqual(float(longitudinal[4]), 1.0, places=6)
        self.assertAlmostEqual(float(vertical[14]), -1.0, places=6)
        self.assertAlmostEqual(float(pitch[14]), 30.0, places=6)
        self.assertGreater(float(np.nanmax(roughness)), 0.0)

    def test_context_summary_supports_balance_and_front_share_responses(self) -> None:
        export_dir = self._copy_export("LOG00016")
        bundle = open_export_session(export_dir)
        context_df, meta = build_context_dataset(
            derived_df=bundle.derived_df,
            wheel_df=bundle.wheel_df,
            imu_frame_df=bundle.imu_frame_df,
            context_config=DEFAULT_CONTEXT_CONFIG,
        )

        balance_rows = summarize_context_bins(
            context_df=context_df,
            meta=meta,
            source=LONGITUDINAL_ACCEL_SOURCE,
            response=BALANCE_RESPONSE,
            bin_count=12,
        )
        front_share_rows = summarize_context_bins(
            context_df=context_df,
            meta=meta,
            source=LONGITUDINAL_ACCEL_SOURCE,
            response=FRONT_SHARE_RESPONSE,
            bin_count=12,
        )
        heatmap = compute_context_heatmap(
            context_df=context_df,
            meta=meta,
            source=LONGITUDINAL_ACCEL_SOURCE,
            response=FRONT_SHARE_RESPONSE,
            source_bins=12,
            response_bins=24,
        )

        self.assertGreater(len(balance_rows), 0)
        self.assertGreater(len(front_share_rows), 0)
        self.assertTrue(heatmap["has_data"])
        self.assertEqual(meta["responses"][FRONT_SHARE_RESPONSE]["units"], "%")

    def test_breakdown_analysis_builds_percentile_speed_bands_for_strong_speed_session(self) -> None:
        export_dir = self._copy_export("LOG00016")
        bundle = open_export_session(export_dir)

        analysis = build_breakdown_analysis(
            derived_df=bundle.derived_df,
            wheel_df=bundle.wheel_df,
            imu_frame_df=bundle.imu_frame_df,
        )

        self.assertTrue(analysis["meta"]["wheel_speed"]["usable"])
        self.assertEqual(len(analysis["band_rows"]), 5)
        first = analysis["band_rows"][0]
        self.assertIn("percentile_label", first)
        self.assertIn("occupancy_s", first)
        self.assertIn("occupancy_pct", first)
        self.assertGreater(sum(float(row["occupancy_s"]) for row in analysis["band_rows"]), 0.0)
        self.assertGreater(len(analysis["state_rows"]), 0)
        self.assertGreater(len(analysis["event_rows"]), 0)
        self.assertGreater(len(analysis["finding_rows"]), 0)
        self.assertGreater(len(analysis["quality_rows"]), 0)
        state = analysis["state_rows"][0]
        self.assertIn("activity_state", state)
        self.assertIn("front_p95_stroke_pct", state)
        self.assertIn("front_time_above_85_pct", state)
        event = analysis["event_rows"][0]
        self.assertIn("timestamp_s", event)
        self.assertIn("front_peak_stroke_pct", event)

    def test_breakdown_analysis_disables_percentile_summary_for_weak_speed_session(self) -> None:
        export_dir = self._copy_export("LOG00010")
        bundle = open_export_session(export_dir)

        analysis = build_breakdown_analysis(
            derived_df=bundle.derived_df,
            wheel_df=bundle.wheel_df,
            imu_frame_df=bundle.imu_frame_df,
        )

        self.assertFalse(analysis["meta"]["wheel_speed"]["usable"])
        self.assertEqual(analysis["band_rows"], [])
        self.assertEqual(analysis["state_rows"], [])
        self.assertGreater(len(analysis["event_rows"]), 0)
        self.assertTrue(any("Wheel speed coverage" in row["title"] for row in analysis["finding_rows"]))

    def test_breakdown_analysis_classifies_motion_activity_and_events(self) -> None:
        time_s = np.arange(0.0, 9.0, 0.02)
        speed_kph = np.piecewise(
            time_s,
            [time_s < 3.0, (time_s >= 3.0) & (time_s < 6.0), time_s >= 6.0],
            [
                lambda values: 10.0 + values * 8.0,
                lambda values: 34.0 - (values - 3.0) * 8.0,
                12.0,
            ],
        )
        braking = (time_s >= 3.0) & (time_s < 6.0)
        accelerating = time_s < 3.0
        front_travel = 25.0 + 8.0 * np.sin(time_s * 8.0)
        front_travel[braking] += 62.0
        rear_travel = 25.0 + 7.0 * np.sin(time_s * 7.0)
        rear_travel[accelerating] += 48.0
        impact_index = int(np.argmin(np.abs(time_s - 4.0)))
        front_travel[impact_index:impact_index + 4] = [85.0, 96.0, 100.0, 88.0]
        rear_travel[impact_index:impact_index + 4] = [40.0, 78.0, 92.0, 50.0]
        front_velocity = np.gradient(front_travel, time_s)
        rear_velocity = np.gradient(rear_travel, time_s)
        derived_df = pl.DataFrame(
            {
                "host_timestamp_us": np.round(time_s * 1_000_000.0).astype(np.int64),
                "dt_s": np.full(time_s.shape, 0.02),
                "front_raw": np.full(time_s.shape, 1200.0),
                "rear_raw": np.full(time_s.shape, 1300.0),
                "front_travel_counts": front_travel,
                "front_filtered_travel_counts": front_travel,
                "front_velocity_counts_per_s_filtered": front_velocity,
                "rear_travel_counts": rear_travel,
                "rear_filtered_travel_counts": rear_travel,
                "rear_velocity_counts_per_s_filtered": rear_velocity,
                "front_travel_stroke_pct": front_travel,
                "front_filtered_travel_stroke_pct": front_travel,
                "front_velocity_stroke_pct_per_s_filtered": front_velocity,
                "rear_travel_stroke_pct": rear_travel,
                "rear_filtered_travel_stroke_pct": rear_travel,
                "rear_velocity_stroke_pct_per_s_filtered": rear_velocity,
            }
        )
        wheel_df = pl.DataFrame(
            {
                "host_time_s": time_s,
                "period_s": np.full(time_s.shape, 0.02),
                "speed_kph": speed_kph,
            }
        )

        analysis = build_breakdown_analysis(
            derived_df=derived_df,
            wheel_df=wheel_df,
            imu_frame_df=pl.DataFrame(),
            breakdown_config={
                "motion_smoothing_ms": 100,
                "braking_threshold_mps2": 0.5,
                "accel_threshold_mps2": 0.5,
                "min_segment_ms": 120,
                "active_activity_threshold": 0.4,
                "high_compression_velocity_min_pct_s": 250.0,
            },
        )

        motion_states = {row["motion_state"] for row in analysis["state_rows"]}
        activity_states = {row["activity_state"] for row in analysis["state_rows"]}
        event_types = {row["event_type"] for row in analysis["event_rows"]}
        finding_titles = {row["title"] for row in analysis["finding_rows"]}

        self.assertIn("braking", motion_states)
        self.assertIn("accelerating", motion_states)
        self.assertIn("high_speed_impact_candidate", activity_states)
        self.assertIn("bottom_out", event_types)
        self.assertIn("high_compression_velocity", event_types)
        self.assertIn("Braking fork usage high", finding_titles)

    def test_export_session_writes_expected_files(self) -> None:
        output_dir = self._workspace_run_dir("synthetic_export") / "synthetic_export"
        session = {
            "path": Path("synthetic.BIN"),
            "file_size_bytes": 128,
            "header": {
                "analog_rate_hz": 1000,
                "analog_resolution_bits": 12,
            },
            "config": {
                "analog_rate_hz": 1000,
                "analog_resolution_bits": 12,
                "analog_averaging": 1,
                "button_debounce_ms": 30,
                "button_hold_start_ms": 500,
                "button_hold_stop_ms": 500,
                "reed_debounce_us": 1000,
                "wheel_circumference_m": 2.1,
                "imu_accel_range_g": 16,
                "imu_gyro_range_dps": 2000,
                "imu_odr_hz": 833,
                "imu_watermark_frames": 32,
                "imu_fifo_mode": 6,
                "imu_timestamp_decimation": 1,
            },
            "record_counts": {"CFG ": 1, "STAT": 1, "ANLG": 2},
            "records": [
                {
                    "header": {"tag": "CFG ", "seq": 1, "flags": 0, "timestamp_us": 0, "payload_len": 0},
                    "payload": {
                        "analog_rate_hz": 1000,
                        "analog_resolution_bits": 12,
                        "analog_averaging": 1,
                        "button_debounce_ms": 30,
                        "button_hold_start_ms": 500,
                        "button_hold_stop_ms": 500,
                        "reed_debounce_us": 1000,
                        "wheel_circumference_m": 2.1,
                        "imu_accel_range_g": 16,
                        "imu_gyro_range_dps": 2000,
                        "imu_odr_hz": 833,
                        "imu_watermark_frames": 32,
                        "imu_fifo_mode": 6,
                        "imu_timestamp_decimation": 1,
                    },
                },
                {
                    "header": {"tag": "STAT", "seq": 2, "flags": 0, "timestamp_us": 500, "payload_len": 0},
                    "payload": {"code": 1, "name": "BOOT", "flags": 0, "value0": 0, "value1": 0},
                },
                {
                    "header": {"tag": "ANLG", "seq": 3, "flags": 0, "timestamp_us": 1000, "payload_len": 0},
                    "payload": {"sample_index": 0, "front_raw": 1000, "rear_raw": 1100},
                },
                {
                    "header": {"tag": "ANLG", "seq": 4, "flags": 0, "timestamp_us": 2000, "payload_len": 0},
                    "payload": {"sample_index": 1, "front_raw": 1010, "rear_raw": 1090},
                },
            ],
        }

        summary = export_session(
            session=session,
            output_dir=output_dir,
            front_mm_per_count=None,
            rear_mm_per_count=None,
            front_zero_count=0,
            rear_zero_count=0,
            front_sign=1,
            rear_sign=1,
        )

        self.assertTrue((output_dir / "analog.csv").exists())
        self.assertTrue((output_dir / "status.csv").exists())
        self.assertTrue((output_dir / "summary.json").exists())
        self.assertEqual(summary["counts"]["analog_rows"], 2)
        self.assertEqual(summary["counts"]["stat_rows"], 1)

    def test_export_session_unwraps_host_timestamp_rollover(self) -> None:
        output_dir = self._workspace_run_dir("wrap_export") / "wrap_export"
        wrap = 1 << 32
        session = {
            "path": Path("wrap.BIN"),
            "file_size_bytes": 128,
            "header": {
                "analog_rate_hz": 500,
                "analog_resolution_bits": 12,
            },
            "config": {
                "analog_rate_hz": 500,
                "analog_resolution_bits": 12,
                "analog_averaging": 8,
                "button_debounce_ms": 20,
                "button_hold_start_ms": 0,
                "button_hold_stop_ms": 0,
                "reed_debounce_us": 3000,
                "wheel_circumference_m": 2.213,
                "imu_accel_range_g": 0,
                "imu_gyro_range_dps": 0,
                "imu_odr_hz": 0,
                "imu_watermark_frames": 0,
                "imu_fifo_mode": 0,
                "imu_timestamp_decimation": 0,
            },
            "record_counts": {"CFG ": 1, "ANLG": 3, "STAT": 1},
            "records": [
                {
                    "header": {"tag": "CFG ", "seq": 1, "flags": 0, "timestamp_us": wrap - 2000, "payload_len": 0},
                    "payload": {
                        "analog_rate_hz": 500,
                        "analog_resolution_bits": 12,
                        "analog_averaging": 8,
                        "button_debounce_ms": 20,
                        "button_hold_start_ms": 0,
                        "button_hold_stop_ms": 0,
                        "reed_debounce_us": 3000,
                        "wheel_circumference_m": 2.213,
                        "imu_accel_range_g": 0,
                        "imu_gyro_range_dps": 0,
                        "imu_odr_hz": 0,
                        "imu_watermark_frames": 0,
                        "imu_fifo_mode": 0,
                        "imu_timestamp_decimation": 0,
                    },
                },
                {
                    "header": {"tag": "ANLG", "seq": 2, "flags": 0, "timestamp_us": wrap - 1000, "payload_len": 0},
                    "payload": {"sample_index": 0, "front_raw": 1000, "rear_raw": 1100},
                },
                {
                    "header": {"tag": "ANLG", "seq": 3, "flags": 0, "timestamp_us": 1000, "payload_len": 0},
                    "payload": {"sample_index": 1, "front_raw": 1010, "rear_raw": 1110},
                },
                {
                    "header": {"tag": "ANLG", "seq": 4, "flags": 0, "timestamp_us": 3000, "payload_len": 0},
                    "payload": {"sample_index": 2, "front_raw": 1020, "rear_raw": 1120},
                },
                {
                    "header": {"tag": "STAT", "seq": 5, "flags": 0, "timestamp_us": 4000, "payload_len": 0},
                    "payload": {"code": 10, "name": "CLEAN_CLOSE", "flags": 0, "value0": 1, "value1": 0},
                },
            ],
        }

        summary = export_session(
            session=session,
            output_dir=output_dir,
            front_mm_per_count=None,
            rear_mm_per_count=None,
            front_zero_count=0,
            rear_zero_count=0,
            front_sign=1,
            rear_sign=1,
        )
        with (output_dir / "analog.csv").open(newline="", encoding="utf-8") as handle:
            analog_rows = list(csv.DictReader(handle))

        self.assertEqual(summary["timing"]["first_host_timestamp_us"], wrap - 2000)
        self.assertEqual(summary["timing"]["last_host_timestamp_us"], wrap + 4000)
        self.assertEqual(summary["timing"]["duration_us"], 6000)
        self.assertEqual([int(row["host_timestamp_us"]) for row in analog_rows], [wrap - 1000, wrap + 1000, wrap + 3000])
        self.assertEqual([row["dt_us"] for row in analog_rows], ["", "2000", "2000"])

    def test_decode_stat_labels_session_stop_reason(self) -> None:
        payload = STAT_STRUCT.pack(9, 0, 16, 1)
        decoded = decode_stat(payload)

        self.assertEqual(decoded["name"], "SESSION_STOP")
        self.assertEqual(decoded["stop_reason_name"], "SWITCH_OFF")

    def test_open_export_session_handles_zero_imu_metadata_with_empty_imu_exports(self) -> None:
        output_dir = self._workspace_run_dir("imu_free_export") / "imu_free_export"
        session = {
            "path": Path("imu_free.BIN"),
            "file_size_bytes": 128,
            "header": {
                "magic": "STLOG1",
                "format_version": 1,
                "header_size": 100,
                "firmware_version": 65536,
                "build_epoch": 1775406721,
                "build_id": "test",
                "start_epoch": 1776000000,
                "start_micros": 1000,
                "session_index": 42,
                "rtc_valid": 1,
                "enabled_mask": 0x17,
                "analog_rate_hz": 500,
                "analog_resolution_bits": 12,
                "analog_averaging": 8,
                "imu_rate_hz": 0,
                "imu_watermark_frames": 0,
                "record_header_size": 16,
                "button_pin": 17,
                "led_pin": 14,
                "reed_pin": 18,
                "imu_int1_pin": 2,
                "front_pot_pin": 20,
                "rear_pot_pin": 19,
                "imu_cs_pin": 10,
            },
            "config": {
                "analog_rate_hz": 500,
                "analog_resolution_bits": 12,
                "analog_averaging": 8,
                "button_debounce_ms": 20,
                "button_hold_start_ms": 3000,
                "button_hold_stop_ms": 3000,
                "reed_debounce_us": 3000,
                "wheel_circumference_m": 2.213,
                "imu_accel_range_g": 0,
                "imu_gyro_range_dps": 0,
                "imu_odr_hz": 0,
                "imu_watermark_frames": 0,
                "imu_fifo_mode": 0,
                "imu_timestamp_decimation": 0,
            },
            "record_counts": {"CFG ": 1, "STAT": 2, "ANLG": 2, "WSPD": 1},
            "records": [
                {
                    "header": {"tag": "CFG ", "seq": 0, "flags": 0, "timestamp_us": 1000, "payload_len": 0},
                    "payload": {
                        "analog_rate_hz": 500,
                        "analog_resolution_bits": 12,
                        "analog_averaging": 8,
                        "button_debounce_ms": 20,
                        "button_hold_start_ms": 3000,
                        "button_hold_stop_ms": 3000,
                        "reed_debounce_us": 3000,
                        "wheel_circumference_m": 2.213,
                        "imu_accel_range_g": 0,
                        "imu_gyro_range_dps": 0,
                        "imu_odr_hz": 0,
                        "imu_watermark_frames": 0,
                        "imu_fifo_mode": 0,
                        "imu_timestamp_decimation": 0,
                    },
                },
                {
                    "header": {"tag": "STAT", "seq": 1, "flags": 0, "timestamp_us": 1200, "payload_len": 0},
                    "payload": {"code": 1, "name": "BOOT", "flags": 0, "value0": 5, "value1": 42},
                },
                {
                    "header": {"tag": "STAT", "seq": 2, "flags": 0, "timestamp_us": 2000, "payload_len": 0},
                    "payload": {
                        "code": 9,
                        "name": "SESSION_STOP",
                        "flags": 0,
                        "value0": 42,
                        "value1": 1,
                        "stop_reason_name": "SWITCH_OFF",
                    },
                },
                {
                    "header": {"tag": "ANLG", "seq": 3, "flags": 0, "timestamp_us": 3000, "payload_len": 0},
                    "payload": {"sample_index": 0, "front_raw": 1000, "rear_raw": 1100},
                },
                {
                    "header": {"tag": "ANLG", "seq": 4, "flags": 0, "timestamp_us": 5000, "payload_len": 0},
                    "payload": {"sample_index": 1, "front_raw": 1010, "rear_raw": 1090},
                },
                {
                    "header": {"tag": "WSPD", "seq": 5, "flags": 0, "timestamp_us": 5500, "payload_len": 0},
                    "payload": {"pulse_count": 1, "interval_us": 100000},
                },
            ],
        }

        export_session(
            session=session,
            output_dir=output_dir,
            front_mm_per_count=None,
            rear_mm_per_count=None,
            front_zero_count=0,
            rear_zero_count=0,
            front_sign=1,
            rear_sign=1,
        )

        bundle = open_export_session(output_dir)

        self.assertEqual(bundle.summary["header"]["analog_rate_hz"], 500)
        self.assertEqual(bundle.summary["header"]["imu_rate_hz"], 0)
        self.assertEqual(bundle.imu_frame_df.height, 0)
        self.assertGreater(bundle.analog_df.height, 0)

    def test_export_bin_to_directory_rejects_malformed_bin(self) -> None:
        temp_dir = self._workspace_run_dir("malformed_bin")
        bin_path = temp_dir / "bad.BIN"
        bin_path.write_bytes(b"not a real log")
        with self.assertRaises(ValueError):
            export_bin_to_directory(bin_path, temp_dir / "out")

    def test_bin_export_is_current_accepts_matching_legacy_summary(self) -> None:
        temp_dir = self._workspace_run_dir("bin_export_current")
        bin_path = temp_dir / "demo.BIN"
        bin_path.write_bytes(b"demo-session")
        export_dir = temp_dir / "exports" / "demo"
        export_dir.mkdir(parents=True, exist_ok=True)

        summary_path = export_dir / "summary.json"
        summary_path.write_text(
            '{\n'
            '  "source_path": "demo.BIN",\n'
            f'  "file_size_bytes": {bin_path.stat().st_size}\n'
            '}\n',
            encoding="utf-8",
        )
        for filename in ("analog.csv", "config.csv", "status.csv", "timeline.csv"):
            (export_dir / filename).write_text("", encoding="utf-8")

        future_time = int(bin_path.stat().st_mtime_ns + 5_000_000_000)
        os.utime(summary_path, ns=(future_time, future_time))

        self.assertTrue(bin_export_is_current(bin_path, export_dir))

    def test_open_bin_session_skips_reexport_when_export_is_current(self) -> None:
        sentinel = object()
        with mock.patch.object(session_service, "bin_export_is_current", return_value=True), mock.patch.object(
            session_service,
            "_prepare_bundle",
            return_value=sentinel,
        ) as prepare_mock, mock.patch.object(session_service, "export_bin_to_directory") as export_mock:
            result = session_service.open_bin_session(Path("LOG00010.BIN"))

        self.assertIs(result, sentinel)
        export_mock.assert_not_called()
        prepare_mock.assert_called_once_with(Path("exports") / "LOG00010", source_path=Path("LOG00010.BIN"))

    def test_open_bin_session_reexports_when_export_is_stale(self) -> None:
        sentinel = object()
        with mock.patch.object(session_service, "bin_export_is_current", return_value=False), mock.patch.object(
            session_service,
            "_prepare_bundle",
            return_value=sentinel,
        ) as prepare_mock, mock.patch.object(session_service, "export_bin_to_directory") as export_mock:
            result = session_service.open_bin_session(Path("LOG00010.BIN"))

        self.assertIs(result, sentinel)
        export_mock.assert_called_once_with(
            bin_path=Path("LOG00010.BIN"),
            output_dir=Path("exports") / "LOG00010",
        )
        prepare_mock.assert_called_once_with(Path("exports") / "LOG00010", source_path=Path("LOG00010.BIN"))

    def test_run_quicklook_generates_artifacts_for_export_dir(self) -> None:
        export_dir = self._copy_export("LOG00016")
        result = run_quicklook(export_dir)

        self.assertTrue(result.summary_json_path.exists())
        self.assertTrue(result.summary_text_path.exists())
        self.assertTrue(result.artifact_paths["travel_overview_png"].exists())
        self.assertTrue(result.artifact_paths["velocity_histograms_png"].exists())
        self.assertIn("Trackside Quicklook", result.summary_text)

        summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["session"]["analog_rows"], int(result.bundle.analog_df.height))
        self.assertIn("front", summary["channels"])
        self.assertIn("rear", summary["channels"])
        self.assertTrue(summary["status"]["clean_close_recorded"])
        self.assertGreater(summary["status"]["highlight_count"], 0)

    def test_run_quicklook_handles_missing_optional_export_files(self) -> None:
        export_dir = self._copy_export("LOG00010")
        (export_dir / "wheel.csv").unlink()
        (export_dir / "imu_frames.csv").unlink()
        (export_dir / "imu_bursts.csv").unlink()

        result = run_quicklook(export_dir)
        summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["session"]["wheel_rows"], 0)
        self.assertEqual(summary["session"]["imu_frame_rows"], 0)
        self.assertTrue(result.artifact_paths["travel_overview_png"].exists())
        self.assertTrue(result.artifact_paths["velocity_histograms_png"].exists())

    def test_quicklook_cli_accepts_bin_input_and_writes_default_output(self) -> None:
        temp_dir = self._workspace_run_dir("quicklook_bin")
        bin_path = (temp_dir / "LOG00010.BIN").resolve()
        shutil.copy2(self._bin_fixture_path("LOG00010"), bin_path)

        repo_root = Path(__file__).resolve().parents[1]
        command = [
            sys.executable,
            str(repo_root / "scripts" / "postprocess_quicklook.py"),
            "--input",
            str(bin_path),
        ]
        completed = subprocess.run(
            command,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        output_dir = temp_dir / "exports" / "LOG00010" / "analysis" / "quicklook"
        self.assertTrue(output_dir.exists())
        self.assertTrue((output_dir / "quicklook_summary.json").exists())
        self.assertTrue((output_dir / "quicklook_summary.txt").exists())
        self.assertTrue((output_dir / "travel_overview.png").exists())
        self.assertTrue((output_dir / "velocity_histograms.png").exists())
        self.assertIn("Quicklook outputs written to", completed.stdout)


if __name__ == "__main__":
    unittest.main()
