from __future__ import annotations

import copy
import os
import re
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from scripts.postprocess_gui_app.app import (
    MainWindow,
    WheelPlotWidget,
    apply_calibration_template,
    build_summary_text,
    downsample_xy,
    downsample_xy_preserve_extrema,
    open_path_in_file_manager,
    parse_optional_float,
)
from scripts.postprocess_gui_app.backend.export_service import export_bin_to_directory
from scripts.postprocess_gui_app.backend.session_config import default_session_config
from scripts.postprocess_gui_app.backend.session_metadata_service import SessionDatabaseEntry
from scripts.postprocess_gui_app.backend.session_service import (
    SessionBundle,
    open_export_session,
    rebuild_session_analysis,
    sanitize_wheel_speed_frame,
)


class PostprocessGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

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

    def _pump_events(self, duration_s: float = 0.2) -> None:
        deadline = time.time() + duration_s
        while time.time() < deadline:
            self.app.processEvents()
            time.sleep(0.01)

    def _synthetic_compare_bundle(
        self,
        session_id: str,
        *,
        track: str,
        front_offset: float = 0.0,
        rear_offset: float = 0.0,
    ) -> SessionBundle:
        derived_df = pl.DataFrame(
            {
                "host_timestamp_us": [0, 1_000_000, 2_000_000, 3_000_000],
                "dt_s": [1.0, 1.0, 1.0, 1.0],
                "front_raw": [1200.0, 1400.0, 1600.0, 1500.0],
                "rear_raw": [1300.0, 1450.0, 1550.0, 1650.0],
                "front_travel_stroke_pct": [10.0 + front_offset, 30.0 + front_offset, 55.0 + front_offset, 25.0 + front_offset],
                "front_filtered_travel_stroke_pct": [10.0 + front_offset, 30.0 + front_offset, 55.0 + front_offset, 25.0 + front_offset],
                "front_velocity_stroke_pct_per_s_filtered": [0.0, 22.0 + front_offset, -18.0 - front_offset, 8.0 + front_offset],
                "rear_travel_stroke_pct": [12.0 + rear_offset, 28.0 + rear_offset, 44.0 + rear_offset, 18.0 + rear_offset],
                "rear_filtered_travel_stroke_pct": [12.0 + rear_offset, 28.0 + rear_offset, 44.0 + rear_offset, 18.0 + rear_offset],
                "rear_velocity_stroke_pct_per_s_filtered": [0.0, 15.0 + rear_offset, -12.0 - rear_offset, 6.0 + rear_offset],
            }
        )
        source_path = Path("data") / f"{session_id}.BIN"
        export_dir = Path(".codex_tmp") / "test_tmp" / "gui_synthetic" / session_id
        return SessionBundle(
            source_path=source_path,
            export_dir=export_dir,
            summary={
                "header": {"analog_resolution_bits": 12},
                "timing": {"duration_us": 3_000_000},
                "counts": {},
                "record_counts": {},
            },
            status_rows=[],
            config_rows=[],
            session_config=default_session_config(source_path, export_dir, 0, 0),
            derived_df=derived_df,
            analog_df=pl.DataFrame(),
            wheel_df=pl.DataFrame(),
            imu_frame_df=pl.DataFrame(),
            session_metadata={
                "session_id": session_id,
                "date": "2026-04-18",
                "track": track,
                "set_label": session_id,
                "comment": "",
            },
        )

    def test_main_window_initializes(self) -> None:
        window = MainWindow()
        self.assertIn("Workbench", window.windowTitle())

    def test_main_window_includes_compare_tab(self) -> None:
        window = MainWindow()
        tab_names = [window.tabs.tabText(index) for index in range(window.tabs.count())]

        self.assertEqual(
            tab_names,
            ["Session", "Histograms", "Position/Velocity Heatmap", "Balance", "Braking", "Breakdown", "Speed", "Metrics", "Compare"],
        )

    def test_main_window_populates_database_panel_from_scan(self) -> None:
        entry = SessionDatabaseEntry(
            session_id="LOG00047",
            export_dir=Path("exports/LOG00047"),
            source_path=Path("data/LOG00047.BIN"),
            metadata={
                "session_id": "LOG00047",
                "date": "2026-04-18",
                "track": "Test Track",
                "set_label": "Set 2",
                "comment": "dry",
            },
            summary={"header": {"start_epoch": 1_776_470_400}},
            sort_timestamp=1_776_470_400,
        )
        with mock.patch("scripts.postprocess_gui_app.app.scan_session_database", return_value=[entry]):
            window = MainWindow()

        self.assertEqual(window.database_table.rowCount(), 2)
        self.assertEqual(window.database_table.columnCount(), 3)
        self.assertEqual(window.database_table.horizontalHeaderItem(0).text(), "BIN / Group")
        self.assertIn("2026-04-18 | Test Track", window.database_table.item(0, 0).text())
        self.assertEqual(window.database_table.item(0, 1).text(), "1 set(s)")
        self.assertEqual(window.database_table.item(0, 2).text(), "")
        self.assertEqual(window.database_table.item(1, 0).text(), "LOG00047")
        self.assertEqual(window.database_table.item(1, 1).text(), "Set 2")
        self.assertEqual(window.database_table.item(1, 2).text(), "dry")
        self.assertEqual(window.current_session_label.text(), "Currently open: -")

    def test_open_bin_dialog_accepts_multiple_bin_files(self) -> None:
        window = MainWindow()
        with mock.patch(
            "scripts.postprocess_gui_app.app.QtWidgets.QFileDialog.getOpenFileNames",
            return_value=(["data/LOG00047.BIN", "data/LOG00020.BIN"], "BIN files (*.BIN *.bin)"),
        ), mock.patch.object(window, "run_task") as run_task:
            window.open_bin_dialog()

        run_task.assert_called_once()
        message, _, callback = run_task.call_args.args
        self.assertEqual(message, "Opening 2 BIN sessions")
        self.assertEqual(callback.__func__, window._on_bin_import_loaded.__func__)

    def test_bin_import_callback_loads_latest_session_and_reports_count(self) -> None:
        older = self._synthetic_compare_bundle("LOG00047", track="Test Track")
        older.summary["header"]["start_epoch"] = 1_776_470_400
        older.session_config = {"front": {}, "rear": {}, "plot_defaults": {}}
        older.session_metadata["set_label"] = "Set 1"
        newer = self._synthetic_compare_bundle("LOG00020", track="Test Track")
        newer.summary["header"]["start_epoch"] = 1_776_470_500
        newer.session_config = {"front": {}, "rear": {}, "plot_defaults": {}}
        newer.session_metadata["set_label"] = "Set 2"
        for bundle in (older, newer):
            bundle.derived_df = bundle.derived_df.with_columns(
                [
                    pl.col("front_travel_stroke_pct").alias("front_travel_counts"),
                    pl.col("front_filtered_travel_stroke_pct").alias("front_filtered_travel_counts"),
                    pl.col("front_velocity_stroke_pct_per_s_filtered").alias("front_velocity_counts_per_s_filtered"),
                    pl.col("rear_travel_stroke_pct").alias("rear_travel_counts"),
                    pl.col("rear_filtered_travel_stroke_pct").alias("rear_filtered_travel_counts"),
                    pl.col("rear_velocity_stroke_pct_per_s_filtered").alias("rear_velocity_counts_per_s_filtered"),
                ]
            )
        window = MainWindow()
        with mock.patch("scripts.postprocess_gui_app.app.scan_session_database", return_value=[]):
            window._on_bin_import_loaded([older, newer])

        self.assertIs(window.current_bundle, newer)
        self.assertEqual(window.metadata_set_edit.text(), "Set 2")
        self.assertEqual(window.status_label.text(), "Imported 2 BIN sessions")

    def test_database_panel_groups_sets_by_date(self) -> None:
        entries = [
            SessionDatabaseEntry(
                session_id="LOG00053",
                export_dir=Path("exports/LOG00053"),
                source_path=Path("data/LOG00053.BIN"),
                metadata={
                    "session_id": "LOG00053",
                    "date": "2026-04-24",
                    "track": "",
                    "set_label": "LOG00053",
                    "comment": "",
                },
                summary={"header": {"start_epoch": 1_777_057_492}},
                sort_timestamp=1_777_057_492,
            ),
            SessionDatabaseEntry(
                session_id="LOG00047",
                export_dir=Path("exports/LOG00047"),
                source_path=Path("data/LOG00047.BIN"),
                metadata={
                    "session_id": "LOG00047",
                    "date": "2026-04-24",
                    "track": "",
                    "set_label": "LOG00047",
                    "comment": "",
                },
                summary={"header": {"start_epoch": 1_777_054_858}},
                sort_timestamp=1_777_054_858,
            ),
        ]
        with mock.patch("scripts.postprocess_gui_app.app.scan_session_database", return_value=entries):
            window = MainWindow()

        self.assertEqual(window.database_label.text(), "Data base (2 processed sets, 1 day, 1 group)")
        self.assertIn("2026-04-24 | No track (2 sets)", window.database_table.item(0, 0).text())
        self.assertEqual(window.database_table.item(0, 1).text(), "2 set(s)")
        self.assertEqual(window.database_table.item(0, 2).text(), "")
        self.assertEqual(window.database_table.item(1, 0).text(), "LOG00053")
        self.assertEqual(window.database_table.item(1, 1).text(), "LOG00053")
        self.assertEqual(window.database_table.item(2, 0).text(), "LOG00047")

        window.on_database_item_clicked(window.database_table.item(0, 0))

        self.assertTrue(window.database_table.isRowHidden(1))
        self.assertTrue(window.database_table.isRowHidden(2))
        self.assertTrue(window.database_table.item(0, 0).text().startswith(">"))

    def test_delete_selected_bin_files_removes_only_confirmed_source_bin(self) -> None:
        temp_dir = self._workspace_run_dir("delete_bin")
        bin_path = temp_dir / "LOG00999.BIN"
        bin_path.write_bytes(b"demo")
        entry = SessionDatabaseEntry(
            session_id="LOG00999",
            export_dir=temp_dir / "exports" / "LOG00999",
            source_path=bin_path,
            metadata={
                "session_id": "LOG00999",
                "date": "2026-04-24",
                "track": "",
                "set_label": "LOG00999",
                "comment": "",
            },
            summary={"header": {"start_epoch": 1_777_000_000}},
            sort_timestamp=1_777_000_000,
        )
        with mock.patch("scripts.postprocess_gui_app.app.scan_session_database", return_value=[entry]):
            window = MainWindow()

        window.database_table.blockSignals(True)
        window.database_table.selectRow(1)
        window.database_table.blockSignals(False)
        window._update_database_actions()

        self.assertTrue(window.delete_bin_button.isEnabled())
        with mock.patch(
            "scripts.postprocess_gui_app.app.QtWidgets.QMessageBox.question",
            return_value=QtWidgets.QMessageBox.StandardButton.Yes,
        ):
            window.delete_selected_bin_files()

        self.assertFalse(bin_path.exists())
        self.assertEqual(window.status_label.text(), "Deleted 1 .BIN source file(s)")

    def test_summary_text_includes_saved_metadata(self) -> None:
        bundle = SessionBundle(
            source_path=Path("data/LOG00047.BIN"),
            export_dir=Path("exports/LOG00047"),
            summary={"timing": {"duration_us": 1_000_000}, "counts": {}, "record_counts": {}},
            status_rows=[],
            config_rows=[],
            session_config={},
            derived_df=pl.DataFrame(),
            analog_df=pl.DataFrame(),
            wheel_df=pl.DataFrame(),
            imu_frame_df=pl.DataFrame(),
            session_metadata={
                "session_id": "LOG00047",
                "date": "2026-04-18",
                "track": "Test Track",
                "set_label": "Set 2",
                "comment": "dry",
            },
        )

        summary_text = build_summary_text(bundle)

        self.assertIn("Track: Test Track", summary_text)
        self.assertIn("Set: Set 2", summary_text)
        self.assertIn("Duration: 0:01", summary_text)

    def test_loading_export_populates_tabs(self) -> None:
        export_dir = self._copy_export("LOG00053")
        bundle = open_export_session(export_dir)
        window = MainWindow()
        window._on_bundle_loaded(bundle)

        self.assertIsNotNone(window.current_bundle)
        self.assertGreater(window.metrics_table.rowCount(), 0)
        self.assertGreater(window.status_table.rowCount(), 0)
        self.assertIsNotNone(window.current_occupancy)
        self.assertEqual(window.travel_bins_spin.value(), 400)
        self.assertEqual(window.velocity_bins_spin.value(), 400)
        self.assertEqual(window.velocity_axis_combo.currentData(), "linear")
        self.assertEqual(window.current_occupancy["front"]["travel_units"], "%")
        self.assertEqual(window.current_occupancy["front"]["velocity_units"], "%/s")
        self.assertEqual(window.current_occupancy["front"]["velocity_axis_mode"], "linear")
        tab_names = [window.tabs.tabText(index) for index in range(window.tabs.count())]
        self.assertEqual(
            tab_names,
            ["Session", "Histograms", "Position/Velocity Heatmap", "Balance", "Braking", "Breakdown", "Speed", "Metrics", "Compare"],
        )
        self.assertIsNone(window.balance_widget.layout())
        self.assertGreater(window.braking_widget.key_table.rowCount(), 0)
        self.assertGreater(window.braking_widget.speed_bin_table.rowCount(), 0)
        self.assertIn("Wheel-speed deceleration analysis", window.braking_widget.method_label.text())
        self.assertIn("speed bins", window.braking_widget.method_label.text())
        self.assertNotIn("FORK BRAKING SUMMARY", window.braking_widget.summary_label.text())
        self.assertNotIn("Headline metrics", window.braking_widget.summary_label.text())
        self.assertNotIn("Speed-bin summary", window.braking_widget.summary_label.text())
        self.assertNotIn("Loaded median stroke", window.braking_widget.summary_label.text())
        self.assertIn("Selected:", window.braking_widget.summary_label.text())
        self.assertIn("Coverage:", window.braking_widget.summary_label.text())
        self.assertTrue(window.braking_widget.scroll_area.widgetResizable())
        for table in (
            window.braking_widget.key_table,
            window.braking_widget.speed_bin_table,
            window.braking_widget.event_table,
            window.braking_widget.flag_table,
        ):
            self.assertEqual(table.verticalScrollBarPolicy(), QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.assertEqual(table.minimumHeight(), table.maximumHeight())
            for row in range(table.rowCount()):
                for column in range(table.columnCount()):
                    item = table.item(row, column)
                    if item is not None:
                        self.assertIsNone(re.search(r"\d+\.\d{3,}", item.text()))
        self.assertIn("Max. Speed", window.wheel_widget.summary_label.text())
        self.assertIn("Mean Speed", window.wheel_widget.summary_label.text())
        self.assertNotIn("Signals", tab_names)
        self.assertNotIn("Context", tab_names)
        self.assertNotIn("IMU", tab_names)
        self.assertEqual(window.histograms_widget.front_pane.travel_plot.getPlotItem().titleLabel.text, "Front position distribution")
        self.assertEqual(window.histograms_widget.rear_pane.travel_plot.getPlotItem().titleLabel.text, "Rear position distribution")
        self.assertEqual(window.histograms_widget.front_pane.travel_plot.getPlotItem().getAxis("left").labelText, "Relative time")
        self.assertEqual(window.histograms_widget.front_pane.travel_plot.getPlotItem().getAxis("left").labelUnits, "%")
        self.assertEqual(window.histograms_widget.front_pane.velocity_plot.getPlotItem().getAxis("left").labelText, "Relative time")
        self.assertEqual(window.histograms_widget.front_pane.velocity_plot.getPlotItem().getAxis("left").labelUnits, "%")
        self.assertEqual(window.occupancy_widget.front_pane.plot_widget.getPlotItem().titleLabel.text, "Front position/velocity heatmap")
        self.assertEqual(window.occupancy_widget.rear_pane.plot_widget.getPlotItem().titleLabel.text, "Rear position/velocity heatmap")
        self.assertIsNone(window.breakdown_widget.layout())

    def test_compare_tab_populates_from_two_loaded_bundles(self) -> None:
        left_bundle = self._synthetic_compare_bundle("LOG00053", track="Track A")
        right_bundle = self._synthetic_compare_bundle("LOG00047", track="Track B", front_offset=6.0, rear_offset=4.0)

        window = MainWindow()
        window._on_compare_bundles_loaded([left_bundle, right_bundle])

        self.assertEqual(window.tabs.tabText(window.tabs.currentIndex()), "Compare")
        self.assertIn("LOG00053", window.compare_widget.summary_label.text())
        self.assertIn("LOG00047", window.compare_widget.summary_label.text())
        self.assertEqual(window.compare_widget.front_metrics_table.rowCount(), 47)
        self.assertEqual(window.compare_widget.rear_metrics_table.rowCount(), 47)
        self.assertEqual(window.compare_widget.front_metrics_table.horizontalHeaderItem(1).text(), "2026-04-18 | Track A | LOG00053")
        self.assertEqual(window.compare_widget.front_metrics_table.horizontalHeaderItem(2).text(), "2026-04-18 | Track B | LOG00047")
        self.assertEqual(window.compare_widget.front_metrics_table.horizontalHeaderItem(3).text(), "Units")
        self.assertEqual(window.compare_widget.rear_metrics_group.title(), "Rear riding metrics")
        self.assertNotIn(
            "Delta",
            [
                window.compare_widget.front_metrics_table.horizontalHeaderItem(index).text()
                for index in range(window.compare_widget.front_metrics_table.columnCount())
            ],
        )
        metric_names = [
            window.compare_widget.front_metrics_table.item(row, 0).text()
            for row in range(window.compare_widget.front_metrics_table.rowCount())
        ]
        self.assertIn("Mean position", metric_names)
        self.assertIn("Position P0-10", metric_names)
        self.assertIn("Position P90-100", metric_names)
        self.assertIn("Mean compression velocity", metric_names)
        self.assertIn("Compression velocity P90-100", metric_names)
        self.assertIn("Mean rebound velocity", metric_names)
        self.assertIn("Rebound velocity P90-100", metric_names)
        self.assertIn("Peak compression velocity", metric_names)
        self.assertNotIn("RMS velocity", metric_names)
        self.assertNotIn("Used stroke", metric_names)
        self.assertEqual(
            window.compare_widget.front_metrics_table.verticalScrollBarPolicy(),
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.assertEqual(
            window.compare_widget.rear_metrics_table.verticalScrollBarPolicy(),
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.assertEqual(window.compare_widget.front_travel_plot.getPlotItem().getAxis("left").labelText, "Relative time")
        self.assertEqual(window.compare_widget.front_travel_plot.getPlotItem().getAxis("left").labelUnits, "%")
        self.assertEqual(window.compare_widget.front_velocity_plot.getPlotItem().getAxis("left").labelText, "Relative time")
        self.assertEqual(window.compare_widget.front_velocity_plot.getPlotItem().getAxis("left").labelUnits, "%")
        self.assertIn("BarGraphItem", [type(item).__name__ for item in window.compare_widget.front_velocity_plot.getPlotItem().items])

    def test_compare_tab_accepts_ten_loaded_bundles(self) -> None:
        bundles = [
            self._synthetic_compare_bundle(f"SYN{index:02d}", track="Track A", front_offset=float(index), rear_offset=float(index))
            for index in range(10)
        ]

        window = MainWindow()
        window._on_compare_bundles_loaded(bundles)

        self.assertEqual(len(window.compare_bundles), 10)
        self.assertEqual(window.compare_widget.front_metrics_table.columnCount(), 12)
        self.assertEqual(window.compare_widget.front_metrics_table.horizontalHeaderItem(10).text(), "2026-04-18 | Track A | SYN09")
        self.assertEqual(window.compare_widget.front_metrics_table.horizontalHeaderItem(11).text(), "Units")
        self.assertIn("SYN00", window.status_label.text())
        self.assertIn("SYN09", window.status_label.text())

    def test_signed_log_velocity_axis_updates_plot_views(self) -> None:
        bundle = self._synthetic_compare_bundle("LOG00997", track="Track A")
        bundle.derived_df = bundle.derived_df.with_columns(
            [
                pl.col("front_travel_stroke_pct").alias("front_travel_counts"),
                pl.col("front_filtered_travel_stroke_pct").alias("front_filtered_travel_counts"),
                pl.col("front_velocity_stroke_pct_per_s_filtered").alias("front_velocity_counts_per_s_filtered"),
                pl.col("rear_travel_stroke_pct").alias("rear_travel_counts"),
                pl.col("rear_filtered_travel_stroke_pct").alias("rear_filtered_travel_counts"),
                pl.col("rear_velocity_stroke_pct_per_s_filtered").alias("rear_velocity_counts_per_s_filtered"),
            ]
        )
        channel_config = {
            "zero_count": 1000,
            "invert": False,
            "mm_per_count": None,
            "full_scale_mm": 300.0,
            "sensor_full_scale_mm": 635.0,
            "travel_reference": "manual",
            "manual_reference_count": None,
            "velocity_filter_window": 3,
        }
        bundle.session_config = {
            "front": copy.deepcopy(channel_config),
            "rear": copy.deepcopy(channel_config),
            "plot_defaults": {
                "selected_channel": "front",
                "occupancy_travel_bins": 80,
                "occupancy_velocity_bins": 80,
                "occupancy_color_scale": "sqrt",
                "velocity_axis_mode": "signed_log",
            },
            "breakdown": {},
        }

        window = MainWindow()
        window._on_bundle_loaded(bundle)

        self.assertEqual(window.velocity_axis_combo.currentData(), "signed_log")
        self.assertEqual(window.current_occupancy["front"]["velocity_axis_mode"], "signed_log")
        self.assertIn("Rebound | Compression", window.current_occupancy["front"]["velocity_label"])
        self.assertIn(
            "Rebound | Compression",
            window.histograms_widget.front_pane.velocity_plot.getPlotItem().getAxis("bottom").labelText,
        )

    def test_metrics_tab_separates_riding_and_hardware_tables(self) -> None:
        export_dir = self._copy_export("LOG00053")
        bundle = open_export_session(export_dir)
        window = MainWindow()
        window._on_bundle_loaded(bundle)

        self.assertEqual(window.metrics_table.columnCount(), 13)
        self.assertEqual(window.metrics_table.rowCount(), 17)
        self.assertEqual(window.metrics_table.horizontalHeaderItem(1).text(), "Front position")
        self.assertEqual(window.metrics_table.horizontalHeaderItem(3).text(), "Front compression")
        self.assertEqual(window.metrics_table.horizontalHeaderItem(5).text(), "Front rebound")
        self.assertEqual(window.metrics_table.horizontalHeaderItem(7).text(), "Rear position")
        self.assertEqual(window.metrics_table.horizontalHeaderItem(9).text(), "Rear compression")
        self.assertEqual(window.metrics_table.horizontalHeaderItem(11).text(), "Rear rebound")
        self.assertEqual(window.metrics_table.item(0, 0).text(), "Session duration")
        self.assertRegex(window.metrics_table.item(0, 1).text(), r"\d+\.\d+ \(\d+:\d{2}\)")
        metric_names = [window.metrics_table.item(row, 0).text() for row in range(window.metrics_table.rowCount())]
        self.assertNotIn("Used stroke", metric_names)
        self.assertIn("Mean", metric_names)
        self.assertIn("Median", metric_names)
        self.assertIn("Mode", metric_names)
        self.assertIn("Geometric SD", metric_names)
        self.assertIn("P0-10", metric_names)
        self.assertIn("P90-100", metric_names)
        self.assertNotIn("RMS", metric_names)
        self.assertEqual(metric_names.index("P0-10"), metric_names.index("Geometric SD") + 1)
        mean_row = metric_names.index("Mean")
        p90_row = metric_names.index("P90-100")
        self.assertTrue(window.metrics_table.item(mean_row, 1).text())
        self.assertEqual(window.metrics_table.item(mean_row, 2).text(), "%")
        self.assertTrue(window.metrics_table.item(mean_row, 3).text())
        self.assertEqual(window.metrics_table.item(mean_row, 4).text(), "%/s")
        self.assertTrue(window.metrics_table.item(mean_row, 5).text())
        self.assertEqual(window.metrics_table.item(mean_row, 6).text(), "%/s")
        self.assertTrue(window.metrics_table.item(p90_row, 3).text())
        self.assertEqual(window.metrics_table.item(p90_row, 4).text(), "%")
        self.assertTrue(window.metrics_table.item(p90_row, 5).text())
        self.assertEqual(window.metrics_table.item(p90_row, 6).text(), "%")
        self.assertEqual(window.hardware_metrics_table.rowCount(), 7)
        self.assertEqual(window.hardware_metrics_table.item(0, 0).text(), "Analog sample count")

    def test_breakdown_tab_stays_blank_for_loaded_session(self) -> None:
        bundle = self._synthetic_compare_bundle("LOG00998", track="Track A")
        window = MainWindow()
        window._on_bundle_loaded(bundle)

        self.assertIsNone(window.breakdown_widget.layout())

    def test_braking_tab_shows_no_data_state_without_wheel_speed(self) -> None:
        bundle = self._synthetic_compare_bundle("LOG00998", track="Track A")
        bundle.wheel_df = pl.DataFrame()
        window = MainWindow()
        window._on_bundle_loaded(bundle)

        self.assertEqual(window.braking_widget.key_table.rowCount(), 0)
        self.assertEqual(window.braking_widget.speed_bin_table.rowCount(), 0)
        self.assertIn("Wheel speed", window.braking_widget.summary_label.text())
        self.assertGreater(window.braking_widget.flag_table.rowCount(), 0)

    def test_histogram_velocity_plot_omits_all_series(self) -> None:
        export_dir = self._copy_export("LOG00053")
        bundle = open_export_session(export_dir)
        window = MainWindow()
        window._on_bundle_loaded(bundle)

        bar_items = [
            item
            for item in window.histograms_widget.front_pane.velocity_plot.getPlotItem().items
            if type(item).__name__ == "BarGraphItem"
        ]
        self.assertEqual(len(bar_items), 2)

    def test_breakdown_tab_stays_blank_for_weak_speed_session(self) -> None:
        bundle = self._synthetic_compare_bundle("LOG00999", track="Track A")
        bundle.wheel_df = pl.DataFrame()
        window = MainWindow()
        window._on_bundle_loaded(bundle)

        self.assertIsNone(window.breakdown_widget.layout())

    def test_wheel_plot_starts_at_zero_before_first_pulse(self) -> None:
        widget = WheelPlotWidget()
        wheel_df = pl.DataFrame(
            {
                "host_time_s": [1.0, 2.0],
                "speed_kph": [10.0, 20.0],
                "period_s": [1.0, 1.0],
                "pulse_count": [1, 2],
            }
        )

        widget.set_wheel_data(wheel_df, start_time_s=0.25)

        speed_time, speed_values = widget.speed_curve.getData()
        self.assertEqual(float(speed_time[0]), 0.25)
        self.assertEqual(float(speed_values[0]), 0.0)
        self.assertEqual(float(speed_time[1]), 1.0)
        self.assertEqual(float(speed_values[1]), 0.0)

    def test_wheel_plot_uses_sanitized_speed_and_reports_summary_stats(self) -> None:
        widget = WheelPlotWidget()
        wheel_df = sanitize_wheel_speed_frame(
            pl.DataFrame(
                {
                    "host_time_s": [1.0, 2.0, 3.0],
                    "speed_kph": [20.0, 1200.0, 25.0],
                    "period_s": [0.40, 0.006, 0.32],
                    "pulse_count": [1, 2, 3],
                }
            )
        )

        widget.set_wheel_data(wheel_df)

        _, speed_values = widget.speed_curve.getData()
        self.assertEqual(float(np.nanmax(speed_values)), 25.0)
        self.assertEqual(widget.summary_label.text(), "Max. Speed 25.00 km/h | Mean Speed 22.50 km/h")

    def test_downsample_xy_preserve_extrema_keeps_rare_minimum(self) -> None:
        x = np.arange(10_000, dtype=np.float64)
        y = np.full(10_000, 20.0, dtype=np.float64)
        y[123] = 0.0

        sampled_x, sampled_y = downsample_xy_preserve_extrema(x, y, 200)

        self.assertLessEqual(sampled_x.shape[0], 200)
        self.assertAlmostEqual(float(np.min(sampled_y)), 0.0, places=6)

    def test_summary_text_includes_rtc_start(self) -> None:
        export_dir = self._copy_export("LOG00053")
        bundle = open_export_session(export_dir)

        summary_text = build_summary_text(bundle)

        self.assertIn("RTC start:", summary_text)

    def test_rebuild_with_calibration_updates_loaded_bundle(self) -> None:
        export_dir = self._copy_export("LOG00053")
        bundle = open_export_session(export_dir)
        config = copy.deepcopy(bundle.session_config)
        config["front"]["mm_per_count"] = 0.1
        rebuilt = rebuild_session_analysis(export_dir, config)

        window = MainWindow()
        window._on_bundle_loaded(rebuilt)

        self.assertIn("front_travel_mm", rebuilt.derived_df.columns)
        self.assertGreater(window.metrics_table.rowCount(), 0)

    def test_refresh_views_previews_unsaved_calibration_controls(self) -> None:
        export_dir = self._copy_export("LOG00053")
        bundle = open_export_session(export_dir)
        window = MainWindow()
        window._on_bundle_loaded(bundle)

        controls = window.channel_controls["front"]
        controls["invert"].setChecked(True)
        window.refresh_views_from_controls()

        self.assertIsNotNone(window._preview_derived_df)

        preview_first = float(window._preview_derived_df.get_column("front_calibrated_counts")[0])
        self.assertEqual(window.status_label.text(), "Views refreshed from current controls (not saved)")

        config = window.gather_session_config()
        rebuilt = rebuild_session_analysis(export_dir, copy.deepcopy(config))
        rebuilt_first = float(rebuilt.derived_df.get_column("front_calibrated_counts")[0])

        self.assertAlmostEqual(preview_first, rebuilt_first, places=9)

    def test_calibration_controls_gather_manual_anchor_only(self) -> None:
        bundle = self._synthetic_compare_bundle("LOG00998", track="Track A")
        channel_config = {
            "zero_count": 1000,
            "invert": False,
            "mm_per_count": None,
            "full_scale_mm": 300.0,
            "sensor_full_scale_mm": 635.0,
            "travel_reference": "manual",
            "manual_reference_count": None,
            "velocity_filter_window": 3,
        }
        bundle.session_config = {
            "front": copy.deepcopy(channel_config),
            "rear": copy.deepcopy(channel_config),
            "plot_defaults": {},
            "breakdown": {},
        }
        window = MainWindow()
        window.current_bundle = bundle
        window.populate_controls_from_bundle(bundle)

        controls = window.channel_controls["front"]
        controls["manual_reference"].setText("1048.5")
        window.velocity_axis_combo.setCurrentIndex(window.velocity_axis_combo.findData("signed_log"))

        config = window.gather_session_config()

        self.assertEqual(config["front"]["travel_reference"], "manual")
        self.assertNotIn("reference_method", config["front"])
        self.assertNotIn("reference_percentile", config["front"])
        self.assertNotIn("reference_window_samples", config["front"])
        self.assertAlmostEqual(config["front"]["manual_reference_count"], 1048.5, places=6)
        self.assertEqual(config["plot_defaults"]["velocity_axis_mode"], "signed_log")

    def test_apply_calibration_template_copies_only_channel_calibration_fields(self) -> None:
        template = {
            "source_path": "data/template.BIN",
            "plot_defaults": {"selected_channel": "rear"},
            "front": {
                "zero_count": 1200,
                "invert": True,
                "travel_reference": "manual",
                "manual_reference_count": 1203.0,
                "velocity_filter_window": 7,
            },
            "rear": {
                "zero_count": 2200,
                "invert": False,
                "travel_reference": "manual",
                "manual_reference_count": None,
                "velocity_filter_window": 3,
            },
        }
        target = {
            "source_path": "data/target.BIN",
            "export_dir": "exports/target",
            "plot_defaults": {"selected_channel": "front"},
            "front": {
                "zero_count": 999,
                "invert": False,
                "travel_reference": "manual",
                "manual_reference_count": None,
            },
            "rear": {
                "zero_count": 1999,
                "invert": True,
                "travel_reference": "manual",
                "manual_reference_count": 2400.0,
            },
        }

        updated = apply_calibration_template(target, template)

        self.assertEqual(updated["source_path"], "data/target.BIN")
        self.assertEqual(updated["plot_defaults"]["selected_channel"], "front")
        self.assertEqual(updated["front"]["zero_count"], 999)
        self.assertTrue(updated["front"]["invert"])
        self.assertEqual(updated["front"].get("travel_reference"), "manual")
        self.assertNotIn("reference_method", updated["front"])
        self.assertAlmostEqual(updated["front"]["manual_reference_count"], 1203.0, places=6)
        self.assertEqual(updated["rear"]["zero_count"], 1999)
        self.assertFalse(updated["rear"]["invert"])
        self.assertEqual(updated["rear"].get("travel_reference"), "manual")

    def test_apply_calibration_to_all_confirms_and_starts_background_task(self) -> None:
        bundle = self._synthetic_compare_bundle("LOG00998", track="Track A")
        channel_config = {
            "zero_count": 1000,
            "invert": False,
            "mm_per_count": None,
            "full_scale_mm": 300.0,
            "sensor_full_scale_mm": 635.0,
            "travel_reference": "manual",
            "manual_reference_count": None,
            "velocity_filter_window": 3,
        }
        bundle.session_config = {
            "front": copy.deepcopy(channel_config),
            "rear": copy.deepcopy(channel_config),
            "plot_defaults": {},
            "breakdown": {},
        }
        entries = [
            SessionDatabaseEntry(
                session_id="LOG00047",
                export_dir=Path("exports/LOG00047"),
                source_path=Path("data/LOG00047.BIN"),
                metadata={"session_id": "LOG00047"},
                summary={"header": {"start_epoch": 1}},
                sort_timestamp=1,
            ),
            SessionDatabaseEntry(
                session_id="LOG00053",
                export_dir=Path("exports/LOG00053"),
                source_path=Path("data/LOG00053.BIN"),
                metadata={"session_id": "LOG00053"},
                summary={"header": {"start_epoch": 2}},
                sort_timestamp=2,
            ),
        ]
        window = MainWindow()
        window.current_bundle = bundle
        window.session_database_entries = entries
        window.populate_controls_from_bundle(bundle)
        self.assertFalse(window.apply_all_button.isEnabled())

        with (
            mock.patch(
                "scripts.postprocess_gui_app.app.QtWidgets.QMessageBox.question",
                return_value=QtWidgets.QMessageBox.StandardButton.Yes,
            ) as question,
            mock.patch.object(window, "run_task") as run_task,
        ):
            window.apply_calibration_to_all()

        question.assert_called_once()
        run_task.assert_called_once()
        message, _, callback = run_task.call_args.args
        self.assertEqual(message, "Applying calibration to 2 sets")
        self.assertEqual(callback.__func__, window._on_all_calibrations_applied.__func__)

    def test_view_settings_change_refreshes_only_plot_views(self) -> None:
        export_dir = self._copy_export("LOG00053")
        bundle = open_export_session(export_dir)
        window = MainWindow()
        window._on_bundle_loaded(bundle)

        with (
            mock.patch("scripts.postprocess_gui_app.app.save_session_config") as save_mock,
            mock.patch.object(window, "_refresh_plot_views") as refresh_plot_views,
            mock.patch.object(window, "_refresh_analysis_views") as refresh_analysis_views,
            mock.patch.object(window, "refresh_views") as refresh_views,
        ):
            window.on_view_settings_changed()

        save_mock.assert_called_once()
        refresh_plot_views.assert_called_once()
        refresh_analysis_views.assert_not_called()
        refresh_views.assert_not_called()

    def test_run_task_delivers_finished_for_fast_background_work(self) -> None:
        window = MainWindow()
        state = {"done": False, "result": None}

        window.run_task(
            "Quick task",
            lambda: 123,
            lambda result: state.update(done=True, result=result),
        )

        deadline = time.time() + 2.0
        while time.time() < deadline and not state["done"]:
            self.app.processEvents()
            time.sleep(0.01)

        self.assertTrue(state["done"])
        self.assertEqual(state["result"], 123)
        self.assertEqual(window.status_label.text(), "Quick task complete")

    def test_parse_optional_float_accepts_ratio_input(self) -> None:
        self.assertAlmostEqual(parse_optional_float("315/4035"), 315.0 / 4035.0, places=12)

    def test_downsample_xy_limits_point_count(self) -> None:
        x = np.arange(10000, dtype=np.float64)
        y = x * 2.0

        reduced_x, reduced_y = downsample_xy(x, y, 2500)

        self.assertEqual(reduced_x.size, 2500)
        self.assertEqual(reduced_y.size, 2500)
        self.assertEqual(float(reduced_x[0]), 0.0)
        self.assertEqual(float(reduced_x[-1]), 9999.0)

    def test_open_path_in_file_manager_uses_windows_startfile(self) -> None:
        with mock.patch("scripts.postprocess_gui_app.app.os.startfile") as startfile_mock:
            result = open_path_in_file_manager(Path("exports"))

        self.assertTrue(result)
        startfile_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
