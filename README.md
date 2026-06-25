# Suspension Telemetry

DIY suspension telemetry logger and post-processing tools for enduro and motocross bikes.

The current project state is a prototype. The active logger slice uses a Teensy 4.1 to record front fork travel, rear shock stroke, front wheel pulse timing, status records, and RTC-backed session start timestamps.

This repo is intended to be useful to riders and builders who want to inspect, adapt, or improve the system. It is not a certified product or a complete suspension tuning solution.

## Prototype Status
- Active firmware scope: analog travel logging, front wheel pulse logging, status diagnostics, RTC-backed session metadata, and microSD binary session files.
- Active hardware baseline: Teensy 4.1, string potentiometer travel sensors (front only for now), front wheel reed switch, handlebar logging switch, microSD storage, RTC battery, and USB power-bank supply.
- Active analysis scope: binary inspection/export, desktop post-processing, occupancy/histogram views, speed checks, balance views (tbd), braking analysis, breakdown tables (tbd), and comparison workflows.

Read [`DISCLAIMER.md`](DISCLAIMER.md) before using this on a bike.

For the first public snapshot, see [`RELEASE_NOTES.md`](RELEASE_NOTES.md).

## Repository Layout
- [`HARDWARE.md`](HARDWARE.md): current hardware stack, mounting assumptions, and pinout.
- [`platformio.ini`](platformio.ini): PlatformIO project configuration for Teensy 4.1 firmware.
- [`src/main.cpp`](src/main.cpp): current integrated logger firmware.
- [`src/uploadcheck_blink.cpp`](src/uploadcheck_blink.cpp): minimal upload-check firmware using the external status LED on pin `14`.
- [`scripts/inspect_log.py`](scripts/inspect_log.py): parser and structural sanity checker for `.BIN` session files.
- [`scripts/export_log.py`](scripts/export_log.py): exporter from `.BIN` session files to CSV outputs.
- [`scripts/log_parser.py`](scripts/log_parser.py): shared binary parser used by the inspection and export tools.
- [`scripts/plot_occupancy_surface.py`](scripts/plot_occupancy_surface.py): CLI occupancy heatmap plotter.
- [`scripts/postprocess_gui.py`](scripts/postprocess_gui.py): desktop post-processing workbench launcher.
- [`scripts/postprocess_gui_app/`](scripts/postprocess_gui_app): shared backend services and PySide6 GUI implementation.
- [`scripts/postprocess_quicklook.py`](scripts/postprocess_quicklook.py): experimental Qt-free summary generator for `.BIN` files or existing exports.
- [`notebooks/trackside_quicklook.ipynb`](notebooks/trackside_quicklook.ipynb): experimental Colab wrapper for the quicklook path.
- [`data/`](data): sample raw `.BIN` sessions for parser, exporter, GUI, and analysis experiments.
- [`installation-pics/`](installation-pics): installation reference photos.
- [`3d-printed-parts-stl/`](3d-printed-parts-stl): STL files for prototype printed parts.
- [`tests/`](tests): backend and GUI regression tests for the analysis tooling.
- [`requirements.txt`](requirements.txt): Python dependencies for the post-processing tools.

Generated exports are intentionally not tracked. Running the exporter or GUI creates `exports/<session>/` locally.

## Quick Start
1. Read [`HARDWARE.md`](HARDWARE.md) before changing firmware, wiring, mounting, or sensor assumptions.
2. Install Python dependencies for post-processing:
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. Inspect a sample log:
   ```powershell
   python scripts/inspect_log.py data/LOG00036.BIN
   ```
4. Export a sample log:
   ```powershell
   python scripts/export_log.py data/LOG00036.BIN --output-dir exports/LOG00036
   ```
5. Launch the desktop workbench on Windows:
   ```powershell
   .\postprocess_gui.cmd
   ```

The workbench can open a raw `.BIN` file and refresh `exports/<session>/`, or open an existing local export folder directly. GUI-derived caches and plots live under `exports/<session>/analysis/` and are generated locally.

## Firmware
The current integrated firmware target is the analog-plus-wheel ride logger in [`src/main.cpp`](src/main.cpp). It is designed around:
- deterministic analog sampling,
- wheel pulses as timestamped events,
- status records for boot/session/fault evidence,
- RTC-backed session start metadata,
- SD storage as a background path rather than an acquisition timing dependency.

Build with PlatformIO:
```powershell
platformio run -e teensy41
```

Use the upload-check firmware only when USB programming behavior needs to be isolated from the logger:
```powershell
platformio run -e teensy41_uploadcheck
```

## Desktop Workbench
Main views:
- `Session`: export summary, record counts, and status/fault rows.
- `Histograms`: front and rear travel plus velocity occupancy distributions.
- `Occupancy`: front and rear velocity-travel heatmaps.
- `Balance`: front/rear normalized travel comparison (tbd).
- `Braking`: front braking behavior.
- `Breakdown`: riding-state, event, finding, and data-quality tables (tbd).
- `Speed`: front wheel pulse-derived speed and pulse period.
- `Metrics`: per-channel stroke, velocity, and ADC checks.
- `Compare`: multi-session comparison workflow.

Per-session calibration is stored locally in `exports/<session>/analysis/session_config.json`.

## Experimental Quicklook
The quicklook path exists, but it is not the primary supported workflow yet. It can generate a small text/JSON summary and two figures without the PySide6 desktop GUI:
```powershell
python scripts/postprocess_quicklook.py --input data/LOG00036.BIN
```

Default outputs are written under `exports/<session>/analysis/quicklook/`.

## Sample Data
The repo keeps selected raw `.BIN` files under [`data/`](data), including the newer `LOG00041.BIN` through `LOG00053.BIN` sessions, so users can try the parser, exporter, quicklook script, GUI, and braking-analysis workflow without first building hardware.

Generated CSV exports, derived Parquet caches, plots, and per-session local calibration files are ignored because they can be regenerated from the raw `.BIN` files.

## Safety And Secrets
- Do not commit secrets, credentials, tokens, private keys, or private ride metadata.
- Treat bike-mounted hardware changes as safety-relevant.
- Verify wiring, power, waterproofing, and mechanical clearance before riding.
- Do not rely on the analysis output as professional suspension advice.

## License
This project is released under the [`MIT License`](LICENSE). MIT is permissive: others may use, copy, modify, merge, publish, distribute, sublicense, and sell copies, but the copyright and permission notice must be included with copies or substantial portions of the project.

## Contributing
Focused improvements are welcome. Please keep hardware assumptions explicit, preserve raw timing/data-model traceability, and update README/HARDWARE when architecture, hardware, wiring, or scope changes materially.
