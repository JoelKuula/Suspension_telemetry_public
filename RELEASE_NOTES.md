# v0.1-prototype

First public prototype snapshot of the DIY suspension telemetry project.

## Included
- Teensy 4.1 analog-plus-wheel logger firmware.
- Hardware wiring reference.
- Sample `.BIN` log files for parser, exporter, quicklook, and GUI experiments.
- Python parser/export tools.
- Desktop post-processing GUI.
- Experimental quicklook script and Colab notebook.
- Installation reference photos.
- Prototype STL files for printed parts.

## Known Limitations
- Prototype only; not a certified safety device or professional suspension tuning system.
- Active IMU logging is deferred until the logger can be mounted rigidly enough for useful inertial data.
- Analysis outputs are experimental and may be affected by calibration, sensor noise, mounting flex, missed pulses, or software bugs.
- Generated `exports/` are not tracked and should be regenerated from `.BIN` files.
- Rear channel represents shock stroke, not rear wheel travel.

## Safety
Read `DISCLAIMER.md` before using this project on a motorcycle.
