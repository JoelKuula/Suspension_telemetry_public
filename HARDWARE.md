# HARDWARE

Current hardware baseline for the suspension telemetry logger.

## In Use Now
- MCU and logger: Teensy 4.1
- RTC backup: Teensy 4.1 RTC with CR2032 backup battery
- IMU hardware retained for later: SparkFun 6DoF IMU Breakout, `ISM330DHCX` (`SEN-19764`)
- Travel sensor currently installed: TE Connectivity `SP1-25` string potentiometer for fork testing
- Planned second travel sensor: likely TE Connectivity `SP1-12` for shock testing after fork validation
- Speed sensor: front wheel reed switch plus magnet
- Storage: Teensy 4.1 microSD
- User control: handlebar-mounted latching rocker switch for logging start and stop
- Power: USB power bank into the Teensy USB connector
- Packaging target: rigid under-seat enclosure near the bike center mass

## Current State
- The logger has already powered up and run successfully in bench-style living-room testing.
- The active firmware slice logs analog travel, front wheel speed, status events, and RTC-backed session start metadata.
- The IMU hardware is the `ISM330DHCX`, but active IMU logging is deferred until the enclosure and mounting arrangement are rigid enough for meaningful data.
- When IMU logging returns, it will use primary 4-wire SPI, not Qwiic or I2C.
- The intended long-term IMU mounting location remains under the seat on a rigidly mounted logger assembly.
- Logged bike-axis convention is `+X forward`, `+Y left`, `+Z up`.
- Rear suspension measurement remains raw shock stroke only; any wheel-travel conversion is an offline step.
- Wheel speed comes from the front wheel only.
- The string pot excitation baseline is the Teensy `3.3V` rail.
- Teensy 4.1 `LED_BUILTIN` shares pin `13`, so firmware must not use the built-in LED while pin `13` is the IMU SPI clock.
- The RTC remains in use because wall-clock session start times are useful for later BIN indexing and database work.
- Remaining hardware details to finalize later: analog filtering and protection, power-bank auto-shutdown mitigation, IMU remounting, and environmental sealing.

## IMU SPI Wiring
Reserved for the future IMU-return slice; not used by the current analog-plus-wheel firmware.

SparkFun labels the SPI pins from the sensor-board side:
- `SCL/SPC` -> Teensy pin `13` (`SCK`)
- `SDA/SDI` -> Teensy pin `11` (`PICO/MOSI`)
- `SDO/SA0` -> Teensy pin `12` (`POCI/MISO`)
- `CS` -> Teensy pin `10`
- `INT1` -> Teensy pin `2`
- `INT2` -> Teensy pin `3` if needed later

Do not confuse these with the auxiliary interface pins such as `SDX`, `SCX`, `SDO_AUX`, `POCI_AUX`, or `OCS_AUX`.

## Pinout

| Function | Device wire / signal | Teensy 4.1 pin | Notes |
| --- | --- | --- | --- |
| Fork travel sensor | `SP1-25 +in` | `3.3V` | Excitation supply |
| Fork travel sensor | `SP1-25 common` | `GND` | Sensor ground |
| Fork travel sensor | `SP1-25 +out / wiper` | `20 (A6)` | Analog input |
| Shock travel sensor, planned | `SP1-12 +in` | `3.3V` | Planned second analog channel |
| Shock travel sensor, planned | `SP1-12 common` | `GND` | Planned second analog ground |
| Shock travel sensor, planned | `SP1-12 +out / wiper` | `19 (A5)` | Planned second analog input |
| IMU power | `VCC` | `3.3V` | SparkFun board is `3.3V` logic and supply |
| IMU power | `GND` | `GND` | Common ground |
| IMU SPI | `SCL / SPC` | `13` | SPI clock |
| IMU SPI | `SDA / SDI` | `11` | Teensy `PICO/MOSI` |
| IMU SPI | `SDO / SA0` | `12` | Teensy `POCI/MISO` |
| IMU SPI | `CS` | `10` | Chip select |
| IMU interrupt | `INT1` | `2` | Reserved for future IMU-return slice |
| IMU interrupt, optional | `INT2` | `3` | Reserve if needed later |
| Wheel speed | Reed switch wire 1 | `18` | Configure as `INPUT_PULLUP` |
| Wheel speed | Reed switch wire 2 | `GND` | Switch closes to ground |
| Handlebar logging switch | Switch wire 1 | `17` | Configure as `INPUT_PULLUP`; flipped on pulls the pin low |
| Handlebar logging switch | Switch wire 2 | `GND` | Latching switch returns high when flipped off |
| USB power | Teensy USB connector | USB power bank | Main supply path |
