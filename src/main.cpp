#include <Arduino.h>
#include <IntervalTimer.h>
#include <SD.h>
#include <TimeLib.h>

#include <cstdio>
#include <cstring>

namespace {

constexpr uint8_t kButtonPin = 17;
constexpr uint8_t kLedPin = 14;
constexpr uint8_t kReedPin = 18;
constexpr uint8_t kFrontPotPin = A6;
constexpr uint8_t kRearPotPin = A5;
constexpr uint8_t kImuCsPin = 10;
constexpr uint8_t kImuInt1Pin = 2;

constexpr uint32_t kDebounceMs = 20;
constexpr uint32_t kArmBlinkMs = 120;

constexpr uint32_t kAnalogSampleRateHz = 500;
constexpr uint32_t kAnalogTimerPeriodUs = 1000000UL / kAnalogSampleRateHz;
constexpr uint8_t kAnalogResolutionBits = 12;
constexpr uint8_t kAnalogAveraging = 8;
// At 500 Hz each pending slot represents 2 ms of service latency. Eight slots
// were not enough to ride through occasional SD flush stalls seen in ride logs,
// so keep a larger timestamp cushion without changing the sample schedule.
constexpr uint16_t kAnalogMaxPendingTicks = 32;
constexpr uint8_t kAnalogTimestampCapacity = kAnalogMaxPendingTicks;

constexpr uint32_t kReedDebounceUs = 3000;
constexpr float kWheelCircumferenceM = 2.213f;

constexpr size_t kWriterBufferBytes = 65536;
constexpr size_t kWriterChunkBytes = 1024;
constexpr uint32_t kWriterFlushIntervalMs = 250;
constexpr uint32_t kFlushTimeoutMs = 5000;

constexpr uint32_t kMinValidRtcEpoch = 1704067200UL;
constexpr uint16_t kFormatVersion = 1;
constexpr uint32_t kFirmwareVersion = 0x00010000UL;

#ifdef LOGGER_VALIDATION_AUTORUN
constexpr uint32_t kValidationStartDelayMs = 12000;
constexpr uint32_t kValidationRunDurationMs = LOGGER_VALIDATION_DURATION_MS;
#endif

constexpr uint8_t kSensorMaskFrontAnalog = 0x01;
constexpr uint8_t kSensorMaskRearAnalog = 0x02;
constexpr uint8_t kSensorMaskWheel = 0x04;
constexpr uint8_t kSensorMaskRtc = 0x10;

enum class LoggerState : uint8_t {
  IDLE,
  LOGGING,
};

enum class StatCode : uint16_t {
  BOOT = 1,
  RTC_VALID = 2,
  RTC_INVALID = 3,
  SD_READY = 4,
  SD_ERROR = 5,
  IMU_READY = 6,
  IMU_ERROR = 7,
  SESSION_START = 8,
  SESSION_STOP = 9,
  CLEAN_CLOSE = 10,
  QUEUE_OVERFLOW = 11,
  ANALOG_OVERRUN = 12,
  REED_OVERFLOW = 13,
  IMU_FIFO_OVERFLOW = 14,
  WRITER_ERROR = 15,
  IMU_READ_ERROR = 16,
};

enum class SessionStopReason : int32_t {
  UNKNOWN = 0,
  SWITCH_OFF = 1,
  VALIDATION_AUTORUN = 2,
};

struct RingBuffer {
  uint8_t data[kWriterBufferBytes] = {};
  size_t head = 0;
  size_t tail = 0;
  size_t size = 0;

  void reset() {
    head = 0;
    tail = 0;
    size = 0;
  }

  size_t freeSpace() const { return kWriterBufferBytes - size; }

  bool push(const uint8_t* bytes, size_t length) {
    if (length > freeSpace()) {
      return false;
    }

    size_t first_chunk = length;
    const size_t head_to_end = kWriterBufferBytes - head;
    if (first_chunk > head_to_end) {
      first_chunk = head_to_end;
    }

    memcpy(&data[head], bytes, first_chunk);
    const size_t remaining = length - first_chunk;
    if (remaining > 0) {
      memcpy(data, bytes + first_chunk, remaining);
    }

    head = (head + length) % kWriterBufferBytes;
    size += length;
    return true;
  }

  size_t contiguousReadable() const {
    if (size == 0) {
      return 0;
    }
    if (head > tail) {
      return head - tail;
    }
    return kWriterBufferBytes - tail;
  }

  const uint8_t* readPtr() const { return &data[tail]; }

  void pop(size_t length) {
    if (length > size) {
      length = size;
    }
    tail = (tail + length) % kWriterBufferBytes;
    size -= length;
  }
};

struct ReedEvent {
  uint32_t timestamp_us;
  uint32_t interval_us;
  uint32_t pulse_count;
};

constexpr uint8_t kReedEventCapacity = 32;
volatile ReedEvent gReedEvents[kReedEventCapacity] = {};
volatile uint8_t gReedHead = 0;
volatile uint8_t gReedTail = 0;
volatile uint8_t gReedCount = 0;
volatile uint32_t gReedDropped = 0;
volatile uint32_t gReedPulseCount = 0;
volatile uint32_t gReedLastPulseUs = 0;

volatile bool gSessionActive = false;
volatile uint32_t gAnalogTickTimestampsUs[kAnalogTimestampCapacity] = {};
volatile uint8_t gAnalogTimestampHead = 0;
volatile uint8_t gAnalogTimestampTail = 0;
volatile uint16_t gAnalogTicksPending = 0;
volatile uint32_t gAnalogTickOverruns = 0;

IntervalTimer gAnalogTimer;
File gSessionFile;
RingBuffer gWriterBuffer;

LoggerState gLoggerState = LoggerState::IDLE;
bool gLoggerRunning = false;
bool gLastRawButtonPressed = false;
bool gStableButtonPressed = false;
bool gLastStableButtonPressed = false;
uint32_t gLastRawChangeMs = 0;

bool gSdReady = false;
bool gRtcValid = false;
bool gFaultLatched = false;
uint32_t gRtcEpochAtBoot = 0;

uint32_t gSessionIndex = 0;
char gSessionFilename[16] = {};
uint32_t gSessionStartEpoch = 0;
uint32_t gSessionStartMicros = 0;
uint32_t gRecordSequence = 0;
uint32_t gAnalogSampleIndex = 0;
uint32_t gLastWriterFlushMs = 0;

uint32_t gDroppedRecordCount = 0;
uint32_t gLastDroppedTag = 0;
uint32_t gReportedDroppedRecordCount = 0;

uint32_t gReportedAnalogTickOverruns = 0;
uint32_t gReportedReedDropped = 0;
uint32_t gWriterErrorCount = 0;
uint32_t gReportedWriterErrorCount = 0;

#ifdef LOGGER_VALIDATION_AUTORUN
bool gValidationAutoStartPending = false;
bool gValidationDumpPending = false;
uint32_t gValidationArmedAtMs = 0;
uint32_t gValidationStopAtMs = 0;
#endif

#pragma pack(push, 1)
struct SessionFileHeader {
  char magic[8];
  uint16_t format_version;
  uint16_t header_size;
  uint32_t firmware_version;
  uint32_t build_epoch;
  char build_id[24];
  uint32_t start_epoch;
  uint32_t start_micros;
  uint32_t session_index;
  uint8_t rtc_valid;
  uint8_t enabled_mask;
  uint8_t analog_resolution_bits;
  uint8_t analog_averaging;
  uint16_t analog_rate_hz;
  uint16_t imu_rate_hz;
  uint16_t imu_watermark_frames;
  uint16_t record_header_size;
  uint8_t button_pin;
  uint8_t led_pin;
  uint8_t reed_pin;
  uint8_t imu_int1_pin;
  uint8_t front_pot_pin;
  uint8_t rear_pot_pin;
  uint8_t imu_cs_pin;
  uint8_t reserved0;
  uint32_t reserved[6];
};

struct RecordHeader {
  char tag[4];
  uint16_t payload_length;
  uint16_t flags;
  uint32_t sequence;
  uint32_t host_timestamp_us;
};

struct ConfigPayload {
  uint16_t analog_rate_hz;
  uint8_t analog_resolution_bits;
  uint8_t analog_averaging;
  uint32_t button_debounce_ms;
  uint32_t button_hold_start_ms;
  uint32_t button_hold_stop_ms;
  uint32_t reed_debounce_us;
  float wheel_circumference_m;
  uint16_t imu_accel_range_g;
  uint16_t imu_gyro_range_dps;
  uint16_t imu_odr_hz;
  uint16_t imu_watermark_frames;
  uint8_t imu_fifo_mode;
  uint8_t imu_timestamp_decimation;
  uint16_t reserved;
};

struct StatPayload {
  uint16_t code;
  uint16_t flags;
  int32_t value0;
  int32_t value1;
};

struct AnalogPayload {
  uint32_t sample_index;
  uint16_t front_raw;
  uint16_t rear_raw;
};

struct WheelPayload {
  uint32_t pulse_count;
  uint32_t interval_us;
};
#pragma pack(pop)

static_assert(sizeof(SessionFileHeader) == 100, "SessionFileHeader size changed");
static_assert(sizeof(RecordHeader) == 16, "RecordHeader size changed");

bool startLogging();
void stopLogging(SessionStopReason reason = SessionStopReason::UNKNOWN);

uint32_t packFourCc(const char* tag) {
  return uint32_t(uint8_t(tag[0])) | (uint32_t(uint8_t(tag[1])) << 8) |
         (uint32_t(uint8_t(tag[2])) << 16) | (uint32_t(uint8_t(tag[3])) << 24);
}

time_t getTeensy3Time() { return Teensy3Clock.get(); }

uint8_t monthFromAbbrev(const char* mon) {
  if (strcmp(mon, "Jan") == 0) return 1;
  if (strcmp(mon, "Feb") == 0) return 2;
  if (strcmp(mon, "Mar") == 0) return 3;
  if (strcmp(mon, "Apr") == 0) return 4;
  if (strcmp(mon, "May") == 0) return 5;
  if (strcmp(mon, "Jun") == 0) return 6;
  if (strcmp(mon, "Jul") == 0) return 7;
  if (strcmp(mon, "Aug") == 0) return 8;
  if (strcmp(mon, "Sep") == 0) return 9;
  if (strcmp(mon, "Oct") == 0) return 10;
  if (strcmp(mon, "Nov") == 0) return 11;
  if (strcmp(mon, "Dec") == 0) return 12;
  return 1;
}

uint32_t compileTimeToEpoch() {
  char mon[4] = {};
  int day = 1;
  int year = 2026;
  int hour_value = 0;
  int minute_value = 0;
  int second_value = 0;

  sscanf(__DATE__, "%3s %d %d", mon, &day, &year);
  sscanf(__TIME__, "%d:%d:%d", &hour_value, &minute_value, &second_value);

  tmElements_t tm = {};
  tm.Year = CalendarYrToTm(year);
  tm.Month = monthFromAbbrev(mon);
  tm.Day = day;
  tm.Hour = hour_value;
  tm.Minute = minute_value;
  tm.Second = second_value;
  return makeTime(tm);
}

void formatDateTime(time_t value, char* buffer, size_t buffer_size) {
  if (buffer_size == 0) {
    return;
  }

  if (value == 0) {
    snprintf(buffer, buffer_size, "0000-00-00 00:00:00");
    return;
  }

  tmElements_t tm = {};
  breakTime(value, tm);
  snprintf(buffer, buffer_size, "%04d-%02d-%02d %02d:%02d:%02d", tmYearToCalendar(tm.Year), tm.Month, tm.Day,
           tm.Hour, tm.Minute, tm.Second);
}

void setLed(bool on) { digitalWrite(kLedPin, on ? HIGH : LOW); }

bool loggingPattern(uint32_t now_ms) {
  const uint32_t t = now_ms % 1200UL;
  return (t < 80UL) || (t >= 200UL && t < 280UL);
}

bool fastBlinkPattern(uint32_t now_ms) { return ((now_ms / kArmBlinkMs) % 2UL) == 0; }

void analogTimerIsr() {
  if (!gSessionActive) {
    return;
  }

  if (gAnalogTicksPending < kAnalogMaxPendingTicks) {
    // Capture the timer tick time in the ISR so ANLG records reflect sampling schedule,
    // not the later main-loop service latency.
    gAnalogTickTimestampsUs[gAnalogTimestampHead] = micros();
    gAnalogTimestampHead = uint8_t((gAnalogTimestampHead + 1U) % kAnalogTimestampCapacity);
    ++gAnalogTicksPending;
  } else {
    ++gAnalogTickOverruns;
  }
}

void resetReedState() {
  noInterrupts();
  gReedHead = 0;
  gReedTail = 0;
  gReedCount = 0;
  gReedDropped = 0;
  gReedPulseCount = 0;
  gReedLastPulseUs = 0;
  interrupts();
}

void reedIsr() {
  if (!gSessionActive) {
    return;
  }

  const uint32_t now_us = micros();
  const uint32_t last_pulse_us = gReedLastPulseUs;
  if (last_pulse_us != 0 && (uint32_t)(now_us - last_pulse_us) < kReedDebounceUs) {
    return;
  }

  const uint32_t next_pulse_count = gReedPulseCount + 1UL;
  const uint32_t interval_us = (last_pulse_us == 0) ? 0UL : (now_us - last_pulse_us);

  gReedPulseCount = next_pulse_count;
  gReedLastPulseUs = now_us;

  if (gReedCount >= kReedEventCapacity) {
    ++gReedDropped;
    return;
  }

  gReedEvents[gReedHead].timestamp_us = now_us;
  gReedEvents[gReedHead].interval_us = interval_us;
  gReedEvents[gReedHead].pulse_count = next_pulse_count;
  gReedHead = uint8_t((gReedHead + 1U) % kReedEventCapacity);
  ++gReedCount;
}

void waitForSerial() {
  Serial.begin(115200);
  const uint32_t start_ms = millis();
  while (!Serial && (millis() - start_ms) < 3000UL) {
    delay(10);
  }
}

bool isRtcUsable(time_t epoch_value) { return epoch_value >= kMinValidRtcEpoch; }

void initRtc() {
  setSyncProvider(getTeensy3Time);
  const time_t rtc_now = now();
  gRtcValid = (timeStatus() == timeSet) && isRtcUsable(rtc_now);
  gRtcEpochAtBoot = gRtcValid ? uint32_t(rtc_now) : 0UL;
}

bool initSd() {
  if (!SD.begin(BUILTIN_SDCARD)) {
    return false;
  }
  return true;
}

uint8_t enabledSensorMask() {
  uint8_t mask = kSensorMaskFrontAnalog | kSensorMaskRearAnalog | kSensorMaskWheel;
  if (gRtcValid) {
    mask |= kSensorMaskRtc;
  }
  return mask;
}

void fillBuildId(char* buffer, size_t buffer_size) {
  snprintf(buffer, buffer_size, "%s %s", __DATE__, __TIME__);
}

bool findNextSessionFilename(char* buffer, size_t buffer_size, uint32_t* session_index) {
  for (uint32_t candidate = 1; candidate <= 99999UL; ++candidate) {
    snprintf(buffer, buffer_size, "/LOG%05lu.BIN", static_cast<unsigned long>(candidate));
    if (!SD.exists(buffer)) {
      *session_index = candidate;
      return true;
    }
  }
  return false;
}

bool writeSessionHeader(uint32_t session_index, const char* filename, uint32_t start_epoch, uint32_t start_micros) {
  SessionFileHeader header = {};
  memcpy(header.magic, "STLOG1\0", 8);
  header.format_version = kFormatVersion;
  header.header_size = sizeof(header);
  header.firmware_version = kFirmwareVersion;
  header.build_epoch = compileTimeToEpoch();
  fillBuildId(header.build_id, sizeof(header.build_id));
  header.start_epoch = start_epoch;
  header.start_micros = start_micros;
  header.session_index = session_index;
  header.rtc_valid = gRtcValid ? 1U : 0U;
  header.enabled_mask = enabledSensorMask();
  header.analog_resolution_bits = kAnalogResolutionBits;
  header.analog_averaging = kAnalogAveraging;
  header.analog_rate_hz = kAnalogSampleRateHz;
  header.imu_rate_hz = 0;
  header.imu_watermark_frames = 0;
  header.record_header_size = sizeof(RecordHeader);
  header.button_pin = kButtonPin;
  header.led_pin = kLedPin;
  header.reed_pin = kReedPin;
  header.imu_int1_pin = kImuInt1Pin;
  header.front_pot_pin = kFrontPotPin;
  header.rear_pot_pin = kRearPotPin;
  header.imu_cs_pin = kImuCsPin;
  header.reserved[0] = packFourCc("CFG ");
  header.reserved[1] = packFourCc("STAT");
  header.reserved[2] = packFourCc("ANLG");
  header.reserved[3] = packFourCc("WSPD");
  header.reserved[4] = packFourCc("IMU ");
  header.reserved[5] = filename[1] ? filename[1] : 0;

  const size_t written = gSessionFile.write(reinterpret_cast<const uint8_t*>(&header), sizeof(header));
  return written == sizeof(header);
}

bool queueRecordRaw(const char* tag, uint16_t flags, uint32_t host_timestamp_us, const void* payload, uint16_t payload_length,
                    bool count_drops) {
  RecordHeader header = {};
  memcpy(header.tag, tag, 4);
  header.payload_length = payload_length;
  header.flags = flags;
  header.sequence = gRecordSequence++;
  header.host_timestamp_us = host_timestamp_us;

  const size_t total_size = sizeof(header) + payload_length;
  if (gWriterBuffer.freeSpace() < total_size) {
    if (count_drops) {
      ++gDroppedRecordCount;
      gLastDroppedTag = packFourCc(tag);
    }
    return false;
  }

  gWriterBuffer.push(reinterpret_cast<const uint8_t*>(&header), sizeof(header));
  if (payload_length > 0) {
    gWriterBuffer.push(reinterpret_cast<const uint8_t*>(payload), payload_length);
  }
  return true;
}

template <typename T>
bool queueRecord(const char* tag, uint16_t flags, uint32_t host_timestamp_us, const T& payload, bool count_drops = true) {
  return queueRecordRaw(tag, flags, host_timestamp_us, &payload, sizeof(T), count_drops);
}

bool queueStat(StatCode code, int32_t value0, int32_t value1, uint16_t flags = 0, bool count_drops = false,
               uint32_t host_timestamp_us = micros()) {
  StatPayload payload = {};
  payload.code = static_cast<uint16_t>(code);
  payload.flags = flags;
  payload.value0 = value0;
  payload.value1 = value1;
  return queueRecord("STAT", 0, host_timestamp_us, payload, count_drops);
}

bool queueConfigRecord() {
  ConfigPayload payload = {};
  payload.analog_rate_hz = kAnalogSampleRateHz;
  payload.analog_resolution_bits = kAnalogResolutionBits;
  payload.analog_averaging = kAnalogAveraging;
  payload.button_debounce_ms = kDebounceMs;
  payload.button_hold_start_ms = 0;
  payload.button_hold_stop_ms = 0;
  payload.reed_debounce_us = kReedDebounceUs;
  payload.wheel_circumference_m = kWheelCircumferenceM;
  payload.imu_accel_range_g = 0;
  payload.imu_gyro_range_dps = 0;
  payload.imu_odr_hz = 0;
  payload.imu_watermark_frames = 0;
  payload.imu_fifo_mode = 0;
  payload.imu_timestamp_decimation = 0;
  return queueRecord("CFG ", 0, gSessionStartMicros, payload);
}

bool flushWriterChunk() {
  if (!gSessionFile || gWriterBuffer.size == 0) {
    return true;
  }

  size_t length = gWriterBuffer.contiguousReadable();
  if (length > kWriterChunkBytes) {
    length = kWriterChunkBytes;
  }
  const size_t written = gSessionFile.write(gWriterBuffer.readPtr(), length);
  if (written == 0) {
    return false;
  }
  gWriterBuffer.pop(written);
  return true;
}

void noteWriterError() {
  ++gWriterErrorCount;
  gFaultLatched = true;
  Serial.println("Writer error");
}

bool flushWriterAll(uint32_t timeout_ms) {
  const uint32_t start_ms = millis();
  while (gWriterBuffer.size > 0) {
    if (!flushWriterChunk()) {
      noteWriterError();
      return false;
    }
    if ((millis() - start_ms) > timeout_ms) {
      noteWriterError();
      return false;
    }
  }
  gSessionFile.flush();
  gLastWriterFlushMs = millis();
  return true;
}

bool serviceWriter(bool force_flush) {
  if (!gSessionFile) {
    return false;
  }

  if (gWriterBuffer.size == 0) {
    if (force_flush) {
      gSessionFile.flush();
      gLastWriterFlushMs = millis();
    }
    return true;
  }

  if (!flushWriterChunk()) {
    noteWriterError();
    return false;
  }

  const uint32_t now_ms = millis();
  if (force_flush || (now_ms - gLastWriterFlushMs) >= kWriterFlushIntervalMs) {
    gSessionFile.flush();
    gLastWriterFlushMs = now_ms;
  }
  return true;
}

void emitStartupStats() {
  const int32_t ready_mask = (gSdReady ? 0x01 : 0x00) | (gRtcValid ? 0x04 : 0x00);
  queueStat(StatCode::BOOT, ready_mask, int32_t(gSessionIndex));
  if (gRtcValid) {
    queueStat(StatCode::RTC_VALID, int32_t(gSessionStartEpoch), 0);
  } else {
    queueStat(StatCode::RTC_INVALID, 0, 0);
  }
  queueStat(gSdReady ? StatCode::SD_READY : StatCode::SD_ERROR, gSdReady ? 1 : 0, 0);
  queueStat(StatCode::SESSION_START, int32_t(gSessionIndex), 0, 0, false, gSessionStartMicros);
}

void emitDeferredStats() {
  if (!gSessionFile) {
    return;
  }

  if (gDroppedRecordCount > gReportedDroppedRecordCount) {
    const uint32_t delta = gDroppedRecordCount - gReportedDroppedRecordCount;
    if (queueStat(StatCode::QUEUE_OVERFLOW, int32_t(delta), int32_t(gLastDroppedTag), 0, false)) {
      gReportedDroppedRecordCount = gDroppedRecordCount;
    }
  }

  if (gAnalogTickOverruns > gReportedAnalogTickOverruns) {
    const uint32_t delta = gAnalogTickOverruns - gReportedAnalogTickOverruns;
    if (queueStat(StatCode::ANALOG_OVERRUN, int32_t(delta), 0, 0, false)) {
      gReportedAnalogTickOverruns = gAnalogTickOverruns;
    }
  }

  if (gReedDropped > gReportedReedDropped) {
    const uint32_t delta = gReedDropped - gReportedReedDropped;
    if (queueStat(StatCode::REED_OVERFLOW, int32_t(delta), 0, 0, false)) {
      gReportedReedDropped = gReedDropped;
    }
  }

  if (gWriterErrorCount > gReportedWriterErrorCount) {
    const uint32_t delta = gWriterErrorCount - gReportedWriterErrorCount;
    if (queueStat(StatCode::WRITER_ERROR, int32_t(delta), 0, 0, false)) {
      gReportedWriterErrorCount = gWriterErrorCount;
    }
  }
}

bool popReedEvent(ReedEvent* event) {
  noInterrupts();
  if (gReedCount == 0) {
    interrupts();
    return false;
  }

  event->timestamp_us = gReedEvents[gReedTail].timestamp_us;
  event->interval_us = gReedEvents[gReedTail].interval_us;
  event->pulse_count = gReedEvents[gReedTail].pulse_count;
  gReedTail = uint8_t((gReedTail + 1U) % kReedEventCapacity);
  --gReedCount;
  interrupts();
  return true;
}

void serviceWheelEvents() {
  ReedEvent event = {};
  while (popReedEvent(&event)) {
    WheelPayload payload = {};
    payload.pulse_count = event.pulse_count;
    payload.interval_us = event.interval_us;
    queueRecord("WSPD", 0, event.timestamp_us, payload);
  }
}

uint16_t readPotSample(uint8_t pin) {
  // analogReadAveraging() already applies hardware-side averaging. Doing an
  // extra software loop here multiplies acquisition time and wastes margin in
  // the timer-driven analog service path.
  return uint16_t(analogRead(pin));
}

void serviceAnalogAcquisition() {
  while (true) {
    uint32_t sample_timestamp_us = 0;
    noInterrupts();
    if (gAnalogTicksPending == 0) {
      interrupts();
      break;
    }
    sample_timestamp_us = gAnalogTickTimestampsUs[gAnalogTimestampTail];
    gAnalogTimestampTail = uint8_t((gAnalogTimestampTail + 1U) % kAnalogTimestampCapacity);
    --gAnalogTicksPending;
    interrupts();

    AnalogPayload payload = {};
    payload.sample_index = gAnalogSampleIndex++;
    payload.front_raw = readPotSample(kFrontPotPin);
    payload.rear_raw = readPotSample(kRearPotPin);
    queueRecord("ANLG", 0, sample_timestamp_us, payload);
  }
}

#ifdef LOGGER_VALIDATION_AUTORUN
void dumpSessionFileHex() {
  if (gSessionFilename[0] == '\0') {
    Serial.println("VALIDATION_DUMP_ERROR missing filename");
    return;
  }

  File dump_file = SD.open(gSessionFilename, FILE_READ);
  if (!dump_file) {
    Serial.print("VALIDATION_DUMP_ERROR open ");
    Serial.println(gSessionFilename);
    return;
  }

  Serial.print("VALIDATION_DUMP_BEGIN ");
  Serial.print(gSessionFilename);
  Serial.print(" ");
  Serial.println(dump_file.size());

  constexpr char kHexDigits[] = "0123456789ABCDEF";
  uint8_t buffer[32] = {};
  char line[sizeof(buffer) * 2 + 1] = {};
  while (true) {
    const int bytes_read = dump_file.read(buffer, sizeof(buffer));
    if (bytes_read <= 0) {
      break;
    }

    for (int i = 0; i < bytes_read; ++i) {
      line[i * 2] = kHexDigits[(buffer[i] >> 4) & 0x0F];
      line[i * 2 + 1] = kHexDigits[buffer[i] & 0x0F];
    }
    line[bytes_read * 2] = '\0';
    Serial.println(line);
    Serial.send_now();
    delay(1);
  }

  dump_file.close();
  Serial.println("VALIDATION_DUMP_END");
  Serial.flush();
}

void serviceValidationAutorun(uint32_t now_ms) {
  if (gValidationAutoStartPending && !gLoggerRunning) {
    if ((now_ms - gValidationArmedAtMs) >= kValidationStartDelayMs) {
      gValidationAutoStartPending = false;
      if (startLogging()) {
        gValidationStopAtMs = now_ms + kValidationRunDurationMs;
      }
    }
  }

  if (gLoggerRunning && gValidationStopAtMs != 0 && (int32_t)(now_ms - gValidationStopAtMs) >= 0) {
    gValidationStopAtMs = 0;
    stopLogging(SessionStopReason::VALIDATION_AUTORUN);
    gLoggerState = LoggerState::IDLE;
    gValidationDumpPending = true;
  }

  if (gValidationDumpPending && !gLoggerRunning) {
    gValidationDumpPending = false;
    dumpSessionFileHex();
  }
}
#endif

void resetSessionRuntimeState() {
  gWriterBuffer.reset();
  gRecordSequence = 0;
  gAnalogSampleIndex = 0;
  gDroppedRecordCount = 0;
  gLastDroppedTag = 0;
  gReportedDroppedRecordCount = 0;
  gReportedAnalogTickOverruns = 0;
  gReportedReedDropped = 0;
  gWriterErrorCount = 0;
  gReportedWriterErrorCount = 0;
  gAnalogTimestampHead = 0;
  gAnalogTimestampTail = 0;
  gAnalogTicksPending = 0;
  gAnalogTickOverruns = 0;
  resetReedState();
}

bool startLogging() {
  if (!gSdReady || gFaultLatched) {
    Serial.println("Start refused: logger not ready");
    gFaultLatched = true;
    return false;
  }

  resetSessionRuntimeState();
  if (!findNextSessionFilename(gSessionFilename, sizeof(gSessionFilename), &gSessionIndex)) {
    Serial.println("Start refused: no free log filename");
    gFaultLatched = true;
    return false;
  }

  gSessionStartMicros = micros();
  gSessionStartEpoch = gRtcValid ? uint32_t(now()) : 0UL;

  gSessionFile = SD.open(gSessionFilename, FILE_WRITE);
  if (!gSessionFile) {
    Serial.println("Start refused: failed to open log file");
    gFaultLatched = true;
    return false;
  }

  if (!writeSessionHeader(gSessionIndex, gSessionFilename, gSessionStartEpoch, gSessionStartMicros)) {
    Serial.println("Start refused: failed to write file header");
    gFaultLatched = true;
    gSessionFile.close();
    return false;
  }

  gSessionActive = true;
  gLoggerRunning = true;
  gLoggerState = LoggerState::LOGGING;
  gLastWriterFlushMs = millis();
  queueConfigRecord();
  emitStartupStats();
  gAnalogTimer.begin(analogTimerIsr, kAnalogTimerPeriodUs);
  serviceWriter(true);

  Serial.print("LOGGING STARTED: ");
  Serial.println(gSessionFilename);
  return true;
}

void stopLogging(SessionStopReason reason) {
  if (!gLoggerRunning) {
    return;
  }

  gSessionActive = false;
  gLoggerRunning = false;
  gAnalogTimer.end();

  serviceAnalogAcquisition();
  serviceWheelEvents();

  queueStat(StatCode::SESSION_STOP, int32_t(gSessionIndex), int32_t(reason), 0, false, micros());
  emitDeferredStats();
  queueStat(StatCode::CLEAN_CLOSE, int32_t(gSessionIndex), 0, 0, false, micros());
  flushWriterAll(kFlushTimeoutMs);

  if (gSessionFile) {
    gSessionFile.close();
  }

  Serial.print("LOGGING STOPPED: ");
  Serial.println(gSessionFilename);
}

void updateDebouncedButton(uint32_t now_ms) {
  const bool raw_pressed = (digitalRead(kButtonPin) == LOW);
  if (raw_pressed != gLastRawButtonPressed) {
    gLastRawButtonPressed = raw_pressed;
    gLastRawChangeMs = now_ms;
  }

  if ((now_ms - gLastRawChangeMs) >= kDebounceMs) {
    gStableButtonPressed = raw_pressed;
  }
}

void updateStateMachine(uint32_t now_ms) {
  (void)now_ms;

  if (gStableButtonPressed == gLastStableButtonPressed) {
    return;
  }
  gLastStableButtonPressed = gStableButtonPressed;

  if (gStableButtonPressed) {
    Serial.println("Logging switch ON");
    if (!gLoggerRunning && !startLogging()) {
      gLoggerState = LoggerState::IDLE;
    }
    return;
  }

  Serial.println("Logging switch OFF");
  if (gLoggerRunning) {
    stopLogging(SessionStopReason::SWITCH_OFF);
    gLoggerState = LoggerState::IDLE;
  }
}

void updateLed(uint32_t now_ms) {
  if (gFaultLatched && !gLoggerRunning) {
    setLed(fastBlinkPattern(now_ms));
    return;
  }

  switch (gLoggerState) {
    case LoggerState::IDLE:
      setLed(true);
      break;
    case LoggerState::LOGGING:
      setLed(loggingPattern(now_ms));
      break;
  }
}

void printBootSummary() {
  char rtc_buffer[32] = {};
  formatDateTime(gRtcEpochAtBoot, rtc_buffer, sizeof(rtc_buffer));

  Serial.println();
  Serial.println("Integrated suspension telemetry logger");
  Serial.println("Board: Teensy 4.1");
  Serial.println("Pins: LED=14 BTN=17 REED=18 FRONT=A6 REAR=A5 IMU_CS=10 IMU_INT1=2");
  Serial.print("RTC valid: ");
  Serial.println(gRtcValid ? rtc_buffer : "no");
  Serial.print("SD ready: ");
  Serial.println(gSdReady ? "yes" : "no");
  Serial.println("IMU logging: deferred in this firmware slice.");

  if (gSdReady) {
    Serial.println("Flip the logging switch ON to start and OFF to stop.");
  } else {
    Serial.println("Logger faulted at boot. Fix SD readiness before starting a session.");
  }
}

}  // namespace

void setup() {
  pinMode(kLedPin, OUTPUT);
  pinMode(kButtonPin, INPUT_PULLUP);
  pinMode(kReedPin, INPUT_PULLUP);
  setLed(true);

  waitForSerial();
  delay(200);

  analogReadResolution(kAnalogResolutionBits);
  analogReadAveraging(kAnalogAveraging);

  initRtc();
  gSdReady = initSd();
  gFaultLatched = !gSdReady;

  attachInterrupt(digitalPinToInterrupt(kReedPin), reedIsr, FALLING);

  printBootSummary();

#ifdef LOGGER_VALIDATION_AUTORUN
  if (gSdReady) {
    gValidationAutoStartPending = true;
    gValidationArmedAtMs = millis();
    Serial.print("Validation autorun armed: start in ");
    Serial.print(kValidationStartDelayMs);
    Serial.print(" ms, duration ");
    Serial.print(kValidationRunDurationMs);
    Serial.println(" ms");
  }
#endif
}

void loop() {
  const uint32_t now_ms = millis();

  updateDebouncedButton(now_ms);
  updateStateMachine(now_ms);
  updateLed(now_ms);

#ifdef LOGGER_VALIDATION_AUTORUN
  serviceValidationAutorun(now_ms);
#endif

  if (gLoggerRunning) {
    serviceAnalogAcquisition();
    serviceWheelEvents();
    emitDeferredStats();
    serviceWriter(false);
  }
}
