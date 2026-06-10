# esp32-gamma-stim — ESP32 gamma stimulation actuator (ADR-250 §21 M2)

The **device harness** for `ruview-gamma`: an ESP32 that drives a light + sound
flicker at a commanded frequency, gated by a hardware emergency stop, with a
compiled-in safety envelope that mirrors `SafetyEnvelope::conservative()` in the
Rust crate. This is the actuator the `hil::verify_hil` contract grades.

> **Not a medical device.** Research/engineering harness. The host
> (`ruview-gamma`) decides *what* to play and never claims a therapeutic effect;
> this firmware only plays it safely and reports exactly what it did.

## Design: safety core vs hardware binding

| File | Role | Tested |
|------|------|--------|
| `main/stim_core.{h,c}` | Pure C safety core: envelope validation, START/STOP/e-stop **latched** state machine, exact integer timing math, line protocol parser. No ESP-IDF deps. | `tests/test_stim_core.c` on the host (gcc), 15 tests |
| `main/main.c` | ESP-IDF binding: GPTimer ISR, LEDC PWM (LED + audio), sync GPIO, e-stop ISR, USB-CDC console. Only moves registers. | on hardware (HIL) |

Every safety decision lives in the host-tested core — **defense in depth**: the
Rust host gates the stimulus *and* the device gates it again independently, so a
buggy or compromised host still cannot command an out-of-envelope output.

## Run the safety-core tests (no hardware, no ESP-IDF)

```bash
cd firmware/esp32-gamma-stim
gcc -Wall -Wextra -Werror -O2 -I main tests/test_stim_core.c main/stim_core.c -o /tmp/test_stim && /tmp/test_stim
# -> all 15 stim_core tests passed
```

## Build & flash (ESP-IDF v5.2+)

```bash
idf.py set-target esp32s3        # or esp32c6
idf.py menuconfig                # Gamma Stimulation -> pins, tone freq
idf.py build flash monitor
```

Default pins (Kconfig-overridable): LED GPIO 4, audio GPIO 5, sync-out GPIO 6,
e-stop button GPIO 7 (to GND, active-low).

## Host protocol (line-based, 115200, USB-CDC/UART0)

```
START <freq_mhz> <brightness_pct> <volume_pct> <duration_s>
STOP
STATUS
UNLOCK            # clear a latched e-stop
VERSION
```

Frequency is **millihertz** (40.0 Hz = `40000`) so the ±0.1 Hz HIL target is
exact integer math (±100 mHz). Example — 40.0 Hz, 30% brightness, 28% volume,
10 min:

```
> START 40000 30 28 600
OK start seq=1 half_period_us=12500
... (session runs) ...
SESSION {"seq":1,"freq_mhz":40000,"brightness_pct":30,"volume_pct":28,"duration_s":600,"half_periods":48000,"stop":"completed","fw":"0.1.0"}
```

The `SESSION {...}` line is canonical (quantized integers, fixed field order) so
the host pairs it with the RuFlo session builder to reproduce the witness hash
(HIL: 100% hash reproducibility).

## How it maps to the HIL targets (`v2/crates/ruview-gamma/src/hil.rs`)

| HIL target | How this firmware meets it |
|------------|----------------------------|
| LED frequency ±0.1 Hz | GPTimer at 1 MHz crystal-derived ticks; half-period from exact integer division; worst-case truncation at 44 Hz is ~3 mHz (35× inside budget) |
| A/V sync < 5 ms | LED and audio duty written in the **same ISR**; skew is a few register writes |
| Stop → actuator off < 100 ms | e-stop GPIO ISR turns outputs off **in the ISR** before latching — microseconds |
| Session-hash reproducibility 100% | canonical integer `SESSION {...}` record, no float formatting |
| EEG lift ≥ 20% vs fixed 40 Hz | provided by the host's adaptive optimizer choosing the frequency this firmware plays |

## Hardware notes

- **Drive the LED through a MOSFET/constant-current driver**, not the GPIO
  directly. Keep brightness within eye-safe flicker limits — the firmware caps
  duty at the envelope's 40%, but the optical design owns absolute luminance.
- **Photosensitivity/epilepsy is a hard exclusion** at the host
  (`ExclusionScreen`); the device is the last line, not the only line.
- The e-stop button is mandatory for any human-facing bench run.
