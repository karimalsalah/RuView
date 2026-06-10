/*
 * stim_core.h — pure, host-testable core of the gamma stimulation firmware.
 *
 * Everything safety-critical lives here, with NO ESP-IDF dependencies, so the
 * exact code that ships on the device is unit-tested on the host (gcc) and in
 * CI. main.c is a thin hardware binding (timers, LEDC, GPIO, UART).
 *
 * Mirrors the ruview-gamma crate's SafetyEnvelope::conservative() (ADR-250
 * §5/§12): the firmware enforces the same hard caps *independently*, so even a
 * compromised or buggy host cannot command an out-of-envelope stimulus.
 * Defense in depth: host gate (Rust) AND device gate (this file).
 *
 * Units: frequency in millihertz (exact integer math — the ±0.1 Hz HIL target
 * is ±100 mHz), intensity in percent (0–100), duration in seconds.
 */
#ifndef STIM_CORE_H
#define STIM_CORE_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Hard safety envelope (device-side; never widened at runtime) ------- */

typedef struct {
    uint32_t min_freq_mhz;      /* 36000 = 36.0 Hz  */
    uint32_t max_freq_mhz;      /* 44000 = 44.0 Hz  */
    uint8_t  max_brightness_pct;/* 40 = SafetyEnvelope::conservative cap 0.40 */
    uint8_t  max_volume_pct;    /* 40 */
    uint32_t max_duration_s;    /* 900 = 15 min */
} stim_envelope_t;

/* The compiled-in conservative envelope (ADR-250 §5). Kconfig may narrow it,
 * never widen it (enforced by range limits in Kconfig.projbuild). */
stim_envelope_t stim_envelope_conservative(void);

/* ---- Session state machine ---------------------------------------------- */

typedef enum {
    STIM_IDLE = 0,     /* outputs off, ready for START            */
    STIM_RUNNING,      /* stimulation active                      */
    STIM_LOCKED,       /* emergency-stopped; START refused until UNLOCK */
} stim_state_t;

typedef enum {
    STIM_STOP_NONE = 0,
    STIM_STOP_COMPLETED,    /* duration elapsed (not a safety stop)  */
    STIM_STOP_HOST,         /* host STOP command                     */
    STIM_STOP_BUTTON,       /* hardware e-stop button                */
    STIM_STOP_FAULT,        /* internal fault (watchdog, bad state)  */
} stim_stop_reason_t;

typedef struct {
    uint32_t freq_mhz;       /* commanded envelope frequency          */
    uint8_t  brightness_pct; /* LED intensity during ON half-period   */
    uint8_t  volume_pct;     /* tone intensity during ON half-period  */
    uint32_t duration_s;     /* session length                        */
} stim_params_t;

typedef struct {
    stim_envelope_t   envelope;
    stim_state_t      state;
    stim_params_t     active;          /* valid when state == RUNNING   */
    stim_stop_reason_t last_stop;
    uint32_t          session_seq;     /* increments on each START      */
    uint32_t          elapsed_half_periods; /* advanced by the timer ISR */
    bool              envelope_on;     /* current half-period phase     */
} stim_ctx_t;

/* Initialize a context with the given envelope, in IDLE. */
void stim_init(stim_ctx_t *ctx, stim_envelope_t envelope);

/* ---- Validation (fail closed) ------------------------------------------- */

typedef enum {
    STIM_OK = 0,
    STIM_ERR_FREQ_RANGE,     /* outside [min,max] mHz                  */
    STIM_ERR_BRIGHTNESS_CAP,
    STIM_ERR_VOLUME_CAP,
    STIM_ERR_DURATION_CAP,
    STIM_ERR_ZERO_DURATION,
    STIM_ERR_BUSY,           /* START while RUNNING                    */
    STIM_ERR_LOCKED,         /* START while LOCKED (e-stop latched)    */
    STIM_ERR_PARSE,          /* malformed command line                 */
    STIM_ERR_UNKNOWN_CMD,
} stim_rc_t;

/* Validate params against the context envelope. Pure; no state change. */
stim_rc_t stim_validate(const stim_ctx_t *ctx, const stim_params_t *p);

/* ---- Transitions (the only mutators) ------------------------------------ */

/* START: validate + transition IDLE->RUNNING. Fails closed on any violation,
 * on BUSY, and on LOCKED. */
stim_rc_t stim_start(stim_ctx_t *ctx, const stim_params_t *p);

/* STOP from the host: RUNNING->IDLE (graceful; not latched). */
stim_rc_t stim_stop_host(stim_ctx_t *ctx);

/* Emergency stop (button ISR or fault): any state -> LOCKED. Latched —
 * further STARTs are refused until stim_unlock(). Mirrors the Rust
 * SafetyMonitor latch (a session must never silently resume). */
void stim_estop(stim_ctx_t *ctx, stim_stop_reason_t why);

/* Operator unlock after an e-stop: LOCKED -> IDLE. */
stim_rc_t stim_unlock(stim_ctx_t *ctx);

/* Timer ISR tick: advance one half-period. Returns true while RUNNING; when
 * the session's duration is reached it transitions to IDLE (COMPLETED) and
 * returns false. Pure integer math, ISR-safe. */
bool stim_tick(stim_ctx_t *ctx);

/* Half-period length in microseconds for a commanded frequency:
 * 500'000'000'000 / freq_mhz / 1000 — exact for the supported range.
 * (40.0 Hz = 40000 mHz -> 12'500 us.) */
uint32_t stim_half_period_us(uint32_t freq_mhz);

/* Total half-periods in a session of duration_s at freq_mhz (rounded down). */
uint32_t stim_session_half_periods(uint32_t freq_mhz, uint32_t duration_s);

/* ---- Host command protocol (line-based, UART) ---------------------------
 *
 *   START <freq_mhz> <brightness_pct> <volume_pct> <duration_s>
 *   STOP
 *   STATUS
 *   UNLOCK
 *   VERSION
 *
 * stim_parse_line() parses one trimmed line into a command. Pure.
 */

typedef enum {
    STIM_CMD_NONE = 0,
    STIM_CMD_START,
    STIM_CMD_STOP,
    STIM_CMD_STATUS,
    STIM_CMD_UNLOCK,
    STIM_CMD_VERSION,
} stim_cmd_kind_t;

typedef struct {
    stim_cmd_kind_t kind;
    stim_params_t   params; /* valid when kind == STIM_CMD_START */
} stim_cmd_t;

stim_rc_t stim_parse_line(const char *line, stim_cmd_t *out);

/* Human-readable tag for a return code (for "ERR <tag>" replies). */
const char *stim_rc_str(stim_rc_t rc);

/* Human-readable tag for a stop reason (for the session log). */
const char *stim_stop_str(stim_stop_reason_t r);

#ifdef __cplusplus
}
#endif

#endif /* STIM_CORE_H */
