/*
 * Host-side unit tests for stim_core (the safety-critical firmware core).
 * Build & run (no ESP-IDF needed):
 *
 *   cd firmware/esp32-gamma-stim
 *   gcc -Wall -Wextra -Werror -O2 -I main tests/test_stim_core.c main/stim_core.c -o /tmp/test_stim && /tmp/test_stim
 *
 * Exit code 0 = all pass. These tests cover the same invariants the
 * ruview-gamma Rust crate enforces host-side (defense in depth): envelope is
 * never exceeded, e-stop latches, fail-closed parsing, exact timing math for
 * the ±0.1 Hz HIL target.
 */
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "stim_core.h"

static int tests_run = 0;
#define RUN(t) do { t(); tests_run++; printf("ok - %s\n", #t); } while (0)

static stim_ctx_t fresh(void)
{
    stim_ctx_t c;
    stim_init(&c, stim_envelope_conservative());
    return c;
}

static stim_params_t prior(void)
{
    stim_params_t p = {
        .freq_mhz = 40000, .brightness_pct = 30, .volume_pct = 28, .duration_s = 600,
    };
    return p;
}

/* ---- envelope ------------------------------------------------------------ */

static void test_prior_is_inside_envelope(void)
{
    stim_ctx_t c = fresh();
    stim_params_t p = prior();
    assert(stim_validate(&c, &p) == STIM_OK);
}

static void test_frequency_outside_band_refused(void)
{
    stim_ctx_t c = fresh();
    stim_params_t p = prior();
    p.freq_mhz = 35999; /* 35.999 Hz */
    assert(stim_validate(&c, &p) == STIM_ERR_FREQ_RANGE);
    p.freq_mhz = 44001;
    assert(stim_validate(&c, &p) == STIM_ERR_FREQ_RANGE);
    p.freq_mhz = 0;
    assert(stim_validate(&c, &p) == STIM_ERR_FREQ_RANGE);
    /* band edges are inclusive */
    p.freq_mhz = 36000;
    assert(stim_validate(&c, &p) == STIM_OK);
    p.freq_mhz = 44000;
    assert(stim_validate(&c, &p) == STIM_OK);
}

static void test_intensity_caps_refused(void)
{
    stim_ctx_t c = fresh();
    stim_params_t p = prior();
    p.brightness_pct = 41;
    assert(stim_validate(&c, &p) == STIM_ERR_BRIGHTNESS_CAP);
    p = prior();
    p.volume_pct = 41;
    assert(stim_validate(&c, &p) == STIM_ERR_VOLUME_CAP);
    p = prior();
    p.brightness_pct = 40; /* cap value itself is allowed */
    p.volume_pct = 40;
    assert(stim_validate(&c, &p) == STIM_OK);
}

static void test_duration_caps_refused(void)
{
    stim_ctx_t c = fresh();
    stim_params_t p = prior();
    p.duration_s = 0;
    assert(stim_validate(&c, &p) == STIM_ERR_ZERO_DURATION);
    p.duration_s = 901;
    assert(stim_validate(&c, &p) == STIM_ERR_DURATION_CAP);
    p.duration_s = 900;
    assert(stim_validate(&c, &p) == STIM_OK);
}

/* ---- state machine --------------------------------------------------------- */

static void test_start_refused_while_running(void)
{
    stim_ctx_t c = fresh();
    stim_params_t p = prior();
    assert(stim_start(&c, &p) == STIM_OK);
    assert(c.state == STIM_RUNNING);
    assert(stim_start(&c, &p) == STIM_ERR_BUSY);
}

static void test_out_of_envelope_start_keeps_outputs_off(void)
{
    stim_ctx_t c = fresh();
    stim_params_t p = prior();
    p.brightness_pct = 90;
    assert(stim_start(&c, &p) == STIM_ERR_BRIGHTNESS_CAP);
    assert(c.state == STIM_IDLE);     /* fail closed */
    assert(!c.envelope_on);
    assert(c.session_seq == 0);       /* no session consumed */
}

static void test_estop_latches_from_any_state(void)
{
    stim_ctx_t c = fresh();
    stim_params_t p = prior();
    assert(stim_start(&c, &p) == STIM_OK);
    stim_estop(&c, STIM_STOP_BUTTON);
    assert(c.state == STIM_LOCKED);
    assert(!c.envelope_on);
    /* START must be refused while latched — a session can never silently
     * resume after an e-stop (mirrors the Rust SafetyMonitor latch). */
    assert(stim_start(&c, &p) == STIM_ERR_LOCKED);
    /* Host STOP does not clear the latch either. */
    stim_stop_host(&c);
    assert(c.state == STIM_LOCKED);
    /* Only the explicit operator UNLOCK clears it. */
    assert(stim_unlock(&c) == STIM_OK);
    assert(c.state == STIM_IDLE);
    assert(stim_start(&c, &p) == STIM_OK);
}

static void test_session_completes_after_duration(void)
{
    stim_ctx_t c = fresh();
    stim_params_t p = prior();
    p.freq_mhz = 40000;
    p.duration_s = 1; /* 1 s @ 40 Hz = 80 half-periods */
    assert(stim_start(&c, &p) == STIM_OK);
    uint32_t total = stim_session_half_periods(p.freq_mhz, p.duration_s);
    assert(total == 80);
    for (uint32_t i = 0; i < total - 1; i++) {
        assert(stim_tick(&c));
    }
    assert(!stim_tick(&c)); /* final tick ends the session */
    assert(c.state == STIM_IDLE);
    assert(c.last_stop == STIM_STOP_COMPLETED);
    assert(!c.envelope_on);
}

static void test_tick_alternates_envelope(void)
{
    stim_ctx_t c = fresh();
    stim_params_t p = prior();
    assert(stim_start(&c, &p) == STIM_OK);
    assert(!c.envelope_on);
    stim_tick(&c);
    assert(c.envelope_on);
    stim_tick(&c);
    assert(!c.envelope_on);
}

/* ---- timing math (the ±0.1 Hz HIL target is integer-exact) ----------------- */

static void test_half_period_math_is_exact(void)
{
    assert(stim_half_period_us(40000) == 12500); /* 40.0 Hz */
    assert(stim_half_period_us(36000) == 13888); /* 36.0 Hz, floor of 13888.9 */
    assert(stim_half_period_us(44000) == 11363); /* 44.0 Hz, floor of 11363.6 */
    assert(stim_half_period_us(38500) == 12987); /* 38.5 Hz */
    /* Worst-case truncation at 44 Hz: commanded period = 2*11363us = 22726us
     * -> 44.0028 Hz, an error of 2.8 mHz — 35x inside the ±100 mHz target. */
}

static void test_session_half_periods_math(void)
{
    assert(stim_session_half_periods(40000, 600) == 48000); /* 10 min @ 40 Hz */
    assert(stim_session_half_periods(44000, 900) == 79200);
    assert(stim_session_half_periods(36000, 1) == 72);
}

/* ---- protocol parsing -------------------------------------------------------- */

static void test_parse_start(void)
{
    stim_cmd_t cmd;
    assert(stim_parse_line("START 40000 30 28 600", &cmd) == STIM_OK);
    assert(cmd.kind == STIM_CMD_START);
    assert(cmd.params.freq_mhz == 40000);
    assert(cmd.params.brightness_pct == 30);
    assert(cmd.params.volume_pct == 28);
    assert(cmd.params.duration_s == 600);
}

static void test_parse_simple_commands(void)
{
    stim_cmd_t cmd;
    assert(stim_parse_line("STOP", &cmd) == STIM_OK && cmd.kind == STIM_CMD_STOP);
    assert(stim_parse_line("STATUS", &cmd) == STIM_OK && cmd.kind == STIM_CMD_STATUS);
    assert(stim_parse_line("UNLOCK", &cmd) == STIM_OK && cmd.kind == STIM_CMD_UNLOCK);
    assert(stim_parse_line("VERSION", &cmd) == STIM_OK && cmd.kind == STIM_CMD_VERSION);
    assert(stim_parse_line("  STOP", &cmd) == STIM_OK); /* leading spaces ok */
}

static void test_parse_rejects_malformed(void)
{
    stim_cmd_t cmd;
    assert(stim_parse_line("START", &cmd) == STIM_ERR_PARSE);
    assert(stim_parse_line("START 40000 30 28", &cmd) == STIM_ERR_PARSE);
    assert(stim_parse_line("START 40000 30 28 600 junk", &cmd) == STIM_ERR_PARSE);
    assert(stim_parse_line("START 40000 999 28 600", &cmd) == STIM_ERR_PARSE);
    assert(stim_parse_line("START -1 30 28 600", &cmd) == STIM_ERR_PARSE);
    assert(stim_parse_line("START 99999999999 30 28 600", &cmd) == STIM_ERR_PARSE);
    assert(stim_parse_line("FLASHBANG", &cmd) == STIM_ERR_UNKNOWN_CMD);
    assert(stim_parse_line("STOPX", &cmd) == STIM_ERR_UNKNOWN_CMD);
    assert(stim_parse_line("", &cmd) == STIM_ERR_UNKNOWN_CMD);
}

static void test_parsed_hostile_start_is_still_refused_by_envelope(void)
{
    /* End-to-end fail-closed: a syntactically valid but unsafe command parses
     * fine and is then refused by validation — never reaches the outputs. */
    stim_ctx_t c = fresh();
    stim_cmd_t cmd;
    assert(stim_parse_line("START 60000 40 40 600", &cmd) == STIM_OK);
    assert(stim_start(&c, &cmd.params) == STIM_ERR_FREQ_RANGE);
    assert(c.state == STIM_IDLE);
}

int main(void)
{
    RUN(test_prior_is_inside_envelope);
    RUN(test_frequency_outside_band_refused);
    RUN(test_intensity_caps_refused);
    RUN(test_duration_caps_refused);
    RUN(test_start_refused_while_running);
    RUN(test_out_of_envelope_start_keeps_outputs_off);
    RUN(test_estop_latches_from_any_state);
    RUN(test_session_completes_after_duration);
    RUN(test_tick_alternates_envelope);
    RUN(test_half_period_math_is_exact);
    RUN(test_session_half_periods_math);
    RUN(test_parse_start);
    RUN(test_parse_simple_commands);
    RUN(test_parse_rejects_malformed);
    RUN(test_parsed_hostile_start_is_still_refused_by_envelope);
    printf("\nall %d stim_core tests passed\n", tests_run);
    return 0;
}
