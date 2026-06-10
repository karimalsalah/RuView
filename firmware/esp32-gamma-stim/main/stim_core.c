/*
 * stim_core.c — pure, host-testable core of the gamma stimulation firmware.
 * See stim_core.h for the contract. No ESP-IDF includes in this file.
 */
#include "stim_core.h"

#include <string.h>
#include <stdlib.h>
#include <ctype.h>

/* ---- Envelope ------------------------------------------------------------ */

stim_envelope_t stim_envelope_conservative(void)
{
    stim_envelope_t e = {
        .min_freq_mhz       = 36000,  /* 36.0 Hz */
        .max_freq_mhz       = 44000,  /* 44.0 Hz */
        .max_brightness_pct = 40,     /* SafetyEnvelope::conservative 0.40 */
        .max_volume_pct     = 40,
        .max_duration_s     = 900,    /* 15 min */
    };
    return e;
}

void stim_init(stim_ctx_t *ctx, stim_envelope_t envelope)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->envelope = envelope;
    ctx->state = STIM_IDLE;
    ctx->last_stop = STIM_STOP_NONE;
}

/* ---- Validation ----------------------------------------------------------- */

stim_rc_t stim_validate(const stim_ctx_t *ctx, const stim_params_t *p)
{
    const stim_envelope_t *e = &ctx->envelope;
    if (p->freq_mhz < e->min_freq_mhz || p->freq_mhz > e->max_freq_mhz) {
        return STIM_ERR_FREQ_RANGE;
    }
    if (p->brightness_pct > e->max_brightness_pct) {
        return STIM_ERR_BRIGHTNESS_CAP;
    }
    if (p->volume_pct > e->max_volume_pct) {
        return STIM_ERR_VOLUME_CAP;
    }
    if (p->duration_s == 0) {
        return STIM_ERR_ZERO_DURATION;
    }
    if (p->duration_s > e->max_duration_s) {
        return STIM_ERR_DURATION_CAP;
    }
    return STIM_OK;
}

/* ---- Transitions ----------------------------------------------------------- */

stim_rc_t stim_start(stim_ctx_t *ctx, const stim_params_t *p)
{
    if (ctx->state == STIM_LOCKED) {
        return STIM_ERR_LOCKED;
    }
    if (ctx->state == STIM_RUNNING) {
        return STIM_ERR_BUSY;
    }
    stim_rc_t rc = stim_validate(ctx, p);
    if (rc != STIM_OK) {
        return rc; /* fail closed: state unchanged, outputs stay off */
    }
    ctx->active = *p;
    ctx->elapsed_half_periods = 0;
    ctx->envelope_on = false;
    ctx->session_seq += 1;
    ctx->last_stop = STIM_STOP_NONE;
    ctx->state = STIM_RUNNING;
    return STIM_OK;
}

stim_rc_t stim_stop_host(stim_ctx_t *ctx)
{
    if (ctx->state == STIM_RUNNING) {
        ctx->state = STIM_IDLE;
        ctx->envelope_on = false;
        ctx->last_stop = STIM_STOP_HOST;
    }
    /* STOP while idle/locked is a harmless no-op (idempotent). */
    return STIM_OK;
}

void stim_estop(stim_ctx_t *ctx, stim_stop_reason_t why)
{
    ctx->state = STIM_LOCKED;       /* latched, from ANY state */
    ctx->envelope_on = false;
    ctx->last_stop = why;
}

stim_rc_t stim_unlock(stim_ctx_t *ctx)
{
    if (ctx->state == STIM_LOCKED) {
        ctx->state = STIM_IDLE;
    }
    return STIM_OK;
}

bool stim_tick(stim_ctx_t *ctx)
{
    if (ctx->state != STIM_RUNNING) {
        ctx->envelope_on = false;
        return false;
    }
    ctx->envelope_on = !ctx->envelope_on;
    ctx->elapsed_half_periods += 1;
    uint32_t total =
        stim_session_half_periods(ctx->active.freq_mhz, ctx->active.duration_s);
    if (ctx->elapsed_half_periods >= total) {
        ctx->state = STIM_IDLE;
        ctx->envelope_on = false;
        ctx->last_stop = STIM_STOP_COMPLETED;
        return false;
    }
    return true;
}

uint32_t stim_half_period_us(uint32_t freq_mhz)
{
    if (freq_mhz == 0) {
        return 0;
    }
    /* half period [us] = 1e6 / (2 * f[Hz]) = 5e8 / f[mHz].
     * 64-bit intermediate; exact division for e.g. 40000 -> 12500 us. */
    return (uint32_t)(500000000ULL / (uint64_t)freq_mhz);
}

uint32_t stim_session_half_periods(uint32_t freq_mhz, uint32_t duration_s)
{
    /* half periods = duration * 2 * f[Hz] = duration * f[mHz] / 500.
     * 64-bit intermediate: 900 s * 44000 = 39.6e6, fine. */
    return (uint32_t)(((uint64_t)duration_s * (uint64_t)freq_mhz) / 500ULL);
}

/* ---- Protocol parsing ------------------------------------------------------ */

/* Parse an unsigned decimal field; returns false on junk/overflow. */
static bool parse_u32(const char **cursor, uint32_t *out)
{
    const char *s = *cursor;
    while (*s == ' ') {
        s++;
    }
    if (!isdigit((unsigned char)*s)) {
        return false;
    }
    uint64_t v = 0;
    while (isdigit((unsigned char)*s)) {
        v = v * 10ULL + (uint64_t)(*s - '0');
        if (v > 0xFFFFFFFFULL) {
            return false;
        }
        s++;
    }
    *out = (uint32_t)v;
    *cursor = s;
    return true;
}

static bool token_is(const char *line, const char *word, const char **rest)
{
    size_t n = strlen(word);
    if (strncmp(line, word, n) != 0) {
        return false;
    }
    if (line[n] != '\0' && line[n] != ' ') {
        return false;
    }
    *rest = line + n;
    return true;
}

stim_rc_t stim_parse_line(const char *line, stim_cmd_t *out)
{
    memset(out, 0, sizeof(*out));
    while (*line == ' ') {
        line++;
    }
    const char *rest = NULL;
    if (token_is(line, "START", &rest)) {
        uint32_t f, b, v, d;
        if (!parse_u32(&rest, &f) || !parse_u32(&rest, &b) ||
            !parse_u32(&rest, &v) || !parse_u32(&rest, &d)) {
            return STIM_ERR_PARSE;
        }
        while (*rest == ' ') {
            rest++;
        }
        if (*rest != '\0') {
            return STIM_ERR_PARSE; /* trailing junk */
        }
        if (b > 255 || v > 255) {
            return STIM_ERR_PARSE; /* fields must fit their types */
        }
        out->kind = STIM_CMD_START;
        out->params.freq_mhz = f;
        out->params.brightness_pct = (uint8_t)b;
        out->params.volume_pct = (uint8_t)v;
        out->params.duration_s = d;
        return STIM_OK;
    }
    if (token_is(line, "STOP", &rest)) {
        out->kind = STIM_CMD_STOP;
        return STIM_OK;
    }
    if (token_is(line, "STATUS", &rest)) {
        out->kind = STIM_CMD_STATUS;
        return STIM_OK;
    }
    if (token_is(line, "UNLOCK", &rest)) {
        out->kind = STIM_CMD_UNLOCK;
        return STIM_OK;
    }
    if (token_is(line, "VERSION", &rest)) {
        out->kind = STIM_CMD_VERSION;
        return STIM_OK;
    }
    return STIM_ERR_UNKNOWN_CMD;
}

const char *stim_rc_str(stim_rc_t rc)
{
    switch (rc) {
    case STIM_OK:                 return "ok";
    case STIM_ERR_FREQ_RANGE:     return "freq_out_of_envelope";
    case STIM_ERR_BRIGHTNESS_CAP: return "brightness_above_cap";
    case STIM_ERR_VOLUME_CAP:     return "volume_above_cap";
    case STIM_ERR_DURATION_CAP:   return "duration_above_cap";
    case STIM_ERR_ZERO_DURATION:  return "zero_duration";
    case STIM_ERR_BUSY:           return "busy";
    case STIM_ERR_LOCKED:         return "estop_locked";
    case STIM_ERR_PARSE:          return "parse_error";
    case STIM_ERR_UNKNOWN_CMD:    return "unknown_command";
    default:                      return "unknown_rc";
    }
}

const char *stim_stop_str(stim_stop_reason_t r)
{
    switch (r) {
    case STIM_STOP_NONE:      return "none";
    case STIM_STOP_COMPLETED: return "completed";
    case STIM_STOP_HOST:      return "host_stop";
    case STIM_STOP_BUTTON:    return "estop_button";
    case STIM_STOP_FAULT:     return "fault";
    default:                  return "unknown";
    }
}
