/*
 * esp32-gamma-stim — ESP-IDF hardware binding for the gamma stimulation core.
 *
 * Architecture (ADR-250 §21 M2 device harness, HIL targets in
 * v2/crates/ruview-gamma/src/hil.rs):
 *
 *   GPTimer (1 MHz, crystal-derived) ─ ISR every half-period
 *        ├── LED:    LEDC channel 0, 19.5 kHz carrier; duty = brightness or 0
 *        ├── Audio:  LEDC channel 1, tone carrier; duty = volume or 0
 *        └── SYNC:   bare GPIO mirroring the envelope (logic-analyzer capture)
 *
 *   E-STOP button ─ GPIO ISR -> outputs off in the ISR itself, state LOCKED.
 *      Stop path is interrupt -> register write: microseconds, vs the 100 ms
 *      HIL budget. The latch is enforced by stim_core (host-tested).
 *
 *   Host protocol: line-based over USB-CDC/UART0 console at 115200
 *      (START/STOP/STATUS/UNLOCK/VERSION — see stim_core.h). Every session
 *      ends with one "SESSION {...}" JSON line for the host to witness-hash.
 *
 * All safety decisions (envelope, latch, session math) are in stim_core.c,
 * which is unit-tested on the host. This file only moves registers.
 */
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "driver/gptimer.h"
#include "driver/ledc.h"
#include "driver/gpio.h"
#include "esp_log.h"

#include "stim_core.h"

static const char *TAG = "gamma-stim";

#define FIRMWARE_VERSION "0.1.0"

/* ---- Pins / peripherals (Kconfig-overridable) ----------------------------- */
#define PIN_LED      CONFIG_GAMMA_STIM_LED_GPIO
#define PIN_AUDIO    CONFIG_GAMMA_STIM_AUDIO_GPIO
#define PIN_SYNC     CONFIG_GAMMA_STIM_SYNC_GPIO
#define PIN_ESTOP    CONFIG_GAMMA_STIM_ESTOP_GPIO

#define LEDC_LED_CH     LEDC_CHANNEL_0
#define LEDC_AUDIO_CH   LEDC_CHANNEL_1
#define LEDC_LED_TIMER  LEDC_TIMER_0
#define LEDC_AUDIO_TIMER LEDC_TIMER_1
/* 13-bit duty at ~19.5 kHz LED carrier: flicker-free dimming far above the
 * envelope band; the 36-44 Hz stimulus is the *envelope*, not the carrier. */
#define LED_CARRIER_HZ   19500
#define LED_DUTY_RES     LEDC_TIMER_12_BIT
#define LED_DUTY_MAX     ((1 << 12) - 1)
/* Audio: square tone carrier gated by the envelope. */
#define AUDIO_TONE_HZ    CONFIG_GAMMA_STIM_AUDIO_TONE_HZ
#define AUDIO_DUTY_RES   LEDC_TIMER_12_BIT
#define AUDIO_DUTY_MAX   ((1 << 12) - 1)

/* ---- Shared state ---------------------------------------------------------- */

static stim_ctx_t s_ctx;                 /* guarded: ISR + main task        */
static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
static gptimer_handle_t s_timer = NULL;
static QueueHandle_t s_evt_queue = NULL; /* session-finished events to task */

typedef enum { EVT_SESSION_DONE = 1, EVT_ESTOP = 2 } stim_evt_t;

/* Apply outputs for the current envelope phase. ISR-safe (register writes). */
static void IRAM_ATTR apply_outputs(bool on, uint8_t brightness_pct, uint8_t volume_pct)
{
    uint32_t led_duty = on ? ((uint32_t)brightness_pct * LED_DUTY_MAX) / 100U : 0U;
    /* Volume cap is 40% -> max audio duty 20% of full scale: keep the square
     * tone gentle; real loudness control belongs to the analog stage. */
    uint32_t aud_duty = on ? ((uint32_t)volume_pct * (AUDIO_DUTY_MAX / 2U)) / 100U : 0U;
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_LED_CH, led_duty);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_LED_CH);
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_AUDIO_CH, aud_duty);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_AUDIO_CH);
    gpio_set_level(PIN_SYNC, on ? 1 : 0);
}

static void IRAM_ATTR outputs_off(void)
{
    apply_outputs(false, 0, 0);
}

/* GPTimer alarm ISR: one half-period boundary. */
static bool IRAM_ATTR on_half_period(gptimer_handle_t timer,
                                     const gptimer_alarm_event_data_t *edata,
                                     void *user)
{
    (void)timer; (void)edata; (void)user;
    BaseType_t hpw = pdFALSE;
    portENTER_CRITICAL_ISR(&s_mux);
    bool running = stim_tick(&s_ctx);
    if (running) {
        apply_outputs(s_ctx.envelope_on, s_ctx.active.brightness_pct,
                      s_ctx.active.volume_pct);
    } else {
        outputs_off();
        gptimer_stop(timer);
        stim_evt_t e = EVT_SESSION_DONE;
        xQueueSendFromISR(s_evt_queue, &e, &hpw);
    }
    portEXIT_CRITICAL_ISR(&s_mux);
    return hpw == pdTRUE;
}

/* E-stop button ISR: outputs off *here*, then latch + notify. The full stop
 * path is ISR latency + two LEDC register writes — microseconds. */
static void IRAM_ATTR on_estop(void *arg)
{
    (void)arg;
    BaseType_t hpw = pdFALSE;
    portENTER_CRITICAL_ISR(&s_mux);
    outputs_off();
    stim_estop(&s_ctx, STIM_STOP_BUTTON);
    if (s_timer) {
        gptimer_stop(s_timer);
    }
    portEXIT_CRITICAL_ISR(&s_mux);
    stim_evt_t e = EVT_ESTOP;
    xQueueSendFromISR(s_evt_queue, &e, &hpw);
    if (hpw == pdTRUE) {
        portYIELD_FROM_ISR();
    }
}

/* ---- Peripheral setup -------------------------------------------------------- */

static void setup_ledc(void)
{
    ledc_timer_config_t led_t = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .timer_num = LEDC_LED_TIMER,
        .duty_resolution = LED_DUTY_RES,
        .freq_hz = LED_CARRIER_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&led_t));
    ledc_channel_config_t led_c = {
        .gpio_num = PIN_LED,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = LEDC_LED_CH,
        .timer_sel = LEDC_LED_TIMER,
        .duty = 0,
        .hpoint = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&led_c));

    ledc_timer_config_t aud_t = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .timer_num = LEDC_AUDIO_TIMER,
        .duty_resolution = AUDIO_DUTY_RES,
        .freq_hz = AUDIO_TONE_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&aud_t));
    ledc_channel_config_t aud_c = {
        .gpio_num = PIN_AUDIO,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel = LEDC_AUDIO_CH,
        .timer_sel = LEDC_AUDIO_TIMER,
        .duty = 0,
        .hpoint = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&aud_c));
}

static void setup_gpio(void)
{
    gpio_config_t sync = {
        .pin_bit_mask = 1ULL << PIN_SYNC,
        .mode = GPIO_MODE_OUTPUT,
    };
    ESP_ERROR_CHECK(gpio_config(&sync));
    gpio_set_level(PIN_SYNC, 0);

    gpio_config_t estop = {
        .pin_bit_mask = 1ULL << PIN_ESTOP,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,    /* button to GND, active low */
        .intr_type = GPIO_INTR_NEGEDGE,
    };
    ESP_ERROR_CHECK(gpio_config(&estop));
    ESP_ERROR_CHECK(gpio_install_isr_service(0));
    ESP_ERROR_CHECK(gpio_isr_handler_add(PIN_ESTOP, on_estop, NULL));
}

static void setup_timer(void)
{
    gptimer_config_t cfg = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT,
        .direction = GPTIMER_COUNT_UP,
        .resolution_hz = 1000000, /* 1 us ticks, crystal-derived */
    };
    ESP_ERROR_CHECK(gptimer_new_timer(&cfg, &s_timer));
    gptimer_event_callbacks_t cbs = { .on_alarm = on_half_period };
    ESP_ERROR_CHECK(gptimer_register_event_callbacks(s_timer, &cbs, NULL));
    ESP_ERROR_CHECK(gptimer_enable(s_timer));
}

/* ---- Session lifecycle ---------------------------------------------------------- */

static void print_session_record(void)
{
    /* One canonical JSON line per finished session; the host pairs it with the
     * RuFlo session builder to compute the witness hash (HIL: 100% hash
     * reproducibility). Quantized integers only — no float formatting drift. */
    portENTER_CRITICAL(&s_mux);
    stim_ctx_t snap = s_ctx;
    portEXIT_CRITICAL(&s_mux);
    printf("SESSION {\"seq\":%u,\"freq_mhz\":%u,\"brightness_pct\":%u,"
           "\"volume_pct\":%u,\"duration_s\":%u,\"half_periods\":%u,"
           "\"stop\":\"%s\",\"fw\":\"%s\"}\n",
           (unsigned)snap.session_seq, (unsigned)snap.active.freq_mhz,
           (unsigned)snap.active.brightness_pct, (unsigned)snap.active.volume_pct,
           (unsigned)snap.active.duration_s, (unsigned)snap.elapsed_half_periods,
           stim_stop_str(snap.last_stop), FIRMWARE_VERSION);
}

static void handle_start(const stim_params_t *p)
{
    portENTER_CRITICAL(&s_mux);
    stim_rc_t rc = stim_start(&s_ctx, p);
    portEXIT_CRITICAL(&s_mux);
    if (rc != STIM_OK) {
        printf("ERR %s\n", stim_rc_str(rc));
        return;
    }
    uint32_t half_us = stim_half_period_us(p->freq_mhz);
    gptimer_alarm_config_t alarm = {
        .alarm_count = half_us,
        .reload_count = 0,
        .flags.auto_reload_on_alarm = true,
    };
    ESP_ERROR_CHECK(gptimer_set_raw_count(s_timer, 0));
    ESP_ERROR_CHECK(gptimer_set_alarm_action(s_timer, &alarm));
    ESP_ERROR_CHECK(gptimer_start(s_timer));
    printf("OK start seq=%u half_period_us=%u\n",
           (unsigned)s_ctx.session_seq, (unsigned)half_us);
}

static void handle_line(const char *line)
{
    stim_cmd_t cmd;
    stim_rc_t rc = stim_parse_line(line, &cmd);
    if (rc != STIM_OK) {
        printf("ERR %s\n", stim_rc_str(rc));
        return;
    }
    switch (cmd.kind) {
    case STIM_CMD_START:
        handle_start(&cmd.params);
        break;
    case STIM_CMD_STOP:
        portENTER_CRITICAL(&s_mux);
        outputs_off();
        gptimer_stop(s_timer);
        stim_stop_host(&s_ctx);
        portEXIT_CRITICAL(&s_mux);
        print_session_record();
        printf("OK stop\n");
        break;
    case STIM_CMD_STATUS: {
        portENTER_CRITICAL(&s_mux);
        stim_ctx_t snap = s_ctx;
        portEXIT_CRITICAL(&s_mux);
        const char *st = snap.state == STIM_RUNNING ? "running"
                       : snap.state == STIM_LOCKED  ? "locked"
                                                    : "idle";
        printf("OK status state=%s seq=%u last_stop=%s\n", st,
               (unsigned)snap.session_seq, stim_stop_str(snap.last_stop));
        break;
    }
    case STIM_CMD_UNLOCK:
        portENTER_CRITICAL(&s_mux);
        stim_unlock(&s_ctx);
        portEXIT_CRITICAL(&s_mux);
        printf("OK unlock\n");
        break;
    case STIM_CMD_VERSION:
        printf("OK version fw=%s envelope=36000-44000mHz b<=%u%% v<=%u%% d<=%us\n",
               FIRMWARE_VERSION,
               (unsigned)s_ctx.envelope.max_brightness_pct,
               (unsigned)s_ctx.envelope.max_volume_pct,
               (unsigned)s_ctx.envelope.max_duration_s);
        break;
    default:
        printf("ERR %s\n", stim_rc_str(STIM_ERR_UNKNOWN_CMD));
    }
}

/* Console reader: line-buffered stdin (USB-CDC / UART0). */
static void console_task(void *arg)
{
    (void)arg;
    char buf[96];
    size_t n = 0;
    for (;;) {
        int ch = fgetc(stdin);
        if (ch == EOF) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        if (ch == '\r') {
            continue;
        }
        if (ch == '\n') {
            buf[n] = '\0';
            if (n > 0) {
                handle_line(buf);
            }
            n = 0;
            continue;
        }
        if (n + 1 < sizeof(buf)) {
            buf[n++] = (char)ch;
        } else {
            n = 0; /* overlong line: drop, fail closed */
            printf("ERR %s\n", stim_rc_str(STIM_ERR_PARSE));
        }
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "gamma-stim v%s (ADR-250 M2 device harness)", FIRMWARE_VERSION);
    s_evt_queue = xQueueCreate(8, sizeof(stim_evt_t));
    stim_init(&s_ctx, stim_envelope_conservative());
    setup_ledc();
    setup_gpio();
    setup_timer();
    outputs_off();
    xTaskCreate(console_task, "console", 4096, NULL, 5, NULL);
    ESP_LOGI(TAG, "ready: envelope 36.0-44.0 Hz, brightness<=%u%%, volume<=%u%%",
             (unsigned)s_ctx.envelope.max_brightness_pct,
             (unsigned)s_ctx.envelope.max_volume_pct);

    stim_evt_t evt;
    for (;;) {
        if (xQueueReceive(s_evt_queue, &evt, portMAX_DELAY) == pdTRUE) {
            if (evt == EVT_SESSION_DONE) {
                print_session_record();
            } else if (evt == EVT_ESTOP) {
                print_session_record();
                printf("EVT estop_latched\n");
            }
        }
    }
}
