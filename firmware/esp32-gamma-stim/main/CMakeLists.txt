idf_component_register(
    SRCS "main.c" "stim_core.c"
    INCLUDE_DIRS "."
    REQUIRES driver esp_timer
)
