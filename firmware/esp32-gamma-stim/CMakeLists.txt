# ESP32 Gamma Stimulation Firmware (ADR-250 §21 M2 device harness)
# Requires ESP-IDF v5.2+
cmake_minimum_required(VERSION 3.16)

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/version.txt" PROJECT_VER LIMIT_COUNT 1)
string(STRIP "${PROJECT_VER}" PROJECT_VER)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(esp32-gamma-stim VERSION ${PROJECT_VER})
