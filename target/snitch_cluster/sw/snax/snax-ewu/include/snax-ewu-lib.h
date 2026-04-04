// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "snrt.h"

#include <stdbool.h>
#include <stdint.h>

// Accelerator Register Addresses
#define EWU_RW_MODE 978
#define EWU_RW_DATALEN 979
#define EWU_RW_START 980
#define EWU_RO_BUSY 981
#define EWU_RO_PERF_COUNT 982

#define EWU_MODE_ADD 0
#define EWU_MODE_MUL 1

void configure_streamer_a(uint32_t base_ptr_low, uint32_t base_ptr_high,
                          uint32_t spatial_stride, uint32_t temporal_bound,
                          uint32_t temporal_stride);

void configure_streamer_b(uint32_t base_ptr_low, uint32_t base_ptr_high,
                          uint32_t spatial_stride, uint32_t temporal_bound,
                          uint32_t temporal_stride);

void configure_streamer_o(uint32_t base_ptr_low, uint32_t base_ptr_high,
                          uint32_t spatial_stride, uint32_t temporal_bound,
                          uint32_t temporal_stride);

void start_streamer(void);

uint32_t read_busy_streamer(void);

void configure_ewu(uint32_t mode, uint32_t data_len);

void start_ewu(void);

uint32_t read_busy_ewu(void);
