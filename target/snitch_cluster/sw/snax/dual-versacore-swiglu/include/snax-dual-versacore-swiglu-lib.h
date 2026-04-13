// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdbool.h>

#include "snrt.h"
#include "stdint.h"
#include "streamer_csr_addr_map.h"

#pragma once

// Accelerator CSR addresses (after streamer CSRs)
#define DUAL_VC_CSR_ADDR_BASE (STREAMER_PERFORMANCE_COUNTER_CSR + 1)
#define DUAL_VC_OVERWRITE_ACCUM (DUAL_VC_CSR_ADDR_BASE)
#define DUAL_VC_ACCUM_BOUND (DUAL_VC_OVERWRITE_ACCUM + 1)
#define DUAL_VC_OUTPUT_BOUND (DUAL_VC_ACCUM_BOUND + 1)
#define DUAL_VC_SUBTRACTIONS (DUAL_VC_OUTPUT_BOUND + 1)
#define DUAL_VC_ARRAY_SHAPE_CFG (DUAL_VC_SUBTRACTIONS + 1)
#define DUAL_VC_DATA_TYPE_CFG (DUAL_VC_ARRAY_SHAPE_CFG + 1)

// Start CSR
#define DUAL_VC_START (DUAL_VC_DATA_TYPE_CFG + 1)

// Read-only CSRs
#define DUAL_VC_BUSY (DUAL_VC_START + 1)
#define DUAL_VC_PERFORMANCE_COUNTER (DUAL_VC_BUSY + 1)

// Pack two subtraction values to one CSR
int32_t gen_dual_vc_subtraction_config(int8_t subtraction_a, int8_t subtraction_b);

// Configure all streamer CSRs (3 readers + 1 writer)
void set_dual_versacore_streamer_csr(
    int32_t delta_local_a, int32_t* Aslstride, int32_t* Atlbound,
    int32_t* Atlstride, int32_t set_addr_remap_index_A,
    int32_t* channel_en_A,

    int32_t delta_local_b0, int32_t* B0slstride, int32_t* B0tlbound,
    int32_t* B0tlstride, int32_t set_addr_remap_index_B0,
    int32_t* channel_en_B0,

    int32_t delta_local_b1, int32_t* B1slstride, int32_t* B1tlbound,
    int32_t* B1tlstride, int32_t set_addr_remap_index_B1,
    int32_t* channel_en_B1,

    int32_t delta_local_d, int32_t* Dslstride, int32_t* Dtlbound,
    int32_t* Dtlstride, int32_t set_addr_remap_index_D,
    int32_t* channel_en_D);

// Start streamer
inline void set_dual_versacore_streamer_start() {
    csrw_ss(STREAMER_START_CSR, 1);
}

// Configure accelerator CSRs (shared by both VersaCores)
void set_dual_versacore_csr(uint32_t take_in_new_c,
                            uint32_t a_b_input_times_one_output,
                            uint32_t output_times, uint32_t subtractions,
                            uint32_t array_shape, uint32_t data_type);

// Start accelerator
inline void set_dual_versacore_start() { csrw_ss(DUAL_VC_START, 1); }

// Poll until accelerator (VersaCore) finishes (does NOT wait for streamer)
void wait_dual_versacore();

// Poll until streamer finishes (call after wait_dual_versacore)
void wait_dual_versacore_streamer();

// Poll until both accelerator and streamer finish
void wait_dual_versacore_and_streamer();

// Read performance counter
uint32_t read_dual_versacore_perf_counter();

// Read streamer performance counter
uint32_t read_dual_versacore_streamer_perf_counter();

// Check result
uint32_t check_dual_versacore_result(int8_t* output, int8_t* output_golden,
                                     int32_t data_length);
