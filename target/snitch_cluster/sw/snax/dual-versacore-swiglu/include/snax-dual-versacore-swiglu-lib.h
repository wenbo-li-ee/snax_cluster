// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdbool.h>

#include "snrt.h"
#include "stdint.h"
#include "streamer_csr_addr_map.h"

#pragma once

// Writer-only busy CSRs (streamer RO CSRs for block pipeline)
#define WRITER_BUSY_CSR STREAMER_WRITER_BUSY_CSR
#define WRITER1_BUSY_CSR STREAMER_WRITER1_BUSY_CSR

// Accelerator CSR addresses (after streamer CSRs, including both writer_busy)
#define DUAL_VC_CSR_ADDR_BASE (STREAMER_WRITER1_BUSY_CSR + 1)
#define DUAL_VC_OVERWRITE_ACCUM (DUAL_VC_CSR_ADDR_BASE)      // [0]
#define DUAL_VC_ACCUM_BOUND (DUAL_VC_OVERWRITE_ACCUM + 1)    // [1]
#define DUAL_VC_OUTPUT_BOUND (DUAL_VC_ACCUM_BOUND + 1)       // [2]
#define DUAL_VC_SUBTRACTIONS (DUAL_VC_OUTPUT_BOUND + 1)       // [3]
#define DUAL_VC_ARRAY_SHAPE_CFG (DUAL_VC_SUBTRACTIONS + 1)    // [4]
#define DUAL_VC_DATA_TYPE_CFG (DUAL_VC_ARRAY_SHAPE_CFG + 1)   // [5]
#define DUAL_VC_MODE (DUAL_VC_DATA_TYPE_CFG + 1)              // [6]

// Rescale0 parameters (VC0 path)
#define DUAL_VC_RESCALE0_INPUT_ZP (DUAL_VC_MODE + 1)          // [7]
#define DUAL_VC_RESCALE0_MULTIPLIER (DUAL_VC_RESCALE0_INPUT_ZP + 1) // [8]
#define DUAL_VC_RESCALE0_OUTPUT_ZP (DUAL_VC_RESCALE0_MULTIPLIER + 1) // [9]
#define DUAL_VC_RESCALE0_SHIFT (DUAL_VC_RESCALE0_OUTPUT_ZP + 1) // [10]

// Rescale1 parameters (VC1 path)
#define DUAL_VC_RESCALE1_INPUT_ZP (DUAL_VC_RESCALE0_SHIFT + 1)  // [11]
#define DUAL_VC_RESCALE1_MULTIPLIER (DUAL_VC_RESCALE1_INPUT_ZP + 1) // [12]
#define DUAL_VC_RESCALE1_OUTPUT_ZP (DUAL_VC_RESCALE1_MULTIPLIER + 1) // [13]
#define DUAL_VC_RESCALE1_SHIFT (DUAL_VC_RESCALE1_OUTPUT_ZP + 1) // [14]

// Rescale_mul parameters (after elem_mul, mode 0 only)
#define DUAL_VC_RESCALE_MUL_INPUT_ZP (DUAL_VC_RESCALE1_SHIFT + 1) // [15]
#define DUAL_VC_RESCALE_MUL_MULTIPLIER (DUAL_VC_RESCALE_MUL_INPUT_ZP + 1) // [16]
#define DUAL_VC_RESCALE_MUL_OUTPUT_ZP (DUAL_VC_RESCALE_MUL_MULTIPLIER + 1) // [17]
#define DUAL_VC_RESCALE_MUL_SHIFT (DUAL_VC_RESCALE_MUL_OUTPUT_ZP + 1) // [18]

// Read-only CSRs follow all accelerator RW CSRs in the generated CSR manager.
#define DUAL_VC_NUM_RW_CSR 23
// ReqRspManager emits csr_reg_set_valid when the last RW CSR is written with bit0=1.
#define DUAL_VC_START (DUAL_VC_CSR_ADDR_BASE + DUAL_VC_NUM_RW_CSR - 1)
#define DUAL_VC_BUSY (DUAL_VC_CSR_ADDR_BASE + DUAL_VC_NUM_RW_CSR)
#define DUAL_VC_PERFORMANCE_COUNTER (DUAL_VC_BUSY + 1)

// Pack two subtraction values to one CSR
int32_t gen_dual_vc_subtraction_config(int8_t subtraction_a, int8_t subtraction_b);

// Configure all streamer CSRs (3 readers + 2 writers)
void set_dual_versacore_streamer_csr(
    int32_t delta_local_a, const int32_t* Aslstride, const int32_t* Atlbound,
    const int32_t* Atlstride, int32_t set_addr_remap_index_A,
    const int32_t* channel_en_A,

    int32_t delta_local_b0, const int32_t* B0slstride, const int32_t* B0tlbound,
    const int32_t* B0tlstride, int32_t set_addr_remap_index_B0,
    const int32_t* channel_en_B0,

    int32_t delta_local_b1, const int32_t* B1slstride, const int32_t* B1tlbound,
    const int32_t* B1tlstride, int32_t set_addr_remap_index_B1,
    const int32_t* channel_en_B1,

    int32_t delta_local_d0, const int32_t* D0slstride, const int32_t* D0tlbound,
    const int32_t* D0tlstride, int32_t set_addr_remap_index_D0,
    const int32_t* channel_en_D0,

    int32_t delta_local_d1, const int32_t* D1slstride, const int32_t* D1tlbound,
    const int32_t* D1tlstride, int32_t set_addr_remap_index_D1,
    const int32_t* channel_en_D1);

// Configure 3 readers + Writer0, and explicitly idle Writer1.
// Use for Mode0 single-writer flows.
void set_dual_versacore_streamer_csr_d0_only(
    int32_t delta_local_a, const int32_t* Aslstride, const int32_t* Atlbound,
    const int32_t* Atlstride, int32_t set_addr_remap_index_A,
    const int32_t* channel_en_A,

    int32_t delta_local_b0, const int32_t* B0slstride, const int32_t* B0tlbound,
    const int32_t* B0tlstride, int32_t set_addr_remap_index_B0,
    const int32_t* channel_en_B0,

    int32_t delta_local_b1, const int32_t* B1slstride, const int32_t* B1tlbound,
    const int32_t* B1tlstride, int32_t set_addr_remap_index_B1,
    const int32_t* channel_en_B1,

    int32_t delta_local_d0, const int32_t* D0slstride, const int32_t* D0tlbound,
    const int32_t* D0tlstride, int32_t set_addr_remap_index_D0,
    const int32_t* channel_en_D0);

// Start streamer
inline void set_dual_versacore_streamer_start() {
    csrw_ss(STREAMER_START_CSR, 1);
}

// Configure accelerator CSRs (shared by both VersaCores)
void set_dual_versacore_csr(uint32_t take_in_new_c,
                            uint32_t a_b_input_times_one_output,
                            uint32_t output_times, uint32_t subtractions,
                            uint32_t array_shape, uint32_t data_type);

// Set mode (0=SwiGLU, 1=GEMM)
void set_dual_versacore_mode(uint32_t mode);

// Set rescale parameters
void set_dual_versacore_rescale0(int32_t input_zp, uint32_t multiplier,
                                 int32_t output_zp, uint32_t shift);
void set_dual_versacore_rescale1(int32_t input_zp, uint32_t multiplier,
                                 int32_t output_zp, uint32_t shift);
void set_dual_versacore_rescale_mul(int32_t input_zp, uint32_t multiplier,
                                    int32_t output_zp, uint32_t shift);

// Start accelerator
inline void set_dual_versacore_start() { csrw_ss(DUAL_VC_START, 1); }

// Poll until accelerator (VersaCore) finishes (does NOT wait for streamer)
void wait_dual_versacore();

// Poll until streamer finishes
void wait_dual_versacore_streamer();

// Poll until both accelerator and streamer finish
void wait_dual_versacore_and_streamer();

// Read performance counter
uint32_t read_dual_versacore_perf_counter();

// Read streamer performance counter
uint32_t read_dual_versacore_streamer_perf_counter();

// Check result (int16)
uint32_t check_dual_versacore_result_i16(const int16_t* output, const int16_t* output_golden,
                                         int32_t num_elements);

// Mode1 writes IEEE FP16 bit patterns (not scaled INT16 values).
uint32_t check_dual_versacore_result_fp16_bits(const uint16_t* output,
                                               const uint16_t* output_golden,
                                               int32_t num_elements);

// Poll until both streamer writers finish.
void wait_dual_versacore_writer();
