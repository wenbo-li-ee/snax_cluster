// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "snax-dual-versacore-swiglu-lib.h"
#include <stdbool.h>
#include "snax_dual_versacore_stationarity.h"
#include "snrt.h"
#include "stdint.h"
#include "streamer_csr_addr_map.h"

int32_t gen_dual_vc_subtraction_config(int8_t subtraction_a,
                                       int8_t subtraction_b) {
    return ((uint8_t)subtraction_b << 8) | (uint8_t)subtraction_a;
}

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

    int32_t delta_local_d0, int32_t* D0slstride, int32_t* D0tlbound,
    int32_t* D0tlstride, int32_t set_addr_remap_index_D0,
    int32_t* channel_en_D0,

    int32_t delta_local_d1, int32_t* D1slstride, int32_t* D1tlbound,
    int32_t* D1tlstride, int32_t set_addr_remap_index_D1,
    int32_t* channel_en_D1) {

    // ----------------------------------A (Reader 0)----------------------------
    csrw_ss(BASE_PTR_READER_0_LOW, (uint32_t)(delta_local_a + snrt_l1_next()));

    for (int i = 0; i < S_STRIDE_NUM_READER_0; i++) {
        csrw_ss(S_STRIDE_BASE_READER_0 + i, Aslstride[i]);
    }
    for (int i = 0; i < T_BOUND_NUM_READER_0; i++) {
        csrw_ss(T_BOUND_BASE_READER_0 + i, Atlbound[i]);
    }
    for (int i = 0; i < T_STRIDE_NUM_READER_0; i++) {
        csrw_ss(T_STRIDE_BASE_READER_0 + i, Atlstride[i]);
    }
#ifdef ADDR_REMAP_INDEX_READER_0
    csrw_ss(ADDR_REMAP_INDEX_READER_0, set_addr_remap_index_A);
#endif
#ifdef ENABLED_CHANNEL_READER_0
    for (int i = 0; i < ENABLED_CHANNEL_READER_0_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_READER_0 + i, channel_en_A[i]);
    }
#endif

    // ----------------------------------B0 (Reader 1)----------------------------
    csrw_ss(BASE_PTR_READER_1_LOW,
            (uint32_t)(delta_local_b0 + snrt_l1_next()));

    for (int i = 0; i < S_STRIDE_NUM_READER_1; i++) {
        csrw_ss(S_STRIDE_BASE_READER_1 + i, B0slstride[i]);
    }
    for (int i = 0; i < T_BOUND_NUM_READER_1; i++) {
        csrw_ss(T_BOUND_BASE_READER_1 + i, B0tlbound[i]);
    }
    for (int i = 0; i < T_STRIDE_NUM_READER_1; i++) {
        csrw_ss(T_STRIDE_BASE_READER_1 + i, B0tlstride[i]);
    }
#ifdef ADDR_REMAP_INDEX_READER_1
    csrw_ss(ADDR_REMAP_INDEX_READER_1, set_addr_remap_index_B0);
#endif
#ifdef ENABLED_CHANNEL_READER_1
    for (int i = 0; i < ENABLED_CHANNEL_READER_1_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_READER_1 + i, channel_en_B0[i]);
    }
#endif

    // ----------------------------------B1 (Reader 2)----------------------------
    csrw_ss(BASE_PTR_READER_2_LOW,
            (uint32_t)(delta_local_b1 + snrt_l1_next()));

    for (int i = 0; i < S_STRIDE_NUM_READER_2; i++) {
        csrw_ss(S_STRIDE_BASE_READER_2 + i, B1slstride[i]);
    }
    for (int i = 0; i < T_BOUND_NUM_READER_2; i++) {
        csrw_ss(T_BOUND_BASE_READER_2 + i, B1tlbound[i]);
    }
    for (int i = 0; i < T_STRIDE_NUM_READER_2; i++) {
        csrw_ss(T_STRIDE_BASE_READER_2 + i, B1tlstride[i]);
    }
#ifdef ADDR_REMAP_INDEX_READER_2
    csrw_ss(ADDR_REMAP_INDEX_READER_2, set_addr_remap_index_B1);
#endif
#ifdef ENABLED_CHANNEL_READER_2
    for (int i = 0; i < ENABLED_CHANNEL_READER_2_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_READER_2 + i, channel_en_B1[i]);
    }
#endif

    // ----------------------------------D0 (Writer 0)----------------------------
    csrw_ss(BASE_PTR_WRITER_0_LOW,
            (uint32_t)(delta_local_d0 + snrt_l1_next()));

    for (int i = 0; i < S_STRIDE_NUM_WRITER_0; i++) {
        csrw_ss(S_STRIDE_BASE_WRITER_0 + i, D0slstride[i]);
    }
    for (int i = 0; i < T_BOUND_NUM_WRITER_0; i++) {
        csrw_ss(T_BOUND_BASE_WRITER_0 + i, D0tlbound[i]);
    }
    for (int i = 0; i < T_STRIDE_NUM_WRITER_0; i++) {
        csrw_ss(T_STRIDE_BASE_WRITER_0 + i, D0tlstride[i]);
    }
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, set_addr_remap_index_D0);
#endif
#ifdef ENABLED_CHANNEL_WRITER_0
    for (int i = 0; i < ENABLED_CHANNEL_WRITER_0_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_WRITER_0 + i, channel_en_D0[i]);
    }
#endif

    // ----------------------------------D1 (Writer 1)----------------------------
    csrw_ss(BASE_PTR_WRITER_1_LOW,
            (uint32_t)(delta_local_d1 + snrt_l1_next()));

    for (int i = 0; i < S_STRIDE_NUM_WRITER_1; i++) {
        csrw_ss(S_STRIDE_BASE_WRITER_1 + i, D1slstride[i]);
    }
    for (int i = 0; i < T_BOUND_NUM_WRITER_1; i++) {
        csrw_ss(T_BOUND_BASE_WRITER_1 + i, D1tlbound[i]);
    }
    for (int i = 0; i < T_STRIDE_NUM_WRITER_1; i++) {
        csrw_ss(T_STRIDE_BASE_WRITER_1 + i, D1tlstride[i]);
    }
#ifdef ADDR_REMAP_INDEX_WRITER_1
    csrw_ss(ADDR_REMAP_INDEX_WRITER_1, set_addr_remap_index_D1);
#endif
#ifdef ENABLED_CHANNEL_WRITER_1
    for (int i = 0; i < ENABLED_CHANNEL_WRITER_1_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_WRITER_1 + i, channel_en_D1[i]);
    }
#endif
}

void set_dual_versacore_csr(uint32_t take_in_new_c,
                            uint32_t a_b_input_times_one_output,
                            uint32_t output_times, uint32_t subtractions,
                            uint32_t array_shape, uint32_t data_type) {
    csrw_ss(DUAL_VC_OVERWRITE_ACCUM, take_in_new_c);
    csrw_ss(DUAL_VC_ACCUM_BOUND, a_b_input_times_one_output);
    csrw_ss(DUAL_VC_OUTPUT_BOUND, output_times);
    csrw_ss(DUAL_VC_SUBTRACTIONS, subtractions);
    csrw_ss(DUAL_VC_ARRAY_SHAPE_CFG, array_shape);
    csrw_ss(DUAL_VC_DATA_TYPE_CFG, data_type);
}

void set_dual_versacore_mode(uint32_t mode) {
    csrw_ss(DUAL_VC_MODE, mode);
}

void set_dual_versacore_rescale0(int32_t input_zp, uint32_t multiplier,
                                 int32_t output_zp, uint32_t shift) {
    csrw_ss(DUAL_VC_RESCALE0_INPUT_ZP, (uint32_t)input_zp);
    csrw_ss(DUAL_VC_RESCALE0_MULTIPLIER, multiplier);
    csrw_ss(DUAL_VC_RESCALE0_OUTPUT_ZP, (uint32_t)output_zp);
    csrw_ss(DUAL_VC_RESCALE0_SHIFT, shift);
}

void set_dual_versacore_rescale1(int32_t input_zp, uint32_t multiplier,
                                 int32_t output_zp, uint32_t shift) {
    csrw_ss(DUAL_VC_RESCALE1_INPUT_ZP, (uint32_t)input_zp);
    csrw_ss(DUAL_VC_RESCALE1_MULTIPLIER, multiplier);
    csrw_ss(DUAL_VC_RESCALE1_OUTPUT_ZP, (uint32_t)output_zp);
    csrw_ss(DUAL_VC_RESCALE1_SHIFT, shift);
}

void set_dual_versacore_rescale_mul(int32_t input_zp, uint32_t multiplier,
                                    int32_t output_zp, uint32_t shift) {
    csrw_ss(DUAL_VC_RESCALE_MUL_INPUT_ZP, (uint32_t)input_zp);
    csrw_ss(DUAL_VC_RESCALE_MUL_MULTIPLIER, multiplier);
    csrw_ss(DUAL_VC_RESCALE_MUL_OUTPUT_ZP, (uint32_t)output_zp);
    csrw_ss(DUAL_VC_RESCALE_MUL_SHIFT, shift);
}

void wait_dual_versacore() {
    csrw_ss(DUAL_VC_START, 0);
    csrw_ss(DUAL_VC_START, 0);
    while (csrr_ss(DUAL_VC_BUSY)) {
    }
}

void wait_dual_versacore_streamer() {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    while (csrr_ss(STREAMER_BUSY_CSR)) {
    }
}

void wait_dual_versacore_and_streamer() {
    wait_dual_versacore();
    wait_dual_versacore_streamer();
}

uint32_t read_dual_versacore_streamer_perf_counter() {
    uint32_t perf_counter = csrr_ss(STREAMER_PERFORMANCE_COUNTER_CSR);
    return perf_counter;
}

uint32_t read_dual_versacore_perf_counter() {
    uint32_t perf_counter = csrr_ss(DUAL_VC_PERFORMANCE_COUNTER);
    return perf_counter;
}

uint32_t check_dual_versacore_result_i16(int16_t* output, int16_t* output_golden,
                                         int32_t num_elements) {
    uint32_t err = 0;
    for (int i = 0; i < num_elements; i++) {
        if (output[i] != output_golden[i]) {
            err++;
            printf("Unequals. output[%d] = %d, output_golden[%d] = %d\n", i,
                   output[i], i, output_golden[i]);
        }
    }
    return err;
}

void wait_dual_versacore_writer() {
    while (csrr_ss(WRITER_BUSY_CSR) || csrr_ss(WRITER1_BUSY_CSR)) {
    }
}
