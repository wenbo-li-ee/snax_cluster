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

    int32_t delta_local_d, int32_t* Dslstride, int32_t* Dtlbound,
    int32_t* Dtlstride, int32_t set_addr_remap_index_D,
    int32_t* channel_en_D) {

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

    // ----------------------------------D (Writer 0)----------------------------
    csrw_ss(BASE_PTR_WRITER_0_LOW,
            (uint32_t)(delta_local_d + snrt_l1_next()));

    for (int i = 0; i < S_STRIDE_NUM_WRITER_0; i++) {
        csrw_ss(S_STRIDE_BASE_WRITER_0 + i, Dslstride[i]);
    }
    for (int i = 0; i < T_BOUND_NUM_WRITER_0; i++) {
        csrw_ss(T_BOUND_BASE_WRITER_0 + i, Dtlbound[i]);
    }
    for (int i = 0; i < T_STRIDE_NUM_WRITER_0; i++) {
        csrw_ss(T_STRIDE_BASE_WRITER_0 + i, Dtlstride[i]);
    }
#ifdef ADDR_REMAP_INDEX_WRITER_0
    csrw_ss(ADDR_REMAP_INDEX_WRITER_0, set_addr_remap_index_D);
#endif
#ifdef ENABLED_CHANNEL_WRITER_0
    for (int i = 0; i < ENABLED_CHANNEL_WRITER_0_CSR_NUM; i++) {
        csrw_ss(ENABLED_CHANNEL_WRITER_0 + i, channel_en_D[i]);
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

uint32_t check_dual_versacore_result(int8_t* output, int8_t* output_golden,
                                     int32_t data_length) {
    uint32_t err = 0;
    for (int i = 0; i < data_length; i++) {
        if (output[i] != output_golden[i]) {
            err++;
            printf("Unequals. output[%d] = %d, output_golden[%d] = %d\n", i,
                   output[i], i, output_golden[i]);
        }
    }
    return err;
}
