// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// FP16xINT4 Dual VersaCore closed-loop test.
//
// Mode 0:
//   D0 = fp16(SiLU_fp32(A @ W) * (A @ V))
//
// Mode 1:
//   D0 = fp16(mode0_D0 @ W2_left)
//   D1 = fp16(mode0_D0 @ W2_right)
//
// The important closed-loop detail is that Mode 1 Reader A points directly to
// DELTA_LOCAL_MODE0_D0.  Mode 0 already writes FP16, so no software cast or
// intermediate copy is needed.

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

static const int32_t a_spatial_stride[2] = {8, 16};
static const int32_t a_temporal_bound[6] = {1, 1, 1, 1, 1, 1};
static const int32_t a_temporal_stride[6] = {0, 0, 0, 0, 0, 0};
static const int32_t a_channel_enable[1] = {0x00ff};

static const int32_t b_spatial_stride[2] = {8, 16};
static const int32_t b_temporal_bound[4] = {1, 1, 1, 1};
static const int32_t b_temporal_stride[4] = {0, 0, 0, 0};
static const int32_t b_channel_enable[1] = {0x000f};

static const int32_t d_spatial_stride[1] = {8};
static const int32_t d_temporal_bound[4] = {8, 1, 1, 1};
static const int32_t d_temporal_stride[4] = {8, 0, 0, 0};
static const int32_t d_channel_enable[1] = {0x0001};

static int check_fp16_bits(const char *name, const uint16_t *actual,
                           const uint16_t *expected, int elements) {
    int errors = 0;
    for (int i = 0; i < elements; i++) {
        if (actual[i] != expected[i]) {
            if (errors < 8) {
                printf("%s mismatch[%d]: got=0x%04x expected=0x%04x\n",
                       name, i, actual[i], expected[i]);
            }
            errors++;
        }
    }
    printf("%s: %s, errors=%d\n", name, errors ? "FAIL" : "PASS", errors);
    return errors;
}

static void configure_accelerator(int mode) {
    uint32_t subtraction = gen_dual_vc_subtraction_config(0, 0);

    set_dual_versacore_csr(1, K_TILES, OUTPUT_TILES, subtraction,
                           ARRAY_SHAPE, DATA_TYPE);
    set_dual_versacore_mode(mode);

    // CSR[7..18] remain present for ABI compatibility with the integer
    // accelerator, but the FP32 postprocess does not use any rescale CSR.
}

static int run_mode0(void) {
    uint16_t *mode0_d0 =
        (uint16_t *)(snrt_l1_next() + DELTA_LOCAL_MODE0_D0);

    set_dual_versacore_streamer_csr_d0_only(
        DELTA_LOCAL_A, a_spatial_stride, a_temporal_bound, a_temporal_stride,
        0, a_channel_enable,
        DELTA_LOCAL_B0, b_spatial_stride, b_temporal_bound, b_temporal_stride,
        0, b_channel_enable,
        DELTA_LOCAL_B1, b_spatial_stride, b_temporal_bound, b_temporal_stride,
        0, b_channel_enable,
        DELTA_LOCAL_MODE0_D0, d_spatial_stride, d_temporal_bound,
        d_temporal_stride, 0, d_channel_enable);

    configure_accelerator(0);

    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    wait_dual_versacore_and_streamer();

    printf("Mode 0 cycles: accelerator=%u, streamer=%u\n",
           read_dual_versacore_perf_counter(),
           read_dual_versacore_streamer_perf_counter());
    return check_fp16_bits("Mode 0 SwiGLU", mode0_d0,
                           mode0_golden_fp16, OUTPUT_ELEMENTS);
}

static int run_mode1(void) {
    uint16_t *mode1_d0 =
        (uint16_t *)(snrt_l1_next() + DELTA_LOCAL_MODE1_D0);
    uint16_t *mode1_d1 =
        (uint16_t *)(snrt_l1_next() + DELTA_LOCAL_MODE1_D1);

    set_dual_versacore_streamer_csr(
        // Closed loop: read the FP16 tokens written by Mode 0 in place.
        DELTA_LOCAL_MODE0_D0, a_spatial_stride, a_temporal_bound,
        a_temporal_stride, 0, a_channel_enable,
        DELTA_LOCAL_W2_LEFT, b_spatial_stride, b_temporal_bound,
        b_temporal_stride, 0, b_channel_enable,
        DELTA_LOCAL_W2_RIGHT, b_spatial_stride, b_temporal_bound,
        b_temporal_stride, 0, b_channel_enable,
        DELTA_LOCAL_MODE1_D0, d_spatial_stride, d_temporal_bound,
        d_temporal_stride, 0, d_channel_enable,
        DELTA_LOCAL_MODE1_D1, d_spatial_stride, d_temporal_bound,
        d_temporal_stride, 0, d_channel_enable);

    configure_accelerator(1);

    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    wait_dual_versacore_and_streamer();

    printf("Mode 1 cycles: accelerator=%u, streamer=%u\n",
           read_dual_versacore_perf_counter(),
           read_dual_versacore_streamer_perf_counter());

    int errors = 0;
    errors += check_fp16_bits("Mode 1 Writer 0", mode1_d0,
                              mode1_d0_golden_fp16, OUTPUT_ELEMENTS);
    errors += check_fp16_bits("Mode 1 Writer 1", mode1_d1,
                              mode1_d1_golden_fp16, OUTPUT_ELEMENTS);
    return errors;
}

int main(void) {
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d((void *)(snrt_l1_next() + DELTA_LOCAL_A),
                          input_a_fp16, A_DATA_LENGTH);
        snrt_dma_start_1d((void *)(snrt_l1_next() + DELTA_LOCAL_B0),
                          mode0_weight_w, WEIGHT_DATA_LENGTH);
        snrt_dma_start_1d((void *)(snrt_l1_next() + DELTA_LOCAL_B1),
                          mode0_weight_v, WEIGHT_DATA_LENGTH);
        snrt_dma_start_1d((void *)(snrt_l1_next() + DELTA_LOCAL_W2_LEFT),
                          mode1_weight_left, WEIGHT_DATA_LENGTH);
        snrt_dma_start_1d((void *)(snrt_l1_next() + DELTA_LOCAL_W2_RIGHT),
                          mode1_weight_right, WEIGHT_DATA_LENGTH);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    int errors = 0;
    if (snrt_global_core_idx() == 0) {
        errors += run_mode0();
        errors += run_mode1();
        printf("FP16xINT4 dual VersaCore total errors: %d\n", errors);
    }

    return errors;
}
