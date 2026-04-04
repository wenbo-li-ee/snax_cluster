// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "snrt.h"

#include <stdint.h>

#include "data.h"
#include "snax-ewu-lib.h"

static inline void split_addr(const void *ptr, uint32_t *low, uint32_t *high) {
    uint64_t addr = (uint64_t)(uintptr_t)ptr;
    *low = (uint32_t)addr;
    *high = (uint32_t)(addr >> 32);
}

static void run_ewu(uint32_t mode, const uint64_t *local_a, const uint64_t *local_b,
                    uint64_t *local_o, const uint64_t *golden, int *err) {
    uint32_t base_low;
    uint32_t base_high;

    split_addr(local_a, &base_low, &base_high);
    configure_streamer_a(base_low, base_high, sizeof(uint64_t), LOOP_ITER,
                         INPUT_WORDS_PER_ITER * sizeof(uint64_t));

    split_addr(local_b, &base_low, &base_high);
    configure_streamer_b(base_low, base_high, sizeof(uint64_t), LOOP_ITER,
                         INPUT_WORDS_PER_ITER * sizeof(uint64_t));

    split_addr(local_o, &base_low, &base_high);
    configure_streamer_o(base_low, base_high, sizeof(uint64_t), LOOP_ITER,
                         OUTPUT_WORDS_PER_ITER * sizeof(uint64_t));

    configure_ewu(mode, LOOP_ITER);
    start_streamer();
    start_ewu();

    while (read_busy_ewu()) {
    }

    while (read_busy_streamer()) {
    }

    for (uint32_t i = 0; i < OUTPUT_WORDS; i++) {
        if (golden[i] != local_o[i]) {
            (*err)++;
        }
    }
}

int main() {
    int err = 0;

    uint64_t *local_a = (uint64_t *)snrt_l1_next();
    uint64_t *local_b = local_a + INPUT_WORDS;
    uint64_t *local_add_o = local_b + INPUT_WORDS;
    uint64_t *local_mul_o = local_add_o + OUTPUT_WORDS;

    if (snrt_is_dm_core()) {
        size_t vector_size = INPUT_WORDS * sizeof(uint64_t);
        snrt_dma_start_1d(local_a, A_PACKED, vector_size);
        snrt_dma_start_1d(local_b, B_PACKED, vector_size);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    if (snrt_is_compute_core()) {
        run_ewu(EWU_MODE_ADD, local_a, local_b, local_add_o, OUT_ADD_PACKED, &err);
        uint32_t add_perf_count = csrr_ss(EWU_RO_PERF_COUNT);

        run_ewu(EWU_MODE_MUL, local_a, local_b, local_mul_o, OUT_MUL_PACKED, &err);
        uint32_t mul_perf_count = csrr_ss(EWU_RO_PERF_COUNT);

        printf("EWU Done!\n");
        printf("EWU add cycles: %d\n", add_perf_count);
        printf("EWU mul cycles: %d\n", mul_perf_count);
        printf("Number of errors: %d\n", err);
    }

    return err;
}
