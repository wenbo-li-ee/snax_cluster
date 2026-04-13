// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Dual VersaCore SwiGLU test
// Computes: output = (A @ W >> 2) + (A @ V >> 2)
// where W = B0, V = B1

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

int main() {
    int err = 0;

    // Prepare TCDM addresses
    int8_t *local_a, *local_b0, *local_b1;
    int32_t *local_d;

    local_a  = (int8_t *)(snrt_l1_next() + delta_local_a);
    local_b0 = (int8_t *)(snrt_l1_next() + delta_local_b0);
    local_b1 = (int8_t *)(snrt_l1_next() + delta_local_b1);
    local_d  = (int32_t *)(snrt_l1_next() + delta_local_d);

    // DMA: transfer data from L3 to L1
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, A, a_data_length);
        snrt_dma_start_1d(local_b0, W, b0_data_length);
        snrt_dma_start_1d(local_b1, V, b1_data_length);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    // Compute core
    if (snrt_global_core_idx() == 0) {
        // Configure A streamer (Reader 0)
        int32_t Aslstride[] = {Aslstride0};
        int32_t Atlbound[] = {Atlbound0, Atlbound1, Atlbound2,
                              Atlbound3, Atlbound4, Atlbound5};
        int32_t Atlstride[] = {Atlstride0, Atlstride1, Atlstride2,
                               Atlstride3, Atlstride4, Atlstride5};

        // Configure B0 streamer (Reader 1)
        int32_t B0slstride[] = {B0slstride0};
        int32_t B0tlbound[] = {B0tlbound0, B0tlbound1, B0tlbound2};
        int32_t B0tlstride[] = {B0tlstride0, B0tlstride1, B0tlstride2};

        // Configure B1 streamer (Reader 2)
        int32_t B1slstride[] = {B1slstride0};
        int32_t B1tlbound[] = {B1tlbound0, B1tlbound1, B1tlbound2};
        int32_t B1tlstride[] = {B1tlstride0, B1tlstride1, B1tlstride2};

        // Configure D streamer (Writer 0)
        int32_t Dslstride[] = {Dslstride0};
        int32_t Dtlbound[] = {Dtlbound0, Dtlbound1, Dtlbound2, Dtlbound3};
        int32_t Dtlstride[] = {Dtlstride0, Dtlstride1, Dtlstride2, Dtlstride3};

        // Set streamer CSRs
        set_dual_versacore_streamer_csr(
            delta_local_a, Aslstride, Atlbound, Atlstride,
            set_addr_remap_index_A, channel_en_A,

            delta_local_b0, B0slstride, B0tlbound, B0tlstride,
            set_addr_remap_index_B0, channel_en_B0,

            delta_local_b1, B1slstride, B1tlbound, B1tlstride,
            set_addr_remap_index_B1, channel_en_B1,

            delta_local_d, Dslstride, Dtlbound, Dtlstride,
            set_addr_remap_index_D, channel_en_D);

        // Configure accelerator CSRs (output stationary)
        uint32_t subtraction_setting =
            gen_dual_vc_subtraction_config(subtraction_a, subtraction_b);

        // take_in_new_c = 1: use zero C to reset accumulator on each tile's first beat
        // a_b_input_times_one_output = K
        // output_times = M * N
        set_dual_versacore_csr(1, K, N * M, subtraction_setting, array_shape,
                               data_type);

        printf("DBG: Streamer + accel start\n");

        // Start streamer and accelerator
        set_dual_versacore_streamer_start();
        set_dual_versacore_start();

        // Wait for VersaCore computation to finish
        printf("DBG: Polling DUAL_VC_BUSY\n");
        wait_dual_versacore();
        printf("DBG: VC BUSY done\n");

        // Wait for Streamer to finish writing results to TCDM
        printf("DBG: Polling STREAMER_BUSY\n");
        wait_dual_versacore_streamer();
        printf("DBG: STREAMER done\n");

        // Check result
        err += check_dual_versacore_result((int8_t *)local_d, (int8_t *)D,
                                           d_data_length);

        printf(
            "Dual VersaCore SwiGLU: %s, Error: %d.\n",
            err ? "FAIL" : "PASS", err);

        int32_t cycles = read_dual_versacore_perf_counter();
        int32_t streamer_cycles = read_dual_versacore_streamer_perf_counter();
        printf("Workload: M=%d, N=%d, K=%d, meshRow=%d, tileSize=%d, meshCol=%d\n",
               M, N, K, meshRow, tileSize, meshCol);
        printf("Accelerator cycles: %d\n", cycles);
        printf("Streamer cycles: %d\n", streamer_cycles);
    }

    return err;
}
