// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Dual VersaCore SwiGLU test — Block-Level Pipeline (optimized)
// Computes: output = (A @ W >> 2) + (A @ V >> 2)
// Split M=20 into two blocks of 10 to verify block pipeline

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

int main() {
    int err = 0;

    const uint32_t M_BLOCK = 10;

    int8_t *local_a, *local_b0, *local_b1;
    int32_t *local_d;

    local_a  = (int8_t *)(snrt_l1_next() + delta_local_a);
    local_b0 = (int8_t *)(snrt_l1_next() + delta_local_b0);
    local_b1 = (int8_t *)(snrt_l1_next() + delta_local_b1);
    local_d  = (int32_t *)(snrt_l1_next() + delta_local_d);

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, A, a_data_length);
        snrt_dma_start_1d(local_b0, W, b0_data_length);
        snrt_dma_start_1d(local_b1, V, b1_data_length);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        uint32_t subtraction_setting =
            gen_dual_vc_subtraction_config(subtraction_a, subtraction_b);

        // ============================================================
        // Block 0: M_tile 0~9
        // ============================================================

        // Reader A/B0/B1: M-bound = M_BLOCK
        int32_t Aslstride_b0[] = {Aslstride0};
        int32_t Atlbound_b0[]  = {Atlbound0, Atlbound1, M_BLOCK,
                                  Atlbound3, Atlbound4, Atlbound5};
        int32_t Atlstride_b0[] = {Atlstride0, Atlstride1, Atlstride2,
                                  Atlstride3, Atlstride4, Atlstride5};
        int32_t B0slstride_b0[] = {B0slstride0};
        int32_t B0tlbound_b0[]  = {B0tlbound0, B0tlbound1, M_BLOCK};
        int32_t B0tlstride_b0[] = {B0tlstride0, B0tlstride1, B0tlstride2};
        int32_t B1slstride_b0[] = {B1slstride0};
        int32_t B1tlbound_b0[]  = {B1tlbound0, B1tlbound1, M_BLOCK};
        int32_t B1tlstride_b0[] = {B1tlstride0, B1tlstride1, B1tlstride2};

        // Writer D: configured for FULL M=20 output (runs continuously)
        int32_t Dslstride_all[] = {Dslstride0};
        int32_t Dtlbound_all[]  = {Dtlbound0, Dtlbound1, M, Dtlbound3};
        int32_t Dtlstride_all[] = {Dtlstride0, Dtlstride1, Dtlstride2, Dtlstride3};

        // Configure ALL streamer CSRs (readers + writer) and start
        set_dual_versacore_streamer_csr(
            delta_local_a, Aslstride_b0, Atlbound_b0, Atlstride_b0,
            set_addr_remap_index_A, channel_en_A,
            delta_local_b0, B0slstride_b0, B0tlbound_b0, B0tlstride_b0,
            set_addr_remap_index_B0, channel_en_B0,
            delta_local_b1, B1slstride_b0, B1tlbound_b0, B1tlstride_b0,
            set_addr_remap_index_B1, channel_en_B1,
            delta_local_d, Dslstride_all, Dtlbound_all, Dtlstride_all,
            set_addr_remap_index_D, channel_en_D);

        set_dual_versacore_csr(1, K, N * M_BLOCK, subtraction_setting,
                               array_shape, data_type);
        set_dual_versacore_streamer_start();
        set_dual_versacore_start();

        // ============================================================
        // Block 1: M_tile 10~19
        // Minimal CSR + overlap + implicit stall
        // ============================================================
        int32_t a_block1_base = delta_local_a +
            (int32_t)(M_BLOCK * Atlstride2);

        // Pre-write new A base addr while VC still computing Block 0
        // (non-start CSR writes always accepted, running readers unaffected)
        csrw_ss(BASE_PTR_READER_0_LOW,
                (uint32_t)(a_block1_base + snrt_l1_next()));

        // Restart readers: stalls until readers_all_done, writer keeps running
        csrw_ss(STREAMER_START_CSR, 1);

        // Restart VC: stalls until Block 0 VC finishes, then re-triggers
        set_dual_versacore_start();

        // Wait for Block 1 VC to finish
        wait_dual_versacore();

        // Wait for Writer to finish writing all results
        wait_dual_versacore_writer();

        // Check full result (M=20)
        err += check_dual_versacore_result((int8_t *)local_d, (int8_t *)D,
                                           d_data_length);

        printf(
            "Dual VersaCore SwiGLU: %s, Error: %d.\n",
            err ? "FAIL" : "PASS", err);

        int32_t cycles = read_dual_versacore_perf_counter();
        int32_t streamer_cycles = read_dual_versacore_streamer_perf_counter();
        printf("Workload: M=%d (2 blocks of %d), N=%d, K=%d\n",
               M, M_BLOCK, N, K);
        printf("Accelerator cycles: %d\n", cycles);
        printf("Streamer cycles: %d\n", streamer_cycles);
    }

    return err;
}
