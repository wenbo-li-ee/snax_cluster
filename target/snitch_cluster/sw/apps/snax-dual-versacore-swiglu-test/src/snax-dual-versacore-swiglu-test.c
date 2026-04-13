// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Dual VersaCore SwiGLU test — Block-Level Pipeline
// Computes: output = (A @ W >> 2) + (A @ V >> 2)
// Split M=20 into two blocks of 10 to verify block pipeline

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

int main() {
    int err = 0;

    // Block pipeline parameters
    // Split M=20 into 2 blocks of M_BLOCK=10
    const uint32_t M_BLOCK = 10;

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
        uint32_t subtraction_setting =
            gen_dual_vc_subtraction_config(subtraction_a, subtraction_b);

        // ============================================================
        // Block 0: M_tile 0~9
        // ============================================================
        printf("DBG: Block 0 start (M_tile 0~%d)\n", M_BLOCK - 1);

        // Reader A: same base, but M-bound = M_BLOCK instead of M
        int32_t Aslstride_b0[] = {Aslstride0};
        int32_t Atlbound_b0[]  = {Atlbound0, Atlbound1, M_BLOCK,
                                  Atlbound3, Atlbound4, Atlbound5};
        int32_t Atlstride_b0[] = {Atlstride0, Atlstride1, Atlstride2,
                                  Atlstride3, Atlstride4, Atlstride5};

        // Reader B0: M-bound = M_BLOCK (stride=0, weight stationary)
        int32_t B0slstride_b0[] = {B0slstride0};
        int32_t B0tlbound_b0[]  = {B0tlbound0, B0tlbound1, M_BLOCK};
        int32_t B0tlstride_b0[] = {B0tlstride0, B0tlstride1, B0tlstride2};

        // Reader B1: M-bound = M_BLOCK
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

        // VersaCore: output_times = N * M_BLOCK
        set_dual_versacore_csr(1, K, N * M_BLOCK, subtraction_setting,
                               array_shape, data_type);

        set_dual_versacore_streamer_start();
        set_dual_versacore_start();

        // Wait for VersaCore to finish Block 0 (Writer may still be writing)
        wait_dual_versacore();
        printf("DBG: Block 0 VC done\n");

        // ============================================================
        // Block 1: M_tile 10~19 (start immediately, don't wait for Writer!)
        // ============================================================
        printf("DBG: Block 1 start (M_tile %d~%d)\n", M_BLOCK, M - 1);

        // A base offset: skip first M_BLOCK M-tiles of A data
        // Each M-tile = meshRow * tileSize * K bytes (int8)
        int32_t a_block1_base = delta_local_a +
            (int32_t)(M_BLOCK * Atlstride2);

        // Reader A for Block 1
        int32_t Atlbound_b1[]  = {Atlbound0, Atlbound1, M_BLOCK,
                                  Atlbound3, Atlbound4, Atlbound5};
        int32_t Atlstride_b1[] = {Atlstride0, Atlstride1, Atlstride2,
                                  Atlstride3, Atlstride4, Atlstride5};

        // Reader B0 for Block 1 (weight stationary: same base, same config)
        int32_t B0tlbound_b1[]  = {B0tlbound0, B0tlbound1, M_BLOCK};
        int32_t B0tlstride_b1[] = {B0tlstride0, B0tlstride1, B0tlstride2};

        // Reader B1 for Block 1 (weight stationary: same base, same config)
        int32_t B1tlbound_b1[]  = {B1tlbound0, B1tlbound1, M_BLOCK};
        int32_t B1tlstride_b1[] = {B1tlstride0, B1tlstride1, B1tlstride2};

        // Restart only readers (Writer continues from Block 0's config!)
        restart_dual_versacore_readers(
            a_block1_base, Aslstride_b0, Atlbound_b1, Atlstride_b1,
            set_addr_remap_index_A, channel_en_A,

            delta_local_b0, B0slstride_b0, B0tlbound_b1, B0tlstride_b1,
            set_addr_remap_index_B0, channel_en_B0,

            delta_local_b1, B1slstride_b0, B1tlbound_b1, B1tlstride_b1,
            set_addr_remap_index_B1, channel_en_B1);

        // VersaCore: output_times = N * M_BLOCK
        set_dual_versacore_csr(1, K, N * M_BLOCK, subtraction_setting,
                               array_shape, data_type);
        set_dual_versacore_start();

        // Wait for VersaCore to finish Block 1
        wait_dual_versacore();
        printf("DBG: Block 1 VC done\n");

        // Now wait for Writer to finish writing all results
        wait_dual_versacore_writer();
        printf("DBG: Writer done\n");

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
