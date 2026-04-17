// Copyright 2025 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Dual VersaCore end-to-end SwiGLU test
// Step 1 — Mode 0 (SwiGLU): output = rescale_mul( (rescale0(A@W)>>2) * rescale1(A@V) )
// Step 2 — SW cast int16 → int8
// Step 3 — Mode 1 (GEMM):  D0 = rescale0(A1 @ W2_left), D1 = rescale1(A1 @ W2_right)

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

int main() {
    int err = 0;

    int8_t *local_a, *local_b0, *local_b1;
    int16_t *local_d0;
    int8_t *local_w2l, *local_w2r;
    int16_t *local_mode1_d0, *local_mode1_d1;

    local_a  = (int8_t *)(snrt_l1_next() + delta_local_a);
    local_b0 = (int8_t *)(snrt_l1_next() + delta_local_b0);
    local_b1 = (int8_t *)(snrt_l1_next() + delta_local_b1);
    local_d0 = (int16_t *)(snrt_l1_next() + delta_local_d0);
    local_w2l = (int8_t *)(snrt_l1_next() + delta_local_w2l);
    local_w2r = (int8_t *)(snrt_l1_next() + delta_local_w2r);
    local_mode1_d0 = (int16_t *)(snrt_l1_next() + delta_local_mode1_d0);
    local_mode1_d1 = (int16_t *)(snrt_l1_next() + delta_local_mode1_d1);

    // ================================================================
    // DMA: load all inputs (mode 0 + mode 1)
    // ================================================================
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, A, a_data_length);
        snrt_dma_start_1d(local_b0, W, b0_data_length);
        snrt_dma_start_1d(local_b1, V, b1_data_length);
        snrt_dma_start_1d(local_w2l, W2_left, w2l_data_length);
        snrt_dma_start_1d(local_w2r, W2_right, w2r_data_length);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        uint32_t subtraction_setting =
            gen_dual_vc_subtraction_config(subtraction_a, subtraction_b);

        // ============================================================
        // Mode 0 (SwiGLU)
        // ============================================================

        // Streamer CSR arrays for Reader A
        int32_t Aslstride_arr[] = {Aslstride0};
        int32_t Atlbound_arr[]  = {Atlbound0, Atlbound1, Atlbound2,
                                   Atlbound3, Atlbound4, Atlbound5};
        int32_t Atlstride_arr[] = {Atlstride0, Atlstride1, Atlstride2,
                                   Atlstride3, Atlstride4, Atlstride5};

        // Streamer CSR arrays for Reader B0
        int32_t B0slstride_arr[] = {B0slstride0};
        int32_t B0tlbound_arr[]  = {B0tlbound0, B0tlbound1, B0tlbound2};
        int32_t B0tlstride_arr[] = {B0tlstride0, B0tlstride1, B0tlstride2};

        // Streamer CSR arrays for Reader B1
        int32_t B1slstride_arr[] = {B1slstride0};
        int32_t B1tlbound_arr[]  = {B1tlbound0, B1tlbound1, B1tlbound2};
        int32_t B1tlstride_arr[] = {B1tlstride0, B1tlstride1, B1tlstride2};

        // Streamer CSR arrays for Writer D0
        int32_t D0slstride_arr[] = {D0slstride0};
        int32_t D0tlbound_arr[]  = {Dtlbound0, Dtlbound1, Dtlbound2, Dtlbound3};
        int32_t D0tlstride_arr[] = {Dtlstride0, Dtlstride1, Dtlstride2, Dtlstride3};

        // Streamer CSR arrays for Writer D1 (same data as D0 in mode 0, dummy addr)
        int32_t D1slstride_arr[] = {D1slstride0};
        int32_t D1tlbound_arr[]  = {Dtlbound0, Dtlbound1, Dtlbound2, Dtlbound3};
        int32_t D1tlstride_arr[] = {Dtlstride0, Dtlstride1, Dtlstride2, Dtlstride3};

        // Configure all streamer CSRs
        set_dual_versacore_streamer_csr(
            delta_local_a, Aslstride_arr, Atlbound_arr, Atlstride_arr,
            set_addr_remap_index_A, channel_en_A,
            delta_local_b0, B0slstride_arr, B0tlbound_arr, B0tlstride_arr,
            set_addr_remap_index_B0, channel_en_B0,
            delta_local_b1, B1slstride_arr, B1tlbound_arr, B1tlstride_arr,
            set_addr_remap_index_B1, channel_en_B1,
            delta_local_d0, D0slstride_arr, D0tlbound_arr, D0tlstride_arr,
            set_addr_remap_index_D0, channel_en_D0,
            delta_local_d1_mode0, D1slstride_arr, D1tlbound_arr, D1tlstride_arr,
            set_addr_remap_index_D1, channel_en_D1);

        // Configure accelerator CSRs
        set_dual_versacore_csr(1, K, N * M, subtraction_setting,
                               array_shape, data_type);

        // Set mode 0 (SwiGLU)
        set_dual_versacore_mode(0);

        // Set rescale parameters (identity)
        set_dual_versacore_rescale0(rescale_input_zp, rescale_multiplier,
                                    rescale_output_zp, rescale_shift);
        set_dual_versacore_rescale1(rescale_input_zp, rescale_multiplier,
                                    rescale_output_zp, rescale_shift);
        set_dual_versacore_rescale_mul(rescale_input_zp, rescale_multiplier,
                                       rescale_output_zp, rescale_shift);

        // Start
        set_dual_versacore_streamer_start();
        set_dual_versacore_start();

        // Wait for completion
        wait_dual_versacore();
        wait_dual_versacore_writer();

        // Check mode 0 result
        err += check_dual_versacore_result_i16(
            local_d0, (int16_t *)mode0_golden, mode0_output_elems);

        int32_t cycles_m0 = read_dual_versacore_perf_counter();
        int32_t streamer_cycles_m0 = read_dual_versacore_streamer_perf_counter();
        printf("Mode 0 SwiGLU: %s, Error: %d.\n",
               err ? "FAIL" : "PASS", err);
        printf("  Workload: M=%d, N=%d, K=%d\n", M, N, K);
        printf("  Accelerator cycles: %d, Streamer cycles: %d\n",
               cycles_m0, streamer_cycles_m0);

        // ============================================================
        // SW cast: int16 -> int8 (saturating clamp to [-128, 127])
        // Write back to local_a (overwrites original A)
        // ============================================================
        for (int i = 0; i < mode0_output_elems; i++) {
            int16_t val = local_d0[i];
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            local_a[i] = (int8_t)val;
        }

        // ============================================================
        // Mode 1 (GEMM): A1 @ W2_left -> D0, A1 @ W2_right -> D1
        // ============================================================

        // Reader A (reusing local_a which now has cast result)
        int32_t M1_Aslstride_arr[] = {Aslstride0};
        int32_t M1_Atlbound_arr[]  = {M1_Atlbound0, M1_Atlbound1, M1_Atlbound2,
                                      1, 1, 1};
        int32_t M1_Atlstride_arr[] = {M1_Atlstride0, M1_Atlstride1, M1_Atlstride2,
                                      0, 0, 0};

        // Reader B0 (W2_left)
        int32_t M1_B0slstride_arr[] = {B0slstride0};
        int32_t M1_B0tlbound_arr[]  = {M1_B0tlbound0, M1_B0tlbound1, M1_B0tlbound2};
        int32_t M1_B0tlstride_arr[] = {M1_B0tlstride0, M1_B0tlstride1, M1_B0tlstride2};

        // Reader B1 (W2_right)
        int32_t M1_B1slstride_arr[] = {B1slstride0};
        int32_t M1_B1tlbound_arr[]  = {M1_B1tlbound0, M1_B1tlbound1, M1_B1tlbound2};
        int32_t M1_B1tlstride_arr[] = {M1_B1tlstride0, M1_B1tlstride1, M1_B1tlstride2};

        // Writer D0 (mode 1 VC0 output)
        int32_t M1_D0slstride_arr[] = {D0slstride0};
        int32_t M1_D0tlbound_arr[]  = {M1_Dtlbound0, M1_Dtlbound1,
                                       M1_Dtlbound2, M1_Dtlbound3};
        int32_t M1_D0tlstride_arr[] = {M1_Dtlstride0, M1_Dtlstride1,
                                       M1_Dtlstride2, M1_Dtlstride3};

        // Writer D1 (mode 1 VC1 output)
        int32_t M1_D1slstride_arr[] = {D1slstride0};
        int32_t M1_D1tlbound_arr[]  = {M1_Dtlbound0, M1_Dtlbound1,
                                       M1_Dtlbound2, M1_Dtlbound3};
        int32_t M1_D1tlstride_arr[] = {M1_Dtlstride0, M1_Dtlstride1,
                                       M1_Dtlstride2, M1_Dtlstride3};

        // Configure all streamer CSRs for mode 1
        set_dual_versacore_streamer_csr(
            delta_local_a, M1_Aslstride_arr, M1_Atlbound_arr, M1_Atlstride_arr,
            set_addr_remap_index_A, channel_en_A,
            delta_local_w2l, M1_B0slstride_arr, M1_B0tlbound_arr, M1_B0tlstride_arr,
            set_addr_remap_index_B0, channel_en_B0,
            delta_local_w2r, M1_B1slstride_arr, M1_B1tlbound_arr, M1_B1tlstride_arr,
            set_addr_remap_index_B1, channel_en_B1,
            delta_local_mode1_d0, M1_D0slstride_arr, M1_D0tlbound_arr, M1_D0tlstride_arr,
            set_addr_remap_index_D0, channel_en_D0,
            delta_local_mode1_d1, M1_D1slstride_arr, M1_D1tlbound_arr, M1_D1tlstride_arr,
            set_addr_remap_index_D1, channel_en_D1);

        // Configure accelerator CSRs for mode 1
        set_dual_versacore_csr(1, K1, N1 * M1, subtraction_setting,
                               array_shape, data_type);

        // Set mode 1 (GEMM)
        set_dual_versacore_mode(1);

        // Set rescale parameters (identity, rescale_mul unused in mode 1)
        set_dual_versacore_rescale0(rescale_input_zp, rescale_multiplier,
                                    rescale_output_zp, rescale_shift);
        set_dual_versacore_rescale1(rescale_input_zp, rescale_multiplier,
                                    rescale_output_zp, rescale_shift);
        set_dual_versacore_rescale_mul(rescale_input_zp, rescale_multiplier,
                                       rescale_output_zp, rescale_shift);

        // Start
        set_dual_versacore_streamer_start();
        set_dual_versacore_start();

        // Wait for completion
        wait_dual_versacore();
        wait_dual_versacore_writer();

        // Check mode 1 results
        int err_d0 = check_dual_versacore_result_i16(
            local_mode1_d0, (int16_t *)mode1_golden_d0, mode1_output_elems);
        int err_d1 = check_dual_versacore_result_i16(
            local_mode1_d1, (int16_t *)mode1_golden_d1, mode1_output_elems);

        err += err_d0 + err_d1;

        int32_t cycles_m1 = read_dual_versacore_perf_counter();
        int32_t streamer_cycles_m1 = read_dual_versacore_streamer_perf_counter();
        printf("Mode 1 GEMM D0: %s, Error: %d.\n",
               err_d0 ? "FAIL" : "PASS", err_d0);
        printf("Mode 1 GEMM D1: %s, Error: %d.\n",
               err_d1 ? "FAIL" : "PASS", err_d1);
        printf("  Workload: M1=%d, N1=%d, K1=%d\n", M1, N1, K1);
        printf("  Accelerator cycles: %d, Streamer cycles: %d\n",
               cycles_m1, streamer_cycles_m1);
    }

    return err;
}
