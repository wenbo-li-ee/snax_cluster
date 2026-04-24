// Int16x4 scaled 1/16 batch test
// Mode 0 (SwiGLU): A[M,K] @ W[K,N], V[K,N] → Output0[M,N] (SwiGLU activation)
// Mode 1 (GEMM): A'=compact(Output0) @ W2_left[N,N1], W2_right[N,N1]
// Mode 1 A input = compact Mode 0 D0 output (SW strips per-tile padding into local_a1)

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

int main() {
    int err = 0;

    int16_t *local_a;
    uint8_t *local_b0, *local_b1;
    int16_t *local_d0, *local_d1_mode0;
    int16_t *local_a1;
    uint8_t *local_w2l, *local_w2r;
    int16_t *local_mode1_d0, *local_mode1_d1;

    local_a  = (int16_t *)(snrt_l1_next() + delta_local_a);
    local_b0 = (uint8_t *)(snrt_l1_next() + delta_local_b0);
    local_b1 = (uint8_t *)(snrt_l1_next() + delta_local_b1);
    local_d0 = (int16_t *)(snrt_l1_next() + delta_local_d0);
    local_d1_mode0 = (int16_t *)(snrt_l1_next() + delta_local_d1_mode0);
    local_a1 = (int16_t *)(snrt_l1_next() + delta_local_a1);
    local_w2l = (uint8_t *)(snrt_l1_next() + delta_local_w2l);
    local_w2r = (uint8_t *)(snrt_l1_next() + delta_local_w2r);
    local_mode1_d0 = (int16_t *)(snrt_l1_next() + delta_local_mode1_d0);
    local_mode1_d1 = (int16_t *)(snrt_l1_next() + delta_local_mode1_d1);

    // DMA: load all inputs (A1 is filled by SW compact loop after Mode 0, not DMA)
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
        int32_t Aslstride_arr[] = {Aslstride0};
        int32_t Atlbound_arr[]  = {Atlbound0, Atlbound1, Atlbound2,
                                   Atlbound3, Atlbound4, Atlbound5};
        int32_t Atlstride_arr[] = {Atlstride0, Atlstride1, Atlstride2,
                                   Atlstride3, Atlstride4, Atlstride5};

        int32_t B0slstride_arr[] = {B0slstride0};
        int32_t B0tlbound_arr[]  = {B0tlbound0, B0tlbound1, B0tlbound2, B0tlbound3};
        int32_t B0tlstride_arr[] = {B0tlstride0, B0tlstride1, B0tlstride2, B0tlstride3};

        int32_t B1slstride_arr[] = {B1slstride0};
        int32_t B1tlbound_arr[]  = {B1tlbound0, B1tlbound1, B1tlbound2, B1tlbound3};
        int32_t B1tlstride_arr[] = {B1tlstride0, B1tlstride1, B1tlstride2, B1tlstride3};

        int32_t D0slstride_arr[] = {D0slstride0};
        int32_t D0tlbound_arr[]  = {Dtlbound0, Dtlbound1, Dtlbound2, Dtlbound3};
        int32_t D0tlstride_arr[] = {Dtlstride0, Dtlstride1, Dtlstride2, Dtlstride3};

        int32_t D1slstride_arr[] = {D1slstride0};
        int32_t D1tlbound_arr[]  = {Dtlbound0, Dtlbound1, Dtlbound2, Dtlbound3};
        int32_t D1tlstride_arr[] = {Dtlstride0, Dtlstride1, Dtlstride2, Dtlstride3};

        uint32_t m0_start = snrt_mcycle();

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

        set_dual_versacore_csr(1, K, N * M, subtraction_setting,
                               array_shape, data_type);
        set_dual_versacore_mode(0);
        set_dual_versacore_rescale0(rescale_input_zp, rescale_multiplier,
                                    rescale_output_zp, rescale_shift);
        set_dual_versacore_rescale1(rescale_input_zp, rescale_multiplier,
                                    rescale_output_zp, rescale_shift);
        set_dual_versacore_rescale_mul(rescale_input_zp, rescale_multiplier,
                                       rescale_output_zp, rescale_shift);

        set_dual_versacore_streamer_start();
        set_dual_versacore_start();
        wait_dual_versacore();
        wait_dual_versacore_writer();

        uint32_t m0_end = snrt_mcycle();

        err += check_dual_versacore_result_i16(
            local_d0, (int16_t *)mode0_golden_padded, mode0_output_elems_padded);

        int32_t cycles_m0 = read_dual_versacore_perf_counter();
        int32_t str_cycles_m0 = read_dual_versacore_streamer_perf_counter();
        printf("Mode 0 SwiGLU: %s, Error: %d\n", err ? "FAIL" : "PASS", err);
        printf("  M0 Cycles: accel=%d, streamer=%d, wall=%u\n",
               cycles_m0, str_cycles_m0, m0_end - m0_start);

        // Compact Mode 0 D0 output for Mode 1 A input.
        // Each D0 tile is mode0_output_elems_padded/(M*N) int16 (fixed 64-byte beat)
        // but only the first mode0_output_elems/(M*N) int16 are real SwiGLU values.
        // Strip the padding so Mode 1 A reader sees a dense [M,N,meshRow,meshCol] buffer.
        {
            int tile_padded = (int)(mode0_output_elems_padded / (M * N));
            int tile_real   = (int)(mode0_output_elems / (M * N));
            for (int idx = 0; idx < (int)(M * N); idx++) {
                int16_t *src = local_d0 + (int32_t)idx * tile_padded;
                int16_t *dst = local_a1 + (int32_t)idx * tile_real;
                for (int i = 0; i < tile_real; i++) {
                    dst[i] = src[i];
                }
            }
        }

        // ============================================================
        // Mode 1 (GEMM) — chained: compact Mode 0 output as A1 input
        // ============================================================
        int32_t M1_Aslstride_arr[] = {Aslstride0};
        int32_t M1_Atlbound_arr[]  = {M1_Atlbound0, M1_Atlbound1, M1_Atlbound2,
                                      1, 1, 1};
        int32_t M1_Atlstride_arr[] = {M1_Atlstride0, M1_Atlstride1, M1_Atlstride2,
                                      0, 0, 0};

        int32_t M1_B0slstride_arr[] = {B0slstride0};
        int32_t M1_B0tlbound_arr[]  = {M1_B0tlbound0, M1_B0tlbound1,
                                       M1_B0tlbound2, M1_B0tlbound3};
        int32_t M1_B0tlstride_arr[] = {M1_B0tlstride0, M1_B0tlstride1,
                                       M1_B0tlstride2, M1_B0tlstride3};

        int32_t M1_B1slstride_arr[] = {B1slstride0};
        int32_t M1_B1tlbound_arr[]  = {M1_B1tlbound0, M1_B1tlbound1,
                                       M1_B1tlbound2, M1_B1tlbound3};
        int32_t M1_B1tlstride_arr[] = {M1_B1tlstride0, M1_B1tlstride1,
                                       M1_B1tlstride2, M1_B1tlstride3};

        int32_t M1_D0slstride_arr[] = {D0slstride0};
        int32_t M1_D0tlbound_arr[]  = {M1_Dtlbound0, M1_Dtlbound1,
                                       M1_Dtlbound2, M1_Dtlbound3};
        int32_t M1_D0tlstride_arr[] = {M1_Dtlstride0, M1_Dtlstride1,
                                       M1_Dtlstride2, M1_Dtlstride3};

        int32_t M1_D1slstride_arr[] = {D1slstride0};
        int32_t M1_D1tlbound_arr[]  = {M1_Dtlbound0, M1_Dtlbound1,
                                       M1_Dtlbound2, M1_Dtlbound3};
        int32_t M1_D1tlstride_arr[] = {M1_Dtlstride0, M1_Dtlstride1,
                                       M1_Dtlstride2, M1_Dtlstride3};

        uint32_t m1_start = snrt_mcycle();

        set_dual_versacore_streamer_csr(
            delta_local_a1, M1_Aslstride_arr, M1_Atlbound_arr, M1_Atlstride_arr,
            set_addr_remap_index_A, channel_en_A,
            delta_local_w2l, M1_B0slstride_arr, M1_B0tlbound_arr, M1_B0tlstride_arr,
            set_addr_remap_index_B0, channel_en_B0,
            delta_local_w2r, M1_B1slstride_arr, M1_B1tlbound_arr, M1_B1tlstride_arr,
            set_addr_remap_index_B1, channel_en_B1,
            delta_local_mode1_d0, M1_D0slstride_arr, M1_D0tlbound_arr, M1_D0tlstride_arr,
            set_addr_remap_index_D0, channel_en_D0,
            delta_local_mode1_d1, M1_D1slstride_arr, M1_D1tlbound_arr, M1_D1tlstride_arr,
            set_addr_remap_index_D1, channel_en_D1);

        set_dual_versacore_csr(1, K1, N1 * M1, subtraction_setting,
                               array_shape, data_type);
        set_dual_versacore_mode(1);
        set_dual_versacore_rescale0(rescale_input_zp, rescale_multiplier,
                                    rescale_output_zp, rescale_shift);
        set_dual_versacore_rescale1(rescale_input_zp, rescale_multiplier,
                                    rescale_output_zp, rescale_shift);
        set_dual_versacore_rescale_mul(rescale_input_zp, rescale_multiplier,
                                       rescale_output_zp, rescale_shift);

        set_dual_versacore_streamer_start();
        set_dual_versacore_start();
        wait_dual_versacore();
        wait_dual_versacore_writer();

        uint32_t m1_end = snrt_mcycle();

        int err_d0 = check_dual_versacore_result_i16(
            local_mode1_d0, (int16_t *)mode1_golden_d0_padded, mode1_output_elems_padded);
        int err_d1 = check_dual_versacore_result_i16(
            local_mode1_d1, (int16_t *)mode1_golden_d1_padded, mode1_output_elems_padded);
        err += err_d0 + err_d1;

        int32_t cycles_m1 = read_dual_versacore_perf_counter();
        int32_t str_cycles_m1 = read_dual_versacore_streamer_perf_counter();
        printf("Mode 1 GEMM D0: %s, Error: %d\n",
               err_d0 ? "FAIL" : "PASS", err_d0);
        printf("Mode 1 GEMM D1: %s, Error: %d\n",
               err_d1 ? "FAIL" : "PASS", err_d1);
        printf("  M1 Cycles: accel=%d, streamer=%d, wall=%u\n",
               cycles_m1, str_cycles_m1, m1_end - m1_start);
    }

    return err;
}
