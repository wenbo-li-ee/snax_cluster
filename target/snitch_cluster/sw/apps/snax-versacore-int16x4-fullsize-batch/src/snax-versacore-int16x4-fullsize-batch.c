// Int16x4 full-size batch test — sequential N-tiling (no double-buffer)
// Mode 0 (SwiGLU): A[8,2048] @ W[2048,1408], V[2048,1408] → Out0[8,1408]
// Mode 1 (GEMM): A'=Out0 @ W2l[1408,1024], W2r[1408,1024]
// B0/B1 buffers reused for W2l/W2r in Mode 1.

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

int main() {
    int err = 0;

    int16_t *local_a = (int16_t *)(snrt_l1_next() + delta_local_a);
    uint8_t *local_b0 = (uint8_t *)(snrt_l1_next() + delta_local_b0);
    uint8_t *local_b1 = (uint8_t *)(snrt_l1_next() + delta_local_b1);
    int16_t *local_d0 = (int16_t *)(snrt_l1_next() + delta_local_d0);
    int16_t *local_d1_mode0 = (int16_t *)(snrt_l1_next() + delta_local_d1_mode0);
    int16_t *local_mode1_d0 = (int16_t *)(snrt_l1_next() + delta_local_mode1_d0);
    int16_t *local_mode1_d1 = (int16_t *)(snrt_l1_next() + delta_local_mode1_d1);

    // DMA: load A (stays for all chunks)
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, A, a_data_length);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // ================================================================
    // Mode 0 (SwiGLU) — sequential N-chunk tiling
    // ================================================================
    for (uint32_t c = 0; c < num_chunks; c++) {
        // DMA: load B chunk c
        if (snrt_is_dm_core()) {
            uint32_t b_offset = c * b_chunk_data_length;
            snrt_dma_start_1d(local_b0, (uint8_t *)W + b_offset,
                              b_chunk_data_length);
            snrt_dma_start_1d(local_b1, (uint8_t *)V + b_offset,
                              b_chunk_data_length);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        if (snrt_global_core_idx() == 0) {
            uint32_t subtraction_setting =
                gen_dual_vc_subtraction_config(subtraction_a, subtraction_b);

            int32_t Aslstride_arr[] = {Aslstride0};
            int32_t Atlbound_arr[]  = {Atlbound0, Atlbound1, Atlbound2,
                                       1, 1, 1};
            int32_t Atlstride_arr[] = {Atlstride0, Atlstride1, Atlstride2,
                                       0, 0, 0};

            int32_t B0slstride_arr[] = {B0slstride0};
            int32_t B0tlbound_arr[]  = {B0tlbound0, B0tlbound1, B0tlbound2, 1};
            int32_t B0tlstride_arr[] = {B0tlstride0, B0tlstride1, B0tlstride2, 0};

            int32_t B1slstride_arr[] = {B1slstride0};
            int32_t B1tlbound_arr[]  = {B1tlbound0, B1tlbound1, B1tlbound2, 1};
            int32_t B1tlstride_arr[] = {B1tlstride0, B1tlstride1, B1tlstride2, 0};

            int32_t d0_base = delta_local_d0 + c * d_chunk_bytes;
            int32_t d1_base = delta_local_d1_mode0 + c * d_chunk_bytes;

            int32_t D0slstride_arr[] = {D0slstride0};
            int32_t D0tlbound_arr[]  = {Dtlbound0, Dtlbound1, Dtlbound2, 1};
            int32_t D0tlstride_arr[] = {Dtlstride0, Dtlstride1, Dtlstride2, 0};

            int32_t D1slstride_arr[] = {D1slstride0};
            int32_t D1tlbound_arr[]  = {Dtlbound0, Dtlbound1, Dtlbound2, 1};
            int32_t D1tlstride_arr[] = {Dtlstride0, Dtlstride1, Dtlstride2, 0};

            set_dual_versacore_streamer_csr(
                delta_local_a, Aslstride_arr, Atlbound_arr, Atlstride_arr,
                set_addr_remap_index_A, channel_en_A,
                delta_local_b0, B0slstride_arr, B0tlbound_arr, B0tlstride_arr,
                set_addr_remap_index_B0, channel_en_B0,
                delta_local_b1, B1slstride_arr, B1tlbound_arr, B1tlstride_arr,
                set_addr_remap_index_B1, channel_en_B1,
                d0_base, D0slstride_arr, D0tlbound_arr, D0tlstride_arr,
                set_addr_remap_index_D0, channel_en_D0,
                d1_base, D1slstride_arr, D1tlbound_arr, D1tlstride_arr,
                set_addr_remap_index_D1, channel_en_D1);

            set_dual_versacore_csr(1, K, N_chunk * M, subtraction_setting,
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
        }
        snrt_cluster_hw_barrier();
    }

    if (snrt_global_core_idx() == 0) {
        err += check_dual_versacore_result_i16(
            local_d0, (int16_t *)mode0_golden, mode0_output_elems);

        int32_t cycles_m0 = read_dual_versacore_perf_counter();
        int32_t str_cycles_m0 = read_dual_versacore_streamer_perf_counter();
        printf("Mode 0 SwiGLU: %s, Error: %d\n", err ? "FAIL" : "PASS", err);
        printf("  M0 Cycles: accel=%d, streamer=%d\n", cycles_m0, str_cycles_m0);
    }
    snrt_cluster_hw_barrier();

    // ================================================================
    // Mode 1 (GEMM) — reuse B0/B1 buffers for W2l/W2r
    // ================================================================
    for (uint32_t c = 0; c < num_chunks1; c++) {
        if (snrt_is_dm_core()) {
            uint32_t b_offset = c * m1_b_chunk_data_length;
            snrt_dma_start_1d(local_b0, (uint8_t *)W2_left + b_offset,
                              m1_b_chunk_data_length);
            snrt_dma_start_1d(local_b1, (uint8_t *)W2_right + b_offset,
                              m1_b_chunk_data_length);
            snrt_dma_wait_all();
        }
        snrt_cluster_hw_barrier();

        if (snrt_global_core_idx() == 0) {
            uint32_t subtraction_setting =
                gen_dual_vc_subtraction_config(subtraction_a, subtraction_b);

            int32_t M1_Aslstride_arr[] = {Aslstride0};
            int32_t M1_Atlbound_arr[]  = {M1_Atlbound0, M1_Atlbound1, M1_Atlbound2,
                                          1, 1, 1};
            int32_t M1_Atlstride_arr[] = {M1_Atlstride0, M1_Atlstride1, M1_Atlstride2,
                                          0, 0, 0};

            int32_t M1_B0slstride_arr[] = {B0slstride0};
            int32_t M1_B0tlbound_arr[]  = {M1_B0tlbound0, M1_B0tlbound1,
                                           M1_B0tlbound2, 1};
            int32_t M1_B0tlstride_arr[] = {M1_B0tlstride0, M1_B0tlstride1,
                                           M1_B0tlstride2, 0};

            int32_t M1_B1slstride_arr[] = {B1slstride0};
            int32_t M1_B1tlbound_arr[]  = {M1_B1tlbound0, M1_B1tlbound1,
                                           M1_B1tlbound2, 1};
            int32_t M1_B1tlstride_arr[] = {M1_B1tlstride0, M1_B1tlstride1,
                                           M1_B1tlstride2, 0};

            int32_t md0_base = delta_local_mode1_d0 + c * m1_d_chunk_bytes;
            int32_t md1_base = delta_local_mode1_d1 + c * m1_d_chunk_bytes;

            int32_t M1_D0slstride_arr[] = {D0slstride0};
            int32_t M1_D0tlbound_arr[]  = {M1_Dtlbound0, M1_Dtlbound1,
                                           M1_Dtlbound2, 1};
            int32_t M1_D0tlstride_arr[] = {M1_Dtlstride0, M1_Dtlstride1,
                                           M1_Dtlstride2, 0};

            int32_t M1_D1slstride_arr[] = {D1slstride0};
            int32_t M1_D1tlbound_arr[]  = {M1_Dtlbound0, M1_Dtlbound1,
                                           M1_Dtlbound2, 1};
            int32_t M1_D1tlstride_arr[] = {M1_Dtlstride0, M1_Dtlstride1,
                                           M1_Dtlstride2, 0};

            set_dual_versacore_streamer_csr(
                delta_local_d0, M1_Aslstride_arr, M1_Atlbound_arr, M1_Atlstride_arr,
                set_addr_remap_index_A, channel_en_A,
                delta_local_b0, M1_B0slstride_arr, M1_B0tlbound_arr, M1_B0tlstride_arr,
                set_addr_remap_index_B0, channel_en_B0,
                delta_local_b1, M1_B1slstride_arr, M1_B1tlbound_arr, M1_B1tlstride_arr,
                set_addr_remap_index_B1, channel_en_B1,
                md0_base, M1_D0slstride_arr, M1_D0tlbound_arr, M1_D0tlstride_arr,
                set_addr_remap_index_D0, channel_en_D0,
                md1_base, M1_D1slstride_arr, M1_D1tlbound_arr, M1_D1tlstride_arr,
                set_addr_remap_index_D1, channel_en_D1);

            set_dual_versacore_csr(1, K1, N1_chunk * M1, subtraction_setting,
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
        }
        snrt_cluster_hw_barrier();
    }

    if (snrt_global_core_idx() == 0) {
        int err_d0 = check_dual_versacore_result_i16(
            local_mode1_d0, (int16_t *)mode1_golden_d0, mode1_output_elems);
        int err_d1 = check_dual_versacore_result_i16(
            local_mode1_d1, (int16_t *)mode1_golden_d1, mode1_output_elems);
        err += err_d0 + err_d1;

        int32_t cycles_m1 = read_dual_versacore_perf_counter();
        int32_t str_cycles_m1 = read_dual_versacore_streamer_perf_counter();
        printf("Mode 1 GEMM D0: %s, Error: %d\n",
               err_d0 ? "FAIL" : "PASS", err_d0);
        printf("Mode 1 GEMM D1: %s, Error: %d\n",
               err_d1 ? "FAIL" : "PASS", err_d1);
        printf("  M1 Cycles: accel=%d, streamer=%d\n", cycles_m1, str_cycles_m1);
    }

    return err;
}
