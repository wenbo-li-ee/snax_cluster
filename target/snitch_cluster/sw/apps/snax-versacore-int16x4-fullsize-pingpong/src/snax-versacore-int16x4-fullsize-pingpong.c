// Int16x4 scaled 1/16 ping-pong (B tile double-buffering) test
// Mode 0 (SwiGLU): A[8,128] @ W[128,88], V[128,88] → Output0[8,88]
// Mode 1 (GEMM): A'=Output0 @ W2_left[88,64], W2_right[88,64]
//
// Double-buffering strategy: split N dimension into chunks.
// Two B0/B1 buffers alternate. After wait_dual_versacore() (accel done,
// B reader idle), start DMA for next B chunk into alternate buffer.
// Then wait_dual_versacore_writer() before starting next compute.

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

int main() {
    int err = 0;

    // A: single buffer (shared across chunks)
    int16_t *local_a = (int16_t *)(snrt_l1_next() + delta_local_a);

    // B0: ping-pong buffers (each holds N_chunk * K padded tiles)
    uint8_t *local_b0_buf[2];
    local_b0_buf[0] = (uint8_t *)(snrt_l1_next() + delta_local_b0_buf0);
    local_b0_buf[1] = (uint8_t *)(snrt_l1_next() + delta_local_b0_buf1);

    // B1: ping-pong buffers
    uint8_t *local_b1_buf[2];
    local_b1_buf[0] = (uint8_t *)(snrt_l1_next() + delta_local_b1_buf0);
    local_b1_buf[1] = (uint8_t *)(snrt_l1_next() + delta_local_b1_buf1);

    // D0, D1: full output buffers
    int16_t *local_d0 = (int16_t *)(snrt_l1_next() + delta_local_d0);
    int16_t *local_d1_mode0 = (int16_t *)(snrt_l1_next() + delta_local_d1_mode0);

    // Mode 1 buffers
    uint8_t *local_w2l_buf[2];
    local_w2l_buf[0] = (uint8_t *)(snrt_l1_next() + delta_local_w2l_buf0);
    local_w2l_buf[1] = (uint8_t *)(snrt_l1_next() + delta_local_w2l_buf1);

    uint8_t *local_w2r_buf[2];
    local_w2r_buf[0] = (uint8_t *)(snrt_l1_next() + delta_local_w2r_buf0);
    local_w2r_buf[1] = (uint8_t *)(snrt_l1_next() + delta_local_w2r_buf1);

    int16_t *local_mode1_d0 = (int16_t *)(snrt_l1_next() + delta_local_mode1_d0);
    int16_t *local_mode1_d1 = (int16_t *)(snrt_l1_next() + delta_local_mode1_d1);

    // ================================================================
    // DMA: preload A + first B chunk (chunk 0) for Mode 0
    // ================================================================
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, A, a_data_length);
        // First B chunk: N_chunk * K tiles from offset 0
        snrt_dma_start_1d(local_b0_buf[0], W, b_chunk_data_length);
        snrt_dma_start_1d(local_b1_buf[0], V, b_chunk_data_length);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    // ================================================================
    // Mode 0 (SwiGLU) with B tile double-buffering along N
    // ================================================================
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

        int32_t D0slstride_arr[] = {D0slstride0};
        int32_t D0tlbound_arr[]  = {Dtlbound0, Dtlbound1, Dtlbound2, 1};
        int32_t D0tlstride_arr[] = {Dtlstride0, Dtlstride1, Dtlstride2, 0};

        int32_t D1slstride_arr[] = {D1slstride0};
        int32_t D1tlbound_arr[]  = {Dtlbound0, Dtlbound1, Dtlbound2, 1};
        int32_t D1tlstride_arr[] = {Dtlstride0, Dtlstride1, Dtlstride2, 0};

        for (uint32_t c = 0; c < num_chunks; c++) {
            int cur = c % 2;
            int32_t b0_base = (uint32_t)((uintptr_t)local_b0_buf[cur] -
                                          (uintptr_t)snrt_l1_next());
            int32_t b1_base = (uint32_t)((uintptr_t)local_b1_buf[cur] -
                                          (uintptr_t)snrt_l1_next());
            int32_t d0_base = delta_local_d0 + c * d_chunk_bytes;
            int32_t d1_base = delta_local_d1_mode0 + c * d_chunk_bytes;

            set_dual_versacore_streamer_csr(
                delta_local_a, Aslstride_arr, Atlbound_arr, Atlstride_arr,
                set_addr_remap_index_A, channel_en_A,
                b0_base, B0slstride_arr, B0tlbound_arr, B0tlstride_arr,
                set_addr_remap_index_B0, channel_en_B0,
                b1_base, B1slstride_arr, B1tlbound_arr, B1tlstride_arr,
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

            // Signal DMA core: ready for next chunk load
            snrt_cluster_hw_barrier();

            set_dual_versacore_streamer_start();
            set_dual_versacore_start();
            wait_dual_versacore();

            // Accel done → B reader idle → DMA can load next B into alt buffer
            // (DMA core handles this after barrier above)
            // Now wait for writer to finish before reusing D buffer
            wait_dual_versacore_writer();

            // Wait for DMA to finish loading next chunk
            snrt_cluster_hw_barrier();
        }

        err += check_dual_versacore_result_i16(
            local_d0, (int16_t *)mode0_golden, mode0_output_elems);

        int32_t cycles_m0 = read_dual_versacore_perf_counter();
        int32_t str_cycles_m0 = read_dual_versacore_streamer_perf_counter();
        printf("Mode 0 SwiGLU: %s, Error: %d\n", err ? "FAIL" : "PASS", err);
        printf("  M0 Cycles: accel=%d, streamer=%d\n", cycles_m0, str_cycles_m0);
    } else if (snrt_is_dm_core()) {
        for (uint32_t c = 0; c < num_chunks; c++) {
            // Wait for compute core to start current chunk
            snrt_cluster_hw_barrier();

            // Load next B chunk into alternate buffer
            if (c + 1 < num_chunks) {
                int nxt = (c + 1) % 2;
                uint32_t b_src_offset = (c + 1) * b_chunk_data_length;
                snrt_dma_start_1d(local_b0_buf[nxt],
                                  (uint8_t *)W + b_src_offset,
                                  b_chunk_data_length);
                snrt_dma_start_1d(local_b1_buf[nxt],
                                  (uint8_t *)V + b_src_offset,
                                  b_chunk_data_length);
                snrt_dma_wait_all();
            }

            // Signal compute core: next chunk ready
            snrt_cluster_hw_barrier();
        }
    } else {
        for (uint32_t c = 0; c < num_chunks; c++) {
            snrt_cluster_hw_barrier();
            snrt_cluster_hw_barrier();
        }
    }

    snrt_cluster_hw_barrier();

    // ================================================================
    // Mode 1 (GEMM) with B tile double-buffering along N1
    // ================================================================
    // Preload first W2 chunk
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_w2l_buf[0], W2_left, m1_b_chunk_data_length);
        snrt_dma_start_1d(local_w2r_buf[0], W2_right, m1_b_chunk_data_length);
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

        for (uint32_t c = 0; c < num_chunks1; c++) {
            int cur = c % 2;
            int32_t w2l_base = (uint32_t)((uintptr_t)local_w2l_buf[cur] -
                                           (uintptr_t)snrt_l1_next());
            int32_t w2r_base = (uint32_t)((uintptr_t)local_w2r_buf[cur] -
                                           (uintptr_t)snrt_l1_next());
            int32_t md0_base = delta_local_mode1_d0 + c * m1_d_chunk_bytes;
            int32_t md1_base = delta_local_mode1_d1 + c * m1_d_chunk_bytes;

            set_dual_versacore_streamer_csr(
                delta_local_d0, M1_Aslstride_arr, M1_Atlbound_arr, M1_Atlstride_arr,
                set_addr_remap_index_A, channel_en_A,
                w2l_base, M1_B0slstride_arr, M1_B0tlbound_arr, M1_B0tlstride_arr,
                set_addr_remap_index_B0, channel_en_B0,
                w2r_base, M1_B1slstride_arr, M1_B1tlbound_arr, M1_B1tlstride_arr,
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

            snrt_cluster_hw_barrier();

            set_dual_versacore_streamer_start();
            set_dual_versacore_start();
            wait_dual_versacore();
            wait_dual_versacore_writer();

            snrt_cluster_hw_barrier();
        }

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
    } else if (snrt_is_dm_core()) {
        for (uint32_t c = 0; c < num_chunks1; c++) {
            snrt_cluster_hw_barrier();
            if (c + 1 < num_chunks1) {
                int nxt = (c + 1) % 2;
                uint32_t b_src_offset = (c + 1) * m1_b_chunk_data_length;
                snrt_dma_start_1d(local_w2l_buf[nxt],
                                  (uint8_t *)W2_left + b_src_offset,
                                  m1_b_chunk_data_length);
                snrt_dma_start_1d(local_w2r_buf[nxt],
                                  (uint8_t *)W2_right + b_src_offset,
                                  m1_b_chunk_data_length);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();
        }
    } else {
        for (uint32_t c = 0; c < num_chunks1; c++) {
            snrt_cluster_hw_barrier();
            snrt_cluster_hw_barrier();
        }
    }

    return err;
}
