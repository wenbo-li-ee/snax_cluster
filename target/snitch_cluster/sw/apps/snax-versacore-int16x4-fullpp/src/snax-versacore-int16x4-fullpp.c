// Int16x4 ping-pong (double-buffered) test — Mode 0 (SwiGLU) + Mode 1 (GEMM)
// M=2, K=2, N=2 multi-tile workload with M-dimension double-buffering:
// DMA core loads A[m+1] while compute core runs tile m.

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

int main() {
    int err = 0;

    // Two A buffers for double-buffering along M dimension
    // Each A buffer holds K tiles: K * meshRow * tileSize * 2 bytes
    int32_t a_tile_bytes = (int32_t)(K * meshRow * tileSize * 2);  // int16
    int32_t b_tile_padded_bytes = b0_data_length / K;  // per-K-N tile (already padded)

    // Mode 0 buffers: 2x A (ping-pong), 1x B0, 1x B1, 1x D0, 1x D1
    int16_t *local_a_buf[2];
    uint8_t *local_b0, *local_b1;
    int16_t *local_d0, *local_d1_mode0;

    local_a_buf[0] = (int16_t *)(snrt_l1_next() + delta_local_a);
    // Second A buffer starts after first A buffer
    // (delta_local_b0 - delta_local_a gives A allocation, so buf1 = halfway)
    local_a_buf[1] = (int16_t *)(snrt_l1_next() + delta_local_a + a_tile_bytes);
    local_b0 = (uint8_t *)(snrt_l1_next() + delta_local_b0);
    local_b1 = (uint8_t *)(snrt_l1_next() + delta_local_b1);
    local_d0 = (int16_t *)(snrt_l1_next() + delta_local_d0);
    local_d1_mode0 = (int16_t *)(snrt_l1_next() + delta_local_d1_mode0);

    // Mode 1 buffers
    int16_t *local_a1_buf[2];
    uint8_t *local_w2l, *local_w2r;
    int16_t *local_mode1_d0, *local_mode1_d1;

    local_a1_buf[0] = (int16_t *)(snrt_l1_next() + delta_local_a1);
    local_a1_buf[1] = (int16_t *)(snrt_l1_next() + delta_local_a1 +
                        (int32_t)(K1 * meshRow * tileSize * 2));
    local_w2l = (uint8_t *)(snrt_l1_next() + delta_local_w2l);
    local_w2r = (uint8_t *)(snrt_l1_next() + delta_local_w2r);
    local_mode1_d0 = (int16_t *)(snrt_l1_next() + delta_local_mode1_d0);
    local_mode1_d1 = (int16_t *)(snrt_l1_next() + delta_local_mode1_d1);

    // Per-M-tile output sizes
    int32_t d_tile_bytes = (int32_t)(N * meshRow * meshCol * 2);  // int16

    // ================================================================
    // Mode 0 (SwiGLU) with M-dimension double-buffering
    // ================================================================

    // DMA: preload B data (shared across all M tiles) + first A tile
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_b0, W, b0_data_length);
        snrt_dma_start_1d(local_b1, V, b1_data_length);
        snrt_dma_start_1d(local_a_buf[0], A, a_tile_bytes);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        uint32_t subtraction_setting =
            gen_dual_vc_subtraction_config(subtraction_a, subtraction_b);

        // Streamer arrays for per-tile config (M=1 per invocation)
        int32_t Aslstride_arr[] = {Aslstride0};
        int32_t Atlbound_arr[]  = {Atlbound0, Atlbound1, 1, 1, 1, 1};  // M=1
        int32_t Atlstride_arr[] = {Atlstride0, Atlstride1, 0, 0, 0, 0};

        int32_t B0slstride_arr[] = {B0slstride0};
        int32_t B0tlbound_arr[]  = {B0tlbound0, B0tlbound1, 1, B0tlbound3};
        int32_t B0tlstride_arr[] = {B0tlstride0, B0tlstride1, 0, B0tlstride3};

        int32_t B1slstride_arr[] = {B1slstride0};
        int32_t B1tlbound_arr[]  = {B1tlbound0, B1tlbound1, 1, B1tlbound3};
        int32_t B1tlstride_arr[] = {B1tlstride0, B1tlstride1, 0, B1tlstride3};

        int32_t D0slstride_arr[] = {D0slstride0};
        int32_t D0tlbound_arr[]  = {Dtlbound0, Dtlbound1, 1, Dtlbound3};
        int32_t D0tlstride_arr[] = {Dtlstride0, Dtlstride1, 0, Dtlstride3};

        int32_t D1slstride_arr[] = {D1slstride0};
        int32_t D1tlbound_arr[]  = {Dtlbound0, Dtlbound1, 1, Dtlbound3};
        int32_t D1tlstride_arr[] = {Dtlstride0, Dtlstride1, 0, Dtlstride3};

        for (uint32_t m = 0; m < M; m++) {
            int cur = m % 2;
            int32_t a_base = (uint32_t)((uintptr_t)local_a_buf[cur] -
                                         (uintptr_t)snrt_l1_next());
            int32_t d0_base = delta_local_d0 + m * d_tile_bytes;
            int32_t d1_base = delta_local_d1_mode0 + m * d_tile_bytes;

            set_dual_versacore_streamer_csr(
                a_base, Aslstride_arr, Atlbound_arr, Atlstride_arr,
                set_addr_remap_index_A, channel_en_A,
                delta_local_b0, B0slstride_arr, B0tlbound_arr, B0tlstride_arr,
                set_addr_remap_index_B0, channel_en_B0,
                delta_local_b1, B1slstride_arr, B1tlbound_arr, B1tlstride_arr,
                set_addr_remap_index_B1, channel_en_B1,
                d0_base, D0slstride_arr, D0tlbound_arr, D0tlstride_arr,
                set_addr_remap_index_D0, channel_en_D0,
                d1_base, D1slstride_arr, D1tlbound_arr, D1tlstride_arr,
                set_addr_remap_index_D1, channel_en_D1);

            set_dual_versacore_csr(1, K, N * 1, subtraction_setting,
                                   array_shape, data_type);
            set_dual_versacore_mode(0);
            set_dual_versacore_rescale0(rescale_input_zp, rescale_multiplier,
                                        rescale_output_zp, rescale_shift);
            set_dual_versacore_rescale1(rescale_input_zp, rescale_multiplier,
                                        rescale_output_zp, rescale_shift);
            set_dual_versacore_rescale_mul(rescale_input_zp, rescale_multiplier,
                                           rescale_output_zp, rescale_shift);

            // Signal DMA core to start loading next A tile
            snrt_cluster_hw_barrier();

            set_dual_versacore_streamer_start();
            set_dual_versacore_start();
            wait_dual_versacore();
            wait_dual_versacore_writer();

            // Wait for DMA to finish loading next tile before reusing buffer
            snrt_cluster_hw_barrier();
        }

        err += check_dual_versacore_result_i16(
            local_d0, (int16_t *)mode0_golden, mode0_output_elems);

        int32_t cycles_m0 = read_dual_versacore_perf_counter();
        int32_t str_cycles_m0 = read_dual_versacore_streamer_perf_counter();
        printf("Mode 0 SwiGLU: %s, Error: %d\n", err ? "FAIL" : "PASS", err);
        printf("  Cycles: accel=%d, streamer=%d\n", cycles_m0, str_cycles_m0);
    } else if (snrt_is_dm_core()) {
        // DMA core: load next A tiles in ping-pong fashion
        for (uint32_t m = 0; m < M; m++) {
            // Wait for compute core to signal it's ready
            snrt_cluster_hw_barrier();

            // Load next A tile into alternate buffer (if not last tile)
            if (m + 1 < M) {
                int nxt = (m + 1) % 2;
                snrt_dma_start_1d(local_a_buf[nxt],
                                  A + (m + 1) * a_tile_bytes / 2,  // int16 offset
                                  a_tile_bytes);
                snrt_dma_wait_all();
            }

            // Signal compute core that next tile is ready
            snrt_cluster_hw_barrier();
        }
    } else {
        // Other cores: just participate in barriers
        for (uint32_t m = 0; m < M; m++) {
            snrt_cluster_hw_barrier();
            snrt_cluster_hw_barrier();
        }
    }

    snrt_cluster_hw_barrier();

    // ================================================================
    // Mode 1 (GEMM) with M-dimension double-buffering
    // ================================================================
    int32_t a1_tile_bytes = (int32_t)(K1 * meshRow * tileSize * 2);
    int32_t d1_tile_bytes = (int32_t)(N1 * meshRow * meshCol * 2);

    // DMA: preload B data + first A1 tile
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_w2l, W2_left, w2l_data_length);
        snrt_dma_start_1d(local_w2r, W2_right, w2r_data_length);
        snrt_dma_start_1d(local_a1_buf[0], A1, a1_tile_bytes);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        uint32_t subtraction_setting =
            gen_dual_vc_subtraction_config(subtraction_a, subtraction_b);

        int32_t M1_Aslstride_arr[] = {Aslstride0};
        int32_t M1_Atlbound_arr[]  = {M1_Atlbound0, M1_Atlbound1, 1, 1, 1, 1};
        int32_t M1_Atlstride_arr[] = {M1_Atlstride0, M1_Atlstride1, 0, 0, 0, 0};

        int32_t M1_B0slstride_arr[] = {B0slstride0};
        int32_t M1_B0tlbound_arr[]  = {M1_B0tlbound0, M1_B0tlbound1,
                                       1, M1_B0tlbound3};
        int32_t M1_B0tlstride_arr[] = {M1_B0tlstride0, M1_B0tlstride1,
                                       0, M1_B0tlstride3};

        int32_t M1_B1slstride_arr[] = {B1slstride0};
        int32_t M1_B1tlbound_arr[]  = {M1_B1tlbound0, M1_B1tlbound1,
                                       1, M1_B1tlbound3};
        int32_t M1_B1tlstride_arr[] = {M1_B1tlstride0, M1_B1tlstride1,
                                       0, M1_B1tlstride3};

        int32_t M1_D0slstride_arr[] = {D0slstride0};
        int32_t M1_D0tlbound_arr[]  = {M1_Dtlbound0, M1_Dtlbound1, 1, M1_Dtlbound3};
        int32_t M1_D0tlstride_arr[] = {M1_Dtlstride0, M1_Dtlstride1, 0, M1_Dtlstride3};

        int32_t M1_D1slstride_arr[] = {D1slstride0};
        int32_t M1_D1tlbound_arr[]  = {M1_Dtlbound0, M1_Dtlbound1, 1, M1_Dtlbound3};
        int32_t M1_D1tlstride_arr[] = {M1_Dtlstride0, M1_Dtlstride1, 0, M1_Dtlstride3};

        for (uint32_t m = 0; m < M1; m++) {
            int cur = m % 2;
            int32_t a1_base = (uint32_t)((uintptr_t)local_a1_buf[cur] -
                                          (uintptr_t)snrt_l1_next());
            int32_t md0_base = delta_local_mode1_d0 + m * d1_tile_bytes;
            int32_t md1_base = delta_local_mode1_d1 + m * d1_tile_bytes;

            set_dual_versacore_streamer_csr(
                a1_base, M1_Aslstride_arr, M1_Atlbound_arr, M1_Atlstride_arr,
                set_addr_remap_index_A, channel_en_A,
                delta_local_w2l, M1_B0slstride_arr, M1_B0tlbound_arr,
                M1_B0tlstride_arr,
                set_addr_remap_index_B0, channel_en_B0,
                delta_local_w2r, M1_B1slstride_arr, M1_B1tlbound_arr,
                M1_B1tlstride_arr,
                set_addr_remap_index_B1, channel_en_B1,
                md0_base, M1_D0slstride_arr, M1_D0tlbound_arr, M1_D0tlstride_arr,
                set_addr_remap_index_D0, channel_en_D0,
                md1_base, M1_D1slstride_arr, M1_D1tlbound_arr, M1_D1tlstride_arr,
                set_addr_remap_index_D1, channel_en_D1);

            set_dual_versacore_csr(1, K1, N1 * 1, subtraction_setting,
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
        printf("  Cycles: accel=%d, streamer=%d\n", cycles_m1, str_cycles_m1);
    } else if (snrt_is_dm_core()) {
        for (uint32_t m = 0; m < M1; m++) {
            snrt_cluster_hw_barrier();
            if (m + 1 < M1) {
                int nxt = (m + 1) % 2;
                snrt_dma_start_1d(local_a1_buf[nxt],
                                  A1 + (m + 1) * a1_tile_bytes / 2,
                                  a1_tile_bytes);
                snrt_dma_wait_all();
            }
            snrt_cluster_hw_barrier();
        }
    } else {
        for (uint32_t m = 0; m < M1; m++) {
            snrt_cluster_hw_barrier();
            snrt_cluster_hw_barrier();
        }
    }

    return err;
}
