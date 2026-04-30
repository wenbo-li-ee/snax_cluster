// Multi-shape S6 16x2 test for dual VersaCore int16x4 SwiGLU.

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

#define WAIT_TIMEOUT_CYCLES 50000000u
#define WAIT_TIMEOUT_ITERS 100000000u

static void copy_i32(int32_t *dst, const int32_t *src, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

static int wait_accel_diag(int shape, int mode) {
    csrw_ss(DUAL_VC_START, 0);
    csrw_ss(DUAL_VC_START, 0);
    uint32_t start = snrt_mcycle();
    while (csrr_ss(DUAL_VC_BUSY)) {
        if ((uint32_t)(snrt_mcycle() - start) > WAIT_TIMEOUT_CYCLES) {
            printf("S%d Mode%d accelerator wait timeout: busy=%u\n", shape,
                   mode, csrr_ss(DUAL_VC_BUSY));
            return 1;
        }
    }
    return 0;
}

static int wait_writer_diag(int shape, int mode) {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    uint32_t start = snrt_mcycle();
    while (csrr_ss(STREAMER_BUSY_CSR)) {
        if ((uint32_t)(snrt_mcycle() - start) > WAIT_TIMEOUT_CYCLES) {
            printf("S%d Mode%d writer wait timeout: streamer=%u w0=%u w1=%u\n",
                   shape, mode, csrr_ss(STREAMER_BUSY_CSR),
                   csrr_ss(WRITER_BUSY_CSR), csrr_ss(WRITER1_BUSY_CSR));
            return 1;
        }
    }
    return 0;
}

static int run_shape(const shape_cfg_t *cfg) {
    int err = 0;

    int16_t *local_d0 = (int16_t *)(snrt_l1_next() + cfg->delta_local_d0);
    int16_t *local_d1_mode0 =
        (int16_t *)(snrt_l1_next() + cfg->delta_local_d1_mode0);
    int16_t *local_mode1_d0 =
        (int16_t *)(snrt_l1_next() + cfg->delta_local_mode1_d0);
    int16_t *local_mode1_d1 =
        (int16_t *)(snrt_l1_next() + cfg->delta_local_mode1_d1);

    uint32_t subtraction_setting =
        gen_dual_vc_subtraction_config(SUBTRACTION_A, SUBTRACTION_B);

    int32_t A_sstride[4];
    int32_t A_tbound[6];
    int32_t A_tstride[6];
    int32_t B0_sstride[3];
    int32_t B0_tbound[4];
    int32_t B0_tstride[4];
    int32_t B1_sstride[3];
    int32_t B1_tbound[4];
    int32_t B1_tstride[4];
    int32_t D0_sstride[1];
    int32_t D0_tbound[4];
    int32_t D0_tstride[4];
    int32_t D1_sstride[1];
    int32_t D1_tbound[4];
    int32_t D1_tstride[4];
    int32_t A_channel_en[1];
    int32_t B_channel_en[1];
    int32_t D_channel_en[1];

    copy_i32(B0_sstride, cfg->B_sstride, 3);
    copy_i32(B1_sstride, cfg->B_sstride, 3);
    copy_i32(D0_sstride, cfg->D_sstride, 1);
    copy_i32(D1_sstride, cfg->D_sstride, 1);
    copy_i32(A_channel_en, cfg->A_channel_en, 1);
    copy_i32(B_channel_en, cfg->B_channel_en, 1);
    copy_i32(D_channel_en, cfg->D_channel_en, 1);

    printf("S%d start\n", cfg->array_shape);

    // Mode 0: per-token-contiguous A, fixed S0-tiled B readers.
    copy_i32(A_sstride, cfg->mode0_A_sstride, 4);
    copy_i32(A_tbound, cfg->mode0_A_tbound, 6);
    copy_i32(A_tstride, cfg->mode0_A_tstride, 6);
    copy_i32(B0_tbound, cfg->mode0_B_tbound, 4);
    copy_i32(B0_tstride, cfg->mode0_B_tstride, 4);
    copy_i32(B1_tbound, cfg->mode0_B_tbound, 4);
    copy_i32(B1_tstride, cfg->mode0_B_tstride, 4);
    copy_i32(D0_tbound, cfg->mode0_D_tbound, 4);
    copy_i32(D0_tstride, cfg->mode0_D_tstride, 4);
    copy_i32(D1_tbound, cfg->mode0_D_tbound, 4);
    copy_i32(D1_tstride, cfg->mode0_D_tstride, 4);

    printf("S%d Mode0 configure streamer\n", cfg->array_shape);
    uint32_t m0_start = snrt_mcycle();
    set_dual_versacore_streamer_csr(
        cfg->delta_local_a, A_sstride, A_tbound, A_tstride,
        SET_ADDR_REMAP_INDEX_A, A_channel_en,
        cfg->delta_local_b0, B0_sstride, B0_tbound, B0_tstride,
        SET_ADDR_REMAP_INDEX_B0, B_channel_en,
        cfg->delta_local_b1, B1_sstride, B1_tbound, B1_tstride,
        SET_ADDR_REMAP_INDEX_B1, B_channel_en,
        cfg->delta_local_d0, D0_sstride, D0_tbound, D0_tstride,
        SET_ADDR_REMAP_INDEX_D0, D_channel_en,
        cfg->delta_local_d1_mode0, D1_sstride, D1_tbound, D1_tstride,
        SET_ADDR_REMAP_INDEX_D1, D_channel_en);

    printf("S%d Mode0 configure core\n", cfg->array_shape);
    set_dual_versacore_csr(1, cfg->K_tiles, cfg->N_tiles * cfg->M_tiles,
                           subtraction_setting, cfg->array_shape, DATA_TYPE);
    set_dual_versacore_mode(0);
    set_dual_versacore_rescale0(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale1(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale_mul(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                   RESCALE_OUTPUT_ZP, RESCALE_SHIFT);

    printf("S%d Mode0 start\n", cfg->array_shape);
    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    printf("S%d Mode0 wait accel\n", cfg->array_shape);
    if (wait_accel_diag(cfg->array_shape, 0)) {
        return 1000 + cfg->array_shape;
    }
    printf("S%d Mode0 wait writer\n", cfg->array_shape);
    if (wait_writer_diag(cfg->array_shape, 0)) {
        return 2000 + cfg->array_shape;
    }
    uint32_t m0_end = snrt_mcycle();

    int err_m0_d0 = check_dual_versacore_result_i16(
        local_d0, (int16_t *)cfg->mode0_d0_golden, cfg->mode0_output_elems);
    int err_m0_d1 = check_dual_versacore_result_i16(
        local_d1_mode0, (int16_t *)cfg->mode0_d1_golden,
        cfg->mode0_output_elems);
    err += err_m0_d0 + err_m0_d1;

    int32_t cycles_m0 = read_dual_versacore_perf_counter();
    int32_t str_cycles_m0 = read_dual_versacore_streamer_perf_counter();
    printf("S%d Mode0 D0: %s, Error: %d\n", cfg->array_shape,
           err_m0_d0 ? "FAIL" : "PASS", err_m0_d0);
    printf("S%d Mode0 D1: %s, Error: %d\n", cfg->array_shape,
           err_m0_d1 ? "FAIL" : "PASS", err_m0_d1);
    printf("S%d Mode0 Cycles: accel=%d, streamer=%d, wall=%u\n",
           cfg->array_shape, cycles_m0, str_cycles_m0, m0_end - m0_start);

    // Mode 1: direct dynamic base pointer to the Mode 0 D0 TCDM buffer.
    copy_i32(A_sstride, cfg->mode1_A_sstride, 4);
    copy_i32(A_tbound, cfg->mode1_A_tbound, 6);
    copy_i32(A_tstride, cfg->mode1_A_tstride, 6);
    copy_i32(B0_tbound, cfg->mode1_B_tbound, 4);
    copy_i32(B0_tstride, cfg->mode1_B_tstride, 4);
    copy_i32(B1_tbound, cfg->mode1_B_tbound, 4);
    copy_i32(B1_tstride, cfg->mode1_B_tstride, 4);
    copy_i32(D0_tbound, cfg->mode1_D_tbound, 4);
    copy_i32(D0_tstride, cfg->mode1_D_tstride, 4);
    copy_i32(D1_tbound, cfg->mode1_D_tbound, 4);
    copy_i32(D1_tstride, cfg->mode1_D_tstride, 4);

    printf("S%d Mode1 configure streamer\n", cfg->array_shape);
    uint32_t m1_start = snrt_mcycle();
    set_dual_versacore_streamer_csr(
        cfg->delta_local_d0, A_sstride, A_tbound, A_tstride,
        SET_ADDR_REMAP_INDEX_A, A_channel_en,
        cfg->delta_local_w2l, B0_sstride, B0_tbound, B0_tstride,
        SET_ADDR_REMAP_INDEX_B0, B_channel_en,
        cfg->delta_local_w2r, B1_sstride, B1_tbound, B1_tstride,
        SET_ADDR_REMAP_INDEX_B1, B_channel_en,
        cfg->delta_local_mode1_d0, D0_sstride, D0_tbound, D0_tstride,
        SET_ADDR_REMAP_INDEX_D0, D_channel_en,
        cfg->delta_local_mode1_d1, D1_sstride, D1_tbound, D1_tstride,
        SET_ADDR_REMAP_INDEX_D1, D_channel_en);

    printf("S%d Mode1 configure core\n", cfg->array_shape);
    set_dual_versacore_csr(1, cfg->K1, cfg->N1 * cfg->M_tiles,
                           subtraction_setting, cfg->array_shape, DATA_TYPE);
    set_dual_versacore_mode(1);
    set_dual_versacore_rescale0(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale1(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale_mul(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                   RESCALE_OUTPUT_ZP, RESCALE_SHIFT);

    printf("S%d Mode1 start\n", cfg->array_shape);
    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    printf("S%d Mode1 wait accel\n", cfg->array_shape);
    if (wait_accel_diag(cfg->array_shape, 1)) {
        return 3000 + cfg->array_shape;
    }
    printf("S%d Mode1 wait writer\n", cfg->array_shape);
    if (wait_writer_diag(cfg->array_shape, 1)) {
        return 4000 + cfg->array_shape;
    }
    uint32_t m1_end = snrt_mcycle();

    int err_m1_d0 = check_dual_versacore_result_i16(
        local_mode1_d0, (int16_t *)cfg->mode1_d0_golden,
        cfg->mode1_output_elems);
    int err_m1_d1 = check_dual_versacore_result_i16(
        local_mode1_d1, (int16_t *)cfg->mode1_d1_golden,
        cfg->mode1_output_elems);
    err += err_m1_d0 + err_m1_d1;

    int32_t cycles_m1 = read_dual_versacore_perf_counter();
    int32_t str_cycles_m1 = read_dual_versacore_streamer_perf_counter();
    printf("S%d Mode1 D0: %s, Error: %d\n", cfg->array_shape,
           err_m1_d0 ? "FAIL" : "PASS", err_m1_d0);
    printf("S%d Mode1 D1: %s, Error: %d\n", cfg->array_shape,
           err_m1_d1 ? "FAIL" : "PASS", err_m1_d1);
    printf("S%d Mode1 Cycles: accel=%d, streamer=%d, wall=%u\n",
           cfg->array_shape, cycles_m1, str_cycles_m1, m1_end - m1_start);

    return err;
}

int main() {
    int err = 0;
    int16_t *local_a = (int16_t *)(snrt_l1_next() + shape_cfg[0].delta_local_a);
    uint8_t *local_b0 = (uint8_t *)(snrt_l1_next() + shape_cfg[0].delta_local_b0);
    uint8_t *local_b1 = (uint8_t *)(snrt_l1_next() + shape_cfg[0].delta_local_b1);
    uint8_t *local_w2l = (uint8_t *)(snrt_l1_next() + shape_cfg[0].delta_local_w2l);
    uint8_t *local_w2r = (uint8_t *)(snrt_l1_next() + shape_cfg[0].delta_local_w2r);

    if (shape_cfg[0].tcdm_end > TCDM_CAPACITY_BYTES) {
        printf("S6 TCDM placement exceeds capacity: end=%d cap=%d\n",
               shape_cfg[0].tcdm_end, TCDM_CAPACITY_BYTES);
        return 1;
    }

    if (snrt_global_core_idx() == 0) {
        printf("S6 multishape 16x2 staging shared dataset: bytes=%d\n",
               shape_cfg[0].tcdm_end);
    }

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, A, A_DATA_LENGTH);
        snrt_dma_start_1d(local_b0, W, B_DATA_LENGTH);
        snrt_dma_start_1d(local_b1, V, B_DATA_LENGTH);
        snrt_dma_start_1d(local_w2l, W2_left, W2_DATA_LENGTH);
        snrt_dma_start_1d(local_w2r, W2_right, W2_DATA_LENGTH);
        snrt_dma_wait_all();
    }

    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        printf("S6 multishape 16x2 shared dataset ready\n");
        for (int shape = 0; shape < NUM_SHAPES; shape++) {
            err += run_shape(&shape_cfg[shape]);
        }
        printf("S6 multishape 16x2 total checks: 12, total error: %d\n", err);
    }

    return err;
}
