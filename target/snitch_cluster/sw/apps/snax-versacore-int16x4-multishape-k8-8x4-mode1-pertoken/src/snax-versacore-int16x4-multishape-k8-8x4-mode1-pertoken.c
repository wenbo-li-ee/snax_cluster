// Multi-shape K=8 8x4 Mode1 per-token output layout test for dual VersaCore int16x4 SwiGLU.

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

#define WAIT_TIMEOUT_CYCLES 300000u
#define CHECK_PRINT_LIMIT 16u

static int check_result_i16_limited(const int16_t *output,
                                    const int16_t *output_golden,
                                    int32_t num_elements) {
    uint32_t err = 0;
    for (int i = 0; i < num_elements; i++) {
        if (output[i] != output_golden[i]) {
            if (err < CHECK_PRINT_LIMIT) {
                printf("Unequals. output[%d] = %d, output_golden[%d] = %d\n",
                       i, output[i], i, output_golden[i]);
            }
            err++;
        }
    }
    if (err > CHECK_PRINT_LIMIT) {
        printf("Mismatch print capped at %u of %u errors\n",
               CHECK_PRINT_LIMIT, err);
    }
    return err;
}

static int wait_accel_diag(int shape, int mode) {
    csrw_ss(DUAL_VC_START, 0);
    csrw_ss(DUAL_VC_START, 0);
    uint32_t start = snrt_mcycle();
    while (csrr_ss(DUAL_VC_BUSY)) {
        if ((uint32_t)(snrt_mcycle() - start) > WAIT_TIMEOUT_CYCLES) {
            printf("S%d Mode%d accelerator wait timeout: accel=%u streamer=%u w0=%u w1=%u\n",
                   shape, mode, csrr_ss(DUAL_VC_BUSY),
                   csrr_ss(STREAMER_BUSY_CSR), csrr_ss(WRITER_BUSY_CSR),
                   csrr_ss(WRITER1_BUSY_CSR));
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
    int16_t *local_mode1_d0 =
        (int16_t *)(snrt_l1_next() + cfg->delta_local_mode1_d0);
    int16_t *local_mode1_d1 =
        (int16_t *)(snrt_l1_next() + cfg->delta_local_mode1_d1);

    uint32_t subtraction_setting =
        gen_dual_vc_subtraction_config(SUBTRACTION_A, SUBTRACTION_B);

    printf("S%d start\n", cfg->array_shape);

    // Mode 0: per-token-contiguous A, fixed S0-tiled B readers.
    printf("S%d Mode0 configure streamer\n", cfg->array_shape);
    uint32_t m0_start = snrt_mcycle();
    set_dual_versacore_streamer_csr_d0_only(
        cfg->delta_local_a, cfg->mode0_A_sstride, cfg->mode0_A_tbound,
        cfg->mode0_A_tstride, SET_ADDR_REMAP_INDEX_A, cfg->A_channel_en,
        cfg->delta_local_b0, cfg->mode0_B_sstride, cfg->mode0_B_tbound,
        cfg->mode0_B_tstride, SET_ADDR_REMAP_INDEX_B0, cfg->B_channel_en,
        cfg->delta_local_b1, cfg->mode0_B_sstride, cfg->mode0_B_tbound,
        cfg->mode0_B_tstride, SET_ADDR_REMAP_INDEX_B1, cfg->B_channel_en,
        cfg->delta_local_d0, cfg->D_sstride, cfg->mode0_D_tbound,
        cfg->mode0_D_tstride, SET_ADDR_REMAP_INDEX_D0, cfg->D_channel_en);

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

    int err_m0_d0 = check_result_i16_limited(
        local_d0, cfg->mode0_d0_golden, cfg->mode0_output_elems);
    err += err_m0_d0;

    int32_t cycles_m0 = read_dual_versacore_perf_counter();
    int32_t str_cycles_m0 = read_dual_versacore_streamer_perf_counter();
    printf("S%d Mode0 D0: %s, Error: %d\n", cfg->array_shape,
           err_m0_d0 ? "FAIL" : "PASS", err_m0_d0);
    printf("S%d Mode0 Cycles: accel=%d, streamer=%d, wall=%u\n",
           cfg->array_shape, cycles_m0, str_cycles_m0, m0_end - m0_start);
    if (err_m0_d0) {
        return 5000 + cfg->array_shape;
    }

    // Mode 1: per-token output layout — D writer tbound/tstride encode per-token order.
    printf("S%d Mode1 configure streamer\n", cfg->array_shape);
    uint32_t m1_start = snrt_mcycle();
    set_dual_versacore_streamer_csr(
        cfg->delta_local_d0, cfg->mode1_A_sstride, cfg->mode1_A_tbound,
        cfg->mode1_A_tstride, SET_ADDR_REMAP_INDEX_A, cfg->A_channel_en,
        cfg->delta_local_w2l, cfg->mode1_B_sstride, cfg->mode1_B_tbound,
        cfg->mode1_B_tstride, SET_ADDR_REMAP_INDEX_B0, cfg->B_channel_en,
        cfg->delta_local_w2r, cfg->mode1_B_sstride, cfg->mode1_B_tbound,
        cfg->mode1_B_tstride, SET_ADDR_REMAP_INDEX_B1, cfg->B_channel_en,
        cfg->delta_local_mode1_d0, cfg->D_sstride, cfg->mode1_D_tbound,
        cfg->mode1_D_tstride, SET_ADDR_REMAP_INDEX_D0, cfg->D_channel_en,
        cfg->delta_local_mode1_d1, cfg->D_sstride, cfg->mode1_D_tbound,
        cfg->mode1_D_tstride, SET_ADDR_REMAP_INDEX_D1, cfg->D_channel_en);

    printf("S%d Mode1 configure core\n", cfg->array_shape);
    set_dual_versacore_csr(1, cfg->K1, cfg->N1 * cfg->M_tiles,
                           subtraction_setting, cfg->array_shape, DATA_TYPE);
    set_dual_versacore_mode(1);
    set_dual_versacore_rescale0(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale1(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
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

    int err_m1_d0 = check_result_i16_limited(
        local_mode1_d0, cfg->mode1_d0_golden, cfg->mode1_output_elems);
    int err_m1_d1 = check_result_i16_limited(
        local_mode1_d1, cfg->mode1_d1_golden, cfg->mode1_output_elems);
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
        printf("K8 8x4 mode1-pertoken TCDM placement exceeds capacity: end=%d cap=%d\n",
               shape_cfg[0].tcdm_end, TCDM_CAPACITY_BYTES);
        return 1;
    }

    if (snrt_global_core_idx() == 0) {
        printf("Mode1 per-token multishape K8 8x4 staging shared dataset: bytes=%d\n",
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
        printf("Mode1 per-token multishape K8 8x4 shared dataset ready\n");
        for (int shape = 0; shape < NUM_SHAPES; shape++) {
            err += run_shape(&shape_cfg[shape]);
        }
        printf("Mode1 per-token multishape K8 8x4 total checks: 9, total error: %d\n", err);
    }

    return err;
}
