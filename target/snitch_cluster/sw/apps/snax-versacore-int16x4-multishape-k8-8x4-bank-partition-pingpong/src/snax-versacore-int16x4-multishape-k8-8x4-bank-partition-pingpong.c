// Bank-partitioned full-workload experiment.  DMA is completed before each
// compute phase, but B is placed exactly as complete-K ping/pong panels.

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

#define WAIT_TIMEOUT_CYCLES 500000u
#define CHECK_PRINT_LIMIT 16u
#define TCDM_ROW_BYTES 512u

#ifndef SELECT_SHAPE
#define SELECT_SHAPE 0
#endif
#ifndef RUN_MODE1
#define RUN_MODE1 1
#endif

static int wait_accel_diag(int shape, int mode) {
    csrw_ss(DUAL_VC_START, 0);
    csrw_ss(DUAL_VC_START, 0);
    uint32_t start = snrt_mcycle();
    while (csrr_ss(DUAL_VC_BUSY)) {
        if ((uint32_t)(snrt_mcycle() - start) > WAIT_TIMEOUT_CYCLES) {
            printf("S%d Mode%d accelerator timeout: accel=%u streamer=%u w0=%u w1=%u\n",
                   shape, mode, csrr_ss(DUAL_VC_BUSY),
                   csrr_ss(STREAMER_BUSY_CSR), csrr_ss(WRITER_BUSY_CSR),
                   csrr_ss(WRITER1_BUSY_CSR));
            return 1;
        }
    }
    return 0;
}

static int wait_streamer_diag(int shape, int mode) {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    uint32_t start = snrt_mcycle();
    while (csrr_ss(STREAMER_BUSY_CSR)) {
        if ((uint32_t)(snrt_mcycle() - start) > WAIT_TIMEOUT_CYCLES) {
            printf("S%d Mode%d streamer timeout: streamer=%u w0=%u w1=%u\n",
                   shape, mode, csrr_ss(STREAMER_BUSY_CSR),
                   csrr_ss(WRITER_BUSY_CSR), csrr_ss(WRITER1_BUSY_CSR));
            return 1;
        }
    }
    return 0;
}

static void stage_a(const shape_cfg_t *cfg) {
    uint8_t *tcdm = (uint8_t *)snrt_l1_next();
    for (int token = 0; token < M_TOTAL; token++) {
        const int16_t *src = A + token * K0_TOTAL;
        uint8_t *dst = tcdm + cfg->delta_local_a + token * 16;
        snrt_dma_start_2d(dst, src, 16, TCDM_ROW_BYTES, 16,
                          K0_TOTAL / 8);
    }
    snrt_dma_wait_all();
}

static void stage_weight_pair(const shape_cfg_t *cfg,
                              const uint8_t *src_b0,
                              const uint8_t *src_b1,
                              int n_tiles, int panel_bytes,
                              int panel_span) {
    uint8_t *tcdm = (uint8_t *)snrt_l1_next();
    int pending = 0;
    for (int n = 0; n < n_tiles; n++) {
        int pair = n >> 1;
        int pong = n & 1;
        int b0_base = pong ? cfg->delta_local_b0_pong
                           : cfg->delta_local_b0_ping;
        int b1_base = pong ? cfg->delta_local_b1_pong
                           : cfg->delta_local_b1_ping;
        for (int sub = 0; sub < cfg->q_shape0_cols; sub++) {
            int src_panel = n * cfg->q_shape0_cols + sub;
            int dst_panel = pair * cfg->q_shape0_cols + sub;
            uint8_t *dst0 = tcdm + b0_base + dst_panel * panel_span;
            uint8_t *dst1 = tcdm + b1_base + dst_panel * panel_span;
            const uint8_t *s0 = src_b0 + src_panel * panel_bytes;
            const uint8_t *s1 = src_b1 + src_panel * panel_bytes;
            snrt_dma_start_2d(dst0, s0, 64, TCDM_ROW_BYTES, 64,
                              panel_bytes / 64);
            snrt_dma_start_2d(dst1, s1, 64, TCDM_ROW_BYTES, 64,
                              panel_bytes / 64);
            pending += 2;
            if (pending >= 8) {
                snrt_dma_wait_all();
                pending = 0;
            }
        }
    }
    snrt_dma_wait_all();
}

static int check_mode0_token_stripes(const shape_cfg_t *cfg) {
    const uint8_t *tcdm = (const uint8_t *)snrt_l1_next();
    uint32_t errors = 0;
    for (int token = 0; token < cfg->meshRow; token++) {
        for (int elem = 0; elem < N0_TOTAL; elem++) {
            int chunk = elem / 8;
            int lane = elem % 8;
            int token_offset;
            int chunk_offset;
            if (cfg->array_shape == 0) {
                token_offset = token * 16;
                chunk_offset = chunk * TCDM_ROW_BYTES;
            } else if (cfg->array_shape == 1) {
                token_offset = token * 16;
                chunk_offset = (chunk % 2) * 64 +
                               (chunk / 2) * TCDM_ROW_BYTES;
            } else {
                token_offset = token * 32;
                chunk_offset = (chunk % 2) * 16 +
                               ((chunk / 2) % 2) * 64 +
                               (chunk / 4) * TCDM_ROW_BYTES;
            }
            const int16_t *p = (const int16_t *)(
                tcdm + cfg->delta_local_mode0_d + token_offset +
                chunk_offset + lane * 2);
            int16_t expected = cfg->mode0_token_golden[
                token * N0_TOTAL + elem];
            if (*p != expected) {
                if (errors < CHECK_PRINT_LIMIT) {
                    printf("M0 mismatch token=%d elem=%d got=%d expected=%d\n",
                           token, elem, *p, expected);
                }
                errors++;
            }
        }
    }
    return (int)errors;
}

static int check_mode1_half(const shape_cfg_t *cfg, int d1) {
    const uint8_t *tcdm = (const uint8_t *)snrt_l1_next();
    int base = d1 ? cfg->delta_local_mode1_d1
                  : cfg->delta_local_mode1_d0;
    const int16_t *golden = d1 ? cfg->mode1_d1_token_golden
                               : cfg->mode1_d0_token_golden;
    uint32_t errors = 0;
    for (int token = 0; token < cfg->meshRow; token++) {
        for (int elem = 0; elem < N1_TOTAL; elem++) {
            int beat = elem / 4;
            int lane = elem % 4;
            const int16_t *p = (const int16_t *)(
                tcdm + base + token * 8 + beat * TCDM_ROW_BYTES + lane * 2);
            int16_t expected = golden[token * N1_TOTAL + elem];
            if (*p != expected) {
                if (errors < CHECK_PRINT_LIMIT) {
                    printf("M1D%d mismatch token=%d elem=%d got=%d expected=%d\n",
                           d1, token, elem, *p, expected);
                }
                errors++;
            }
        }
    }
    return (int)errors;
}

static int run_mode0(const shape_cfg_t *cfg) {
    uint32_t subtraction =
        gen_dual_vc_subtraction_config(SUBTRACTION_A, SUBTRACTION_B);
    uint32_t wall_start = snrt_mcycle();
    set_dual_versacore_streamer_csr_d0_only(
        cfg->delta_local_a, cfg->mode0_A_sstride, cfg->mode0_A_tbound,
        cfg->mode0_A_tstride, SET_ADDR_REMAP_INDEX_A, cfg->A_channel_en,
        cfg->delta_local_b0_ping, cfg->mode0_B_sstride,
        cfg->mode0_B_tbound, cfg->mode0_B_tstride,
        SET_ADDR_REMAP_INDEX_B0, cfg->B_channel_en,
        cfg->delta_local_b1_ping, cfg->mode0_B_sstride,
        cfg->mode0_B_tbound, cfg->mode0_B_tstride,
        SET_ADDR_REMAP_INDEX_B1, cfg->B_channel_en,
        cfg->delta_local_mode0_d, cfg->D_sstride, cfg->mode0_D_tbound,
        cfg->mode0_D_tstride, SET_ADDR_REMAP_INDEX_D0,
        cfg->D_channel_en);
    set_dual_versacore_csr(1, cfg->K_tiles, cfg->N_tiles,
                           subtraction, cfg->array_shape, DATA_TYPE);
    set_dual_versacore_mode(0);
    set_dual_versacore_rescale0(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale1(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale_mul(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                   RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    if (wait_accel_diag(cfg->array_shape, 0) ||
        wait_streamer_diag(cfg->array_shape, 0)) {
        return 1000 + cfg->array_shape;
    }
    uint32_t wall = snrt_mcycle() - wall_start;
    int accel = read_dual_versacore_perf_counter();
    int streamer = read_dual_versacore_streamer_perf_counter();
    int errors = check_mode0_token_stripes(cfg);
    printf("S%d Mode0 token-striped: %s, errors=%d, accel=%d, streamer=%d, wall=%u\n",
           cfg->array_shape, errors ? "FAIL" : "PASS", errors,
           accel, streamer, wall);
    return errors;
}

static int run_mode1(const shape_cfg_t *cfg) {
    uint32_t subtraction =
        gen_dual_vc_subtraction_config(SUBTRACTION_A, SUBTRACTION_B);
    uint32_t wall_start = snrt_mcycle();
    set_dual_versacore_streamer_csr(
        cfg->delta_local_mode0_d, cfg->mode1_A_sstride, cfg->mode1_A_tbound,
        cfg->mode1_A_tstride, SET_ADDR_REMAP_INDEX_A, cfg->A_channel_en,
        cfg->delta_local_b0_ping, cfg->mode1_B_sstride,
        cfg->mode1_B_tbound, cfg->mode1_B_tstride,
        SET_ADDR_REMAP_INDEX_B0, cfg->B_channel_en,
        cfg->delta_local_b1_ping, cfg->mode1_B_sstride,
        cfg->mode1_B_tbound, cfg->mode1_B_tstride,
        SET_ADDR_REMAP_INDEX_B1, cfg->B_channel_en,
        cfg->delta_local_mode1_d0, cfg->D_sstride, cfg->mode1_D_tbound,
        cfg->mode1_D_tstride, SET_ADDR_REMAP_INDEX_D0,
        cfg->D_channel_en,
        cfg->delta_local_mode1_d1, cfg->D_sstride, cfg->mode1_D_tbound,
        cfg->mode1_D_tstride, SET_ADDR_REMAP_INDEX_D1,
        cfg->D_channel_en);
    set_dual_versacore_csr(1, cfg->K1_tiles, cfg->N1_tiles,
                           subtraction, cfg->array_shape, DATA_TYPE);
    set_dual_versacore_mode(1);
    set_dual_versacore_rescale0(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale1(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    if (wait_accel_diag(cfg->array_shape, 1) ||
        wait_streamer_diag(cfg->array_shape, 1)) {
        return 2000 + cfg->array_shape;
    }
    uint32_t wall = snrt_mcycle() - wall_start;
    int accel = read_dual_versacore_perf_counter();
    int streamer = read_dual_versacore_streamer_perf_counter();
    int err0 = check_mode1_half(cfg, 0);
    int err1 = check_mode1_half(cfg, 1);
    printf("S%d Mode1 per-token: D0=%s D1=%s, errors=%d, accel=%d, streamer=%d, wall=%u\n",
           cfg->array_shape, err0 ? "FAIL" : "PASS",
           err1 ? "FAIL" : "PASS", err0 + err1, accel, streamer, wall);
    return err0 + err1;
}

int main(void) {
    if (SELECT_SHAPE < 0 || SELECT_SHAPE >= NUM_SHAPES) {
        if (snrt_global_core_idx() == 0) {
            printf("Invalid SELECT_SHAPE=%d\n", SELECT_SHAPE);
        }
        return 1;
    }
    const shape_cfg_t *cfg = &shape_cfgs[SELECT_SHAPE];
    int error = 0;

    if (snrt_global_core_idx() == 0) {
        printf("Bank partition S%d (%dx%dx%d), no padding/coloring, granularity=1\n",
               cfg->array_shape, cfg->meshRow, cfg->tileSize, cfg->meshCol);
        printf("banks: A=0..15 Bping=(16..31) Bpong=(32..47) D=48..63; q=%d\n",
               cfg->q_shape0_cols);
        printf("Mode0 DMA2D: panel=%d span=%d repeat=%d; A size=16 dst_stride=512 repeat=256\n",
               cfg->mode0_panel_bytes, cfg->mode0_panel_span,
               cfg->mode0_panel_bytes / 64);
    }

    if (snrt_is_dm_core()) {
        uint32_t start = snrt_mcycle();
        stage_a(cfg);
        stage_weight_pair(cfg, W, V, cfg->N_tiles,
                          cfg->mode0_panel_bytes, cfg->mode0_panel_span);
        printf("S%d Mode0 DMA staging cycles: %u\n", cfg->array_shape,
               snrt_mcycle() - start);
    }
    snrt_cluster_hw_barrier();
    if (snrt_global_core_idx() == 0) {
        error += run_mode0(cfg);
    }
    snrt_cluster_hw_barrier();

#if RUN_MODE1
    if (snrt_is_dm_core()) {
        uint32_t start = snrt_mcycle();
        stage_weight_pair(cfg, W2_left, W2_right, cfg->N1_tiles,
                          cfg->mode1_panel_bytes, cfg->mode1_panel_span);
        printf("S%d Mode1 DMA staging cycles: %u\n", cfg->array_shape,
               snrt_mcycle() - start);
    }
    snrt_cluster_hw_barrier();
    if (snrt_global_core_idx() == 0) {
        error += run_mode1(cfg);
    }
    snrt_cluster_hw_barrier();
#endif

    if (snrt_global_core_idx() == 0) {
        printf("S%d total error: %d\n", cfg->array_shape, error);
    }
    return snrt_global_core_idx() == 0 ? error : 0;
}
