// K=8 8x4 L15 Mode1 app with A-format padded Mode1 output rows.

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

#define WAIT_TIMEOUT_CYCLES 300000u
#define CHECK_PRINT_LIMIT 16u
#ifndef SELECT_LAYOUT
#define SELECT_LAYOUT 0
#endif
#ifndef SELECT_SHAPE
#define SELECT_SHAPE -1
#endif
#ifndef RUN_MODE1
#define RUN_MODE1 1
#endif

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

static int wait_accel_diag(int layout, int shape, int mode) {
    csrw_ss(DUAL_VC_START, 0);
    csrw_ss(DUAL_VC_START, 0);
    uint32_t start = snrt_mcycle();
    while (csrr_ss(DUAL_VC_BUSY)) {
        if ((uint32_t)(snrt_mcycle() - start) > WAIT_TIMEOUT_CYCLES) {
                 printf("L%d S%d Mode%d accelerator wait timeout: accel=%u streamer=%u w0=%u w1=%u\n",
                     layout, shape, mode, csrr_ss(DUAL_VC_BUSY),
                   csrr_ss(STREAMER_BUSY_CSR), csrr_ss(WRITER_BUSY_CSR),
                   csrr_ss(WRITER1_BUSY_CSR));
            return 1;
        }
    }
    return 0;
}

static int wait_writer_diag(int layout, int shape, int mode) {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    uint32_t start = snrt_mcycle();
    while (csrr_ss(STREAMER_BUSY_CSR)) {
        if ((uint32_t)(snrt_mcycle() - start) > WAIT_TIMEOUT_CYCLES) {
                 printf("L%d S%d Mode%d writer wait timeout: streamer=%u w0=%u w1=%u\n",
                     layout, shape, mode, csrr_ss(STREAMER_BUSY_CSR),
                   csrr_ss(WRITER_BUSY_CSR), csrr_ss(WRITER1_BUSY_CSR));
            return 1;
        }
    }
    return 0;
}

static int run_shape(int layout, const shape_cfg_t *cfg) {
    int err = 0;

    int16_t *local_d0 = (int16_t *)(snrt_l1_next() + cfg->delta_local_d0);
#if RUN_MODE1
    int16_t *local_mode1_d =
        (int16_t *)(snrt_l1_next() + cfg->delta_local_mode1_d0);
    int16_t *local_mode1_d1 =
        (int16_t *)(snrt_l1_next() + cfg->delta_local_mode1_d1);
#endif

    uint32_t subtraction_setting =
        gen_dual_vc_subtraction_config(SUBTRACTION_A, SUBTRACTION_B);

    printf("L%d S%d start\n", layout, cfg->array_shape);

    // Mode 0: per-token-contiguous A, fixed S0-tiled B readers.
    printf("L%d S%d Mode0 configure streamer\n", layout, cfg->array_shape);
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

    printf("L%d S%d Mode0 configure core\n", layout, cfg->array_shape);
    set_dual_versacore_csr(1, cfg->K_tiles, cfg->N_tiles * cfg->M_tiles,
                           subtraction_setting, cfg->array_shape, DATA_TYPE);
    set_dual_versacore_mode(0);
    set_dual_versacore_rescale0(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale1(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale_mul(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                   RESCALE_OUTPUT_ZP, RESCALE_SHIFT);

    printf("L%d S%d Mode0 start\n", layout, cfg->array_shape);
    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    printf("L%d S%d Mode0 wait accel\n", layout, cfg->array_shape);
    if (wait_accel_diag(layout, cfg->array_shape, 0)) {
        return 1000 + cfg->array_shape;
    }
    printf("L%d S%d Mode0 wait writer\n", layout, cfg->array_shape);
    if (wait_writer_diag(layout, cfg->array_shape, 0)) {
        return 2000 + cfg->array_shape;
    }
    uint32_t m0_end = snrt_mcycle();

    int err_m0_d0 = check_result_i16_limited(
        local_d0, cfg->mode0_d0_golden, cfg->mode0_output_elems);
    err += err_m0_d0;

    int32_t cycles_m0 = read_dual_versacore_perf_counter();
    int32_t str_cycles_m0 = read_dual_versacore_streamer_perf_counter();
        printf("L%d S%d Mode0 D0: %s, Error: %d\n", layout, cfg->array_shape,
            err_m0_d0 ? "FAIL" : "PASS", err_m0_d0);
        printf("L%d S%d Mode0 Cycles: accel=%d, streamer=%d, wall=%u\n",
            layout, cfg->array_shape, cycles_m0, str_cycles_m0,
            m0_end - m0_start);
    if (err_m0_d0) {
        return 5000 + cfg->array_shape;
    }

#if RUN_MODE1
    // Mode 1: read Mode0 D0 and write [left1024, right1024, pad] per token.
    printf("L%d S%d Mode1 configure streamer\n", layout, cfg->array_shape);
    uint32_t m1_start = snrt_mcycle();
    for (int i = 0; i < cfg->mode1_padded_output_elems; i++) {
        local_mode1_d[i] = 0;
    }
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

    printf("L%d S%d Mode1 configure core\n", layout, cfg->array_shape);
    set_dual_versacore_csr(1, cfg->K1, cfg->N1 * cfg->M_tiles,
                           subtraction_setting, cfg->array_shape, DATA_TYPE);
    set_dual_versacore_mode(1);
    set_dual_versacore_rescale0(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale1(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);

    printf("L%d S%d Mode1 start\n", layout, cfg->array_shape);
    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    printf("L%d S%d Mode1 wait accel\n", layout, cfg->array_shape);
    if (wait_accel_diag(layout, cfg->array_shape, 1)) {
        return 3000 + cfg->array_shape;
    }
    printf("L%d S%d Mode1 wait writer\n", layout, cfg->array_shape);
    if (wait_writer_diag(layout, cfg->array_shape, 1)) {
        return 4000 + cfg->array_shape;
    }
    uint32_t m1_end = snrt_mcycle();

    int err_m1_d = check_result_i16_limited(
        local_mode1_d, cfg->mode1_padded_golden,
        cfg->mode1_padded_output_elems);
    err += err_m1_d;

    int32_t cycles_m1 = read_dual_versacore_perf_counter();
    int32_t str_cycles_m1 = read_dual_versacore_streamer_perf_counter();
        printf("L%d S%d Mode1 D padded-contiguous: %s, Error: %d\n", layout,
            cfg->array_shape, err_m1_d ? "FAIL" : "PASS", err_m1_d);
    printf("L%d S%d Mode1 Cycles: accel=%d, streamer=%d, wall=%u\n",
           layout, cfg->array_shape, cycles_m1, str_cycles_m1,
           m1_end - m1_start);
#endif

    return err;
}

static int stage_layout(const layout_cfg_t *layout) {
    const shape_cfg_t *cfg0 = &layout->shapes[0];
    int16_t *local_a = (int16_t *)(snrt_l1_next() + cfg0->delta_local_a);
    uint8_t *local_b0 = (uint8_t *)(snrt_l1_next() + cfg0->delta_local_b0);
    uint8_t *local_b1 = (uint8_t *)(snrt_l1_next() + cfg0->delta_local_b1);
    uint8_t *local_w2l = (uint8_t *)(snrt_l1_next() + cfg0->delta_local_w2l);
    uint8_t *local_w2r = (uint8_t *)(snrt_l1_next() + cfg0->delta_local_w2r);

    if (cfg0->tcdm_end > TCDM_CAPACITY_BYTES) {
        printf("L%d TCDM placement exceeds capacity: end=%d cap=%d\n",
               layout->layout_id, cfg0->tcdm_end, TCDM_CAPACITY_BYTES);
        return 1;
    }

    if (snrt_global_core_idx() == 0) {
                                 printf("L%d %s staging dataset: bytes=%d A_stride=%d B0_bank=%d B1_bank=%d D0_bank=%d M1D0_bank=%d M1D1_minus_D0=%d M1_row_stride=%d\n",
               layout->layout_id, layout->name, cfg0->tcdm_end,
               layout->a_row_stride, cfg0->delta_local_b0 / 8 % 64,
             cfg0->delta_local_b1 / 8 % 64,
                         cfg0->delta_local_d0 / 8 % 64,
                         cfg0->delta_local_mode1_d0 / 8 % 64,
                                                 cfg0->delta_local_mode1_d1 - cfg0->delta_local_mode1_d0,
                                                 cfg0->mode1_output_row_stride_bytes);
    }

    if (snrt_is_dm_core()) {
        uint32_t dma_start = snrt_mcycle();
        snrt_dma_start_1d(local_a, layout->a_data, layout->a_data_length);
        snrt_dma_start_1d(local_b0, layout->w_data, layout->b_data_length);
        snrt_dma_start_1d(local_b1, layout->v_data, layout->b_data_length);
        snrt_dma_start_1d(local_w2l, layout->w2_left_data,
                          layout->w2_data_length);
        snrt_dma_start_1d(local_w2r, layout->w2_right_data,
                          layout->w2_data_length);
        snrt_dma_wait_all();
        uint32_t dma_end = snrt_mcycle();
        printf("L%d %s DMA staging cycles: %u\n", layout->layout_id,
               layout->name, dma_end - dma_start);
    }

    snrt_cluster_hw_barrier();
    return 0;
}

int main() {
    int err = 0;

    if (snrt_global_core_idx() == 0) {
        printf("K8 8x4 Mode1 padded-contiguous L15 app: layouts=%d shapes=%d\n",
               NUM_LAYOUTS, NUM_SHAPES);
    }

    if (SELECT_LAYOUT < 0 || SELECT_LAYOUT >= NUM_LAYOUTS) {
        if (snrt_global_core_idx() == 0) {
            printf("Invalid SELECT_LAYOUT=%d, NUM_LAYOUTS=%d\n", SELECT_LAYOUT,
                   NUM_LAYOUTS);
        }
        return 1;
    }

    if (SELECT_SHAPE < -1 || SELECT_SHAPE >= NUM_SHAPES) {
        if (snrt_global_core_idx() == 0) {
            printf("Invalid SELECT_SHAPE=%d, NUM_SHAPES=%d\n", SELECT_SHAPE,
                   NUM_SHAPES);
        }
        return 1;
    }

    const layout_cfg_t *layout_cfg = &layout_cfgs[SELECT_LAYOUT];
    int stage_err = stage_layout(layout_cfg);
    if (stage_err) {
        return stage_err;
    }

    if (snrt_global_core_idx() != 0) {
        return 0;
    }

        int shape_begin = SELECT_SHAPE < 0 ? 0 : SELECT_SHAPE;
        int shape_end = SELECT_SHAPE < 0 ? NUM_SHAPES : SELECT_SHAPE + 1;
        int checks = (shape_end - shape_begin) * (RUN_MODE1 ? 2 : 1);

        printf("L%d %s dataset ready\n", layout_cfg->layout_id,
            layout_cfg->name);
        for (int shape = shape_begin; shape < shape_end; shape++) {
        err += run_shape(layout_cfg->layout_id, &layout_cfg->shapes[shape]);
    }
#if RUN_MODE1
        printf("L%d %s total checks: %d, total error: %d\n", layout_cfg->layout_id,
            layout_cfg->name, checks, err);
#else
        printf("L%d %s total checks: %d, total error: %d\n", layout_cfg->layout_id,
            layout_cfg->name, checks, err);
#endif

    return err;
}
