// Small MoE-style K=8 8x4 L15 app with weights-first TCDM placement.
//
// This test intentionally runs one fixed layout and all generated shapes.  The
// datagen script owns the tensor placement, streamer strides, and golden data;
// this C file only stages the data, programs the CSRs, starts the accelerator,
// and checks the results.

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

#define WAIT_TIMEOUT_CYCLES 300000u
#define CHECK_PRINT_LIMIT 16u
#ifndef SELECT_LAYOUT
#define SELECT_LAYOUT 0
#endif

static int check_result_i16_limited(const int16_t *output,
                                    const int16_t *golden,
                                    int32_t num_elements) {
    uint32_t err = 0;

    for (int i = 0; i < num_elements; i++) {
        if (output[i] != golden[i]) {
            if (err < CHECK_PRINT_LIMIT) {
                printf("Unequals. output[%d] = %d, golden[%d] = %d\n", i,
                       output[i], i, golden[i]);
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

static int wait_accelerator_done(int layout_id, int shape_id, int mode) {
    uint32_t start_cycle;

    csrw_ss(DUAL_VC_START, 0);
    csrw_ss(DUAL_VC_START, 0);

    start_cycle = snrt_mcycle();
    while (csrr_ss(DUAL_VC_BUSY)) {
        if ((uint32_t)(snrt_mcycle() - start_cycle) >
            WAIT_TIMEOUT_CYCLES) {
            printf("L%d S%d Mode%d accelerator timeout: accel=%u streamer=%u "
                   "w0=%u w1=%u\n",
                   layout_id, shape_id, mode, csrr_ss(DUAL_VC_BUSY),
                   csrr_ss(STREAMER_BUSY_CSR), csrr_ss(WRITER_BUSY_CSR),
                   csrr_ss(WRITER1_BUSY_CSR));
            return 1;
        }
    }

    return 0;
}

static int wait_streamer_done(int layout_id, int shape_id, int mode) {
    uint32_t start_cycle;

    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);

    start_cycle = snrt_mcycle();
    while (csrr_ss(STREAMER_BUSY_CSR)) {
        if ((uint32_t)(snrt_mcycle() - start_cycle) >
            WAIT_TIMEOUT_CYCLES) {
            printf("L%d S%d Mode%d streamer timeout: streamer=%u w0=%u "
                   "w1=%u\n",
                   layout_id, shape_id, mode, csrr_ss(STREAMER_BUSY_CSR),
                   csrr_ss(WRITER_BUSY_CSR), csrr_ss(WRITER1_BUSY_CSR));
            return 1;
        }
    }

    return 0;
}

static int stage_layout_to_tcdm(const layout_cfg_t *layout) {
    const shape_cfg_t *cfg = &layout->shapes[0];

    uint8_t *local_b0 = (uint8_t *)(snrt_l1_next() + cfg->delta_local_b0);
    uint8_t *local_b1 = (uint8_t *)(snrt_l1_next() + cfg->delta_local_b1);
    uint8_t *local_w2_left =
        (uint8_t *)(snrt_l1_next() + cfg->delta_local_w2l);
    uint8_t *local_w2_right =
        (uint8_t *)(snrt_l1_next() + cfg->delta_local_w2r);

    if (cfg->tcdm_end > TCDM_CAPACITY_BYTES) {
        if (snrt_global_core_idx() == 0) {
            printf("L%d TCDM placement exceeds capacity: end=%d cap=%d\n",
                   layout->layout_id, cfg->tcdm_end, TCDM_CAPACITY_BYTES);
        }
        return 1;
    }

    if (snrt_global_core_idx() == 0) {
        printf("L%d %s staging dataset: bytes=%d A_stride=%d B0_bank=%d "
               "B1_bank=%d D0_bank=%d M1D0_bank=%d M1D1_minus_D0=%d "
               "M1_row_stride=%d shape_A_strides=[%d,%d,%d]\n",
               layout->layout_id, layout->name, cfg->tcdm_end,
               layout->a_row_stride, cfg->delta_local_b0 / 8 % 64,
               cfg->delta_local_b1 / 8 % 64,
               cfg->delta_local_d0 / 8 % 64,
               cfg->delta_local_mode1_d0 / 8 % 64,
               cfg->delta_local_mode1_d1 - cfg->delta_local_mode1_d0,
               cfg->mode1_output_row_stride_bytes,
               layout->shapes[0].a_row_stride, layout->shapes[1].a_row_stride,
               layout->shapes[2].a_row_stride);
    }

    if (snrt_is_dm_core()) {
        uint32_t dma_start = snrt_mcycle();

        for (int shape = 0; shape < NUM_SHAPES; shape++) {
            int16_t *local_a =
                (int16_t *)(snrt_l1_next() +
                            layout->shapes[shape].delta_local_a);
            snrt_dma_start_1d(local_a, layout->shapes[shape].a_data,
                              layout->shapes[shape].a_data_length);
        }
        snrt_dma_start_1d(local_b0, layout->w_data, layout->b_data_length);
        snrt_dma_start_1d(local_b1, layout->v_data, layout->b_data_length);
        snrt_dma_start_1d(local_w2_left, layout->w2_left_data,
                          layout->w2_data_length);
        snrt_dma_start_1d(local_w2_right, layout->w2_right_data,
                          layout->w2_data_length);
        snrt_dma_wait_all();

        printf("L%d %s DMA staging cycles: %u\n", layout->layout_id,
               layout->name, snrt_mcycle() - dma_start);
    }

    snrt_cluster_hw_barrier();
    return 0;
}

static void configure_identity_rescale_for_mode0(void) {
    set_dual_versacore_rescale0(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale1(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale_mul(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                   RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
}

static void configure_identity_rescale_for_mode1(void) {
    set_dual_versacore_rescale0(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale1(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
}

static int run_mode0(int layout_id, const shape_cfg_t *cfg,
                     uint32_t subtraction_setting) {
    int16_t *local_d0 = (int16_t *)(snrt_l1_next() + cfg->delta_local_d0);

    for (int i = 0; i < cfg->mode0_output_elems; i++) {
        local_d0[i] = 0;
    }

    printf("L%d S%d Mode0 configure streamer\n", layout_id,
           cfg->array_shape);
    uint32_t start_cycle = snrt_mcycle();
    set_dual_versacore_streamer_csr_d0_only(
        cfg->delta_local_a, cfg->mode0_A_sstride, cfg->mode0_A_tbound,
        cfg->mode0_A_tstride, SET_ADDR_REMAP_INDEX_A, cfg->A_channel_en,
        cfg->delta_local_b0, cfg->mode0_B_sstride, cfg->mode0_B_tbound,
        cfg->mode0_B_tstride, SET_ADDR_REMAP_INDEX_B0, cfg->B_channel_en,
        cfg->delta_local_b1, cfg->mode0_B_sstride, cfg->mode0_B_tbound,
        cfg->mode0_B_tstride, SET_ADDR_REMAP_INDEX_B1, cfg->B_channel_en,
        cfg->delta_local_d0, cfg->D_sstride, cfg->mode0_D_tbound,
        cfg->mode0_D_tstride, SET_ADDR_REMAP_INDEX_D0, cfg->D_channel_en);

    printf("L%d S%d Mode0 configure core\n", layout_id, cfg->array_shape);
    set_dual_versacore_csr(1, cfg->K_tiles, cfg->N_tiles * cfg->M_tiles,
                           subtraction_setting, cfg->array_shape, DATA_TYPE);
    set_dual_versacore_mode(0);
    configure_identity_rescale_for_mode0();

    printf("L%d S%d Mode0 start\n", layout_id, cfg->array_shape);
    uint32_t streamer_start_cycle = snrt_mcycle();
    set_dual_versacore_streamer_start();
    uint32_t accel_start_cycle = snrt_mcycle();
    set_dual_versacore_start();
    if (wait_accelerator_done(layout_id, cfg->array_shape, 0)) {
        return 1000 + cfg->array_shape;
    }
    uint32_t accel_done_cycle = snrt_mcycle();
    if (wait_streamer_done(layout_id, cfg->array_shape, 0)) {
        return 2000 + cfg->array_shape;
    }
    uint32_t streamer_done_cycle = snrt_mcycle();

    int err = check_result_i16_limited(local_d0, cfg->mode0_d0_golden,
                                       cfg->mode0_output_elems);
    int32_t accel_cycles = read_dual_versacore_perf_counter();
    int32_t streamer_cycles = read_dual_versacore_streamer_perf_counter();

    printf("L%d S%d Mode0 D0: %s, Error: %d\n", layout_id,
           cfg->array_shape, err ? "FAIL" : "PASS", err);
    printf("L%d S%d Mode0 Cycles: accel=%d, streamer=%d, wall=%u\n",
           layout_id, cfg->array_shape, accel_cycles, streamer_cycles,
           snrt_mcycle() - start_cycle);
    printf("L%d S%d Mode0 mcycle start-finish: accel=%u, streamer=%u\n",
           layout_id, cfg->array_shape, accel_done_cycle - accel_start_cycle,
           streamer_done_cycle - streamer_start_cycle);

    if (err) {
        return 5000 + cfg->array_shape;
    }

    return 0;
}

static int run_mode1(int layout_id, const shape_cfg_t *cfg,
                     uint32_t subtraction_setting) {
    int16_t *local_mode1_d =
        (int16_t *)(snrt_l1_next() + cfg->delta_local_mode1_d0);

    printf("L%d S%d Mode1 configure streamer\n", layout_id,
           cfg->array_shape);
    for (int i = 0; i < cfg->mode1_padded_output_elems; i++) {
        local_mode1_d[i] = 0;
    }
    uint32_t start_cycle = snrt_mcycle();

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

    printf("L%d S%d Mode1 configure core\n", layout_id, cfg->array_shape);
    set_dual_versacore_csr(1, cfg->K1, cfg->N1 * cfg->M_tiles,
                           subtraction_setting, cfg->array_shape, DATA_TYPE);
    set_dual_versacore_mode(1);
    configure_identity_rescale_for_mode1();

    printf("L%d S%d Mode1 start\n", layout_id, cfg->array_shape);
    uint32_t streamer_start_cycle = snrt_mcycle();
    set_dual_versacore_streamer_start();
    uint32_t accel_start_cycle = snrt_mcycle();
    set_dual_versacore_start();
    if (wait_accelerator_done(layout_id, cfg->array_shape, 1)) {
        return 3000 + cfg->array_shape;
    }
    uint32_t accel_done_cycle = snrt_mcycle();
    if (wait_streamer_done(layout_id, cfg->array_shape, 1)) {
        return 4000 + cfg->array_shape;
    }
    uint32_t streamer_done_cycle = snrt_mcycle();

    int err = check_result_i16_limited(local_mode1_d,
                                       cfg->mode1_padded_golden,
                                       cfg->mode1_padded_output_elems);
    int32_t accel_cycles = read_dual_versacore_perf_counter();
    int32_t streamer_cycles = read_dual_versacore_streamer_perf_counter();

    printf("L%d S%d Mode1 D padded-contiguous: %s, Error: %d\n", layout_id,
           cfg->array_shape, err ? "FAIL" : "PASS", err);
    printf("L%d S%d Mode1 Cycles: accel=%d, streamer=%d, wall=%u\n",
           layout_id, cfg->array_shape, accel_cycles, streamer_cycles,
           snrt_mcycle() - start_cycle);
    printf("L%d S%d Mode1 mcycle start-finish: accel=%u, streamer=%u\n",
           layout_id, cfg->array_shape, accel_done_cycle - accel_start_cycle,
           streamer_done_cycle - streamer_start_cycle);

    return err;
}

static int run_shape(int layout_id, const shape_cfg_t *cfg) {
    uint32_t subtraction_setting =
        gen_dual_vc_subtraction_config(SUBTRACTION_A, SUBTRACTION_B);

    printf("L%d S%d start\n", layout_id, cfg->array_shape);

    int mode0_err = run_mode0(layout_id, cfg, subtraction_setting);
    if (mode0_err) {
        return mode0_err;
    }

    return run_mode1(layout_id, cfg, subtraction_setting);
}

int main(void) {
    if (SELECT_LAYOUT < 0 || SELECT_LAYOUT >= NUM_LAYOUTS) {
        if (snrt_global_core_idx() == 0) {
            printf("Invalid SELECT_LAYOUT=%d, NUM_LAYOUTS=%d\n",
                   SELECT_LAYOUT, NUM_LAYOUTS);
        }
        return 1;
    }

    const layout_cfg_t *layout = &layout_cfgs[SELECT_LAYOUT];
    int err = 0;

    if (snrt_global_core_idx() == 0) {
        printf("Small MoE K8 8x4 padded-contiguous L15 app: layouts=%d "
               "shapes=%d\n",
               NUM_LAYOUTS, NUM_SHAPES);
    }

    if (stage_layout_to_tcdm(layout)) {
        return 1;
    }

    if (snrt_global_core_idx() != 0) {
        return 0;
    }

    printf("L%d %s dataset ready\n", layout->layout_id, layout->name);

    for (int shape = 0; shape < NUM_SHAPES; shape++) {
        err += run_shape(layout->layout_id, &layout->shapes[shape]);
    }

    printf("L%d %s total checks: %d, total error: %d\n", layout->layout_id,
           layout->name, NUM_SHAPES * 2, err);

    return err;
}
