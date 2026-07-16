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
#ifndef NUM_LAYOUTS_TO_RUN
#define NUM_LAYOUTS_TO_RUN 1
#endif
#ifndef SELECT_SHAPE
#define SELECT_SHAPE -1
#endif
#ifndef RUN_MODE1
#define RUN_MODE1 1
#endif

static int check_result_i16_limited(const int16_t *output,
                                    const int16_t *golden,
                                    int32_t num_elements) {
    uint32_t err = 0;

    // The output and generated golden are 64-bit aligned.  Most runs pass, so
    // compare four int16 values at once and only fall back to scalar checks on
    // a mismatching word.  This preserves exhaustive checking while reducing
    // VLT instruction count substantially.
    int words = num_elements / 4;
    const uint64_t *output64 = (const uint64_t *)output;
    const uint64_t *golden64 = (const uint64_t *)golden;
    for (int word = 0; word < words; word++) {
        if (output64[word] != golden64[word]) {
            for (int lane = 0; lane < 4; lane++) {
                int i = word * 4 + lane;
                if (output[i] != golden[i]) {
                    if (err < CHECK_PRINT_LIMIT) {
                        printf("Unequals. output[%d] = %d, golden[%d] = %d\n",
                               i, output[i], i, golden[i]);
                    }
                    err++;
                }
            }
        }
    }
    for (int i = words * 4; i < num_elements; i++) {
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

static int stage_layout_to_tcdm(const layout_cfg_t *layout, int reuse_static) {
    const shape_cfg_t *cfg = &layout->shapes[0];

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

    if (reuse_static) {
        if (snrt_global_core_idx() == 0) {
            printf("L%d %s reusing identical A/weight TCDM image\n",
                   layout->layout_id, layout->name);
        }
        snrt_cluster_hw_barrier();
        return 0;
    }

    if (snrt_is_dm_core()) {
        uint32_t dma_start = snrt_mcycle();

        for (int shape = 0; shape < NUM_SHAPES; shape++) {
            int16_t *local_a =
                (int16_t *)(snrt_l1_next() +
                            layout->shapes[shape].delta_local_a);
            const shape_cfg_t *shape_cfg = &layout->shapes[shape];
            if (shape_cfg->a_panel_pitch) {
                // L3/source remains per-token. Pack only the TCDM compute
                // image into channel-linear 64-bit words per K tile. One 2D
                // DMA per active token copies its contiguous 16B K slice into
                // the selected panel pitch; inactive panel holes are unread.
                for (int token = 0; token < shape_cfg->meshRow; token++) {
                    const int16_t *src_row =
                        shape_cfg->a_data + token * (shape_cfg->a_row_stride / 2);
                    int16_t *dst = local_a +
                                   token * (shape_cfg->a_panel_token_stride / 2);
                    snrt_dma_start_2d(dst, src_row, 16,
                                      shape_cfg->a_panel_pitch, 16,
                                      shape_cfg->K_tiles);
                }
            } else {
                snrt_dma_start_1d(local_a, shape_cfg->a_data,
                                  shape_cfg->a_data_length);
            }
            int weights_already_staged = 0;
            if (shape > 0) {
                const shape_cfg_t *prev = &layout->shapes[shape - 1];
                weights_already_staged =
                    prev->delta_local_b0 == shape_cfg->delta_local_b0 &&
                    prev->delta_local_b1 == shape_cfg->delta_local_b1 &&
                    prev->delta_local_w2l == shape_cfg->delta_local_w2l &&
                    prev->delta_local_w2r == shape_cfg->delta_local_w2r &&
                    prev->w_data == shape_cfg->w_data &&
                    prev->v_data == shape_cfg->v_data &&
                    prev->w2_left_data == shape_cfg->w2_left_data &&
                    prev->w2_right_data == shape_cfg->w2_right_data;
            }
            if (!weights_already_staged) {
                uint8_t *local_b0 =
                    (uint8_t *)(snrt_l1_next() + shape_cfg->delta_local_b0);
                uint8_t *local_b1 =
                    (uint8_t *)(snrt_l1_next() + shape_cfg->delta_local_b1);
                uint8_t *local_w2_left =
                    (uint8_t *)(snrt_l1_next() + shape_cfg->delta_local_w2l);
                uint8_t *local_w2_right =
                    (uint8_t *)(snrt_l1_next() + shape_cfg->delta_local_w2r);
                snrt_dma_start_1d(local_b0, shape_cfg->w_data,
                                  shape_cfg->b_data_length);
                snrt_dma_start_1d(local_b1, shape_cfg->v_data,
                                  shape_cfg->b_data_length);
                snrt_dma_start_1d(local_w2_left, shape_cfg->w2_left_data,
                                  shape_cfg->w2_data_length);
                snrt_dma_start_1d(local_w2_right, shape_cfg->w2_right_data,
                                  shape_cfg->w2_data_length);
            }
        }
        snrt_dma_wait_all();

        printf("L%d %s DMA staging cycles: %u\n", layout->layout_id,
               layout->name, snrt_mcycle() - dma_start);
    }

    snrt_cluster_hw_barrier();
    return 0;
}

static int can_reuse_static_image(const layout_cfg_t *prev,
                                  const layout_cfg_t *next) {
    if (prev == 0) {
        return 0;
    }
    for (int shape = 0; shape < NUM_SHAPES; shape++) {
        const shape_cfg_t *a = &prev->shapes[shape];
        const shape_cfg_t *b = &next->shapes[shape];
        int same_a_image =
            a->a_panel_pitch && b->a_panel_pitch
                ? a->a_panel_pitch == b->a_panel_pitch
                : (a->a_data == b->a_data &&
                   a->a_data_length == b->a_data_length &&
                   a->a_row_stride == b->a_row_stride &&
                   a->a_panel_pitch == b->a_panel_pitch);
        if (a->delta_local_a != b->delta_local_a || !same_a_image ||
            a->a_panel_token_stride != b->a_panel_token_stride ||
            a->delta_local_b0 != b->delta_local_b0 ||
            a->delta_local_b1 != b->delta_local_b1 ||
            a->delta_local_w2l != b->delta_local_w2l ||
            a->delta_local_w2r != b->delta_local_w2r ||
            a->w_data != b->w_data || a->v_data != b->v_data ||
            a->w2_left_data != b->w2_left_data ||
            a->w2_right_data != b->w2_right_data ||
            a->b_data_length != b->b_data_length ||
            a->w2_data_length != b->w2_data_length) {
            return 0;
        }
    }
    return 1;
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
    // Both writers overwrite the complete 2*N1 payload.  Only clear the
    // per-token padding that is intentionally checked by the closed-loop
    // golden; clearing the whole output dominated VLT wall time.
    int row_elems = cfg->mode1_output_row_stride_bytes / 2;
    int payload_elems = (2 * cfg->mode1_output_elems) / cfg->meshRow;
    for (int token = 0; token < cfg->meshRow; token++) {
        for (int i = payload_elems; i < row_elems; i++) {
            local_mode1_d[token * row_elems + i] = 0;
        }
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

#if RUN_MODE1
    return run_mode1(layout_id, cfg, subtraction_setting);
#else
    return 0;
#endif
}

int main(void) {
    if (SELECT_LAYOUT < 0 || SELECT_LAYOUT + NUM_LAYOUTS_TO_RUN > NUM_LAYOUTS) {
        if (snrt_global_core_idx() == 0) {
            printf("Invalid SELECT_LAYOUT=%d, NUM_LAYOUTS=%d\n",
                   SELECT_LAYOUT, NUM_LAYOUTS);
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

    int err = 0;
    const layout_cfg_t *previous_layout = 0;

    if (snrt_global_core_idx() == 0) {
        printf("Small MoE K8 8x4 padded-contiguous L15 app: layouts=%d "
               "shapes=%d\n",
               NUM_LAYOUTS, NUM_SHAPES);
    }

    for (int layout_index = SELECT_LAYOUT;
         layout_index < SELECT_LAYOUT + NUM_LAYOUTS_TO_RUN; layout_index++) {
        const layout_cfg_t *layout = &layout_cfgs[layout_index];
        int layout_err = 0;
        int reuse_static = can_reuse_static_image(previous_layout, layout);
        if (stage_layout_to_tcdm(layout, reuse_static)) {
            return 1;
        }
        if (snrt_global_core_idx() == 0) {
            printf("L%d %s dataset ready\n", layout->layout_id, layout->name);
            int first_shape = SELECT_SHAPE < 0 ? 0 : SELECT_SHAPE;
            int shape_end = SELECT_SHAPE < 0 ? NUM_SHAPES : SELECT_SHAPE + 1;
            for (int shape = first_shape; shape < shape_end; shape++) {
                layout_err += run_shape(layout->layout_id,
                                        &layout->shapes[shape]);
            }
            int checks = (shape_end - first_shape) * (RUN_MODE1 ? 2 : 1);
            printf("L%d %s total checks: %d, total error: %d\n",
                   layout->layout_id, layout->name, checks,
                   layout_err);
            err += layout_err;
        }
        snrt_cluster_hw_barrier();
        previous_layout = layout;
    }

    return err;
}
