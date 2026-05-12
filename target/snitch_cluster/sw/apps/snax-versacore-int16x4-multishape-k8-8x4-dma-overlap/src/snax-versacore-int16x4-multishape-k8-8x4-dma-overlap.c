// K=8 8x4 DMA-start timing and full-K N-panel overlap experiments.

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

#define WAIT_TIMEOUT_CYCLES 300000u
#define CHECK_PRINT_LIMIT 16u

#ifndef SELECT_SHAPE
#define SELECT_SHAPE 0
#endif

#ifndef SELECT_STRATEGY
#define SELECT_STRATEGY 0
#endif

#ifndef RUN_MODE1
#define RUN_MODE1 1
#endif

static uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

static uint32_t align_up_u32(uint32_t value, uint32_t alignment) {
    return ((value + alignment - 1u) / alignment) * alignment;
}

static void copy_i32(int32_t *dst, const int32_t *src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

static uint32_t strategy_chunk_tiles(void) {
    switch (SELECT_STRATEGY) {
        case 1:
            return 1;
        case 2:
            return 2;
        case 3:
            return 4;
        case 4:
            return 8;
        case 5:
            return 16;
        case 6:
            return 32;
        case 7:
            return 64;
        case 8:
            return 128;
        case 9:
            return 256;
        case 10:
            return 512;
        case 11:
            return 24;
        case 12:
            return 48;
        case 13:
            return 80;
        case 14:
            return 96;
        case 15:
            return 192;
        case 16:
            return 16;
        case 17:
            return 32;
        case 18:
            return 64;
        case 19:
            return 8;
        case 20:
            return 128;
        case 21:
            return 4;
        case 22:
            return 6;
        case 23:
            return 10;
        case 24:
            return 12;
        case 25:
            return 14;
        default:
            return 0;
    }
}

static const char *strategy_name(void) {
    switch (SELECT_STRATEGY) {
        case 0:
            return "baseline_full_dma_then_compute";
        case 1:
            return "overlap_nchunk_1";
        case 2:
            return "overlap_nchunk_2";
        case 3:
            return "overlap_nchunk_4";
        case 4:
            return "overlap_nchunk_8";
        case 5:
            return "overlap_nchunk_16";
        case 6:
            return "overlap_nchunk_32";
        case 7:
            return "overlap_nchunk_64";
        case 8:
            return "overlap_nchunk_128";
        case 9:
            return "overlap_nchunk_256";
        case 10:
            return "overlap_nchunk_512";
        case 11:
            return "overlap_nchunk_24";
        case 12:
            return "overlap_nchunk_48";
        case 13:
            return "overlap_nchunk_80";
        case 14:
            return "overlap_nchunk_96";
        case 15:
            return "overlap_nchunk_192";
        case 16:
            return "base_only_overlap_nchunk_16";
        case 17:
            return "base_only_overlap_nchunk_32";
        case 18:
            return "base_only_overlap_nchunk_64";
        case 19:
            return "base_only_overlap_nchunk_8";
        case 20:
            return "base_only_overlap_nchunk_128";
        case 21:
            return "base_only_overlap_nchunk_4";
        case 22:
            return "base_only_overlap_nchunk_6";
        case 23:
            return "base_only_overlap_nchunk_10";
        case 24:
            return "base_only_overlap_nchunk_12";
        case 25:
            return "base_only_overlap_nchunk_14";
        default:
            return "invalid";
    }
}

static int strategy_base_only_updates(void) {
    return SELECT_STRATEGY >= 16 && SELECT_STRATEGY <= 25;
}

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

static int wait_accel_diag(int shape, int mode, int chunk) {
    csrw_ss(DUAL_VC_START, 0);
    csrw_ss(DUAL_VC_START, 0);
    uint32_t start = snrt_mcycle();
    while (csrr_ss(DUAL_VC_BUSY)) {
        if ((uint32_t)(snrt_mcycle() - start) > WAIT_TIMEOUT_CYCLES) {
            printf("S%d strategy=%d Mode%d chunk=%d accelerator wait timeout: accel=%u streamer=%u w0=%u w1=%u\n",
                   shape, SELECT_STRATEGY, mode, chunk, csrr_ss(DUAL_VC_BUSY),
                   csrr_ss(STREAMER_BUSY_CSR), csrr_ss(WRITER_BUSY_CSR),
                   csrr_ss(WRITER1_BUSY_CSR));
            return 1;
        }
    }
    return 0;
}

static int wait_writer_diag(int shape, int mode, int chunk) {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    uint32_t start = snrt_mcycle();
    while (csrr_ss(STREAMER_BUSY_CSR)) {
        if ((uint32_t)(snrt_mcycle() - start) > WAIT_TIMEOUT_CYCLES) {
            printf("S%d strategy=%d Mode%d chunk=%d writer wait timeout: streamer=%u w0=%u w1=%u\n",
                   shape, SELECT_STRATEGY, mode, chunk,
                   csrr_ss(STREAMER_BUSY_CSR), csrr_ss(WRITER_BUSY_CSR),
                   csrr_ss(WRITER1_BUSY_CSR));
            return 1;
        }
    }
    return 0;
}

static int run_mode0_full(const shape_cfg_t *cfg, uint32_t *accel_cycles,
                          uint32_t *streamer_cycles, uint32_t *wall_cycles) {
    uint32_t subtraction_setting =
        gen_dual_vc_subtraction_config(SUBTRACTION_A, SUBTRACTION_B);

    uint32_t start = snrt_mcycle();
    set_dual_versacore_streamer_csr_d0_only(
        cfg->delta_local_a, cfg->mode0_A_sstride, cfg->mode0_A_tbound,
        cfg->mode0_A_tstride, SET_ADDR_REMAP_INDEX_A, cfg->A_channel_en,
        cfg->delta_local_b0, cfg->mode0_B_sstride, cfg->mode0_B_tbound,
        cfg->mode0_B_tstride, SET_ADDR_REMAP_INDEX_B0, cfg->B_channel_en,
        cfg->delta_local_b1, cfg->mode0_B_sstride, cfg->mode0_B_tbound,
        cfg->mode0_B_tstride, SET_ADDR_REMAP_INDEX_B1, cfg->B_channel_en,
        cfg->delta_local_d0, cfg->D_sstride, cfg->mode0_D_tbound,
        cfg->mode0_D_tstride, SET_ADDR_REMAP_INDEX_D0, cfg->D_channel_en);

    set_dual_versacore_csr(1, cfg->K_tiles, cfg->N_tiles * cfg->M_tiles,
                           subtraction_setting, cfg->array_shape, DATA_TYPE);
    set_dual_versacore_mode(0);
    set_dual_versacore_rescale0(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale1(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale_mul(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                   RESCALE_OUTPUT_ZP, RESCALE_SHIFT);

    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    if (wait_accel_diag(cfg->array_shape, 0, -1)) {
        return 1000 + cfg->array_shape;
    }
    if (wait_writer_diag(cfg->array_shape, 0, -1)) {
        return 2000 + cfg->array_shape;
    }
    *wall_cycles = snrt_mcycle() - start;
    *accel_cycles = read_dual_versacore_perf_counter();
    *streamer_cycles = read_dual_versacore_streamer_perf_counter();
    return 0;
}

static int run_mode1_full(const shape_cfg_t *cfg, uint32_t *accel_cycles,
                          uint32_t *streamer_cycles, uint32_t *wall_cycles) {
    uint32_t subtraction_setting =
        gen_dual_vc_subtraction_config(SUBTRACTION_A, SUBTRACTION_B);

    uint32_t start = snrt_mcycle();
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

    set_dual_versacore_csr(1, cfg->K1, cfg->N1 * cfg->M_tiles,
                           subtraction_setting, cfg->array_shape, DATA_TYPE);
    set_dual_versacore_mode(1);
    set_dual_versacore_rescale0(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale1(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);

    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    if (wait_accel_diag(cfg->array_shape, 1, -1)) {
        return 3000 + cfg->array_shape;
    }
    if (wait_writer_diag(cfg->array_shape, 1, -1)) {
        return 4000 + cfg->array_shape;
    }
    *wall_cycles = snrt_mcycle() - start;
    *accel_cycles = read_dual_versacore_perf_counter();
    *streamer_cycles = read_dual_versacore_streamer_perf_counter();
    return 0;
}

static int run_mode0_chunk(const shape_cfg_t *cfg, int32_t b0_base,
                           int32_t b1_base, uint32_t n_start,
                           uint32_t n_tiles, uint32_t chunk_idx,
                           uint32_t *accel_sum, uint32_t *streamer_sum,
                           uint32_t *wall_sum) {
    int32_t a_tbound[6], b_tbound[4], d_tbound[4];
    int32_t d_base = cfg->delta_local_d0 + n_start * cfg->mode0_D_tstride[1];
    uint32_t subtraction_setting =
        gen_dual_vc_subtraction_config(SUBTRACTION_A, SUBTRACTION_B);

    copy_i32(a_tbound, cfg->mode0_A_tbound, 6);
    copy_i32(b_tbound, cfg->mode0_B_tbound, 4);
    copy_i32(d_tbound, cfg->mode0_D_tbound, 4);
    a_tbound[1] = n_tiles;
    b_tbound[1] = n_tiles;
    d_tbound[1] = n_tiles;

    uint32_t start = snrt_mcycle();
    set_dual_versacore_streamer_csr_d0_only(
        cfg->delta_local_a, cfg->mode0_A_sstride, a_tbound,
        cfg->mode0_A_tstride, SET_ADDR_REMAP_INDEX_A, cfg->A_channel_en,
        b0_base, cfg->mode0_B_sstride, b_tbound, cfg->mode0_B_tstride,
        SET_ADDR_REMAP_INDEX_B0, cfg->B_channel_en,
        b1_base, cfg->mode0_B_sstride, b_tbound, cfg->mode0_B_tstride,
        SET_ADDR_REMAP_INDEX_B1, cfg->B_channel_en,
        d_base, cfg->D_sstride, d_tbound, cfg->mode0_D_tstride,
        SET_ADDR_REMAP_INDEX_D0, cfg->D_channel_en);

    set_dual_versacore_csr(1, cfg->K_tiles, n_tiles * cfg->M_tiles,
                           subtraction_setting, cfg->array_shape, DATA_TYPE);
    set_dual_versacore_mode(0);
    set_dual_versacore_rescale0(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale1(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale_mul(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                   RESCALE_OUTPUT_ZP, RESCALE_SHIFT);

    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    if (wait_accel_diag(cfg->array_shape, 0, chunk_idx)) {
        return 1000 + cfg->array_shape;
    }
    if (wait_writer_diag(cfg->array_shape, 0, chunk_idx)) {
        return 2000 + cfg->array_shape;
    }
    *wall_sum += snrt_mcycle() - start;
    *accel_sum += read_dual_versacore_perf_counter();
    *streamer_sum += read_dual_versacore_streamer_perf_counter();
    return 0;
}

static int run_mode1_chunk(const shape_cfg_t *cfg, int32_t b0_base,
                           int32_t b1_base, uint32_t n_start,
                           uint32_t n_tiles, uint32_t chunk_idx,
                           uint32_t *accel_sum, uint32_t *streamer_sum,
                           uint32_t *wall_sum) {
    int32_t a_tbound[6], b_tbound[4], d_tbound[4];
    int32_t d0_base =
        cfg->delta_local_mode1_d0 + n_start * cfg->mode1_D_tstride[2];
    int32_t d1_base =
        cfg->delta_local_mode1_d1 + n_start * cfg->mode1_D_tstride[2];
    uint32_t subtraction_setting =
        gen_dual_vc_subtraction_config(SUBTRACTION_A, SUBTRACTION_B);

    copy_i32(a_tbound, cfg->mode1_A_tbound, 6);
    copy_i32(b_tbound, cfg->mode1_B_tbound, 4);
    copy_i32(d_tbound, cfg->mode1_D_tbound, 4);
    a_tbound[1] = n_tiles;
    b_tbound[1] = n_tiles;
    d_tbound[2] = n_tiles;

    uint32_t start = snrt_mcycle();
    set_dual_versacore_streamer_csr(
        cfg->delta_local_d0, cfg->mode1_A_sstride, a_tbound,
        cfg->mode1_A_tstride, SET_ADDR_REMAP_INDEX_A, cfg->A_channel_en,
        b0_base, cfg->mode1_B_sstride, b_tbound, cfg->mode1_B_tstride,
        SET_ADDR_REMAP_INDEX_B0, cfg->B_channel_en,
        b1_base, cfg->mode1_B_sstride, b_tbound, cfg->mode1_B_tstride,
        SET_ADDR_REMAP_INDEX_B1, cfg->B_channel_en,
        d0_base, cfg->D_sstride, d_tbound, cfg->mode1_D_tstride,
        SET_ADDR_REMAP_INDEX_D0, cfg->D_channel_en,
        d1_base, cfg->D_sstride, d_tbound, cfg->mode1_D_tstride,
        SET_ADDR_REMAP_INDEX_D1, cfg->D_channel_en);

    set_dual_versacore_csr(1, cfg->K1, n_tiles * cfg->M_tiles,
                           subtraction_setting, cfg->array_shape, DATA_TYPE);
    set_dual_versacore_mode(1);
    set_dual_versacore_rescale0(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale1(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);

    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    if (wait_accel_diag(cfg->array_shape, 1, chunk_idx)) {
        return 3000 + cfg->array_shape;
    }
    if (wait_writer_diag(cfg->array_shape, 1, chunk_idx)) {
        return 4000 + cfg->array_shape;
    }
    *wall_sum += snrt_mcycle() - start;
    *accel_sum += read_dual_versacore_perf_counter();
    *streamer_sum += read_dual_versacore_streamer_perf_counter();
    return 0;
}

static void set_base_ptr(uint32_t csr, int32_t delta_local) {
    csrw_ss(csr, (uint32_t)(delta_local + snrt_l1_next()));
}

static int run_mode0_chunk_base_only(const shape_cfg_t *cfg, int32_t b0_base,
                                     int32_t b1_base, uint32_t n_start,
                                     uint32_t n_tiles, uint32_t chunk_idx,
                                     uint32_t full_chunk_tiles,
                                     uint32_t *accel_sum,
                                     uint32_t *streamer_sum,
                                     uint32_t *wall_sum) {
    int32_t d_base = cfg->delta_local_d0 + n_start * cfg->mode0_D_tstride[1];

    uint32_t start = snrt_mcycle();
    set_base_ptr(BASE_PTR_READER_1_LOW, b0_base);
    set_base_ptr(BASE_PTR_READER_2_LOW, b1_base);
    set_base_ptr(BASE_PTR_WRITER_0_LOW, d_base);
    if (n_tiles != full_chunk_tiles) {
        csrw_ss(T_BOUND_READER_0_1, n_tiles);
        csrw_ss(T_BOUND_READER_1_1, n_tiles);
        csrw_ss(T_BOUND_READER_2_1, n_tiles);
        csrw_ss(T_BOUND_WRITER_0_1, n_tiles);
        csrw_ss(DUAL_VC_OUTPUT_BOUND, n_tiles * cfg->M_tiles);
    }

    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    if (wait_accel_diag(cfg->array_shape, 0, chunk_idx)) {
        return 1000 + cfg->array_shape;
    }
    if (wait_writer_diag(cfg->array_shape, 0, chunk_idx)) {
        return 2000 + cfg->array_shape;
    }
    *wall_sum += snrt_mcycle() - start;
    *accel_sum += read_dual_versacore_perf_counter();
    *streamer_sum += read_dual_versacore_streamer_perf_counter();
    return 0;
}

static int run_mode1_chunk_base_only(const shape_cfg_t *cfg, int32_t b0_base,
                                     int32_t b1_base, uint32_t n_start,
                                     uint32_t n_tiles, uint32_t chunk_idx,
                                     uint32_t full_chunk_tiles,
                                     uint32_t *accel_sum,
                                     uint32_t *streamer_sum,
                                     uint32_t *wall_sum) {
    int32_t d0_base =
        cfg->delta_local_mode1_d0 + n_start * cfg->mode1_D_tstride[2];
    int32_t d1_base =
        cfg->delta_local_mode1_d1 + n_start * cfg->mode1_D_tstride[2];

    uint32_t start = snrt_mcycle();
    set_base_ptr(BASE_PTR_READER_1_LOW, b0_base);
    set_base_ptr(BASE_PTR_READER_2_LOW, b1_base);
    set_base_ptr(BASE_PTR_WRITER_0_LOW, d0_base);
    set_base_ptr(BASE_PTR_WRITER_1_LOW, d1_base);
    if (n_tiles != full_chunk_tiles) {
        csrw_ss(T_BOUND_READER_0_1, n_tiles);
        csrw_ss(T_BOUND_READER_1_1, n_tiles);
        csrw_ss(T_BOUND_READER_2_1, n_tiles);
        csrw_ss(T_BOUND_WRITER_0_2, n_tiles);
        csrw_ss(T_BOUND_WRITER_1_2, n_tiles);
        csrw_ss(DUAL_VC_OUTPUT_BOUND, n_tiles * cfg->M_tiles);
    }

    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    if (wait_accel_diag(cfg->array_shape, 1, chunk_idx)) {
        return 3000 + cfg->array_shape;
    }
    if (wait_writer_diag(cfg->array_shape, 1, chunk_idx)) {
        return 4000 + cfg->array_shape;
    }
    *wall_sum += snrt_mcycle() - start;
    *accel_sum += read_dual_versacore_perf_counter();
    *streamer_sum += read_dual_versacore_streamer_perf_counter();
    return 0;
}

static void dma_mode0_panel(const shape_cfg_t *cfg, uint8_t *local_b0,
                            uint8_t *local_b1, uint32_t n_start,
                            uint32_t n_tiles) {
    uint32_t stride = cfg->mode0_B_tstride[1];
    uint32_t bytes = stride * n_tiles;
    uint32_t offset = stride * n_start;
    snrt_dma_start_1d(local_b0, (uint8_t *)W + offset, bytes);
    snrt_dma_start_1d(local_b1, (uint8_t *)V + offset, bytes);
    snrt_dma_wait_all();
}

static void dma_mode1_panel(const shape_cfg_t *cfg, uint8_t *local_b0,
                            uint8_t *local_b1, uint32_t n_start,
                            uint32_t n_tiles) {
    uint32_t stride = cfg->mode1_B_tstride[1];
    uint32_t bytes = stride * n_tiles;
    uint32_t offset = stride * n_start;
    snrt_dma_start_1d(local_b0, (uint8_t *)W2_left + offset, bytes);
    snrt_dma_start_1d(local_b1, (uint8_t *)W2_right + offset, bytes);
    snrt_dma_wait_all();
}

static int run_mode0_overlap_all_roles(const shape_cfg_t *cfg,
                                       uint32_t chunk_tiles,
                                       uint32_t *accel_sum,
                                       uint32_t *streamer_sum,
                                       uint32_t *wall_sum) {
    uint32_t chunks = (cfg->N_tiles + chunk_tiles - 1u) / chunk_tiles;
    uint32_t panel_max = align_up_u32(cfg->mode0_B_tstride[1] * chunk_tiles, 64);
    uint8_t *b0_buf[2] = {
        (uint8_t *)(snrt_l1_next() + cfg->delta_local_b0),
        (uint8_t *)(snrt_l1_next() + cfg->delta_local_b0 + panel_max),
    };
    uint8_t *b1_buf[2] = {
        (uint8_t *)(snrt_l1_next() + cfg->delta_local_b1),
        (uint8_t *)(snrt_l1_next() + cfg->delta_local_b1 + panel_max),
    };

    if (snrt_is_dm_core()) {
        uint32_t first_tiles = min_u32(chunk_tiles, cfg->N_tiles);
        dma_mode0_panel(cfg, b0_buf[0], b1_buf[0], 0, first_tiles);
    }
    snrt_cluster_hw_barrier();

    for (uint32_t c = 0; c < chunks; c++) {
        uint32_t cur = c & 1u;
        uint32_t next = (c + 1u) & 1u;
        uint32_t n_start = c * chunk_tiles;
        uint32_t n_tiles = min_u32(chunk_tiles, cfg->N_tiles - n_start);

        snrt_cluster_hw_barrier();
        if (snrt_global_core_idx() == 0) {
            int32_t b0_base =
                cfg->delta_local_b0 + (cur ? (int32_t)panel_max : 0);
            int32_t b1_base =
                cfg->delta_local_b1 + (cur ? (int32_t)panel_max : 0);
            int rc;
            if (strategy_base_only_updates() && c > 0) {
                rc = run_mode0_chunk_base_only(
                    cfg, b0_base, b1_base, n_start, n_tiles, c, chunk_tiles,
                    accel_sum, streamer_sum, wall_sum);
            } else {
                rc = run_mode0_chunk(cfg, b0_base, b1_base, n_start, n_tiles,
                                     c, accel_sum, streamer_sum, wall_sum);
            }
            if (rc) {
                return rc;
            }
        } else if (snrt_is_dm_core() && c + 1u < chunks) {
            uint32_t next_start = (c + 1u) * chunk_tiles;
            uint32_t next_tiles =
                min_u32(chunk_tiles, cfg->N_tiles - next_start);
            dma_mode0_panel(cfg, b0_buf[next], b1_buf[next], next_start,
                            next_tiles);
        }
        snrt_cluster_hw_barrier();
    }
    return 0;
}

static int run_mode1_overlap_all_roles(const shape_cfg_t *cfg,
                                       uint32_t chunk_tiles,
                                       uint32_t *accel_sum,
                                       uint32_t *streamer_sum,
                                       uint32_t *wall_sum) {
    uint32_t chunks = (cfg->N1 + chunk_tiles - 1u) / chunk_tiles;
    uint32_t panel_max = align_up_u32(cfg->mode1_B_tstride[1] * chunk_tiles, 64);
    uint8_t *b0_buf[2] = {
        (uint8_t *)(snrt_l1_next() + cfg->delta_local_w2l),
        (uint8_t *)(snrt_l1_next() + cfg->delta_local_w2l + panel_max),
    };
    uint8_t *b1_buf[2] = {
        (uint8_t *)(snrt_l1_next() + cfg->delta_local_w2r),
        (uint8_t *)(snrt_l1_next() + cfg->delta_local_w2r + panel_max),
    };

    if (snrt_is_dm_core()) {
        uint32_t first_tiles = min_u32(chunk_tiles, cfg->N1);
        dma_mode1_panel(cfg, b0_buf[0], b1_buf[0], 0, first_tiles);
    }
    snrt_cluster_hw_barrier();

    for (uint32_t c = 0; c < chunks; c++) {
        uint32_t cur = c & 1u;
        uint32_t next = (c + 1u) & 1u;
        uint32_t n_start = c * chunk_tiles;
        uint32_t n_tiles = min_u32(chunk_tiles, cfg->N1 - n_start);

        snrt_cluster_hw_barrier();
        if (snrt_global_core_idx() == 0) {
            int32_t b0_base =
                cfg->delta_local_w2l + (cur ? (int32_t)panel_max : 0);
            int32_t b1_base =
                cfg->delta_local_w2r + (cur ? (int32_t)panel_max : 0);
            int rc;
            if (strategy_base_only_updates() && c > 0) {
                rc = run_mode1_chunk_base_only(
                    cfg, b0_base, b1_base, n_start, n_tiles, c, chunk_tiles,
                    accel_sum, streamer_sum, wall_sum);
            } else {
                rc = run_mode1_chunk(cfg, b0_base, b1_base, n_start, n_tiles,
                                     c, accel_sum, streamer_sum, wall_sum);
            }
            if (rc) {
                return rc;
            }
        } else if (snrt_is_dm_core() && c + 1u < chunks) {
            uint32_t next_start = (c + 1u) * chunk_tiles;
            uint32_t next_tiles = min_u32(chunk_tiles, cfg->N1 - next_start);
            dma_mode1_panel(cfg, b0_buf[next], b1_buf[next], next_start,
                            next_tiles);
        }
        snrt_cluster_hw_barrier();
    }
    return 0;
}

static int run_baseline_all_roles(const shape_cfg_t *cfg) {
    int err = 0;
    int16_t *local_a = (int16_t *)(snrt_l1_next() + cfg->delta_local_a);
    uint8_t *local_b0 = (uint8_t *)(snrt_l1_next() + cfg->delta_local_b0);
    uint8_t *local_b1 = (uint8_t *)(snrt_l1_next() + cfg->delta_local_b1);
    uint8_t *local_w2l = (uint8_t *)(snrt_l1_next() + cfg->delta_local_w2l);
    uint8_t *local_w2r = (uint8_t *)(snrt_l1_next() + cfg->delta_local_w2r);

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
        uint32_t m0_accel = 0, m0_streamer = 0, m0_wall = 0;
        uint32_t m1_accel = 0, m1_streamer = 0, m1_wall = 0;
        int rc = run_mode0_full(cfg, &m0_accel, &m0_streamer, &m0_wall);
        if (rc) {
            return rc;
        }

        int16_t *local_d0 = (int16_t *)(snrt_l1_next() + cfg->delta_local_d0);
        int err_m0 = check_result_i16_limited(
            local_d0, cfg->mode0_d0_golden, cfg->mode0_output_elems);
        err += err_m0;

#if RUN_MODE1
        rc = run_mode1_full(cfg, &m1_accel, &m1_streamer, &m1_wall);
        if (rc) {
            return rc;
        }
        int16_t *local_mode1_d0 =
            (int16_t *)(snrt_l1_next() + cfg->delta_local_mode1_d0);
        int16_t *local_mode1_d1 =
            (int16_t *)(snrt_l1_next() + cfg->delta_local_mode1_d1);
        int err_m1_d0 = check_result_i16_limited(
            local_mode1_d0, cfg->mode1_d0_golden, cfg->mode1_output_elems);
        int err_m1_d1 = check_result_i16_limited(
            local_mode1_d1, cfg->mode1_d1_golden, cfg->mode1_output_elems);
        err += err_m1_d0 + err_m1_d1;
#endif

        printf("RESULT strategy=%d shape=%u mode=0 status=%s accel=%u streamer=%u compute_wall=%u\n",
               SELECT_STRATEGY, cfg->array_shape, err_m0 ? "FAIL" : "PASS",
               m0_accel, m0_streamer, m0_wall);
#if RUN_MODE1
        printf("RESULT strategy=%d shape=%u mode=1 status=%s accel=%u streamer=%u compute_wall=%u\n",
               SELECT_STRATEGY, cfg->array_shape,
               err ? "FAIL" : "PASS", m1_accel, m1_streamer, m1_wall);
#endif
    }

    snrt_cluster_hw_barrier();
    return err;
}

static int run_overlap_all_roles(const shape_cfg_t *cfg, uint32_t chunk_tiles) {
    int err = 0;
    int16_t *local_a = (int16_t *)(snrt_l1_next() + cfg->delta_local_a);

    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_a, A, A_DATA_LENGTH);
        snrt_dma_wait_all();
    }
    snrt_cluster_hw_barrier();

    uint32_t m0_accel = 0, m0_streamer = 0, m0_wall = 0;
    int rc = run_mode0_overlap_all_roles(cfg, chunk_tiles, &m0_accel,
                                         &m0_streamer, &m0_wall);
    if (rc) {
        return rc;
    }

    if (snrt_global_core_idx() == 0) {
        int16_t *local_d0 = (int16_t *)(snrt_l1_next() + cfg->delta_local_d0);
        int err_m0 = check_result_i16_limited(
            local_d0, cfg->mode0_d0_golden, cfg->mode0_output_elems);
        err += err_m0;
        printf("RESULT strategy=%d shape=%u mode=0 status=%s accel_sum=%u streamer_sum=%u compute_wall_sum=%u chunk_tiles=%u\n",
               SELECT_STRATEGY, cfg->array_shape,
               err_m0 ? "FAIL" : "PASS", m0_accel, m0_streamer, m0_wall,
               chunk_tiles);
    }

#if RUN_MODE1
    snrt_cluster_hw_barrier();
    uint32_t m1_accel = 0, m1_streamer = 0, m1_wall = 0;
    rc = run_mode1_overlap_all_roles(cfg, chunk_tiles, &m1_accel,
                                     &m1_streamer, &m1_wall);
    if (rc) {
        return rc;
    }

    if (snrt_global_core_idx() == 0) {
        int16_t *local_mode1_d0 =
            (int16_t *)(snrt_l1_next() + cfg->delta_local_mode1_d0);
        int16_t *local_mode1_d1 =
            (int16_t *)(snrt_l1_next() + cfg->delta_local_mode1_d1);
        int err_m1_d0 = check_result_i16_limited(
            local_mode1_d0, cfg->mode1_d0_golden, cfg->mode1_output_elems);
        int err_m1_d1 = check_result_i16_limited(
            local_mode1_d1, cfg->mode1_d1_golden, cfg->mode1_output_elems);
        err += err_m1_d0 + err_m1_d1;
        printf("RESULT strategy=%d shape=%u mode=1 status=%s accel_sum=%u streamer_sum=%u compute_wall_sum=%u chunk_tiles=%u\n",
               SELECT_STRATEGY, cfg->array_shape,
               (err_m1_d0 || err_m1_d1) ? "FAIL" : "PASS", m1_accel,
               m1_streamer, m1_wall, chunk_tiles);
    }
#endif

    snrt_cluster_hw_barrier();
    return err;
}

int main() {
    if (SELECT_SHAPE < 0 || SELECT_SHAPE >= NUM_SHAPES) {
        if (snrt_global_core_idx() == 0) {
            printf("Invalid SELECT_SHAPE=%d, NUM_SHAPES=%d\n", SELECT_SHAPE,
                   NUM_SHAPES);
        }
        return 1;
    }
    if (SELECT_STRATEGY < 0 || SELECT_STRATEGY > 25) {
        if (snrt_global_core_idx() == 0) {
            printf("Invalid SELECT_STRATEGY=%d\n", SELECT_STRATEGY);
        }
        return 1;
    }

    const shape_cfg_t *cfg = &shape_cfg[SELECT_SHAPE];
    if (cfg->tcdm_end > TCDM_CAPACITY_BYTES) {
        if (snrt_global_core_idx() == 0) {
            printf("DMA-overlap TCDM placement exceeds capacity: end=%d cap=%d\n",
                   cfg->tcdm_end, TCDM_CAPACITY_BYTES);
        }
        return 1;
    }

    uint32_t chunk_tiles = strategy_chunk_tiles();
    if (SELECT_STRATEGY != 0 && chunk_tiles == 0) {
        return 1;
    }

    if (snrt_global_core_idx() == 0) {
        printf("DMA-overlap app strategy=%d %s shape=%u chunk_tiles=%u\n",
               SELECT_STRATEGY, strategy_name(), cfg->array_shape,
               chunk_tiles);
    }

    snrt_cluster_hw_barrier();
    uint32_t total_start = 0;
    if (snrt_global_core_idx() == 0) {
        total_start = snrt_mcycle();
    }

    int err = 0;
    if (SELECT_STRATEGY == 0) {
        err = run_baseline_all_roles(cfg);
    } else {
        err = run_overlap_all_roles(cfg, chunk_tiles);
    }

    if (snrt_global_core_idx() == 0) {
        uint32_t total_wall = snrt_mcycle() - total_start;
        printf("RESULT strategy=%d shape=%u mode=total status=%s total_wall=%u chunk_tiles=%u error=%d\n",
               SELECT_STRATEGY, cfg->array_shape, err ? "FAIL" : "PASS",
               total_wall, chunk_tiles, err);
    }

    return err;
}
