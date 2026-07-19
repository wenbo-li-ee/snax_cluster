// Resident-weight common-16-column chunk experiment.
//
// W/V=[2048,1024] and logical W2=[1024,2048] are staged exactly once.  W2 is
// physically split into W2_left/right=[1024,1024].  Mode0 occupies the first
// 8192 TCDM rows of banks 16..47 and Mode1 the next 4096 rows.  Every chunk
// keeps its ping/pong bank phase and receives a unique depth slot, so all
// weights remain resident while S0/S1/S2 run back-to-back.

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

#define WAIT_TIMEOUT_CYCLES 1000000u
#define CHECK_PRINT_LIMIT 4u
#define TCDM_ROW_BYTES 512u
#define BANK_WORD_BYTES 8u

#define A_BASE 0
#define B0_PING_BASE (16 * BANK_WORD_BYTES)
#define B1_PING_BASE (24 * BANK_WORD_BYTES)
#define B0_PONG_BASE (32 * BANK_WORD_BYTES)
#define B1_PONG_BASE (40 * BANK_WORD_BYTES)
#define MODE0_D_BASE (48 * BANK_WORD_BYTES)
#define MODE1_D0_BASE 0
#define MODE1_D1_BASE (8 * BANK_WORD_BYTES)

#ifndef SELECT_SHAPE
#define SELECT_SHAPE -1
#endif
#ifndef CHUNK_COLS
#define CHUNK_COLS 16
#endif
#ifndef RUN_MODE1
#define RUN_MODE1 1
#endif

static uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

static uint32_t ceil_div_u32(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
}

static uint32_t mode_panel_bytes(int mode) {
    return mode ? (K1_TOTAL / 8u) * 16u : (K0_TOTAL / 8u) * 16u;
}

static uint32_t mode_panel_span(int mode) {
    return (mode_panel_bytes(mode) / 64u) * TCDM_ROW_BYTES;
}

static uint32_t mode_total_cols(int mode) {
    return mode ? N1_TOTAL : N0_TOTAL;
}

static uint32_t chunk_slot_span(int mode) {
    return (CHUNK_COLS / 4u) * mode_panel_span(mode);
}

static uint32_t chunk_count(int mode) {
    return ceil_div_u32(mode_total_cols(mode), CHUNK_COLS);
}

static uint32_t mode_region_span(int mode) {
    uint32_t slots = ceil_div_u32(chunk_count(mode), 2u);
    return slots * chunk_slot_span(mode);
}

static uint32_t mode_region_offset(int mode) {
    return mode ? mode_region_span(0) : 0u;
}

static int32_t weight_base(int mode, int right, uint32_t chunk) {
    uint32_t slot = chunk >> 1;
    uint32_t pong = chunk & 1u;
    uint32_t base;
    if (!pong) {
        base = right ? B1_PING_BASE : B0_PING_BASE;
    } else {
        base = right ? B1_PONG_BASE : B0_PONG_BASE;
    }
    return (int32_t)(mode_region_offset(mode) + base +
                     slot * chunk_slot_span(mode));
}

static uint32_t mode_region_end(int mode) {
    return mode_region_offset(mode) + mode_region_span(mode);
}

static int wait_accel_diag(int shape, int mode, uint32_t chunk) {
    csrw_ss(DUAL_VC_START, 0);
    csrw_ss(DUAL_VC_START, 0);
    uint32_t start = snrt_mcycle();
    while (csrr_ss(DUAL_VC_BUSY)) {
        if ((uint32_t)(snrt_mcycle() - start) > WAIT_TIMEOUT_CYCLES) {
            printf("TIMEOUT shape=%d mode=%d chunk=%u unit=accelerator accel=%u streamer=%u w0=%u w1=%u\n",
                   shape, mode, chunk, csrr_ss(DUAL_VC_BUSY),
                   csrr_ss(STREAMER_BUSY_CSR), csrr_ss(WRITER_BUSY_CSR),
                   csrr_ss(WRITER1_BUSY_CSR));
            return 1;
        }
    }
    return 0;
}

static int wait_streamer_diag(int shape, int mode, uint32_t chunk) {
    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    uint32_t start = snrt_mcycle();
    while (csrr_ss(STREAMER_BUSY_CSR)) {
        if ((uint32_t)(snrt_mcycle() - start) > WAIT_TIMEOUT_CYCLES) {
            printf("TIMEOUT shape=%d mode=%d chunk=%u unit=streamer streamer=%u w0=%u w1=%u\n",
                   shape, mode, chunk, csrr_ss(STREAMER_BUSY_CSR),
                   csrr_ss(WRITER_BUSY_CSR), csrr_ss(WRITER1_BUSY_CSR));
            return 1;
        }
    }
    return 0;
}

static void stage_tokens(const shape_cfg_t *cfg) {
    uint8_t *tcdm = (uint8_t *)snrt_l1_next();
    uint32_t start = snrt_mcycle();
    for (uint32_t token = 0; token < M_TOTAL; token++) {
        const int16_t *src = A + token * K0_TOTAL;
        uint8_t *dst = tcdm + A_BASE + token * 16u;
        snrt_dma_start_2d(dst, src, 16, TCDM_ROW_BYTES, 16,
                          K0_TOTAL / 8u);
    }
    snrt_dma_wait_all();
    printf("TOKEN_DMA shape=%d active_tokens=%d cycles=%u base=%d banks=0..15 size=16 dst_stride=512 src_stride=16 repeat=256\n",
           cfg->array_shape, cfg->meshRow, snrt_mcycle() - start, A_BASE);
}

static void stage_weight_chunks(int mode) {
    uint8_t *tcdm = (uint8_t *)snrt_l1_next();
    const uint8_t *left = mode ? W2_left : W;
    const uint8_t *right = mode ? W2_right : V;
    uint32_t total_cols = mode_total_cols(mode);
    uint32_t panel_bytes = mode_panel_bytes(mode);
    uint32_t panels_per_chunk = CHUNK_COLS / 4u;
    uint32_t chunks = chunk_count(mode);
    uint32_t total_start = snrt_mcycle();

    for (uint32_t chunk = 0; chunk < chunks; chunk++) {
        uint32_t col_start = chunk * CHUNK_COLS;
        uint32_t cols = min_u32(CHUNK_COLS, total_cols - col_start);
        uint32_t panels = cols / 4u;
        int32_t left_base = weight_base(mode, 0, chunk);
        int32_t right_base = weight_base(mode, 1, chunk);
        uint32_t chunk_start = snrt_mcycle();
        uint32_t global_panel = chunk * panels_per_chunk;
        uint32_t chunk_bytes = panels * panel_bytes;
        uint32_t repeats = chunk_bytes / 64u;
        const uint8_t *src_left = left + global_panel * panel_bytes;
        const uint8_t *src_right = right + global_panel * panel_bytes;

        // The four canonical 4-column panels of a 16-column chunk are
        // contiguous both in L3 and along TCDM depth.  Collapse them into one
        // 2D descriptor per tensor; panel boundaries do not change either
        // stride.  The overlap/synchronization granularity is therefore one
        // complete CHUNK_COLS-wide full-K transfer.
        snrt_dma_start_2d(tcdm + left_base, src_left, 64, TCDM_ROW_BYTES, 64,
                          repeats);
        snrt_dma_start_2d(tcdm + right_base, src_right, 64, TCDM_ROW_BYTES, 64,
                          repeats);
        snrt_dma_wait_all();
        printf("WEIGHT_DMA_CHUNK mode=%d chunk=%u cols=%u..%u buffer=%s slot=%u left_base=%d right_base=%d descriptors=2 panels=%u panel_bytes=%u chunk_bytes_per_tensor=%u size=64 dst_stride=512 src_stride=64 repeat=%u cycles=%u\n",
               mode, chunk, col_start, col_start + cols,
               (chunk & 1u) ? "pong" : "ping", chunk >> 1, left_base,
               right_base, panels, panel_bytes, chunk_bytes, repeats,
               snrt_mcycle() - chunk_start);
    }
    printf("WEIGHT_DMA_SUMMARY mode=%d chunks=%u chunk_cols=%d commands=%u total_cycles=%u region_offset=%u region_span=%u region_end=%u\n",
           mode, chunks, CHUNK_COLS, chunks * 2u,
           snrt_mcycle() - total_start, mode_region_offset(mode),
           mode_region_span(mode), mode_region_end(mode));
}

static int check_mode0_chunk(const shape_cfg_t *cfg, uint32_t col_start,
                             uint32_t cols) {
    const uint8_t *tcdm = (const uint8_t *)snrt_l1_next();
    uint32_t errors = 0;
    for (int token = 0; token < cfg->meshRow; token++) {
        for (uint32_t elem = col_start; elem < col_start + cols; elem++) {
            uint32_t group8 = elem / 8u;
            uint32_t lane = elem % 8u;
            uint32_t token_offset;
            uint32_t group_offset;
            if (cfg->array_shape == 0) {
                token_offset = token * 16u;
                group_offset = group8 * TCDM_ROW_BYTES;
            } else if (cfg->array_shape == 1) {
                token_offset = token * 16u;
                group_offset = (group8 % 2u) * 64u +
                               (group8 / 2u) * TCDM_ROW_BYTES;
            } else {
                token_offset = token * 32u;
                group_offset = (group8 % 2u) * 16u +
                               ((group8 / 2u) % 2u) * 64u +
                               (group8 / 4u) * TCDM_ROW_BYTES;
            }
            const int16_t *actual = (const int16_t *)(
                tcdm + MODE0_D_BASE + token_offset + group_offset + lane * 2u);
            int16_t expected = cfg->mode0_token_golden[token * N0_TOTAL + elem];
            if (*actual != expected) {
                if (errors < CHECK_PRINT_LIMIT) {
                    printf("MISMATCH shape=%d mode=0 token=%d col=%u got=%d expected=%d\n",
                           cfg->array_shape, token, elem, *actual, expected);
                }
                errors++;
            }
        }
    }
    return (int)errors;
}

static int check_mode1_chunk(const shape_cfg_t *cfg, int right,
                             uint32_t col_start, uint32_t cols) {
    const uint8_t *tcdm = (const uint8_t *)snrt_l1_next();
    const int16_t *golden = right ? cfg->mode1_d1_token_golden
                                  : cfg->mode1_d0_token_golden;
    uint32_t base = right ? MODE1_D1_BASE : MODE1_D0_BASE;
    uint32_t errors = 0;
    for (int token = 0; token < cfg->meshRow; token++) {
        for (uint32_t elem = col_start; elem < col_start + cols; elem++) {
            uint32_t beat = elem / 4u;
            uint32_t lane = elem % 4u;
            const int16_t *actual = (const int16_t *)(
                tcdm + base + token * 8u + beat * TCDM_ROW_BYTES + lane * 2u);
            int16_t expected = golden[token * N1_TOTAL + elem];
            if (*actual != expected) {
                if (errors < CHECK_PRINT_LIMIT) {
                    printf("MISMATCH shape=%d mode=1 writer=D%d token=%d col=%u got=%d expected=%d\n",
                           cfg->array_shape, right, token, elem, *actual,
                           expected);
                }
                errors++;
            }
        }
    }
    return (int)errors;
}

static void mode0_writer_cfg(const shape_cfg_t *cfg, uint32_t col_start,
                             uint32_t n_tiles, int32_t *base,
                             int32_t bound[4], int32_t stride[4]) {
    if (cfg->array_shape == 0) {
        *base = MODE0_D_BASE + (int32_t)((col_start / 8u) * TCDM_ROW_BYTES);
        bound[0] = 1;
        bound[1] = 8;
        bound[2] = 2;
        bound[3] = n_tiles / 2u;
        stride[0] = 8;
        stride[1] = 16;
        stride[2] = 8;
        stride[3] = TCDM_ROW_BYTES;
    } else if (cfg->array_shape == 1) {
        *base = MODE0_D_BASE + (int32_t)((col_start / 16u) * TCDM_ROW_BYTES);
        bound[0] = 2;
        bound[1] = 4;
        bound[2] = 2;
        bound[3] = n_tiles / 2u;
        stride[0] = 8;
        stride[1] = 16;
        stride[2] = 64;
        stride[3] = TCDM_ROW_BYTES;
    } else if (n_tiles == 1u) {
        uint32_t global_tile = col_start / 16u;
        *base = MODE0_D_BASE +
                (int32_t)((global_tile / 2u) * TCDM_ROW_BYTES) +
                (int32_t)((global_tile & 1u) * 64u);
        bound[0] = 4;
        bound[1] = 2;
        bound[2] = 1;
        bound[3] = 1;
        stride[0] = 8;
        stride[1] = 32;
        stride[2] = 64;
        stride[3] = TCDM_ROW_BYTES;
    } else {
        uint32_t global_tile = col_start / 16u;
        *base = MODE0_D_BASE +
                (int32_t)((global_tile / 2u) * TCDM_ROW_BYTES);
        bound[0] = 4;
        bound[1] = 2;
        bound[2] = 2;
        bound[3] = n_tiles / 2u;
        stride[0] = 8;
        stride[1] = 32;
        stride[2] = 64;
        stride[3] = TCDM_ROW_BYTES;
    }
}

static int run_mode0_chunk(const shape_cfg_t *cfg, uint32_t chunk,
                           uint32_t col_start, uint32_t cols,
                           uint32_t *accel_sum, uint32_t *streamer_sum,
                           uint32_t *wall_sum) {
    uint32_t n_tiles = cols / cfg->meshCol;
    uint32_t panel_span = mode_panel_span(0);
    int32_t a_sstride[2] = {8, 16};
    int32_t a_tbound[6] = {cfg->K0_tiles, n_tiles, 1, 1, 1, 1};
    int32_t a_tstride[6] = {TCDM_ROW_BYTES, 0, 0, 0, 0, 0};
    int32_t b_sstride[2] = {8, (int32_t)panel_span};
    int32_t b_tbound[4] = {4, cfg->K0_tiles / 4, n_tiles, 1};
    int32_t b_tstride[4] = {
        16, TCDM_ROW_BYTES, cfg->q_shape0_cols * (int32_t)panel_span, 0};
    int32_t d_sstride[1] = {8};
    int32_t d_tbound[4], d_tstride[4], d_base;
    int32_t b0_base = weight_base(0, 0, chunk);
    int32_t b1_base = weight_base(0, 1, chunk);
    uint32_t subtraction =
        gen_dual_vc_subtraction_config(SUBTRACTION_A, SUBTRACTION_B);

    mode0_writer_cfg(cfg, col_start, n_tiles, &d_base, d_tbound, d_tstride);
    uint32_t start = snrt_mcycle();
    set_dual_versacore_streamer_csr_d0_only(
        A_BASE, a_sstride, a_tbound, a_tstride,
        SET_ADDR_REMAP_INDEX_A, cfg->A_channel_en,
        b0_base, b_sstride, b_tbound, b_tstride,
        SET_ADDR_REMAP_INDEX_B0, cfg->B_channel_en,
        b1_base, b_sstride, b_tbound, b_tstride,
        SET_ADDR_REMAP_INDEX_B1, cfg->B_channel_en,
        d_base, d_sstride, d_tbound, d_tstride,
        SET_ADDR_REMAP_INDEX_D0, cfg->D_channel_en);
    set_dual_versacore_csr(1, cfg->K0_tiles, n_tiles, subtraction,
                           cfg->array_shape, DATA_TYPE);
    set_dual_versacore_mode(0);
    set_dual_versacore_rescale0(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale1(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale_mul(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                   RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    if (wait_accel_diag(cfg->array_shape, 0, chunk) ||
        wait_streamer_diag(cfg->array_shape, 0, chunk)) {
        return 1000000;
    }
    uint32_t wall = snrt_mcycle() - start;
    uint32_t accel = read_dual_versacore_perf_counter();
    uint32_t streamer = read_dual_versacore_streamer_perf_counter();
    int errors = check_mode0_chunk(cfg, col_start, cols);
    *wall_sum += wall;
    *accel_sum += accel;
    *streamer_sum += streamer;
    printf("CHUNK_RESULT shape=%d mode=0 chunk=%u cols=%u..%u n_tiles=%u buffer=%s b0_base=%d b1_base=%d d_base=%d status=%s errors=%d accel=%u streamer=%u wall=%u\n",
           cfg->array_shape, chunk, col_start, col_start + cols, n_tiles,
           (chunk & 1u) ? "pong" : "ping", b0_base, b1_base, d_base,
           errors ? "FAIL" : "PASS", errors, accel, streamer, wall);
    return errors;
}

static void mode1_a_cfg(const shape_cfg_t *cfg, uint32_t n_tiles,
                        int32_t sstride[2], int32_t bound[6],
                        int32_t stride[6]) {
    for (int i = 0; i < 6; i++) {
        bound[i] = 1;
        stride[i] = 0;
    }
    if (cfg->array_shape == 0) {
        sstride[0] = 8;
        sstride[1] = 16;
        bound[0] = cfg->K1_tiles;
        bound[1] = n_tiles;
        stride[0] = TCDM_ROW_BYTES;
    } else if (cfg->array_shape == 1) {
        sstride[0] = 8;
        sstride[1] = 16;
        bound[0] = 2;
        bound[1] = cfg->K1_tiles / 2;
        bound[2] = n_tiles;
        stride[0] = 64;
        stride[1] = TCDM_ROW_BYTES;
    } else {
        sstride[0] = 8;
        sstride[1] = 32;
        bound[0] = 2;
        bound[1] = 2;
        bound[2] = cfg->K1_tiles / 4;
        bound[3] = n_tiles;
        stride[0] = 16;
        stride[1] = 64;
        stride[2] = TCDM_ROW_BYTES;
    }
}

static int run_mode1_chunk(const shape_cfg_t *cfg, uint32_t chunk,
                           uint32_t col_start, uint32_t cols,
                           uint32_t *accel_sum, uint32_t *streamer_sum,
                           uint32_t *wall_sum) {
    uint32_t n_tiles = cols / cfg->meshCol;
    uint32_t panel_span = mode_panel_span(1);
    uint32_t beats_per_tile = cfg->meshCol / 4u;
    int32_t a_sstride[2], a_tbound[6], a_tstride[6];
    int32_t b_sstride[2] = {8, (int32_t)panel_span};
    int32_t b_tbound[4] = {4, cfg->K1_tiles / 4, n_tiles, 1};
    int32_t b_tstride[4] = {
        16, TCDM_ROW_BYTES, cfg->q_shape0_cols * (int32_t)panel_span, 0};
    int32_t d_sstride[1] = {8};
    int32_t d_tbound[4] = {
        (int32_t)beats_per_tile, cfg->meshRow, (int32_t)n_tiles, 1};
    int32_t d_tstride[4] = {
        TCDM_ROW_BYTES, 8, (int32_t)(beats_per_tile * TCDM_ROW_BYTES), 0};
    int32_t b0_base = weight_base(1, 0, chunk);
    int32_t b1_base = weight_base(1, 1, chunk);
    int32_t d0_base = MODE1_D0_BASE +
                      (int32_t)((col_start / 4u) * TCDM_ROW_BYTES);
    int32_t d1_base = MODE1_D1_BASE +
                      (int32_t)((col_start / 4u) * TCDM_ROW_BYTES);
    uint32_t subtraction =
        gen_dual_vc_subtraction_config(SUBTRACTION_A, SUBTRACTION_B);

    mode1_a_cfg(cfg, n_tiles, a_sstride, a_tbound, a_tstride);
    uint32_t start = snrt_mcycle();
    set_dual_versacore_streamer_csr(
        MODE0_D_BASE, a_sstride, a_tbound, a_tstride,
        SET_ADDR_REMAP_INDEX_A, cfg->A_channel_en,
        b0_base, b_sstride, b_tbound, b_tstride,
        SET_ADDR_REMAP_INDEX_B0, cfg->B_channel_en,
        b1_base, b_sstride, b_tbound, b_tstride,
        SET_ADDR_REMAP_INDEX_B1, cfg->B_channel_en,
        d0_base, d_sstride, d_tbound, d_tstride,
        SET_ADDR_REMAP_INDEX_D0, cfg->D_channel_en,
        d1_base, d_sstride, d_tbound, d_tstride,
        SET_ADDR_REMAP_INDEX_D1, cfg->D_channel_en);
    set_dual_versacore_csr(1, cfg->K1_tiles, n_tiles, subtraction,
                           cfg->array_shape, DATA_TYPE);
    set_dual_versacore_mode(1);
    set_dual_versacore_rescale0(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale1(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    if (wait_accel_diag(cfg->array_shape, 1, chunk) ||
        wait_streamer_diag(cfg->array_shape, 1, chunk)) {
        return 1000000;
    }
    uint32_t wall = snrt_mcycle() - start;
    uint32_t accel = read_dual_versacore_perf_counter();
    uint32_t streamer = read_dual_versacore_streamer_perf_counter();
    int err0 = check_mode1_chunk(cfg, 0, col_start, cols);
    int err1 = check_mode1_chunk(cfg, 1, col_start, cols);
    *wall_sum += wall;
    *accel_sum += accel;
    *streamer_sum += streamer;
    printf("CHUNK_RESULT shape=%d mode=1 chunk=%u cols=%u..%u n_tiles=%u buffer=%s b0_base=%d b1_base=%d d0_base=%d d1_base=%d D0=%s D1=%s errors=%d accel=%u streamer=%u wall=%u\n",
           cfg->array_shape, chunk, col_start, col_start + cols, n_tiles,
           (chunk & 1u) ? "pong" : "ping", b0_base, b1_base, d0_base,
           d1_base, err0 ? "FAIL" : "PASS", err1 ? "FAIL" : "PASS",
           err0 + err1, accel, streamer, wall);
    return err0 + err1;
}

static void print_streamer_contract(const shape_cfg_t *cfg, int mode) {
    uint32_t cols = min_u32(CHUNK_COLS, mode_total_cols(mode));
    uint32_t n_tiles = cols / cfg->meshCol;
    uint32_t span = mode_panel_span(mode);
    if (!mode) {
        printf("STREAMER_CONTRACT shape=%d mode=0 chunk_cols=%u A_sstride=[8,16] A_bound=[%d,%u,1,1,1,1] A_stride=[512,0,0,0,0,0] B_sstride=[8,%u] B_bound=[4,%d,%u,1] B_stride=[16,512,%u,0]\n",
               cfg->array_shape, cols, cfg->K0_tiles, n_tiles, span,
               cfg->K0_tiles / 4, n_tiles, cfg->q_shape0_cols * span);
    } else {
        int32_t as[2], ab[6], at[6];
        mode1_a_cfg(cfg, n_tiles, as, ab, at);
        printf("STREAMER_CONTRACT shape=%d mode=1 chunk_cols=%u A_sstride=[%d,%d] A_bound=[%d,%d,%d,%d,%d,%d] A_stride=[%d,%d,%d,%d,%d,%d] B_sstride=[8,%u] B_bound=[4,%d,%u,1] B_stride=[16,512,%u,0] D_bound=[%u,%d,%u,1] D_stride=[512,8,%u,0]\n",
               cfg->array_shape, cols, as[0], as[1], ab[0], ab[1], ab[2],
               ab[3], ab[4], ab[5], at[0], at[1], at[2], at[3], at[4],
               at[5], span, cfg->K1_tiles / 4, n_tiles,
               cfg->q_shape0_cols * span, cfg->meshCol / 4u, cfg->meshRow,
               n_tiles, (cfg->meshCol / 4u) * TCDM_ROW_BYTES);
    }
}

static int run_shape(const shape_cfg_t *cfg) {
    int errors = 0;
    if (snrt_global_core_idx() == 0) {
        printf("SHAPE_BEGIN shape=%d dims=(%d,8,%d) active_tokens=%d chunk_cols=%d mode0_chunks=%u mode1_chunks=%u\n",
               cfg->array_shape, cfg->meshRow, cfg->meshCol, cfg->meshRow,
               CHUNK_COLS, chunk_count(0), chunk_count(1));
        print_streamer_contract(cfg, 0);
#if RUN_MODE1
        print_streamer_contract(cfg, 1);
#endif
    }

    if (snrt_is_dm_core()) {
        stage_tokens(cfg);
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        uint32_t accel_sum = 0, streamer_sum = 0, wall_sum = 0;
        for (uint32_t chunk = 0; chunk < chunk_count(0); chunk++) {
            uint32_t col_start = chunk * CHUNK_COLS;
            uint32_t cols = min_u32(CHUNK_COLS, N0_TOTAL - col_start);
            errors += run_mode0_chunk(cfg, chunk, col_start, cols,
                                      &accel_sum, &streamer_sum, &wall_sum);
        }
        printf("MODE_SUMMARY shape=%d mode=0 status=%s errors=%d chunks=%u accel_sum=%u streamer_sum=%u wall_sum=%u\n",
               cfg->array_shape, errors ? "FAIL" : "PASS", errors,
               chunk_count(0), accel_sum, streamer_sum, wall_sum);
    }
    snrt_cluster_hw_barrier();

#if RUN_MODE1
    if (snrt_global_core_idx() == 0) {
        int mode1_errors = 0;
        uint32_t accel_sum = 0, streamer_sum = 0, wall_sum = 0;
        for (uint32_t chunk = 0; chunk < chunk_count(1); chunk++) {
            uint32_t col_start = chunk * CHUNK_COLS;
            uint32_t cols = min_u32(CHUNK_COLS, N1_TOTAL - col_start);
            mode1_errors += run_mode1_chunk(cfg, chunk, col_start, cols,
                                            &accel_sum, &streamer_sum,
                                            &wall_sum);
        }
        errors += mode1_errors;
        printf("MODE_SUMMARY shape=%d mode=1 status=%s errors=%d chunks=%u accel_sum=%u streamer_sum=%u wall_sum=%u output_banks=0..15\n",
               cfg->array_shape, mode1_errors ? "FAIL" : "PASS",
               mode1_errors, chunk_count(1), accel_sum, streamer_sum,
               wall_sum);
    }
    snrt_cluster_hw_barrier();
#endif

    if (snrt_global_core_idx() == 0) {
        printf("SHAPE_END shape=%d status=%s total_errors=%d\n",
               cfg->array_shape, errors ? "FAIL" : "PASS", errors);
    }
    return snrt_global_core_idx() == 0 ? errors : 0;
}

int main(void) {
    if (CHUNK_COLS < 16 || (CHUNK_COLS % 16) != 0) {
        if (snrt_global_core_idx() == 0) {
            printf("Invalid CHUNK_COLS=%d: common B granularity must be a multiple of 16 columns\n",
                   CHUNK_COLS);
        }
        return 1;
    }
    if (CHUNK_COLS != 16 && (CHUNK_COLS % 32) != 0) {
        if (snrt_global_core_idx() == 0) {
            printf("Unsupported CHUNK_COLS=%d: B supports every 16-column multiple, but the preserved S2 Mode0-D layout needs 16 or a multiple of 32 for one accelerator command per chunk\n",
                   CHUNK_COLS);
        }
        return 1;
    }
    if (SELECT_SHAPE < -1 || SELECT_SHAPE >= NUM_SHAPES) {
        if (snrt_global_core_idx() == 0) {
            printf("Invalid SELECT_SHAPE=%d\n", SELECT_SHAPE);
        }
        return 1;
    }
    uint32_t required = mode_region_end(1);
    if (required > TCDM_CAPACITY_BYTES) {
        if (snrt_global_core_idx() == 0) {
            printf("TCDM placement exceeds capacity: required=%u capacity=%u chunk_cols=%d\n",
                   required, TCDM_CAPACITY_BYTES, CHUNK_COLS);
        }
        return 1;
    }

    if (snrt_global_core_idx() == 0) {
        printf("WEIGHT_RESIDENT_LAYOUT logical_W=2048x1024 logical_V=2048x1024 logical_W2=1024x2048 physical_W2_half=1024x1024 banks=16..47 mode0_rows=0..8191 mode1_rows=8192..12287 free_rows=12288..16383 required=%u capacity=%u\n",
               required, TCDM_CAPACITY_BYTES);
    }
    if (snrt_is_dm_core()) {
        stage_weight_chunks(0);
#if RUN_MODE1
        stage_weight_chunks(1);
#endif
    }
    snrt_cluster_hw_barrier();
    if (snrt_global_core_idx() == 0) {
        printf("WEIGHTS_RESIDENT_READY mode0_chunks=%u mode1_chunks=%u no_more_weight_dma=1\n",
               chunk_count(0), chunk_count(1));
    }

    int total_errors = 0;
    for (int shape = 0; shape < NUM_SHAPES; shape++) {
        if (SELECT_SHAPE >= 0 && shape != SELECT_SHAPE) {
            continue;
        }
        total_errors += run_shape(&shape_cfgs[shape]);
    }
    if (snrt_global_core_idx() == 0) {
        printf("FINAL_RESULT selected_shape=%d chunk_cols=%d status=%s total_errors=%d\n",
               SELECT_SHAPE, CHUNK_COLS,
               total_errors ? "FAIL" : "PASS", total_errors);
    }
    return snrt_global_core_idx() == 0 ? total_errors : 0;
}
