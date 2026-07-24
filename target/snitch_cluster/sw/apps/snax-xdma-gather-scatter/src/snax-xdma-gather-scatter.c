// Copyright 2026 KU Leuven.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "snax-xdma-lib.h"
#include "snrt.h"

#define LANE_BYTES 8u
#define BEAT_BYTES 64u
#define ROW_BYTES 512u
#define ROWS 3u
#define GUARD_WORD 0xdeadbeefdeadbeefULL

static const uint32_t token_offsets[XDMA_SPATIAL_CHAN] = {
    3u * LANE_BYTES,  4u * LANE_BYTES,  17u * LANE_BYTES,
    18u * LANE_BYTES, 29u * LANE_BYTES, 30u * LANE_BYTES,
    45u * LANE_BYTES, 46u * LANE_BYTES,
};

static const uint32_t scatter_offsets[XDMA_SPATIAL_CHAN] = {
    6u * LANE_BYTES,  7u * LANE_BYTES,  20u * LANE_BYTES,
    21u * LANE_BYTES, 35u * LANE_BYTES, 36u * LANE_BYTES,
    51u * LANE_BYTES, 52u * LANE_BYTES,
};

static uint64_t pattern(uint32_t row, uint32_t lane) {
    return 0x4700000000000000ULL | ((uint64_t)row << 16) | lane;
}

static int configure_copy(void *src, void *dst, uint32_t src_row_stride,
                          uint32_t dst_row_stride) {
    uint32_t src_strides[1] = {src_row_stride};
    uint32_t dst_strides[1] = {dst_row_stride};
    uint32_t bounds[1] = {ROWS};
    return snax_xdma_memcpy_nd(src, dst, LANE_BYTES, LANE_BYTES, 1,
                              src_strides, bounds, 1, dst_strides, bounds,
                              0xffu, 0xffu, 0xffu);
}

static int run_task(void) {
    int task_id = snax_xdma_start();
    if (task_id < 0) return -1;
    snax_xdma_local_wait((uint32_t)task_id);
    return 0;
}

static int check_sparse_destination(const uint8_t *dst) {
    for (uint32_t row = 0; row < ROWS; row++) {
        const uint64_t *dst_row =
            (const uint64_t *)(dst + row * ROW_BYTES);
        for (uint32_t bank = 0; bank < ROW_BYTES / LANE_BYTES; bank++) {
            uint64_t expected = GUARD_WORD;
            for (uint32_t lane = 0; lane < XDMA_SPATIAL_CHAN; lane++) {
                if (scatter_offsets[lane] / LANE_BYTES == bank) {
                    expected = pattern(row, lane);
                    break;
                }
            }
            if (dst_row[bank] != expected) return 1;
        }
    }
    return 0;
}

static int test_stride(uint8_t *src, uint8_t *dst) {
    uint64_t *src_words = (uint64_t *)src;
    uint64_t *dst_words = (uint64_t *)dst;
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t lane = 0; lane < XDMA_SPATIAL_CHAN; lane++) {
            src_words[row * XDMA_SPATIAL_CHAN + lane] = pattern(row, lane);
            dst_words[row * XDMA_SPATIAL_CHAN + lane] = 0;
        }
    }
    if (configure_copy(src, dst, BEAT_BYTES, BEAT_BYTES) != 0 ||
        run_task() != 0)
        return 1;
    for (uint32_t i = 0; i < ROWS * XDMA_SPATIAL_CHAN; i++) {
        if (dst_words[i] != src_words[i]) return 1;
    }
    return 0;
}

static int test_gather(uint8_t *src, uint8_t *dst) {
    for (uint32_t row = 0; row < ROWS; row++) {
        uint64_t *src_row = (uint64_t *)(src + row * ROW_BYTES);
        uint64_t *dst_row = (uint64_t *)(dst + row * BEAT_BYTES);
        for (uint32_t lane = 0; lane < ROW_BYTES / LANE_BYTES; lane++)
            src_row[lane] = GUARD_WORD;
        for (uint32_t lane = 0; lane < XDMA_SPATIAL_CHAN; lane++) {
            *(uint64_t *)(src + row * ROW_BYTES + token_offsets[lane]) =
                pattern(row, lane);
            dst_row[lane] = 0;
        }
    }
    if (configure_copy(src, dst, ROW_BYTES, BEAT_BYTES) != 0 ||
        snax_xdma_set_src_address_mode(XDMA_ADDR_MODE_INDEXED,
                                       token_offsets) != 0 ||
        run_task() != 0)
        return 1;
    for (uint32_t row = 0; row < ROWS; row++) {
        uint64_t *dst_row = (uint64_t *)(dst + row * BEAT_BYTES);
        for (uint32_t lane = 0; lane < XDMA_SPATIAL_CHAN; lane++) {
            if (dst_row[lane] != pattern(row, lane)) return 1;
        }
    }
    return 0;
}

static int test_scatter(uint8_t *src, uint8_t *dst) {
    for (uint32_t row = 0; row < ROWS; row++) {
        uint64_t *src_row = (uint64_t *)(src + row * BEAT_BYTES);
        uint64_t *dst_row = (uint64_t *)(dst + row * ROW_BYTES);
        for (uint32_t lane = 0; lane < XDMA_SPATIAL_CHAN; lane++)
            src_row[lane] = pattern(row, lane);
        for (uint32_t lane = 0; lane < ROW_BYTES / LANE_BYTES; lane++)
            dst_row[lane] = GUARD_WORD;
    }
    if (configure_copy(src, dst, BEAT_BYTES, ROW_BYTES) != 0 ||
        snax_xdma_set_dst_address_mode(XDMA_ADDR_MODE_INDEXED,
                                       scatter_offsets) != 0 ||
        run_task() != 0)
        return 1;
    return check_sparse_destination(dst);
}

static int test_gather_scatter(uint8_t *src, uint8_t *dst) {
    for (uint32_t row = 0; row < ROWS; row++) {
        uint64_t *src_row = (uint64_t *)(src + row * ROW_BYTES);
        uint64_t *dst_row = (uint64_t *)(dst + row * ROW_BYTES);
        for (uint32_t bank = 0; bank < ROW_BYTES / LANE_BYTES; bank++) {
            src_row[bank] = GUARD_WORD;
            dst_row[bank] = GUARD_WORD;
        }
        for (uint32_t lane = 0; lane < XDMA_SPATIAL_CHAN; lane++) {
            *(uint64_t *)(src + row * ROW_BYTES + token_offsets[lane]) =
                pattern(row, lane);
        }
    }
    if (configure_copy(src, dst, ROW_BYTES, ROW_BYTES) != 0 ||
        snax_xdma_set_src_address_mode(XDMA_ADDR_MODE_INDEXED,
                                       token_offsets) != 0 ||
        snax_xdma_set_dst_address_mode(XDMA_ADDR_MODE_INDEXED,
                                       scatter_offsets) != 0 ||
        run_task() != 0)
        return 1;
    return check_sparse_destination(dst);
}

int main(void) {
    if (!snrt_is_dm_core() || snrt_cluster_idx() != 0) return 0;

#if !XDMA_HAS_GATHER_SCATTER
    printf("XDMA_GATHER_SCATTER FAIL feature_not_generated\n");
    return 1;
#else
    uint8_t *base = (uint8_t *)snrt_l1_next();
    uint8_t *stride_src = base;
    uint8_t *stride_dst = base + 0x1000;
    uint8_t *gather_src = base + 0x2000;
    uint8_t *gather_dst = base + 0x4000;
    uint8_t *scatter_src = base + 0x5000;
    uint8_t *scatter_dst = base + 0x6000;
    uint8_t *gather_scatter_src = base + 0x8000;
    uint8_t *gather_scatter_dst = base + 0xa000;

    int stride_err = test_stride(stride_src, stride_dst);
    printf("XDMA_STRIDE_MODE %s\n", stride_err ? "FAIL" : "PASS");

    int gather_err = test_gather(gather_src, gather_dst);
    printf("XDMA_GATHER_MODE %s\n", gather_err ? "FAIL" : "PASS");

    int scatter_err = test_scatter(scatter_src, scatter_dst);
    printf("XDMA_SCATTER_MODE %s\n", scatter_err ? "FAIL" : "PASS");

    int gather_scatter_err =
        test_gather_scatter(gather_scatter_src, gather_scatter_dst);
    printf("XDMA_GATHER_TO_SCATTER %s\n",
           gather_scatter_err ? "FAIL" : "PASS");

    int errors =
        stride_err + gather_err + scatter_err + gather_scatter_err;
    printf("XDMA_GATHER_SCATTER %s errors=%d\n",
           errors ? "FAIL" : "PASS", errors);
    return errors;
#endif
}
