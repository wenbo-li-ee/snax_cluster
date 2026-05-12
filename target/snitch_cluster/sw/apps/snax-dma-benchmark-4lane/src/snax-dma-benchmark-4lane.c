// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// DMA bandwidth benchmark: L3 (DRAM @ 0x80000000) -> TCDM (L1 @ 0x10000000)
// Transfer sizes: 1x/2x/3x/4x/8x/16x N, where N = 2048*8 int16 (32768 bytes).
// Uses snrt_dma_start_1d_wideptr to be explicit about L3 vs TCDM addresses.

#include "snrt.h"

// Base unit: N = 2048*8 = 16384 int16 = 32768 bytes
#define N_BASE  (2048 * 8)
#define N_MAX   (N_BASE * 16)  // 262144 int16 = 524288 bytes (full L1)

// Source data in L3 (DRAM). Placed in the .dram section by the linker.
static int16_t l3_src[N_MAX] __attribute__((section(".dram")));

static const int kMults[] = {1, 2, 3, 4, 8, 16};
#define NUM_TESTS 6

int main() {
    // Only the DMA core performs the transfers; compute cores exit immediately.
    if (!snrt_is_dm_core()) {
        return 0;
    }

    // TCDM destination buffer (L1).
    int16_t *tcdm_dst = (int16_t *)snrt_l1_next();

    // Precompute wide addresses:
    //   L3 src: 64-bit physical address (addrh = 0 for DRAM in this platform)
    //   TCDM dst: 64-bit address = (uint32_t ptr) + (addrh << 32)
    uint64_t src_wide = (uint64_t)(uintptr_t)l3_src;
    uint32_t addrh    = snrt_cluster_base_addrh();

    for (int i = 0; i < NUM_TESTS; i++) {
        int mult   = kMults[i];
        size_t n_elem  = (size_t)N_BASE * mult;
        size_t n_bytes = n_elem * sizeof(int16_t);

        uint64_t dst_wide = (uint64_t)(uintptr_t)tcdm_dst
                          + ((uint64_t)addrh << 32);

        uint32_t t0 = snrt_mcycle();
        snrt_dma_txid_t tid = snrt_dma_start_1d_wideptr(dst_wide, src_wide,
                                                         n_bytes);
        snrt_dma_wait_all();
        uint32_t t1 = snrt_mcycle();

        printf("DMA L3->TCDM %2dN = %6u int16 (%7u bytes): %u cycles\n",
               mult, (unsigned)n_elem, (unsigned)n_bytes,
               (uint32_t)(t1 - t0));
        (void)tid;
    }

    return 0;
}

