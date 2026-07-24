// Shape-1 common-16 experiment with real weight DMA/compute overlap.
//
// Four tokens are staged before Mode 0.  Two fixed ping/pong weight buffers
// hold the current/next common-16 panels.  Chunk 0 is prefetched outside the
// loop.  In loop iteration c, core 0 starts compute c and immediately stages
// the moving streamer base CSRs for c+1.  In parallel, the DM core starts one
// already-prepared interleaved-W/V DMA for c+1, then prepares the complete
// interleaved-W/V descriptor for c+2 while c+1 transfers.  One hardware
// barrier ends the iteration and protects the reused buffer.

#include "data.h"
#include "snax-dual-versacore-swiglu-lib.h"

#ifndef USE_XDMA
#define USE_XDMA 0
#endif

#if USE_XDMA
#include "snax-xdma-lib.h"
#if USE_XDMA == 2
#define WEIGHT_DMA_ENGINE_NAME "idma_to_xdma_mmio"
#else
#define WEIGHT_DMA_ENGINE_NAME "idma+xdma_local"
#endif
#else
#define WEIGHT_DMA_ENGINE_NAME "idma"
#endif

#define SHAPE_INDEX 1
#ifndef CHUNK_COLS
#define CHUNK_COLS 16u
#endif
#define CHUNKS_PER_MODE (N0_TOTAL / CHUNK_COLS)
#define WAIT_TIMEOUT_CYCLES 1000000u
#define CHECK_PRINT_LIMIT 4u
#define TCDM_ROW_BYTES 512u
#define BANK_WORD_BYTES 8u
#define WV_INTERLEAVED_ROW_BYTES 128u
#define XDMA_STAGE0_BASE (1024u * 1024u)
// The generated top-level passes ClusterAddrSpace=16384 KiB to the cluster
// and ClusterAddressSpace=0x1000000 to its XDMA wrapper.
#define CLUSTER_ADDRESS_SPACE_BYTES (1u << 24)
#define XDMA_MMIO_BYTES (16u * 1024u)
#define XDMA_DATA_WINDOW_BYTES (4u * 1024u)

#define A_BASE 0
#define B0_PING_BASE (16 * BANK_WORD_BYTES)
#define B1_PING_BASE (24 * BANK_WORD_BYTES)
#define B0_PONG_BASE (32 * BANK_WORD_BYTES)
#define B1_PONG_BASE (40 * BANK_WORD_BYTES)
#define MODE0_D_BASE (48 * BANK_WORD_BYTES)
#define MODE1_D0_BASE 0
#define MODE1_D1_BASE (8 * BANK_WORD_BYTES)

typedef struct {
    uint32_t prologue_cycles;
    uint32_t first_dma_cycles;
    uint32_t loop_cycles;
    uint32_t compute_path_sum;
    uint32_t preconfig_sum;
    uint32_t barrier_wait_sum;
    uint32_t accel_sum;
    uint32_t streamer_sum;
    uint32_t dma_loop_sum;
    uint32_t dma_prepared_start_sum;
    uint32_t dma_next_prepare_sum;
    uint32_t dma_wait_sum;
    uint32_t ideal_overlap_sum;
    uint32_t sync_control_overhead;
} pipeline_stats_t;

static volatile uint32_t dma_cycles[2][CHUNKS_PER_MODE];
static volatile uint32_t dma_prepared_start_cycles[CHUNKS_PER_MODE];
static volatile uint32_t dma_next_prepare_cycles[CHUNKS_PER_MODE];
static volatile uint32_t dma_wait_cycles[CHUNKS_PER_MODE];
static volatile uint32_t compute_path_cycles[2][CHUNKS_PER_MODE];
static volatile uint32_t pipeline_error;
static volatile uint32_t calibration_sink;
static volatile uint32_t token_dma_cycles_shared;
static volatile uint32_t calibration_iterations = CHUNKS_PER_MODE;
static uint32_t empty_loop_cycles;
static uint32_t barrier_loop_cycles;
static pipeline_stats_t stats[2];

static uint32_t mode_panel_bytes(int mode) {
    return mode ? (K1_TOTAL / 8u) * 16u : (K0_TOTAL / 8u) * 16u;
}

static uint32_t mode_panel_span(int mode) {
    return (mode_panel_bytes(mode) / 64u) * TCDM_ROW_BYTES;
}

// snrt_dma_start_2d() is a five-instruction sequence:
// DMSRC, DMDST, DMSTR, DMREP, DMCPYI.  The first four instructions update the
// iDMA frontend's staging request; DMCPYI copies that staged request into the
// request FIFO.  Split the sequence so the first descriptor of the next
// iteration can be configured while the current iteration is still running.
static inline __attribute__((always_inline)) void dma_prepare_2d(
    void *dst, const void *src, size_t dst_stride, size_t src_stride,
    size_t repeat) {
    uint64_t dst_wide = (uint64_t)dst +
                        ((uint64_t)snrt_cluster_base_addrh() << 32);
    uint64_t src_wide = (uint64_t)src +
                        ((uint64_t)snrt_cluster_base_addrh() << 32);
    register uint32_t dst_low asm("a0") = dst_wide;
    register uint32_t dst_high asm("a1") = dst_wide >> 32;
    register uint32_t src_low asm("a2") = src_wide;
    register uint32_t src_high asm("a3") = src_wide >> 32;
    register uint32_t dst_stride_reg asm("a5") = dst_stride;
    register uint32_t src_stride_reg asm("a6") = src_stride;
    register uint32_t repeat_reg asm("a7") = repeat;

    asm volatile(
        ".word (0b0000000 << 25) | ((13) << 20) | ((12) << 15) | "
        "(0b000 << 12) | (0b0101011 << 0)\n"
        :
        : "r"(src_high), "r"(src_low));
    asm volatile(
        ".word (0b0000001 << 25) | ((11) << 20) | ((10) << 15) | "
        "(0b000 << 12) | (0b0101011 << 0)\n"
        :
        : "r"(dst_high), "r"(dst_low));
    asm volatile(
        ".word (0b0000110 << 25) | ((15) << 20) | ((16) << 15) | "
        "(0b000 << 12) | (0b0101011 << 0)\n"
        :
        : "r"(dst_stride_reg), "r"(src_stride_reg));
    asm volatile(
        ".word (0b0000111 << 25) | ((17) << 15) | "
        "(0b000 << 12) | (0b0101011 << 0)\n"
        :
        : "r"(repeat_reg));
}

// After one complete 2D descriptor has established the invariant
// stride/repeat fields, only source and destination move between chunks.
// DMCPYI does not clear the staged request, so these two address instructions
// are sufficient for all following full-size chunks.
static inline __attribute__((always_inline)) void dma_prepare_addresses(
    void *dst, const void *src) {
    uint64_t dst_wide = (uint64_t)dst +
                        ((uint64_t)snrt_cluster_base_addrh() << 32);
    uint64_t src_wide = (uint64_t)src +
                        ((uint64_t)snrt_cluster_base_addrh() << 32);
    register uint32_t dst_low asm("a0") = dst_wide;
    register uint32_t dst_high asm("a1") = dst_wide >> 32;
    register uint32_t src_low asm("a2") = src_wide;
    register uint32_t src_high asm("a3") = src_wide >> 32;

    asm volatile(
        ".word (0b0000000 << 25) | ((13) << 20) | ((12) << 15) | "
        "(0b000 << 12) | (0b0101011 << 0)\n"
        :
        : "r"(src_high), "r"(src_low));
    asm volatile(
        ".word (0b0000001 << 25) | ((11) << 20) | ((10) << 15) | "
        "(0b000 << 12) | (0b0101011 << 0)\n"
        :
        : "r"(dst_high), "r"(dst_low));
}

static inline __attribute__((always_inline)) snrt_dma_txid_t
dma_start_prepared_2d(size_t size) {
    register uint32_t size_reg asm("a4") = size;
    register uint32_t txid asm("a0");
    asm volatile(
        ".word (0b0000010 << 25) | (0b00010 << 20) | ((14) << 15) | "
        "(0b000 << 12) | ((10) << 7) | (0b0101011 << 0)\n"
        : "=r"(txid)
        : "r"(size_reg));
    return txid;
}

static int32_t weight_base(int right, uint32_t chunk) {
    if (chunk & 1u) {
        return right ? B1_PONG_BASE : B0_PONG_BASE;
    }
    return right ? B1_PING_BASE : B0_PING_BASE;
}

static int32_t mode0_d_base(uint32_t chunk) {
    return MODE0_D_BASE +
           (int32_t)(chunk * (CHUNK_COLS / 16u) * TCDM_ROW_BYTES);
}

static int32_t mode1_d_base(int right, uint32_t chunk) {
    int32_t base = right ? MODE1_D1_BASE : MODE1_D0_BASE;
    return base + (int32_t)(chunk * 4u * TCDM_ROW_BYTES);
}

static inline __attribute__((always_inline)) void set_local_base(
    uint32_t csr, int32_t delta_local) {
    csrw_ss(csr, (uint32_t)((uintptr_t)snrt_l1_next() +
                            (uintptr_t)delta_local));
}

static void preconfigure_next_bases(int mode, uint32_t chunk) {
    set_local_base(BASE_PTR_READER_1_LOW, weight_base(0, chunk));
    set_local_base(BASE_PTR_READER_2_LOW, weight_base(1, chunk));
    if (mode == 0) {
        set_local_base(BASE_PTR_WRITER_0_LOW, mode0_d_base(chunk));
    } else {
        set_local_base(BASE_PTR_WRITER_0_LOW, mode1_d_base(0, chunk));
        set_local_base(BASE_PTR_WRITER_1_LOW, mode1_d_base(1, chunk));
    }
}

#if USE_XDMA
static uint64_t xdma_full_addr(const void *ptr) {
    return (uint64_t)(uintptr_t)ptr +
           ((uint64_t)snrt_cluster_base_addrh() << 32);
}

static uint32_t mode0_chunk_bytes(void) {
    return 2u * (CHUNK_COLS / 4u) * mode_panel_bytes(0);
}

static uint8_t *xdma_stage_ptr(uint32_t chunk) {
    return (uint8_t *)snrt_l1_next() + XDMA_STAGE0_BASE +
           (chunk & 1u) * mode0_chunk_bytes();
}

// The single-cluster testbench has no XDMA endpoint beside main memory.
// Therefore iDMA brings one contiguous W/V chunk from L3 into a ping/pong
// staging area, after which local XDMA performs the strided TCDM placement.
static void idma_stage_mode0_wv(uint32_t chunk) {
    uint32_t chunk_bytes = mode0_chunk_bytes();
    snrt_dma_start_1d(xdma_stage_ptr(chunk),
                      WV_interleaved + chunk * chunk_bytes, chunk_bytes);
    snrt_dma_wait_all();
}

static void idma_prepare_mode0_stage(uint32_t chunk) {
    uint32_t chunk_bytes = mode0_chunk_bytes();
    dma_prepare_2d(xdma_stage_ptr(chunk),
                   WV_interleaved + chunk * chunk_bytes, chunk_bytes,
                   chunk_bytes, 1);
}

static void idma_prepare_mode0_stage_addresses(uint32_t chunk) {
    uint32_t chunk_bytes = mode0_chunk_bytes();
    dma_prepare_addresses(xdma_stage_ptr(chunk),
                          WV_interleaved + chunk * chunk_bytes);
}

// Configure the invariant two-dimensional XDMA walk once. Dimension 0 selects
// adjacent W/V beats in staging; dimension 1 advances to the next weight row.
static int xdma_configure_mode0_wv(uint32_t chunk) {
    uint8_t *tcdm = (uint8_t *)snrt_l1_next();
    uint32_t chunk_bytes = mode0_chunk_bytes();
    uint32_t bounds[2] = {2u, chunk_bytes / WV_INTERLEAVED_ROW_BYTES};
    uint32_t src_strides[2] = {64u, WV_INTERLEAVED_ROW_BYTES};
    uint32_t dst_strides[2] = {64u, TCDM_ROW_BYTES};
    const uint8_t *src = xdma_stage_ptr(chunk);
    void *dst = tcdm + weight_base(0, chunk);

    return snax_xdma_memcpy_nd_full_addr(
        xdma_full_addr(src), xdma_full_addr(dst), BANK_WORD_BYTES,
        BANK_WORD_BYTES, 2, src_strides, bounds, 2, dst_strides, bounds,
        0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
}

// Once the static AGU fields have been configured, only these four address
// CSRs move between common-16 chunks.  Calling this after START is safe
// because the active task has already latched the previous configuration.
static void xdma_prepare_mode0_wv_addresses(uint32_t chunk) {
    uint8_t *tcdm = (uint8_t *)snrt_l1_next();
    uint64_t src = xdma_full_addr(xdma_stage_ptr(chunk));
    uint64_t dst = xdma_full_addr(tcdm + weight_base(0, chunk));

    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_LSB, (uint32_t)src);
    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_MSB, (uint32_t)(src >> 32));
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB, (uint32_t)dst);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB, (uint32_t)(dst >> 32));
}

// Both staging buffers and both weight buffers are in the same local TCDM
// address region. The high halves were configured by the first full XDMA
// descriptor and remain invariant, so steady state only needs two low CSRs.
static void xdma_prepare_mode0_wv_addresses_low(uint32_t chunk) {
    uint8_t *tcdm = (uint8_t *)snrt_l1_next();
    uint64_t src = xdma_full_addr(xdma_stage_ptr(chunk));
    uint64_t dst = xdma_full_addr(tcdm + weight_base(0, chunk));

    snax_write_xdma_cfg_reg(XDMA_SRC_ADDR_PTR_LSB, (uint32_t)src);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB, (uint32_t)dst);
}

static void *xdma_data_mmio_ptr(void) {
    return (void *)(uintptr_t)(
        snrt_cluster_base_addr() + CLUSTER_ADDRESS_SPACE_BYTES -
        XDMA_MMIO_BYTES);
}

// In the DRAM/TCDM hybrid mode the XDMA reader pointer is zero, so its reader
// frame is deliberately discarded. iDMA writes the payload into XDMA's 4 KiB
// data window and the local XDMA writer performs the strided TCDM placement.
static int xdma_configure_mode0_mmio_writer(uint32_t chunk) {
    uint8_t *tcdm = (uint8_t *)snrt_l1_next();
    uint32_t chunk_bytes = mode0_chunk_bytes();
    uint32_t bounds[2] = {2u, chunk_bytes / WV_INTERLEAVED_ROW_BYTES};
    uint32_t src_strides[2] = {64u, WV_INTERLEAVED_ROW_BYTES};
    uint32_t dst_strides[2] = {64u, TCDM_ROW_BYTES};
    uint64_t dst = xdma_full_addr(tcdm + weight_base(0, chunk));

    return snax_xdma_memcpy_nd_full_addr(
        0, dst, BANK_WORD_BYTES, BANK_WORD_BYTES, 2, src_strides, bounds, 2,
        dst_strides, bounds, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
}

static void xdma_prepare_mode0_mmio_writer_address(uint32_t chunk) {
    uint8_t *tcdm = (uint8_t *)snrt_l1_next();
    uint64_t dst = xdma_full_addr(tcdm + weight_base(0, chunk));

    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_LSB, (uint32_t)dst);
    snax_write_xdma_cfg_reg(XDMA_DST_ADDR_PTR_MSB, (uint32_t)(dst >> 32));
}

static void idma_prepare_mode0_mmio_stream(uint32_t chunk) {
    uint32_t chunk_bytes = mode0_chunk_bytes();
    dma_prepare_2d(xdma_data_mmio_ptr(),
                   WV_interleaved + chunk * chunk_bytes, 0,
                   XDMA_DATA_WINDOW_BYTES,
                   chunk_bytes / XDMA_DATA_WINDOW_BYTES);
}

static int xdma_wait_writer_idle(void) {
    uint32_t start = snrt_mcycle();
    uint32_t previous = snax_xdma_last_write_cycle();
    uint32_t stable_reads = 0;
    while (1) {
        uint32_t current = snax_xdma_last_write_cycle();
        if (current == previous) {
            stable_reads++;
            if (stable_reads == 16u) {
                return 0;
            }
        } else {
            stable_reads = 0;
        }
        previous = current;
        if ((uint32_t)(snrt_mcycle() - start) > WAIT_TIMEOUT_CYCLES) {
            return 1;
        }
    }
}
#endif

static uint32_t dma_weight_chunk(int mode, uint32_t chunk) {
    uint8_t *tcdm = (uint8_t *)snrt_l1_next();
    uint32_t panel_bytes = mode_panel_bytes(mode);
    uint32_t chunk_bytes = (CHUNK_COLS / 4u) * panel_bytes;
    uint32_t repeats = chunk_bytes / 64u;
    uint32_t start = snrt_mcycle();

#if USE_XDMA == 1
    if (mode != 0) {
        pipeline_error = 100u;
        return snrt_mcycle() - start;
    }
    idma_stage_mode0_wv(chunk);
    if (xdma_configure_mode0_wv(chunk) != 0) {
        pipeline_error = 100u;
        return snrt_mcycle() - start;
    }
    uint32_t task_id = snax_xdma_start();
    snax_xdma_local_wait(task_id);
#elif USE_XDMA == 2
    if (mode != 0 || xdma_configure_mode0_mmio_writer(chunk) != 0) {
        pipeline_error = 100u;
        return snrt_mcycle() - start;
    }
    snax_xdma_start();
    snrt_dma_start_2d(
        xdma_data_mmio_ptr(), WV_interleaved + chunk * mode0_chunk_bytes(),
        XDMA_DATA_WINDOW_BYTES, 0, XDMA_DATA_WINDOW_BYTES,
        mode0_chunk_bytes() / XDMA_DATA_WINDOW_BYTES);
    snrt_dma_wait_all();
    if (xdma_wait_writer_idle() != 0) {
        pipeline_error = 101u;
    }
#else
    if (mode == 0) {
        const uint8_t *src =
            WV_interleaved + chunk * 2u * chunk_bytes;
        // Each L3 row is W[64 B] followed by V[64 B].  B0 and B1 are
        // adjacent 64-byte bank groups in TCDM, so one two-beat descriptor
        // fills both without a second DMA command.
        snrt_dma_start_2d(tcdm + weight_base(0, chunk), src,
                          WV_INTERLEAVED_ROW_BYTES, TCDM_ROW_BYTES,
                          WV_INTERLEAVED_ROW_BYTES, repeats);
    } else {
        const uint8_t *src_left = W2_left + chunk * chunk_bytes;
        const uint8_t *src_right = W2_right + chunk * chunk_bytes;
        snrt_dma_start_2d(tcdm + weight_base(0, chunk), src_left, 64,
                          TCDM_ROW_BYTES, 64, repeats);
        snrt_dma_start_2d(tcdm + weight_base(1, chunk), src_right, 64,
                          TCDM_ROW_BYTES, 64, repeats);
    }
    snrt_dma_wait_all();
#endif
    return snrt_mcycle() - start;
}

static void dma_prepare_mode0_wv(uint32_t chunk) {
#if USE_XDMA == 1
    xdma_prepare_mode0_wv_addresses(chunk);
#elif USE_XDMA == 2
    xdma_prepare_mode0_mmio_writer_address(chunk);
#else
    uint8_t *tcdm = (uint8_t *)snrt_l1_next();
    uint32_t chunk_bytes = (CHUNK_COLS / 4u) * mode_panel_bytes(0);
    uint32_t repeats = chunk_bytes / 64u;
    dma_prepare_2d(tcdm + weight_base(0, chunk),
                   WV_interleaved + chunk * 2u * chunk_bytes,
                   TCDM_ROW_BYTES, WV_INTERLEAVED_ROW_BYTES, repeats);
#endif
}

// The interleaved W/V descriptor was prepared in the previous iteration.
// Start it with only DMCPYI, then overwrite the frontend staging registers
// with the following iteration's complete W/V descriptor.
static uint32_t dma_mode0_prepared_wv(uint32_t chunk) {
    uint32_t total_start = snrt_mcycle();

    uint32_t phase_start = snrt_mcycle();
#if USE_XDMA == 1
    uint32_t task_id = snax_xdma_start();
    if (chunk + 1u < CHUNKS_PER_MODE) {
        dma_start_prepared_2d(mode0_chunk_bytes());
    }
#elif USE_XDMA == 2
    snax_xdma_start();
    dma_start_prepared_2d(XDMA_DATA_WINDOW_BYTES);
#else
    dma_start_prepared_2d(WV_INTERLEAVED_ROW_BYTES);
#endif
    dma_prepared_start_cycles[chunk] = snrt_mcycle() - phase_start;

    phase_start = snrt_mcycle();
#if USE_XDMA == 1
    if (chunk + 1u < CHUNKS_PER_MODE) {
        xdma_prepare_mode0_wv_addresses_low(chunk + 1u);
    }
    if (chunk + 2u < CHUNKS_PER_MODE) {
        idma_prepare_mode0_stage_addresses(chunk + 2u);
    }
#elif USE_XDMA == 2
    if (chunk + 1u < CHUNKS_PER_MODE) {
        xdma_prepare_mode0_mmio_writer_address(chunk + 1u);
        idma_prepare_mode0_mmio_stream(chunk + 1u);
    }
#else
    if (chunk + 1u < CHUNKS_PER_MODE) {
        uint8_t *tcdm = (uint8_t *)snrt_l1_next();
        uint32_t chunk_bytes =
            (CHUNK_COLS / 4u) * mode_panel_bytes(0);
        dma_prepare_addresses(
            tcdm + weight_base(0, chunk + 1u),
            WV_interleaved + (chunk + 1u) * 2u * chunk_bytes);
    }
#endif
    dma_next_prepare_cycles[chunk] = snrt_mcycle() - phase_start;

    phase_start = snrt_mcycle();
#if USE_XDMA == 1
    snax_xdma_local_wait(task_id);
    if (chunk + 1u < CHUNKS_PER_MODE) {
        snrt_dma_wait_all();
    }
#elif USE_XDMA == 2
    snrt_dma_wait_all();
    if (xdma_wait_writer_idle() != 0) {
        pipeline_error = 101u;
    }
#else
    snrt_dma_wait_all();
#endif
    dma_wait_cycles[chunk] = snrt_mcycle() - phase_start;
    return snrt_mcycle() - total_start;
}

static uint32_t stage_four_tokens(void) {
    uint8_t *tcdm = (uint8_t *)snrt_l1_next();
    uint32_t start = snrt_mcycle();
    for (uint32_t token = 0; token < 4u; token++) {
        snrt_dma_start_2d(tcdm + A_BASE + token * 16u,
                          A + token * K0_TOTAL, 16, TCDM_ROW_BYTES, 16,
                          K0_TOTAL / 8u);
    }
    snrt_dma_wait_all();
    return snrt_mcycle() - start;
}

static void configure_mode0_first(const shape_cfg_t *cfg) {
    uint32_t n_tiles = CHUNK_COLS / (uint32_t)cfg->meshCol;
    int32_t a_sstride[2] = {8, 16};
    int32_t a_tbound[6] = {
        cfg->K0_tiles, (int32_t)n_tiles, 1, 1, 1, 1};
    int32_t a_tstride[6] = {TCDM_ROW_BYTES, 0, 0, 0, 0, 0};
    int32_t b_sstride[2] = {8, (int32_t)mode_panel_span(0)};
    int32_t b_tbound[4] = {
        4, cfg->K0_tiles / 4, (int32_t)n_tiles, 1};
    int32_t b_tstride[4] = {
        16, TCDM_ROW_BYTES, 2 * (int32_t)mode_panel_span(0), 0};
    int32_t d_sstride[1] = {8};
    int32_t d_tbound[4] = {2, 4, 2, (int32_t)(n_tiles / 2u)};
    int32_t d_tstride[4] = {8, 16, 64, TCDM_ROW_BYTES};
    uint32_t subtraction =
        gen_dual_vc_subtraction_config(SUBTRACTION_A, SUBTRACTION_B);

    set_dual_versacore_streamer_csr_d0_only(
        A_BASE, a_sstride, a_tbound, a_tstride, SET_ADDR_REMAP_INDEX_A,
        cfg->A_channel_en, B0_PING_BASE, b_sstride, b_tbound, b_tstride,
        SET_ADDR_REMAP_INDEX_B0, cfg->B_channel_en, B1_PING_BASE, b_sstride,
        b_tbound, b_tstride, SET_ADDR_REMAP_INDEX_B1, cfg->B_channel_en,
        MODE0_D_BASE, d_sstride, d_tbound, d_tstride,
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
}

static void configure_mode1_first(const shape_cfg_t *cfg) {
    int32_t a_sstride[2] = {8, 16};
    int32_t a_tbound[6] = {2, cfg->K1_tiles / 2, 2, 1, 1, 1};
    int32_t a_tstride[6] = {64, TCDM_ROW_BYTES, 0, 0, 0, 0};
    int32_t b_sstride[2] = {8, (int32_t)mode_panel_span(1)};
    int32_t b_tbound[4] = {4, cfg->K1_tiles / 4, 2, 1};
    int32_t b_tstride[4] = {
        16, TCDM_ROW_BYTES, 2 * (int32_t)mode_panel_span(1), 0};
    int32_t d_sstride[1] = {8};
    int32_t d_tbound[4] = {2, 4, 2, 1};
    int32_t d_tstride[4] = {TCDM_ROW_BYTES, 8, 2 * TCDM_ROW_BYTES, 0};
    uint32_t subtraction =
        gen_dual_vc_subtraction_config(SUBTRACTION_A, SUBTRACTION_B);

    set_dual_versacore_streamer_csr(
        MODE0_D_BASE, a_sstride, a_tbound, a_tstride,
        SET_ADDR_REMAP_INDEX_A, cfg->A_channel_en, B0_PING_BASE, b_sstride,
        b_tbound, b_tstride, SET_ADDR_REMAP_INDEX_B0, cfg->B_channel_en,
        B1_PING_BASE, b_sstride, b_tbound, b_tstride,
        SET_ADDR_REMAP_INDEX_B1, cfg->B_channel_en, MODE1_D0_BASE, d_sstride,
        d_tbound, d_tstride, SET_ADDR_REMAP_INDEX_D0, cfg->D_channel_en,
        MODE1_D1_BASE, d_sstride, d_tbound, d_tstride,
        SET_ADDR_REMAP_INDEX_D1, cfg->D_channel_en);

    set_dual_versacore_csr(1, cfg->K1_tiles, 2, subtraction,
                           cfg->array_shape, DATA_TYPE);
    set_dual_versacore_mode(1);
    set_dual_versacore_rescale0(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
    set_dual_versacore_rescale1(RESCALE_INPUT_ZP, RESCALE_MULTIPLIER,
                                RESCALE_OUTPUT_ZP, RESCALE_SHIFT);
}

static int wait_current_iteration(void) {
    uint32_t start;
    csrw_ss(DUAL_VC_START, 0);
    csrw_ss(DUAL_VC_START, 0);
    start = snrt_mcycle();
    while (csrr_ss(DUAL_VC_BUSY)) {
        if ((uint32_t)(snrt_mcycle() - start) > WAIT_TIMEOUT_CYCLES) {
            return 1;
        }
    }

    csrw_ss(STREAMER_START_CSR, 0);
    csrw_ss(STREAMER_START_CSR, 0);
    start = snrt_mcycle();
    while (csrr_ss(STREAMER_BUSY_CSR)) {
        if ((uint32_t)(snrt_mcycle() - start) > WAIT_TIMEOUT_CYCLES) {
            return 2;
        }
    }
    return 0;
}

static void clear_mode_stats(int mode) {
    if (snrt_global_core_idx() == 0) {
        stats[mode].prologue_cycles = 0;
        stats[mode].first_dma_cycles = 0;
        stats[mode].loop_cycles = 0;
        stats[mode].compute_path_sum = 0;
        stats[mode].preconfig_sum = 0;
        stats[mode].barrier_wait_sum = 0;
        stats[mode].accel_sum = 0;
        stats[mode].streamer_sum = 0;
        stats[mode].dma_loop_sum = 0;
        stats[mode].dma_prepared_start_sum = 0;
        stats[mode].dma_next_prepare_sum = 0;
        stats[mode].dma_wait_sum = 0;
        stats[mode].ideal_overlap_sum = 0;
        stats[mode].sync_control_overhead = 0;
        pipeline_error = 0;
        for (uint32_t c = 0; c < CHUNKS_PER_MODE; c++) {
            dma_cycles[mode][c] = 0;
            compute_path_cycles[mode][c] = 0;
            if (mode == 0) {
                dma_prepared_start_cycles[c] = 0;
                dma_next_prepare_cycles[c] = 0;
                dma_wait_cycles[c] = 0;
            }
        }
    }
    snrt_cluster_hw_barrier();
}

static int run_mode_pipeline(const shape_cfg_t *cfg, int mode) {
    clear_mode_stats(mode);

    uint32_t prologue_start = 0;
    if (snrt_global_core_idx() == 0) {
        prologue_start = snrt_mcycle();
        if (mode == 0) {
            configure_mode0_first(cfg);
        } else {
            configure_mode1_first(cfg);
        }
    } else if (snrt_is_dm_core()) {
        dma_cycles[mode][0] = dma_weight_chunk(mode, 0);
        if (mode == 0 && CHUNKS_PER_MODE > 1u) {
#if USE_XDMA == 1
            // Chunk 1 must already be in the alternate staging buffer when
            // iteration 0 launches its local XDMA reshape.
            idma_stage_mode0_wv(1);
            xdma_prepare_mode0_wv_addresses_low(1);
            if (CHUNKS_PER_MODE > 2u) {
                // Hide all descriptor CSR writes for chunk 2 behind
                // iteration 0; only DMCPYI remains in the timed path.
                idma_prepare_mode0_stage(2);
            }
#elif USE_XDMA == 2
            // Both engines latch their current task at START/DMCPYI, so stage
            // the next XDMA destination and iDMA-to-MMIO descriptor now.
            xdma_prepare_mode0_mmio_writer_address(1);
            idma_prepare_mode0_mmio_stream(1);
#else
            // Prepare chunk 1 before entering the timed loop.  Its DMCPYI is
            // deliberately delayed until iteration 0 starts, because pong is
            // the next free destination buffer.
            dma_prepare_mode0_wv(1);
#endif
        }
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        stats[mode].prologue_cycles = snrt_mcycle() - prologue_start;
        stats[mode].first_dma_cycles = dma_cycles[mode][0];
    }

    uint32_t loop_start = 0;
    if (snrt_global_core_idx() == 0) {
        loop_start = snrt_mcycle();
    }

    for (uint32_t c = 0; c < CHUNKS_PER_MODE; c++) {
        if (snrt_global_core_idx() == 0) {
            uint32_t compute_start = snrt_mcycle();
            set_dual_versacore_streamer_start();
            set_dual_versacore_start();

            // The active streamer/accelerator configurations were latched by
            // START.  Stage only the moving bases for the next chunk while
            // the current chunk is running.  Accelerator CSRs are invariant
            // for all full-size chunks and therefore need no loop writes.
            uint32_t preconfig_start = snrt_mcycle();
            if (c + 1u < CHUNKS_PER_MODE) {
                preconfigure_next_bases(mode, c + 1u);
            }
            stats[mode].preconfig_sum += snrt_mcycle() - preconfig_start;

            int wait_rc = wait_current_iteration();
            if (wait_rc != 0) {
                pipeline_error = (uint32_t)(mode * 10 + wait_rc);
            }
            stats[mode].accel_sum += read_dual_versacore_perf_counter();
            stats[mode].streamer_sum +=
                read_dual_versacore_streamer_perf_counter();
            compute_path_cycles[mode][c] =
                snrt_mcycle() - compute_start;
            stats[mode].compute_path_sum +=
                compute_path_cycles[mode][c];
        } else if (snrt_is_dm_core() && c + 1u < CHUNKS_PER_MODE) {
            dma_cycles[mode][c + 1u] =
                mode == 0 ? dma_mode0_prepared_wv(c + 1u)
                          : dma_weight_chunk(mode, c + 1u);
        }

        uint32_t barrier_start = 0;
        if (snrt_global_core_idx() == 0) {
            barrier_start = snrt_mcycle();
        }
        snrt_cluster_hw_barrier();
        if (snrt_global_core_idx() == 0) {
            stats[mode].barrier_wait_sum +=
                snrt_mcycle() - barrier_start;
        }
        if (pipeline_error != 0) {
            break;
        }
    }

    if (snrt_global_core_idx() == 0) {
        stats[mode].loop_cycles = snrt_mcycle() - loop_start;
        for (uint32_t c = 0; c < CHUNKS_PER_MODE; c++) {
            uint32_t next_dma =
                c + 1u < CHUNKS_PER_MODE ? dma_cycles[mode][c + 1u] : 0;
            uint32_t compute = compute_path_cycles[mode][c];
            stats[mode].dma_loop_sum += next_dma;
            if (mode == 0 && c + 1u < CHUNKS_PER_MODE) {
                stats[mode].dma_prepared_start_sum +=
                    dma_prepared_start_cycles[c + 1u];
                stats[mode].dma_next_prepare_sum +=
                    dma_next_prepare_cycles[c + 1u];
                stats[mode].dma_wait_sum += dma_wait_cycles[c + 1u];
            }
            stats[mode].ideal_overlap_sum +=
                compute > next_dma ? compute : next_dma;
        }
        if (stats[mode].loop_cycles >= stats[mode].ideal_overlap_sum) {
            stats[mode].sync_control_overhead =
                stats[mode].loop_cycles - stats[mode].ideal_overlap_sum;
        }
    }
    snrt_cluster_hw_barrier();
    return (int)pipeline_error;
}

static int check_mode0(const shape_cfg_t *cfg) {
    const uint8_t *tcdm = (const uint8_t *)snrt_l1_next();
    uint32_t errors = 0;
    for (uint32_t token = 0; token < 4u; token++) {
        for (uint32_t elem = 0; elem < N0_TOTAL; elem++) {
            uint32_t group8 = elem / 8u;
            uint32_t lane = elem % 8u;
            uint32_t token_offset = token * 16u;
            uint32_t group_offset =
                (group8 % 2u) * 64u +
                (group8 / 2u) * TCDM_ROW_BYTES;
            const int16_t *actual = (const int16_t *)(
                tcdm + MODE0_D_BASE + token_offset + group_offset +
                lane * 2u);
            int16_t expected =
                cfg->mode0_token_golden[token * N0_TOTAL + elem];
            if (*actual != expected) {
                if (errors < CHECK_PRINT_LIMIT) {
                    printf("MISMATCH mode=0 token=%u col=%u got=%d expected=%d\n",
                           token, elem, *actual, expected);
                }
                errors++;
            }
        }
    }
    return (int)errors;
}

static int check_mode1(const shape_cfg_t *cfg) {
    const uint8_t *tcdm = (const uint8_t *)snrt_l1_next();
    uint32_t errors = 0;
    for (uint32_t right = 0; right < 2u; right++) {
        const int16_t *golden =
            right ? cfg->mode1_d1_token_golden
                  : cfg->mode1_d0_token_golden;
        uint32_t base = right ? MODE1_D1_BASE : MODE1_D0_BASE;
        for (uint32_t token = 0; token < 4u; token++) {
            for (uint32_t elem = 0; elem < N1_TOTAL; elem++) {
                uint32_t beat = elem / 4u;
                uint32_t lane = elem % 4u;
                const int16_t *actual = (const int16_t *)(
                    tcdm + base + token * 8u +
                    beat * TCDM_ROW_BYTES + lane * 2u);
                int16_t expected = golden[token * N1_TOTAL + elem];
                if (*actual != expected) {
                    if (errors < CHECK_PRINT_LIMIT) {
                        printf("MISMATCH mode=1 D%u token=%u col=%u got=%d expected=%d\n",
                               right, token, elem, *actual, expected);
                    }
                    errors++;
                }
            }
        }
    }
    return (int)errors;
}

static __attribute__((noinline)) uint32_t measure_branch_loop(
    uint32_t iterations) {
    uint32_t start = snrt_mcycle();
    for (uint32_t c = 0; c < iterations; c++) {
        // Keep one real instruction in the body so the optimized loop cannot
        // be replaced by a closed-form expression.
        asm volatile("nop" ::: "memory");
    }
    return snrt_mcycle() - start;
}

static void run_control_calibration(void) {
    snrt_cluster_hw_barrier();
    if (snrt_global_core_idx() == 0) {
        uint32_t iterations = calibration_iterations;
        empty_loop_cycles = measure_branch_loop(iterations);
        calibration_sink = iterations;
    }
    snrt_cluster_hw_barrier();

    uint32_t barrier_start = 0;
    if (snrt_global_core_idx() == 0) {
        barrier_start = snrt_mcycle();
    }
    for (uint32_t c = 0; c < CHUNKS_PER_MODE; c++) {
        snrt_cluster_hw_barrier();
    }
    if (snrt_global_core_idx() == 0) {
        barrier_loop_cycles = snrt_mcycle() - barrier_start;
    }
    snrt_cluster_hw_barrier();
}

static void print_stats(int mode, int errors) {
    const pipeline_stats_t *s = &stats[mode];
    printf("PIPELINE_RESULT shape=1 mode=%d chunks=%u chunk_cols=%u status=%s errors=%d prologue=%u first_dma=%u loop=%u accel_sum=%u streamer_sum=%u compute_path_sum=%u dma_loop_sum=%u streamer_preconfig_sum=%u dma_prepared_start_sum=%u dma_next_prepare_sum=%u dma_wait_sum=%u barrier_wait_sum=%u ideal_overlap_sum=%u sync_control_overhead=%u\n",
           mode, CHUNKS_PER_MODE, CHUNK_COLS,
           errors ? "FAIL" : "PASS", errors, s->prologue_cycles,
           s->first_dma_cycles, s->loop_cycles, s->accel_sum,
           s->streamer_sum, s->compute_path_sum, s->dma_loop_sum,
           s->preconfig_sum, s->dma_prepared_start_sum,
           s->dma_next_prepare_sum, s->dma_wait_sum,
           s->barrier_wait_sum, s->ideal_overlap_sum,
           s->sync_control_overhead);
}

int main(void) {
    const shape_cfg_t *cfg = &shape_cfgs[SHAPE_INDEX];
    if (cfg->array_shape != 1 || cfg->meshRow != 4 ||
        cfg->meshCol != 8 || N0_TOTAL != N1_TOTAL ||
        CHUNK_COLS < 16u || (CHUNK_COLS % 16u) != 0u ||
        (CHUNK_COLS % (uint32_t)cfg->meshCol) != 0u ||
        (N0_TOTAL % CHUNK_COLS) != 0) {
        if (snrt_global_core_idx() == 0) {
            printf("Invalid Shape-1/common16 configuration\n");
        }
        return 1;
    }

    if (snrt_global_core_idx() == 0) {
#if USE_XDMA == 1
        const char *dma_shape = "idma_stage+xdma_nd";
#elif USE_XDMA == 2
        const char *dma_shape = "idma_to_xdma_mmio+xdma_nd";
#else
        const char *dma_shape = "idma_2d_size_128B";
#endif
        uint32_t wv_payload_bytes =
            2u * (CHUNK_COLS / 4u) * mode_panel_bytes(0);
        printf("WEIGHT_OVERLAP_BEGIN shape=1 mode=0 dims=(4,8,8) active_tokens=4 chunk_cols=%u chunks=%u n_tiles=%u ping_banks=16..31 pong_banks=32..47 mode0_output_banks=48..63 l3_layout=wv_interleaved_64B weight_dma_engine=%s dma_shape=%s dma_payload_bytes=%u dma_repeat=%u dma_csr_pipeline=1\n",
               CHUNK_COLS, CHUNKS_PER_MODE,
               CHUNK_COLS / (uint32_t)cfg->meshCol,
               WEIGHT_DMA_ENGINE_NAME, dma_shape, wv_payload_bytes,
               wv_payload_bytes / WV_INTERLEAVED_ROW_BYTES);
    }

    if (snrt_is_dm_core()) {
        token_dma_cycles_shared = stage_four_tokens();
    }
    snrt_cluster_hw_barrier();
    if (snrt_global_core_idx() == 0) {
        printf("TOKEN_DMA active_tokens=4 banks=0..15 cycles=%u\n",
               token_dma_cycles_shared);
    }

    run_control_calibration();
    if (snrt_global_core_idx() == 0) {
        printf("CONTROL_CALIBRATION iterations=%u branch_nop_loop=%u barrier_loop=%u barrier_increment=%u sink=%u\n",
               CHUNKS_PER_MODE, empty_loop_cycles, barrier_loop_cycles,
               barrier_loop_cycles > empty_loop_cycles
                   ? barrier_loop_cycles - empty_loop_cycles
                   : 0,
               calibration_sink);
    }

    int total_errors = 0;
    int rc = run_mode_pipeline(cfg, 0);
    if (snrt_global_core_idx() == 0 && rc == 0) {
        int errors = check_mode0(cfg);
        total_errors += errors;
        print_stats(0, errors);
    }
    snrt_cluster_hw_barrier();

    if (snrt_global_core_idx() == 0) {
        if (rc != 0) {
            printf("PIPELINE_TIMEOUT code=%d\n", rc);
            total_errors += rc;
        }
        printf("FINAL_RESULT shape=1 mode=0_only weight_overlap=1 weight_dma_engine=%s accelerator_streamer_csr_pipeline=1 dma_csr_pipeline=1 status=%s total_errors=%d\n",
               WEIGHT_DMA_ENGINE_NAME, total_errors ? "FAIL" : "PASS",
               total_errors);
    }
    return snrt_global_core_idx() == 0 ? total_errors : 0;
}
