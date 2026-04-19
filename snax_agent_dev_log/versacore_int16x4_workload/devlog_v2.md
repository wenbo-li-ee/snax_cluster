# Dev Log v2: Dual VersaCore Int16x4 Workload — Fix & Completion

## Date: 2026-04-19
### Base: commit `68308f20` on branch `swiglue`

---

## Phase 1: HJSON Config Fix

### Action: Remove int8x8 data type
- Changed all VersaCore parameter arrays from 2-element to 1-element (int16x4 only)
- Changed `spatial_unrolling` from 2 layers to 1 layer (kept int16x4: S0(8,8,4), S1(4,8,8), S2(2,8,16))
- After fix: `data_type=0` selects int16x4 (the only type)
- File: `cfg/snax_dual_versacore_int16x4_cluster.hjson`

---

## Phase 2: RTL-gen + Simulator Build

### Action: Build RTL and simulator for int16x4-only config
- RTL generation: `make rtl-gen CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_cluster.hjson`
- Simulator build: `make bin/snitch_cluster.vlt CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_cluster.hjson`
- TCDM configured to 8192 KB (8 MB), 64 banks, sparse interconnect
- Streamer: 48 TCDM ports (A:16ch, B0:8ch, B1:8ch, D0:8ch, D1:8ch)

---

## Phase 3: Software Workloads (5 apps)

### 3.1 Sanity Test (`snax-versacore-int16x4-sanity`)
- Minimal functional verification
- Result: **PASS**

### 3.2 Scaled 1/16 Batch (`snax-versacore-int16x4-scale16-batch`)
- Dimensions: A[8,128] × W[128,88], Mode 1: A'[8,88] × W2[88,64]
- Tiles: M=1, K=16, N=22 (Mode 0); M1=1, K1=11, N1=16 (Mode 1)
- N_chunk=22, N1_chunk=16 (single batch per mode)
- Result: **PASS** — M0 accel=705, M1 accel=353

### 3.3 Scaled 1/16 Pingpong (`snax-versacore-int16x4-scale16-pingpong`)
- Same dimensions, N_chunk=11, N1_chunk=8 (2 chunks, double-buffered)
- Result: **PASS** — M0 accel=357/chunk, M1 accel=180/chunk

### 3.4 Full-size Batch (`snax-versacore-int16x4-fullsize-batch`)
- Dimensions: A[8,2048] × W[2048,1408], Mode 1: A'[8,1408] × W2[1408,1024]
- Tiles: M=1, K=256, N=352 (Mode 0); M1=1, K1=176, N1=256 (Mode 1)
- N_chunk=176, N1_chunk=128 (2 chunks per mode)
- TCDM usage: 5,878,208 B (5.6 MB / 8 MB)
- Result: **PASS** — M0 accel=90,117, M1 accel=45,061

### 3.5 Full-size Pingpong (`snax-versacore-int16x4-fullsize-pingpong`)
- Same dimensions, N_chunk=88, N1_chunk=64 (4 chunks, double-buffered)
- Mode 1 W2 buffers overlaid on Mode 0 B buffers
- Result: **PASS** — M0 accel=45,061/chunk, M1 accel=22,533/chunk

### Critical Bug Fix: Stale Runtime Library
- **Symptom**: Fullsize-batch crashed with `Illegal Instruction` at PC 0x800030d0 after 188M cycles
- **Root cause**: `snitch_cluster_start.o` compiled April 17 with old 256KB TCDM config, but `snitch_cluster_addrmap.h` regenerated April 19 with 8MB TCDM
- **Impact**: Stack pointer at ~0x10040000 (262KB offset) fell INSIDE the B0 DMA buffer range (0x10008000–0x102C8000), causing DMA to overwrite the stack
- **Fix**: `rm -f sw/runtime/rtl-generic/build/*` → `make -C sw/runtime/rtl-generic` → rebuild all apps
- **Verification**: Disassembly showed `lui t0, 0x800` (8MB) instead of stale `lui t0, 0x40` (256KB)

---

## Phase 4: Cycle Analysis

Full measurements documented in `cycle_analysis.md`. Key findings:

| Workload | Mode | Accel Cycles | Tiles | Cycles/Tile |
|----------|------|-------------|-------|-------------|
| Scale16 Batch | M0 | 705 | 352 | 2.00 |
| Scale16 Batch | M1 | 353 | 176 | 2.01 |
| Scale16 Pingpong | M0 | 357 | 176/chunk | 2.03 |
| Scale16 Pingpong | M1 | 180 | 88/chunk | 2.05 |
| Fullsize Batch | M0 | 90,117 | 90,112 | 1.0001 |
| Fullsize Batch | M1 | 45,061 | 45,056 | 1.0001 |
| Fullsize Pingpong | M0 | 45,061 | 22,528/chunk | 2.00 |
| Fullsize Pingpong | M1 | 22,533 | 11,264/chunk | 2.00 |

- MAC pipeline depth = 2 cycles (multiply + accumulate)
- K=256: fully pipelined → 1 cycle/tile (tile k+1 multiply overlaps tile k accumulate)
- K=16: pipeline startup dominates → 2 cycles/tile
- Streamer overhead: 5–15 cycles total (negligible)

---

## Phase 5: Granularity Exploration

Written in Chinese as `granularity_zh.md`. Analyzed 4 alternative configurations:

| Config | a | b | c_d | Characteristics |
|--------|---|---|-----|-----------------|
| Current | 4 | 2 | 8 | Balanced, matches tcdm_logic_word_size |
| Config 1 | 1 | 1 | 1 | Minimum alignment, max bank conflict risk |
| Config 2 | 8 | 4 | 8 | Symmetric reader alignment |
| Config 3 | 16 | 8 | 8 | Maximum regularity, perfect tile alignment |
| Config 4 | 4 | 2 | 4 | Reduced D alignment |

**Conclusion**: For Int16x4 Shape S0, all tile sizes are already multiples of the largest granularity, so the choice only affects hardware-level bank access regularity, not memory usage. Current (4,2,8) is optimal.

---

## Phase 6: Chinese User Guide

Written as `user_guide_zh.md` covering 9 sections:
1. Hardware configuration overview
2. Workload dimensions and tiling
3. Dual-mode execution flow (Mode 0 SwiGLU + Mode 1 GEMM)
4. TCDM memory layout (batch and pingpong)
5. Batch vs pingpong programming models
6. Streamer CSR configuration
7. Build and simulation steps
8. Performance analysis
9. Troubleshooting (5 common issues)

---

## Final Progress
- [x] Read spec, devlog, understand current state
- [x] Fix HJSON config (single int16x4)
- [x] RTL-gen + simulator build
- [x] Sanity test — PASS
- [x] Scaled batch — PASS
- [x] Scaled ping-pong — PASS
- [x] Full-size batch — PASS (M0: 90,117 cycles, M1: 45,061 cycles)
- [x] Full-size ping-pong — PASS (M0: 45,061 cycles, M1: 22,533 cycles)
- [x] Cycle analysis — complete with all measurements
- [x] Granularity exploration — 4 configs analyzed (Chinese)
- [x] Chinese user guide — 9 sections
- [x] Skills update
- [x] All docs verified in correct paths
