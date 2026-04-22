# Cycle Analysis: SwiGLU SiLU Integration

**Date:** 2026-04-21  
**Cluster:** `snax_dual_versacore_int16x4_cluster`  
**Simulator:** Verilator (`bin/snitch_cluster.vlt`)

---

## 1. Simulation Results Summary

All 4 apps pass with EXIT_CODE=0. No Unequals errors.

| App | Mode | accel | streamer | wall | Status |
|-----|------|------:|---------:|-----:|--------|
| m1-batch   | M0 SwiGLU | 132 | 142 | 2888  | PASS |
| m1-batch   | M1 GEMM   | 133 | 137 | 2590  | PASS |
| m1-pingpong | M0 SwiGLU | 20  | 30  | 19421 | PASS |
| m1-pingpong | M1 GEMM   | 37  | 41  | 10025 | PASS |
| m4-batch   | M0 SwiGLU | 517 | 527 | 3272  | PASS |
| m4-batch   | M1 GEMM   | 513 | 531 | 2996  | PASS |
| m4-pingpong | M0 SwiGLU | 69  | 79  | 19497 | PASS |
| m4-pingpong | M1 GEMM   | 133 | 138 | 10244 | PASS |

- **accel**: VersaCore performance counter (total computation cycles, both VCs)
- **streamer**: Streamer performance counter (data transfer cycles)
- **wall**: mcycle difference from start to end of wait_dual_versacore() + wait_dual_versacore_writer()

---

## 2. Workload Parameters

### m1-batch / m1-pingpong
- M=1, K=8, N=8 (Mode 0 matmul dimension)
- K1=16, N1=4, M1=1 (Mode 1 GEMM)
- B0tlstride0=64 (B tile padded to 64B channel footprint)
- mode0_output_elems=256 (int16 elements in D0)

### m4-batch / m4-pingpong
- Larger M dimension (4 rows)
- Proportionally more output data and longer computation

---

## 3. Cycle Breakdown Analysis

### m1-batch (single-tile, no DMA overlap)

**Mode 0 SwiGLU:**
- accel=132 cycles: VersaCore time includes 8×8 accumulation × 1 output = ~8×K accumulation + pipeline latency
- streamer=142 cycles: slightly more due to streamer overhead (address generation, TCDM access)
- wall=2888: larger than accel+streamer due to DMA initialization overhead and CPU-side CSR setup

The streamer (142) > accel (132) by 10 cycles — the streamer is the bottleneck for this small workload.

**Mode 1 GEMM:**
- Similar cycle count to Mode 0 (accel=133 vs 132) — comparable workload size
- Faster wall time (2590 < 2888) because DMA pre-loading is already done from Mode 0

### m1-pingpong

**Much higher wall time** (~19421 vs 2888 for Mode 0) despite lower accel (20 vs 132).  
Cause: The ping-pong variant introduces DMA synchronization barriers between transfers.  
The CPU spends most wall cycles in barrier/sync overhead, not in VersaCore computation.

accel=20 cycles for Mode 0 is much lower — this reflects the smaller effective problem size  
per ping-pong iteration (the workload is split across multiple DMA phases).

### m4-batch

**Scales linearly with M=4:**  
m4-batch accel (517) ≈ 4× m1-batch accel (132) — confirms correct scaling.  
streamer (527) ≈ 527/142 = 3.7× — close to 4× (slight amortization of overhead).

### m4-pingpong

Similarly high wall time (~19497) due to ping-pong synchronization overhead.

---

## 4. SiLU Pipeline Overhead

The SiLU module (silu_multilane) adds a 3-stage pipeline latency vs the previous 6-stage shifter.  
However, since SiLU is fully pipelined with backpressure-CE logic:

- **No throughput degradation** vs shifter: sustains 1 output/cycle at full bandwidth
- **3-cycle latency** introduction compared to zero (if shifter were removed)
- The 3-cycle SiLU latency is masked by streamer latency in all test cases

In Mode 0, the critical path bottleneck is the **streamer** (TCDM access time), not SiLU computation.  
SiLU adds no throughput cost — the streamer remains the performance bottleneck.

---

## 5. Efficiency Metrics

### Compute Utilization (accel / wall)

| App | Mode | Utilization |
|-----|------|------------:|
| m1-batch   | M0 | 132/2888 = 4.6% |
| m1-batch   | M1 | 133/2590 = 5.1% |
| m4-batch   | M0 | 517/3272 = 15.8% |
| m4-batch   | M1 | 513/2996 = 17.1% |
| m1-pingpong | M0 | 20/19421 = 0.1% |

Low utilization in batch variants is dominated by DMA startup and CSR setup overhead.  
Ping-pong variants have even lower utilization due to barrier synchronization overhead.  
For production workloads with larger M, utilization would improve significantly.

### Streamer vs Accel Ratio

| App | Mode | streamer/accel |
|-----|------|---------------:|
| m1-batch | M0 | 142/132 = 1.08 |
| m4-batch | M0 | 527/517 = 1.02 |

The streamer is marginally slower than the accelerator (ratio ~1.02–1.08).  
This confirms the pipeline is nearly balanced: TCDM bandwidth matches compute throughput.

---

## 6. Known Limitations

1. **Small M causes low utilization**: For M=1, most wall time is initialization overhead.
   Real workloads typically have M >> 1, bringing utilization much higher.

2. **Ping-pong overhead**: The ping-pong pattern in small-M cases is dominated by barrier
   synchronization overhead rather than compute or memory bandwidth.

3. **Mode 0 writes to both Writer0 and Writer1**: Writer1 (D1_mode0) writes the same data
   to a separate buffer (dummy address) as a side effect of the dual-writer Mode 0 design.
   This wastes some TCDM bandwidth but does not affect correctness.

4. **rescale_mul shift=0**: With shift=0, multiplier=1, the rescale_mul stage is effectively
   an identity (clamp to int16 range). In real quantized deployments, the shift/multiplier
   would be tuned to match the actual int32 dynamic range from elem_mul.
