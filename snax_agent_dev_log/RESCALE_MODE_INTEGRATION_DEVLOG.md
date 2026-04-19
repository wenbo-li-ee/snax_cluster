# RescaleDown + Mode Integration Dev Log

## 2026-04-17 Session Start

### Current State
- Branch: `swiglue`, base commit: `6c7cf48a`
- Current architecture: 2 VersaCores + shifter_6stage + shifter_2stage + elem_adder_32b + 1 Writer
- Golden: `D = (A@W >> 2) + (A@V >> 2)`, int32 output
- CSR count: 7 RW, 2 RO (after block pipeline: 3 RO with writer_busy)

### Target Architecture
- Mode 0 (SwiGLU): VC0→rescale0→shifter6(16b)→ElemMul16b←rescale1←VC1 → rescale_mul → Writer0
- Mode 1 (GEMM): VC0→rescale0→Writer0, VC1→rescale1→Writer1
- 20 RW CSRs, 4 RO CSRs (busy, perf, writer0_busy, writer1_busy)
- 2 Writers (32ch each), DataWidthOut=2048b (int16 output)

### Implementation Plan
1. Create `rescale_down_32to16.sv` (NEW)
2. Create `elem_mul_16b.sv` (NEW)
3. Modify `DualVersaCoreSwigluGen.scala` (major rewrite)
4. Modify `Streamer.scala` (numReadOnlyReg 3→4)
5. Modify `snaxgen.py` (+1→+2 for writer busy)
6. Modify hjson config (dual writers, 20 RW CSRs)
7. Modify SW header/lib
8. Modify datagen.py (golden model)
9. Modify test.c (3-step SwiGLU flow)
10. Build and test

---

## Changes Log

### Task 1: rescale_down_32to16.sv
- Created based on RescaleDownPE from RescaleDown.scala
- Identity params: input_zp=0, multiplier=1, output_zp=0, shift=0

### Task 2: elem_mul_16b.sv
- Based on elem_adder_32b.sv structure
- int16×int16→int32 per lane

### Task 3: DualVersaCoreSwigluGen.scala
- Major rewrite of post-proc section
- New acc2stream_1 port, RegRWCount=20, DataWidthOut=DataWidthD/2
- Mode mux, 3 rescale instances, elem_mul, remove shifter_2stage + elem_adder

### Task 4-6: Streamer/snaxgen/hjson
- numReadOnlyReg 3→4
- snaxgen.py +1→+2
- hjson: 20 RW CSRs, dual writers

### Task 7-9: SW changes
- New CSR defines, setter functions
- Updated golden model with rescale + mode switching
- Full 3-step test flow
