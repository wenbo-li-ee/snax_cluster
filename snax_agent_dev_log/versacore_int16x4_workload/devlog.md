# Dev Log: Dual VersaCore Multi-Shape Int16×Int4 Workload

## Phase 0 — Current State Assessment (dc66da1f)

### CSR Layout (20 RW + 2 RO)
- CSR[0]: OVERWRITE_ACCUM
- CSR[1]: ACCUM_BOUND (a_b_input_times_one_output)
- CSR[2]: OUTPUT_BOUND (output_times)
- CSR[3]: SUBTRACTIONS
- CSR[4]: ARRAY_SHAPE_CFG (selects shape index within current data_type)
- CSR[5]: DATA_TYPE_CFG (selects data type index)
- CSR[6]: MODE (0=SwiGLU, 1=GEMM)
- CSR[7-10]: RESCALE0 params (input_zp, multiplier, output_zp, shift)
- CSR[11-14]: RESCALE1 params
- CSR[15-18]: RESCALE_MUL params
- CSR[19]: START
- RO[0]: BUSY (vc0_busy || vc1_busy)
- RO[1]: PERF_COUNTER (max of both VCs)

### RescaleDown Modules — IMPLEMENTED
- `rescale_down_32to16.sv`: parameterized NUM_LANES, registered output, backpressure
- Three instances: u_rescale0, u_rescale1, u_rescale_mul
- `elem_mul_16b.sv`: int16x int16 -> int32, joint handshake, registered output
- `shifter_6stage.sv`: 6-stage pipeline, DATA_WIDTH parameterized, >>2 total

### MODE CSR — IMPLEMENTED
- mode_sel = csr_reg_set_i[6][0]
- Mode 0 (SwiGLU): rescale_mul -> both writers
- Mode 1 (GEMM): rescale0 -> Writer0, rescale1 -> Writer1

### Current Streamer Config
- Reader A: 16 channels, 6 temporal dims
- Reader B0: 128 channels, 3 temporal dims
- Reader B1: 128 channels, 3 temporal dims
- Writer D0: 32 channels, 4 temporal dims
- Writer D1: 32 channels, 4 temporal dims
- Total: 336 TCDM ports

### Current VersaCore Config
- Shape: meshRow=16, tileSize=8, meshCol=8 (1024 MACs, 128 D elems/cycle)
- Data types: int8x int8 -> int32 only
- PostprocLanes: 64
- ElemsPerBeat: 128, NumChunks: 2
- TCDM: 256 KB, 64 banks

### Key Insight
The VersaCore ALREADY supports multiple data types and shapes at design time via `snax_versacore_spatial_unrolling[data_type][array_shape]`. At runtime, `dataTypeCfg` and `arrayShapeCfg` CSRs index into this structure. Adding int16x int4 support ONLY requires adding entries in the HJSON config arrays. No Chisel code changes needed for the VersaCore PE itself.

### New Config Width Analysis
Target shapes (all 256 MACs, 32 D elements/cycle):
- S0: meshRow=8, tileSize=8, meshCol=4
- S1: meshRow=4, tileSize=8, meshCol=8
- S2: meshRow=2, tileSize=8, meshCol=16

Width calculations:
- arrayInputAWidth: max A = S0+int16: 8*8*16 = 1024 bits -> 16 channels
- arrayInputBWidth: max B = S2+int4: 8*16*4 = 512 bits -> 8 channels (also S1+int8: 8*8*8=512)
- arrayOutputDWidth: 32*32 = 1024 bits (all shapes, int32)
- DataWidthOut (int16): 32*16 = 512 bits -> 8 channels per writer
- PostprocLanes: 32 (= meshRow*meshCol for all shapes)
- ElemsPerBeat: 32, NumChunks: 1

Note: data_type=0 (int8) cannot use S2 (B width 1024 > 512). S2 only for data_type=1.

Granularity: a=4, b=2, c_d=8 (all strides align)

---

## Task 1.1 — Create HJSON Config

Status: IN PROGRESS

---

## Current Progress
- Phase 0: COMPLETE
- Next: Task 1.1 - Create HJSON config
