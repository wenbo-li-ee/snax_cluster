# Design Notes: 4-Lane Postproc Redo

## Source/Generated Mismatch Before Edits

Compared:

- `snax_cluster/hw/chisel_acc/src/main/scala/snax_acc/versacore/DualVersaCoreSwigluGen.scala`
- `snax_cluster/target/snitch_cluster/generated/snax_dual_versacore_swiglu/snax_dual_versacore_swiglu_shell_wrapper.sv`

Findings:

- Generator source has `val DataWidthOut = DataWidthD / 2`. With current `DataWidthD = 1024`, that regenerates a 512-bit writer-facing output width.
- Generated shell wrapper currently has `parameter int unsigned DataWidthOut = 64`, which matches `PostprocLanes = 4` and `4 * 16` bits.
- Generator source emits `assign out_chunk_last_0 = (NumChunks <= 1) || (out_chunk_cnt_0 == NumChunks - 1);` and the same for path 1.
- Generated shell wrapper currently has `assign out_chunk_last_0 = 1'b1;` and `assign out_chunk_last_1 = 1'b1;`.
- This means the current generated wrapper contains manual/generated-output-only fixes that would not survive a clean `rtl-gen`.

## elem_mul_16b Before Edits

- Resource source and generated copy match.
- Both use strict joint input ready:
  - `ready_o_0 = valid_i_1 && out_can_accept`
  - `ready_o_1 = valid_i_0 && out_can_accept`
- This is not a robust source-level elasticity fix.

## Current Workload/Memory Truth

- Current 4-lane app is chained Mode0->Mode1 and uses `M=1, K=8, N=8`.
- `N=8` is a workaround to make Mode1 legal for all shapes; it is off-spec for the original `M=1, K=8, N=1` Mode 0 debug target.
- Current generated data places `delta_local_d0 = 9216` and `delta_local_d1_mode0 = 9728`.
- TCDM bank formula: `bank = (addr / 8) % 64`.
- D0 bank: `(9216 / 8) % 64 = 0`.
- D1 bank: `(9728 / 8) % 64 = 0`.
- `d_alloc = 512` for the current S0 `M=1, K=8, N=8` workload, exactly one full bank rotation, causing D0 and D1 to start in the same bank class.

## Source Repair Intent

- Correct writer width formula: `DataWidthOut = PostprocLanes * 16`.
- For `PostprocLanes = 4`, writer beat width is `64` bits, matching one 64-bit TCDM writer channel.
- Input chunk serialization still uses `NumChunks = ceil((DataWidthD / 32) / PostprocLanes)` to break a 1024-bit int32 D tile into 4-lane chunks.
- Output assembly uses `OutChunks = ceil((DataWidthOut / 16) / PostprocLanes)`. In 4-lane mode this is `1`, so each postproc chunk is one writer beat.
- D1 placement is now computed by trying `d_alloc + bank_word_bytes`, applying normal `granularity_c_d` alignment, then advancing until `(addr / 8) % 64` differs from D0. This avoids the raw `+8` pitfall when alignment rounds the address.

## Post-rtl-gen Verification

- Regenerated shell wrapper has `parameter int unsigned DataWidthOut = 64`.
- Regenerated shell wrapper has `localparam int unsigned OutChunks = (ElemsPerBeatOut + PostprocLanes - 1) / PostprocLanes`.
- `out_chunk_last_0/1` are derived from `OutChunks`, not input `NumChunks`.
- Regenerated `elem_mul_16b.sv` has `fifo0_valid`, `fifo1_valid`, and independent `ready_o_0 = !fifo0_valid || fire`, `ready_o_1 = !fifo1_valid || fire`.

## Sparse Interconnect Writer-Port Bug

- Temporary generated probes showed writer ports 32 and 33 presented valid requests but never received `q_ready`.
- Root cause in source config: `sparse_interconnect_config` used `[1, 4]` for each single-channel writer.
- In `SparseConfig`, the tuple means `(width, access_granularity)` and `inputsPerBank = width / access_granularity`.
- For `[1, 4]`, integer division gives `0` writer inputs per bank, so the single-channel writer ports are not connected to any memory-bank arbiter.
- Source config fixed to `[1, 1]` for both writers. This gives each single writer one input per bank and full bank reachability, while preserving total SNAX TCDM ports at 34.

## Post Sparse-Fix rtl-gen Verification

- Clean `rtl-gen` completed with sparse config `[[16,4],[8,4],[8,4],[1,1],[1,1],[16,1],[1,1],[1,1],[1,1]]`.
- Temporary generated `$display` probes were overwritten by regeneration; no final behavior depends on generated-only instrumentation.
- Regenerated shell still has `DataWidthOut = 64` and `OutChunks`-based output completion.
- Regenerated `elem_mul_16b.sv` still matches the source elastic-buffer implementation.
- Regenerated `SparseInterconnect.sv` now has non-constant ready and response routing for writer inputs 32 and 33.
