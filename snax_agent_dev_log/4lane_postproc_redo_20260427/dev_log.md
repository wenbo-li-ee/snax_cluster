# Dev Log: 4-Lane Postproc Redo

Date: 2026-04-27

## Phase 0 - Establish Current Truth

- Created required redo log folder and files.
- Using SNAX cluster workflow: source-first fixes, container build/sim flow, no final generated-only RTL patches.
- Relevant skills/workflows loaded: SNAX cluster workflow, VersaCore SNAX fusion design, VersaCore post-D2S integration.
- Confirmed the current generated shell wrapper and generator source diverge before edits.
- Validation target choice: start with a Mode 0-only debug app, not the current chained Mode0->Mode1 app. Rationale: Mode 0 directly validates 4-lane postproc and writer serialization, while the current chained app uses `N=8` only to keep Mode1 legal and therefore does not match the original `N=1` debug target.

## Phase 2/3/4 - Source Repairs Before Regeneration

- Updated `DualVersaCoreSwigluGen.scala`: `DataWidthOut = PostprocLanes * 16`.
- Updated generated wrapper template output assembly to use `OutChunks = ceil(ElemsPerBeatOut / PostprocLanes)` instead of input `NumChunks`.
- Updated resource `elem_mul_16b.sv` with independent 1-deep input buffers.
- Updated `wait_dual_versacore_writer()` to poll both writer busy CSRs.
- Added `snax-versacore-int16x4-4lane-mode0-debug` as focused Mode 0 validation app.
- Added `mode0_only: true` params with `M=1, K=8, N=1`.
- Updated 4-lane datagen allocator to stagger D1 after alignment until its start bank differs from D0.
- Container datagen smoke test for Mode 0 produced `delta_local_d0 = 2048` and `delta_local_d1_mode0 = 2128`, banks 0 and 10 respectively. Host datagen was not usable because host Python lacks `numpy`.

## Phase 5 - Regeneration

- Ran inside `barnard3`:
  `make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_4lane_postproc_v2.hjson`
- Result: completed successfully with existing Chisel dynamic-index warnings.
- Regenerated wrapper now has `DataWidthOut = 64`.
- Regenerated wrapper output assembly now uses `OutChunks`.
- Regenerated `elem_mul_16b.sv` contains independent input buffering, confirming the resource-source fix propagated through `rtl-gen`.

## Phase 6 - Focused Simulation Diagnosis

- Mode 0 debug app and simulator build succeeded.
- Initial sim timed out in `wait_dual_versacore_writer()`; accelerator busy had already cleared.
- Temporary generated-only probes show both postproc outputs emit one beat and both streamer writer inputs accept it.
- Both writer TCDM requests then stall forever: port 32 at `0x800`, port 33 at `0x850`, with no `q_ready`.
- Next focus: writer TCDM port connectivity/arbitration/port-count mapping, not postproc arithmetic or shell output assembly.

## Phase 6 - Sparse Interconnect Source Fix

- Identified writer TCDM stall cause: single-channel writer sparse entries were `[1,4]`, which produces `inputsPerBank = 1 / 4 = 0` in `SparseConfig`.
- Updated `cfg/snax_dual_versacore_int16x4_4lane_postproc_v2.hjson` writer sparse entries to `[1,1]`.
- Next step: clean `rtl-gen`; this should remove temporary generated probes and regenerate a writer-connected sparse interconnect.

## Phase 6 - Focused Mode 0 Pass

- Clean rebuilt shared SW library after generated-header/config changes.
- Rebuilt focused Mode 0 app and simulator from regenerated RTL.
- Clean direct simulation passed: `Mode 0 SwiGLU: PASS, Error: 0`.
- Cycle counters: `accel=21`, `streamer=45`, `wall=2772`.
- Proceeding to chained `N=8` regression only after the focused Mode 0 path is stable.

## Phase 6 - Chained Regression

- Rebuilt the existing chained `snax-versacore-int16x4-4lane-test` app using the same 4-lane config.
- Clean direct simulation completed with status 0.
- Mode 0 passed with `Error: 0`, cycles `accel=133`, `streamer=157`, `wall=2887`.
- Mode 1 D0 and D1 both passed with `Error: 0`, cycles `accel=251`, `streamer=271`, `wall=2720`.
- Updated `snax-cluster-workflow` skill note with the sparse interconnect `[width, access_granularity]` pitfall learned here.
