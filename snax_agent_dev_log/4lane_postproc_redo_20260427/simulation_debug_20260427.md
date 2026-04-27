# Simulation Debug: 4-Lane Postproc Redo

Date: 2026-04-27

## Initial State

- No new simulation run yet.
- Previous evidence showed writer ready collapse after one beat when D0 and D1 both targeted bank class 0.
- This redo will first validate a Mode 0-only app after source regeneration.

## Datagen Smoke Test

- Host command failed because host Python lacks `numpy`.
- Container command generated Mode 0 data successfully.
- Mode 0 `M=1, K=8, N=1` addresses from `/tmp/mode0_data_test.h`:
  - `delta_local_d0 = 2048`, bank `(2048 / 8) % 64 = 0`
  - `delta_local_d1_mode0 = 2128`, bank `(2128 / 8) % 64 = 10`
- This confirms the allocator stagger survives `granularity_c_d = 2` alignment for the focused workload.

## Focused Mode 0 Simulation - First Run

- Built `snax-versacore-int16x4-4lane-mode0-debug` and `bin/snitch_cluster.vlt` after clean `rtl-gen` and clean shared library rebuild.
- Direct single-ELF simulation timed out after 120 seconds with no app-level PASS/FAIL printf.
- Trace diagnosis:
  - Hart 0 spins in `wait_dual_versacore_writer()`.
  - Disassembly shows the loop now polls both writer busy CSRs.
  - Repeated CSR read shows `0x408` returning `1`, so writer 0 remains busy.
  - `wait_dual_versacore()` already returned before this loop, so the first precise blocker is writer completion/drain, not accelerator-compute busy.
- Next diagnostic step: add temporary generated-RTL probes only in the regenerated shell/streamer wrappers to observe shell output fires/stalls, streamer acc2stream acceptance, and writer TCDM port 32/33 request drain.

## Temporary Generated Probe Run

- Added temporary generated-only `$display` probes in:
  - `target/snitch_cluster/generated/snax_dual_versacore_swiglu/snax_dual_versacore_swiglu_shell_wrapper.sv`
  - `target/snitch_cluster/generated/snax_dual_versacore_swiglu/snax_dual_versacore_swiglu_streamer_wrapper.sv`
- Rebuilt `bin/snitch_cluster.vlt` successfully; instrumentation is diagnostic only and not a source-of-truth fix.
- 45-second focused Mode 0 sim timed out again.
- Probe evidence from `/tmp/4lane_mode0_dbg.txt`:
  - Shell output path 0 fired once with data `02260134002004c2`.
  - Shell output path 1 fired once with the same data.
  - Streamer accepted one beat on acc2stream 0 and one beat on acc2stream 1.
  - Writer TCDM port 32 then stalled at address `0x800` with `pvalid=0`.
  - Writer TCDM port 33 then stalled at address `0x850` with `pvalid=0`.
  - No writer TCDM fire was observed before timeout.
- Current classification: postproc produces data and shell/streamer handshake works for the first beat; failure is in writer TCDM request drain/connectivity/arbitration or streamer writer port mapping.

## Regeneration After Sparse-Fix

- Reran `rtl-gen` after changing writer sparse entries to `[1,1]`.
- Regeneration removed temporary generated probes.
- Verification confirms writer inputs 32/33 now have generated ready and response routing instead of being effectively disconnected.

## Clean Mode 0 Validation After Sparse-Fix

- Rebuilt focused Mode 0 app and rebuilt `bin/snitch_cluster.vlt` from regenerated RTL with no temporary probes.
- Direct single-ELF simulation command completed with status 0.
- App output:
  - `Mode 0 SwiGLU: PASS, Error: 0`
  - `M0 Cycles: accel=21, streamer=45, wall=2772`
- This validates the focused `M=1,K=8,N=1` Mode 0 writer/postproc path after clean regeneration.

## Chained N=8 Regression After Focused Pass

- Rebuilt `snax-versacore-int16x4-4lane-test` with `CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_4lane_postproc_v2.hjson`.
- Direct single-ELF simulation completed with status 0.
- App output:
  - `Mode 0 SwiGLU: PASS, Error: 0`
  - `M0 Cycles: accel=133, streamer=157, wall=2887`
  - `Mode 1 GEMM D0: PASS, Error: 0`
  - `Mode 1 GEMM D1: PASS, Error: 0`
  - `M1 Cycles: accel=251, streamer=271, wall=2720`
- This regression keeps the known chained `N=8` workload as a secondary check after the smaller Mode 0-only target passed.
