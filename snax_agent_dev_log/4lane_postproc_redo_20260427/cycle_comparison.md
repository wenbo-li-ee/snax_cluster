# Cycle Comparison: 4-Lane Postproc Redo

## 2026-04-27 Mode 0 Debug First Sim

- App: `snax-versacore-int16x4-4lane-mode0-debug`, `M=1,K=8,N=1`, Mode 0 only.
- Result: timeout after 120 seconds, no cycle count printed by app.
- Earliest classified stall: software reaches `wait_dual_versacore_writer()` after accelerator busy clears; writer 0 busy CSR remains asserted.

## 2026-04-27 Mode 0 Debug After Sparse-Fix

- App: `snax-versacore-int16x4-4lane-mode0-debug`, `M=1,K=8,N=1`, Mode 0 only.
- Result: PASS, `Error: 0`.
- Cycles: `accel=21`, `streamer=45`, `wall=2772`.
- Classification: no remaining blocker for focused Mode 0 writer/postproc validation.

## 2026-04-27 Chained N=8 Regression After Sparse-Fix

- App: `snax-versacore-int16x4-4lane-test`, chained Mode0->Mode1 workload.
- Result: PASS, Mode 0 `Error: 0`, Mode 1 D0 `Error: 0`, Mode 1 D1 `Error: 0`.
- Mode 0 cycles: `accel=133`, `streamer=157`, `wall=2887`.
- Mode 1 cycles: `accel=251`, `streamer=271`, `wall=2720`.
- Classification: no remaining blocker for the existing chained regression workload.
