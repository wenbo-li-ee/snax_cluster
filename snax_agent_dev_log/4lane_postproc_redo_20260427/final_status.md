# Final Status: 4-Lane Postproc Redo

Status: Complete. Focused Mode 0 validation and chained N=8 regression both pass after clean source regeneration.

## Source Changes

- `DualVersaCoreSwigluGen.scala`: writer output width now derives from `PostprocLanes * 16`; output beat completion now uses `OutChunks` rather than input-side `NumChunks`.
- `elem_mul_16b.sv` resource source: strict joint input ready was replaced with independent 1-deep input buffering.
- 4-lane config: single-channel writer sparse interconnect entries changed from `[1,4]` to `[1,1]`; `[1,4]` generated zero inputs per bank and disconnected the writer TCDM ports.
- Shared dual-VersaCore SwiGLU library: `wait_dual_versacore_writer()` now waits for both writer busy CSRs.
- 4-lane datagen/app sources: added Mode 0-only generation path, added D1 bank-stagger logic that survives alignment, and added focused `snax-versacore-int16x4-4lane-mode0-debug` app.

## Temporary Debug Only

- Temporary `$display` probes were added to generated shell/streamer wrappers to classify the first timeout.
- The probes showed both postproc paths and streamer writer inputs accepted the first beat, while writer TCDM ports 32/33 never received `q_ready`.
- These generated-only probes were removed by the subsequent clean `rtl-gen`; the final state does not depend on generated manual edits.

## Workloads And Validation

- Primary validation target: Mode 0-only `M=1,K=8,N=1`, because it isolates 4-lane postproc/writer behavior and avoids the chained Mode1 `N=8` legality workaround.
- Secondary regression: existing chained Mode0->Mode1 `N=8` app, run only after the focused Mode 0 target passed.
- Clean regeneration/build flow completed inside `barnard3`: `rtl-gen`, clean shared SW library rebuild, focused app build, chained app build, and simulator rebuild.
- Focused Mode 0 direct simulation passed with status 0: `Mode 0 SwiGLU: PASS, Error: 0`, cycles `accel=21`, `streamer=45`, `wall=2772`.
- Chained direct simulation passed with status 0: Mode 0 `PASS`, Mode 1 D0 `PASS`, Mode 1 D1 `PASS`; cycles M0 `accel=133`, `streamer=157`, `wall=2887`; cycles M1 `accel=251`, `streamer=271`, `wall=2720`.

## Blocker Classification

- Remaining blocker: none observed in the focused validation or chained regression.
- The original post-regeneration hang is classified as a config/generator integration issue in sparse interconnect writer-port connectivity, not postproc arithmetic, shell output assembly, allocator bank placement, or software wait logic.
- The repository is left with behavior fixed in source/config/app/resource files and verified to survive clean `rtl-gen`.
