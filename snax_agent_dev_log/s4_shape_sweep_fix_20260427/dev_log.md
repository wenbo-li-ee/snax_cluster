# S4 Shape Sweep Fix Dev Log

Date: 2026-04-27
Workspace: `/esat/studscratch/r1015498/Thesis/original_snax`
Config: `CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_4lane_postproc_v2.hjson`

## Initial Setup

- Created the required live log folder and files before new source edits.
- Loaded and followed `snax-cluster-workflow`.
- Required references were read during planning:
  - `snax_agent_dev_log/direct_mode1_read_mode0_output_20260427/`
  - `review_log/s4_direct_mode1_review_20260427/review_en.md`
  - `dev_prompt_spec/s4_direct_mode1_read_mode0_output/prompt_spec/spec_en.md`
  - `snax_agent_dev_log/4lane_postproc_redo_20260427/`
- Current task starts from the previous direct-read state:
  - Mode1 A base is `delta_local_d0`.
  - No SW compact/copy loop remains in the chained app.
  - `delta_local_a1` is still allocated as unused compatibility padding.

## Phase 0 - S0 Baseline

- Starting phase by rereading `dev_log.md`, `debug_log_20260427.md`, and `design_notes.md`.
- Built S0 direct-read app in `barnard3` from `snax_cluster/`:
  - `make -C target/snitch_cluster/sw/apps/snax-versacore-int16x4-4lane-test CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_4lane_postproc_v2.hjson`
  - Result: up to date.
- Ran direct S0 simulation:
  - `timeout 180s ./target/snitch_cluster/bin/snitch_cluster.vlt ./target/snitch_cluster/sw/apps/snax-versacore-int16x4-4lane-test/build/snax-versacore-int16x4-4lane-test.elf`
  - Result: PASS.
  - Mode0 cycles: `accel=133`, `streamer=157`, `wall=2859`.
  - Mode1 cycles: `accel=251`, `streamer=271`, `wall=2718`.

## Phase 1 - S1 Timeout Diagnosis

- Starting phase by rereading `dev_log.md`, `debug_log_20260427.md`, and `design_notes.md`.
- Set `params.hjson` to `array_shape: 1`.
- Regenerated/rebuilt the chained app in `barnard3`:
  - `make -C target/snitch_cluster/sw/apps/snax-versacore-int16x4-4lane-test CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_4lane_postproc_v2.hjson`
  - Result: success, with existing static inline warnings.
- Generated a temporary S0 header at `/tmp/s4_s0_data.h` for comparison only.
- S1 layout from regenerated `data/data.h`:
  - Shape: `meshRow=4`, `tileSize=16`, `meshCol=4`, `M=1`, `K=8`, `N=8`, `M1=1`, `K1=2`, `N1=32`.
  - `beats_per_tile=4`.
  - Mode0 D0/D1 bounds and strides are shared by the C app:
    - bounds `{4, 8, 1, 1}`
    - strides `{8, 32, 256, 0}`
  - Mode1 A bounds `{2, 32, 1}` and strides `{128, 0, 256}`.
  - Mode1 D bounds `{4, 32, 1, 1}` and strides `{8, 32, 1024, 0}`.
  - Padded counts match real counts: Mode0 `128 == 128`, Mode1 `512 == 512`.
  - Buffer start banks:
    - `delta_local_a=0`, bank 0.
    - `delta_local_b0=1024`, bank 0.
    - `delta_local_b1=5120`, bank 0.
    - `delta_local_d0=9216`, bank 0.
    - `delta_local_d1_mode0=9488`, bank 34.
    - `delta_local_a1=9760`, bank 4.
    - `delta_local_w2l=10016`, bank 36.
    - `delta_local_w2r=14112`, bank 36.
    - `delta_local_mode1_d0=18208`, bank 36.
    - `delta_local_mode1_d1=19248`, bank 38.
  - Buffer intervals by emitted data lengths do not overlap.
- S1 direct simulation with `timeout 600s` produced no app output and exited with status 124.
- Next diagnostic edit: temporary progress prints in the C app around DMA, Mode0 CSR/start, `wait_dual_versacore()`, and `wait_dual_versacore_writer()`.
- Temporary app diagnostics localized the S1 hang to `wait_dual_versacore()` after accelerator start.
- Temporary generated RTL diagnostics showed the shell emitted extra inactive zero chunks for S1, exhausting the writer's 4-beat-per-tile quota and backpressuring VC output.
- Applied source fix in `hw/chisel_acc/src/main/scala/snax_acc/versacore/DualVersaCoreSwigluGen.scala` to stop postproc chunk serialization at the active shape chunk count.
- Ran `make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_4lane_postproc_v2.hjson`.
  - Result: success with existing dynamic-index warnings.
  - Regenerated shell wrapper contains `active_num_chunks` and no temporary `[rtl-diag]` probes.
- Removed temporary app progress prints.
- Added explicit Mode0 D1 check:
  - `datagen.py` emits `mode0_d1_golden_padded`.
  - C app compares `local_d1_mode0` after Mode0 and prints separate D0/D1 pass/fail lines.
- S1 validation after fix:
  - Command: `timeout 180s ./target/snitch_cluster/bin/snitch_cluster.vlt ./target/snitch_cluster/sw/apps/snax-versacore-int16x4-4lane-test/build/snax-versacore-int16x4-4lane-test.elf`
  - Result: PASS.
  - Mode0 D0 PASS, Mode0 D1 PASS.
  - Mode1 D0 PASS, Mode1 D1 PASS.
  - Mode0 cycles: `accel=133`, `streamer=149`, `wall=2888`.
  - Mode1 cycles: `accel=254`, `streamer=266`, `wall=2748`.

## Phase 1 - S2 Timeout Diagnosis

- Starting phase by rereading `dev_log.md`, `debug_log_20260427.md`, and `design_notes.md`.
- Set `params.hjson` to `array_shape: 2`.
- Regenerated/rebuilt the chained app successfully.
- S2 layout from regenerated `data/data.h`:
  - Shape: `meshRow=2`, `tileSize=32`, `meshCol=4`, `M=1`, `K=8`, `N=8`, `M1=1`, `K1=1`, `N1=64`.
  - `beats_per_tile=2`.
  - Mode0 D0/D1 bounds and strides are shared by the C app:
    - bounds `{2, 8, 1, 1}`
    - strides `{8, 16, 128, 0}`
  - Mode1 A bounds `{1, 64, 1}` and strides `{128, 0, 128}`.
  - Padded counts match real counts: Mode0 `64 == 64`, Mode1 `512 == 512`.
  - Buffer start banks:
    - `delta_local_a=0`, bank 0.
    - `delta_local_b0=1024`, bank 0.
    - `delta_local_b1=5120`, bank 0.
    - `delta_local_d0=9216`, bank 0.
    - `delta_local_d1_mode0=9360`, bank 18.
    - `delta_local_a1=9504`, bank 36.
    - `delta_local_w2l=9632`, bank 52.
    - `delta_local_w2r=13728`, bank 52.
    - `delta_local_mode1_d0=17824`, bank 52.
    - `delta_local_mode1_d1=18864`, bank 54.
- S2 validation after fix:
  - Result: PASS.
  - Mode0 D0 PASS, Mode0 D1 PASS.
  - Mode1 D0 PASS, Mode1 D1 PASS.
  - Mode0 cycles: `accel=133`, `streamer=145`, `wall=2888`.
  - Mode1 cycles: `accel=256`, `streamer=264`, `wall=2752`.

## Phase 3 - Final S0 Restore Sweep

- Starting phase by rereading `dev_log.md`, `debug_log_20260427.md`, and `design_notes.md`.
- The first S0 restore attempt exposed a width bug in the active chunk-count helper: S0 needs to represent active count `8`, so the helper return width must be `$clog2(NumChunks + 1)` rather than the chunk-counter width.
- Updated the generator source and reran `rtl-gen` plus `bin/snitch_cluster.vlt` rebuild successfully.
- Next validation step: rerun S0/S1/S2 after this final RTL fix.
- Final post-width-fix S0 run:
  - Mode0 D0 PASS, Mode0 D1 PASS.
  - Mode1 D0 PASS, Mode1 D1 PASS.
  - Mode0 cycles: `accel=133`, `streamer=157`, `wall=2896`.
  - Mode1 cycles: `accel=251`, `streamer=271`, `wall=2731`.
- Final post-width-fix S1 run:
  - Mode0 D0 PASS, Mode0 D1 PASS.
  - Mode1 D0 PASS, Mode1 D1 PASS.
  - Mode0 cycles: `accel=133`, `streamer=149`, `wall=2888`.
  - Mode1 cycles: `accel=254`, `streamer=266`, `wall=2748`.
- Final post-width-fix S2 run:
  - Mode0 D0 PASS, Mode0 D1 PASS.
  - Mode1 D0 PASS, Mode1 D1 PASS.
  - Mode0 cycles: `accel=133`, `streamer=145`, `wall=2888`.
  - Mode1 cycles: `accel=256`, `streamer=264`, `wall=2752`.
- Restored `params.hjson` to `array_shape: 0`, rebuilt, and reran final S0 direct simulation:
  - Mode0 D0 PASS, Mode0 D1 PASS.
  - Mode1 D0 PASS, Mode1 D1 PASS.
  - Mode0 cycles: `accel=133`, `streamer=157`, `wall=2896`.
  - Mode1 cycles: `accel=251`, `streamer=271`, `wall=2731`.
