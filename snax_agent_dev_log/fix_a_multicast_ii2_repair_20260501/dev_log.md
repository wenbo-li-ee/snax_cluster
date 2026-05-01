# dev_log.md

Created: 2026-05-01

- 2026-05-01: Read the SNAX cluster workflow skill and the repair spec. The active target is the existing S6 cfg `snax_dual_versacore_int16x4_multidim_spatial_16x2.hjson` and app `snax-versacore-int16x4-multishape-16x2`.
- 2026-05-01: Read the independent II=2 analysis, second-round review, deep handshake analysis, and prior S6/fix logs. The current generator checkout still contains the original single registered A replay slot, not the unverified 2-slot FIFO variant described in the second-round review.
- 2026-05-01: Baseline S6 cycle data from the prior clean log is functionally correct but follows `2 * K_tiles * N_tiles + constant`: S0M0 180236, S0M1 90117, S1M0 90120, S1M1 45061, S2M0 45062, S2M1 22533.
- 2026-05-01: Starting source edit in `DualVersaCoreSwigluGen.scala`, localized to the generated shell-wrapper shared-A path. VersaCore internal files are intentionally untouched.
- 2026-05-01: Implemented the first repair attempt: a two-entry A FIFO with atomic dual issue, keepOutput mirror capability gating, direct B pass-through, and full-state same-cycle pop+push support.
- 2026-05-01: Started `rtl-gen` inside `barnard3` with `CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_16x2.hjson`.
- 2026-05-01: `rtl-gen` completed successfully. Inspected generated `snax_dual_versacore_swiglu_shell_wrapper.sv`; it contains the expected `a_fifo_count`, `a_joint_fire`, and `a_src_data` wiring.
- 2026-05-01: `make -C snax_cluster/target/snitch_cluster sw CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_16x2.hjson` completed successfully inside `barnard3`; the S6 multishape ELF was rebuilt.
- 2026-05-01: `make -C snax_cluster/target/snitch_cluster bin/snitch_cluster.vlt -j$(nproc) CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_16x2.hjson` completed successfully inside `barnard3`.
- 2026-05-01: Direct S6 simulation of Attempt 1 stalled in `S0 Mode0 wait accel`; stopped manually after several minutes at full CPU.
- 2026-05-01: Root-cause refinement for the stall: `vc*_can_accept` did not require `vc*_busy`, so A could be popped while the VC was not yet in `sBUSY` and its internal subtraction/control valid was still false. Starting Attempt 2 by adding `vc0_busy/vc1_busy` to the local can-accept gates.
- 2026-05-01: Attempt 2 `rtl-gen` completed successfully inside `barnard3`.
- 2026-05-01: Attempt 2 `sw` and `bin/snitch_cluster.vlt` builds completed successfully inside `barnard3`.
- 2026-05-01: Direct S6 simulation of Attempt 2 again stalled in `S0 Mode0 wait accel`; stopped manually. Starting Attempt 3: keep A valid assertion based on loop-free local capability, but pop the A FIFO only on actual dual A handshake (`vc0_in_a_ready && vc1_in_a_ready` while both A valids are asserted).
- 2026-05-01: Attempt 3 `rtl-gen` completed successfully inside `barnard3`. Re-read the current logs and inspected the generated shell wrapper; generated SV contains `a_issue_valid` and `a_joint_fire = a_issue_valid && vc0_in_a_ready && vc1_in_a_ready` as intended.
- 2026-05-01: Attempt 3 `sw` build completed successfully inside `barnard3`; the S6 multishape ELF was rebuilt.
- 2026-05-01: Attempt 3 `bin/snitch_cluster.vlt` build completed successfully inside `barnard3`.
- 2026-05-01: Attempt 3 direct S6 simulation again stalled in `S0 Mode0 wait accel`; stopped manually. Re-read relevant VersaCore/Array/DecoupledCat ready logic and identified the likely deadlock: the wrapper waited for B valid before asserting A valid, while VC A/B ready is itself enabled by all inputs being valid. Starting Attempt 4 by removing B-valid from the wrapper A issue capability gate while retaining confirmed dual-ready FIFO pop.
- 2026-05-01: Attempt 4 `rtl-gen` completed successfully inside `barnard3`.
- 2026-05-01: Attempt 4 `sw` build completed successfully inside `barnard3`; the S6 multishape ELF was rebuilt again after header regeneration.
- 2026-05-01: Attempt 4 `bin/snitch_cluster.vlt` build completed successfully inside `barnard3`.
- 2026-05-01: Attempt 4 direct S6 simulation still stalled in `S0 Mode0 wait accel`; stopped manually. Starting Attempt 5: remove wrapper-side VC capability mirroring from A valid assertion and make actual dual VC A ready the only consume authority.
- 2026-05-01: Attempt 5 `rtl-gen`, `sw`, and `bin/snitch_cluster.vlt` completed successfully inside `barnard3`.
- 2026-05-01: Attempt 5 direct S6 simulation still stalled in `S0 Mode0 wait accel`; stopped manually after no progress. Starting targeted wrapper signal instrumentation to identify which ready/valid path is blocking the first tile.
- 2026-05-01: Temporary instrumentation showed Attempt 5 accepts A/B input but later wedges with output writer ready low. Counters showed the two-entry A prefetch experiment allows output handshakes to run ahead of matched VC D accepts. Replaced it with Attempt 6: a single fall-through A slot with same-cycle pop/push and atomic dual ready consumption.
- 2026-05-01: Attempt 6 `rtl-gen` completed successfully inside `barnard3`. The source no longer contains the temporary debug display names and the A path is now the single fall-through slot.
- 2026-05-01: Attempt 6 `sw` build completed successfully inside `barnard3`; the S6 multishape ELF was regenerated.
- 2026-05-01: Attempt 6 `bin/snitch_cluster.vlt` Verilator model build completed successfully inside `barnard3`.
- 2026-05-01: Attempt 6 direct S6 simulation still stalled in `S0 Mode0 wait accel`; stopped manually after no cycle print. Continuing with targeted shell-wrapper inspection/debug.
- 2026-05-01: Attempt 7 added an output quota drain after debug showed the repaired A path can expose one surplus postprocess/output beat beyond the streamer writer count. The quota is derived from `output_times * active_num_chunks(array_shape)` and drops only post-quota surplus beats.
- 2026-05-01: Attempt 7 moved S0 Mode0 past accelerator wait but then stalled at streamer wait with A reader incomplete (`a=88800` vs B/output complete). Starting Attempt 8: add a post-compute A reader drain that is armed only after an accepted CSR start has completed.
- 2026-05-01: Attempt 8 completed far enough to compare results but produced wrong S0 Mode0 data, proving the post-compute drain/output quota approach discards real compute work. Starting Attempt 9: remove those drains and use a two-entry A queue with joint issue pulse and loop-free B/output capability gating.

- 2026-05-01: Attempt 9 direct S6 run still stalled in S0 Mode0 after aligned A/B issue. Debug at cyc=196608 showed a=b0=b1=35328, o0=o1=2817, afifo=2, both writer readies low, and both VC D valids high. This confirms the joint A issue pulse fixes the launch/atomicity side but exposes the output assembler emitting one beat beyond the streamer writer program.

- 2026-05-01: Starting Attempt 10: removed the `ctrl_accept` FIFO clear. The app starts streamer before accelerator CSR start, so pre-start A beats in the two-entry queue are valid data and must be preserved for the first logical K steps.

- 2026-05-01: Interrupted the Attempt 10 rebuild before simulation and pivoted to Attempt 11. Attempt 11 restores the original per-VC A replay/sent semantics, but changes the slot to accept the next A beat in the same cycle the current beat has been accepted by both VCs. This directly removes the refill bubble without changing VersaCore internal ready logic or adding B throttling.

- 2026-05-01: Attempt 11 direct S6 run still stalled at S0 Mode0 with output writer overrun (`o0=o1=2817` for a 2816-beat writer quota) while both VCs remained busy. Starting Attempt 12: add output quota counters in the shell output assemblers, derived from `csr_reg_set_i[2] * active_num_chunks(array_shape)`, to stop emitting beyond the programmed writer length while allowing postprocess surplus to drain locally. No post-compute A drain is reintroduced.

- 2026-05-01: Attempt 12 reached result comparison but failed S0 Mode0 from output index 188 onward. This falsifies output quota/drain as a correctness-preserving fix. Starting Attempt 13: remove output quota and instead prevent each B stream from handshaking unless the matching A replay slot is valid for that same VC (`vc*_in_a_valid`). This preserves independent VC B paths but prevents B from advancing without its A beat.

- 2026-05-01: Later attempts converged on the final input repair: preserve the original shared-A replay/sent contract, allow same-cycle A refill when both VCs have accepted the old beat, and make each VC's B handshake conditional on the matching paired A/B acceptance. This kept A/B counters locked without touching VersaCore internals.

- 2026-05-01: Output-side debugging showed that the repaired input path exposed an independent shell issue. For S6 Mode0, `OutChunks == 1`, and the direct duplicated output path can let writer0 and writer1 accept a shared result beat at different times. Added a Mode0 direct-output holding register with per-writer sent bits so each duplicated postprocess beat is delivered exactly once to each writer.

- 2026-05-01: Removed temporary Verilator debug displays. Final source edits are in `DualVersaCoreSwigluGen.scala` and `silu_multilane.sv`.

- 2026-05-01: Ran the required clean validation loop inside `barnard3` with the existing S6 cfg:
  - `make -C snax_cluster/target/snitch_cluster rtl-gen CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_16x2.hjson`
  - `make -C snax_cluster/target/snitch_cluster sw CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_16x2.hjson`
  - `make -C snax_cluster/target/snitch_cluster bin/snitch_cluster.vlt -j$(nproc) CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_16x2.hjson`

- 2026-05-01: Final direct ELF simulation completed successfully:
  - S0 Mode0: PASS/PASS, accel=115688
  - S0 Mode1: PASS/PASS, accel=55877
  - S1 Mode0: PASS/PASS, accel=59113
  - S1 Mode1: PASS/PASS, accel=27542
  - S2 Mode0: PASS/PASS, accel=28246
  - S2 Mode1: PASS/PASS, accel=12968
  - Total checks: 12, total error: 0
