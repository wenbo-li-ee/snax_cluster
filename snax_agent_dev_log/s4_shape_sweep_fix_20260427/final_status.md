# Final Status

Status: complete.

## Required Answers

1. S0 direct-read still passed at task start:
   - Initial S0: Mode0 PASS, Mode1 D0 PASS, Mode1 D1 PASS.
   - Initial cycles: Mode0 `accel=133`, `streamer=157`, `wall=2859`; Mode1 `accel=251`, `streamer=271`, `wall=2718`.
2. S1 timeout root cause:
   - The generated dual-VersaCore SwiGLU shell serialized all 8 physical chunks from the fixed 1024-bit VersaCore D output. S1 only has 4 active chunks/tile. The extra inactive zero chunks filled the shape-aware writer quota, writer ready dropped, VC output backpressured, and `wait_dual_versacore()` never completed.
   - Fixed in `DualVersaCoreSwigluGen.scala` by stopping the chunk serializer at `active_num_chunks(array_shape)` and using a return width that can represent `NumChunks`.
3. S2 timeout root cause:
   - Same shell/datagen contract mismatch as S1. S2 only has 2 active chunks/tile, but the shell previously emitted 8.
   - Fixed by the same active chunk-count logic.
4. S1 and S2 direct-read pass after fixes:
   - S1 final: Mode0 D0/D1 PASS, Mode1 D0/D1 PASS.
   - S2 final: Mode0 D0/D1 PASS, Mode1 D0/D1 PASS.
5. Mode0 D1 is now checked:
   - Confirmed source semantics first: Mode0 routes the same `rescale_mul` postprocessed stream to both output ports in this shell.
   - `datagen.py` emits `mode0_d1_golden_padded`; the app compares `local_d1_mode0` before Mode1.
   - Final S0 restored-state run: Mode0 D1 PASS.
6. Source files changed:
   - `snax_cluster/hw/chisel_acc/src/main/scala/snax_acc/versacore/DualVersaCoreSwigluGen.scala`
   - `snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-4lane-test/data/datagen.py`
   - `snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-4lane-test/src/snax-versacore-int16x4-4lane-test.c`
   - `.codex/skills/versacore-snax-fusion-design/SKILL.md`
   - New logs under `snax_agent_dev_log/s4_shape_sweep_fix_20260427/`
7. Exact commands run:
   - `podman exec barnard3 bash -lc 'cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster && export PATH=$PWD/.pixi/envs/default/bin:/tools/riscv/bin:$PATH && export VERILATOR_ROOT=$PWD/.pixi/envs/default/share/verilator && make -C target/snitch_cluster/sw/apps/snax-versacore-int16x4-4lane-test CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_4lane_postproc_v2.hjson'`
   - `podman exec barnard3 bash -lc 'cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster && export PATH=$PWD/.pixi/envs/default/bin:/tools/riscv/bin:$PATH && export VERILATOR_ROOT=$PWD/.pixi/envs/default/share/verilator && timeout 180s ./target/snitch_cluster/bin/snitch_cluster.vlt ./target/snitch_cluster/sw/apps/snax-versacore-int16x4-4lane-test/build/snax-versacore-int16x4-4lane-test.elf'`
   - `podman exec barnard3 bash -lc 'cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster && export PATH=$PWD/.pixi/envs/default/bin:/tools/riscv/bin:$PATH && export VERILATOR_ROOT=$PWD/.pixi/envs/default/share/verilator && timeout 600s ./target/snitch_cluster/bin/snitch_cluster.vlt ./target/snitch_cluster/sw/apps/snax-versacore-int16x4-4lane-test/build/snax-versacore-int16x4-4lane-test.elf'`
   - `podman exec barnard3 bash -lc 'cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster && export PATH=$PWD/.pixi/envs/default/bin:/tools/riscv/bin:$PATH && export VERILATOR_ROOT=$PWD/.pixi/envs/default/share/verilator && make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_4lane_postproc_v2.hjson'`
   - `podman exec barnard3 bash -lc 'cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster && export PATH=$PWD/.pixi/envs/default/bin:/tools/riscv/bin:$PATH && export VERILATOR_ROOT=$PWD/.pixi/envs/default/share/verilator && make -C target/snitch_cluster bin/snitch_cluster.vlt -j$(nproc) CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_4lane_postproc_v2.hjson'`
   - Final per-shape rebuild/sim used the same environment with build and simulation chained in one shell:
     `podman exec barnard3 bash -lc 'cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster && export PATH=$PWD/.pixi/envs/default/bin:/tools/riscv/bin:$PATH && export VERILATOR_ROOT=$PWD/.pixi/envs/default/share/verilator && make -C target/snitch_cluster/sw/apps/snax-versacore-int16x4-4lane-test CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_4lane_postproc_v2.hjson && timeout 180s ./target/snitch_cluster/bin/snitch_cluster.vlt ./target/snitch_cluster/sw/apps/snax-versacore-int16x4-4lane-test/build/snax-versacore-int16x4-4lane-test.elf'`
8. Final cycle counts:
   - S0 final/restored: Mode0 `accel=133`, `streamer=157`, `wall=2896`; Mode1 `accel=251`, `streamer=271`, `wall=2731`.
   - S1 final: Mode0 `accel=133`, `streamer=149`, `wall=2888`; Mode1 `accel=254`, `streamer=266`, `wall=2748`.
   - S2 final: Mode0 `accel=133`, `streamer=145`, `wall=2888`; Mode1 `accel=256`, `streamer=264`, `wall=2752`.
9. `delta_local_a1` Route B cleanup:
   - Deferred. Removing it would shift W2 and Mode1 D buffers across all shapes and require another full bank/layout validation plus simulation sweep. It remains unused compatibility padding.
10. Skill update:
   - Yes. Updated `versacore-snax-fusion-design` with the reusable rule that fixed-width VC output serializers must use active shape chunk counts and must size the count signal to represent `NumChunks`.

Final state:

- `params.hjson` restored to `array_shape: 0`.
- Final restored S0 direct single-ELF simulation passed.
