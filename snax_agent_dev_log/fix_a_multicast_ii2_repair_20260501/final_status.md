# final_status.md

Created: 2026-05-01

## Status

Complete. The dual-VersaCore int16x4 SwiGLU A-multicast bottleneck has been repaired for the existing S6 cfg/app, and the final clean direct ELF simulation passes all checks.

## Source Changes

- `snax_cluster/hw/chisel_acc/src/main/scala/snax_acc/versacore/DualVersaCoreSwigluGen.scala`
  - Converts the original shared-A slot into an elastic replay slot with same-cycle refill.
  - Gates each VC's A offer by its matching B-valid and advances B only on paired A/B ready.
  - Adds output quota handling and Mode0 post-quota local drain.
  - Adds a Mode0 direct-output holding register with per-writer sent bits for `OutChunks <= 1`.
- `snax_cluster/hw/chisel_acc/src/main/resources/snax_acc/versacore/silu_multilane.sv`
  - Drives `valid_o` from the actual `silu_top.valid_out` signal instead of a shadow valid pipe.

No `Array.scala`, `Accumulator.scala`, or `VersaCore.scala` internal handshake semantics were modified.

## Validation

Commands run inside `barnard3`:

```bash
make -C snax_cluster/target/snitch_cluster rtl-gen \
  CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_16x2.hjson

make -C snax_cluster/target/snitch_cluster sw \
  CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_16x2.hjson

make -C snax_cluster/target/snitch_cluster bin/snitch_cluster.vlt -j$(nproc) \
  CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_16x2.hjson

./snax_cluster/target/snitch_cluster/bin/snitch_cluster.vlt \
  ./snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-multishape-16x2/build/snax-versacore-int16x4-multishape-16x2.elf
```

Final result:

```text
S6 multishape 16x2 total checks: 12, total error: 0
```

## Cycle Summary

| Shape | Mode | Baseline accel | Final accel |
|---|---:|---:|---:|
| S0 | Mode0 | 180236 | 115688 |
| S0 | Mode1 | 90117 | 55877 |
| S1 | Mode0 | 90120 | 59113 |
| S1 | Mode1 | 45061 | 27542 |
| S2 | Mode0 | 45062 | 28246 |
| S2 | Mode1 | 22533 | 12968 |

The final design removes the structural II=2 A-refill bottleneck and improves all measured shapes/modes while preserving correctness.
