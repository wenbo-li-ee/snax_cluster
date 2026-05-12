# Cycles: small MoE K8 L15 weights-first app

**Date:** 2026-05-12  
**Config:** `cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson`  
**App:** `target/snitch_cluster/sw/apps/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15`  
**ELF:** `target/snitch_cluster/sw/apps/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15/build/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15.elf`

## Build and Run

After the C refactor, the app no longer uses `SELECT_LAYOUT`, `SELECT_SHAPE`, or `RUN_MODE1` to choose the run. It always uses `layout_cfgs[0]`, then runs S0/S1/S2, and for each shape runs Mode0 followed by Mode1.

```bash
make -C target/snitch_cluster/sw/apps/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15 all \
  CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson \
  FAST_BUILD=1

./target/snitch_cluster/bin/snitch_cluster.vlt \
  ./target/snitch_cluster/sw/apps/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15/build/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15.elf
```

## Dataset Placement

```text
layout              : L15 l15_weights_first_padded_1024_per_token
DMA staging cycles  : 3520
TCDM used           : 235008 bytes
A row stride        : 2080 bytes
B0 bank             : 0
B1 bank             : 34
D0 bank             : 0
Mode1 D0 bank       : 32
Mode1 D1 - D0       : 1024 bytes
Mode1 row stride    : 2080 bytes
```

## Cycle Results

| Shape | Mode | Correctness | Accelerator cycles | Streamer cycles | Wall cycles |
|---|---|---|---:|---:|---:|
| S0 | Mode0 | PASS | 4102 | 4126 | 42311 |
| S0 | Mode1 | PASS | 2180 | 2199 | 235108 |
| S1 | Mode0 | PASS | 2058 | 2082 | 28702 |
| S1 | Mode1 | PASS | 1110 | 1129 | 126607 |
| S2 | Mode0 | PASS | 1037 | 1061 | 22794 |
| S2 | Mode1 | PASS | 544 | 563 | 72594 |

Final simulation result:

```text
total checks: 6
total error : 0
```

## Notes

- The VLT binary was reused; no RTL generation or hardware rebuild was run.
- Software was rebuilt inside the `barnard3` container.
- The two compiler warnings are the existing `static function 'csrw_ss' is used in an inline function with external linkage` warnings from `snax-dual-versacore-swiglu-lib.h`; the ELF linked and simulated correctly.
- Wall cycles changed after the readability refactor because the program now has a cleaner function structure and still prints progress messages around each phase. Correctness is unchanged. Accelerator/streamer counters are the values to compare for hardware work.
