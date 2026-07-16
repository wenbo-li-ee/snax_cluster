# V144 full-workload cycle result

Workload:

```text
M=8, K0=2048, N0=1408, K1=1408, N1=1024
```

Hardware configuration:

```text
cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane_a4_b2_layout_v144.hjson
sparse entries: A4 / B0-2 / B1-2 / D0-1 / D1-1
```

App:

```text
target/snitch_cluster/sw/apps/snax-versacore-int16x4-multishape-k8-8x4-v144-layout
```

## V144 RTL result

| Shape | Mode | Accelerator cycles | Streamer cycles | Wall cycles | Result |
|---|---|---:|---:|---:|---|
| S0 | Mode0 | 92984 | 93009 | 260311 | PASS |
| S0 | Mode1 | 46577 | 46608 | 287231 | PASS |
| S1 | Mode0 | 49456 | 49481 | 139010 | PASS |
| S1 | Mode1 | 23385 | 23416 | 149439 | PASS |
| S2 | Mode0 | 23284 | 23309 | 267619 | PASS |
| S2 | Mode1 | 11705 | 11736 | 82446 | PASS |

All six checks passed with total error 0.  DMA staging took 77363 cycles and
the TCDM image ended at byte 4494208.

## Streamer comparison with padded-contiguous L15

The baseline numbers are from
`snax_agent_dev_log/multishape_k8_4lane_mode1_contiguous_l15_20260507/cycle_comparison.md`.

| Shape | Mode | L15 | V144 | Delta | Delta % |
|---|---|---:|---:|---:|---:|
| S0 | Mode0 | 90141 | 93009 | +2868 | +3.18% |
| S0 | Mode1 | 46613 | 46608 | -5 | -0.01% |
| S1 | Mode0 | 45110 | 49481 | +4371 | +9.69% |
| S1 | Mode1 | 23606 | 23416 | -190 | -0.80% |
| S2 | Mode0 | 22573 | 23309 | +736 | +3.26% |
| S2 | Mode1 | 11342 | 11736 | +394 | +3.47% |

Summary:

| Group | L15 | V144 | Delta | Delta % |
|---|---:|---:|---:|---:|
| Mode0 sum | 157824 | 165799 | +7975 | +5.05% |
| Mode1 sum | 81561 | 81760 | +199 | +0.24% |
| All six | 239385 | 247559 | +8174 | +3.41% |

The streamer counters are the primary comparison metric.  Wall cycles also
include software clearing/checking, and V144 checks sparse intermediate panel
holes, so they are not directly comparable to the older L15 harness wall
numbers.
