# Full-workload V144 S1 optimization

Workload:

```text
M=8, K0=2048, N0=1408, K1=1408, N1=1024
```

Hardware remained the validated A4/B2/B2 configuration:

```text
cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane_a4_b2_layout_v144.hjson
```

## Final layout

```text
v270_full_A4_V144_S1in144_pitch576_bank56

S1 token spatial stride = 144 B
S1 K panel pitch        = 576 B
S1 A base bank          = 56
```

A4 legality:

```text
144 / 8 = 18 = 2 mod 4
576 / 8 = 72 = 0 mod 4
56 mod 4 = 0
```

The external per-token layout and Mode1 closed loop are unchanged.  The larger
S1 panel adds exactly 128 KiB to the TCDM image: 4,494,208 B becomes
4,625,280 B.  DMA still copies only the active 16-byte token slices, so final
staging remained 77,363 cycles.

## Final RTL counters

| Shape | Mode | V144 streamer | V270 streamer | Delta | Result |
|---|---|---:|---:|---:|---|
| S0 | Mode0 | 93009 | 93009 | 0 | PASS |
| S0 | Mode1 | 46608 | 46608 | 0 | PASS |
| S1 | Mode0 | 49481 | 46667 | -2814 (-5.69%) | PASS |
| S1 | Mode1 | 23416 | 23416 | 0 | PASS |
| S2 | Mode0 | 23309 | 23308 | -1 | PASS |
| S2 | Mode1 | 11736 | 11736 | 0 | PASS |

All six checks passed with total error 0.

| Aggregate | L15 | Original V144 | V270 | V270 vs V144 | V270 vs L15 |
|---|---:|---:|---:|---:|---:|
| Mode0 sum | 157824 | 165799 | 162984 | -1.70% | +3.27% |
| Mode1 sum | 81561 | 81760 | 81760 | 0.00% | +0.24% |
| All six | 239385 | 247559 | 244744 | -1.14% | +2.24% |

The optimized A4 point therefore reduces the full-workload gap to L15 from
3.41% to 2.24%, while retaining A4 hardware fan-in.

## Search summary

Forty-six A4-legal S1 runs were evaluated.  Every completed check passed.

### Token-stride sweep

| S1 token stride | Implied minimum pitch | Streamer cycles |
|---:|---:|---:|
| 16 B | 64 B | 49481 |
| 48 B | 160 B | 49356 |
| 80 B | 256 B | 49283 |
| 112 B | 352 B | 50788 |
| 144 B | 448 B | 48470 |
| 176 B | 544 B | 49338 |

Stride 144 B was the best spatial layout, but its natural base was bank 24.

### Base-bank sweep at pitch 448 B

All 16 legal base banks (`bank mod 4 = 0`) were tested.  The best was bank 40
at 47,660 cycles.  Examples of the strong phase sensitivity are bank 48 at
51,446 and bank 56 at 52,252 cycles.

### Pitch sweep and coupled bank search

At bank 40, pitch 576 B improved to 47,598 cycles.  Re-sweeping base phase at
pitch 576 B moved the optimum to bank 56 and 46,667 cycles.  Adjacent phases
were much worse: bank 52 was 50,339 and bank 60 was 47,465 cycles.

Extending pitch at bank 56 did not improve the result:

| Pitch | Streamer cycles |
|---:|---:|
| 576 B | 46667 |
| 608 B | 49103 |
| 640 B | 50092 |
| 672 B | 48711 |
| 704 B | 47871 |
| 736 B | 47925 |
| 768 B | 47393 |
| 800 B | 48453 |

This shows that token stride, K pitch, and base bank cannot be optimized
independently.  Routing congruence proves legality, but the performance optimum
comes from the joint dynamic phase against B0, B1, D, FIFOs, and arbitration.

## Disk-space handling

Final simulations used `--disable-tracing`.  Regenerable `.dasm`, `.dump`,
`.bin`, `.dwarf`, multi-layout `data.h`, and Python cache files were removed.
The source, ELF, result logs, and this report were retained.
