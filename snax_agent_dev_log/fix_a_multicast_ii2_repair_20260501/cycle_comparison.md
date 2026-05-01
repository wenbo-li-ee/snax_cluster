# cycle_comparison.md

Created: 2026-05-01

Final validation used the original S6 cfg and app:

- cfg: `snax_cluster/target/snitch_cluster/cfg/snax_dual_versacore_int16x4_multidim_spatial_16x2.hjson`
- ELF: `snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-multishape-16x2/build/snax-versacore-int16x4-multishape-16x2.elf`

| Shape | Mode | Baseline accel cycles | Final accel cycles | Target approx | Reduction vs baseline | Correctness |
|---|---:|---:|---:|---:|---:|---|
| S0 | Mode0 | 180236 | 115688 | 90112 | 35.8% | PASS/PASS |
| S0 | Mode1 | 90117 | 55877 | 45056 | 38.0% | PASS/PASS |
| S1 | Mode0 | 90120 | 59113 | 45056 | 34.4% | PASS/PASS |
| S1 | Mode1 | 45061 | 27542 | 22528 | 38.9% | PASS/PASS |
| S2 | Mode0 | 45062 | 28246 | 22528 | 37.3% | PASS/PASS |
| S2 | Mode1 | 22533 | 12968 | 11264 | 42.4% | PASS/PASS |

## Final Clean Run

```text
S0 Mode0 D0: PASS, Error: 0
S0 Mode0 D1: PASS, Error: 0
S0 Mode0 Cycles: accel=115688, streamer=115704, wall=124807
S0 Mode1 D0: PASS, Error: 0
S0 Mode1 D1: PASS, Error: 0
S0 Mode1 Cycles: accel=55877, streamer=55886, wall=64594
S1 Mode0 D0: PASS, Error: 0
S1 Mode0 D1: PASS, Error: 0
S1 Mode0 Cycles: accel=59113, streamer=59125, wall=67795
S1 Mode1 D0: PASS, Error: 0
S1 Mode1 D1: PASS, Error: 0
S1 Mode1 Cycles: accel=27542, streamer=27547, wall=36089
S2 Mode0 D0: PASS, Error: 0
S2 Mode0 D1: PASS, Error: 0
S2 Mode0 Cycles: accel=28246, streamer=28256, wall=36907
S2 Mode1 D0: PASS, Error: 0
S2 Mode1 D1: PASS, Error: 0
S2 Mode1 Cycles: accel=12968, streamer=12971, wall=21592
S6 multishape 16x2 total checks: 12, total error: 0
```

## Interpretation

- The old near-`2 * K_tiles * N_tiles` structural behavior is removed.
- The final cycles do not reach the ideal lower bound exactly, but they move substantially toward `K_tiles * N_tiles + constant` while preserving correctness for all 12 S6 checks.
- The remaining gap is expected to include shell/postprocess/output register latency and streamer/writer completion overhead.
