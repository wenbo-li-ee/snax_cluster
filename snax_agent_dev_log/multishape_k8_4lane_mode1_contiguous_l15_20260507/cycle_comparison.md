# Mode1 Padded-Contiguous L15 Cycle Record

Date: 2026-05-07

## App and Configuration

App:

```text
snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-multishape-k8-8x4-mode1-contiguous-l15
```

Hardware/software configuration:

```text
cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson
```

Simulation log directory:

```text
snax_cluster/layout_explore_logs/mode1_padded_contiguous_l15_20260507
```

## Storage Contract

Mode1 writes one logical output row per token. Each row uses the same stride as input A:

```text
A row stride = 4128 bytes = 2048 int16 payload + 16 int16 padding
Mode1 row stride = 4128 bytes
D1 base - D0 base = 2048 bytes
```

Therefore each token is stored as:

```text
[left 1024 int16][right 1024 int16][padding 16 int16]
```

## Final Simulation Results

| Shape | Mesh | Writer base delta | Mode1 row stride | Mode0 streamer | Mode1 streamer | Status |
|---|---|---:|---:|---:|---:|---|
| S0 | 8x8x4 | 2048 B | 4128 B | 90141 | 46613 | PASS |
| S1 | 4x8x8 | 2048 B | 4128 B | 45110 | 23606 | PASS |
| S2 | 2x8x16 | 2048 B | 4128 B | 22573 | 11342 | PASS |

Detailed counters:

| Shape | Mode | Accel cycles | Streamer cycles | Wall cycles | Validation |
|---|---|---:|---:|---:|---|
| S0 | Mode0 | 90117 | 90141 | 100718 | D0 PASS |
| S0 | Mode1 | 46594 | 46613 | 106914 | padded-contiguous D PASS |
| S1 | Mode0 | 45086 | 45110 | 55712 | D0 PASS |
| S1 | Mode1 | 23574 | 23606 | 59115 | padded-contiguous D PASS |
| S2 | Mode0 | 22549 | 22573 | 33206 | D0 PASS |
| S2 | Mode1 | 11310 | 11342 | 34421 | padded-contiguous D PASS |

## Comparison

| Shape | Old split-base L15 Mode1 | Contiguous no-padding L15 Mode1 | Padded-contiguous L15 Mode1 |
|---|---:|---:|---:|
| S0 | 46539 | 47330 | 46613 |
| S1 | 23558 | 23484 | 23606 |
| S2 | 11305 | 11331 | 11342 |

Delta of padded-contiguous versus contiguous no-padding:

| Shape | No-padding | Padded | Delta |
|---|---:|---:|---:|
| S0 | 47330 | 46613 | -717 |
| S1 | 23484 | 23606 | +122 |
| S2 | 11331 | 11342 | +11 |

The padded-contiguous version is the semantically correct version for input/output-compatible token rows.
