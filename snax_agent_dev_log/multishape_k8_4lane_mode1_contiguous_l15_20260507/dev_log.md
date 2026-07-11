# Mode1 Contiguous L15 Development Log

Started: 2026-05-07

## Goal

Implement a new K8 8x4 4-lane app for the workload:

```text
(Swish(xW) * xV) W2
```

Mode1 still sends W2 as two physical halves to the dual VersaCore, but those halves are logically the left and right halves of one output matrix. Therefore, for each token, the Mode1 output must be stored as one contiguous 2048-element int16 row:

```text
[token left 1024 int16][token right 1024 int16]
```

This requires the two Mode1 writer base addresses to differ by exactly `1024 * 2 = 2048` bytes. The app uses the best balanced layout found in the previous exploration, L15:

```text
A pad 32 B + B1 bank34 + W2_left bank16 + Mode1 D0 bank32
```

## New App

Created app:

```text
snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-multishape-k8-8x4-mode1-contiguous-l15
```

Registered it under:

```text
cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson
```

The app was copied from the validated `mode1-pertoken-layout-explore` harness and reduced to one layout only.

## Datagen Changes

The new datagen keeps activation and weight constraints unchanged:

- A is generated per token.
- W/V/W2 are generated in shape0 physical tile layout.
- W2 is still physically split into `W2_left` and `W2_right` for the accelerator readers.

Placement changes:

```text
delta_local_mode1_d0 = colored_offset(..., m1d0_color=256)
delta_local_mode1_d1 = delta_local_mode1_d0 + 2048
```

Mode1 writer stride changes:

```text
old per-half token stride: 2048 B
new combined token stride: 4096 B
```

For each shape, datagen emits:

```text
mode1_D_tstride = {8, 4096, meshCol * 2, 0}
```

The golden output is now one combined array per shape. It concatenates the two logical halves per token:

```text
mode1_combined[token] = concat(mode1_left[token, 0:1024], mode1_right[token, 0:1024])
```

## Runtime Changes

The C harness now checks one contiguous Mode1 output block from `delta_local_mode1_d0`:

```text
Mode1 D contiguous: PASS/FAIL
```

It also prints the runtime placement sanity field:

```text
M1D1_minus_D0=2048
```

This confirms the two writer bases are separated by exactly `1024 * 2` bytes.

## Build and Simulation

Command pattern used:

```bash
APP=snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-multishape-k8-8x4-mode1-contiguous-l15
CFG=cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson
SIM=snax_cluster/target/snitch_cluster/bin/snitch_cluster.vlt
LOGDIR=snax_cluster/layout_explore_logs/mode1_contiguous_l15_20260507
```

Built three shape-split ELFs with `FAST_BUILD=1` and ran all three simulations in parallel with `timeout 600`.

Log directory:

```text
snax_cluster/layout_explore_logs/mode1_contiguous_l15_20260507
```

## Results

All three shape-split simulations exited 0. Each run performed Mode0 D0 checking and the new Mode1 contiguous-output checking. Total error was 0 for all shapes.

| Shape | M1D1 - M1D0 | Mode0 streamer | Mode1 streamer | Result |
|---|---:|---:|---:|---|
| S0 | 2048 B | 90141 | 47330 | PASS |
| S1 | 2048 B | 45110 | 23484 | PASS |
| S2 | 2048 B | 22573 | 11331 | PASS |

Detailed counters:

| Shape | Mode | Accel cycles | Streamer cycles | Wall cycles | Check |
|---|---|---:|---:|---:|---|
| S0 | Mode0 | 90117 | 90141 | 100716 | D0 PASS |
| S0 | Mode1 | 47311 | 47330 | 58124 | contiguous D PASS |
| S1 | Mode0 | 45086 | 45110 | 55710 | D0 PASS |
| S1 | Mode1 | 23465 | 23484 | 34306 | contiguous D PASS |
| S2 | Mode0 | 22549 | 22573 | 33110 | D0 PASS |
| S2 | Mode1 | 11312 | 11331 | 22123 | contiguous D PASS |

## Comparison Against Previous L15 Split-Base Result

Previous L15 split-base Mode1 streamer cycles:

```text
S0/S1/S2 = 46539 / 23558 / 11305
```

New contiguous-output L15 Mode1 streamer cycles:

```text
S0/S1/S2 = 47330 / 23484 / 11331
```

Delta versus previous L15:

| Shape | Previous L15 | Contiguous L15 | Delta |
|---|---:|---:|---:|
| S0 | 46539 | 47330 | +791 |
| S1 | 23558 | 23484 | -74 |
| S2 | 11305 | 11331 | +26 |

The new storage contract is functionally correct and keeps Mode0 unchanged. It improves S1 slightly, while S0 and S2 are slower than the earlier split-base layout. This is expected because the new semantic requirement fixes D1 directly after D0 for each token and changes the Mode1 writer token stride to 4096 B.

## Final Status

The requested contiguous-output L15 app is implemented, registered, built, and simulation-validated for all three shapes.

## 2026-05-07 Update: Token Padding Added to Mode1 Output

The follow-up requirement is to make Mode1 writeback use the same per-token row format as the input A tensor. The previous version made the two writer halves contiguous inside one token but packed tokens at a 4096-byte stride:

```text
token i: [left1024][right1024]
token i+1 immediately follows
```

The updated version keeps the same intra-token semantic rule but adds the same 32-byte padding used by input A:

```text
token i: [left1024][right1024][32-byte padding]
```

The decisive streamer rule is:

```text
D0 base = delta_local_mode1_d0
D1 base = delta_local_mode1_d0 + 2048
Mode1 writer token stride = A row stride = 4128
```

This means writer0 writes the left half at `D0 + token * 4128`, writer1 writes the right half at `D0 + token * 4128 + 2048`, and the final 32 bytes of every row are padding.

The datagen now emits a padded golden array named `mode1_padded_golden`. The C harness clears the entire Mode1 padded output region before Mode1 starts, then compares all `meshRow * 2064` int16 values. Therefore, the check validates both the 2048 payload values and the 16 int16 padding values per token.

Final padded-output simulation results:

| Shape | M1D1 - M1D0 | Mode1 row stride | Mode0 streamer | Mode1 streamer | Result |
|---|---:|---:|---:|---:|---|
| S0 | 2048 B | 4128 B | 90141 | 46613 | PASS |
| S1 | 2048 B | 4128 B | 45110 | 23606 | PASS |
| S2 | 2048 B | 4128 B | 22573 | 11342 | PASS |

Detailed padded-output counters:

| Shape | Mode | Accel cycles | Streamer cycles | Wall cycles | Check |
|---|---|---:|---:|---:|---|
| S0 | Mode0 | 90117 | 90141 | 100718 | D0 PASS |
| S0 | Mode1 | 46594 | 46613 | 106914 | padded-contiguous D PASS |
| S1 | Mode0 | 45086 | 45110 | 55712 | D0 PASS |
| S1 | Mode1 | 23574 | 23606 | 59115 | padded-contiguous D PASS |
| S2 | Mode0 | 22549 | 22573 | 33206 | D0 PASS |
| S2 | Mode1 | 11310 | 11342 | 34421 | padded-contiguous D PASS |

Compared with the previous no-padding contiguous version, the padded stride changes Mode1 streamer cycles from:

```text
no padding: S0/S1/S2 = 47330 / 23484 / 11331
with padding: S0/S1/S2 = 46613 / 23606 / 11342
```

The padded version is now the correct app for the requested input/output-compatible memory format.
