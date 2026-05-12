# Review: small MoE K8 L15 weights-first app

**Date:** 2026-05-12  
**Config:** `snax_cluster/target/snitch_cluster/cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson`  
**New app:** `snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15`  
**Base app copied from:** `snax-versacore-int16x4-multishape-k8-8x4-mode1-contiguous-l15`  
**Primary audience:** Hemaia agent implementing or generating full MoE-layer data.

## 1. Purpose

This app is a smaller, faster-running version of the previous L15 contiguous app, shaped like one small MoE expert path:

```text
A tokens          : 8 x 1024 int16
Mode0 W/V         : 1024 x 128 int4 each
Mode0 output      : 8 x 128 int16
Mode1 W2_left     : 128 x 512 int4
Mode1 W2_right    : 128 x 512 int4
Mode1 combined out: 8 x 1024 int16
```

It keeps the same hardware config and shape family:

| Shape | array_shape | meshRow | tileSize | meshCol |
|---|---:|---:|---:|---:|
| S0 | 0 | 8 | 8 | 4 |
| S1 | 1 | 4 | 8 | 8 |
| S2 | 2 | 2 | 8 | 16 |

The two functional changes are:

1. **Weights-first TCDM placement.** Expert weights are placed at the beginning of TCDM, and the token buffer is placed after all weights. This matches a MoE setting where expert weight storage is deterministic while the number or routing of tokens can vary.
2. **Smaller dimensions.** The old `K0=2048, N0=1408, K1=1408, N1=1024` workload is reduced to `K0=1024, N0=128, K1=128, N1=512`, so simulation is much faster.

The L15 padding/coloring recipe is kept:

```text
A pad     : 32 bytes
B1 color  : 272 bytes = bank34
W2L color : 128 bytes = bank16
M1D0 color: 256 bytes = bank32
```

## 2. Files Added or Changed

New app directory:

```text
snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15/
```

Important files:

```text
Makefile
data/Makefile
data/params.hjson
data/datagen.py
src/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15.c
```

The top-level app Makefile was also updated:

```text
snax_cluster/target/snitch_cluster/sw/apps/Makefile
```

The new app is now included under:

```make
ifeq ($(CFG_OVERRIDE), cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson)
```

## 3. Datagen Changes

### 3.1 Size Parameters

`data/params.hjson` now contains:

```hjson
{
    M_total: 8
    K0_total: 1024
    N0_total: 128
    K1_total: 128
    N1_total: 512
}
```

`datagen.py` still asserts these exact values. This app is intentionally a fixed validation workload, not a fully generic matrix-size generator.

### 3.2 Weights-First Placement

The previous app placed A first. This app places all weights first:

```text
B0/W -> B1/V -> W2_left -> W2_right -> A tokens -> Mode0 D0 -> Mode1 D0/D1
```

Actual generated offsets from `data.h`:

| Tensor | Offset bytes | Start bank | Role |
|---|---:|---:|---|
| B0/W | 0 | 0 | Mode0 first expert weight |
| B1/V | 65808 | 34 | Mode0 second expert weight |
| W2_left | 132224 | 16 | Mode1 left down-projection half |
| W2_right | 165888 | 0 | Mode1 right down-projection half |
| A | 198656 | 0 | padded token buffer, after all weights |
| Mode0 D0 | 216064 | 0 | Mode0 output, Mode1 A source |
| Mode1 D0 | 218368 | 32 | Mode1 left output |
| Mode1 D1 | 219392 | 32 | Mode1 right output |
| End | 235008 | 0 | total TCDM footprint |

This is the key MoE-relevant structural change. A future full-MoE generator can allocate a deterministic expert-weight prefix and then place routed token blocks after that prefix.

### 3.3 Tensor Sizes

With the small dimensions:

| Tensor | Size |
|---|---:|
| W/V each | 65536 bytes |
| W2_left/W2_right each | 32768 bytes |
| A padded | 16640 bytes |
| Mode0 D0 | 2048 bytes |
| Mode1 padded output area | 16640 bytes |
| Total used | 235008 bytes |

The app therefore fits very comfortably in the 8 MiB TCDM.

### 3.4 A Padding

The A row is now:

```text
K0_total * 2 = 1024 * 2 = 2048 bytes
A pad         = 32 bytes
A row stride  = 2080 bytes
row_elems     = 1040 int16
```

Each token row contains 1024 live int16 values followed by 16 zero padding int16 values. The bank-phase idea is unchanged from L15: +32 B means each successive token row starts four TCDM banks later.

### 3.5 B Reader Stride Fix

The old app had B spatial strides matching large K:

```text
Mode0 B sstride[1] = 4096
Mode1 B sstride[1] = 2816
```

Those would be wrong for the small matrices. The new app derives them from tile counts:

```python
mode0_b_sstride = k0_s0_tiles * 16  # 128 * 16 = 2048
mode1_b_sstride = k1_s0_tiles * 16  # 16 * 16 = 256
```

Generated CSR fields are:

```text
mode0_B_sstride = { 8, 2048 }
mode1_B_sstride = { 8, 256 }
```

Without this change, the app would either read wrong W/V/W2 locations or depend on accidental zero-filled gaps.

### 3.6 Mode1 Output Row

Mode1 now writes two 512-wide halves:

```text
per token row = [D0/left 512 int16][D1/right 512 int16][16 int16 padding]
row_elems     = 1040
row_bytes     = 2080
```

Writer1 starts at:

```text
delta_local_mode1_d1 = delta_local_mode1_d0 + N1_total * 2
                     = delta_local_mode1_d0 + 1024 bytes
```

So D0 and D1 are adjacent halves of the same padded token row.

## 4. Runtime Shape Parameters

The generated `shape_cfg_t` values are:

| Shape | K_tiles | N_tiles | K1 | N1 | Mode0 B sstride | Mode1 B sstride |
|---|---:|---:|---:|---:|---|---|
| S0 | 128 | 32 | 16 | 128 | `{8,2048}` | `{8,256}` |
| S1 | 128 | 16 | 16 | 64 | `{8,2048}` | `{8,256}` |
| S2 | 128 | 8 | 16 | 32 | `{8,2048}` | `{8,256}` |

The Mode1 D writer still uses the same pattern:

```python
mode1_D_tbound  = [meshCol/4, meshRow, N1_tiles, 1]
mode1_D_tstride = [8, 2080, meshCol*2, 0]
```

Per shape:

| Shape | mode1_D_tbound | mode1_D_tstride |
|---|---|---|
| S0 | `[1, 8, 128, 1]` | `[8, 2080, 8, 0]` |
| S1 | `[2, 4, 64, 1]` | `[8, 2080, 16, 0]` |
| S2 | `[4, 2, 32, 1]` | `[8, 2080, 32, 0]` |

## 5. Simulation Results

The app was built in `barnard3` and run on the existing VLT binary. No RTL generation or hardware rebuild was run.

```text
total checks: 6
total error : 0
```

Cycle table:

| Shape | Mode | Correctness | Accelerator cycles | Streamer cycles | Wall cycles |
|---|---|---|---:|---:|---:|
| S0 | Mode0 | PASS | 4102 | 4126 | 42311 |
| S0 | Mode1 | PASS | 2180 | 2199 | 235108 |
| S1 | Mode0 | PASS | 2058 | 2082 | 28702 |
| S1 | Mode1 | PASS | 1110 | 1129 | 126607 |
| S2 | Mode0 | PASS | 1037 | 1061 | 22794 |
| S2 | Mode1 | PASS | 544 | 563 | 72594 |

The C program has been refactored to always run `layout_cfgs[0]`, all three shapes, and both modes. It no longer depends on `SELECT_LAYOUT`, `SELECT_SHAPE`, or `RUN_MODE1` for this normal run.

The cycle report is in:

```text
review_log/snax_versacore_moe_small_k8_l15_20260512/cycles.md
```

## 6. Validation Scope

Validated:

- `data.h` generation with the small dimensions.
- Weights-first TCDM placement.
- Derived Mode0/Mode1 B spatial strides.
- All S0/S1/S2 Mode0 outputs against golden.
- All S0/S1/S2 Mode1 padded contiguous outputs against golden.
- Single-ELF simulation on `target/snitch_cluster/bin/snitch_cluster.vlt`.

Not changed:

- Hardware config.
- Streamer hardware.
- Dual VersaCore library.
- SiLU golden model.
- Rescale parameters.

## 7. Notes for Hemaia Agent

This app is a useful small correctness and cycle probe, but it is still not a full MoE-layer generator.

For full MoE data generation, the likely next abstraction is:

1. Allocate expert weights in a deterministic prefix:

```text
expert0 W/V/W2_left/W2_right
expert1 W/V/W2_left/W2_right
...
```

2. Allocate routed token blocks after all expert weights.
3. For each expert invocation, select that expert's weight offsets and that expert's token block offset.
4. Keep the same streamer shape formulas for a fixed per-expert hidden/intermediate size.
5. Recompute or at least verify bank coloring if expert count, expert size, or token block placement changes.

The important lesson from this app is that putting A after weights is functionally safe as long as every `delta_local_*` field is regenerated consistently. The streamer only sees base pointers plus strides; it does not require A to appear before weights in TCDM.

The performance coloring is inherited from the L15 sweep. It is correct and passes for this small case, but it was not re-swept for the new size. If performance matters for the final full MoE workload, resweep at least:

```text
A pad
B1 bank
W2_left bank
Mode1 D0 bank
```
