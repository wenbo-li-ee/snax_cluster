# Detailed Report: Small MoE K8 L15 App

**Date:** 2026-05-12  
**Config:** `snax_cluster/target/snitch_cluster/cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson`  
**App:** `snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15`  
**Reference layout study:** `review_log/layout_explore_k8_4lane_20260506_review/review_en.md`

This app is a smaller MoE-style version of the previous L15 multishape app. It keeps the same padding/coloring idea, but changes the TCDM placement to put all fixed expert weights before the token buffer.

The current problem size is:

```text
M_total  = 8
K0_total = 1024
N0_total = 128
K1_total = 128
N1_total = 512
```

Mode0 computes the SwiGLU gate/up part with two `1024 x 128` int4 weight matrices. Mode1 computes the down projection as two `128 x 512` matrices. The two Mode1 outputs are written as `[left 512, right 512, padding]` per token row.

## 1. Source Files

- `data/params.hjson`: matrix sizes.
- `data/datagen.py`: emits `data.h`, including tensors, golden outputs, TCDM offsets, and streamer CSR fields.
- `src/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15.c`: DMA staging, streamer/core configuration, execution, wait loops, and result checking.
- `target/snitch_cluster/sw/snax/dual-versacore-swiglu`: the software CSR helper library.
- `util/silu_pkg/silu_out16_balanced_golden.py`: bit-true SiLU golden model used by datagen.

## 2. Streamer Hardware Configuration

The relevant hardware template is `snax_dual_versacore_int16x4_streamer_template`.

The accelerator has 34 TCDM ports:

```text
A reader  : 16 channels
B0 reader :  8 channels
B1 reader :  8 channels
D0 writer :  1 channel
D1 writer :  1 channel
total     : 34
```

The sparse interconnect config is:

```hjson
sparse_interconnect_config: [
  [16, 1],
  [8,  1],
  [8,  1],
  [1,  1],
  [1,  1],
]
```

The second field is `1` for every port group. This matters because the app uses multidimensional streamer strides and bank-colored base addresses. With a coarser access granularity, some generated addresses could become unroutable through the sparse interconnect.

Reader parameters:

```text
A  spatial bounds: [2, 8], channels: 16, temporal dim: 6
B0 spatial bounds: [2, 4], channels:  8, temporal dim: 4
B1 spatial bounds: [2, 4], channels:  8, temporal dim: 4
```

Writer parameters:

```text
D0 spatial bounds: [1], channels: 1, temporal dim: 4
D1 spatial bounds: [1], channels: 1, temporal dim: 4
```

All readers and writers keep `tcdm_logic_word_size = [256, 128, 64]`. The app uses remap index 0, so the 256-bit entry must remain present.

The shape family is:

| Shape | array_shape | meshRow | tileSize | meshCol |
|---|---:|---:|---:|---:|
| S0 | 0 | 8 | 8 | 4 |
| S1 | 1 | 4 | 8 | 8 |
| S2 | 2 | 2 | 8 | 16 |

All three shapes use 256 MACs, but distribute them differently across M and N.

## 3. SiLU Golden Source

`datagen.py` imports:

```python
from silu_out16_balanced_golden import silu_out16_balanced_eval_q
```

The preferred path is:

```text
snax_cluster/util/silu_pkg/silu_out16_balanced_golden.py
```

If that directory is not present, the script falls back to:

```text
/esat/studscratch/r1015498/Thesis/original_snax/silu/pkg
```

`apply_silu_vectorized()` calls `silu_out16_balanced_eval_q()` element by element. This is not a floating-point SiLU approximation for convenience; it is the bit-true fixed-point golden model matching the hardware SiLU block used by the dual-VersaCore SwiGLU path.

Mode0 golden is:

```text
vc0      = A * W
vc1      = A * V
vc0_i16  = rescale(vc0)
vc0_silu = silu_out16_balanced(vc0_i16)
vc1_i16  = rescale(vc1)
mode0    = rescale(vc0_silu * vc1_i16)
```

The current rescale settings are identity.

## 4. Datagen Flow

`data/datagen.py` emits a complete C header. Its main work is:

1. Read matrix sizes from `params.hjson`.
2. Generate logical A.
3. Generate padded physical A.
4. Generate packed int4 weights: `W`, `V`, `W2_left`, `W2_right`.
5. Generate Mode0 and Mode1 golden arrays by emulating streamer access.
6. Compute TCDM offsets.
7. Emit per-shape streamer CSR fields.
8. Emit `layout_cfg_t layout_cfgs[]`.

The only active layout is:

```python
{"id": 15, "name": "l15_weights_first_padded_1024_per_token",
 "a_pad": 32, "b1_color": 272, "w2l_color": 128, "m1d0_color": 256}
```

## 5. A Padding

Logical A is generated as an `8 x 1024` int16 matrix:

```python
data[m, k] = ((m * 5 + k * 3) % 11) - 5
```

Each live token row is:

```text
1024 int16 = 2048 bytes
```

L15 adds 32 bytes of padding:

```text
A row stride = 2048 + 32 = 2080 bytes
row elements  = 2080 / 2 = 1040 int16
live elements = 1024 int16
padding       = 16 int16
```

Implementation:

```python
def make_padded_a(logical_a, row_stride_bytes):
    row_elems = row_stride_bytes // 2
    out = np.zeros((logical_a.shape[0], row_elems), dtype=np.int16)
    out[:, :logical_a.shape[1]] = logical_a
    return out.reshape(-1)
```

The padding exists to rotate the TCDM bank phase across token rows. With 64 banks and 8-byte bank words:

```text
2048 / 8 = 256 = 4 * 64       => no row-to-row bank rotation
2080 / 8 = 260 = 4 * 64 + 4   => each token row starts 4 banks later
```

This follows the 2026-05-06 layout exploration result: A row padding is the primary fix for token-row bank conflicts.

## 6. Weights-First TCDM Placement

The placement is computed in `place_tensors()`. The order is:

```text
B0 / W
B1 / V
W2_left
W2_right
A token buffer
Mode0 D0
Mode1 D0 left output
Mode1 D1 right output
```

This is the key MoE-oriented change. Expert weights are fixed-size; routed token counts are variable. Keeping weights in a deterministic prefix makes it easier to extend this into a full MoE layer data generator.

Offsets use:

```python
def colored_offset(offset, color_bytes=0, alignment=1024):
    return align_up(offset, alignment) + int(color_bytes)
```

Since `1024 / 8 = 128 = 2 * 64`, 1024-byte alignment preserves bank phase. The extra color bytes determine the bank.

For the current small app:

| Tensor | Offset | Bank | Notes |
|---|---:|---:|---|
| B0/W | 0 | 0 | first Mode0 weight |
| B1/V | 65808 | 34 | `b1_color = 272` |
| W2_left | 132224 | 16 | `w2l_color = 128` |
| W2_right | 165888 | 0 | aligned, no extra color |
| A | 198656 | 0 | after all weights |
| Mode0 D0 | 216064 | 0 | after A |
| Mode1 D0 | 218368 | 32 | `m1d0_color = 256` |
| Mode1 D1 | 219392 | 32 | D0 + 1024 bytes |
| TCDM end | 235008 | - | total footprint |

## 7. Weight Coloring

Mode0 uses:

```text
B0/W: bank 0
B1/V: bank 34
```

`b1_color = 272` gives `272 / 8 = 34` bank words. This comes from the 2026-05-06 layout exploration. After A padding fixed the dominant A-row conflict, B1 bank coloring fixed the remaining B0/B1 phase conflict, especially for S2. Bank 34 was the earliest clean combined-best phase for S0/S1/S2 in that study.

Mode1 uses:

```text
W2_left : bank 16
W2_right: bank 0
Mode1 D0/D1 output: bank 32
```

This comes from:

```python
w2l_color = 128   # bank 16
m1d0_color = 256  # bank 32
```

Mode1 writes two output streams into the same padded token-row layout. D1 starts `N1_total * 2 = 1024` bytes after D0, so it writes the right half of the row.

## 8. Per-Shape Streamer CSR Generation

The CSR fields are emitted by `build_shape_cfg()`.

Channel enables:

| Shape | A channels | A mask | B channels | B mask |
|---|---:|---:|---:|---:|
| S0 | 16 | `0xFFFF` | 2 | `0x03` |
| S1 | 8 | `0x00FF` | 4 | `0x0F` |
| S2 | 4 | `0x000F` | 8 | `0xFF` |

Mode0 A:

```python
mode0_A_sstride = [8, a_row_stride]          # [8, 2080]
mode0_A_tbound  = [K_tiles, N_tiles, 1, 1, 1, 1]
mode0_A_tstride = [tileSize * 2, 0, meshRow * a_row_stride, 0, 0, 0]
```

Mode0 B:

```python
mode0_B_sstride = [8, k0_s0_tiles * 16]      # [8, 2048]
mode0_B_tbound  = [K_tiles, N_tiles, 1, 1]
mode0_B_tstride = [16, (meshCol // 4) * 2048, 0, 0]
```

Mode0 D:

```python
mode0_D_tbound  = [8, N_tiles, 1, 1]
mode0_D_tstride = [8, 64, N_tiles * 64, 0]
```

Mode1 A reads Mode0 D0:

```python
mode1_A_sstride = {
  S0: [64, 8],
  S1: [8, 16],
  S2: [8, 32],
}
mode1_A_tstride[0] = {
  S0: 128,
  S1: 64,
  S2: 16,
}
```

Mode1 B:

```python
mode1_B_sstride = [8, k1_s0_tiles * 16]      # [8, 256]
mode1_B_tbound  = [K1, N1, 1, 1]
mode1_B_tstride = [16, (meshCol // 4) * 256, 0, 0]
```

Mode1 D:

```python
beats_per_row = meshCol // 4
mode1_D_tbound  = [beats_per_row, meshRow, N1, 1]
mode1_D_tstride = [8, a_row_stride, meshCol * 2, 0]
```

The important point is that Mode1 D uses `a_row_stride = 2080`, so its output has the same per-token padded layout as A.

## 9. Golden Layout Model

`streamer_i16_flat()` emulates streamer reads. It builds spatial offsets from bounds/strides, applies `channel_en`, then walks temporal K/M loops and reads 4 int16 values per spatial offset.

Mode0 golden uses logical A with logical row stride `k0_bytes = 2048`, while hardware reads physical padded A with `a_row_stride = 2080`. This is intentional: padding is never part of the valid K range, so logical and physical reads produce the same mathematical input.

Mode1 golden converts natural accelerator output order into per-token output order:

```python
mode1_d0.reshape(M_tiles, N_tiles, meshRow, meshCol)
        .transpose(0, 2, 1, 3)
```

Then it concatenates left and right outputs and writes them into a zero-initialized padded row. The final 16 int16 padding elements per row remain zero.

## 10. L3-to-TCDM DMA Staging

C staging is in `stage_layout_to_tcdm()`.

Destination pointers are built from `delta_local_*`:

```c
local_a   = snrt_l1_next() + delta_local_a;
local_b0  = snrt_l1_next() + delta_local_b0;
local_b1  = snrt_l1_next() + delta_local_b1;
local_w2l = snrt_l1_next() + delta_local_w2l;
local_w2r = snrt_l1_next() + delta_local_w2r;
```

The DM core issues five 1D DMA transfers:

```c
snrt_dma_start_1d(local_a,   layout->a_data,        layout->a_data_length);
snrt_dma_start_1d(local_b0,  layout->w_data,        layout->b_data_length);
snrt_dma_start_1d(local_b1,  layout->v_data,        layout->b_data_length);
snrt_dma_start_1d(local_w2l, layout->w2_left_data,  layout->w2_data_length);
snrt_dma_start_1d(local_w2r, layout->w2_right_data, layout->w2_data_length);
snrt_dma_wait_all();
```

The DMA call order is not the TCDM layout order. The destination offsets define the layout. D0 and Mode1 outputs are not staged from L3; they are written by the accelerator writers.

## 11. C Program Flow

The C program has been refactored for readability. It no longer uses `SELECT_LAYOUT`, `SELECT_SHAPE`, or `RUN_MODE1` to choose the normal path. It always uses `layout_cfgs[0]`, runs all three generated shapes, and runs Mode0 followed by Mode1 for every shape.

`main()`:

1. Print app metadata on core0.
2. Select `layout_cfgs[0]`.
3. Call `stage_layout_to_tcdm()`.
4. Non-core0 harts exit after the barrier.
5. Core0 runs every shape in `layout->shapes[]`.
6. Print `NUM_SHAPES * 2` checks and total errors.

`run_shape()`:

1. Build the subtraction setting.
2. Call `run_mode0()`.
3. Stop early if Mode0 failed.
4. Call `run_mode1()`.

`run_mode0()`:

1. Configure three readers and D0 writer with `set_dual_versacore_streamer_csr_d0_only()`.
2. Configure core with `K_tiles`, `N_tiles * M_tiles`, array shape, and data type.
3. Set mode 0 and identity rescale settings.
4. Start streamer and accelerator.
5. Wait for accelerator busy and streamer busy to clear.
6. Compare D0 against `mode0_d0_golden`.

`run_mode1()`:

1. Zero the padded Mode1 output region.
2. Configure three readers and two writers with `set_dual_versacore_streamer_csr()`.
3. Mode1 A base is `delta_local_d0`.
4. B0/B1 bases are `delta_local_w2l` and `delta_local_w2r`.
5. D0/D1 bases are `delta_local_mode1_d0` and `delta_local_mode1_d1`.
6. Configure core with `K1`, `N1 * M_tiles`, array shape, and data type.
7. Set mode 1.
8. Start, wait, and compare padded output against `mode1_padded_golden`.

The wait helpers include timeout diagnostics to avoid silent simulation hangs.

## 12. Current Simulation Result

From `cycles.md`:

| Shape | Mode | Correctness | Accelerator cycles | Streamer cycles | Wall cycles |
|---|---|---|---:|---:|---:|
| S0 | Mode0 | PASS | 4102 | 4126 | 42311 |
| S0 | Mode1 | PASS | 2180 | 2199 | 235108 |
| S1 | Mode0 | PASS | 2058 | 2082 | 28702 |
| S1 | Mode1 | PASS | 1110 | 1129 | 126607 |
| S2 | Mode0 | PASS | 1037 | 1061 | 22794 |
| S2 | Mode1 | PASS | 544 | 563 | 72594 |

Final:

```text
total checks: 6
total error : 0
```

DMA staging took 3520 cycles in the latest refactored run. Wall cycles include software structure and progress-print effects; use accelerator and streamer counters for hardware-side comparisons.

## 13. Size Parameter Support

The current datagen is not generic yet. It has hard assertions:

```python
assert m_total == 8
assert k0_total == 1024
assert n0_total == 128
assert k1_total == 128
assert n1_total == 512
```

Many formulas are already parameterized, but the script intentionally only accepts this small MoE shape.

To support other K/N sizes, replace those assertions with constraints:

```python
assert m_total == 8
assert k0_total % 8 == 0
assert k1_total % 8 == 0
assert n0_total % 16 == 0
assert n1_total % 16 == 0
assert k1_total == n0_total
```

Also check that the two Mode1 halves fit into the padded token row:

```text
2 * N1_total <= (K0_total * 2 + a_pad) / 2
```

Changing `M_total` is more involved because the generated configs currently use `M_tiles = 1`. A real variable-token MoE generator should add per-expert token counts, padding to shape tile sizes, and per-expert invocation descriptors.

If the shape family, channel count, data type, or writer width changes, this becomes a cfg/RTL-generation change, not just a datagen change.

## 14. Hemaia Agent Notes

This app is not yet a full MoE-layer data generator. It is a validated single-expert, fixed-token-block executable test.

For a full MoE layer, the agent should extend it toward:

1. A fixed expert-weight prefix containing all experts' `W`, `V`, `W2_left`, and `W2_right`.
2. A variable token buffer after the weights.
3. Per-expert descriptors: token base, token count, padded token count, weight bases, output base, and selected shape.
4. Reuse of this app's padding/coloring and streamer CSR generation logic.
5. Separate routing/scatter-gather metadata for full-layer token movement.

The validated local rule is:

```text
weights first,
tokens after weights,
A and Mode1 output use 2080-byte padded token rows,
Mode0 uses A padding plus B1 bank34 coloring,
Mode1 uses W2_left bank16 and output bank32,
datagen owns physical layout, streamer CSRs, and bit-true golden,
C owns DMA staging, CSR programming, execution, and checks.
```
