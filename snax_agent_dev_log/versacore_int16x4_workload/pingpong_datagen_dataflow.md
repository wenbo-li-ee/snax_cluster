# Dual VersaCore Int16x4 Chained SwiGLU: Pingpong Mechanism & Datagen Dataflow

## 1. Overview

完整的 chained SwiGLU pipeline 由两个阶段组成:

```
Mode 0 (SwiGLU gate):   D0 = Rescale(Swish(Rescale(A @ W^T)) * Rescale(A @ V^T))
Mode 1 (GEMM project):  D1 = Rescale(D0 @ W2^T)
```

Mode 1 的输入 A 直接读取 Mode 0 在 TCDM 中的输出 (zero-copy chaining),
不需要额外的 DMA 传输。

---

## 2. 维度体系

### 2.1 Tile 维度

硬件的基本计算单元是一个 tile-level GEMM:

```
A_tile[meshRow, tileSize] @ B_tile[meshCol, tileSize]^T = D_tile[meshRow, meshCol]
```

其中 `(meshRow, tileSize, meshCol)` 由 `array_shape` 和 `data_type` 决定:

| array_shape | meshRow | tileSize | meshCol |
|-------------|---------|----------|---------|
| 0           | 8       | 8        | 4       |
| 1           | 4       | 8        | 8       |
| 2           | 2       | 8        | 16      |

（data_type=0, int16x4）

### 2.2 Workload 维度 (M, K, N)

在 `params.hjson` 中定义的 M, K, N 是 **tile count**（不是 element count）:

- 矩阵 A: `[M, K]` tiles = `[M*meshRow, K*tileSize]` elements (int16)
- 矩阵 B (W/V): `[N, K]` tiles = `[N*meshCol, K*tileSize]` elements (int4, 存储为 packed)
  - 硬件执行 `A @ B^T`，B 以 `(N, K, meshCol, tileSize)` 存储
- 输出 D: `[M, N]` tiles = `[M*meshRow, N*meshCol]` elements (int16)

### 2.3 Mode 1 维度的自动计算

Mode 1 的输入是 Mode 0 的输出。Mode 0 输出形状为 `[M*meshRow, N*meshCol]` elements。
Mode 1 将其作为 A 矩阵读入，按 tile `[meshRow, tileSize]` 切分：

- **列方向**: 总列数 = `N * meshCol`，每 tile 宽 `tileSize`
  → `K1 = N * meshCol / tileSize`
- **行方向**: 总行数 = `M * meshRow`，每 tile 高 `meshRow`
  → `M1 = M`
- **对称 SwiGLU** (d_model → d_ff → d_model): W2 将 d_ff 映射回 d_model
  → `N1 = K * tileSize / meshCol`

验证: Mode 1 输出 `[M1*meshRow, N1*meshCol] = [M*meshRow, K*tileSize]`，
恰好是原始输入 x 的维度（对称投影）。

**各 shape 下的具体数值** (M=4, K=8, N=8):

| Shape | meshRow | tileSize | meshCol | K1=N*meshCol/tileSize | N1=K*tileSize/meshCol | Mode1 Output Elements |
|-------|---------|----------|---------|-----------------------|-----------------------|-----------------------|
| 0     | 8       | 8        | 4       | 4                     | 16                    | 4*8*16*4 = 2048       |
| 1     | 4       | 8        | 8       | 8                     | 8                     | 4*4*8*8 = 1024        |
| 2     | 2       | 8        | 16      | 16                    | 4                     | 4*2*4*16 = 512        |

---

## 3. Batch vs Pingpong

### 3.1 Batch 模式

Batch 模式一次性加载所有数据到 TCDM，然后一次性计算:

```
DMA: A, W(all N tiles), V(all N tiles) → TCDM
Compute: Mode 0 全部 M*N tiles → D0
DMA: W2_left(all N1 tiles), W2_right(all N1 tiles) → TCDM
Compute: Mode 1 全部 M1*N1 tiles → D1
```

**优点**: 简单，无需同步开销
**缺点**: 所有 B 矩阵必须同时在 TCDM 中，占用大量 SRAM

### 3.2 Pingpong (Double-Buffering) 模式

Pingpong 模式将 B 矩阵沿 N 维度切成 `num_chunks = N / N_chunk` 个 chunk。
每个 chunk 包含 `N_chunk` 个 N-tiles。用两组 buffer 交替加载和计算:

```
时间线 (Mode 0):
───────────────────────────────────────────────────
DMA core:   [load chunk0] [load chunk1] [load chunk2] ...
Compute:              [compute chunk0] [compute chunk1] ...
Buffer:        buf[0]        buf[1]        buf[0]      ...
───────────────────────────────────────────────────
```

具体流程:

```
1. 预加载: DMA 将 A 和第一个 B chunk (chunk 0) 加载到 buf[0]
2. 对每个 chunk c = 0, 1, ..., num_chunks-1:
   Compute core:
     a. 设置 streamer CSR (B base = buf[c%2], D base = d0 + c*d_chunk_bytes)
     b. 设置 accelerator CSR (K, N_chunk*M, array_shape, mode)
     c. barrier → 告知 DMA core 可以开始加载下一个 chunk
     d. 启动 streamer + accelerator
     e. 等待 accelerator 完成 (wait_dual_versacore)
     f. 等待 writer 完成 (wait_dual_versacore_writer)
     g. barrier → 等待 DMA 完成下一个 chunk 加载
   DMA core:
     a. barrier → 等待 compute core 准备好
     b. 如果 c+1 < num_chunks: 加载 chunk c+1 到 buf[(c+1)%2]
     c. barrier → 告知 compute core 下一个 chunk 已就绪
```

**优点**: B 矩阵的 TCDM 占用从 `N*K*b_tile_padded` 降低到 `2*N_chunk*K*b_tile_padded`
**缺点**: 额外的 barrier 同步和 CSR 重配置开销

### 3.3 Buffer 布局 (Pingpong)

TCDM 内存布局:

```
Mode 0 buffers:
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│    A     │ W_buf[0] │ W_buf[1] │ V_buf[0] │ V_buf[1] │   D0     │   D1     │
│(全部M*K) │(N_chunk  │(N_chunk  │(N_chunk  │(N_chunk  │ (full    │ (full    │
│          │ *K tiles)│ *K tiles)│ *K tiles)│ *K tiles)│  M*N)    │  M*N)    │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

Mode 1 buffers (接在 Mode 0 之后):
┌───────────┬───────────┬───────────┬───────────┬──────────┬──────────┐
│ W2l_buf[0]│ W2l_buf[1]│ W2r_buf[0]│ W2r_buf[1]│  M1_D0   │  M1_D1   │
│(N1_chunk  │(N1_chunk  │(N1_chunk  │(N1_chunk  │ (full    │ (full    │
│ *K1 tiles)│ *K1 tiles)│ *K1 tiles)│ *K1 tiles)│ M1*N1)   │ M1*N1)   │
└───────────┴───────────┴───────────┴───────────┴──────────┴──────────┘

注意: Mode 1 的 A 输入 = Mode 0 的 D0 输出 (zero-copy, 同一块内存)
```

---

## 4. Datagen Dataflow 详解

### 4.1 输入参数

从 `params.hjson` 读取:
- `M, K, N`: workload tile counts
- `N_chunk, N1_chunk`: pingpong chunk size (仅 pingpong 模式)
- `array_shape, data_type`: 选择 spatial unrolling

从 `hw config .hjson` 读取:
- `meshRow, tileSize, meshCol`: 由 array_shape 和 data_type 确定
- `granularity_a/b/c_d`: streamer 对齐粒度
- `bankWidth = 64`: TCDM bank 宽度 (bits)

### 4.2 B Tile Padding

每个 B tile 包含 `meshCol * tileSize` 个 int4 值 = `meshCol * tileSize / 2` bytes。
但 streamer 按 channel 读取，每个 channel = `bankWidth/8 = 8` bytes。
为避免 TCDM deadlock，每个 tile 需要 pad 到 channel footprint 的整数倍:

```python
b_tile_raw    = meshCol * tileSize * 4 / 8   # 实际数据大小 (bytes)
channel_bits  = ceil(b_tile_raw / 8) * 8      # 向上对齐到 8 channels
b_tile_padded = max(b_tile_raw, channel_bits * 8)  # pad 后大小
```

对所有 shape:
- Shape 0: meshCol=4, raw=16B, padded=64B (pad to 8 channels * 8B)
- Shape 1: meshCol=8, raw=32B, padded=64B
- Shape 2: meshCol=16, raw=64B, padded=64B (无需 pad)

### 4.3 Streamer Loop Nest

硬件 streamer 使用一个多层嵌套循环来遍历 tile:

**A reader** (6 层 temporal loop):
```
for m in range(Atlbound2):           # M tiles (stride = K * a_tile_bytes)
  for n in range(Atlbound1):         # N_chunk tiles (stride = 0, A 被广播)
    for k in range(Atlbound0):       # K tiles (stride = a_tile_bytes)
      读取 A[m,k] tile 送入 accelerator
```

**B0/B1 reader** (4 层 temporal loop):
```
for m in range(B0tlbound2):          # M tiles (stride = 0, B 对所有 M 行重复)
  for n in range(B0tlbound1):        # N_chunk tiles (stride = K * b_tile_padded)
    for k in range(B0tlbound0):      # K tiles (stride = b_tile_padded)
      读取 B[n,k] tile 送入 accelerator
```

**D0/D1 writer** (4 层 temporal loop):
```
for m in range(Dtlbound2):           # M tiles (stride = Dtlstride2)
  for n in range(Dtlbound1):         # N_chunk tiles (stride = Dtlstride1)
    for s in range(Dtlbound0):       # spatial (stride = Dtlstride0)
      写出 D[m,n] tile
```

### 4.4 D Writer Stride 的关键设计 (Pingpong 的核心难点)

**问题**: 在 pingpong 模式下，每个 chunk 只处理 `N_chunk` 个 N-tile。但所有 chunk 的输出
必须拼成和 batch 模式一样的 row-major 布局:

```
Batch 输出布局 (M=4, N=8):
Row 0: [tile(0,0), tile(0,1), ..., tile(0,7)]   ← N=8 tiles, 每个 meshCol*meshRow*2 bytes
Row 1: [tile(1,0), tile(1,1), ..., tile(1,7)]
Row 2: [tile(2,0), tile(2,1), ..., tile(2,7)]
Row 3: [tile(3,0), tile(3,1), ..., tile(3,7)]
```

**Pingpong chunk c 的输出**只包含 N_chunk=1 个 tile per row。
如果 Dtlstride2 (行间步长) = N_chunk * tile_size，输出会变成:

```
错误的布局 (连续存储):
[Row0_N=c, Row1_N=c, Row2_N=c, Row3_N=c]  ← 列优先!
```

**正确做法**: Dtlstride2 必须等于完整的行宽 `N * tile_element_bytes`:

```python
# 每个 output tile = meshRow * meshCol 个 int16 = meshRow * meshCol * 2 bytes
tile_d_bytes = meshRow * meshCol * out_elem_bits // 8  # = 64 bytes (shape 0)

# 行间步长 = 完整行宽 (所有 N tiles)
Dtlstride2 = N * tile_d_bytes   # = 8 * 64 = 512 bytes

# 每个 chunk 的 D base 偏移 = 只偏移一个 tile 列宽
d_chunk_bytes = N_chunk * tile_d_bytes  # = 1 * 64 = 64 bytes
```

C 代码中:
```c
d0_base = delta_local_d0 + c * d_chunk_bytes;
// chunk 0 → offset 0, chunk 1 → offset 64, ..., chunk 7 → offset 448
```

这样每个 chunk 的 D writer 会在正确的位置写入:
```
Chunk 0: 写 (Row0, col 0), (Row1, col 0), (Row2, col 0), (Row3, col 0)
         地址: base+0, base+512, base+1024, base+1536
Chunk 1: 写 (Row0, col 1), (Row1, col 1), (Row2, col 1), (Row3, col 1)
         地址: base+64, base+576, base+1088, base+1600
...
```

**注意**: 对于 M=1 (只有一行)，无论 Dtlstride2 是什么值都不影响结果，
因为 Dtlbound2=1 时 stride 不会被使用。这就是为什么之前 M=1 的 pingpong 能 PASS
而 M=4 的 pingpong FAIL 的原因。

### 4.5 Golden Model

```python
# Mode 0 Golden:
vc0_int32 = block_gemm_int16x4(M, K, N, A, W)   # D = A @ W^T
vc1_int32 = block_gemm_int16x4(M, K, N, A, V)   # D = A @ V^T
vc0_int16 = rescale(vc0_int32)                    # 32→16 bit
vc0_silu  = vc0_int16 >> 2                        # 近似 Swish(x) ≈ x/4
vc1_int16 = rescale(vc1_int32)
mul_int32 = vc0_silu * vc1_int16                  # Hadamard product
mode0_out = rescale(mul_int32)                     # 最终 SwiGLU gate 输出

# Mode 1 Golden (chained):
mode1_A_flat = mode0_out.reshape(-1)              # 直接使用 Mode 0 输出, 不做任何 DMA
golden_d0 = rescale(block_gemm_int16x4(M1, K1, N1, mode1_A_flat, W2_left))
golden_d1 = rescale(block_gemm_int16x4(M1, K1, N1, mode1_A_flat, W2_right))
```

block_gemm_int16x4 的核心:
```python
def block_gemm_int16x4(M, K, N, A_flat, B_flat):
    a = A_flat.reshape(M, K, meshRow, tileSize) - sub_a
    b = B_flat.reshape(N, K, meshCol, tileSize) - sub_b
    d = zeros(M, N, meshRow, meshCol)
    for m, n:
        d[m,n] = tensordot(a[m], b[n], axes=([K,tileSize], [K,tileSize]))
    # 等价于: D = A @ B^T (在 tile 级别)
    return d.reshape(-1)
```

---

## 5. C 代码 Dataflow (Pingpong)

### 5.1 Mode 0 双核协同

```
                    Compute Core                          DMA Core
                    ============                          ========
                                                     load A → local_a
                                                     load W_chunk0 → buf[0]
                                                     load V_chunk0 → buf[0]
                    ─── barrier ───────────────────── barrier ───
Chunk 0:
  setup CSR (B=buf[0], D=d0+0)                       (idle)
  ─── barrier ──────────────────────────────── barrier ──
  start streamer + accel                              load W_chunk1 → buf[1]
  wait accel done                                     load V_chunk1 → buf[1]
  wait writer done
  ─── barrier ──────────────────────────────── barrier ──
Chunk 1:
  setup CSR (B=buf[1], D=d0+64)                       (idle)
  ─── barrier ──────────────────────────────── barrier ──
  start streamer + accel                              load W_chunk2 → buf[0]
  wait accel done
  wait writer done
  ─── barrier ──────────────────────────────── barrier ──
...
```

### 5.2 Mode 1 双核协同

Mode 1 结构与 Mode 0 相同，但:
- A 输入 = `delta_local_d0` (Mode 0 的 D0 输出, zero-copy)
- B 为 W2_left / W2_right (用 w2l_buf / w2r_buf 双缓冲)
- num_chunks1 = N1 / N1_chunk

---

## 6. 实验结果 (All 12 PASS)

### M=1, K=8, N=8

| Shape | Mode |  Batch accel | Batch streamer | Pingpong accel | Pingpong streamer | Speedup (accel) |
|-------|------|-------------|----------------|----------------|-------------------|-----------------|
| 0     | M0   | 133         | 146            | 21             | 34                | 6.3x            |
| 0     | M1   | 133         | 138            | 12             | 17                | 11.1x           |
| 1     | M0   | 133         | 146            | 21             | 34                | 6.3x            |
| 1     | M1   | 133         | 138            | 21             | 26                | 6.3x            |
| 2     | M0   | 132         | 145            | 20             | 33                | 6.6x            |
| 2     | M1   | 133         | 137            | 37             | 41                | 3.6x            |

### M=4, K=8, N=8

| Shape | Mode |  Batch accel | Batch streamer | Pingpong accel | Pingpong streamer | Speedup (accel) |
|-------|------|-------------|----------------|----------------|-------------------|-----------------|
| 0     | M0   | 513         | 539            | 69             | 82                | 7.4x            |
| 0     | M1   | 513         | 531            | 36             | 41                | 14.3x           |
| 1     | M0   | 513         | 539            | 69             | 82                | 7.4x            |
| 1     | M1   | 513         | 531            | 69             | 74                | 7.4x            |
| 2     | M0   | 517         | 530            | 69             | 82                | 7.5x            |
| 2     | M1   | 517         | 522            | 133            | 138               | 3.9x            |

**说明**:
- Pingpong 模式的 cycle 是 **last chunk 的 cycle** (perf counter 只报告最后一次计算)
- Batch 的 cycle 是整个计算的 cycle
- 因此直接比较 batch total vs pingpong last-chunk 不完全公平，
  但它展示了 pingpong 下单个 chunk 的计算确实更快（更小的 tile count per invocation）
- Shape 2 的 Mode 1 speedup 较低，因为 K1=16 (更多 K tiles per chunk)

---

## 7. 文件结构

```
sw/apps/
├── snax-versacore-int16x4-swiglu-m1-batch/
│   ├── data/
│   │   ├── datagen.py       # 数据生成, 无 N_chunk
│   │   ├── params.hjson     # M=1, K=8, N=8
│   │   ├── data.h           # (生成的)
│   │   └── Makefile
│   ├── src/
│   │   └── *.c              # 一次性加载, 一次性计算
│   └── Makefile
├── snax-versacore-int16x4-swiglu-m1-pingpong/
│   ├── data/
│   │   ├── datagen.py       # 数据生成, 有 N_chunk/N1_chunk
│   │   ├── params.hjson     # M=1, K=8, N=8, N_chunk=1, N1_chunk=1
│   │   └── ...
│   ├── src/
│   │   └── *.c              # 双缓冲循环
│   └── Makefile
├── snax-versacore-int16x4-swiglu-m4-batch/    # 同 m1 但 M=4
└── snax-versacore-int16x4-swiglu-m4-pingpong/ # 同 m1-pp 但 M=4
```

---

## 8. Bug Fix 记录

### D Writer Stride Bug (M>1 Pingpong)

**症状**: M=4 pingpong FAIL, M=1 pingpong PASS, 所有 batch PASS

**根因**: `Dtlstride2` (D writer 的 M 维度步长) 被设为 `N_chunk * tile_d_bytes`
而不是 `N * tile_d_bytes`。对于 M>1, 这导致 chunk 输出按列优先而非行优先排列，
与 golden model 的行优先布局不匹配。

**修复** (datagen.py):
```python
# Before (错误):
Dtlstride2 = N_chunk * out_elem_bits * meshRow * meshCol // 8
d_chunk_bytes = M * N_chunk * meshRow * meshCol * out_elem_bits // 8

# After (正确):
Dtlstride2 = N * out_elem_bits * meshRow * meshCol // 8
d_chunk_bytes = N_chunk * meshRow * meshCol * out_elem_bits // 8
```

同样的修复应用于 Mode 1 的 `M1_Dtlstride2` 和 `m1_d_chunk_bytes`。
