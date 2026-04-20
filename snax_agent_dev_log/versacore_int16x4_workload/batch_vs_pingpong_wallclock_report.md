# Batch vs Pingpong Wall-Clock Cycle Comparison

## 实验配置

- Workload: Chained SwiGLU (Mode 0 → Mode 1, zero-copy)
- K=8, N=8, data_type=0 (int16x4)
- Pingpong: N_chunk=1, N1_chunk=1 (最细粒度, 每次只处理 1 个 N-tile)
- wall = `snrt_mcycle()` 测量的总 wall-clock cycles (包含 CSR 配置 + barrier 同步 + 计算 + 写回)
- accel/streamer = 硬件 perf counter (batch 报告整体, pingpong 仅报告最后一个 chunk)

---

## M=1 结果

| Shape | Mode | Batch wall | Pingpong wall | Pingpong / Batch | num_chunks |
|-------|------|-----------|---------------|------------------|------------|
| 0     | M0   | 2888      | 19421         | 6.72x slower     | 8          |
| 0     | M1   | 2587      | 37660         | 14.56x slower    | 16         |
| 1     | M0   | 2868      | 19421         | 6.77x slower     | 8          |
| 1     | M1   | 2587      | 19231         | 7.43x slower     | 8          |
| 2     | M0   | 2888      | 19423         | 6.72x slower     | 8          |
| 2     | M1   | 2590      | 10025         | 3.87x slower     | 4          |

## M=4 结果

| Shape | Mode | Batch wall | Pingpong wall | Pingpong / Batch | num_chunks |
|-------|------|-----------|---------------|------------------|------------|
| 0     | M0   | 3257      | 19559         | 6.00x slower     | 8          |
| 0     | M1   | 3006      | 37691         | 12.54x slower    | 16         |
| 1     | M0   | 3257      | 19559         | 6.00x slower     | 8          |
| 1     | M1   | 3006      | 19314         | 6.42x slower     | 8          |
| 2     | M0   | 3272      | 19496         | 5.96x slower     | 8          |
| 2     | M1   | 2996      | 10244         | 3.42x slower     | 4          |

---

## 分析

### 1. Pingpong 在当前工作量下显著慢于 Batch

所有实验中 pingpong 的 wall-clock 都比 batch **大 3.4x ~ 14.6x**。原因:

**每个 chunk 的固定开销**: 每次 chunk 迭代需要:
- 2 次 `snrt_cluster_hw_barrier()` (compute ↔ DMA 同步)
- 完整的 streamer CSR 配置 (~几十条 CSR 写指令)
- Accelerator CSR 配置 + mode/rescale 设置
- Streamer start + accelerator start 序列

以 M=1 Mode 0 为例:
- Batch: wall=2888 cycles, accel=133 → overhead=2888-146=2742 cycles (一次性 CSR + streamer + barrier)
- Pingpong: wall=19421, 8 chunks → per-chunk≈2428 cycles, 但 accel 只有 21 cycles/chunk
  → 每 chunk overhead≈2400 cycles, 有效计算仅占 ~0.9%

**结论**: 在 N=8 这个小规模下, 硬件计算时间 (accel ~20-130 cycles) 远小于软件配置开销 (~2400 cycles/chunk)。Pingpong 的 DMA 隐藏优势被巨大的 per-chunk overhead 淹没。

### 2. chunk 数越多, 开销越大

| num_chunks | 大致 wall-clock | 每 chunk 平均 |
|------------|----------------|--------------|
| 4          | ~10,000        | ~2,500       |
| 8          | ~19,400        | ~2,425       |
| 16         | ~37,700        | ~2,356       |

per-chunk overhead 非常稳定 (~2,400 cycles), 说明这是由 CSR 配置 + barrier 同步主导的固定成本。

### 3. 什么时候 Pingpong 会有优势?

Pingpong 的价值在于 **SRAM 节约** 而非速度:
- Batch 需要所有 B 矩阵同时在 TCDM: `2 × N × K × b_tile_padded` bytes
- Pingpong 只需: `2 × 2 × N_chunk × K × b_tile_padded` bytes (双缓冲 × 2 个 B 矩阵)

**例**: Shape 0, K=8:
- Batch B (W+V) 内存: 2 × 8 × 8 × 64 = 8192 bytes
- Pingpong B 内存: 2 × 2 × 1 × 8 × 64 = 2048 bytes (节省 75%)

当工作量 scale up (更大的 N, K), 同时 TCDM 容量有限时:
- Batch 可能无法放下所有 B 矩阵 → 必须使用 pingpong
- 计算时间增长 (accel cycles ∝ N*M*K) 而 per-chunk overhead 不变
  → overhead 占比下降, pingpong 的效率提升

**Pingpong 的 break-even point**: 当每 chunk 的 accel cycles ≫ per-chunk overhead (~2400 cycles) 时,
pingpong 的 wall-clock 接近 batch。这大约需要 N_chunk * M * K > 几百个 tile 操作。

### 4. Batch wall-clock 的组成

以 M=4 batch Mode 0 shape 0 为例:
- wall = 3257 cycles
- streamer = 530 cycles (硬件 streamer 活跃时间)
- accel = 517 cycles (硬件加速器活跃时间)
- overhead = 3257 - 530 = 2727 cycles (CSR 配置 + start + wait + barrier)

说明即使是 batch, CSR 配置的开销也占了总时间的 ~84%。这是当前小规模工作量下的根本瓶颈。

---

## 原始数据

### M=1 Batch
```
Shape 0: M0 accel=133, streamer=146, wall=2888  | M1 accel=133, streamer=138, wall=2587
Shape 1: M0 accel=133, streamer=146, wall=2868  | M1 accel=133, streamer=138, wall=2587
Shape 2: M0 accel=132, streamer=145, wall=2888  | M1 accel=133, streamer=137, wall=2590
```

### M=1 Pingpong (N_chunk=1, N1_chunk=1, num_chunks=8/8/8, num_chunks1=16/8/4)
```
Shape 0: M0 accel=21, streamer=34, wall=19421   | M1 accel=12, streamer=17, wall=37660
Shape 1: M0 accel=21, streamer=34, wall=19421   | M1 accel=21, streamer=26, wall=19231
Shape 2: M0 accel=20, streamer=33, wall=19423   | M1 accel=37, streamer=41, wall=10025
```

### M=4 Batch
```
Shape 0: M0 accel=517, streamer=530, wall=3257   | M1 accel=517, streamer=522, wall=3006
Shape 1: M0 accel=517, streamer=530, wall=3257   | M1 accel=517, streamer=522, wall=3006
Shape 2: M0 accel=517, streamer=530, wall=3272   | M1 accel=513, streamer=531, wall=2996
```

### M=4 Pingpong (N_chunk=1, N1_chunk=1, num_chunks=8/8/8, num_chunks1=16/8/4)
```
Shape 0: M0 accel=69, streamer=82, wall=19559    | M1 accel=35, streamer=41, wall=37691
Shape 1: M0 accel=69, streamer=82, wall=19559    | M1 accel=68, streamer=74, wall=19314
Shape 2: M0 accel=69, streamer=82, wall=19496    | M1 accel=133, streamer=138, wall=10244
```
