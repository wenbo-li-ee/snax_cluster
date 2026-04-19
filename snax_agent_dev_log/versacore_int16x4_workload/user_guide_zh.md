# VersaCore Int16x4 双核工作负载 — 用户指南

## 1. 硬件配置概览

### 集群架构

本工作负载运行在 SNAX 双核 VersaCore Int16x4 集群上，配置文件位于：
```
cfg/snax_dual_versacore_int16x4_cluster.hjson
```

| 组件 | 规格 |
|------|------|
| 加速器 | snax_dual_versacore_swiglu, 256 MAC 单元 |
| 数据类型 | A: SInt16, B: SInt4, C/D: SInt32, 输出: SInt16 (rescale后) |
| TCDM | 8192 KB (8 MB), 64 banks, 稀疏互连 |
| Streamer 端口 | 48 个 (A:16ch, B0:8ch, B1:8ch, D0:8ch, D1:8ch) |
| 核心数 | 2 (core 0: 计算核, core 1: DMA 核) |
| 阵列形状 (S0) | meshRow=8, tileSize=8, meshCol=4 |

### 数据格式

- **A tile**: 8×8 的 int16 矩阵 = 128 字节 (16通道 × 8字节/通道)
- **B tile**: 4×8 的 int4 矩阵 = 16 字节原始数据，需填充到 64 字节 (8通道 × 8字节/通道)
- **D tile**: 8×4 的 int16 输出 = 64 字节 (8通道 × 8字节/通道)

B tile 的 4× 填充开销 (16B -> 64B) 是 int16x4 数据类型相对于 int8x8 的主要内存效率差异。

## 2. 工作负载维度与 Tiling

### 维度映射

| 参数 | Scaled 1/16 | Full-size | 含义 |
|------|------------|-----------|------|
| M, K, N | 1, 16, 22 | 1, 256, 352 | Mode 0 tile 计数 |
| M1, K1, N1 | 1, 11, 16 | 1, 176, 256 | Mode 1 tile 计数 |
| 矩阵 A | [8, 128] | [8, 2048] | M×meshRow 行, K×tileSize 列 |
| 矩阵 W | [128, 88] | [2048, 1408] | K×tileSize 行, N×meshCol 列 |
| 输出 D0 | [8, 88] | [8, 1408] | Mode 0 SwiGLU 输出 |
| 矩阵 W2 | [88, 64] | [1408, 1024] | Mode 1 权重 |

### N 方向 Tiling (分块)

当 B 数据不能一次全部放入 TCDM 时，需沿 N 方向分块：

| 模式 | Batch N_chunk | Pingpong N_chunk | 分块数 |
|------|--------------|------------------|--------|
| Mode 0 (Full) | 176 | 88 | 2 / 4 |
| Mode 1 (Full) | 128 | 64 | 2 / 4 |

## 3. 双模式执行流程

### Mode 0: SwiGLU 激活

```
D0 = rescale0(A @ W)     ← 矩阵乘法 + 缩放
D1 = rescale1(A @ V)     ← 矩阵乘法 + 缩放
Output = (D0 >> 2) × D1  ← SwiGLU 门控激活
```

### Mode 1: 双流 GEMM

```
D0 = rescale0(A' @ W2_left)
D1 = rescale1(A' @ W2_right)
```

其中 **A' = Mode 0 的 D0 输出**，直接从 TCDM 读取，无需额外 DMA。

### 模式切换

Mode 0 和 Mode 1 严格顺序执行。Mode 1 的 A 输入基地址设置为 Mode 0 的 D0 输出地址 (`delta_local_d0`)，实现零拷贝数据传递。

两个连续的 [8,4] Mode 0 输出 tile (128字节) 恰好构成一个 [8,8] Mode 1 A tile，因此硬件可以直接读取连续内存而无需重排。

## 4. TCDM 内存布局

### Batch 模式布局 (Full-size)

```
偏移 0          : A 数据        (32,768 B)
偏移 32,768     : B0/W 数据     (2,883,584 B)
偏移 2,916,352  : B1/V 数据     (2,883,584 B)
偏移 5,799,936  : D0 输出       (22,528 B)
偏移 5,822,464  : D1_mode0 输出 (22,528 B)
偏移 5,844,992  : Mode1 D0 输出 (16,384 B)
偏移 5,861,376  : Mode1 D1 输出 (16,384 B)
总计使用: 5,878,208 B (5.6 MB / 8 MB TCDM)
```

Mode 1 的 W2 数据复用 B0/B1 缓冲区（DMA 覆盖写入）。

### Pingpong 模式布局

每个 B 矩阵使用两个交替缓冲区：
```
B0_buf[0], B0_buf[1]: 各 N_chunk × K × 64B
B1_buf[0], B1_buf[1]: 各 N_chunk × K × 64B
```

Mode 1 的 W2 缓冲区叠加在 Mode 0 的 B 缓冲区上。

## 5. Batch 与 Pingpong 编程模型

### Batch 编程流程

```c
// 1. DMA 核加载所有数据到 TCDM
snrt_dma_start_1d(local_a, A_data, ...);
snrt_dma_start_1d(local_b0, W_data, ...);
snrt_dma_start_1d(local_b1, V_data, ...);
snrt_dma_wait_all();
snrt_cluster_hw_barrier();

// 2. 计算核运行 Mode 0
for (chunk = 0; chunk < num_chunks; chunk++) {
    // DMA 加载当前 chunk 的 B 数据
    // Barrier
    set_dual_versacore_streamer_csr(...);
    set_dual_versacore_csr(1, K, N_chunk * M, ...);
    set_dual_versacore_mode(0);
    // 设置 rescale 参数
    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    wait_dual_versacore();
    wait_dual_versacore_writer();
}

// 3. 类似地运行 Mode 1
```

### Pingpong 编程流程

```c
// DMA 核预加载 chunk 0 到 buf[0]
snrt_cluster_hw_barrier();

for (chunk = 0; chunk < num_chunks; chunk++) {
    cur = chunk % 2;
    
    // 计算核: 配置 streamer 使用 buf[cur]
    set_dual_versacore_streamer_csr(... b0_buf[cur] ...);
    
    // Barrier: 通知 DMA 核开始加载下一个 chunk
    snrt_cluster_hw_barrier();
    
    // 计算核: 执行加速器
    set_dual_versacore_streamer_start();
    set_dual_versacore_start();
    wait_dual_versacore();
    wait_dual_versacore_writer();
    
    // Barrier: 等待 DMA 核完成下一个 chunk 的加载
    snrt_cluster_hw_barrier();
    
    // DMA 核 (并行): 加载 chunk+1 到 buf[1-cur]
}
```

**关键区别**：Pingpong 通过双缓冲实现 DMA 与计算重叠，减少等待时间。

## 6. Streamer CSR 配置

### CSR 地址映射

| Streamer | 基地址范围 | 功能 |
|----------|-----------|------|
| Reader 0 (A) | CSR 960-976 | 基地址、空间步幅、6级时间边界/步幅 |
| Reader 1 (B0) | CSR 977-989 | 基地址、空间步幅、4级时间边界/步幅 |
| Reader 2 (B1) | CSR 990-1002 | 基地址、空间步幅、4级时间边界/步幅 |
| Writer 0 (D0) | CSR 1003-1015 | 基地址、空间步幅、4级时间边界/步幅 |
| Writer 1 (D1) | CSR 1016-1028 | 基地址、空间步幅、4级时间边界/步幅 |
| 控制寄存器 | CSR 1029-1033 | 启动、忙状态、性能计数器 |

### 关键步幅参数

| 参数 | Mode 0 值 | 含义 |
|------|----------|------|
| Aslstride0 | 8 B | A 读取器空间步幅 (1 bank) |
| Atlstride0 | 128 B | A 时间步幅 level 0 (K 方向, 1 A-tile) |
| Atlbound0 | K | A 时间边界 level 0 |
| Atlbound1 | N_chunk | A 时间边界 level 1 (N 方向) |
| B0tlstride0 | 64 B | B0 时间步幅 (K 方向, 1 padded B-tile) |
| Dtlstride1 | 64 B | D 时间步幅 (N 方向, 1 output tile) |

## 7. 构建与仿真步骤

### 环境准备

```bash
# 进入 Podman 容器
podman exec -w /path/to/snax_cluster/target/snitch_cluster barnard3 bash
source /pixi/entrypoint.sh
```

### 重要提醒：运行时库重建

如果修改了 HJSON 配置文件（特别是 TCDM 大小、bank 数等），**必须重建运行时库**：

```bash
# 清除旧的运行时对象
rm -f sw/runtime/rtl-generic/build/*

# 重建运行时
make -C sw/runtime/rtl-generic

# 然后重建所有应用
make -C sw/apps/<app-name> CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_cluster.hjson TARGET=all
```

不重建运行时会导致栈地址与 TCDM 数据缓冲区重叠，引发难以调试的内存损坏崩溃。

### 生成测试数据

```bash
# datagen.py 自动从 params.hjson 和 HJSON 配置生成 data.h
make -C sw/apps/<app-name>/data
```

### 构建应用

```bash
make -C sw/apps/snax-versacore-int16x4-scale16-batch \
     CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_cluster.hjson TARGET=all
```

### RTL 仿真

```bash
bin/snitch_cluster.vlt sw/apps/<app-name>/build/<app-name>.elf
```

预期输出：
```
Mode 0 SwiGLU: PASS, Error: 0
  M0 Cycles: accel=XXX, streamer=XXX
Mode 1 GEMM D0: PASS, Error: 0
Mode 1 GEMM D1: PASS, Error: 0
  M1 Cycles: accel=XXX, streamer=XXX
```

### 可用测试应用

| 应用名 | 描述 | 预计仿真时间 |
|--------|------|-------------|
| snax-versacore-int16x4-sanity | 最小功能验证 | < 1 分钟 |
| snax-versacore-int16x4-scale16-batch | 1/16 缩放, 批处理 | ~ 2 分钟 |
| snax-versacore-int16x4-scale16-pingpong | 1/16 缩放, 双缓冲 | ~ 2 分钟 |
| snax-versacore-int16x4-fullsize-batch | 全尺寸, 批处理 | ~ 10 分钟 |
| snax-versacore-int16x4-fullsize-pingpong | 全尺寸, 双缓冲 | ~ 10 分钟 |

## 8. 性能分析

### 测量结果汇总

| 工作负载 | Mode | Accel 周期 | Tile 操作数 | 周期/Tile |
|----------|------|-----------|------------|----------|
| Scale16 Batch | M0 | 705 | 352 | 2.00 |
| Scale16 Batch | M1 | 353 | 176 | 2.01 |
| Scale16 Pingpong | M0 | 357 | 176/chunk | 2.03 |
| Scale16 Pingpong | M1 | 180 | 88/chunk | 2.05 |
| Fullsize Batch | M0 | 90,117 | 90,112 | 1.00 |
| Fullsize Batch | M1 | 45,061 | 45,056 | 1.00 |
| Fullsize Pingpong | M0 | 45,061 | 22,528/chunk | 2.00 |
| Fullsize Pingpong | M1 | 22,533 | 11,264/chunk | 2.00 |

### 关键发现

1. **MAC 流水线深度为 2 周期**：乘法和累加各需 1 周期，可重叠执行
2. **长 K 循环实现 1 周期/tile**：K=256 时流水线完全填满，下一 tile 的乘法与当前 tile 的累加重叠
3. **短 K 循环为 2 周期/tile**：K=16 时流水线启停开销无法隐藏
4. **Streamer 开销极小**：Streamer 周期仅比 Accel 多 5-15 个周期

### 吞吐量计算

以全尺寸 Batch Mode 0 为例：
- 256 MAC 单元 × 1 cycle/tile × 90112 tiles = 90112 cycles
- 实际测量 90117 cycles → 效率 99.99%
- 等效吞吐量：256 MAC × 频率 (假设 1 GHz) = 256 GOPS (int16×int4)

## 9. 常见问题排查

### 问题 1: 仿真崩溃 — Illegal Instruction

**症状**：
```
[Illegal Instruction Core 0] PC: XXXX Data: 00d6002b
%Fatal: snitch.sv: Assertion failed
```

**原因**：运行时库 (`libsnRuntime.a`) 使用了旧的 HJSON 配置编译，TCDM 大小不匹配，导致栈被 DMA 数据覆盖。

**解决方案**：
```bash
rm -f sw/runtime/rtl-generic/build/*
make -C sw/runtime/rtl-generic
# 重新构建所有应用
```

### 问题 2: Mode 1 数据不匹配

**症状**：Mode 0 PASS 但 Mode 1 FAIL

**原因**：Mode 1 的 A 输入是 Mode 0 的 D0 输出在 TCDM 中的连续布局。golden model 中的 reshape 必须匹配硬件的通道映射：

```python
# 正确: 简单 reshape 连续内存
mode1_A_flat = mode0_out.reshape(-1)

# 错误: 复杂的 reshape+transpose
# tiles.reshape([M,K1,tiles_per_k,meshRow,meshCol]).transpose(0,1,3,2,4)
```

硬件通道映射：channel i 读取字节 [i×8, i×8+7]，通道对 (0,1) 构成行 0，(2,3) 构成行 1，依此类推。

### 问题 3: B tile 填充

**症状**：B 数据 TCDM 占用远大于原始数据

**原因**：int4 B tile 原始 16B 必须填充到 64B 以匹配 Streamer 的 8 通道 × 8 字节/通道的访问模式。

**解决方案**：这是硬件设计约束，无法避免。在 datagen.py 中通过 `pad_B_tile()` 函数处理。

### 问题 4: TCDM 空间不足

**症状**：datagen.py 的 `assert total_used <= 8388608` 失败

**解决方案**：
- 减小 N_chunk / N1_chunk 以减少每块 B 缓冲区大小
- 使用 Pingpong 模式允许更小的 B 缓冲区
- 考虑 Mode 1 的 W2 缓冲区与 Mode 0 的 B 缓冲区共享（Batch 模式已实现此优化）

### 问题 5: Podman NFS 错误

**症状**：`podman` 命令报 `xattr` 或 `overlay` 错误

**解决方案**：
```bash
export XDG_RUNTIME_DIR=/tmp/$(whoami)_runtime
mkdir -p $XDG_RUNTIME_DIR
```
确保 Podman 的 runroot 在本地文件系统上，而非 NFS。
