# Dual VersaCore + Rescale 寄存器配置说明

本文档详细介绍 SNAX Dual VersaCore Int16x4 加速器的所有配置寄存器，包括 Streamer（数据搬运器）、VersaCore（矩阵乘法核心）和 Rescale（量化缩放模块）三大部分。

---

## 目录

1. [系统架构概览](#1-系统架构概览)
2. [Streamer 寄存器](#2-streamer-寄存器)
3. [VersaCore 加速器寄存器](#3-versacore-加速器寄存器)
4. [Rescale 模块寄存器](#4-rescale-模块寄存器)
5. [完整配置示例](#5-完整配置示例)
6. [常见问题与注意事项](#6-常见问题与注意事项)

---

## 1. 系统架构概览

Dual VersaCore 系统包含：
- **3 个 Reader**：Reader 0（A 矩阵）、Reader 1（B0/W 权重）、Reader 2（B1/V 权重）
- **2 个 Writer**：Writer 0（D0 输出）、Writer 1（D1 输出）
- **2 个 VersaCore 核心**：VC0 和 VC1，共享相同的配置参数
- **3 个 Rescale 模块**：Rescale0（VC0 路径）、Rescale1（VC1 路径）、Rescale_mul（逐元素乘法后）
- **2 种工作模式**：Mode 0（SwiGLU）和 Mode 1（GEMM）

### 数据流

```
Mode 0 (SwiGLU):
  A ──┬──→ VC0 @ W ──→ Rescale0 ──→ >>2 (SiLU近似) ──┐
      └──→ VC1 @ V ──→ Rescale1 ──────────────────────┤
                                                        ├──→ ElemMul ──→ Rescale_mul ──→ D0
                                                        
Mode 1 (GEMM):
  A ──┬──→ VC0 @ B0 ──→ Rescale0 ──→ D0
      └──→ VC1 @ B1 ──→ Rescale1 ──→ D1
```

### 硬件参数（Int16x4, array_shape=0, data_type=0）

| 参数 | 值 | 说明 |
|------|-----|------|
| meshRow | 8 | A 矩阵的行展开维度 |
| tileSize | 8 | K 维度的空间展开 |
| meshCol | 4 | B 矩阵的列展开维度 |
| A 元素位宽 | 16 bit (int16) | 输入 A 的数据类型 |
| B 元素位宽 | 4 bit (int4) | 权重 B 的数据类型 |
| 输出位宽 | 32 bit (int32) | GEMM 累加结果 |
| Rescale 输出 | 16 bit (int16) | 缩放后的输出 |
| bankWidth | 64 bit | TCDM bank 宽度 |

### 一个 Tile 的物理含义

- **A tile**: `[meshRow, tileSize]` = `[8, 8]` = 64 个 int16 元素 = 128 字节
- **B tile**: `[meshCol, tileSize]` = `[4, 8]` = 32 个 int4 元素 = 16 字节（原始），需 padding 到 64 字节
- **D tile**: `[meshRow, meshCol]` = `[8, 4]` = 32 个 int16 元素 = 64 字节

---

## 2. Streamer 寄存器

Streamer 负责在 TCDM（L1 scratchpad）和加速器之间搬运数据。每个 Reader/Writer 端口有以下寄存器组：

### 2.1 通用寄存器结构

每个 Reader/Writer 端口的寄存器：

| 寄存器 | 功能 | 说明 |
|--------|------|------|
| `BASE_PTR_LOW` | 基地址（低 32 位） | TCDM 中数据的起始地址 |
| `BASE_PTR_HIGH` | 基地址（高 16 位） | 用于 48 位寻址（通常为 0） |
| `S_STRIDE` | 空间步长 | 相邻 channel 之间的字节间隔，通常 = bankWidth/8 = 8 字节 |
| `T_BOUND[i]` | 时间维度循环边界 | 第 i 层嵌套循环的迭代次数 |
| `T_STRIDE[i]` | 时间维度步长 | 第 i 层循环每步地址增量（字节） |
| `ADDR_REMAP_INDEX` | 地址重映射索引 | 通常设为 0 |
| `ENABLED_CHANNEL` | 通道使能位掩码 | 每 bit 对应一个 TCDM 读/写通道 |

### 2.2 Reader 0 — A 矩阵（CSR 960-976）

读取输入矩阵 A（int16），拥有 **16 个 channel**，**6 层时间循环**。

| CSR 地址 | 寄存器名 | 含义 |
|----------|---------|------|
| 960 | `BASE_PTR_READER_0_LOW` | A 矩阵基地址（低 32 位） |
| 961 | `BASE_PTR_READER_0_HIGH` | A 矩阵基地址（高 16 位） |
| 962 | `S_STRIDE_READER_0_0` | 空间步长 = 8 字节 |
| 963 | `T_BOUND_READER_0_0` | 第 0 层循环边界 = K（沿 K 维度的 tile 数） |
| 964 | `T_BOUND_READER_0_1` | 第 1 层循环边界 = N（A 在 N 维度广播） |
| 965 | `T_BOUND_READER_0_2` | 第 2 层循环边界 = M（沿 M 维度的 tile 数） |
| 966-968 | `T_BOUND_READER_0_3..5` | 第 3-5 层边界（未使用，设为 1） |
| 969 | `T_STRIDE_READER_0_0` | K 步长 = meshRow * tileSize * 2 = 128 字节 |
| 970 | `T_STRIDE_READER_0_1` | N 步长 = 0（A 在 N 维度广播，不移动地址） |
| 971 | `T_STRIDE_READER_0_2` | M 步长 = K * meshRow * tileSize * 2 字节 |
| 972-974 | `T_STRIDE_READER_0_3..5` | 未使用步长（设为 0） |
| 975 | `ADDR_REMAP_INDEX_READER_0` | 地址重映射 = 0 |
| 976 | `ENABLED_CHANNEL_READER_0` | 通道使能 = 0xFFFF（16 个 channel 全开） |

**循环执行顺序**（最内层到最外层）：
```
for m in range(M):           // 第 2 层
  for n in range(N):         // 第 1 层
    for k in range(K):       // 第 0 层
      读取 A tile [meshRow, tileSize] = 128 字节
      地址 = base + k * Atlstride0 + n * 0 + m * Atlstride2
```

### 2.3 Reader 1 — B0/W 权重（CSR 977-989）

读取第一组权重矩阵 W/B0（int4 nibble-packed），拥有 **8 个 channel**，**4 层时间循环**。

| CSR 地址 | 寄存器名 | 含义 |
|----------|---------|------|
| 977 | `BASE_PTR_READER_1_LOW` | B0 矩阵基地址 |
| 978 | `BASE_PTR_READER_1_HIGH` | B0 矩阵基地址高位 |
| 979 | `S_STRIDE_READER_1_0` | 空间步长 = 8 字节 |
| 980 | `T_BOUND_READER_1_0` | 第 0 层边界 = K |
| 981 | `T_BOUND_READER_1_1` | 第 1 层边界 = N |
| 982 | `T_BOUND_READER_1_2` | 第 2 层边界 = M |
| 983 | `T_BOUND_READER_1_3` | 第 3 层边界 = 1（未使用） |
| 984 | `T_STRIDE_READER_1_0` | K 步长 = b_tile_padded（64 字节，含 padding） |
| 985 | `T_STRIDE_READER_1_1` | N 步长 = K * b_tile_padded |
| 986 | `T_STRIDE_READER_1_2` | M 步长 = 0（B 在 M 维度广播） |
| 987 | `T_STRIDE_READER_1_3` | 未使用步长 = 0 |
| 988 | `ADDR_REMAP_INDEX_READER_1` | 地址重映射 = 0 |
| 989 | `ENABLED_CHANNEL_READER_1` | 通道使能 = 0xFF（8 个 channel） |

**重要**：B tile 必须从原始 16 字节 padding 到 64 字节（即 channel footprint），否则 streamer 会因为地址重叠而导致 TCDM 死锁。

### 2.4 Reader 2 — B1/V 权重（CSR 990-1002）

与 Reader 1 完全对称，读取第二组权重 V/B1。寄存器结构和含义与 Reader 1 相同。

### 2.5 Writer 0 — D0 输出（CSR 1003-1015）

写入 VC0 路径的输出结果，拥有 **8 个 channel**，**4 层时间循环**。

| CSR 地址 | 寄存器名 | 含义 |
|----------|---------|------|
| 1003 | `BASE_PTR_WRITER_0_LOW` | D0 输出基地址 |
| 1004 | `BASE_PTR_WRITER_0_HIGH` | D0 输出基地址高位 |
| 1005 | `S_STRIDE_WRITER_0_0` | 空间步长 = 8 字节 |
| 1006 | `T_BOUND_WRITER_0_0` | 第 0 层边界 = 1（单次写出） |
| 1007 | `T_BOUND_WRITER_0_1` | 第 1 层边界 = N |
| 1008 | `T_BOUND_WRITER_0_2` | 第 2 层边界 = M |
| 1009 | `T_BOUND_WRITER_0_3` | 第 3 层边界 = 1 |
| 1010 | `T_STRIDE_WRITER_0_0` | 第 0 层步长 = 8 * 8 = 64 字节 |
| 1011 | `T_STRIDE_WRITER_0_1` | N 步长 = meshRow * meshCol * 2 = 64 字节 |
| 1012 | `T_STRIDE_WRITER_0_2` | M 步长 = N * meshRow * meshCol * 2 字节 |
| 1013 | `T_STRIDE_WRITER_0_3` | 未使用 = 0 |
| 1014 | `ADDR_REMAP_INDEX_WRITER_0` | 地址重映射 = 0 |
| 1015 | `ENABLED_CHANNEL_WRITER_0` | 通道使能 = 0xFF（8 channel） |

### 2.6 Writer 1 — D1 输出（CSR 1016-1028）

与 Writer 0 结构相同，写入 VC1 路径的输出。

### 2.7 Streamer 控制/状态寄存器

| CSR 地址 | 寄存器名 | 读/写 | 说明 |
|----------|---------|-------|------|
| 1029 | `STREAMER_START_CSR` | W | 写 1 启动 Streamer |
| 1030 | `STREAMER_BUSY_CSR` | R | 1=Streamer 忙，0=空闲 |
| 1031 | `STREAMER_PERFORMANCE_COUNTER_CSR` | R | Streamer 周期计数器 |
| 1032 | `STREAMER_WRITER_BUSY_CSR` | R | Writer 0 忙状态 |
| 1033 | `STREAMER_WRITER1_BUSY_CSR` | R | Writer 1 忙状态 |

---

## 3. VersaCore 加速器寄存器

VersaCore 的 CSR 地址从 `STREAMER_WRITER1_BUSY_CSR + 1 = 1034` 开始。

### 3.1 计算控制寄存器

| CSR 偏移 | 宏定义 | 功能 | 取值说明 |
|----------|--------|------|---------|
| [0] | `DUAL_VC_OVERWRITE_ACCUM` | 是否重载累加器 | 1 = 每次新计算清零累加器并加载新输入；0 = 保持累加 |
| [1] | `DUAL_VC_ACCUM_BOUND` | 累加边界 | = K（在 K 维度上累加多少个 tile 后产出一个输出 tile） |
| [2] | `DUAL_VC_OUTPUT_BOUND` | 输出次数 | = N * M（总共产出多少个输出 tile） |
| [3] | `DUAL_VC_SUBTRACTIONS` | 减法偏移 | 低 8 位 = subtraction_a，高 8 位 = subtraction_b。用于零点偏移 |
| [4] | `DUAL_VC_ARRAY_SHAPE_CFG` | 阵列形状配置 | 0 = shape S0 (8,8,4)；其他值对应不同 unrolling |
| [5] | `DUAL_VC_DATA_TYPE_CFG` | 数据类型配置 | 0 = int16×int4→int32 |
| [6] | `DUAL_VC_MODE` | 工作模式 | 0 = SwiGLU（两个 VC 结果做逐元素乘法）；1 = GEMM（独立输出） |

### 3.2 各寄存器详解

#### OVERWRITE_ACCUM（重载累加器）

控制 VersaCore 是否在开始新的输出 tile 计算时清零累加器。

- **值 = 1**：每次开始新的 K 累加循环时，先清零累加寄存器（正常模式）
- **值 = 0**：保留上一次的累加值，继续累加（用于分块累加场景）

#### ACCUM_BOUND（累加边界 = K）

告诉硬件每产出一个输出 tile 需要执行多少次乘累加操作。等于 K 维度的 tile 数量。

例如：K=8 表示每个输出 tile 需要读取 8 个 A tile 和 8 个 B tile 做矩阵乘累加。

#### OUTPUT_BOUND（输出次数 = N * M）

告诉硬件总共要产出多少个输出 tile。

例如：M=2, N=16 → OUTPUT_BOUND = 32，表示总共产出 32 个 `[8,4]` 输出 tile。

#### SUBTRACTIONS（减法配置）

用于量化场景的零点偏移。在乘法之前，先从输入中减去指定值：
```
A_actual = A - subtraction_a
B_actual = B - subtraction_b
```
打包格式：`(subtraction_b << 8) | subtraction_a`

#### ARRAY_SHAPE_CFG

选择 VersaCore 阵列的空间展开方式。不同的 shape 对应不同的 (meshRow, tileSize, meshCol) 组合：

| array_shape | meshRow | tileSize | meshCol | 说明 |
|-------------|---------|----------|---------|------|
| 0 | 8 | 8 | 4 | 默认 shape，适合 int16x4 |

#### DATA_TYPE_CFG

选择输入/输出数据类型组合：

| data_type | A 类型 | B 类型 | 累加类型 | 说明 |
|-----------|--------|--------|---------|------|
| 0 | int16 | int4 | int32 | 默认 int16x4 模式 |

#### MODE（工作模式）

| mode | 名称 | 行为 |
|------|------|------|
| 0 | SwiGLU | VC0 和 VC1 分别计算 A@W 和 A@V，结果经过 Rescale，然后做逐元素乘法 |
| 1 | GEMM | VC0 和 VC1 独立计算 A@B0 和 A@B1，分别输出到 D0 和 D1 |

### 3.3 控制/状态寄存器

| CSR 偏移 | 宏定义 | 读/写 | 说明 |
|----------|--------|-------|------|
| [19] | `DUAL_VC_START` | W | 写 1 启动加速器 |
| [20] | `DUAL_VC_BUSY` | R | 1=加速器忙，0=计算完成 |
| [21] | `DUAL_VC_PERFORMANCE_COUNTER` | R | 加速器周期计数器 |

---

## 4. Rescale 模块寄存器

Rescale 模块将 32 位累加结果缩放为 16 位输出。共有 **3 个 Rescale 模块**，每个有 4 个寄存器。

### 4.1 Rescale 计算公式

```
output = clamp( ((input - input_zp) * multiplier + round_bias) >> shift + output_zp, -32768, 32767 )
```

其中 `round_bias` 在 `shift > 0` 时为 `1 << (shift - 1)`，用于实现四舍五入。

### 4.2 Rescale0 寄存器（VC0 路径）

| CSR 偏移 | 宏定义 | 类型 | 说明 |
|----------|--------|------|------|
| [7] | `DUAL_VC_RESCALE0_INPUT_ZP` | int32 | 输入零点。在乘以 multiplier 之前，先减去此值 |
| [8] | `DUAL_VC_RESCALE0_MULTIPLIER` | uint32 | 缩放乘数。定点乘法因子 |
| [9] | `DUAL_VC_RESCALE0_OUTPUT_ZP` | int32 | 输出零点。移位之后加上此值 |
| [10] | `DUAL_VC_RESCALE0_SHIFT` | uint32 | 右移位数。控制缩放精度 |

### 4.3 Rescale1 寄存器（VC1 路径）

| CSR 偏移 | 宏定义 | 类型 | 说明 |
|----------|--------|------|------|
| [11] | `DUAL_VC_RESCALE1_INPUT_ZP` | int32 | 输入零点 |
| [12] | `DUAL_VC_RESCALE1_MULTIPLIER` | uint32 | 缩放乘数 |
| [13] | `DUAL_VC_RESCALE1_OUTPUT_ZP` | int32 | 输出零点 |
| [14] | `DUAL_VC_RESCALE1_SHIFT` | uint32 | 右移位数 |

### 4.4 Rescale_mul 寄存器（逐元素乘法后，仅 Mode 0）

| CSR 偏移 | 宏定义 | 类型 | 说明 |
|----------|--------|------|------|
| [15] | `DUAL_VC_RESCALE_MUL_INPUT_ZP` | int32 | 输入零点 |
| [16] | `DUAL_VC_RESCALE_MUL_MULTIPLIER` | uint32 | 缩放乘数 |
| [17] | `DUAL_VC_RESCALE_MUL_OUTPUT_ZP` | int32 | 输出零点 |
| [18] | `DUAL_VC_RESCALE_MUL_SHIFT` | uint32 | 右移位数 |

### 4.5 Rescale 参数的含义

| 参数 | 典型范围 | 作用 |
|------|---------|------|
| `input_zp` | -128 ~ 127 | 将输入的量化零点移除，还原为真实值。对于对称量化，设为 0 |
| `multiplier` | 1 ~ 2^31 | 定点缩放因子。与 shift 配合实现任意比例缩放：实际缩放 = multiplier / 2^shift |
| `output_zp` | -128 ~ 127 | 为输出添加量化零点偏移。对于对称量化，设为 0 |
| `shift` | 0 ~ 31 | 右移位数。shift=0 表示不缩放（恒等变换） |

---

## 5. 完整配置示例

### 示例 1：恒等缩放（Identity Rescale）

最简单的情况——不做任何缩放，直接截断 32 位到 16 位：

```c
// Rescale 参数：不缩放，直接截断
set_dual_versacore_rescale0(
    0,    // input_zp = 0（不减零点）
    1,    // multiplier = 1（乘以 1）
    0,    // output_zp = 0（不加零点）
    0     // shift = 0（不右移）
);
// 效果：output_int16 = clamp(input_int32, -32768, 32767)
```

### 示例 2：1/256 缩放

将 32 位累加结果缩小 256 倍后输出为 16 位：

```c
set_dual_versacore_rescale0(
    0,      // input_zp = 0
    1,      // multiplier = 1
    0,      // output_zp = 0
    8       // shift = 8（右移 8 位 = 除以 256）
);
// 效果：output = input >> 8（带四舍五入）
```

### 示例 3：带零点的非对称量化

用于 INT8 非对称量化推理场景：

```c
set_dual_versacore_rescale0(
    10,         // input_zp = 10（输入零点）
    1073741824, // multiplier ≈ 2^30（大乘数）
    -5,         // output_zp = -5（输出零点）
    30          // shift = 30
);
// 效果：output = ((input - 10) * 1073741824 + round) >> 30 + (-5)
//      ≈ (input - 10) * 1.0 + (-5)
//      = input - 15
```

### 示例 4：完整的 M=2, K=8, N=16 GEMM 配置

```c
// 1. 配置 VersaCore
set_dual_versacore_csr(
    1,              // overwrite_accum = 1（每次清零累加器）
    8,              // accum_bound = K = 8（累加 8 个 tile）
    32,             // output_bound = N * M = 16 * 2 = 32
    0,              // subtractions = 0（无零点偏移）
    0,              // array_shape = 0（shape S0: 8,8,4）
    0               // data_type = 0（int16 x int4）
);

// 2. 设置模式
set_dual_versacore_mode(0);  // Mode 0 = SwiGLU

// 3. 配置 Rescale（三个模块都用恒等缩放）
set_dual_versacore_rescale0(0, 1, 0, 0);
set_dual_versacore_rescale1(0, 1, 0, 0);
set_dual_versacore_rescale_mul(0, 1, 0, 0);

// 4. 启动
set_dual_versacore_streamer_start();  // 启动 Streamer
set_dual_versacore_start();           // 启动加速器

// 5. 等待完成
wait_dual_versacore();                // 等加速器完成
wait_dual_versacore_writer();         // 等写回完成
```

### 示例 5：Streamer 步长计算

以 M=2, K=8, N=16 为例，详细计算各端口步长：

```
硬件参数：meshRow=8, tileSize=8, meshCol=4, a_len=16bit, b_len=4bit
bankWidth=64bit, out_elem_bits=16bit

=== Reader 0 (A 矩阵) ===
空间步长 = bankWidth/8 = 8 字节
一个 A tile = meshRow * tileSize * a_len / 8 = 8 * 8 * 16 / 8 = 128 字节
使能通道数 = meshRow * tileSize * a_len / bankWidth = 8 * 8 * 16 / 64 = 16

T_BOUND[0] = K = 8        T_STRIDE[0] = 128 字节（相邻 K tile 间距）
T_BOUND[1] = N = 16       T_STRIDE[1] = 0（A 在 N 维广播）
T_BOUND[2] = M = 2        T_STRIDE[2] = K * 128 = 1024 字节（跳到下一个 M）

=== Reader 1 (B0/W 权重) ===
一个 B tile 原始 = meshCol * tileSize * b_len / 8 = 4 * 8 * 4 / 8 = 16 字节
Padding 后 = 64 字节（必须对齐到 channel footprint）
使能通道数 = 8

T_BOUND[0] = K = 8        T_STRIDE[0] = 64 字节（padded tile）
T_BOUND[1] = N = 16       T_STRIDE[1] = K * 64 = 512 字节
T_BOUND[2] = M = 2        T_STRIDE[2] = 0（B 在 M 维广播）

=== Writer 0 (D0 输出) ===
一个 D tile = meshRow * meshCol * out_elem_bits / 8 = 8 * 4 * 16 / 8 = 64 字节
使能通道数 = 8

T_BOUND[0] = 1            T_STRIDE[0] = 64 字节
T_BOUND[1] = N = 16       T_STRIDE[1] = 64 字节（相邻 N tile 间距）
T_BOUND[2] = M = 2        T_STRIDE[2] = N * 64 = 1024 字节
```

---

## 6. 常见问题与注意事项

### 6.1 B Tile Padding

**问题**：Int16x4 模式下，B tile 原始大小为 16 字节，小于 channel footprint（64 字节）。如果不 padding，streamer 在相邻时间步会读取重叠的 TCDM 地址，导致互连死锁。

**解决**：将每个 B tile padding 到 64 字节，确保 `B0tlstride0 >= channel_footprint`。

### 6.2 内存对齐

所有 streamer 端口的基地址必须对齐到 `granularity * bankWidth / 8` 字节边界。使用 `align_wide_addr()` 函数确保对齐。

### 6.3 ACCUM_BOUND 与 OUTPUT_BOUND 的关系

- `ACCUM_BOUND = K`：沿 K 维度累加多少次后产出一个输出
- `OUTPUT_BOUND = N * M`：总共要产出的输出 tile 数量
- 总的乘累加操作次数 = `K * N * M`

### 6.4 Mode 0 vs Mode 1

| 特性 | Mode 0 (SwiGLU) | Mode 1 (GEMM) |
|------|-----------------|---------------|
| VC0 输出 | 经 Rescale0 → >>2 → 送入 ElemMul | 经 Rescale0 → 直接输出到 D0 |
| VC1 输出 | 经 Rescale1 → 送入 ElemMul | 经 Rescale1 → 直接输出到 D1 |
| 最终输出 | ElemMul 结果经 Rescale_mul → D0 | D0, D1 分别独立输出 |
| Rescale_mul | 生效 | 不生效（但仍需配置） |
| D1 Writer | Mode 0 下 D1 写出 VC1 rescale 后的中间结果 | Mode 1 下 D1 写出独立 GEMM 结果 |

### 6.5 启动与等待顺序

```c
// 正确的启动顺序：
set_dual_versacore_streamer_start();  // 1. 先启动 Streamer
set_dual_versacore_start();           // 2. 再启动加速器

// 正确的等待顺序：
wait_dual_versacore();                // 3. 等加速器完成
wait_dual_versacore_writer();         // 4. 等写回完成
```

### 6.6 缩放因子的选择

对于实际量化推理，缩放因子的选取需要根据模型量化参数计算：

```
实际缩放比 = multiplier / 2^shift

例如：想要缩放 0.00123
  → 选 shift = 20, multiplier = round(0.00123 * 2^20) = 1290
  → 实际缩放 = 1290 / 1048576 ≈ 0.001230

例如：想要缩放 1.0（恒等）
  → shift = 0, multiplier = 1
  或 shift = 16, multiplier = 65536
```

---

## 附录：CSR 地址汇总表

| 功能模块 | CSR 范围 | 数量 |
|---------|---------|------|
| Reader 0 (A) | 960-976 | 17 |
| Reader 1 (B0) | 977-989 | 13 |
| Reader 2 (B1) | 990-1002 | 13 |
| Writer 0 (D0) | 1003-1015 | 13 |
| Writer 1 (D1) | 1016-1028 | 13 |
| Streamer 控制 | 1029-1033 | 5 |
| VersaCore 控制 | 1034-1039 | 6 |
| VersaCore Mode | 1040 | 1 |
| Rescale0 | 1041-1044 | 4 |
| Rescale1 | 1045-1048 | 4 |
| Rescale_mul | 1049-1052 | 4 |
| VC Start/Busy/Perf | 1053-1055 | 3 |
| **总计** | 960-1055 | **96** |
