# Dual VersaCore SwiGLU 加速器配置指南

## 1. 架构概述

Dual VersaCore SwiGLU 加速器包含两个 VersaCore 矩阵乘法核心和一个后处理流水线，支持两种运算模式。

```
                          ┌─────────────────────────────┐
                          │        Shared A Input        │
                          │       (stream2acc_0)         │
                          └──────┬──────────┬────────────┘
                                 │          │
                    ┌────────────▼──┐  ┌────▼────────────┐
   B0 (stream2acc_1)│  VersaCore 0  │  │  VersaCore 1   │B1 (stream2acc_2)
                    │   (A @ B0)    │  │   (A @ B1)     │
                    └───────┬───────┘  └───────┬─────────┘
                            │ int32            │ int32
                    ┌───────▼───────┐  ┌───────▼─────────┐
                    │  RescaleDown0 │  │  RescaleDown1   │
                    │  (32→16 bit)  │  │  (32→16 bit)   │
                    └───────┬───────┘  └───────┬─────────┘
                            │ int16            │ int16
                            │                  │
                  ┌─────────▼──────────────────▼──────────┐
                  │          Mode Mux (mode_sel)           │
                  ├─── Mode 0 (SwiGLU) ───┬── Mode 1 ────┤
                  │                       │   (GEMM)      │
                  │  ┌──────────────┐     │               │
                  │  │ Shifter 6stg │     │  rescale0→D0  │
                  │  │  (>>2, SiLU) │     │  rescale1→D1  │
                  │  └──────┬───────┘     │               │
                  │         │ int16       │               │
                  │  ┌──────▼───────┐     │               │
                  │  │ ElemMul 16b  │◄────┤               │
                  │  │(int16×int16) │     │               │
                  │  └──────┬───────┘     │               │
                  │         │ int32       │               │
                  │  ┌──────▼───────┐     │               │
                  │  │ RescaleMul   │     │               │
                  │  │ (32→16 bit)  │     │               │
                  │  └──────┬───────┘     │               │
                  │         │ int16       │               │
                  │   ──────▼─────        │               │
                  │   D0=D1=same          │               │
                  └───────────────────────┴───────────────┘
                            │                    │    │
                    ┌───────▼───────┐    ┌───────▼──┐ ▼
                    │  Writer 0     │    │Writer 0  │Writer 1
                    │(acc2stream_0) │    │          │
                    └───────────────┘    └──────────┴────────┘
```

### 关键参数

| 参数 | 值 | 含义 |
|------|-----|------|
| meshRow | 16 | 每个 VersaCore 的行数 |
| tileSize | 8 | 矩阵乘法的内积维度（K 方向每次处理的元素数） |
| meshCol | 8 | 每个 VersaCore 的列数 |
| MAC 数量 | 1024 | meshRow × tileSize × meshCol = 16×8×8 |
| PostprocLanes | 64 | 后处理流水线宽度（每周期处理 64 个元素） |
| 输入 A 宽度 | 1024 bit | meshRow × tileSize × 8bit = 16×8×8 |
| 输入 B 宽度 | 8192 bit | meshCol × tileSize × 8bit × 128通道 |
| 输出 D 宽度 | 2048 bit | 128 个 int16 元素 |

---

## 2. 模式说明

### Mode 0 — SwiGLU

SwiGLU 激活函数的硬件近似：

```
output = RescaleMul( RescaleDown0(A @ W) >> 2  ×  RescaleDown1(A @ V) )
```

**数据流：**
1. VC0 计算 `A @ W`（int8 × int8 → int32）
2. VC1 计算 `A @ V`（int8 × int8 → int32）
3. RescaleDown0 将 VC0 输出从 int32 量化到 int16
4. Shifter 6-stage 将结果算术右移 2 位（SiLU 近似）
5. RescaleDown1 将 VC1 输出从 int32 量化到 int16
6. ElemMul 16b 逐元素相乘（int16 × int16 → int32）
7. RescaleMul 将乘法结果从 int32 量化到 int16
8. **两个 Writer 都输出相同的数据**（Writer 1 写入 dummy 地址）

### Mode 1 — GEMM

独立双路矩阵乘法：

```
D0 = RescaleDown0(A @ B0)
D1 = RescaleDown1(A @ B1)
```

**数据流：**
1. VC0 计算 `A @ B0`（int8 × int8 → int32）
2. VC1 计算 `A @ B1`（int8 × int8 → int32）
3. RescaleDown0 → Writer 0 (D0)
4. RescaleDown1 → Writer 1 (D1)
5. Shifter/ElemMul/RescaleMul 不参与运算

**典型应用场景：** 将一个大的权重矩阵按列分成两半，分别喂给两个 VersaCore，实现 2× 吞吐。

---

## 3. CSR 寄存器映射

### 3.1 Streamer CSR

Streamer 控制数据在 TCDM（L1 SRAM）和加速器之间的搬运。每个 reader/writer 都有自己的 CSR 组。

#### Reader 0 — 矩阵 A（共享，供两个 VersaCore 使用）

| CSR 地址 | 名称 | 说明 |
|---------|------|------|
| 960 | BASE_PTR_READER_0_LOW | 基地址低 32 位（L1 中的绝对地址） |
| 961 | BASE_PTR_READER_0_HIGH | 基地址高 32 位（通常为 0） |
| 962 | S_STRIDE_READER_0_0 | 空间步长（通道间的字节间隔），通常为 `bankWidth/8 = 8` |
| 963–968 | T_BOUND_READER_0_0~5 | 6 个时间循环边界（详见 §6） |
| 969–974 | T_STRIDE_READER_0_0~5 | 6 个时间循环步长（字节） |
| 975 | ADDR_REMAP_INDEX_READER_0 | 地址重映射索引（通常为 0） |
| 976 | ENABLED_CHANNEL_READER_0 | 通道使能位图（1 个 CSR，16 通道） |

**时间维度映射（Output Stationary）：**
- T_BOUND_0 / T_STRIDE_0 = K 维度（内积累加循环）
- T_BOUND_1 / T_STRIDE_1 = N 维度（A 不随 N 移动，stride=0）
- T_BOUND_2 / T_STRIDE_2 = M 维度（切换到下一块 A）
- T_BOUND_3~5 = 1（未使用）

#### Reader 1 — 矩阵 B0（VersaCore 0 的 B 输入）

| CSR 地址 | 名称 | 说明 |
|---------|------|------|
| 977–978 | BASE_PTR_READER_1 | 基地址 |
| 979 | S_STRIDE_READER_1_0 | 空间步长 = `bankWidth/8 = 8` |
| 980–982 | T_BOUND_READER_1_0~2 | 3 个时间循环边界 |
| 983–985 | T_STRIDE_READER_1_0~2 | 3 个时间循环步长 |
| 986 | ADDR_REMAP_INDEX_READER_1 | 地址重映射索引 |
| 987–990 | ENABLED_CHANNEL_READER_1 | 通道使能（4 个 CSR，最多 128 通道） |

**时间维度映射：**
- T_BOUND_0 / T_STRIDE_0 = K 维度
- T_BOUND_1 / T_STRIDE_1 = N 维度（切换到下一块 B）
- T_BOUND_2 / T_STRIDE_2 = M 维度（B 不随 M 移动，stride=0）

#### Reader 2 — 矩阵 B1（VersaCore 1 的 B 输入）

| CSR 地址 | 名称 | 说明 |
|---------|------|------|
| 991–992 | BASE_PTR_READER_2 | 基地址 |
| 993 | S_STRIDE_READER_2_0 | 空间步长 |
| 994–996 | T_BOUND_READER_2_0~2 | 3 个时间循环边界 |
| 997–999 | T_STRIDE_READER_2_0~2 | 3 个时间循环步长 |
| 1000 | ADDR_REMAP_INDEX_READER_2 | 地址重映射索引 |
| 1001–1004 | ENABLED_CHANNEL_READER_2 | 通道使能（4 个 CSR） |

结构与 Reader 1 完全相同。

#### Writer 0 — 输出 D0

| CSR 地址 | 名称 | 说明 |
|---------|------|------|
| 1005–1006 | BASE_PTR_WRITER_0 | 基地址 |
| 1007 | S_STRIDE_WRITER_0_0 | 空间步长 = 8 |
| 1008–1011 | T_BOUND_WRITER_0_0~3 | 4 个时间循环边界 |
| 1012–1015 | T_STRIDE_WRITER_0_0~3 | 4 个时间循环步长 |
| 1016 | ADDR_REMAP_INDEX_WRITER_0 | 地址重映射索引 |
| 1017 | ENABLED_CHANNEL_WRITER_0 | 通道使能（1 个 CSR，32 通道） |

**时间维度映射：**
- T_BOUND_0 / T_STRIDE_0 = 序列化循环（当 PostprocLanes < ElemsPerBeat 时）
- T_BOUND_1 / T_STRIDE_1 = N 维度
- T_BOUND_2 / T_STRIDE_2 = M 维度
- T_BOUND_3 = 1（未使用）

#### Writer 1 — 输出 D1

| CSR 地址 | 名称 | 说明 |
|---------|------|------|
| 1018–1019 | BASE_PTR_WRITER_1 | 基地址 |
| 1020 | S_STRIDE_WRITER_1_0 | 空间步长 |
| 1021–1024 | T_BOUND_WRITER_1_0~3 | 4 个时间循环边界 |
| 1025–1028 | T_STRIDE_WRITER_1_0~3 | 4 个时间循环步长 |
| 1029 | ADDR_REMAP_INDEX_WRITER_1 | 地址重映射索引 |
| 1030 | ENABLED_CHANNEL_WRITER_1 | 通道使能 |

结构与 Writer 0 完全相同。

#### Streamer 控制/状态

| CSR 地址 | 名称 | 读/写 | 说明 |
|---------|------|-------|------|
| 1031 | STREAMER_START_CSR | W | 写 1 启动 Streamer |
| 1032 | STREAMER_BUSY_CSR | R | Streamer 忙标志 |
| 1033 | STREAMER_PERFORMANCE_COUNTER_CSR | R | Streamer 性能计数器（周期数） |
| 1034 | STREAMER_WRITER_BUSY_CSR | R | Writer 0 忙标志 |
| 1035 | STREAMER_WRITER1_BUSY_CSR | R | Writer 1 忙标志 |

### 3.2 加速器 CSR

加速器 CSR 紧跟在 Streamer CSR 之后，基地址 = `STREAMER_WRITER1_BUSY_CSR + 1 = 1036`。

| CSR 地址 | 索引 | 名称 | 读/写 | 说明 |
|---------|------|------|-------|------|
| 1036 | [0] | DUAL_VC_OVERWRITE_ACCUM | W | 1 = 覆盖累加器（取新的 C=0），0 = 累加模式 |
| 1037 | [1] | DUAL_VC_ACCUM_BOUND | W | 累加次数 = K（每次输出需要 K 次 A×B 累加） |
| 1038 | [2] | DUAL_VC_OUTPUT_BOUND | W | 输出块数 = N × M |
| 1039 | [3] | DUAL_VC_SUBTRACTIONS | W | 减法配置：`(subtraction_b << 8) | subtraction_a` |
| 1040 | [4] | DUAL_VC_ARRAY_SHAPE_CFG | W | 阵列形状索引（通常为 0） |
| 1041 | [5] | DUAL_VC_DATA_TYPE_CFG | W | 数据类型索引（0=INT8） |
| 1042 | [6] | DUAL_VC_MODE | W | 模式选择：0=SwiGLU，1=GEMM |
| 1043 | [7] | DUAL_VC_RESCALE0_INPUT_ZP | W | RescaleDown0 输入零点 |
| 1044 | [8] | DUAL_VC_RESCALE0_MULTIPLIER | W | RescaleDown0 乘数 |
| 1045 | [9] | DUAL_VC_RESCALE0_OUTPUT_ZP | W | RescaleDown0 输出零点 |
| 1046 | [10] | DUAL_VC_RESCALE0_SHIFT | W | RescaleDown0 移位量 |
| 1047 | [11] | DUAL_VC_RESCALE1_INPUT_ZP | W | RescaleDown1 输入零点 |
| 1048 | [12] | DUAL_VC_RESCALE1_MULTIPLIER | W | RescaleDown1 乘数 |
| 1049 | [13] | DUAL_VC_RESCALE1_OUTPUT_ZP | W | RescaleDown1 输出零点 |
| 1050 | [14] | DUAL_VC_RESCALE1_SHIFT | W | RescaleDown1 移位量 |
| 1051 | [15] | DUAL_VC_RESCALE_MUL_INPUT_ZP | W | RescaleMul 输入零点（仅 Mode 0） |
| 1052 | [16] | DUAL_VC_RESCALE_MUL_MULTIPLIER | W | RescaleMul 乘数 |
| 1053 | [17] | DUAL_VC_RESCALE_MUL_OUTPUT_ZP | W | RescaleMul 输出零点 |
| 1054 | [18] | DUAL_VC_RESCALE_MUL_SHIFT | W | RescaleMul 移位量 |
| 1055 | [19] | DUAL_VC_START | W | 写 1 启动加速器 |
| 1056 | [20] | DUAL_VC_BUSY | R | 加速器忙标志（任一 VC 忙则为 1） |
| 1057 | [21] | DUAL_VC_PERFORMANCE_COUNTER | R | 性能计数器（两个 VC 中较大的周期数） |

---

## 4. hjson 配置参数详解

配置文件路径：`target/snitch_cluster/cfg/snax_dual_versacore_swiglu_cluster.hjson`

### 4.1 集群级别参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `name` | `snax_dual_versacore_swiglu_cluster` | 集群名称，决定生成的 wrapper 文件名 |
| `bender_target` | `[snax_dual_versacore_swiglu_cluster, snax_dual_versacore_swiglu, sparse_interconnect]` | Bender 编译目标列表 |
| `tcdm.size` | 256 | TCDM 大小（KB） |
| `tcdm.banks` | 64 | TCDM bank 数量 |
| `tcdm.sparse_interconnect` | true | 启用稀疏互联（减少面积） |
| `dma_data_width` | 512 | DMA 数据宽度（bit） |

### 4.2 加速器核心参数

| 参数 | 值 | 说明 | 约束 |
|------|-----|------|------|
| `snax_acc_name` | `snax_dual_versacore_swiglu` | 加速器名称 | 必须与 Scala Gen 匹配 |
| `snax_tcdm_ports` | 336 | TCDM 端口总数 | = 16(A) + 128(B0) + 128(B1) + 32(D0) + 32(D1) |
| `snax_num_rw_csr` | 20 | 读写 CSR 数量 | [0]~[18] 配置 + [19] 启动 |
| `snax_num_ro_csr` | 2 | 只读 CSR 数量 | [20] busy + [21] perf_counter |

### 4.3 VersaCore 阵列参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `snax_versacore_mac_num` | [1024] | MAC 单元数量 = meshRow × tileSize × meshCol |
| `snax_versacore_input_a_element_width` | [8] | A 输入元素宽度（bit），当前仅支持 8 |
| `snax_versacore_input_a_data_type` | [SInt] | A 输入数据类型 |
| `snax_versacore_input_b_element_width` | [8] | B 输入元素宽度（bit） |
| `snax_versacore_input_b_data_type` | [SInt] | B 输入数据类型 |
| `snax_versacore_input_c_element_width` | [32] | C（累加器）元素宽度 |
| `snax_versacore_output_d_element_width` | [32] | D（VersaCore 原始输出）元素宽度 |
| `snax_versacore_array_input_a_width` | 1024 | A 输入总宽度（bit）= meshRow × tileSize × a_width |
| `snax_versacore_array_input_b_width` | 8192 | B 输入总宽度（bit）= meshCol × tileSize × b_width × 128 |
| `snax_versacore_array_output_d_width` | 4096 | D 输出总宽度（bit）= meshRow × meshCol × 32 |
| `snax_versacore_serial_c_d_width` | 4096 | C/D 串行化宽度 |
| `snax_versacore_adder_tree_delay` | 0 | 加法树延迟（0 = 组合逻辑） |
| `snax_dual_versacore_postproc_lanes` | 64 | 后处理流水线宽度（并行处理的元素数） |

### 4.4 空间展开配置

```hjson
snax_versacore_spatial_unrolling: [
    [
        [16, 8, 8]    // [meshRow, tileSize, meshCol]
        [1, 32, 32]   // 每个维度的空间展开因子
    ]
]
```

含义：
- `[16, 8, 8]` = 硬件阵列维度：16 行 × 8 内积 × 8 列
- `[1, 32, 32]` = 空间展开因子（影响地址生成）

### 4.5 Granularity 参数

| 参数 | 值 | 说明 | 约束 |
|------|-----|------|------|
| `granularity_a` | 4 | A reader 地址对齐粒度 | 步长必须是 `granularity_a × bankWidth/8 = 32` 的倍数 |
| `granularity_b` | 8 | B reader 地址对齐粒度 | 步长必须是 `granularity_b × bankWidth/8 = 64` 的倍数 |
| `granularity_c_d` | 16 | D writer 地址对齐粒度 | 步长必须是 `granularity_c_d × bankWidth/8 = 128` 的倍数 |

### 4.6 Sparse Interconnect 配置

```hjson
sparse_interconnect_config: [
    [16, 4]     // Reader A: 16 通道，每组 4 bank
    [128, 8]    // Reader B0: 128 通道，每组 8 bank
    [128, 8]    // Reader B1: 128 通道，每组 8 bank
    [32, 8]     // Writer D0: 32 通道，每组 8 bank
    [32, 8]     // Writer D1: 32 通道，每组 8 bank
]
```

每一行 `[通道数, bank分组大小]`：
- 通道数 = 该 reader/writer 的 TCDM 端口数
- bank 分组大小 = 每个通道组访问的 bank 数量（影响互联面积）

**约束：** 所有通道数之和 = `snax_tcdm_ports = 336`

### 4.7 Streamer 配置

```hjson
snax_dual_versacore_swiglu_streamer_template: {
    data_reader_params: {
        spatial_bounds: [[16], [128], [128]]     // 3 个 reader 的空间边界
        temporal_dim: [6, 3, 3]                   // 时间循环维度数
        num_channel: [16, 128, 128]               // TCDM 通道数
        fifo_depth: [8, 8, 8]                     // FIFO 深度
        configurable_channel: [1, 1, 1]           // 可配置通道使能
        tcdm_logic_word_size: [                   // 支持的逻辑字宽
            [256, 128, 64],                       // Reader 0
            [256, 128, 64],                       // Reader 1
            [256, 128, 64]                        // Reader 2
        ]
    }
    data_writer_params: {
        spatial_bounds: [[32], [32]]              // 2 个 writer 的空间边界
        temporal_dim: [4, 4]                      // 时间循环维度数
        num_channel: [32, 32]                     // TCDM 通道数
        fifo_depth: [1, 1]
        configurable_channel: [1, 1]
        tcdm_logic_word_size: [
            [256, 128, 64],
            [256, 128, 64]
        ]
    }
    snax_library_name: dual-versacore-swiglu       // SW 库名称
}
```

**关键约束：**
- `num_channel` 必须与 `sparse_interconnect_config` 的通道数一致
- `spatial_bounds` 决定了空间步长 CSR 的数量（通常为 1 个）
- `temporal_dim` 决定了时间边界/步长 CSR 的数量

---

## 5. 参数约束关系

### 5.1 维度约束

对于 Block GEMM（M × K × N 个 tile）：
- **A 矩阵**形状：`(M, K, meshRow, tileSize)` = `(M, K, 16, 8)`
- **B 矩阵**形状：`(N, K, meshCol, tileSize)` = `(N, K, 8, 8)`
- **输出 D**形状：`(M, N, meshRow, meshCol)` = `(M, N, 16, 8)`
- A 数据量 = `M × K × meshRow × tileSize × a_width/8` 字节
- B 数据量 = `N × K × meshCol × tileSize × b_width/8` 字节
- D 数据量 = `M × N × meshRow × meshCol × out_elem_bits/8` 字节（out_elem_bits=16）

### 5.2 加速器 CSR 约束

```
ACCUM_BOUND = K        （累加次数等于 K 维度）
OUTPUT_BOUND = N × M   （输出块总数）
```

### 5.3 Streamer 步长对齐约束

所有时间步长必须满足对齐要求：

| Reader/Writer | 对齐要求（字节） | 计算公式 |
|--------------|----------------|---------|
| Reader A | 32 | `granularity_a × bankWidth/8 = 4 × 8` |
| Reader B0/B1 | 64 | `granularity_b × bankWidth/8 = 8 × 8` |
| Writer D0/D1 | 128 | `granularity_c_d × bankWidth/8 = 16 × 8` |

### 5.4 内存布局约束

所有基地址也必须按对应的对齐粒度对齐。使用 `align_wide_addr()` 函数：

```python
def align_wide_addr(addr, alignment):
    if addr % alignment:
        addr = ((addr // alignment) + 1) * alignment
    return addr
```

### 5.5 通道使能约束

- Reader A: `ceil(meshRow × tileSize × a_width / bankWidth)` 个通道
  - = `ceil(16 × 8 × 8 / 64)` = 16 通道 → 1 个 CSR（16 bit 使能）
- Reader B0/B1: 128 通道 → 4 个 CSR（128 bit 使能，但实际只用 8 个通道）
- Writer D0/D1: 32 通道 → 1 个 CSR（32 bit 使能）

---

## 6. Streamer 配置指南（Output Stationary）

Output Stationary 模式下，输出 tile 保持不动（在加速器内累加），输入按 K → N → M 循环读取。

### 6.1 Reader A 循环嵌套

```
for m in range(M):              # T_BOUND_2, T_STRIDE_2 = K * meshRow * tileSize * a_len/8
  for n in range(N):            # T_BOUND_1, T_STRIDE_1 = 0（A 不随 N 变化）
    for k in range(K):          # T_BOUND_0, T_STRIDE_0 = meshRow * tileSize * a_len/8
      read A[m][k]              # 每次读取一个 (meshRow × tileSize) tile
```

**公式：**
- `Atlstride0 = meshRow × tileSize × a_len / 8` = 16×8×8/8 = **128 字节**
- `Atlstride1 = 0`（A 不随 N 变化）
- `Atlstride2 = K × Atlstride0` = 2×128 = **256 字节**

### 6.2 Reader B 循环嵌套

```
for m in range(M):              # T_BOUND_2, T_STRIDE_2 = 0（B 不随 M 变化）
  for n in range(N):            # T_BOUND_1, T_STRIDE_1 = K * meshCol * tileSize * b_len/8
    for k in range(K):          # T_BOUND_0, T_STRIDE_0 = meshCol * tileSize * b_len/8
      read B[n][k]              # 每次读取一个 (meshCol × tileSize) tile
```

**公式：**
- `Btlstride0 = meshCol × tileSize × b_len / 8` = 8×8×8/8 = **64 字节**
- `Btlstride1 = K × Btlstride0` = 2×64 = **128 字节**
- `Btlstride2 = 0`（B 不随 M 变化）

### 6.3 Writer D 循环嵌套

```
for m in range(M):              # T_BOUND_2, T_STRIDE_2
  for n in range(N):            # T_BOUND_1, T_STRIDE_1
    for ser in range(1):        # T_BOUND_0（序列化，通常为 1）
      write D[m][n]             # 每次写入一个 (meshRow × meshCol) tile 的 int16
```

**公式：**
- `Dtlstride0 = d_spatial_bound × bankWidth/8` = 32×8 = **256 字节**（仅序列化时有效）
- `Dtlstride1 = meshRow × meshCol × out_elem_bits / 8` = 16×8×16/8 = **256 字节**
- `Dtlstride2 = N × Dtlstride1`

---

## 7. RescaleDown 参数说明

RescaleDown 模块将 int32 量化为 int16，公式：

```
zero_compensated = input - input_zp
multiplied = zero_compensated × multiplier        （64-bit 乘法）
if shift > 0:
    shifted = (multiplied + (1 << (shift-1))) >> shift   （带四舍五入的算术右移）
else:
    shifted = multiplied
result = clamp(shifted[31:0] + output_zp, -32768, 32767)
```

### 恒等（Identity / Pass-through）参数

用于调试，不改变数值（仅截断到 int16 范围）：

| 参数 | 值 | 效果 |
|------|-----|------|
| input_zp | 0 | 不减零点 |
| multiplier | 1 | 乘以 1 |
| output_zp | 0 | 不加零点 |
| shift | 0 | 不移位 |

### 量化参数示例

假设原始浮点值的 scale=0.015625，zero_point=0：
- `multiplier` = round(1/scale × 2^shift)
- 例：`shift=8`，`multiplier = round(64 × 256) = 16384`

---

## 8. 内存布局

### Mode 0 + Mode 1 端到端测试的内存布局

以 M=2, K=2, N=2, M1=2, K1=2, N1=1 为例：

```
偏移地址        大小        内容
────────────────────────────────────────────
0x0000          512 B      A 矩阵（Mode 0 输入，Mode 1 中被 cast 结果覆盖）
0x0200          256 B      W 矩阵（Mode 0 B0）
0x0300          256 B      V 矩阵（Mode 0 B1）
0x0400         1024 B      Mode 0 D0 输出（int16）
0x0800         1024 B      Mode 0 D1 dummy 输出
0x0C00          128 B      W2_left（Mode 1 B0）
0x0C80          128 B      W2_right（Mode 1 B1）
0x0D00          512 B      Mode 1 D0 输出（int16）
0x0F00          512 B      Mode 1 D1 输出（int16）
────────────────────────────────────────────
总计           ~4.25 KB    （TCDM 容量 256 KB）
```

**地址计算规则：**
1. 按 reader/writer 的 granularity 对齐
2. 使用 `align_wide_addr(addr, granularity × bankWidth/8)` 计算
3. 各数据区域不能重叠（除非有意覆盖，如 cast 结果覆盖 A）

---

## 9. 测试流程

### 端到端测试步骤

```
1. DMA 加载所有输入数据到 L1
   ├── A, W, V（Mode 0 用）
   └── W2_left, W2_right（Mode 1 用）

2. Mode 0（SwiGLU）
   ├── 配置 Streamer CSR（3 readers + 2 writers）
   ├── 配置加速器 CSR（K, N×M, mode=0, rescale=identity）
   ├── 启动 Streamer + 加速器
   ├── 等待完成（wait_dual_versacore + wait_dual_versacore_writer）
   └── 检查 D0 输出 vs 黄金参考

3. SW Cast（int16 → int8）
   └── 逐元素：clamp(local_d0[i], -128, 127) → local_a[i]

4. Mode 1（GEMM）
   ├── 重新配置 Streamer CSR（新的基地址、边界、步长）
   ├── 配置加速器 CSR（K1, N1×M1, mode=1）
   ├── 启动 Streamer + 加速器
   ├── 等待完成
   ├── 检查 D0 输出 vs 黄金参考
   └── 检查 D1 输出 vs 黄金参考

5. 打印结果和性能计数
```

### 编译和运行

```bash
# 在 barnard3 容器中：
source /pixi/entrypoint.sh
cd target/snitch_cluster

# 编译 SW
make -C sw/apps/snax-dual-versacore-swiglu-test \
     CFG_OVERRIDE=cfg/snax_dual_versacore_swiglu_cluster.hjson

# 运行仿真
bin/snitch_cluster.vlt \
    sw/apps/snax-dual-versacore-swiglu-test/build/snax-dual-versacore-swiglu-test.elf
```

### 预期输出

```
Mode 0 SwiGLU: PASS, Error: 0.
  Workload: M=2, N=2, K=2
  Accelerator cycles: 32, Streamer cycles: 46
Mode 1 GEMM D0: PASS, Error: 0.
Mode 1 GEMM D1: PASS, Error: 0.
  Workload: M1=2, N1=1, K1=2
  Accelerator cycles: 12, Streamer cycles: 18
```
