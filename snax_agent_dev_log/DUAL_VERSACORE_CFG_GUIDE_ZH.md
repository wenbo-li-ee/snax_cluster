# Dual VersaCore SwiGLU 配置参数详解

> 本文档详细介绍 `snax_dual_versacore_swiglu_cluster.hjson` 中所有参数的含义、作用以及它们之间的约束关系。  
> 适用于需要定制后处理计算单元数量、调整矩阵尺寸或修改流水线配置的场景。

---

## 一、架构概览

```
TCDM
 │
 ├── Reader 0 (A,  16 ch × 64bit = 1024bit) ─► [A 共享缓冲] ─►┐
 ├── Reader 1 (B0,128 ch × 64bit = 8192bit) ─────────────────►│ VersaCore 0
 └── Reader 2 (B1,128 ch × 64bit = 8192bit) ─────────────────►│ VersaCore 1
                                                                │
                    VersaCore 0 输出 (DataWidthD=4096 bit) ────►│
                    VersaCore 1 输出 (DataWidthD=4096 bit) ────►│
                                                                ▼
                                               [chunk 序列化] —→ shifter_6stage (>>2)
                                               [chunk 序列化] —→ shifter_2stage (>>2)
                                                                │
                                                 elem_adder_32b (64 lane 加法)
                                                                │
                                             [output 重组 4096 bit]
                                                                │
                                              Writer 0 (D, 64 ch × 64bit = 4096bit)
                                                                │
                                                             TCDM
```

**黄金模型：** `D = (A @ W >> 2) + (A @ V >> 2)`，其中 W=B0，V=B1，`>>` 为算术右移。

---

## 二、硬件核心参数（`snax_versacore_spatial_unrolling`）

### 2.1 空间展开参数（决定硬件面积和数据宽度）

```hjson
snax_versacore_spatial_unrolling:
[                       // 外层: data_type 维度（index）
    [                   // 中层: array_shape 维度（index）
        [               // 内层: 三个数字 [meshRow, tileSize, meshCol]
            16          // meshRow: GEMM 输出行方向展开度
            8           // tileSize: GEMM K 维度（规约方向）展开度
            8           // meshCol: GEMM 输出列方向展开度
        ]
        [           // 第二个 array_shape（若有多组）
            1
            32
            32
        ]
    ]
]
```

| 参数 | 当前值 | 含义 |
|------|--------|------|
| `meshRow` | 16 | 空间展开：输出矩阵行方向并行度 |
| `tileSize` | 8 | 空间展开：K（内积）维度并行度（脉动阵列深度） |
| `meshCol` | 8 | 空间展开：输出矩阵列方向并行度 |

**由此派生的关键width：**

| 派生参数 | 计算公式 | 当前值 |
|----------|----------|--------|
| `snax_versacore_mac_num` | meshRow × tileSize × meshCol | **1024** |
| `snax_versacore_array_input_a_width` | meshRow × tileSize × a\_elem\_width | **1024** (=16×8×8) |
| `snax_versacore_array_output_d_width` | meshRow × meshCol × d\_elem\_width | **4096** (=16×8×32) |
| `snax_versacore_serial_c_d_width` | meshRow × meshCol × c\_elem\_width | **4096** |
| `ElemsPerBeat` (自动计算) | DataWidthD / d\_elem\_width = meshRow × meshCol | **128** |

> ⚠️ 修改 meshRow/tileSize/meshCol 时，必须同步修改 `snax_versacore_mac_num` 和所有 `*_width` 参数。

---

## 三、后处理计算单元数量：`snax_dual_versacore_postproc_lanes`

### 3.1 参数说明

```hjson
snax_dual_versacore_postproc_lanes: 64
```

这是 **后处理流水线中并行运算的 lane 数**，直接决定：
- `shifter_6stage` 和 `shifter_2stage` 的输入位宽：`PostprocLanes × 32`
- `elem_adder_32b` 中的加法器个数：`PostprocLanes`
- 每次进入后处理单元处理的 32-bit 元素数量

### 3.2 与 chunk 序列化的关系

由于 VersaCore 每拍输出 `ElemsPerBeat = meshRow × meshCol = 128` 个 32-bit 元素，但后处理单元每拍只能处理 `PostprocLanes` 个元素，因此需要分 `NumChunks` 拍处理：

```
NumChunks = ceil(ElemsPerBeat / PostprocLanes)
           = ceil(128 / 64) = 2    ← 当前配置
```

**可选 PostprocLanes 值及对应性能：**

| PostprocLanes | NumChunks | 后处理延迟（拍数） | shifter/adder 面积 |
|---------------|-----------|------------------|---|
| 16 | 8 | 8× 慢 | 最小 |
| 32 | 4 | 4× 慢 | 较小 |
| **64** | **2** | **2×（当前）** | **中等** |
| 128 | 1 | 最快，无 chunk | 最大（2× 当前） |

> ✅ **PostprocLanes 必须整除 ElemsPerBeat（= meshRow × meshCol）**，以避免最后一个 chunk 有无效 padding 元素被写入输出。  
> 有效值（当前 meshRow×meshCol=128）：1, 2, 4, 8, 16, 32, **64**, 128

### 3.3 修改 PostprocLanes 时需联动修改的地方

修改 `snax_dual_versacore_postproc_lanes` 后，**还必须同步修改以下参数**：

```hjson
// 1. Writer D 的空间通道数 → 等于 PostprocLanes
data_writer_params:
{
    spatial_bounds: [[ PostprocLanes ]]   // 改这里
    num_channel:    [ PostprocLanes ]     // 改这里

// 2. TCDM 端口数 → A(16) + B0(128) + B1(128) + D(PostprocLanes)
snax_tcdm_ports: +(PostprocLanes 的变化量)  // 改这里

// 3. sparse_interconnect_config 最后一项
sparse_interconnect_config:
[
    [16,  4 ]    // A: 固定
    [128, 8 ]    // B0: 固定
    [128, 8 ]    // B1: 固定
    [PostprocLanes, ...]   // D: 改这里，倒数第二个数是 TCDM crossbar组数
]
```

---

## 四、数据类型参数

```hjson
snax_versacore_input_a_element_width:  [ 8  ]   // A: int8
snax_versacore_input_a_data_type:      [ SInt ]
snax_versacore_input_b_element_width:  [ 8  ]   // B: int8
snax_versacore_input_b_data_type:      [ SInt ]
snax_versacore_input_c_element_width:  [ 32 ]   // 累加器: int32
snax_versacore_input_c_data_type:      [ SInt ]
snax_versacore_output_d_element_width: [ 32 ]   // 输出 D: int32
snax_versacore_output_d_data_type:     [ SInt ]
```

- 外层数组索引对应 `data_type` 参数（软件 `params.hjson` 中的 `data_type`）
- 当前只定义了 `data_type=0`（int8×int8→int32）
- 若要加 fp16 支持，需在每个数组中追加第二个元素

---

## 五、数据总线宽度参数

```hjson
snax_versacore_array_input_a_width:  1024   // meshRow × tileSize × a_elem_width = 16×8×8
snax_versacore_array_input_b_width:  8192   // B 权重总线宽度（含16倍序列化展开）
snax_versacore_array_input_c_width:  4096   // 累加器宽度 = meshRow × meshCol × 32
snax_versacore_array_output_d_width: 4096   // 输出宽度   = meshRow × meshCol × 32
snax_versacore_serial_a_width:       1024   // 串行 A 总线（等于 array_input_a_width）
snax_versacore_serial_b_width:       8192   // 串行 B 总线（等于 array_input_b_width）
snax_versacore_serial_c_d_width:     4096   // 串行 C/D 总线
```

> ⚠️ 这些宽度必须与 `snax_versacore_spatial_unrolling` 中的维度严格一致，由 `SpatialArrayParamParser` 解析后传递给 Chisel。  
> `array_input_b_width = 8192` 对应 8192 / 64（bank宽）= **128 个 TCDM 通道**，与 streamer Reader 1/2 的 `num_channel=128` 一致。

---

## 六、Streamer 参数详解

Streamer 是连接 TCDM 和加速器的地址生成 + FIFO 单元。

### 6.1 Reader 0（矩阵 A，共享）

```hjson
spatial_bounds: [[ 16 ]]       // 每拍读 16 个 64-bit bank = 1024 bit = DataWidthA
num_channel:    [ 16 ]
temporal_dim:   [ 6 ]          // 软件最多配置 6 层时序循环（M, N, K, +3 余量）
fifo_depth:     [ 8 ]
```

### 6.2 Reader 1/2（矩阵 B0/B1，各一个 VersaCore 的权重）

```hjson
spatial_bounds: [[ 128 ]]      // 每拍读 128 个 64-bit bank = 8192 bit = DataWidthB
num_channel:    [ 128 ]
temporal_dim:   [ 3 ]          // K, N, M 三层
fifo_depth:     [ 8 ]
```

### 6.3 Writer 0（输出 D）

```hjson
spatial_bounds: [[ 64 ]]       // 每拍写 64 个 64-bit bank = 4096 bit = DataWidthD
num_channel:    [ 64 ]         // = PostprocLanes
temporal_dim:   [ 4 ]          // chunk, N, M + 余量
fifo_depth:     [ 1 ]
```

---

## 七、地址粒度参数

```hjson
granularity_a:   4    // A 的地址步长必须是 4×(bankWidth/8) = 4×8 = 32 字节的倍数
granularity_b:   8    // B 的地址步长必须是 8×8 = 64 字节的倍数
granularity_c_d: 16   // D 的地址步长必须是 16×8 = 128 字节的倍数
```

这些是 `datagen.py` 中 `assert` 检查的约束。如果修改了 meshRow/meshCol/等，对应的 granularity 值可能需调整。

---

## 八、计算延迟参数

```hjson
snax_versacore_adder_tree_delay: 0
```

VersaCore 内部加法树的额外流水线延迟（时钟周期数）。设为 0 表示组合逻辑加法树。若时序收敛困难可调大，但需同时修改 VersaCore 内部流水线相关代码。

---

## 九、CSR 接口参数

```hjson
snax_num_rw_csr: 7    // 7 个读写 CSR（过写累加、累加边界、输出边界等）
snax_num_ro_csr: 2    // 2 个只读 CSR （busy 状态寄存器）
```

当前 7 个 RW CSR 对应：
1. OVERWRITE_ACCUM
2. ACCUM_BOUND
3. OUTPUT_BOUND
4. SUBTRACTIONS（右移量，当前固定为 2）
5. ARRAY_SHAPE
6. DATA_TYPE
7. START（写 1 启动加速器）

只读 CSR：
1. BUSY（bit0 = VC0 or VC1 busy）
2. PERF_COUNTER

---

## 十、软件测试参数（`params.hjson`）

```hjson
{
    array_shape: 0       // 索引 snax_versacore_spatial_unrolling[data_type][array_shape]
    data_type:   0       // 当前只有 0（int8→int32）
    K: 2                 // K 维度 tile 数（内积维度）
    N: 1                 // N 维度 tile 数（输出列方向）
    M: 20                // M 维度 tile 数（输出行方向）
    channel_en_C: 0      // 必须为 0（不使用累加器 C 输入）
    stationary: 0        // 必须为 0（仅支持 output stationary）
    transposed_A: 0      // 必须为 0（不支持转置）
    transposed_B: 0      // 必须为 0（不支持转置）
}
```

### 10.1 M、K、N 与实际矩阵大小的对应关系

| 参数 | 实际矩阵维度 | 公式 |
|------|-------------|------|
| M | A/D 的行数 | M × meshRow = M × 16 |
| K | A 的列数，B 的行数 | K × tileSize = K × 8 |
| N | B/D 的列数 | N × meshCol = N × 8 |

**示例（当前 M=20, K=2, N=1）：**
- A 矩阵：320 行 × 16 列（in8）
- W/V 矩阵：16 行 × 8 列（int8）
- 输出 D：320 行 × 8 列（int32）

---

## 十一、参数约束汇总表

| 约束 | 条件 | 说明 |
|------|------|------|
| **MAC 数一致** | `snax_versacore_mac_num` == meshRow × tileSize × meshCol | 1024 = 16×8×8 |
| **A 总线宽度** | `array_input_a_width` == meshRow × tileSize × a\_elem\_width | 1024 = 16×8×8 |
| **D 总线宽度** | `array_output_d_width` == meshRow × meshCol × d\_elem\_width | 4096 = 16×8×32 |
| **PostprocLanes 整除** | ElemsPerBeat % PostprocLanes == 0 | 128 % 64 = 0 ✓ |
| **Writer 通道** | D writer `num_channel` == PostprocLanes | 64 = 64 ✓ |
| **Writer 空间边界** | D writer `spatial_bounds` == PostprocLanes | 64 = 64 ✓ |
| **TCDM 端口数** | `snax_tcdm_ports` == A\_ch + B0\_ch + B1\_ch + D\_ch | 16+128+128+64=336 ✓ |
| **Reader A 通道** | A `num_channel` == `array_input_a_width` / bankWidth | 1024/64=16 ✓ |
| **Reader B 通道** | B `num_channel` == `array_input_b_width` / bankWidth | 8192/64=128 ✓ |
| **stationary** | 必须为 0 | 只支持 output stationary |
| **channel\_en\_C** | 必须为 0 | SwiGLU 无 C 累加器输入 |
| **地址对齐 A** | K × tileSize × meshRow × a\_len/8 % (granularity\_a × 8) == 0 | `datagen.py` assert |
| **地址对齐 B** | K × tileSize × meshCol × b\_len/8 % (granularity\_b × 8) == 0 | `datagen.py` assert |

---

## 十二、仿真实测：K=2, N=1, M=20

```
Dual VersaCore SwiGLU: PASS, Error: 0.
Workload: M=20, N=1, K=2, meshRow=16, tileSize=8, meshCol=8
Accelerator cycles: 84
Streamer cycles:    95
EXIT_CODE: 0
```

### 与 M=4, K=4, N=2 的对比

| 配置 | 总 tile 数（M×K×N） | 加速器周期 | Streamer 周期 | 备注 |
|------|---------------------|-----------|--------------|------|
| M=4, K=4, N=2 | 32 | 69 | 80 | 均匀负载 |
| M=20, K=2, N=1 | 40 | 84 | 95 | M 增大，N/K 减小 |

- 每 tile 加速器周期约 **2.1 cycles/tile**
- Streamer 始终略慢于 Compute（约多 10 个周期），说明当前瓶颈在 Streamer 一侧

---

## 十三、快速修改参考

### 场景 A：增大后处理带宽（PostprocLanes 64 → 128）

1. `snax_dual_versacore_postproc_lanes: 128`
2. Writer D: `spatial_bounds: [[128]]`, `num_channel: [128]`
3. `snax_tcdm_ports: 400`（16+128+128+**128**）
4. `sparse_interconnect_config` 最后一项改为 `[128, 16]`（或其他合适的 group 数）
5. 重新 `make CFG_OVERRIDE=... rtl-gen bin/snitch_cluster.vlt`

### 场景 B：更换阵列尺寸（例如 meshRow=32, tileSize=8, meshCol=16）

需同步修改：
```hjson
snax_versacore_mac_num: 4096             // 32×8×16
snax_versacore_array_input_a_width: 2048  // 32×8×8
snax_versacore_array_input_b_width: ...   // 需按 VersaCore 参数重新计算
snax_versacore_array_output_d_width: 16384 // 32×16×32
snax_versacore_serial_c_d_width: 16384
snax_versacore_serial_a_width: 2048
// Reader A num_channel = 2048/64 = 32
// ElemsPerBeat = 32×16 = 512
// PostprocLanes 需可以整除 512（如 128）
// Writer D num_channel = PostprocLanes
// snax_tcdm_ports = 32 + 128 + 128 + PostprocLanes（B 通道数也需重新计算）
```
