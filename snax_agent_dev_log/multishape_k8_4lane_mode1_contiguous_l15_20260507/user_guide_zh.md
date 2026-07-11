# K8 8x4 4-lane L15 Mode1 Padded-Contiguous App 用户指南

日期：2026-05-07

## 1. 这个 app 解决的问题

目标 workload 是：

```text
(Swish(xW) * xV) W2
```

因为 dual VersaCore 的结构，Mode1 的 `W2` 在硬件输入侧被拆成左右两半：

```text
W2_left  -> B0 reader -> writer0 输出左半 1024 个 int16
W2_right -> B1 reader -> writer1 输出右半 1024 个 int16
```

但是从算法语义看，`W2_left` 和 `W2_right` 不是两个独立矩阵，而是同一个大矩阵 `W2` 的左右两部分。因此同一个 token 的最终输出应该是一行 2048 个 int16：

```text
[left 1024 int16][right 1024 int16]
```

这次进一步要求：Mode1 输出也要和输入 A 一样，在 token 之间带相同 padding，使输入输出格式一致。输入 A 每个 token 行格式是：

```text
2048 int16 payload + 32 bytes padding
```

所以 Mode1 输出最终采用：

```text
2048 int16 payload + 32 bytes padding
```

也就是每个 token：

```text
[left 1024 int16][right 1024 int16][16 int16 padding]
```

关键地址约束是：

```text
D1 base - D0 base = 1024 * 2 = 2048 bytes
Mode1 token row stride = A row stride = 4128 bytes
```

这保证同一个 token 内左右半连续，token 与 token 之间留出和输入 A 一样的 32B padding。

## 2. 文件位置

新 app 路径：

```text
snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-multishape-k8-8x4-mode1-contiguous-l15
```

固定 cfg：

```text
cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson
```

主要文件：

| 文件 | 作用 |
|---|---|
| `Makefile` | app 编译入口，支持 `SELECT_SHAPE`、`RUN_MODE1`、`FAST_BUILD` |
| `data/Makefile` | 调用 datagen.py 生成 data.h |
| `data/params.hjson` | workload 参数：M/K/N 等 |
| `data/datagen.py` | 生成 A/W/V/W2 数据、golden、streamer 配置、TCDM placement |
| `data/data.h` | datagen 输出的 C header，包含所有静态数组和 shape 配置 |
| `src/snax-versacore-int16x4-multishape-k8-8x4-mode1-contiguous-l15.c` | runtime harness：搬数据、配置 streamer/accelerator、启动、等待、比较 |

## 3. workload 参数

当前 datagen 检查这些固定参数：

```text
M_total  = 8
K0_total = 2048
N0_total = 1408
K1_total = 1408
N1_total = 1024
```

含义：

| 参数 | 含义 |
|---|---|
| `M_total=8` | 最多处理 8 个 token |
| `K0_total=2048` | Mode0 的输入 hidden size，也就是 x 的长度 |
| `N0_total=1408` | Mode0 输出 hidden size，也是 Mode1 的 K1 |
| `K1_total=1408` | Mode1 输入 K，来自 Mode0 的 SwiGLU 输出 |
| `N1_total=1024` | 每个 writer 的输出宽度，左右各 1024 |

Mode1 逻辑总输出宽度是：

```text
N1_total * 2 = 2048 int16
```

带 padding 后每行实际占用：

```text
2048 int16 payload = 4096 bytes
padding = 32 bytes
row stride = 4128 bytes
```

## 4. 三个 shape

这个 cfg 下使用三种 array shape：

| Shape | array_shape | meshRow | tileSize | meshCol | 使用 token 数 |
|---|---:|---:|---:|---:|---:|
| S0 | 0 | 8 | 8 | 4 | 8 |
| S1 | 1 | 4 | 8 | 8 | 4 |
| S2 | 2 | 2 | 8 | 16 | 2 |

因为 `M_tiles=1`，每次 simulation 只跑一组 tile，但不同 shape 的 `meshRow` 不同，所以每次实际覆盖的 token 数也不同。

## 5. L15 TCDM layout

本 app 固定使用上一轮探索得到的 L15 layout：

```text
A pad = 32 bytes
B1 color = 272 bytes
W2_left color = 128 bytes
Mode1 D0 color = 256 bytes
```

datagen 生成的主要 placement 是：

| Tensor | Offset | Bank 说明 |
|---|---:|---|
| A | 0 | 输入 A 从 TCDM 起始处开始 |
| B0/W | 33792 | bank 0 |
| B1/V | 1475856 | bank 34 |
| Mode0 D0 | 2918400 | bank 0 |
| W2_left | 2941056 | W2 左半 |
| W2_right | 3662848 | W2 右半 |
| Mode1 D0 | 4384000 | bank 32 |
| Mode1 D1 | 4386048 | `D0 + 2048` |

最终 TCDM end 是：

```text
4417024 bytes
```

比无 padding 版本多 256B，因为 S0 最多 8 个 token，每个 token 多 32B padding：

```text
8 * 32 = 256 bytes
```

## 6. data 生成逻辑

`datagen.py` 生成数据的顺序如下。

### 6.1 生成 logical A

函数：

```text
make_logical_a(m_total, k_total)
```

生成一个二维 int16 数组：

```text
A[m, k] = ((m * 5 + k * 3) % 11) - 5
```

这样 A 的值范围小、确定、可复现，方便 golden 验证。

### 6.2 生成 padded A

函数：

```text
make_padded_a(logical_a, row_stride_bytes)
```

当前：

```text
row_stride_bytes = K0_total * 2 + a_pad = 2048 * 2 + 32 = 4128
row_elems = 4128 / 2 = 2064
```

每个 token 的 A 行是：

```text
前 2048 个 int16 = logical A payload
后 16 个 int16 = 0 padding
```

最终 `A_row_stride_4128` 是按 token 顺序 flatten 的一维数组。

### 6.3 生成 W/V

Mode0 需要两个 int4 权重流：

```text
W -> B0 reader
V -> B1 reader
```

datagen 使用 `packed_int4_constant` 生成常量 int4 packed bytes：

```text
W = 全部 int4 值 1
V = 全部 int4 值 2
```

每个 byte 存两个 int4，所以常量 1 被打包为：

```text
0x11
```

常量 2 被打包为：

```text
0x22
```

### 6.4 生成 W2_left/W2_right

Mode1 的 W2 被拆成左右两半：

```text
W2_left  = 全部 int4 值 1
W2_right = 全部 int4 值 2
```

这两个数组分别送到 B0/B1 reader。它们语义上是同一个 `W2` 的左右半边，但硬件上走两个 B reader。

### 6.5 生成 Mode0 golden

Mode0 逻辑是：

```text
Swish(xW) * xV
```

datagen 中对应流程：

1. 用 `streamer_i16_flat` 模拟 Mode0 A reader 从 padded A 中按当前 shape 读出 A。
2. 用 `block_gemm_int16x4` 计算 `xW`。
3. 用 `block_gemm_int16x4` 计算 `xV`。
4. 对 `xW` 做 `rescale_down_32to16`。
5. 对 `xW` 的 int16 结果做 SiLU/Swish：`apply_silu_vectorized`。
6. 对 `xV` 做 `rescale_down_32to16`。
7. 两路相乘，再做一次 `rescale_down_32to16`。

最终得到 `S*_mode0_d0_golden`。

Mode0 golden 的物理顺序与 writer0 写 D0 的顺序一致，也就是后续 C 程序可以直接从 `delta_local_d0` 开始按一维连续数组比较。

### 6.6 生成 Mode1 golden

Mode1 逻辑是：

```text
Mode0_D0 * W2
```

其中 `W2` 被硬件拆成 `W2_left` 和 `W2_right`，因此 datagen 分别计算：

```text
mode1_d0 = Mode0_D0 * W2_left
mode1_d1 = Mode0_D0 * W2_right
```

然后把硬件输出顺序转换成 per-token 顺序：

```text
mode1_d0.reshape(M_tiles, N1_tiles, meshRow, meshCol)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
```

这一步的目的：writer 原始输出按 N tile 组织，而最终希望每个 token 的 1024 个元素连续排列。

之后将左右半按 token 拼接：

```text
for each token:
    combined[token] = [left1024, right1024]
```

最后生成 padded golden：

```text
row_elems = 4128 / 2 = 2064
payload_elems = 2048
padding_elems = 16
```

每个 token 的 golden 行是：

```text
[combined 2048 int16][16 个 0]
```

这个数组名是：

```text
S0_mode1_padded_golden
S1_mode1_padded_golden
S2_mode1_padded_golden
```

C 程序会比较整个 padded row，包括 padding 区。因此如果 writer 错误覆盖 padding，比较也会失败。

## 7. streamer 配置总览

每个 shape 的 streamer 配置都在 `data.h` 的 `shape_cfg_t` 中。字段分为：

| 字段 | 含义 |
|---|---|
| `*_sstride` | spatial stride，描述同一个 tile 内不同 spatial lane 的 byte offset |
| `*_tbound` | temporal loop bound |
| `*_tstride` | temporal loop stride，单位是 byte |
| `*_channel_en` | 启用哪些 spatial channel |
| `delta_local_*` | TCDM base offset |

streamer 地址一般可以理解为：

```text
address = base + spatial_offset + sum(loop_index[i] * tstride[i])
```

其中 spatial offset 由 `sstride` 和 `channel_en` 决定。

## 8. Mode0 streamer 配置

Mode0 做：

```text
A * W, A * V, 然后 Swish(xW) * xV
```

### 8.1 Mode0 A reader

A 是 padded per-token 行格式：

```text
mode0_A_sstride = {8, 4128}
```

两个 spatial 维度含义可以看成：

```text
第 0 spatial 维：同一 token 内每 8B 跨一个 bank beat
第 1 spatial 维：跨 token，stride 是一整行 4128B
```

Mode0 A temporal 配置：

| Shape | tbound | tstride |
|---|---|---|
| S0 | `{256, 352, 1, 1, 1, 1}` | `{16, 0, 33024, 0, 0, 0}` |
| S1 | `{256, 176, 1, 1, 1, 1}` | `{16, 0, 16512, 0, 0, 0}` |
| S2 | `{256, 88, 1, 1, 1, 1}` | `{16, 0, 8256, 0, 0, 0}` |

解释：

- 第 0 维遍历 K tile，`K0_total / tileSize = 2048 / 8 = 256`。
- 第 1 维遍历 N tile，但 A 与 N 无关，所以 stride 是 0，会重复使用同一份 A。
- 第 2 维是 M tile，本 app 中 `M_tiles=1`。
- `mode0_A_tstride[0]=16`，因为每个 K tile 是 8 个 int16，即 16B。
- `mode0_A_tstride[2]=meshRow * 4128`，跨一个 M tile 会跳过当前 shape 使用的所有 token 行。

Mode0 A channel enable：

| Shape | channel_en | 含义 |
|---|---:|---|
| S0 | `0xFFFF` | 8x8x4 shape 使用 16 个 spatial offsets |
| S1 | `0x00FF` | 4x8x8 shape 使用 8 个 spatial offsets |
| S2 | `0x000F` | 2x8x16 shape 使用 4 个 spatial offsets |

### 8.2 Mode0 B0/B1 reader

B0 读 W，B1 读 V。两者配置相同，只是 base 不同。

```text
mode0_B_sstride = {8, 4096}
mode0_B_tstride[0] = 16
```

`mode0_B_tstride[1]` 根据 meshCol 不同：

| Shape | N_tiles | B tstride |
|---|---:|---|
| S0 | 352 | `{16, 4096, 0, 0}` |
| S1 | 176 | `{16, 8192, 0, 0}` |
| S2 | 88 | `{16, 16384, 0, 0}` |

解释：

- 第 0 维遍历 K tile。
- 第 1 维遍历 N tile。
- 权重数据按 S0 physical tile layout 存储，所以不同 meshCol 读取同一份 packed weight 时需要不同 N stride。
- B channel enable 也随 shape 改变：S0=`0x03`，S1=`0x0F`，S2=`0xFF`。

### 8.3 Mode0 D writer

Mode0 只使用 writer0，writer1 在 helper 中被 idle。

```text
D_sstride = {8}
mode0_D_tbound = {8, N_tiles, 1, 1}
mode0_D_tstride = {8, 64, mode0_d_m_stride, 0}
```

各 shape：

| Shape | tbound | tstride |
|---|---|---|
| S0 | `{8, 352, 1, 1}` | `{8, 64, 22528, 0}` |
| S1 | `{8, 176, 1, 1}` | `{8, 64, 11264, 0}` |
| S2 | `{8, 88, 1, 1}` | `{8, 64, 5632, 0}` |

解释：

- 第 0 维是 writer beat 内的 8 个输出位置。
- 第 1 维遍历 N tile。
- `64B` 是固定 writer beat row stride。
- `mode0_d_m_stride = N_tiles * 64`。

Mode0 D0 的输出宽度是 `N0_total=1408`，每个 shape 的 token 数不同，所以 output elements 是：

| Shape | tokens | elems |
|---|---:|---:|
| S0 | 8 | 11264 |
| S1 | 4 | 5632 |
| S2 | 2 | 2816 |

## 9. Mode1 streamer 配置

Mode1 做：

```text
Mode0_D0 * W2_left  -> writer0 -> left1024
Mode0_D0 * W2_right -> writer1 -> right1024
```

### 9.1 Mode1 A reader

Mode1 的 A reader 读的是 Mode0 D0 输出，不是原始输入 A。

各 shape 的 spatial stride：

| Shape | mode1_A_sstride |
|---|---|
| S0 | `{64, 8}` |
| S1 | `{8, 16}` |
| S2 | `{8, 32}` |

各 shape 的 temporal 配置：

| Shape | tbound | tstride |
|---|---|---|
| S0 | `{176, 256, 1, 1, 1, 1}` | `{128, 0, 22528, 0, 0, 0}` |
| S1 | `{176, 128, 1, 1, 1, 1}` | `{64, 0, 11264, 0, 0, 0}` |
| S2 | `{176, 64, 1, 1, 1, 1}` | `{16, 0, 5632, 0, 0, 0}` |

解释：

- `K1 = 1408 / 8 = 176`。
- `N1 = 1024 / meshCol`，S0/S1/S2 分别是 256/128/64。
- Mode1 A 来自 Mode0 D0，Mode0 D0 的物理 layout 和原始 A 不同，所以 `mode1_A_sstride` 和 `mode1_A_tstride[0]` 需要按 shape 特别设置。
- 第 1 维遍历 N1 tile，但 A 与 N1 无关，所以 stride 是 0。
- 第 2 维跨 M tile，stride 是 Mode0 D0 的一个 M tile 大小。

### 9.2 Mode1 B0/B1 reader

B0 读 `W2_left`，B1 读 `W2_right`。

```text
mode1_B_sstride = {8, 2816}
mode1_B_tstride[0] = 16
```

各 shape：

| Shape | N1 | mode1_B_tstride |
|---|---:|---|
| S0 | 256 | `{16, 2816, 0, 0}` |
| S1 | 128 | `{16, 5632, 0, 0}` |
| S2 | 64 | `{16, 11264, 0, 0}` |

解释：

- `2816B = K1_s0_tiles * 16 = 176 * 16`，是 S0 physical tile layout 下一个 N tile 的 K dimension 存储跨度。
- S1/S2 因为 meshCol 更大，一个 N tile 覆盖更多列，所以 N stride 对应放大。
- B0 和 B1 的配置完全相同，只有 base 分别指向 `W2_left` 和 `W2_right`。

### 9.3 Mode1 D writer：本 app 的核心

Mode1 writer 是本次 app 的关键。

要求：同 token 的 2048 个 int16 连续，token 之间有和输入 A 相同的 padding。

最终配置：

```text
D0 base = delta_local_mode1_d0 = 4384000
D1 base = delta_local_mode1_d1 = 4386048
D1 - D0 = 2048 bytes
D_sstride = {8}
mode1_D_tstride[1] = 4128
```

各 shape：

| Shape | mode1_D_tbound | mode1_D_tstride |
|---|---|---|
| S0 | `{1, 8, 256, 1}` | `{8, 4128, 8, 0}` |
| S1 | `{2, 4, 128, 1}` | `{8, 4128, 16, 0}` |
| S2 | `{4, 2, 64, 1}` | `{8, 4128, 32, 0}` |

维度解释：

- 第 0 维：`beats_per_row = meshCol / 4`。因为 writer 每个 beat 写 4 个 int16，对应 8B，所以 meshCol 越大，一个 N tile 内需要的 beat 越多。
- 第 1 维：token 维，bound 是 `meshRow`。这是最关键的维度，stride 设置成 `4128B`，与输入 A 的行 stride 完全一致。
- 第 2 维：N tile 维，stride 是 `meshCol * 2` bytes。因为每个 N tile 沿输出列方向推进 `meshCol` 个 int16。
- 第 3 维：M tile，本 app 是 1。

writer0 和 writer1 使用同一套 D tbound/tstride，只是 base 不同：

```text
writer0 base = D0
writer1 base = D0 + 2048
```

所以 token i 的写入地址是：

```text
writer0: D0 + i * 4128 + n_offset
writer1: D0 + 2048 + i * 4128 + n_offset
```

因此每个 token 形成：

```text
D0 + i*4128 + 0..2047 bytes      -> left 1024 int16
D0 + i*4128 + 2048..4095 bytes   -> right 1024 int16
D0 + i*4128 + 4096..4127 bytes   -> padding
```

padding 区没有 writer 会写入。C 程序在 Mode1 前先把整个 output region 清零，然后比较 padded golden，因此 padding 必须保持 0 才能 PASS。

## 10. C 程序执行逻辑

C 文件入口是：

```text
src/snax-versacore-int16x4-multishape-k8-8x4-mode1-contiguous-l15.c
```

### 10.1 main

`main()` 做这些事：

1. 打印 app banner。
2. 检查 `SELECT_LAYOUT` 是否合法。
3. 检查 `SELECT_SHAPE` 是否合法。
4. 调用 `stage_layout()` 把数据 DMA 到 TCDM。
5. 根据 `SELECT_SHAPE` 决定跑一个 shape 还是三个 shape。
6. 对每个 shape 调用 `run_shape()`。
7. 打印 total checks 和 total error。

常用宏：

| 宏 | 作用 |
|---|---|
| `SELECT_LAYOUT` | 本 app 只有 layout 0，即 L15 |
| `SELECT_SHAPE` | `-1` 跑全部，`0/1/2` 只跑对应 shape |
| `RUN_MODE1` | 是否运行 Mode1 |
| `FAST_BUILD` | Makefile 中用于 `-O1 -g0` 快速编译 |

### 10.2 stage_layout

`stage_layout()` 负责把静态数据搬到 TCDM：

```text
A         -> delta_local_a
W         -> delta_local_b0
V         -> delta_local_b1
W2_left   -> delta_local_w2l
W2_right  -> delta_local_w2r
```

它还会检查：

```text
tcdm_end <= TCDM_CAPACITY_BYTES
```

并打印 sanity 信息：

```text
A_stride=4128
M1D1_minus_D0=2048
M1_row_stride=4128
```

这些打印是验证本 app 地址格式最直接的证据。

### 10.3 run_shape 的 Mode0 阶段

Mode0 流程：

1. 配置 streamer：`set_dual_versacore_streamer_csr_d0_only(...)`。
2. 配置 accelerator：`set_dual_versacore_csr(...)`。
3. 设置 mode 为 0：`set_dual_versacore_mode(0)`。
4. 设置 rescale 参数。
5. 先启动 streamer，再启动 accelerator。
6. 等 accelerator busy 清零。
7. 等 streamer/writer busy 清零。
8. 读取 performance counter。
9. 比较 `local_d0` 和 `mode0_d0_golden`。

Mode0 只启用 writer0。helper 会显式让 writer1 idle，避免 writer1 保持 busy。

### 10.4 run_shape 的 Mode1 阶段

Mode1 流程：

1. 将 Mode1 padded output region 清零：

```text
for i in mode1_padded_output_elems:
    local_mode1_d[i] = 0
```

2. 配置 streamer：`set_dual_versacore_streamer_csr(...)`。
3. A reader base 指向 Mode0 D0。
4. B0 reader base 指向 W2_left。
5. B1 reader base 指向 W2_right。
6. writer0 base 指向 Mode1 D0。
7. writer1 base 指向 Mode1 D1，也就是 D0 + 2048。
8. writer0/writer1 都使用同一套 padded D tstride。
9. 配置 accelerator mode 为 1。
10. 启动 streamer 和 accelerator。
11. 等 accelerator 和 streamer 结束。
12. 比较 `local_mode1_d` 和 `mode1_padded_golden`。

比较长度是：

| Shape | tokens | 每行 int16 | 比较元素数 |
|---|---:|---:|---:|
| S0 | 8 | 2064 | 16512 |
| S1 | 4 | 2064 | 8256 |
| S2 | 2 | 2064 | 4128 |

这比只比较 payload 更严格，因为 padding 也被比较。

## 11. 结果比较方式

比较函数是：

```text
check_result_i16_limited(output, output_golden, num_elements)
```

它逐元素比较 int16：

```text
if output[i] != output_golden[i]: err++
```

最多打印前 16 个 mismatch，避免 log 太大。

Mode0 比较：

```text
local_d0 vs mode0_d0_golden
num_elements = meshRow * N0_total
```

Mode1 padded 比较：

```text
local_mode1_d vs mode1_padded_golden
num_elements = meshRow * 2064
```

Mode1 golden 的每行最后 16 个 int16 是 0。由于 C 程序在 Mode1 前清零输出区，如果 writer 正确不碰 padding 区，则 padding 比较通过；如果 writer stride 配错或 writer 覆盖 padding，则会出现 mismatch。

## 12. 构建和仿真

使用的构建/仿真模式是每个 shape 单独编译一个 ELF：

```bash
APP=snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-multishape-k8-8x4-mode1-contiguous-l15
CFG=cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson
SIM=snax_cluster/target/snitch_cluster/bin/snitch_cluster.vlt
LOGDIR=snax_cluster/layout_explore_logs/mode1_padded_contiguous_l15_20260507
ELF_BASE=$PWD/$APP/build/snax-versacore-int16x4-multishape-k8-8x4-mode1-contiguous-l15.elf
```

构建单个 shape：

```bash
make -C $APP "$ELF_BASE" \
  CFG_OVERRIDE=$CFG \
  SELECT_LAYOUT=0 \
  SELECT_SHAPE=$S \
  RUN_MODE1=1 \
  FAST_BUILD=1
```

运行：

```bash
timeout 600 $SIM $LOGDIR/elf_S${S}.elf > $LOGDIR/S${S}.log 2>&1
```

本次最终 log 目录：

```text
snax_cluster/layout_explore_logs/mode1_padded_contiguous_l15_20260507
```

## 13. 最终验证结果

三形状全部 exit 0，且 `total error: 0`。

| Shape | D1-D0 | Mode1 row stride | Mode0 streamer | Mode1 streamer | 结果 |
|---|---:|---:|---:|---:|---|
| S0 | 2048 B | 4128 B | 90141 | 46613 | PASS |
| S1 | 2048 B | 4128 B | 45110 | 23606 | PASS |
| S2 | 2048 B | 4128 B | 22573 | 11342 | PASS |

详细 cycle：

| Shape | Mode | accel | streamer | wall | check |
|---|---|---:|---:|---:|---|
| S0 | Mode0 | 90117 | 90141 | 100718 | D0 PASS |
| S0 | Mode1 | 46594 | 46613 | 106914 | padded-contiguous D PASS |
| S1 | Mode0 | 45086 | 45110 | 55712 | D0 PASS |
| S1 | Mode1 | 23574 | 23606 | 59115 | padded-contiguous D PASS |
| S2 | Mode0 | 22549 | 22573 | 33206 | D0 PASS |
| S2 | Mode1 | 11310 | 11342 | 34421 | padded-contiguous D PASS |

## 14. 和前两个版本的对比

| Shape | 旧 L15 split-base | 连续无 padding | 连续且带 A-style padding |
|---|---:|---:|---:|
| S0 | 46539 | 47330 | 46613 |
| S1 | 23558 | 23484 | 23606 |
| S2 | 11305 | 11331 | 11342 |

解释：

- split-base 版本不是最终语义，因为两个 writer 的输出不是同 token 连续 2048 payload。
- 连续无 padding 版本满足同 token `[left1024,right1024]`，但 token 间没有 A-style padding。
- 当前 padded-contiguous 版本同时满足两个条件：同 token 左右半连续，token 间 stride 和输入 A 一样。

## 15. 最重要的结论

这个 app 的正确 Mode1 writer streamer 配置不是简单把 D1 base 设置成 D0+2048，也不是简单保持旧 token stride。正确配置是二者同时成立：

```text
D1 base = D0 base + 2048
D token stride = 4128
```

这样地址才是：

```text
token0: D0 + 0      left, D0 + 2048 right, D0 + 4096 padding
token1: D0 + 4128   left, D0 + 6176 right, D0 + 8224 padding
token2: D0 + 8256   left, D0 + 10304 right, ...
```

这与输入 A 的行间格式完全一致：每个 token 一行，每行 4128B，其中前 4096B 是有效 payload，最后 32B 是 padding。

## 16. A 的 padding 字节有没有被 streamer 读给加速器？

**结论：没有。** A reader streamer 的设计保证了每次访问都落在每行 4096B 的数据区内，永远不触碰 `[4096, 4128)` 这 32B padding 区。

### 16.1 A 行格式回顾

每个 token 的 A 行在 TCDM 中的布局：

```text
字节偏移:  [0 ........ 4095] [4096 .. 4127]
内容:       2048 int16 data    32 bytes padding (全 0)
```

行 stride = 4128B，但有效数据只占前 4096B（= 2048 int16 × 2 bytes）。

### 16.2 A reader 地址生成机制

Mode0 A reader 访问地址由两部分叠加：

```
addr = delta_local_a + temporal_offset + spatial_offset
```

**Temporal 部分**（随 K tile 推进）：

```text
temporal_offset = k_tile_idx * tileSize * 2
               = k_tile_idx * 8 * 2
               = k_tile_idx * 16
```

`k_tile_idx` 范围是 `[0, K0_total/tileSize - 1] = [0, 255]`。

**Spatial 部分**（由 sstride 和 spatial bounds 决定）：

```text
mode0_A_sstride = {8, 4128}
spatial bounds  = [2, 8]   (S0: 16 channels; S1: [2,4] 8 channels; S2: [2,2] 4 channels)
spatial_offset(i, j) = i * sstride[0] + j * sstride[1]
                     = i * 8 + j * 4128
```

其中 `i ∈ {0, 1}`，`j ∈ {0, .., meshRow/2 - 1}`。

`j` 维度的作用是**跨 token**：`j * 4128` 使 channel j 访问第 j 个 token 的同一位置，而不是沿当前 token 行向后偏移。

### 16.3 行内字节偏移上界分析

对于给定 token（固定 `j`），该 token 行内的字节偏移由 `k_tile_idx` 和 `i` 决定：

```text
in_row_offset = k_tile_idx * 16 + i * 8
```

代入最大值：

```text
k_tile_idx_max = 255  (共 256 个 K tile，从 0 开始)
i_max          = 1
in_row_offset_max = 255 * 16 + 1 * 8 = 4080 + 8 = 4088
```

而数据区末尾是字节 4095，padding 区从字节 4096 开始。

```text
4088 < 4096  ✓
```

所以 streamer 在任何 K tile 的任何 spatial channel 下，行内偏移最大值是 4088，**始终落在 `[0, 4095]` 数据区内**，不会触及 `[4096, 4127]` padding 区。

### 16.4 Spatial j 维度为什么不会走进 padding 区

可能的疑惑：`j * 4128` 是否会使地址对齐到某行内的 padding 区？

不会。j 维度是**行基址偏移**，每个 j 值对应一个完整的 token 行起始：

```text
token 0 起始: delta_local_a + 0 * 4128
token 1 起始: delta_local_a + 1 * 4128
...
token j 起始: delta_local_a + j * 4128
```

在 token j 的行内，访问的字节偏移仍然是 `k_tile_idx * 16 + i * 8 ∈ [0, 4088]`，因此实际地址是：

```text
delta_local_a + j * 4128 + [0..4088]
```

这等价于在 token j 的数据区（`[j*4128, j*4128+4088]`）内取值，和 token j 的 padding 区（`[j*4128+4096, j*4128+4127]`）没有重叠。

### 16.5 M tile 的情况

当 `M_tiles > 1` 时，`mode0_A_tstride[2] = meshRow * 4128`，即跨一个 M tile 的地址步进是 `meshRow` 个完整 token 行的大小，同样是 4128 的整数倍，不会改变行内偏移分析。本 app 固定 `M_tiles=1`，该问题不出现。

### 16.6 小结

| 分析维度 | 结论 |
|---|---|
| K tile 遍历的最大行内偏移 | 4088 < 4096，数据区内 |
| Spatial `i` 维度（stride=8）| 最多偏移 8B，不超出数据区 |
| Spatial `j` 维度（stride=4128）| 跨 token，每个 token 内偏移不变 |
| A padding 区 `[4096, 4127]` | 从未被访问 |

A reader streamer 通过把 **"跨 token 跳转"（stride=4128=行 stride）放进 spatial 维度**，而把 **"K 方向遍历"（stride=16B）放进 temporal 维度**，自然实现了 padding 的跳过：spatial j 维度负责把多个 channel 同时散布到不同 token，K tile 遍历只在行内移动，二者合作使得 padding 区永远不被访问。
