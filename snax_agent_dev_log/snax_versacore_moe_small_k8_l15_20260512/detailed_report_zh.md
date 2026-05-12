# 小 MoE K8 L15 app 详细报告

**日期:** 2026-05-12  
**配置:** `snax_cluster/target/snitch_cluster/cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson`  
**App:** `snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15`  
**参考:** `review_log/layout_explore_k8_4lane_20260506_review/review_en.md`

这份 app 是从原来的 `snax-versacore-int16x4-multishape-k8-8x4-mode1-contiguous-l15` 改出来的小尺寸 MoE 版本。核心目的有两个：

1. TCDM 里先放固定大小的 expert weight，再放 token。MoE workload 里每个 expert 实际收到多少 token 是运行时才知道的，但是 expert weight 的大小是固定的，所以 weight 放在前面更接近真实部署。
2. 把矩阵规模缩小，便于快速仿真。当前参数是 `M=8` token，Mode0 两个 gate/up 矩阵是 `1024 x 128`，Mode1 down projection 拆成左右两个矩阵，每个是 `128 x 512`，最后在 TCDM 里拼成每 token 一行的 `[left 512, right 512, padding]`。

仿真已经跑通，`cycles.md` 里记录了软件 build 和 Verilator run 的结果。6 个检查全部 PASS。

## 1. 文件入口

关键文件如下：

- `data/params.hjson`: 定义矩阵尺寸。
- `data/datagen.py`: 生成 `data.h`，包括输入数据、weight、golden、TCDM offset、streamer CSR 配置。
- `src/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15.c`: C 端负责 DMA staging、配置 streamer、配置 dual VersaCore、启动、等待、比较 golden。
- `snax_cluster/target/snitch_cluster/sw/snax/dual-versacore-swiglu`: dual VersaCore SwiGLU 软件库，C app 通过 `snax-dual-versacore-swiglu-lib.h` 调用 CSR helper。
- `util/silu_pkg/silu_out16_balanced_golden.py`: Python golden 里的 SiLU 函数来源。

当前 `params.hjson` 是：

```hjson
{
  M_total: 8
  K0_total: 1024
  N0_total: 128
  K1_total: 128
  N1_total: 512
}
```

含义：

- Mode0: `A[M, K0] * W/V[K0, N0]`，这里是 `8 x 1024` 乘两个 `1024 x 128`。
- Mode0 后处理: `SiLU(AW) * AV`，输出 `D0[M, N0] = 8 x 128`。
- Mode1: `D0[M, K1] * W2[K1, N1]`，这里 `K1=N0=128`，两个 down-projection 矩阵分别输出 `512` 列。
- Mode1 输出: 左右两个 `8 x 512` 拼成 `8 x 1024`，再按 A 的 padded row stride 存成每 token 一行。

## 2. cfg 里的 streamer 配置

配置文件里真正决定 streamer 硬件形状的是 `snax_dual_versacore_int16x4_streamer_template` 和 accelerator config 里的 TCDM port 设置。

### 2.1 TCDM 和端口

cfg 中：

```hjson
tcdm: {
  size: 8192
  banks: 64
  sparse_interconnect: true
}
```

这个 app 在 `data.h` 里按 `TCDM_CAPACITY_BYTES (8192 * 1024)` 使用，所以这里按 8 MiB TCDM 理解。bank 数是 64，每个 bank word 是 8 byte，因此一个地址的 bank phase 可以按：

```text
bank = (byte_offset / 8) % 64
```

accelerator 侧配置：

```hjson
snax_tcdm_ports: 34
sparse_interconnect_config: [
  [16, 1]  // A reader
  [8,  1]  // B0 reader
  [8,  1]  // B1 reader
  [1,  1]  // D0 writer
  [1,  1]  // D1 writer
]
```

所以总端口数是：

```text
A  reader: 16 channels
B0 reader:  8 channels
B1 reader:  8 channels
D0 writer:  1 channel
D1 writer:  1 channel
total     : 34 TCDM ports
```

`sparse_interconnect_config` 的第二个数字都设为 `1`。这点很重要，因为 streamer 的空间 stride 和行 stride 会让不同 channel 访问不同 bank phase。如果 access granularity 不是 1，就可能出现某些 channel 在硬件互连里根本路由不到目标 bank 的情况。

### 2.2 Reader 参数

cfg 中 reader 参数：

```hjson
data_reader_params: {
  spatial_bounds: [
    [2, 8]  // A
    [2, 4]  // B0
    [2, 4]  // B1
  ]
  temporal_dim: [6, 4, 4]
  num_channel: [16, 8, 8]
  fifo_depth: [8, 8, 8]
  configurable_channel: [1, 1, 1]
  tcdm_logic_word_size: [
    [256, 128, 64]
    [256, 128, 64]
    [256, 128, 64]
  ]
}
```

解释：

- A reader 最多 16 channels，空间维度是 `[2, 8]`。这对应 16 个空间 lane，可以覆盖最多 `2 * 8 = 16` 个 A spatial offsets。
- B0/B1 reader 各 8 channels，空间维度是 `[2, 4]`，对应 8 个 B spatial offsets。
- A 的 temporal dimension 是 6，因为 datagen 会为 A 生成 6 维 temporal bound/stride：`K tile, N tile, M tile, 1, 1, 1`。
- B 的 temporal dimension 是 4，因为 B 只需要：`K tile, N tile, M tile, 1`。
- `configurable_channel=1` 表示每次运行可以用 CSR 配 channel enable mask。这个 app 对 S0/S1/S2 用不同 channel mask。
- `tcdm_logic_word_size` 保留 `[256,128,64]`，app 里 remap index 都是 0。这里的 256-bit entry 对当前非 remap 物理布局很关键，不能随便删。

### 2.3 Writer 参数

cfg 中 writer 参数：

```hjson
data_writer_params: {
  spatial_bounds: [
    [1]  // D0
    [1]  // D1
  ]
  temporal_dim: [4, 4]
  num_channel: [1, 1]
  fifo_depth: [1, 1]
  configurable_channel: [1, 1]
  tcdm_logic_word_size: [
    [256, 128, 64]
    [256, 128, 64]
  ]
}
```

D0/D1 writer 各只有 1 个 64-bit channel。因为 post-process 是 4-lane int16 输出，一个 writer beat 是 4 个 int16，也就是 8 byte。

Mode0 只用 D0 writer。Mode1 同时用 D0 和 D1 writer：

- D0 写左半个 down projection。
- D1 写右半个 down projection。
- 两个 writer 的 base offset 相差 `N1_total * 2 = 512 * 2 = 1024` byte，所以它们写到同一 token row 的左右两段。

## 3. 三个 shape 的含义

cfg 里的 `snax_versacore_spatial_unrolling` 定义了三种 array shape：

| Shape | array_shape | meshRow | tileSize | meshCol | 含义 |
|---|---:|---:|---:|---:|---|
| S0 | 0 | 8 | 8 | 4 | 8 token row 并行，N 方向每次 4 列 |
| S1 | 1 | 4 | 8 | 8 | 4 token row 并行，N 方向每次 8 列 |
| S2 | 2 | 2 | 8 | 16 | 2 token row 并行，N 方向每次 16 列 |

三种 shape 的 MAC 数都等价于 `meshRow * tileSize * meshCol = 256`。区别只是把 256 个 MAC 分配在 M/K/N 维度上的方式不同。

datagen 里 `SHAPE_DIMS` 就是这个表：

```python
SHAPE_DIMS = [
    ("S0", 0, 8, 8, 4),
    ("S1", 1, 4, 8, 8),
    ("S2", 2, 2, 8, 16),
]
```

每个 shape 会生成一份 `shape_cfg_t`，C 程序运行时遍历这三份配置。

## 4. SiLU 函数从哪里来

datagen 顶部这样导入 SiLU golden：

```python
_silu_pkg = os.path.realpath(os.path.join(_this_dir, "../../../../../../util/silu_pkg"))
if os.path.isdir(_silu_pkg):
    sys.path.insert(0, _silu_pkg)
else:
    sys.path.insert(0, "/esat/studscratch/r1015498/Thesis/original_snax/silu/pkg")
from silu_out16_balanced_golden import silu_out16_balanced_eval_q
```

也就是说优先使用 repo 内的：

```text
snax_cluster/util/silu_pkg/silu_out16_balanced_golden.py
```

如果这个路径不存在，就 fallback 到旧路径：

```text
/esat/studscratch/r1015498/Thesis/original_snax/silu/pkg
```

`apply_silu_vectorized()` 对每个 int16 元素调用 `silu_out16_balanced_eval_q(int(x))`：

```python
def apply_silu_vectorized(arr_int16):
    flat = arr_int16.flatten()
    result = np.array([silu_out16_balanced_eval_q(int(x)) for x in flat], dtype=np.int16)
    return result.reshape(arr_int16.shape)
```

这个函数不是普通 float SiLU，而是硬件对应的定点 golden。它用来匹配 dual VersaCore SwiGLU pipeline 里 Mode0 的 SiLU 输出，所以 golden 的行为和硬件 `silu_out16_balanced` 逻辑对齐。

Mode0 golden 流程是：

```text
vc0 = A * W
vc1 = A * V
vc0_i16 = rescale(vc0)
vc0_silu = silu_out16_balanced(vc0_i16)
vc1_i16 = rescale(vc1)
mode0 = rescale(vc0_silu * vc1_i16)
```

当前 rescale 参数都是 identity：

```c
RESCALE_INPUT_ZP = 0
RESCALE_MULTIPLIER = 1
RESCALE_OUTPUT_ZP = 0
RESCALE_SHIFT = 0
```

## 5. datagen 总体逻辑

`data/datagen.py` 的任务是打印一个完整的 C header，也就是 build 过程中生成的 `data/data.h`。它做了这些事：

1. 读 `params.hjson` 得到矩阵尺寸。
2. 生成 logical A。
3. 给 A 加 row padding，生成 physical A。
4. 生成四份 int4 packed weight：`W`, `V`, `W2_left`, `W2_right`。
5. 用 Python 模拟 streamer 读数和 VersaCore GEMM，生成 Mode0/Mode1 golden。
6. 根据 layout recipe 计算所有 tensor 在 TCDM 内的 byte offset。
7. 对 S0/S1/S2 生成 streamer CSR 配置。
8. 输出 `layout_cfg_t layout_cfgs[]`，供 C 程序直接使用。

当前只生成一个 layout：

```python
LAYOUTS = [
    {"id": 15, "name": "l15_weights_first_padded_1024_per_token", "a_pad": 32,
     "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
]
```

这个 L15 保留了前面 layout exploration 找到的核心 coloring/padding 思路，同时改成 weights-first。

## 6. A 的 padding 怎么加

logical A 是 `M_total x K0_total = 8 x 1024` 的 int16 矩阵：

```python
def make_logical_a(m_total, k_total):
    data = np.zeros((m_total, k_total), dtype=np.int16)
    for m in range(m_total):
        for k in range(k_total):
            data[m, k] = ((m * 5 + k * 3) % 11) - 5
    return data
```

所以每个 token 的真实数据长度是：

```text
K0_total * 2 byte = 1024 * 2 = 2048 byte
```

L15 设置：

```python
a_pad = 32
a_row_stride = k0_bytes + a_pad = 2048 + 32 = 2080 byte
```

padding 的实现：

```python
def make_padded_a(logical_a, row_stride_bytes):
    row_elems = row_stride_bytes // 2
    out = np.zeros((logical_a.shape[0], row_elems), dtype=np.int16)
    out[:, :logical_a.shape[1]] = logical_a
    return out.reshape(-1)
```

也就是说每一行先开 `2080 / 2 = 1040` 个 int16，然后把前 `1024` 个元素填成真实 A，后 `16` 个 int16 保持 0：

```text
每 token row:
  live A : 1024 int16 = 2048 byte
  padding:   16 int16 =   32 byte
  stride : 1040 int16 = 2080 byte
```

padding 的目的不是给计算用，而是改变下一行 token 的 TCDM bank phase。因为 64 bank、8 byte per bank word：

```text
无 padding: 2048 / 8 = 256 = 4 * 64
          每个 token row 都从同一个 bank phase 开始

加 32B padding: 2080 / 8 = 260 = 4 * 64 + 4
              下一 token row 起始 bank phase 每次 +4
```

所以 8 个 token row 的起始 bank phase 变成：

```text
0, 4, 8, 12, 16, 20, 24, 28
```

这正是 2026-05-06 layout exploration 的主要结论：A row padding 是解决 A token row bank conflict 的最大收益项。原始大尺寸里无 padding 时每个 token row 都落在同一个 bank phase，S0 读 8 个 token row 时冲突严重；加 32B padding 后，多个 token row 的访问被摊开到不同 bank phase。

当前小尺寸 app 虽然 `K0_total` 从 2048 改成 1024，但是 `1024 * 2 = 2048B` 仍然是 64-bank 周期的整数倍，所以不加 padding 仍会有同类问题。32B padding 依然有意义。

## 7. weights-first TCDM layout

原始 app 更偏 layout exploration，token/A 可以放在前面。这个小 MoE app 改成 weights-first，原因是 MoE 真实 workload 中：

- expert weight 的大小固定。
- 每个 expert 收到多少 token 不固定。
- 因此固定 weight region 放在 TCDM 前缀，variable token buffer 放后面，更适合后续扩展成一层 MoE 的 data generator。

实现函数是 `place_tensors()`：

```python
def place_tensors(globals_, layout):
    a_row_stride = globals_["k0_bytes"] + layout["a_pad"]
    a_bytes = globals_["m_total"] * a_row_stride
    w_bytes = globals_["k0_s0_tiles"] * globals_["n0_s0_tiles"] * 16
    mode0_d_bytes = globals_["m_total"] * globals_["n0_total"] * 2
    w2_bytes = globals_["k1_s0_tiles"] * globals_["n1_s0_tiles"] * 16
    mode1_padded_d_bytes = globals_["m_total"] * a_row_stride

    delta_local_b0 = colored_offset(0, layout.get("b0_color", 0))
    delta_local_b1 = colored_offset(delta_local_b0 + w_bytes, layout.get("b1_color", 0))
    delta_local_w2l = colored_offset(delta_local_b1 + w_bytes, layout.get("w2l_color", 0))
    delta_local_w2r = colored_offset(delta_local_w2l + w2_bytes, layout.get("w2r_color", 0))
    delta_local_a = colored_offset(delta_local_w2r + w2_bytes, layout.get("a_color", 0))
    delta_local_d0 = colored_offset(delta_local_a + a_bytes, layout.get("d0_color", 0))
    delta_local_mode1_d0 = colored_offset(delta_local_d0 + mode0_d_bytes, layout.get("m1d0_color", 0))
    delta_local_mode1_d1 = delta_local_mode1_d0 + globals_["n1_total"] * 2
```

实际顺序是：

```text
B0 / W
B1 / V
W2_left
W2_right
A token buffer
Mode0 D0
Mode1 D0 left output
Mode1 D1 right output, starts inside same padded token row
```

`colored_offset()` 的逻辑：

```python
def colored_offset(offset, color_bytes=0, alignment=1024):
    return align_up(offset, alignment) + int(color_bytes)
```

也就是先把 region 起点对齐到 1024B，再额外加一个 coloring byte offset。1024B 本身不会改变 bank phase：

```text
1024 / 8 = 128 = 2 * 64
```

所以真正改变 bank 的是 `color_bytes`。

当前小尺寸下几个关键大小：

```text
W/V:
  K0_s0_tiles = 1024 / 8 = 128
  N0_s0_tiles = 128 / 4 = 32
  bytes = 128 * 32 * 16 = 65536

W2_left/right:
  K1_s0_tiles = 128 / 8 = 16
  N1_s0_tiles = 512 / 4 = 128
  bytes = 16 * 128 * 16 = 32768

A:
  bytes = 8 * 2080 = 16640

Mode0 D0:
  bytes = 8 * 128 * 2 = 2048

Mode1 padded output:
  bytes = 8 * 2080 = 16640
```

实际生成的 TCDM offset 是：

| Tensor | offset byte | bank | 说明 |
|---|---:|---:|---|
| B0/W | 0 | 0 | 第一个 Mode0 weight |
| B1/V | 65808 | 34 | `65536` 后对齐再加 `272B` |
| W2_left | 132224 | 16 | B1 后对齐再加 `128B` |
| W2_right | 165888 | 0 | W2_left 后正常 1024B 对齐 |
| A | 198656 | 0 | 所有 weight 后面 |
| Mode0 D0 | 216064 | 0 | A 后面 |
| Mode1 D0 | 218368 | 32 | D0 后对齐再加 `256B` |
| Mode1 D1 | 219392 | 32 | Mode1 D0 + 1024B，bank phase 仍是 32 |
| TCDM end | 235008 | - | 总占用约 229.5 KiB |

注意 `Mode1 D1` 和 `Mode1 D0` 的 bank 一样不是 bug。它们相差 1024B，而 1024B 是完整 bank 周期的 2 倍。它们是左右两个 writer 分别写同一 token row 的两段，空间上没有重叠。

## 8. 两个 mode 的 weight coloring

### 8.1 Mode0 coloring: B1/V 放到 bank 34

Mode0 有两个 weight stream：

- B0 对应 `W`，值全是 int4 常数 1。
- B1 对应 `V`，值全是 int4 常数 2。

weight 数据生成：

```python
def packed_int4_constant(num_shape0_tiles, value):
    assert 0 <= value <= 7
    packed_byte = (value << 4) | value
    return np.full(num_shape0_tiles * 16, packed_byte, dtype=np.uint8)
```

因为 B 是 int4，两个 nibble 打进一个 byte。这里 `W` 用 `0x11`，`V` 用 `0x22`。数组长度是 `num_shape0_tiles * 16` byte。这个 `16` 是一个 S0 physical tile 的 packed byte 数。

Mode0 的 coloring 来自 2026-05-06 layout exploration：

- A row padding 解决最大冲突。
- B1 base bank coloring 进一步解决 S2 上 B0/B1 同相位读冲突。
- B1 bank 34 是最早的 clean combined-best phase：S0/S1/S2 都没有明显退化。

当前 datagen 里：

```python
{"b1_color": 272}
```

因为：

```text
272 / 8 = 34 banks
```

所以 B1/V 的起点被放到 bank 34。B0/W 保持 bank 0。

### 8.2 Mode1 coloring: W2_left bank 16，Mode1 D0 bank 32

Mode1 使用两个 down-projection weight：

- `W2_left`: 左半输出，值全是 int4 常数 1。
- `W2_right`: 右半输出，值全是 int4 常数 2。

datagen 中：

```python
{"w2l_color": 128, "m1d0_color": 256}
```

含义：

```text
W2_left bank = 128 / 8 = 16
Mode1 D0 bank = 256 / 8 = 32
```

W2_right 当前没有额外 color，保持 1024B 对齐后的 bank 0。这样 Mode1 的主要 stream phase 是：

```text
Mode1 A input: 读 Mode0 D0，bank 0 region
W2_left     : bank 16
W2_right    : bank 0
Mode1 D0    : bank 32
Mode1 D1    : bank 32
```

这个配置继承了 L15 padded-contiguous 的思路：Mode1 不把输出按纯连续 `8 x 1024` 无 padding 写，而是写回和 A 一样的 per-token padded row。这样后续如果把这个 Mode1 输出继续当 token-like buffer 使用，行 stride 仍然保持 2080B，bank phase 继续每 token +4。

## 9. streamer 配置是怎么为不同 shape/mode 生成的

每个 shape 的 streamer 配置在 `build_shape_cfg()` 里生成，最后变成 `shape_cfg_t` 字段。

### 9.1 channel enable

cfg 硬件给了最大 channel 数，但是 S0/S1/S2 实际使用不同数量的 channel。datagen 里：

```python
a_channel_en = {0: 0xFFFF, 1: 0x00FF, 2: 0x000F}[array_shape]
b_channel_en = {0: 0x03,   1: 0x0F,   2: 0xFF}[array_shape]
d_channel_en = 0x01
```

解释：

| Shape | A channels | A mask | B channels | B mask |
|---|---:|---:|---:|---:|
| S0 | 16 | `0xFFFF` | 2 | `0x03` |
| S1 | 8 | `0x00FF` | 4 | `0x0F` |
| S2 | 4 | `0x000F` | 8 | `0xFF` |

A 的 channel 数随 `meshRow` 变小而减少，B 的 channel 数随 `meshCol` 变大而增加。

### 9.2 Mode0 A reader

Mode0 A 读 padded A：

```python
mode0_A_sstride = [8, a_row_stride]
mode0_A_tbound  = [K_tiles, N_tiles, M_tiles, 1, 1, 1]
mode0_A_tstride = [tile_size * 2, 0, mesh_row * a_row_stride, 0, 0, 0]
```

其中：

```text
a_row_stride = 2080
K_tiles = K0_total / tileSize = 1024 / 8 = 128
N_tiles = N0_total / meshCol
M_tiles = 1
```

`sstride=[8,2080]` 配合 A reader 的 spatial bounds `[2,8]`，表示空间上先在 row 内按 8 byte 走，再跨 token row 按 2080 byte 走。这样 hardware 读的是 padded physical A。

`tstride[0]=tileSize*2=16`，每个 K tile 前进 8 个 int16，也就是 16 byte。

`tstride[1]=0` 是因为 Mode0 对不同 N tile 重复使用同一份 A。N 方向变了，A 不动。

`tstride[2]=mesh_row*a_row_stride` 是 M tile 的跨度。当前 `M_total=8` 且每个 shape 都只用 `M_tiles=1`，所以这个字段实际没有跨多个 M tile，但保留在配置里。

### 9.3 Mode0 B reader

Mode0 B0/B1 读 S0 physical tiled weight：

```python
mode0_B_sstride = [8, k0_s0_tiles * 16]
mode0_B_tbound  = [K_tiles, N_tiles, M_tiles, 1]
mode0_B_tstride = [16, (mesh_col // 4) * k0_s0_tiles * 16, 0, 0]
```

当前：

```text
k0_s0_tiles = 1024 / 8 = 128
mode0_B_sstride[1] = 128 * 16 = 2048 byte
```

不同 shape 的 `N_tiles` 和 N stride：

| Shape | meshCol | N_tiles | Mode0 B N stride |
|---|---:|---:|---:|
| S0 | 4 | 32 | `1 * 2048 = 2048` |
| S1 | 8 | 16 | `2 * 2048 = 4096` |
| S2 | 16 | 8 | `4 * 2048 = 8192` |

为什么 weight 用 S0 physical layout？因为 datagen 固定按 S0 tile order 存权重，然后通过不同 shape 的 `meshCol//4` 改 N stride，让 S1/S2 跳过更多 S0-col groups。

### 9.4 Mode0 D writer

Mode0 只写 D0：

```python
mode0_D_tbound  = [8, N_tiles, M_tiles, 1]
mode0_D_tstride = [8, 64, N_tiles * 64, 0]
```

含义：

- 第 0 维写 8 个 64-bit beat，形成一个 64B 输出 tile。
- 第 1 维跨 N tile，stride 固定 64B。
- 第 2 维跨 M tile，stride 是 `N_tiles * 64`。

每个 shape 的 Mode0 D0 row group 大小：

| Shape | N_tiles | Mode0 D M stride |
|---|---:|---:|
| S0 | 32 | 2048 |
| S1 | 16 | 1024 |
| S2 | 8 | 512 |

这些 layout 是 accelerator natural output layout，不是普通 row-major `8 x 128`。Mode1 的 A reader 会按对应 shape 的特殊 stride 从 D0 里读回 `K1=128`。

### 9.5 Mode1 A reader

Mode1 的 A 不是原始 token，而是 Mode0 的 D0。datagen 中：

```python
mode1_a_sstride = {0: [64, 8], 1: [8, 16], 2: [8, 32]}[array_shape]
mode1_a_k_stride = {0: 128, 1: 64, 2: 16}[array_shape]

mode1_A_tbound  = [K1, N1, M_tiles, 1, 1, 1]
mode1_A_tstride = [mode1_a_k_stride, 0, mode0_d_m_stride, 0, 0, 0]
```

这组 stride 是为了从 Mode0 D0 的 shape-specific output layout 中，把 `D0[M,128]` 按 Mode1 GEMM 需要的顺序读出来。

当前 `K1 = K1_total / tileSize = 128 / 8 = 16`。

### 9.6 Mode1 B reader

Mode1 B 读 W2_left/W2_right：

```python
mode1_B_sstride = [8, k1_s0_tiles * 16]
mode1_B_tbound  = [K1, N1, M_tiles, 1]
mode1_B_tstride = [16, (mesh_col // 4) * k1_s0_tiles * 16, 0, 0]
```

当前：

```text
k1_s0_tiles = 128 / 8 = 16
mode1_B_sstride[1] = 16 * 16 = 256 byte
```

不同 shape 的 Mode1 B N stride：

| Shape | meshCol | N1 tiles | Mode1 B N stride |
|---|---:|---:|---:|
| S0 | 4 | 128 | 256 |
| S1 | 8 | 64 | 512 |
| S2 | 16 | 32 | 1024 |

### 9.7 Mode1 D writer

Mode1 输出是 padded-contiguous per-token layout。关键字段：

```python
beats_per_row = mesh_col // 4
mode1_D_tbound  = [beats_per_row, mesh_row, n1_tiles, m_tiles]
mode1_D_tstride = [8, a_row_stride, mesh_col * 2, 0]
```

这里 `a_row_stride=2080`，所以 writer 的第 1 维跨 token row 时直接跳 2080B。这就把 Mode1 输出写成和 A 一样的 padded token-row layout。

两个 writer 的 base：

```text
D0 base = delta_local_mode1_d0
D1 base = delta_local_mode1_d1 = delta_local_mode1_d0 + N1_total * 2
```

也就是：

```text
每个 token row:
  D0 writer 写 left  512 int16 = 1024B
  D1 writer 写 right 512 int16 = 1024B
  padding 保持 16 int16 = 32B
  total row stride = 2080B
```

## 10. Python golden 怎么对应 datalayout

datagen 不是只生成数据，它还用 Python 模拟 streamer 访问方式，生成 golden。

### 10.1 streamer_i16_flat

核心函数：

```python
def streamer_i16_flat(source_i16, m_tiles, k_bound, spatial_bounds, spatial_strides,
                      k_stride, m_stride, channel_en):
```

它做的事：

1. 用 `spatial_bounds` 和 `spatial_strides` 算出所有 spatial offset。
2. 用 `channel_en` 筛掉没有启用的 channel。
3. 对每个 M tile、K tile，按 `base = mt * m_stride + kt * k_stride` 加 spatial offset，从 flat array 里取连续 4 个 int16。
4. 得到和硬件 streamer 喂给 VersaCore 类似的 flat 输入序列。

Mode0 golden 的 A 用 logical A 而不是 padded A：

```python
a_flat = streamer_i16_flat(
    logical_a, m_shape_tiles, k_shape_tiles, A_SPATIAL_BOUNDS,
    [8, k0_bytes], tile_size * 2, mesh_row * k0_bytes, a_channel_en)
```

这并不矛盾。golden 只需要算数学上的正确值，所以用 logical row stride `k0_bytes=2048` 即可；硬件实际读 padded A 时用的是 `a_row_stride=2080`。padding 区域不会被读进有效 K 范围，所以两者数学结果相同。

### 10.2 block_gemm_int16x4

GEMM golden：

```python
a = A_flat.reshape(M, K, meshRow, tileSize)
b = B_flat.reshape(N, K, meshCol, tileSize)
d[mm, nn] = np.tensordot(a[mm], b[nn], axes=([0, 2], [0, 2]))
```

也就是对 K tile 和 tileSize 两个维度做乘加，保留 `meshRow x meshCol` 空间输出。

### 10.3 Mode1 golden 的 padded output

Mode1 先生成两个输出：

```python
mode1_d0 = D0 * W2_left
mode1_d1 = D0 * W2_right
```

然后把 accelerator natural output layout 转成 per-token row-major：

```python
mode1_d0_pertoken = mode1_d0.reshape(
    m_shape_tiles, n1_tiles_shape, mesh_row, mesh_col
).transpose(0, 2, 1, 3).reshape(-1)
```

左右拼接后，再塞进 padded row：

```python
mode1_combined = np.concatenate([
    mode1_d0_pertoken.reshape(mesh_row, n1_total),
    mode1_d1_pertoken.reshape(mesh_row, n1_total),
], axis=1).reshape(-1)

row_elems = (k0_bytes + LAYOUTS[0]["a_pad"]) // 2
mode1_padded = np.zeros((mesh_row, row_elems), dtype=np.int16)
mode1_padded[:, :n1_total * 2] = mode1_combined.reshape(mesh_row, n1_total * 2)
```

当前 `n1_total * 2 = 1024` 个 int16，刚好等于原始 A 的 logical width。row_elems 是 1040，所以最后 16 个 int16 是 padding。

## 11. 从 L3 用 DMA 搬到 TCDM 时怎么设置

C 端函数是 `stage_layout_to_tcdm()`。

它先根据 datagen 给的 offset 计算 TCDM 目的地址：

```c
int16_t *local_a  = (int16_t *)(snrt_l1_next() + cfg0->delta_local_a);
uint8_t *local_b0 = (uint8_t *)(snrt_l1_next() + cfg0->delta_local_b0);
uint8_t *local_b1 = (uint8_t *)(snrt_l1_next() + cfg0->delta_local_b1);
uint8_t *local_w2l = (uint8_t *)(snrt_l1_next() + cfg0->delta_local_w2l);
uint8_t *local_w2r = (uint8_t *)(snrt_l1_next() + cfg0->delta_local_w2r);
```

然后检查 TCDM 容量：

```c
if (cfg0->tcdm_end > TCDM_CAPACITY_BYTES) {
    return 1;
}
```

真正 DMA：

```c
if (snrt_is_dm_core()) {
    snrt_dma_start_1d(local_a,   layout->a_data,        layout->a_data_length);
    snrt_dma_start_1d(local_b0,  layout->w_data,        layout->b_data_length);
    snrt_dma_start_1d(local_b1,  layout->v_data,        layout->b_data_length);
    snrt_dma_start_1d(local_w2l, layout->w2_left_data,  layout->w2_data_length);
    snrt_dma_start_1d(local_w2r, layout->w2_right_data, layout->w2_data_length);
    snrt_dma_wait_all();
}
snrt_cluster_hw_barrier();
```

这里有一个容易误解的点：虽然 TCDM layout 是 weights-first，但 DMA call 的顺序是先 A，再 B0/B1/W2。这不影响最终布局，因为每次 DMA 的 destination pointer 都已经是显式 offset。真正决定 TCDM 里谁在前谁在后的是 `delta_local_*`，不是 DMA 调用顺序。

D0、Mode1 D0、Mode1 D1 不从 L3 搬进来，它们是 accelerator writer 在 TCDM 里写出来的 output buffer。

## 12. C 程序详细流程

当前 C 程序已经按可读性重构过：不再用 `SELECT_LAYOUT`、`SELECT_SHAPE`、`RUN_MODE1` 这些宏选项控制运行路径。正常运行时固定使用 `layout_cfgs[0]`，固定跑 S0/S1/S2，且每个 shape 固定跑 Mode0 后接 Mode1。

### 12.1 main

`main()` 的流程：

1. core0 打印 app 名称、layout 数、shape 数。
2. 固定取 `layout_cfgs[0]`，也就是 L15。
3. 调 `stage_layout_to_tcdm()` 把 A 和 weights 搬到 TCDM。
4. 非 core0 在 DMA/barrier 后直接返回。
5. core0 遍历 `shape=0..NUM_SHAPES-1`，调用 `run_shape()`。
6. 最后打印 `NUM_SHAPES * 2` 个 checks 和 total error。

### 12.2 run_shape

`run_shape()` 只做调度：

```c
mode0_err = run_mode0(layout_id, cfg, subtraction_setting);
if (mode0_err) {
    return mode0_err;
}
return run_mode1(layout_id, cfg, subtraction_setting);
```

也就是说 Mode0 如果失败，就不会继续跑 Mode1；Mode0 正确后再进入 Mode1。

### 12.3 run_mode0

Mode0 做 SwiGLU 的 gate/up 阶段。

第一步配置 streamer：

```c
set_dual_versacore_streamer_csr_d0_only(
    A base/stride/bound/channel,
    B0 base/stride/bound/channel,
    B1 base/stride/bound/channel,
    D0 base/stride/bound/channel);
```

这里：

- A base 是 `delta_local_a`。
- B0 base 是 `delta_local_b0`。
- B1 base 是 `delta_local_b1`。
- D0 base 是 `delta_local_d0`。
- 所有 remap index 都是 0。

第二步配置 accelerator core：

```c
set_dual_versacore_csr(1, cfg->K_tiles, cfg->N_tiles * cfg->M_tiles,
                       subtraction_setting, cfg->array_shape, DATA_TYPE);
set_dual_versacore_mode(0);
set_dual_versacore_rescale0(...);
set_dual_versacore_rescale1(...);
set_dual_versacore_rescale_mul(...);
```

`K_tiles=128`。`N_tiles` 根据 shape 是 32/16/8。`array_shape` 是 0/1/2。`DATA_TYPE=0` 表示 int16x4。

第三步启动：

```c
set_dual_versacore_streamer_start();
set_dual_versacore_start();
```

然后先等 accelerator busy 清掉，再等 streamer/writer busy 清掉：

```c
wait_accelerator_done(...)
wait_streamer_done(...)
```

最后拿 TCDM 中 `local_d0` 和 `mode0_d0_golden` 比较。如果有 mismatch，最多打印 16 个 mismatch。

### 12.4 run_mode1

Mode1 做 down projection。

先清空 Mode1 padded output region：

```c
for (int i = 0; i < cfg->mode1_padded_output_elems; i++) {
    local_mode1_d[i] = 0;
}
```

这一步很重要，因为 Mode1 每个 row 后面有 padding，writer 不写 padding 区域，padding 必须保持 0 才能和 golden 对上。

然后配置完整 streamer，也就是 3 readers + 2 writers：

```c
set_dual_versacore_streamer_csr(
    Mode1 A base = delta_local_d0,
    B0 base = delta_local_w2l,
    B1 base = delta_local_w2r,
    D0 base = delta_local_mode1_d0,
    D1 base = delta_local_mode1_d1);
```

Mode1 的 A reader 读的是 Mode0 D0。B0/B1 readers 分别读 `W2_left/W2_right`。D0/D1 writers 分别写 left/right output。

core 配置：

```c
set_dual_versacore_csr(1, cfg->K1, cfg->N1 * cfg->M_tiles,
                       subtraction_setting, cfg->array_shape, DATA_TYPE);
set_dual_versacore_mode(1);
```

当前 `K1=16`。`N1` 根据 shape 是 128/64/32。

最后启动、等待、比较 `mode1_padded_golden`。

### 12.5 wait helper

`wait_accelerator_done()` 会先把 `DUAL_VC_START` 写 0，然后 polling `DUAL_VC_BUSY`。超时会打印 accelerator、streamer、writer 状态。

`wait_streamer_done()` 会先把 `STREAMER_START_CSR` 写 0，然后 polling `STREAMER_BUSY_CSR`。超时会打印 streamer/writer 状态。

这两个 helper 的目的不是计算，而是防止仿真在 deadlock 时一直卡住，并给出最低限度诊断信息。

## 13. cycle 结果

当前仿真结果：

| Shape | Mode | Correctness | Accelerator cycles | Streamer cycles | Wall cycles |
|---|---|---|---:|---:|---:|
| S0 | Mode0 | PASS | 4102 | 4126 | 42311 |
| S0 | Mode1 | PASS | 2180 | 2199 | 235108 |
| S1 | Mode0 | PASS | 2058 | 2082 | 28702 |
| S1 | Mode1 | PASS | 1110 | 1129 | 126607 |
| S2 | Mode0 | PASS | 1037 | 1061 | 22794 |
| S2 | Mode1 | PASS | 544 | 563 | 72594 |

总结果：

```text
total checks: 6
total error : 0
```

DMA staging cycles 是 3520。TCDM 使用量是 235008 byte。重构后 wall cycles 会包含更清晰的函数结构和 progress print 影响；如果比较硬件本体，优先看 accelerator/streamer counters。

## 14. 如果要改矩阵 size，当前 datagen 支持吗

严格说，当前这个小 MoE datagen **还不支持任意 size**。它在 `emit_header()` 里写了硬 assert：

```python
assert m_total == 8
assert k0_total == 1024
assert n0_total == 128
assert k1_total == 128
assert n1_total == 512
```

所以现在只能生成这组尺寸。这是有意保守的，因为当时目标是快速跑通一个小尺寸 MoE app，而不是做完全泛化的 generator。

不过代码里很多计算已经是参数化的，例如：

- `k0_s0_tiles = k0_total // 8`
- `k1_s0_tiles = k1_total // 8`
- `n0_s0_tiles = n0_total // 4`
- `n1_s0_tiles = n1_total // 4`
- B/W2 的 size 和 stride 都由这些 tile 数推出来。
- TCDM offsets 也会按 size 自动推。

### 14.1 可以比较容易支持的改动

如果只是改 `K0_total/N0_total/K1_total/N1_total`，可以把 hard assert 改成约束检查：

```python
assert m_total == 8
assert k0_total % 8 == 0
assert k1_total % 8 == 0
assert n0_total % 16 == 0
assert n1_total % 16 == 0
assert k1_total == n0_total
```

其中 `n0_total/n1_total % 16 == 0` 是为了覆盖最大 `meshCol=16` 的 S2。S0/S1 只要求能被 4/8 整除，但三 shape 都跑就按 16 要求。

还要检查 Mode1 输出能不能塞进 padded row：

```text
2 * N1_total int16 <= row_elems
row_elems = (K0_total * 2 + a_pad) / 2
```

等价于：

```text
4 * N1_total <= 2 * K0_total + a_pad
```

当前：

```text
2 * N1_total = 1024 int16
row_elems = 1040 int16
```

所以刚好可以放下左右两个 512 输出，再剩 16 int16 padding。

### 14.2 如果改 M/token 数

当前 `M_total=8`，datagen 里每个 shape 都写死：

```python
m_tiles = 1
```

所以如果改 token 数，不只是改 assert，还要改 M tile 支持。至少有两种路线：

1. 要求 `M_total` 等于每个 shape 的 `meshRow`，每次只跑刚好一个 tile。这不适合三 shape 同时跑，因为 S0/S1/S2 的 `meshRow` 分别是 8/4/2。
2. 真正支持 `M_tiles = ceil_div(M_total, meshRow)`，并处理最后一个 M tile 的 tail。这需要 datagen、golden、streamer bound、C check 都一起改。

对 MoE 更真实的做法可能是：每个 expert 一个 token block，block 内按 `meshRow` 对齐或 padding 到固定 tile 数。routing 后每个 expert token 数不同，generator 需要输出每个 expert 的 A base、token_count、possibly padded token_count 和 per-expert invocation table。

### 14.3 如果改 shape family

如果只是改矩阵 size，不需要改 cfg。但如果要改：

- `meshRow/tileSize/meshCol`
- A/B channel 数
- writer channel 数
- data type 或 lane 数

那就不是 datagen 级别的改动了，需要改 hjson cfg，重新 `rtl-gen`，重 build software/hardware，再跑 Verilator。

### 14.4 coloring 是否需要重扫

建议重扫。A padding 的基本原则仍成立：row stride 不要是完整 bank 周期，且保持 8B 对齐。但 B1/W2/D coloring 的最佳 phase 可能随矩阵 size、shape、输出 stride 改变。

一个合理的下一步是把 `LAYOUTS` 扩成多个候选：

```python
for a_pad in [32, 48, 56]:
  for b1_color in [272, 320, 352, 384, 416]:
  for w2l_color in [128, 256, 384]:
  for m1d0_color in [128, 256, 384]:
```

当前 C 程序固定跑 `layout_cfgs[0]`。如果要重新做 layout sweep，可以临时把 C 程序改成遍历多个 `layout_cfgs[]`，或者引入一个很小的 layout index 常量；跑完记录 streamer cycles，再选新的 Lxx。

## 15. 给 Hemaia MoE agent 的接口含义

这个 app 目前不是完整 MoE layer generator，只是单 expert、固定 token block 的小型 executable test。它对后续 Hemaia agent 的价值在于：

- 展示了 expert weights-first 的 TCDM placement。
- 展示了 variable token buffer 可以放在所有 fixed weight 后面。
- 保留了 A row padding 和 weight coloring 的实现方式。
- 给出了 Mode0 gate/up + Mode1 down projection 的完整 C 控制流。
- 给出了 Python golden 如何和 hardware streamer layout 对齐。

如果 Hemaia agent 要生成整个 MoE layer 的 data，建议在这个基础上扩展：

1. 全部 expert 的 `W/V/W2_left/W2_right` 固定排在 TCDM 或 L3 layout 前缀。
2. 每个 expert 一段 token buffer，token 数可变但按 tile 对齐。
3. 为每个 expert 生成一条 invocation descriptor：token base、token count、expert weight base、output base、shape 选择。
4. 对每个 expert 独立复用这里的 `shape_cfg_t` 逻辑，只把 base offset 和 M tile count 参数化。
5. 对 routed token 的 scatter/gather 另建 metadata，不要混进这个 single-expert app 的固定 layout 里。

## 16. 结论

这个小 MoE app 的关键点可以总结成一句话：

```text
固定 expert weights 放在 TCDM 前面，variable token A 放在后面；
A 和 Mode1 output 都使用 2080B per-token padded row；
Mode0 继承 A padding + B1 bank34 coloring；
Mode1 使用 W2_left bank16 + Mode1 output bank32；
datagen 同时生成 physical layout、streamer CSR 和 bit-true golden；
C 程序只负责 DMA staging、CSR 配置、启动等待和结果检查。
```

当前版本已经跑通小尺寸：

```text
M=8
K0=1024
N0=128
K1=128
N1=512
```

如果要支持任意 MoE expert size 或 variable token count，下一步需要把 datagen 里的 hard assert 改成约束检查，并把 `M_tiles`、tail/padding、per-expert descriptor 做成真正参数化。
