# K8 common16：1024 中间维度、全权重常驻实验

## 1. 实验目标

本版使用同一个硬件配置：

```text
target/snitch_cluster/cfg/
snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson
```

软件 workload 改为：

```text
A          = [8, 2048]      int16
W, V       = [2048, 1024]   packed int4
logical W2 = [1024, 2048]   packed int4
W2_left    = [1024, 1024]   packed int4, VersaCore 0
W2_right   = [1024, 1024]   packed int4, VersaCore 1
```

所有 weight 在任何 shape 开始前只 DMA 一次，按照未来 overlap 所需的
16-column ping/pong pattern 放到不同 depth slot，并在 S0/S1/S2 全流程中一直驻留。

运行顺序固定为：

```text
stage W/V once
stage W2_left/right once

S0: refresh A -> Mode0 -> Mode0 activation -> Mode1
S1: refresh A -> Mode0 -> Mode0 activation -> Mode1
S2: refresh A -> Mode0 -> Mode0 activation -> Mode1
```

Mode0 输出仍写 banks 48..63；Mode1 直接从该区域读取 activation，并把两个
1024-column 输出分别写回 banks 0..7 和 8..15。下一个 shape 开始时 token DMA
刷新 banks 0..15。

## 2. 上一版 1408 中间维度 run 分析

旧日志：

```text
layout_explore_logs/common16_chunked_20260718/all_shapes_chunk16.log
```

旧 run 完整结束，没有 timeout 或 mismatch：

```text
FINAL_RESULT selected_shape=-1 chunk_cols=16 status=PASS total_errors=0
```

### 2.1 正确性和计算周期

| Shape | Mode | chunks | status | accel sum | streamer sum | wall sum | avg accel/chunk | avg streamer/chunk | avg wall/chunk |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| S0 `(8,8,4)` | 0 | 88 | PASS | 90,640 | 92,752 | 257,592 | 1030.0 | 1054.0 | 2927.2 |
| S0 `(8,8,4)` | 1 | 64 | PASS | 45,440 | 46,592 | 178,200 | 710.0 | 728.0 | 2784.4 |
| S1 `(4,8,8)` | 0 | 88 | PASS | 45,584 | 47,696 | 216,568 | 518.0 | 542.0 | 2461.0 |
| S1 `(4,8,8)` | 1 | 64 | PASS | 22,912 | 24,064 | 157,828 | 358.0 | 376.0 | 2466.1 |
| S2 `(2,8,16)` | 0 | 88 | PASS | 23,232 | 25,344 | 196,042 | 264.0 | 288.0 | 2227.8 |
| S2 `(2,8,16)` | 1 | 64 | PASS | 11,808 | 12,960 | 144,302 | 184.5 | 202.5 | 2254.7 |

计算量随 active token 8/4/2 基本按 2 倍递减。wall 没有同比缩小，是因为每个
16-column chunk 都重复进行 CSR 配置、start/wait 和 compare。

### 2.2 旧 run 的重复 weight DMA

| Shape | Mode0 DMA cycles | Mode1 DMA cycles |
|---|---:|---:|
| S0 | 1,776,354 | 1,250,615 |
| S1 | 1,776,461 | 1,250,615 |
| S2 | 1,776,677 | 1,250,811 |
| 合计 | 5,329,492 | 3,752,041 |

三个 shape 总 weight DMA 为 9,081,533 cycles，而六次计算的 wall sum 合计为
1,150,532 cycles。旧 app 把 weight staging 放在 `run_shape()` 内，导致同一份
weight 被重复搬三次。旧结果可以作为功能 baseline，但其 DMA 流程不是最终的
resident-weight workload。

## 3. 新尺寸的容量计算

三个逻辑 weight 的大小相同：

```text
2048 * 1024 / 2 = 1,048,576 B = 1 MiB
```

因此：

```text
W + V                  = 2 MiB
W2_left + W2_right     = 1 MiB
all resident weights   = 3 MiB
banks 16..47 capacity  = 4 MiB
free weight capacity   = 1 MiB
```

TCDM 每个全 bank row 为 `64 banks * 8 B = 512 B`。weight 使用其中连续的
32 banks，因此一个 512 B flat row 中有 256 B weight payload。

## 4. 常驻 ping/pong 物理映射

固定 bank phase：

| Buffer | B0 banks | B1 banks |
|---|---|---|
| ping | 16..23 | 24..31 |
| pong | 32..39 | 40..47 |

两种 mode 使用相同 bank phase，但使用完全不重叠的 row ranges：

| Region | TCDM rows | flat offset/span | actual weight bytes |
|---|---|---:|---:|
| Mode0 W/V | 0..8191 | offset 0, span 4 MiB | 2 MiB |
| Mode1 W2L/W2R | 8192..12287 | offset 4 MiB, span 2 MiB | 1 MiB |
| weight-bank free | 12288..16383 | flat span 2 MiB | 1 MiB |

flat region end 是 6 MiB，小于整个 8 MiB TCDM；这里的 flat span 包含同一 row
中不属于 weight bank group 的另外 32 banks。

### 4.1 先用 bank/row 坐标理解 weight

TCDM flat offset 与 bank/row 的关系是：

```text
flat_offset(row, bank, byte_in_bank)
    = row * 512 + bank * 8 + byte_in_bank

bank = (flat_offset / 8) % 64
row  = flat_offset / 512
```

因此一个 TCDM row 可以画成：

```text
banks   0........15 | 16.....23 | 24.....31 | 32.....39 | 40.....47 | 48.....63
role    A/Mode1 D   | B0 ping   | B1 ping   | B0 pong   | B1 pong   | Mode0 D
bytes   0.......127 | 128...191 | 192...255 | 256...319 | 320...383 | 384...511
```

weight placement 同时使用三个互相独立的坐标：

1. **tensor/buffer 沿 bank 方向分开**：B0/B1、ping/pong 各占固定 8-bank group；
2. **panel 沿 row 深度分开**：同一个 16-column chunk 的四个 4-column panels
   依次放在四段不重叠的 rows；
3. **resident slot 再沿 row 深度分开**：chunk pair `(0,1)` 共用 slot 0 的 row
   range，但分别占 ping/pong banks；chunk pair `(2,3)` 使用 slot 1 的下一段 rows。

这里的最小逻辑单元是一个 **4-column full-K panel**，而不是一个 64 B row：

```text
Mode0 one W/V panel   = [K=2048, N=4] int4 = 4096 B
Mode1 one W2L/R panel = [K=1024, N=4] int4 = 2048 B
```

一个 panel 在 L3 中连续；进入 TCDM 后，每次取连续 64 B 放进固定 8 banks，下一
64 B 放到下一条 512 B row。因此：

```text
Mode0 panel: 64 repeats * 512 B row stride = 32768 B panel_span = 64 rows
Mode1 panel: 32 repeats * 512 B row stride = 16384 B panel_span = 32 rows
```

`panel_span` 是首地址到下一个 panel 首地址的 flat distance，不是实际 payload。
Mode0 panel 虽然 span 32 KiB，真实数据仍只有 4 KiB；每个 row 只由该 tensor 写其中
一个 64 B bank group。其他 bank group 可同时容纳另一个 tensor、另一个 buffer 或
A/D 数据。

对 chunk `c`：

```text
slot = c >> 1
buffer = (c & 1) ? pong : ping

Mode0 region_offset = 0
Mode1 region_offset = 4,194,304

base(mode,c,tensor) = region_offset(mode)
                    + ping_or_pong_bank_base(tensor)
                    + slot * chunk_slot_span(mode)
```

默认 `CHUNK_COLS=16` 时：

| Mode | full-K panel span | panels/chunk | chunk slot span | chunks | slots/buffer |
|---|---:|---:|---:|---:|---:|
| 0 | 32,768 B | 4 | 131,072 B | 64 | 32 |
| 1 | 16,384 B | 4 | 65,536 B | 64 | 32 |

### 4.2 Mode0 的具体地址例子

下面地址均为相对 `snrt_l1_next()` 的 flat offset。Mode0 的
`region_offset=0`，一个 chunk 有四个 panels、占 256 rows：

| chunk | logical columns | slot/buffer | B0/W base | B1/V base | occupied rows |
|---:|---|---|---:|---:|---|
| 0 | 0..15 | slot0 ping | `0x000080` | `0x0000c0` | 0..255 |
| 1 | 16..31 | slot0 pong | `0x000100` | `0x000140` | 0..255 |
| 2 | 32..47 | slot1 ping | `0x020080` | `0x0200c0` | 256..511 |
| 3 | 48..63 | slot1 pong | `0x020100` | `0x020140` | 256..511 |

以 chunk 0 的 W 为例，四个 panel 的 row range 是：

```text
panel 0, logical cols  0..3  : base 0x000080, rows   0..63, banks 16..23
panel 1, logical cols  4..7  : base 0x008080, rows  64..127, banks 16..23
panel 2, logical cols  8..11 : base 0x010080, rows 128..191, banks 16..23
panel 3, logical cols 12..15 : base 0x018080, rows 192..255, banks 16..23
```

V 使用完全相同的 rows，但 base 的 bank offset 从 `0x80` 换成 `0xc0`，即 banks
24..31。chunk 1 也使用 rows 0..255，但切到 pong banks 32..47，所以不会覆盖
chunk 0。chunk 2 回到 ping banks，同时通过 `slot=1` 下移 256 rows。

### 4.3 Mode1 的具体地址例子

Mode1 从 `region_offset=0x400000`、即 row 8192 开始。K 减半后，一个 panel 是
32 rows，一个 chunk 是 128 rows：

| chunk | logical columns per half | slot/buffer | B0/W2L base | B1/W2R base | occupied rows |
|---:|---|---|---:|---:|---|
| 0 | 0..15 | slot0 ping | `0x400080` | `0x4000c0` | 8192..8319 |
| 1 | 16..31 | slot0 pong | `0x400100` | `0x400140` | 8192..8319 |
| 2 | 32..47 | slot1 ping | `0x410080` | `0x4100c0` | 8320..8447 |
| 3 | 48..63 | slot1 pong | `0x410100` | `0x410140` | 8320..8447 |

这里的 logical columns 是每个 1024-column half 内的列号。W2L 与 W2R 最后分别
形成完整 W2 输出的左、右 1024 columns；两者不是两个 K slice，也不需要做
partial-sum 合并。

这里保留 `slot * chunk_slot_span` 是因为 SNAX app 先搬完再算，所有 chunk 必须
同时存在。这个 resident-slot 方案是可移植的相对布局，不只适用于当前 SNAX app：
HeMAiA 若也让全部 weight chunks 驻留，可以直接保留 slot 编号和 row-depth 展开；若
使用多级预取，也可以保留有限数量的 slots 并在同步后循环复用。只有严格的两块
ping/pong 双缓冲才会退化为始终使用 slot 0，让偶数 chunk 复用固定 ping base、奇数
chunk 复用固定 pong base。

### 4.4 边搬边算粒度变粗时如何复用 resident slot

令真正的 DMA/compute granularity 为 `G` 个输出列，其中 `G` 是 16 的倍数。4-column
canonical panel、ping/pong bank phase、B streamer 的 spatial/K strides 都不需要变；
只需把一个 chunk 中合并的 panel 数和 row-depth span 改为：

```text
panels_per_chunk = G / 4

Mode0:
  rows_per_chunk  = (G/4) * 64 = 16G
  chunk_slot_span = 16G * 512 B = 8192G B
  DMA repeat      = (G/4) * 64 = 16G       # per W or V descriptor

Mode1:
  rows_per_chunk  = (G/4) * 32 = 8G
  chunk_slot_span = 8G * 512 B = 4096G B
  DMA repeat      = (G/4) * 32 = 8G        # per W2L or W2R descriptor
```

例如：

| G | Mode0 slot span / repeat | Mode1 slot span / repeat | chunks for N=1024 |
|---:|---:|---:|---:|
| 16 | 128 KiB / 256 | 64 KiB / 128 | 64 |
| 32 | 256 KiB / 512 | 128 KiB / 256 | 32 |
| 64 | 512 KiB / 1024 | 256 KiB / 512 | 16 |

若所有 chunks 驻留且 `1024/G` 为偶数，仍可使用：

```text
slot = chunk >> 1
slots_per_buffer = (1024/G) / 2
```

此时粒度变粗会让单个 slot 变深，但 slot 数同比减少，因此 Mode0/Mode1 的总 flat
region span 仍分别保持 4 MiB/2 MiB。若只保留 `D` 个预取 slot pairs，则可以使用
`slot=(chunk>>1)%D`，但必须用 DMA-ready/compute-done 同步保证覆盖安全。

streamer 方面，固定的 panel/K stride 可以照搬；主要变化是一次 invocation 的
`N_tiles=G/meshCol` 以及对应 temporal bound。HeMAiA 仍应由自己的 layout/runtime
维护总 region base 和每个 slot 的实际起始地址。

## 5. DMA 参数

### 5.1 Token refresh

每个 shape 开始前刷新全部 8 个 token；shape channel mask 决定实际消费 8/4/2 个：

```text
8 x 2D DMA
size       = 16 B
src_stride = 16 B
dst_stride = 512 B
repeat     = 2048 / 8 = 256
dst        = A_BASE + token * 16
```

### 5.2 Mode0 W/V，一次性搬运

4-column full-K panel 仍是 layout 原子：

```text
panel payload per tensor = 2048 * 4 / 2 = 4096 B
```

但四个相邻 panel 在 L3 中连续，在 TCDM 中也保持同一个 512 B row stride，因此
当前实现将整个 16-column chunk 合并为每个 tensor 一笔 2D DMA：

```text
chunk payload per tensor = 4 * 4096 = 16384 B
size                     = 64 B
src_stride               = 64 B
dst_stride               = 512 B
repeat                   = 16384 / 64 = 256
```

每 chunk 只有 W、V 两笔 descriptor；64 chunks 共 128 笔 2D DMA。实际 W/V
payload 共 2 MiB。

### 5.3 Mode1 W2L/W2R，一次性搬运

```text
panel payload per tensor = 1024 * 4 / 2 = 2048 B
chunk payload per tensor = 4 * 2048 = 8192 B
size                     = 64 B
src_stride               = 64 B
dst_stride               = 512 B
repeat                   = 8192 / 64 = 128
```

同样每 chunk 只有 W2L、W2R 两笔，总计 128 笔 2D DMA。W2L/W2R payload 共
1 MiB。

当前 L3 source 仍是 separated W/V/W2L/W2R arrays。每个 chunk 提交左右 tensor
两笔请求后执行一次 `snrt_dma_wait_all()`；这个 wait 才是 16-column weight-ready
边界。

### 5.4 B0/B1 的 DMA 提交顺序

当前 `stage_weight_chunks()` 的软件循环顺序是：

```text
for chunk in 0..63:
  enqueue B0/left 整个 16-column chunk 的 2D descriptor
  enqueue B1/right 整个 16-column chunk 的 2D descriptor
  wait both descriptors
```

所以 Mode0 的 descriptor 顺序是 `W columns 0..15 -> V columns 0..15`，Mode1
则是 `W2L columns 0..15 -> W2R columns 0..15`。每笔 descriptor 内部仍按
panel0、panel1、panel2、panel3 的 canonical source 顺序前进，但 panel boundary
不再产生新 DMA command。

这仍没有把 W/V 的每个 64 B beat 预先交错成一个 128 B source record，也没有刻意
实现“一个周期 W、下一个周期 V”的 512-bit 交替。iDMA backend 仍按单一 command
stream 执行两笔 chunk-wide 请求。

若 HeMAiA 为减少 descriptor 数而修改 source packing，可以把 B0/B1 的 64 B beats
交错；但那是另一种 source ABI，不能只改 DMA 参数。datagen 的 flatten、source
pointer 公式、DMA descriptor generator 和 byte-level checker 必须同时改变。

### 5.5 W 的 512-bit beat 到逻辑 element/bank 的精确映射

#### “一个 beat”不等于“DMA 启动后的一个固定 clock”

本 cfg 的 `dma_data_width=512`，TCDM bank width 为 64 bit。因此一个 **被 TCDM
接受的、完整 64 B DMA write beat** 会被 wide-to-narrow mux 拆成 8 个并行 8 B
writes，落到同一个 superbank 的连续 8 banks。

但不应把 DMA start 后的 clock 0、1、2……直接称为 beat 0、1、2：descriptor
decode、L3 AXI read、data realignment、FIFO、AXI/TCDM ready-valid 和仲裁都可能产生
空周期或 backpressure。准确说法是：

```text
第 g 个 accepted destination write beat
```

在没有 stall 的稳态区间，它可以连续每 clock 接受一个 512-bit write；有 stall 时，
逻辑 beat 顺序不变，只是相邻 accepted beats 之间插入若干 clock。

#### Mode0 W 的通用公式

对 chunk `c`、chunk 内 panel `p=0..3`、panel 内 64 B beat `g=0..63`：

```text
global_panel P = 4*c + p
logical N      = 4*P .. 4*P+3
logical K      = 32*g .. 32*g+31

slot           = c >> 1
W bank_start   = (c even) ? 16 : 32
TCDM row       = slot*256 + p*64 + g

dst_offset     = row*512 + bank_start*8
               = weight_base(Mode0,W,c) + p*32768 + g*512
```

每个 64 B beat 内包含四个连续 K tiles。每个 K tile 是 16 B，占两个 banks：

```text
beat bytes  0..15 -> K tile 4*g+0 -> logical K 32*g+ 0 .. 32*g+ 7
beat bytes 16..31 -> K tile 4*g+1 -> logical K 32*g+ 8 .. 32*g+15
beat bytes 32..47 -> K tile 4*g+2 -> logical K 32*g+16 .. 32*g+23
beat bytes 48..63 -> K tile 4*g+3 -> logical K 32*g+24 .. 32*g+31
```

以 ping 的 `bank_start=16` 为例，bank mapping 是：

| Destination bank | Beat bytes | 逻辑 W elements |
|---:|---:|---|
| 16 | 0..7 | `N=4P+0,4P+1`，`K=32g+0..7` |
| 17 | 8..15 | `N=4P+2,4P+3`，`K=32g+0..7` |
| 18 | 16..23 | `N=4P+0,4P+1`，`K=32g+8..15` |
| 19 | 24..31 | `N=4P+2,4P+3`，`K=32g+8..15` |
| 20 | 32..39 | `N=4P+0,4P+1`，`K=32g+16..23` |
| 21 | 40..47 | `N=4P+2,4P+3`，`K=32g+16..23` |
| 22 | 48..55 | `N=4P+0,4P+1`，`K=32g+24..31` |
| 23 | 56..63 | `N=4P+2,4P+3`，`K=32g+24..31` |

每个 bank 的 8 B 又包含两列，每列 4 packed bytes：

```text
bank byte 0: first  N column, K lane 0 in low nibble, lane 1 in high nibble
bank byte 1: first  N column, K lane 2 low, lane 3 high
bank byte 2: first  N column, K lane 4 low, lane 5 high
bank byte 3: first  N column, K lane 6 low, lane 7 high
bank byte 4..7: second N column，采用相同 lane-pair 顺序
```

如果 `c` 为奇数，只需把表中的 banks 16..23 换成 pong banks 32..39；逻辑
elements 和 row 公式不变。V 的 element mapping 相同，但 ping/pong bank_start 分别
是 24/40。

#### chunk 0 搬 W 时的 accepted-write 顺序

假设 command FIFO 按提交顺序执行，Mode0 chunk 0 的第一笔 descriptor 覆盖完整
16-column W，第二笔才覆盖完整 16-column V：

| Descriptor/beat | Logical elements | TCDM destination |
|---|---|---|
| W panel0, `g=0` | `W[K=0..31,N=0..3]`，按上表排列 | row 0, banks 16..23 |
| W panel0, `g=1` | `W[K=32..63,N=0..3]` | row 1, banks 16..23 |
| ... | ... | ... |
| W panel0, `g=63` | `W[K=2016..2047,N=0..3]` | row 63, banks 16..23 |
| W panel1, `g=0..63` | `W[K=0..2047,N=4..7]` | rows 64..127, banks 16..23 |
| W panel2, `g=0..63` | `W[K=0..2047,N=8..11]` | rows 128..191, banks 16..23 |
| W panel3, `g=0..63` | `W[K=0..2047,N=12..15]` | rows 192..255, banks 16..23 |
| V panels0..3 | `V[K=0..2047,N=0..15]` | rows 0..255, banks 24..31 |

software 会异步 enqueue 后续 descriptor，但没有把 W 和 V 在单个 descriptor 内
beat-by-beat 交错。逻辑顺序是 W 的 256 个 destination beats 覆盖完整 16 columns，
然后才是 V 的 256 个 beats，而不是 `W beat0, V beat0, W beat1, V beat1`。

#### Mode1 的变化

W2L/W2R 使用完全相同的 beat 内部格式，只把 K 和 row 深度减半：

```text
g             = 0..31
logical K     = 32*g .. 32*g+31
TCDM row      = 8192 + slot*128 + p*32 + g
W2L bank_start= even chunk 16, odd chunk 32
W2R bank_start= even chunk 24, odd chunk 40
```

#### 当前 ELF 的 source alignment 注意事项

2026-07-19 合并为每 chunk 两笔 descriptor 后重新构建的 ELF 中，
`W=0x80084cf4`，即 source address `%64=52`；
W/V/W2L/W2R 都具有相同的 64 B misalignment。destination base 是 64 B aligned，
所以 TCDM 侧仍以上述完整 64 B beats 写入，但 source 侧的一个逻辑 64 B block 会跨
aligned AXI beat boundary，iDMA 必须读取/realign 后再写出。这会影响真实 cycle 数，
也意味着不能从 Mode0 `repeat=256` 推导“整个 16-column W chunk 恰好 256 clocks”。

这个 source 地址是链接布局的构建产物，不是 datagen ABI。HeMAiA 做最终性能版本时，
建议显式保证 packed weight blobs 至少 64 B aligned，并用 ELF symbol/map 或 runtime
assert 检查；否则逻辑 layout 正确，但 DMA 带宽可能因为每行 source misalignment
受到影响。

## 6. 每个 16-column accelerator invocation

所有 invocation 都使用完整 K，沿 N 切分；不存在跨 invocation 的 partial-sum。

| Shape | array shape | active M | meshCol | N tiles/chunk | A mask | B mask | D mask |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | 0 | 8 | 4 | 4 | `0xffff` | `0x03` | `1` |
| S1 | 1 | 4 | 8 | 2 | `0x00ff` | `0x0f` | `1` |
| S2 | 2 | 2 | 16 | 1 | `0x000f` | `0xff` | `1` |

Accelerator CSR：

```text
Mode0: M_tiles=1, K_tiles=2048/8=256, N_tiles=4/2/1
Mode1: M_tiles=1, K_tiles=1024/8=128, N_tiles=4/2/1
```

## 7. Mode0 streamer

### 7.1 A reader

| Shape | spatial bounds | spatial strides | channel mask | temporal bounds | temporal strides |
|---|---|---|---|---|---|
| S0 | `[2,8]` | `[8,16]` | `0xffff` | `[256,4,1,1,1,1]` | `[512,0,0,0,0,0]` |
| S1 | `[2,8]` | `[8,16]` | `0x00ff` | `[256,2,1,1,1,1]` | `[512,0,0,0,0,0]` |
| S2 | `[2,8]` | `[8,16]` | `0x000f` | `[256,1,1,1,1,1]` | `[512,0,0,0,0,0]` |

三种 shape 的 spatial 配置相同并不表示 token 连续拼成一个 matrix stream；token
在 TCDM 中仍然是 per-token 排布。`stage_tokens()` 形成的地址是：

```text
A_addr(token, k_tile) = A_BASE + token * 16 + k_tile * 512
```

一个 token 的一个 K tile 是 8 个 int16，即 16 B，由两个 8 B reader channel
读取。`spatial_bounds=[2,8]`、`spatial_strides=[8,16]` 展平后的 channel offset 为：

```text
offset(channel i) = (i % 2) * 8 + (i / 2) * 16
channel 0,1   -> token 0
channel 2,3   -> token 1
...
channel 14,15 -> token 7
```

所以 shape 的 token 数由 channel mask 选择：S0 使能 16 channels/8 tokens，S1
使能 8 channels/4 tokens，S2 使能 4 channels/2 tokens。第一 temporal dimension
以 512 B stride 走完 256 个 K tiles；第二 dimension 的 bound `4/2/1` 只是在同一
次 16-column invocation 内，为不同 N tile 重放同一个 A，不移动 A base。因此这里
的配置正好对应 per-token 存储，而不是所有 shape 使用相同数量的 token。

### 7.2 W/V readers

Mode0 `panel_span=32768`：

| Shape | spatial bounds | spatial strides | channel mask | temporal bounds | temporal strides |
|---|---|---|---|---|---|
| S0 | `[2,4]` | `[8,32768]` | `0x03` | `[4,64,4,1]` | `[16,512,32768,0]` |
| S1 | `[2,4]` | `[8,32768]` | `0x0f` | `[4,64,2,1]` | `[16,512,65536,0]` |
| S2 | `[2,4]` | `[8,32768]` | `0xff` | `[4,64,1,1]` | `[16,512,131072,0]` |

对 B reader channel `i`，spatial offset 为：

```text
B_offset(i) = (i % 2) * 8 + (i / 2) * panel_span

channels 0,1 -> 当前第 0 个 panel 的两个 8 B halves
channels 2,3 -> 相邻第 1 个 panel
channels 4,5 -> 相邻第 2 个 panel
channels 6,7 -> 相邻第 3 个 panel
```

因此同一个 16-column chunk base 可以被三种 shape 这样消费：

| Shape | 每个 array N tile 空间上同时读取 | invocation 内 N tiles | temporal N stride |
|---|---|---:|---:|
| S0 | panel 0，共 4 columns | 4 | 1 panel = 32768 B |
| S1 | panels 0..1，共 8 columns | 2 | 2 panels = 65536 B |
| S2 | panels 0..3，共 16 columns | 1 | 4 panels = 131072 B |

对 S0，第三 temporal dimension 会依次把 base 推到 panel 0、1、2、3；对 S1，
它依次读取 panel pair `(0,1)`、`(2,3)`；对 S2，一次 spatial access 已经覆盖四个
panels。第一 temporal dimension `bound=4,stride=16` 读取同一 TCDM row 中连续四个
K tiles，第二 dimension `bound=64,stride=512` 走完完整 K。三种 shape 最终消费的
仍是同一组 16 logical columns、同一份 resident bytes。

B0/B1 只在每个 chunk 更新 resident `weight_base()`；stride/bound 不包含 ping/pong
或跨 chunk 维度。

### 7.3 Mode0 D writer，banks 48..63

| Shape | temporal bounds | temporal strides |
|---|---|---|
| S0 | `[1,8,2,2]` | `[8,16,8,512]` |
| S1 | `[2,4,2,1]` | `[8,16,64,512]` |
| S2 | `[4,2,1,1]` | `[8,32,64,512]` |

spatial stride 均为 `[8]`。writer base 随 `col_start` 移动，使完整 1024-column
activation 留在 banks 48..63，并可直接作为同一 shape 的 Mode1 A。

## 8. Mode1 streamer

### 8.1 Activation reader，banks 48..63

| Shape | spatial stride | temporal bounds | temporal strides |
|---|---|---|---|
| S0 | `[8,16]` | `[128,4,1,1,1,1]` | `[512,0,0,0,0,0]` |
| S1 | `[8,16]` | `[2,64,2,1,1,1]` | `[64,512,0,0,0,0]` |
| S2 | `[8,32]` | `[2,2,32,1,1,1]` | `[16,64,512,0,0,0]` |

### 8.2 W2L/W2R readers

Mode1 `panel_span=16384`：

| Shape | spatial stride | temporal bounds | temporal strides |
|---|---|---|---|
| S0 | `[8,16384]` | `[4,32,4,1]` | `[16,512,16384,0]` |
| S1 | `[8,16384]` | `[4,32,2,1]` | `[16,512,32768,0]` |
| S2 | `[8,16384]` | `[4,32,1,1]` | `[16,512,65536,0]` |

Mode1 的 spatial bounds 和 channel masks 仍分别是 `[2,4]` 与
`0x03/0x0f/0xff`，panel 组合方式与第 7.2 节完全相同。区别只有 K 从 2048 降到
1024：一个 panel 缩为 32 rows，K-group temporal bound 从 64 变为 32，panel
span 从 32768 B 变为 16384 B。

### 8.3 D0/D1 writers，banks 0..15

| Shape | temporal bounds | temporal strides |
|---|---|---|
| S0 | `[1,8,4,1]` | `[512,8,512,0]` |
| S1 | `[2,4,2,1]` | `[512,8,1024,0]` |
| S2 | `[4,2,1,1]` | `[512,8,2048,0]` |

D0 base 位于 bank 0，D1 base 位于 bank 8；两者每个 token 各产生 1024 个 int16，
合起来是逻辑 2048-column 输出。

## 9. 日志与正确性约定

weight staging 只允许出现在第一个 `SHAPE_BEGIN` 之前：

```text
WEIGHT_RESIDENT_LAYOUT ...
WEIGHT_DMA_CHUNK mode=0 ...       # 64 lines
WEIGHT_DMA_SUMMARY mode=0 ...
WEIGHT_DMA_CHUNK mode=1 ...       # 64 lines
WEIGHT_DMA_SUMMARY mode=1 ...
WEIGHTS_RESIDENT_READY ... no_more_weight_dma=1
```

之后每个 shape 只出现一个 `TOKEN_DMA`，每个计算 chunk 都立即 compare。成功结束应为：

```text
FINAL_RESULT selected_shape=-1 chunk_cols=16 status=PASS total_errors=0
```

检查是否意外在 shape 内重搬 weight：

```bash
grep -nE 'WEIGHT_DMA|SHAPE_BEGIN|FINAL_RESULT' \
  layout_explore_logs/common16_resident_k1024_20260718/all_shapes_2desc.log
```

所有 `WEIGHT_DMA_*` 必须位于第一个 `SHAPE_BEGIN` 之前。

## 10. 构建和运行

这是纯软件/datagen 改动；硬件 cfg、生成 RTL、CSR header 和现有 VLT 均不变。
目标 app 使用默认：

```text
SELECT_SHAPE=-1
CHUNK_COLS=16
RUN_MODE1=1
```

2026-07-18 已完成 clean datagen；2026-07-19 将每 chunk 的 weight DMA 从八笔
panel descriptors 合并为两笔 chunk-wide descriptors 后，已在 `barnard3` 中使用目标
cfg 完成完整 `make sw`。生成 header 确认：

```text
K0_TOTAL=2048, N0_TOTAL=1024
K1_TOTAL=1024, N1_TOTAL=1024
MODE0_WEIGHT_DATA_LENGTH=1,048,576 B  # W 或 V 单个数组
MODE1_WEIGHT_DATA_LENGTH=524,288 B    # W2L 或 W2R 单个数组
```

静态最大地址检查：

```text
Mode0 last byte exclusive = 4,194,176 < 4,194,304
Mode1 last byte exclusive = 6,291,328 < 6,291,456
resident flat end         = 6,291,456 < 8,388,608
```

目标 ELF 已通过 RISC-V clang/link；仅有 dual-versacore 现有 header 的
`-Wstatic-in-inline` 告警。硬件 cfg、RTL 和 VLT 均未变化，因此复用 2026-07-18
构建的目标 VLT。旧的 `all_shapes.log` 是合并 descriptor 之前的完整 PASS baseline，
日志中每 chunk 的 `repeat=64` 表示一笔 panel descriptor；必须保留，不要覆盖。

新版日志应显示：

```text
Mode0 WEIGHT_DMA_CHUNK: descriptors=2 chunk_bytes_per_tensor=16384 repeat=256
Mode0 WEIGHT_DMA_SUMMARY: chunks=64 commands=128
Mode1 WEIGHT_DMA_CHUNK: descriptors=2 chunk_bytes_per_tensor=8192 repeat=128
Mode1 WEIGHT_DMA_SUMMARY: chunks=64 commands=128
```

完整 VLT 功能仿真留给下面的一键命令执行，因为仍有三个 shape 的 384 次 accelerator
invocation，运行较长。

一键仿真命令会在 host 上通过 `podman exec barnard3` 运行，并把实时输出保存到：

```text
layout_explore_logs/common16_resident_k1024_20260718/all_shapes_2desc.log
```

进入一个新的 tmux session：

```bash
tmux new -s common16_2desc
```

然后在 tmux 内执行：

```bash
cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster && \
mkdir -p layout_explore_logs/common16_resident_k1024_20260718 && \
podman exec barnard3 bash -lc \
  'cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster && \
   stdbuf -oL -eL ./target/snitch_cluster/bin/snitch_cluster.vlt \
   ./target/snitch_cluster/sw/apps/snax-versacore-int16x4-multishape-k8-common16-chunked/build/snax-versacore-int16x4-multishape-k8-common16-chunked.elf' \
  2>&1 | tee layout_explore_logs/common16_resident_k1024_20260718/all_shapes_2desc.log
```

按 `Ctrl-b`、再按 `d` 可 detach；重新进入使用：

```bash
tmux attach -t common16_2desc
```

不进入 tmux、只从另一个终端实时看日志：

```bash
tail -f /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster/layout_explore_logs/common16_resident_k1024_20260718/all_shapes_2desc.log
```

本 workload 仍然很长。默认不加 `--vcd`；完整 VCD 会显著拖慢仿真并产生很大的
`sim.vcd`。

## 11. HeMAiA 移植注意事项

> **给 HeMAiA MoE workload 开发者的移植边界：**本实验真正可以复用的是
> **TCDM 上的 layout 规则、各 streamer 的 bound/stride/channel-mask 配置，以及
> DMA 的 chunk/ping-pong 搬运 pattern**。本文列出的 `A_BASE`、B/D base、
> `region_offset`、resident slot 起始地址等具体数值只是当前 SNAX app 的一种实例，
> 不能直接复制到 HeMAiA；这些起始地址必须由 HeMAiA 自己的 datagen/layout 脚本或
> runtime 根据实际 L1 arena、expert/cluster、ping-pong buffer 和 tensor lifetime
> 统一推导、维护并传给 kernel。换句话说，应复用“相对排布和访问规律”，而不是复用
> “这次实验中的绝对地址常量”。

1. 逻辑 workload 与物理拆分必须区分：W2 是一个 `[1024,2048]` 逻辑矩阵，物理上
   分为两个 `[1024,1024]` reader streams。
2. SNAX resident 实验的 depth-slot 相对布局可以复用：全部驻留时保留所有 slots，
   多级预取时循环使用有限 slots，严格双缓冲时才只保留 slot 0；任何覆盖都必须由
   DMA/compute 同步控制生命周期。
3. streamer 的 bound/stride 可直接复用；每次 invocation 只更新 B base、D base，
   若 CSR retention 已验证，还可避免重复写其他 CSR。
4. datagen 应分别生成逻辑 tensor、canonical S0 4-column panels 和物理地址 manifest，
   不要在多个 app 内复制地址常量。
5. 必须保留容量断言、bank-range 断言和 segment overlap 检查；本版 resident flat end
   为 6 MiB，TCDM capacity 为 8 MiB。

## 12. Datagen 的角色、入口与产物

本实验的 datagen 位于：

```text
target/snitch_cluster/sw/apps/
snax-versacore-int16x4-multishape-k8-common16-chunked/data/
├── params.hjson
├── datagen.py
├── Makefile
└── data.h                 # 生成物，禁止手改
```

`params.hjson` 只描述逻辑尺寸：

```hjson
M_total: 8
K0_total: 2048
N0_total: 1024
K1_total: 1024
N1_total: 1024
```

`data/Makefile` 调用 `datagen.py --swcfg ... --hwcfg ...`，stdout 重定向成
`data.h`。生成器同时读取目标 HJSON，检查 sparse interconnect 以及 reader/writer
temporal dimension。也就是说，`data.h` 不只是测试向量，还固化了“这份数据对应哪种
streamer 结构”的契约；硬件配置不匹配时应在 datagen 阶段直接失败。

`data.h` 包含：

| 类别 | C symbol | 逻辑含义 | 字节数 |
|---|---|---|---:|
| token | `A` | `[8,2048]` int16，普通 token-major L3 数组 | 32,768 |
| Mode0 weight | `W` | `[2048,1024]` packed int4 | 1,048,576 |
| Mode0 weight | `V` | `[2048,1024]` packed int4 | 1,048,576 |
| Mode1 left | `W2_left` | 逻辑 W2 的 columns 0..1023 | 524,288 |
| Mode1 right | `W2_right` | 逻辑 W2 的 columns 1024..2047 | 524,288 |
| golden | `S{0,1,2}_mode0_token_golden` | Mode0 token-major 输出 | 随 active M 变化 |
| golden | `S{0,1,2}_mode1_d{0,1}_token_golden` | 两个 Mode1 writer 输出 | 随 active M 变化 |
| metadata | `shape_cfgs` | shape、tile 数、channel mask、golden pointer | 3 entries |

这些 `static const` 数组随 ELF 位于 L3/source memory。它们不是已经按 TCDM bank
地址展开的 8 MiB image；runtime 的 DMA 才把 canonical source stream scatter 到
正确 bank/row。HeMAiA datagen 也应保持这一区分：

```text
logical tensors -> canonical packed source blobs -> placement manifest
                                                   -> runtime DMA
```

不要让 Python 直接生成一个充满 padding 的完整 TCDM dump，否则逻辑数据、source
packing 和 runtime placement 会粘在一起，shape 或 bank mapping 一改就很难复用。

### 12.1 Source-of-truth 索引

下一位开发者应优先对照下面这些函数，而不是从生成后的巨大 `data.h` 搜索字节：

| 层次 | 文件/函数 | 负责内容 |
|---|---|---|
| logical params | `data/params.hjson` | M/K/N 尺寸 |
| int4 ABI | `datagen.py: pack_int4()` | signed int4 与 low/high nibble 顺序 |
| test values | `make_a()`、`make_sparse_s0_weights()` | deterministic token/weight |
| shape reinterpret | `regroup_s0_panels()` | S0 panels 如何组成 S1/S2 meshCol |
| arithmetic | `block_gemm()`、`make_goldens()` | Mode0/Mode1 golden 与 saturate 顺序 |
| generated ABI | `shape_cfg()`、`emit_header()` | C arrays、mask、tile metadata、HW assert |
| token placement | C app `stage_tokens()` | contiguous L3 token 到 banks 0..15 |
| weight placement | C app `weight_base()`、`stage_weight_chunks()` | region/slot/ping-pong 与 2D DMA |
| Mode0 consumer | C app `run_mode0_chunk()` | A/B/D streamer 和每 chunk 启动 |
| Mode1 consumer | C app `mode1_a_cfg()`、`run_mode1_chunk()` | activation reader、W2 readers、双 writer |
| checking | C app `check_mode0_chunk()`、`check_mode1_chunk()` | per-chunk token-major compare |

C app 的完整路径是：

```text
target/snitch_cluster/sw/apps/
snax-versacore-int16x4-multishape-k8-common16-chunked/src/
snax-versacore-int16x4-multishape-k8-common16-chunked.c
```

正常 app build 会按 `data/Makefile` 的依赖自动重新生成 `data.h`。只改 datagen 或
params 后需要强制刷新时，先运行 app Makefile 的 `clean-data` target，再走正常 build；
不要直接修改 `data.h`，也不要把它当作 HeMAiA 端的长期 source-of-truth。

## 13. 逻辑 tensor 和可重复测试数据

### 13.1 Token A

`A` 的 L3 排布是标准 C-order `[token][k]`，没有 TCDM row padding：

```text
A[token,k] = ((token * 5 + k * 3) % 11) - 5
L3 byte address = &A[0][0] + (token * 2048 + k) * 2
```

DMA 到 TCDM 后才变为第 7.1 节所述的
`A_BASE + token*16 + k_tile*512`。因此需要明确区分：

- L3 source row：一个 token 连续占 `2048*2=4096 B`；
- TCDM consumer row：同一 K tile 的 8 个 token 并排，占 banks 0..15；
- 一个 token 的相邻 K tile 在 TCDM 中相隔 512 B。

每个 shape 都重新 DMA 全部 8 个 token，随后用 mask 选择前 8/4/2 个。这样三种
shape 使用相同输入前缀，且能检查刷新是否覆盖了之前 Mode1 写到 banks 0..15 的结果。

### 13.2 Sparse deterministic weights

四个物理 weight stream 都由 `make_sparse_s0_weights()` 生成。canonical shape 是：

```text
[panel, k_tile, col_in_panel, k_lane]
[N/4,   K/8,    4,            8]
```

每个 output column 只选择一个 non-zero K tile：

```text
active_k = (panel*(7+seed) + col*(11+seed) + seed) % k_tiles
magnitude = 1 + ((panel + col + k_lane + seed) & 1)
```

各 tensor 使用不同 seed/sign：

| Tensor | seed | sign |
|---|---:|---:|
| W | 1 | +1 |
| V | 3 | +1 |
| W2_left | 5 | +1 |
| W2_right | 7 | -1 |

稀疏数据用于控制累加值范围，同时让 panel、chunk、左右 half 或 ping/pong 放错位置
立即形成数值 mismatch；当前 NumPy `tensordot` 仍按 dense tensor 计算，并不会因为
稀疏而减少 golden 的运算量。它不是性能 workload 的真实权重分布；评估带宽/功耗时
可以替换数值，但必须保持完全相同的 packed layout 和地址 manifest。

## 14. int4 编码与 canonical panel 字节顺序

### 14.1 Signed int4 nibble 约定

所有 weight 元素范围为 `[-8,7]`，按 4-bit two's-complement 保存。展平后的相邻两个
元素组成一个 byte：

```text
byte[j] = (value[2*j]   & 0x0f)
        | ((value[2*j+1] & 0x0f) << 4)
```

即第 0 个元素在 low nibble，第 1 个元素在 high nibble。例如 `[-1,2]` 编码为
`0x2f`。HeMAiA 的 packer 必须保持同样顺序；high/low nibble 对调不会触发地址错误，
但所有计算值都会错，是最难从 DMA trace 中发现的一类问题。

### 14.2 C-order flatten 顺序

pack 前按下面顺序展平，最右侧 `k_lane` 变化最快：

```text
for panel in range(N/4):
  for k_tile in range(K/8):
    for col_in_panel in range(4):
      for k_lane in range(8):
        emit weight[panel,k_tile,col_in_panel,k_lane]
```

一个 K tile 对一个 4-column panel 有 `4*8=32` 个 int4，即 16 B；连续 4 个 K
tiles 正好组成一次 64 B DMA beat：

```text
source byte  0..15  -> k_tile 4*g + 0
source byte 16..31  -> k_tile 4*g + 1
source byte 32..47  -> k_tile 4*g + 2
source byte 48..63  -> k_tile 4*g + 3
```

因此一个 full-K panel 的 source payload 为：

```text
Mode0: 256 k_tiles * 16 B = 4096 B
Mode1: 128 k_tiles * 16 B = 2048 B
```

DMA 把每个连续 64 B source beat 放进相隔 512 B 的 TCDM row。B streamer 的 inner
`bound=4,stride=16` 再从该 row 中依次取回四个 K tiles；outer K-group dimension 用
`stride=512`。这就是第 5 节 DMA 参数和第 7.2/8.2 节 streamer stride 能对上的原因。

### 14.3 三种 shape 共享同一份 source bytes

物理 source 永远只有上述 S0 4-column panels，datagen 不为 S1/S2 重排或复制 weight。
`regroup_s0_panels()` 仅在 Python golden 中模拟硬件对相邻 panel 的组合：

```text
S0 meshCol=4:  q=1，1 panel / N tile
S1 meshCol=8:  q=2，2 adjacent panels / N tile
S2 meshCol=16: q=4，4 adjacent panels / N tile
```

其等价 reshape/transpose 为：

```text
(n_tile, q, k_tile, 4, 8)
    -> (n_tile, k_tile, q, 4, 8)
    -> (n_tile, k_tile, meshCol, 8)
```

resident TCDM 里的 bytes 在切换 shape 时完全不动；变化的只有 B reader 的
channel mask、`q` 对应的 temporal bound/stride，以及 invocation base。HeMAiA
实现必须保留这个性质，否则“所有 shape 共用常驻权重”的目标就被破坏了。

## 15. Golden model 的精确计算链

Golden 使用整数运算，不应以浮点 PyTorch SiLU 或一次性高精度 GEMM 近似替代。

### 15.1 Mode0 / SwiGLU

对每种 shape 先取 `A[:meshRow]`，转为 accelerator 消费的
`[k_tile,meshRow,tileSize]` 顺序，再分别计算：

```text
vc0_i32 = A @ W
vc1_i32 = A @ V
w16     = sat_i16(vc0_i32)
v16     = sat_i16(vc1_i32)
silu16  = silu_out16_balanced_eval_q(w16)
mode0   = sat_i16(int32(silu16) * int32(v16))
```

`sat_i16(x)=clip(x,-32768,32767)`。这里有三处不可随意合并：两路 GEMM 后各自先
saturate，W 路再进入 hardware-matched balanced SiLU，最后乘法后再 saturate。
输出从 array order `[n_tile,meshRow,meshCol]` transpose 为 token-major
`[meshRow,1024]`，与 banks 48..63 中 Mode1 activation 的逻辑顺序一致。

### 15.2 Mode1 / W2

Mode0 token-major golden 被重新解释为 Mode1 A stream：

```text
[meshRow,1024]
  -> [meshRow,128,8]
  -> [128,meshRow,8]
```

随后分别执行：

```text
d0 = sat_i16(mode0 @ W2_left)
d1 = sat_i16(mode0 @ W2_right)
```

两者分别转回 `[meshRow,1024]` token-major，匹配 D0 banks 0..7 和 D1 banks
8..15。逻辑上把 `d0 || d1` 在最后一维拼接，才是 `[meshRow,2048]` 的完整 W2
输出；不能把 W2_left/right 当成两个 expert 或两个时间 tile。

### 15.3 Compare 粒度

app 每完成一个 16-column chunk 就只 compare 该 chunk 覆盖的 token/column window，
并打印 PASS/FAIL。这能把错误定位到：

```text
(shape, mode, chunk, token, logical column, left/right writer)
```

HeMAiA correctness workload 建议保留相同粒度的 compare 或 checksum，即使 DFG node
保持粗粒度；不要为了对齐 DFG 而只在整个 expert 结束后给一个总 PASS。

## 16. `shape_cfgs` 与 runtime 的职责边界

datagen 为每种 shape 输出以下 metadata：

| 字段 | S0 | S1 | S2 | 含义 |
|---|---:|---:|---:|---|
| `array_shape` | 0 | 1 | 2 | VersaCore shape CSR |
| `meshRow` | 8 | 4 | 2 | active token 数 |
| `tileSize` | 8 | 8 | 8 | K tile 固定为 8 |
| `meshCol` | 4 | 8 | 16 | 一次 array N 宽度 |
| `K0_tiles` | 256 | 256 | 256 | Mode0 full K |
| `N0_tiles` | 256 | 128 | 64 | 完整 N=1024 的 array tiles |
| `K1_tiles` | 128 | 128 | 128 | Mode1 full K |
| `N1_tiles` | 256 | 128 | 64 | 每个 W2 half 的 N tiles |
| `q_shape0_cols` | 1 | 2 | 4 | 每个 N tile 合并的 S0 panels |
| `A_channel_en` | `ffff` | `00ff` | `000f` | 8/4/2 tokens |
| `B_channel_en` | `03` | `0f` | `ff` | 1/2/4 panel channel groups |
| `D_channel_en` | `1` | `1` | `1` | writer enable |

datagen 不生成每个 chunk 的绝对 TCDM 地址；这些地址由 runtime 根据
`mode/chunk/tensor` 计算。这一边界很重要：逻辑 shape metadata 可以跨平台复用，
而 HeMAiA 的 bank base、arena offset 和 ping/pong lifetime 应进入独立 manifest/layout
模块，不能硬编码回 tensor generator。

## 17. 推荐的 placement manifest

为了让下一阶段不再从 C 地址公式逆向推断，HeMAiA datagen/layout 应显式产生一份
machine-readable manifest，并由 Python 检查器与 device kernel 共用。每个 weight
segment 至少包含：

```text
name, mode, tensor, logical_shape, source_symbol
element_bits, nibble_order, source_layout
panel_cols, panel_count, panel_payload_bytes, panel_span
chunk_cols, panels_per_chunk, chunk_count
region_offset, ping_bank_offset, pong_bank_offset
chunk_slot_span, slots_per_buffer, lifetime
consumer_reader, allowed_shapes
```

对本实验可实例化为：

| Segment | panels | panel payload | panel span | region offset | ping base | pong base | resident lifetime | last consumer |
|---|---:|---:|---:|---:|---:|---:|---|---|
| W | 256 | 4096 B | 32768 B | 0 | bank16 | bank32 | stage 前至完整 sequence 结束 | S2 Mode0 |
| V | 256 | 4096 B | 32768 B | 0 | bank24 | bank40 | stage 前至完整 sequence 结束 | S2 Mode0 |
| W2_left | 256 | 2048 B | 16384 B | 4 MiB | bank16 | bank32 | stage 前至完整 sequence 结束 | S2 Mode1 |
| W2_right | 256 | 2048 B | 16384 B | 4 MiB | bank24 | bank40 | stage 前至完整 sequence 结束 | S2 Mode1 |

这里的 bank offset 是相对每个 512 B row 的 byte offset：bank16/24/32/40 分别为
128/192/256/320 B。manifest checker 至少应验证：

1. tensor payload、panel 数和逻辑 shape 严格相等；
2. 每个 destination 64 B beat 完整落在指定 8-bank group；
3. 同时 live 的 segment 不重叠；
4. 最大地址小于 arena/TCDM capacity；
5. chunk 数能被 ping/pong 或 resident slot 规则覆盖；
6. S0/S1/S2 都只引用同一份 source symbol；
7. Mode1 region 不覆盖 Mode0 region。

## 18. 从 manifest 生成 DMA descriptor

当前 SNAX correctness app 的通用 16-column chunk descriptor 是：

```text
first_global_panel = chunk * panels_per_chunk
chunk_payload_bytes = panels_in_chunk * panel_payload_bytes

src = source_symbol + first_global_panel * panel_payload_bytes
dst = region_offset
    + buffer_bank_offset(tensor, chunk & 1)
    + (chunk >> 1) * chunk_slot_span

size       = 64 B
src_stride = 64 B
dst_stride = 512 B
repeat     = chunk_payload_bytes / 64
```

在本版 resident app 中，16-column chunk 有 4 panels，但 panel boundary 不改变
source/destination stride，因此可以合并。每 mode 每 chunk 只有 B0/B1 两个 2D
descriptor；64 chunks 是 128 descriptors。Mode0/Mode1 总计 256 个 weight
descriptors，并且只在程序开头执行一次。

这不是 HeMAiA 最终实现必须照搬的 descriptor 数。可优化方式是先根据 manifest 找出
destination 地址连续满足同一 affine 关系的最大 segment，再为每段生成一笔 2D DMA。
但必须注意：resident 实验的 chunk 同时占不同 depth slot，且 chunk 奇偶切换 bank
group，所以整个 W 或 V 通常不是一个单一 affine destination。若要减少 descriptor，
可以选择：

- 将 source 预打包成 ping segments 和 pong segments，每个 segment 内 panel/slot
  顺序与 destination affine 顺序一致；
- 分别为 W-ping、W-pong、V-ping、V-pong 生成较大的 descriptor；
- 在真实双缓冲中只保留固定 ping/pong base，每次搬当前 chunk，此时 descriptor 数
  应按 overlap 粒度评估，而不是按“全部 resident”静态镜像评估；
- 将 B0/B1 source 以 64 B beat 交错成 128 B records，但这会改变 source contract，
  packer、descriptor generator 和 trace checker 必须一起改。

不能只因为 API 名为 1D 就假定能表达上述 scatter。这里每 64 B payload 的 destination
跨到下一条 512 B TCDM row，需要 `repeat + dst_stride`，在 SNRT/iDMA API 语义上使用
2D descriptor；source 自身是连续的，只有 destination 带 row stride。

## 19. HeMAiA `multi_cluster_MoE_test` 的具体接入建议

当前 HeMAiA 相关文件的完整位置是：

```text
/esat/studscratch/r1015498/Thesis/hemaia_eval/HeMAiA/
target/sw/host/apps/offload_bingo_hw/single_chip/workloads/
multi_cluster_MoE_test/
├── params.hjson
├── multi_cluster_MoE_test_datagen.py
├── moe_test_layout.py
└── main_bingo.py
```

移植时建议按以下职责拆分：

### 19.1 `params.hjson`

先把逻辑 contract 设为：

```text
hidden_size       = 2048
intermediate_size = 1024
token capacity    = 8
shape active M    = [8,4,2]
common chunk cols = 16
```

W2 仍应在逻辑层表示 `[1024,2048]`，到物理 stream/manifest 层才拆成两个
`[1024,1024]`。参数命名要避免把 `N1=1024` 误解成完整 W2 输出宽度。

### 19.2 `multi_cluster_MoE_test_datagen.py`

保留它已有的 low-nibble-first `pack_int4()`，但将 weight 生成和 flatten 顺序替换/验证
为第 13、14 节的 canonical panel contract。建议明确输出：

```text
tokens_i16
expert_W_packed_u8
expert_V_packed_u8
expert_W2_left_packed_u8
expert_W2_right_packed_u8
shape_cfg metadata
placement manifest
golden/checksum per shape/mode/chunk
```

如果真实 MoE 有多个 expert，最外层可以增加 `expert` dimension，但每个 expert 内仍保持：

```text
[W][V][W2_left][W2_right] as separate named blobs
each blob: [panel][k_tile][col_in_panel][k_lane]
```

不要沿用旧 generator 中仅适合旧尺寸的随机 reshape/transpose 而不做字节级检查。
生成后至少抽查 panel0 的前 64 B，unpack 回 int4，并与原 tensor 的
`[panel0,k_tile0..3,:,:]` 比较。

### 19.3 `moe_test_layout.py`

让它成为物理布局的唯一事实来源：根据 params 推导第 17 节 manifest、所有 base、
span、arena end 和 overlap assertion。当前 workflow 中若 gate/up 区域会被 next expert
或 down weight 复用，需要取消这类 reuse，直到本实验要求的 S0/S1/S2 Mode1 全部结束；
“全部权重驻留”意味着 W/V/W2 四个物理 stream 的 lifetime 同时覆盖所有 shape。

HeMAiA 的 L1 应只分配一个 512 B aligned parent arena，再用 `BingoMemView` 建子视图；
不要为每个 manifest segment 单独 malloc 后假设它们物理相邻。布局模块必须返回
`arena_bytes`，并断言它不超过目标 cluster 可用 L1。

### 19.4 `main_bingo.py` 与 device kernel

保持粗粒度 DFG/task，例如“stage resident expert weights”与“run one shape/mode”。不要把
64 chunks 展开成 64 个 DFG nodes；真实的 chunk loop、ping/pong base 更新、CSR start/wait
和 per-chunk compare 应放在 device kernel 内。这样既保留 DFG 可读性，也能准确复现
SNAX 的每 16-column invocation 行为。

需要传给 kernel 的不应只是旧的 `block_count/block_bank_stride`，而应是 manifest
派生的 region base、panel/chunk span、chunk count、ping/pong bank offset，以及当前
shape 的 streamer cfg。若结构体参数过长，可把只读 manifest/config table 放在 L3，
kernel 只接收 table pointer、expert id、shape id 和 mode。

还要区分两个实验：

- 本文的 correctness sequence 是同一个 expert 上依次运行 S0/S1/S2；
- 现有 HeMAiA MoE DFG 中的节点/阶段命名可能代表不同 expert 或流水阶段。

移植时可以先做一个专用 correctness workload 完整复现本文 sequence，通过后再把相同
manifest 与 kernel 接入原 MoE DFG。不能仅因都叫 S0/S1/S2 就假定两边调度语义相同。

## 20. HeMAiA 开发与验收清单

建议下一个 agent 按下面顺序推进，每一步都留下可独立检查的产物：

1. 用新 params 生成逻辑 W/V/W2 和 token，检查三个逻辑 weight 都是 1 MiB。
2. 检查 W2 物理拆成两个 512 KiB blobs，拼接后的逻辑输出宽度为 2048。
3. 对四个 blobs 做 int4 pack/unpack round-trip，确认 low nibble first 和 signed decode。
4. 输出 canonical panel manifest，确认 Mode0/Mode1 panel 分别为 4096/2048 B。
5. 生成 layout，打印每个 segment 的 first/last address、bank range、row range、lifetime。
6. 静态检查 Mode0 与 Mode1 weight region 不重叠，且所有 shape 引用相同 resident blobs。
7. 只运行 weight stage kernel，用 trace 检查 64 B payload 和 512 B destination stride。
8. 运行 S0 Mode0 的首个 chunk，逐元素 compare；再检查最后一个 chunk，排除 slot/base
   只在边界处出错。
9. 扩展到 S1/S2，重点检查相邻 2/4 panels 的组合顺序，没有重新 DMA weight。
10. 接入 Mode1，确认 activation 从 banks 48..63 读取，D0/D1 写 banks 0..7/8..15。
11. 执行完整 sequence，日志中 resident weight stage 只能出现一次。
12. 最后才启用真实 ping/pong overlap，并用 barrier/event trace 证明 DMA 不覆盖正在消费
    的 buffer。

最终验收日志至少应打印：参数摘要、source blob 大小、manifest、每笔/每组 DMA 摘要、
每个 shape/mode/chunk 的 compare 结果、每段计算周期，以及全局
`status=PASS total_errors=0`。若数值 mismatch，先按以下顺序排查：nibble 顺序、panel
flatten、global panel index、ping/pong bank base、resident slot、S1/S2 panel regroup、
Mode1 left/right writer，最后再检查算术 golden。
