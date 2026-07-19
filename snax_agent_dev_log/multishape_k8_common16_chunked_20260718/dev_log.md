# K8 多 shape 公共 16-column chunk / 真实逐 chunk 启动实验

> 2026-07-18 后续 resident-weight 版本已将中间维度改为 1024，并让 Mode0/Mode1
> 权重一次搬入、跨三个 shape 常驻。新设计、旧 run 分析和完整参数见
> `resident_k1024.md`。本文其余内容保留为旧的 1408/1024 baseline 记录。

日期：2026-07-18

## 1. 实验目标

为 MoE dynamic shape 建立一个所有 array shape 都能消费的 weight 搬运粒度。硬件
shape 为：

| 名称 | `array_shape` | `(meshRow,tileSize,meshCol)` | 一次处理 token | 16 columns 内 N tiles |
|---|---:|---:|---:|---:|
| S0 | 0 | `(8,8,4)` | 8 | 4 |
| S1 | 1 | `(4,8,8)` | 4 | 2 |
| S2 | 2 | `(2,8,16)` | 2 | 1 |

因此选择第三个 shape 的 `meshCol=16` 作为最小公共 chunk。无论运行 S0/S1/S2，
一个 16-column chunk 总是包含四个相邻的 canonical S0 4-column full-K panels。

完整 workload 不变：

```text
Mode0: K0=2048, N0=1408
Mode1: K1=1408, N1=1024
```

默认 `CHUNK_COLS=16`，所以 Mode0/Mode1 分别执行 88/64 次 accelerator start。
每次 invocation 都保留完整 K，只沿 N/输出列切分；不能沿 K 切分，因为 accelerator
内部累加且没有 external partial-sum input。

## 2. “16 的倍数是否只改变 base”结论

### 2.1 对 B streamer

在下面的 layout contract 成立时，答案基本是“是”：

1. L3 继续采用 canonical S0 顺序：每个 4-column panel 存完整 K。
2. L1 chunk 内的 S0 panels 连续排列。
3. 每个 panel 内保持相同的 64 B row / 512 B physical pitch。
4. chunk 大小是 16 columns 的倍数，所以始终包含完整的 1/2/4-panel shape group。

此时对同一 mode：

- B spatial stride 不变；
- K temporal stride 不变；
- shape-N temporal stride 不变；
- 固定粒度的不同 invocation 只改变 B base；
- 如果比较不同 `CHUNK_COLS`，B stride 仍不变，但 N bound 会变成
  `CHUNK_COLS / meshCol`。

所以不能把整个 streamer/accelerator 概括成“只改 B base”：A 的 broadcast bound、
D writer bound、D base 和 accelerator output bound也随 chunk 改变。最后一个 tail
如果尺寸不同，也要更新 bounds。

### 2.2 Mode0 S2 writer 的额外限制

既有 Mode0 token-striped layout 每个 physical row 用 banks 48..63：S2 的一个
16-column N tile 只占其中 64 B，两个相邻 N tiles 才填满一行。因此：

- `CHUNK_COLS=16`：每个 invocation 用一个 half-row，D base 在同一行的 `+0/+64`
  间交替；
- `CHUNK_COLS` 为 32 的倍数：每个 invocation 含偶数个 S2 N tiles，可直接用成对
  writer bound；
- 48、80 等奇数个 16-column group 若要求“一整个 chunk 只发一条 accelerator
  command”，单个 rectangular 4D writer AGU 不能同时自然表达开头/结尾 half-row。

因此本 app 接受 `CHUNK_COLS=16` 或 32 的倍数。这个限制来自保留现有 Mode0-D
layout，不是 B streamer 的限制；B 本身支持任意 16-column 倍数。

## 3. 使用配置与 design-time 事实

统一 cfg：

```text
cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson
```

该 cfg 的 sparse-interconnect access granularity 为：

```text
[16,1], [8,1], [8,1], [1,1], [1,1]
```

所以本实验相关的 bank routing granularity 都是 1。注意 cfg 中另外还有 VersaCore
内部 serialization 参数：

```text
granularity_a=4, granularity_b=2, granularity_c_d=2
```

这两类 granularity 不应混淆。streamer design-time dimensions 为 A/B0/B1=`6/4/4`，
D0/D1=`4/4`，足够本实验使用。

## 4. TCDM bank contract

| 数据/阶段 | bank | base |
|---|---:|---:|
| Mode0 A/token | 0..15 | 0 |
| B0 ping | 16..23 | 128 |
| B1 ping | 24..31 | 192 |
| B0 pong | 32..39 | 256 |
| B1 pong | 40..47 | 320 |
| Mode0 D0 / Mode1 A | 48..63 | 384 |
| Mode1 D0 | 0..7 | 0 |
| Mode1 D1 | 8..15 | 64 |

Mode1 开始后，原始 Mode0 A 已经不再使用，所以 Mode1 writers 可以覆盖 banks
0..15。这样 Mode1 阶段形成完全分区：D=`0..15`，B=`16..47`，A=`48..63`。

## 5. 预搬完实验和真实双缓冲的 base 区别

SNAX cluster 当前不能做目标中的真实 DMA/compute overlap。本 app 先把全部 chunk
搬入 L1，但随后真的每个 chunk 单独发送 streamer/accelerator start。

为同时保留所有预搬数据，第 `c` 个 chunk 使用：

```text
slot = floor(c / 2)
even c: ping_base + slot * chunk_slot_span
odd  c: pong_base + slot * chunk_slot_span
```

真正 HeMAiA ping-pong 在时间上覆盖旧数据，只需：

```text
even c: 固定 ping_base
odd  c: 固定 pong_base
```

即 SNAX app 的 streamer stride/bound 可以照搬；移植 datagen/runtime 时必须去掉
`slot * chunk_slot_span`，并由同步逻辑保证 DMA 不覆盖正在计算的 buffer。

默认 16-column chunk 含四个 S0 panels：

| Mode | 单 panel valid bytes | panel span | chunk slot span |
|---|---:|---:|---:|
| 0 | 4096 | 32768 | 131072 |
| 1 | 2816 | 22528 | 90112 |

## 6. DMA 参数

### 6.1 Token

所有8个 token 在每个 shape 开始前重新搬入；shape channel mask 只消费8/4/2个：

```text
每 token 一笔 2D DMA
size=16 B
src_stride=16 B
dst_stride=512 B
repeat=256
dst=A_BASE + token*16
```

### 6.2 当前 separated B0/B1 L3 layout

每个 4-column full-K panel 对 B0/B1 各发一笔命令：

| Mode | `size` | `src_stride` | `dst_stride` | `repeat` |
|---|---:|---:|---:|---:|
| 0 | 64 | 64 | 512 | 64 |
| 1 | 64 | 64 | 512 | 44 |

16-column chunk 共四个 panels，因此是8笔命令：

```text
B0_panel0, B1_panel0, ..., B0_panel3, B1_panel3
```

app 每累计8笔等待一次，适配当前 `dma_req_fifo_depth=8`。每个 `DMA_CHUNK` 和
最终 `DMA_SUMMARY` 都会打印 cycles 和实际 base。

如果 HeMAiA datagen 将 L3 做成逐 row 的 B0/B1 交织，也可对每个 panel 使用：

```text
size=128 B, src_stride=128 B, dst_stride=512 B, repeat=64/44
```

这样一个 512-bit beat 给 B0 banks，下一 beat 给 B1 banks。该选择改变 DMA/source
packing，不改变下面的 B streamer contract。

## 7. `CHUNK_COLS=16` streamer 参数

所有 stride 单位为 byte，未列出的 temporal dimensions 为 `bound=1,stride=0`。

### 7.1 Mode0

| Shape | A spatial stride | A temporal bound | A temporal stride |
|---|---|---|---|
| S0 | `[8,16]` | `[256,4,1,1,1,1]` | `[512,0,0,0,0,0]` |
| S1 | `[8,16]` | `[256,2,1,1,1,1]` | `[512,0,0,0,0,0]` |
| S2 | `[8,16]` | `[256,1,1,1,1,1]` | `[512,0,0,0,0,0]` |

| Shape | B spatial stride | B temporal bound | B temporal stride | mask |
|---|---|---|---|---:|
| S0 | `[8,32768]` | `[4,64,4,1]` | `[16,512,32768,0]` | `0x03` |
| S1 | `[8,32768]` | `[4,64,2,1]` | `[16,512,65536,0]` | `0x0f` |
| S2 | `[8,32768]` | `[4,64,1,1]` | `[16,512,131072,0]` | `0xff` |

与上次单次 full-N invocation 不同，这里 B bound/stride 中没有 ping/pong 维度。
每次 invocation 的 B base 已经直接指向当前完整 chunk。

| Shape | D0 temporal bound | D0 temporal stride | chunk `c` 的 D base |
|---|---|---|---|
| S0 | `[1,8,2,2]` | `[8,16,8,512]` | `384 + c*1024` |
| S1 | `[2,4,2,1]` | `[8,16,64,512]` | `384 + c*512` |
| S2 | `[4,2,1,1]` | `[8,32,64,512]` | `384 + floor(c/2)*512 + (c%2)*64` |

Mode0 只启用 D0 writer。accelerator 每次使用完整 `K=256 tiles`，output bound
分别为 S0/S1/S2 的 `4/2/1 N tiles`。

### 7.2 Mode1

Mode1 A base 固定为384，直接读取 Mode0 token-striped D。

| Shape | A spatial stride | A temporal bound | A temporal stride |
|---|---|---|---|
| S0 | `[8,16]` | `[176,4,1,1,1,1]` | `[512,0,0,0,0,0]` |
| S1 | `[8,16]` | `[2,88,2,1,1,1]` | `[64,512,0,0,0,0]` |
| S2 | `[8,32]` | `[2,2,44,1,1,1]` | `[16,64,512,0,0,0]` |

| Shape | B spatial stride | B temporal bound | B temporal stride | mask |
|---|---|---|---|---:|
| S0 | `[8,22528]` | `[4,44,4,1]` | `[16,512,22528,0]` | `0x03` |
| S1 | `[8,22528]` | `[4,44,2,1]` | `[16,512,45056,0]` | `0x0f` |
| S2 | `[8,22528]` | `[4,44,1,1]` | `[16,512,90112,0]` | `0xff` |

| Shape | D0/D1 temporal bound | D0/D1 temporal stride |
|---|---|---|
| S0 | `[1,8,4,1]` | `[512,8,512,0]` |
| S1 | `[2,4,2,1]` | `[512,8,1024,0]` |
| S2 | `[4,2,1,1]` | `[512,8,2048,0]` |

chunk `c` 的 Mode1 writer base：

```text
D0 = 0  + c*2048
D1 = 64 + c*2048
```

accelerator 每次使用完整 `K1=176 tiles`，output bound 为 `4/2/1 N tiles`。

## 8. 正确性数据与日志格式

旧实验的 W/V 都是常量，不能充分发现 panel/chunk 交换。本 app 使用 panel、column、
K-tile 相关的稀疏非均匀 int4 weights，并为三个 shape 分别生成完整 Mode0/Mode1
golden。

每个计算 chunk 完成后立即 compare，并打印：

```text
CHUNK_RESULT ... status=PASS/FAIL errors=... accel=... streamer=... wall=...
```

Mode1 同时分别打印 `D0=PASS/FAIL D1=PASS/FAIL`。每个 mode 和 shape 末尾另有：

```text
MODE_SUMMARY ...
SHAPE_END ...
FINAL_RESULT ...
```

`wall` 从本 chunk CSR programming 开始，到 accelerator 和 streamer 都 idle 为止；
`accel/streamer` 是各自 performance counter。`*_sum` 是所有 chunk 的计数和。

## 9. App、构建状态与运行命令

App：

```text
target/snitch_cluster/sw/apps/snax-versacore-int16x4-multishape-k8-common16-chunked
```

已经完成：

- datagen 生成；
- dual-versacore software library clean rebuild；
- app clean build，`SELECT_SHAPE=-1 CHUNK_COLS=16 RUN_MODE1=1`；
- ELF 成功链接，仅有仓库既有 `static-in-inline` warning。

本次按要求没有启动长 VLT 仿真。运行前必须确认 VLT 也是同一个 cfg 构建。

在 `barnard3` 内：

```bash
cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster
export PATH=$PWD/.pixi/envs/default/bin:/tools/riscv/bin:$PATH
export VERILATOR_ROOT=$PWD/.pixi/envs/default/share/verilator

CFG=cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson
APP=target/snitch_cluster/sw/apps/snax-versacore-int16x4-multishape-k8-common16-chunked

make -C $APP clean
make -C target/snitch_cluster/sw/snax/dual-versacore-swiglu clean
make -C target/snitch_cluster/sw/snax/dual-versacore-swiglu all
make -C $APP all CFG_OVERRIDE=$CFG SELECT_SHAPE=-1 CHUNK_COLS=16 RUN_MODE1=1 FAST_BUILD=1

mkdir -p layout_explore_logs/common16_chunked_20260718
./target/snitch_cluster/bin/snitch_cluster.vlt \
  $APP/build/snax-versacore-int16x4-multishape-k8-common16-chunked.elf \
  2>&1 | tee layout_explore_logs/common16_chunked_20260718/all_shapes_chunk16.log
```

`SELECT_SHAPE=-1` 会顺序跑 S0/S1/S2，并在每个 shape 开始前重新 DMA token。
如果希望分 shape 运行，分别使用 `SELECT_SHAPE=0/1/2`；切换宏时先 clean，因为 Make
不会自动把 CFLAGS 变化当成依赖。

若当前 VLT 不是指定 cfg，先按标准流程重建：

```bash
make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=$CFG
make -C target/snitch_cluster sw CFG_OVERRIDE=$CFG
make -C target/snitch_cluster bin/snitch_cluster.vlt -j$(nproc) CFG_OVERRIDE=$CFG
```

## 10. HeMAiA datagen/runtime 移植要点

1. 固定 canonical S0 4-column full-K panel source order。
2. 每个 overlap chunk 含 `chunk_cols/4` 个连续 panels。
3. shape 只改变 `q=meshCol/4`、channel mask、N bound；B stride 表可以直接使用。
4. 真双缓冲 base 只在固定 ping/pong 之间切换，不携带 SNAX 预搬实验的 slot offset。
5. 每次启动必须是完整 K；沿 N 切分。
6. DMA separated source 和128B interleaved source都可行，但 datagen source packing与
   DMA参数必须配套。
7. Mode0 D=`48..63`、Mode1 D=`0..15`，下一层可继续把 activation bank group
   交替使用。
8. tail 不等于 full chunk 时更新 A/B/D bounds和 accelerator output bound。
