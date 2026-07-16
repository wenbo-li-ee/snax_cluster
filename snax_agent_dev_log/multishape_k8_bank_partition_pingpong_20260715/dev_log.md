# K8 多 shape 分 bank / ping-pong layout 实验

日期：2026-07-15

## 目标和边界

本实验沿用 L15 的完整 workload 大小：`K0=2048, N0=1408, K1=1408, N1=1024`。三个 shape 分别使用 8、4、2 个 token：

| Shape | `(meshRow, tileSize, meshCol)` | token 数 | Mode0 `(K tiles, N tiles)` | Mode1 `(K tiles, N tiles)` |
|---|---:|---:|---:|---:|
| S0 | `(8, 8, 4)` | 8 | `(256, 352)` | `(176, 256)` |
| S1 | `(4, 8, 8)` | 4 | `(256, 176)` | `(176, 128)` |
| S2 | `(2, 8, 16)` | 2 | `(256, 88)` | `(176, 64)` |

当前在 `snax_cluster` 层级不能真正把 DMA 和计算重叠，因此程序仍先完成本阶段的 DMA，再启动计算。TCDM 中的数据放置和 B 的地址序列已经按未来的边搬边算 pattern 构造。本轮没有 padding、coloring 或其他 bank-remap trick，所有 accelerator/sparse-interconnect granularity 都为 1。

## Design-time 配置

配置文件：`target/snitch_cluster/cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane_bank_partition_g1.hjson`

- TCDM：64 banks，8 MiB；一个 bank word 为 8 B，一整行跨度为 512 B。
- Accelerator port：A=16，B0=8，B1=8，D0=1，D1=1，共 34 个 TCDM ports。
- Sparse-interconnect 配置：`[16,1], [8,1], [8,1], [1,1], [1,1]`。
- `granularity_a = granularity_b = granularity_c_d = 1`。
- Reader spatial bounds：A `[2,8]`，B0/B1 `[2,4]`。
- Writer spatial bounds：D0/D1 `[1]`。
- Reader temporal dimensions：A/B0/B1 均为 4。
- Writer temporal dimensions：D0/D1 均为 4。

这里将 A temporal dimension 设为 4，是为了让 Mode1 reader 能直接描述 Mode0 单 writer 在 S1/S2 的 16-bank 分散 layout。

## TCDM bank 分区

| 数据 | bank | 首行 byte offset | 说明 |
|---|---:|---:|---|
| A / token | 0..15 | 0 | 每 token 起点相差 16 B；K 方向每次跨 512 B |
| B0 ping | 16..23 | 128 | 当前 meshCol panel |
| B1 ping | 24..31 | 192 | 当前 meshCol panel |
| B0 pong | 32..39 | 256 | 下一 meshCol panel |
| B1 pong | 40..47 | 320 | 下一 meshCol panel |
| Mode0 D0 | 48..63 | 384 | 只启用一个 writer，但三个 shape 都覆盖完整 16 banks |
| Mode1 D0 | 48..55 | 90496 | per-token 输出，D0 writer 启用 |
| Mode1 D1 | 56..63 | 90560 | per-token 输出，D1 writer 启用 |

Mode0 调用 `set_dual_versacore_streamer_csr_d0_only()`，D1 writer 被显式禁用；D0 的 temporal 分解使 S0/S1/S2 的实际输出分别为 22528/11264/5632 个唯一 byte，同时三者的 bank 集合都严格等于 `{48..63}`。Mode1 调用完整 streamer CSR API，D0/D1 两个 writer 都启用。

## DMA pattern

### Token A

所有 8 个输入 token 都搬入 A 分区，不做 padding。每个 token 使用一笔 2D DMA：

```text
size = 16 B
src_stride = 16 B
dst_stride = 512 B
repeat = 256
dst_base(token) = A_BASE + token * 16 B
```

三个 shape 通过 channel mask `0xffff / 0x00ff / 0x000f` 分别消费 8/4/2 个 token。

### Weight B0/B1

L3 数据继续使用 shape0 tile 顺序。每个 shape0 meshCol=4 的完整 reduction panel 单独使用一笔 2D DMA；一个运行时 S0/S1/S2 panel 分别含 `q=1/2/4` 个相邻 shape0 panel。

| 阶段 | panel 有效字节 | DMA `size` | `src_stride` | `dst_stride` | `repeat` | TCDM panel span |
|---|---:|---:|---:|---:|---:|---:|
| Mode0 | 4096 | 64 | 64 | 512 | 64 | 32768 |
| Mode1 | 2816 | 64 | 64 | 512 | 44 | 22528 |

运行时 meshCol index `n` 的偶数列放 ping、奇数列放 pong，buffer 内 slot 为 `floor(n/2)`；B0/B1 分别进入各自 8-bank 子分区。虽然当前先搬完再算，这个地址序列可以直接作为以后 N/N+1 overlap 的 double-buffer pattern。

## Runtime streamer 参数

表中 spatial/temporal stride 的单位都是 byte。

### A reader

| Shape | Mode | spatial stride | temporal bound | temporal stride |
|---|---|---|---|---|
| S0 | 0 | `[8,16]` | `[256,352,1,1]` | `[512,0,0,0]` |
| S1 | 0 | `[8,16]` | `[256,176,1,1]` | `[512,0,0,0]` |
| S2 | 0 | `[8,16]` | `[256,88,1,1]` | `[512,0,0,0]` |
| S0 | 1 | `[8,16]` | `[176,256,1,1]` | `[512,0,0,0]` |
| S1 | 1 | `[8,16]` | `[2,88,128,1]` | `[64,512,0,0]` |
| S2 | 1 | `[8,32]` | `[2,2,44,64]` | `[16,64,512,0]` |

Mode1 A 直接以 Mode0 D0 的 base 为输入，没有 software compact/copy。

### B0/B1 readers

B0 和 B1 使用相同的 bounds/strides，但 base 分别落在对应的 ping bank group。

| Shape | Mode | spatial stride | temporal bound | temporal stride |
|---|---|---|---|---|
| S0 | 0 | `[8,32768]` | `[4,64,2,176]` | `[16,512,128,32768]` |
| S1 | 0 | `[8,32768]` | `[4,64,2,88]` | `[16,512,128,65536]` |
| S2 | 0 | `[8,32768]` | `[4,64,2,44]` | `[16,512,128,131072]` |
| S0 | 1 | `[8,22528]` | `[4,44,2,128]` | `[16,512,128,22528]` |
| S1 | 1 | `[8,22528]` | `[4,44,2,64]` | `[16,512,128,45056]` |
| S2 | 1 | `[8,22528]` | `[4,44,2,32]` | `[16,512,128,90112]` |

B channel mask 为 `0x03 / 0x0f / 0xff`。

### D writers

所有 writer 的 spatial stride 都是 `[8]`，channel enable 为 `1`。

| Shape | Mode | writer | temporal bound | temporal stride |
|---|---|---|---|---|
| S0 | 0 | D0 only | `[1,8,2,176]` | `[8,16,8,512]` |
| S1 | 0 | D0 only | `[2,4,2,88]` | `[8,16,64,512]` |
| S2 | 0 | D0 only | `[4,2,2,44]` | `[8,32,64,512]` |
| S0 | 1 | D0 + D1 | `[1,8,256,1]` | `[512,8,512,0]` |
| S1 | 1 | D0 + D1 | `[2,4,128,1]` | `[512,8,1024,0]` |
| S2 | 1 | D0 + D1 | `[4,2,64,1]` | `[512,8,2048,0]` |

Mode0 的第一个 temporal 维度随着 shape 为 1/2/4，配合 token/meshCol 子维度把唯一的 D0 writer 轮转到整个 16-bank 分区。Mode1 两个 writer 都按 per-token layout 写回各自的 8-bank half。

## VLT 结果

仿真未启用运行时 waveform 选项，未生成 `.vcd/.fst/.fsdb`。

| Shape | Mode0 DMA | Mode0 accel | Mode0 streamer | Mode0 wall | Mode1 DMA | Mode1 accel | Mode1 streamer | Mode1 wall | 正确性 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S0 | 122379 | 90118 | 90142 | 92412 | 60013 | 45317 | 45335 | 47772 | M0 PASS; D0/D1 PASS |
| S1 | 123443 | 45062 | 45086 | 47339 | 60732 | 22534 | 22552 | 24981 | M0 PASS; D0/D1 PASS |
| S2 | 123301 | 22536 | 22560 | 24827 | 60646 | 11272 | 11290 | 13677 | M0 PASS; D0/D1 PASS |

三个 shape 都报告 `total error: 0`。DMA 周期是当前“先全搬完、再计算”的 staging 成本，没有和计算 overlap；accel/streamer 周期才是本轮 layout conflict 探索的主要观察值。

相对原 L15 的计算周期：

| Shape | Mode0 accel / streamer delta | Mode1 accel / streamer delta |
|---|---:|---:|
| S0 | `+1 / +1` | `-1277 / -1278` |
| S1 | `-24 / -24` | `-1040 / -1054` |
| S2 | `-13 / -13` | `-38 / -52` |

## 构建和复现

所有命令均在 `barnard3` container 内、仓库 `snax_cluster/` 根目录执行，且统一使用：

```text
CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane_bank_partition_g1.hjson
```

完成过以下验证：RTL generation、dual-versacore software library clean rebuild、VLT rebuild、顶层 `sw` 构建，以及 `SELECT_SHAPE=0/1/2` 三个 ELF 的直接 VLT 仿真。逐 shape 日志位于 `layout_explore_logs/bank_partition_pingpong_20260715/`。
