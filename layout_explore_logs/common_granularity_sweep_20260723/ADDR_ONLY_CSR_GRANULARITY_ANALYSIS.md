# Shape-1 Mode-0：地址-only CSR 与粒度扫描

日期：2026-07-23

## 范围与配置

- Shape 1：`meshRow=4, meshCol=8, tileSize=8`
- 4 tokens；token 在 weight pipeline 开始前全部搬入 TCDM
- Mode 0 only，W/V 两路并行
- W/V 在 L3 中以 `W[64 B], V[64 B]` 交错
- weight TCDM 使用固定 ping/pong 布局
- cfg：
  - `dma_axi_req_fifo_depth=32`
  - `wide_xbar_latency=CUT_ALL_AX`
  - `wide_xbar_fall_through=true`
  - `register_ext_wide=false`

这轮只修改软件，复用与上述 cfg 匹配的 `snitch_cluster.vlt`。

## CSR 优化

完整 CSR 只在 prologue 配置一次。循环中每次先启动当前任务，再写下一
iteration 的动态地址。

### Accelerator

循环中只写 `START`，不再写配置 CSR。K、N、array shape、mode、data
type、subtraction 和三个 rescale 配置对所有等长 chunk 都不变。

### Streamer

循环中只写三个低位 base：

1. B0 当前 W chunk；
2. B1 当前 V chunk；
3. D0 当前 output chunk。

A base、所有 spatial/temporal bounds、strides、channel enables 和地址
remap 都只配置一次。这里不能只保留一个 base write，因为 B0、B1 和 D0
是三个独立 streamer endpoint，且三者地址都会随 chunk 改变。

### iDMA

完整 descriptor 在 prologue 配置一次：

- source/destination；
- source/destination stride；
- repeat。

后续 chunk 的 stride 和 repeat 不变，所以循环内只写 DMSRC 和 DMDST；
当前 descriptor 用 DMCPYI 启动后，立即覆盖 staging source/destination
供下一次使用。

### local XDMA

完整 XDMA descriptor 只配置一次，包括：

- source/destination 高低地址；
- ND bounds 和 strides；
- channel/byte masks。

staging buffer 与 weight buffer 都在同一 cluster 地址区，高地址部分不变。
循环内只写 source low 和 destination low 两个 CSR。source 和 destination
分别在 staging ping/pong 与 weight ping/pong 之间切换，因此仍需要两个
低位 base write。

## 地址-only CSR 的直接效果（32 columns）

| engine | CSR 方案 | loop | next prepare | DMA wait | DMA-side |
|---|---|---:|---:|---:|---:|
| iDMA | 完整动态 descriptor | 51,773 | 1,267 | 约 47.5k | 49,391 |
| iDMA | 地址-only | **51,717** | **1,082** | 47,671 | 49,326 |
| iDMA + local XDMA | 完整动态地址（含 high） | 47,562 | 7,680 | 19,725 | 33,901 |
| iDMA + local XDMA | low-address-only | **47,170** | **2,238** | 30,596 | 33,939 |

结论：

- iDMA loop 只减少 56 cycles（0.11%）。
- XDMA loop 减少 392 cycles（0.82%）。
- XDMA `next prepare` 明显减少 5,442 cycles，但 DMA-side 总时间基本不变；
  省下的配置等待转移成了真实传输的 `DMA wait`。
- 因而 CSR 优化有效，但原来的主要瓶颈不是 CSR 指令数量，而是 DMA
  完成时间和并发访问 TCDM 的争用。

## 粒度扫描

下表只列 golden check 通过的结果。`loop` 是用户指定的、严格从 for-loop
入口到出口的时间；`total` 是 `prologue + loop`，用于识别首块 DMA 被移出
计时区间的影响。

| engine | chunk cols | iterations | prologue | loop | total | streamer | DMA-side | barrier wait | sync/control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| iDMA | 32 | 32 | 2,601 | 51,717 | 54,318 | 33,728 | 49,326 | 12,298 | 1,269 |
| iDMA | 64 | 16 | 3,424 | 50,260 | **53,684** | 33,248 | 47,447 | 14,161 | 664 |
| iDMA | 128 | 8 | 6,571 | 48,685 | 55,256 | 33,008 | 44,119 | 14,198 | 372 |
| iDMA | 256 | 4 | 12,831 | **46,283** | 59,114 | 32,888 | 37,753 | 12,565 | 239 |
| iDMA + local XDMA | 32 | 32 | 5,360 | 47,170 | 52,530 | 41,507 | 33,939 | 231 | 3,444 |
| iDMA + local XDMA | 64 | 16 | 8,443 | 43,664 | **52,107** | 40,763 | 32,099 | 119 | 1,762 |
| iDMA + local XDMA | 128 | 8 | 14,674 | **41,261** | 55,935 | 39,797 | 29,439 | 40 | 882 |

### 只看 for-loop

- XDMA `32 -> 64`：`47,170 -> 43,664`，减少 3,506 cycles（7.43%）。
- XDMA `64 -> 128`：`43,664 -> 41,261`，减少 2,403 cycles（5.50%）。
- XDMA `32 -> 128`：总共减少 5,909 cycles（12.53%）。
- 当前最短且正确的 XDMA loop 是 128 columns：**41,261 cycles**。
- iDMA loop 随粒度继续下降，但一直是 DMA-bound，改善较慢。

### 与原来纯计算比较

resident-weight 实验
`common16_resident_k1024_20260718/all_shapes.log` 中 Shape 1 Mode 0 的
纯 streamer sum 是 **34,688 cycles**（accelerator sum 33,152）。

因此当前最好的正确 XDMA loop：

```text
41,261 - 34,688 = 6,573 cycles
overhead = 18.95%
```

128-column XDMA run 内部：

- streamer：39,797
- compute path：40,379
- whole loop：41,261
- loop 相对本次 compute path 的同步/控制差值：882 cycles（2.18%）

因此剩余 6,573-cycle 差距中，只有较小部分是 branch/barrier/控制；主要部分
是 concurrent local XDMA 访问 TCDM 后，streamer/accelerator 本身从纯计算
的 34,688 增长到约 39.8k。

### 若把首块预取也算入总时间

粒度越大，第一块 DMA 越大，而它位于 for-loop 计时区间之外：

- XDMA 32：52,530
- XDMA 64：**52,107**
- XDMA 128：55,935

所以：

- 按用户指定的 steady-state for-loop 指标，128 columns 最好；
- 按单次完整调用 `prologue + loop`，64 columns 最好；
- 128 columns 的 loop 优势不能解释成端到端同样改善。

## 256-column XDMA 诊断

二维 XDMA `{2,2048}` 在 256-column 下失败：

- 1,990 mismatches；
- 首个 mismatch 位于 token 0、全局 column 128；
- mismatch 总数接近全部输出的一半，符合每个 256-column reshape 只正确
  产生前半部分的现象，但当前 checker 只打印前四个错误，没有逐 chunk
  记录错误分布；
- iDMA 256-column 使用相同 accelerator/streamer 配置可完整通过。

因此错误不在 VersaCore、streamer 或 32-bit accelerator CSR。

又测试了地址序列等价的三维 XDMA `{2,1024,2}`，错误数量和首个错误位置
完全不变，排除了单个 temporal bound 为 2048 的问题。错误边界对应 XDMA
destination offset 到达 512 KiB。目前可确认它属于 local XDMA/TCDM 地址
路径，具体地址截断点仍需 RTL trace 才能进一步定位。该失败点不纳入性能
结论，当前最大已验证 XDMA 粒度为 128 columns。

## 日志

- `../common32_shape1_weight_overlap_20260723/run_idma_chunk32_addr_only_csr.log`
- `../common32_shape1_weight_overlap_20260723/run_xdma_chunk32_addr_only_csr.log`
- `run_idma_chunk64_addr_only_csr.log`
- `run_idma_chunk128_addr_only_csr.log`
- `run_idma_chunk256_addr_only_csr.log`
- `run_xdma_chunk64_addr_only_csr.log`
- `run_xdma_chunk128_addr_only_csr.log`
- `run_xdma_chunk256_addr_only_csr.log`
- `run_xdma_chunk256_addr_only_csr_split3d.log`
