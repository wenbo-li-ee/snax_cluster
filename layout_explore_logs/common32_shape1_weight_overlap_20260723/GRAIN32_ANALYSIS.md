# Shape-1 Mode-0 weight overlap：32-column 粒度

日期：2026-07-23

## 测试配置

- Shape 1：`meshRow=4, meshCol=8, tileSize=8`
- 4 tokens，token 在 pipeline 前搬完
- Mode 0 only
- `CHUNK_COLS=32`，因此 32 iterations
- 每次 accelerator invocation 使用 `n_tiles=4`
- 每次 weight payload 为 64 KiB：
  - L3：`W[64 B], V[64 B]` 交错
  - iDMA：`size=128 B, repeat=512`
  - XDMA：source/destination ND bounds `{2,512}`
- weight ping/pong 和 output layout 保持不变
- cfg：
  - `dma_axi_req_fifo_depth=32`
  - `wide_xbar_latency=CUT_ALL_AX`
  - `wide_xbar_fall_through=true`
  - `register_ext_wide=false`

软件只改了 chunk/streamer 配置，复用了已经与上述 cfg 匹配的
`snitch_cluster.vlt`，没有重新生成 RTL。

## Streamer/accelerator 改动

相对 16-column：

```text
A temporal bound:  [K0_tiles, 2, 1, 1, 1, 1]
                 -> [K0_tiles, 4, 1, 1, 1, 1]

B temporal bound:  [4, K0_tiles/4, 2, 1]
                 -> [4, K0_tiles/4, 4, 1]

D temporal bound:  [2, 4, 2, 1]
                 -> [2, 4, 2, 2]

accelerator N tiles: 2 -> 4
output base step:     512 B -> 1024 B
```

所有 K bounds、strides、channel enables、mode 和 rescale CSR 仍只在
prologue 配置一次。循环中 START 后只写下一 iteration 的三个 moving
streamer bases，以及下一次 DMA/XDMA address/descriptor。

## 结果

所有结果均通过完整 4×1024 output golden check。

| engine | chunk cols | iterations | loop | streamer | compute path | DMA-side | barrier wait | sync/control overhead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iDMA | 16 | 64 | 53,763 | 34,704 | 39,249 | 50,831 | 8,143 | 2,320 |
| iDMA | 32 | 32 | **51,773** | 33,728 | 35,922 | 49,391 | 12,513 | 1,262 |
| iDMA + local XDMA | 16 | 64 | 53,861 | 42,439 | 46,998 | 35,716 | 320 | 6,863 |
| iDMA + local XDMA | 32 | 32 | **47,562** | 41,528 | 44,025 | 33,901 | 231 | 3,537 |

粒度翻倍带来的 loop 改善：

- iDMA：`53,763 -> 51,773`，减少 1,990 cycles，3.70%
- XDMA：`53,861 -> 47,562`，减少 6,299 cycles，11.69%
- 32-column 下 XDMA 比 iDMA 快 4,211 cycles，8.13%

相对于 32-column 无 XDMA contention 的 streamer time 33,728 cycles，
XDMA overlap loop 仍有 13,834 cycles，即 41.02% overhead；但已经明显低于
16-column 的约 55%。

## CSR 配置假设

结果支持“粒度太细”的判断，但不是单纯的 CSR latency：

- iDMA moving-streamer-CSR sum：`2,485 -> 1,245`
- iDMA next-descriptor-prepare sum：`2,621 -> 1,267`
- XDMA moving-streamer-CSR sum：`2,995 -> 1,800`
- XDMA next-prepare sum：`3,964 -> 7,680`

XDMA 的 next-prepare 在 64 KiB transfer 下反而更长，说明 software CSR
custom instructions 会受到当前更长 iDMA staging transaction 的
backpressure；它不是被简单地缩短了。但是每 iteration 的 compute 时间
也从约 734 cycles 增到约 1,376 cycles，足以把 next-prepare 和 DMA
完成都隐藏在当前计算内。最终 XDMA barrier wait 只有 231 cycles。

更大的收益来自：

1. accelerator/streamer START 次数从 64 减到 32；
2. cluster barrier 和 loop branch 次数减半；
3. moving-base CSR 写次数减半；
4. 同样的总 payload 使用一半 descriptor；
5. XDMA DMA-side 已短于 compute path：
   `33,901 < 44,025`，pipeline 从 DMA-bound 变成 compute-bound。

iDMA 在 32-column 下仍是 DMA-bound：
`49,391 > 35,922`，因此粒度翻倍只改善 3.70%。

## 日志

- `run_idma_chunk32_cut_all_ax_fallthrough.log`
- `run_xdma_chunk32_cut_all_ax_fallthrough.log`

## 结论

32-column 是目前明显更好的粒度，特别适合 XDMA staging 路径。它证明
此前 16-column 的主要问题是粒度过细造成的启动、控制、descriptor 和
同步摊销，而不是 XDMA 带宽本身不足。
