# Shape-1 common16 weight overlap：XDMA 尝试

日期：2026-07-23

## 1. 测试范围

- 只测 Shape 1 / Mode 0：`meshRow=4, meshCol=8, tileSize=8`。
- 4 个 token 在进入 pipeline 前一次性搬入 TCDM。
- 64 个 common-16 weight chunks。
- L3 中每行仍采用 `W[64 B], V[64 B]` 交错布局。
- weight destination 仍采用 16-bank ping/pong：
  - ping：banks 16..31；
  - pong：banks 32..47。
- 统计区间只取 64-iteration for loop，不包含 token DMA 和 chunk 0
  prologue。
- accelerator/streamer base CSR 和 DMA/XDMA address CSR 都采用
  `START` 后立即配置下一 iteration 的方式。

App：

`target/snitch_cluster/sw/apps/snax-versacore-int16x4-shape1-common16-weight-overlap`

编译开关：

- `USE_XDMA=0`：原始 iDMA 2D descriptor；
- `USE_XDMA=1`：iDMA contiguous staging + local XDMA reshape；
- `USE_XDMA=2`：实验性的 iDMA-to-XDMA data-MMIO 路径。

## 2. 为什么不能直接用 XDMA 从 L3 搬到本 cluster

XDMA frontend 不是 iDMA 的直接替代品。`XDMACtrl.scala` 按 source
address 分类：

- source 位于 local TCDM：local XDMA read；
- source 为 0：discard reader，DRAM 一侧应由 iDMA 注入；
- 其他 source：remote XDMA read。

因此把 `WV_interleaved` 的 L3 地址直接写入 XDMA source CSR，会被分类为
remote XDMA request。当前 `snitch_cluster.vlt` 单 cluster testbench 没有
HeMAiA memory-side XDMA endpoint，remote submitted counter 会增加，但
remote finish counter 不会返回，仿真不能完成。

## 3. 可工作的方案：iDMA staging + local XDMA

每 chunk 的 32 KiB interleaved W/V payload 先由一条 contiguous iDMA
搬到 TCDM staging ping/pong，随后 local XDMA 用下面的 ND walk 写入正式
weight ping/pong：

```text
source bounds/strides = {2, 256} / {64, 128}
dest   bounds/strides = {2, 256} / {64, 512}
spatial transfer      = 8 B × 8 lanes = 64 B
```

steady state 中：

1. compute core 启动 chunk `c`，随后配置 chunk `c+1` 的 streamer bases；
2. DM core 启动 chunk `c+1` 的 XDMA reshape；
3. 启动已准备的 iDMA descriptor，将 chunk `c+2` 搬入另一 staging
   buffer；
4. START/DMCPYI 后立即配置下一次 XDMA address 和 iDMA descriptor；
5. iteration 末尾用一次 cluster hardware barrier 保护 ping/pong reuse。

功能检查为 PASS。

## 4. 同配置结果

### 4.1 保留 AXI cuts：`CUT_ALL_AX + FallThrough=true`

| 路径 | loop | streamer sum | accelerator sum | DMA-side sum | barrier wait |
|---|---:|---:|---:|---:|---:|
| iDMA，W/V 单 descriptor | **53,763** | 34,704 | 33,023 | 50,831 | 8,143 |
| iDMA staging + local XDMA | 53,861 | 42,439 | 40,903 | 35,716 | 320 |

XDMA 将 DMA-side 时间减少 15,115 cycles（29.7%），但 streamer 增加
7,735 cycles（22.3%），compute path 增加 7,749 cycles（19.7%）。原因是
staging iDMA 与 XDMA reshape 都访问 TCDM，而
compute 同时占用 token、weight 和 output banks；新增的 TCDM bank
arbitration使计算路径明显变慢。最终 loop 反而比 iDMA 多 98
cycles，基本持平但没有收益。

日志：

- `run_xdma_local_staging_cut_all_ax_fallthrough_final.log`
- `run_idma_cut_all_ax_fallthrough_final.log`

### 4.2 性能上界：`NO_LATENCY + FallThrough=true`

为了验证 iDMA-to-XDMA MMIO handshake，临时去掉 wide-xbar cuts，并对两条
可工作路径做了同 RTL 对照：

| 路径 | loop | compute path | streamer sum | accelerator sum | DMA-side sum | barrier wait |
|---|---:|---:|---:|---:|---:|---:|
| iDMA | 53,918 | 39,083 | 34,688 | 33,152 | 50,922 | 8,474 |
| iDMA staging + local XDMA | **53,657** | 46,727 | 42,454 | 40,918 | 36,211 | 320 |

同配置下 XDMA 快 261 cycles，即 0.48%。它把 DMA-side sum 减少 14,711
cycles（28.9%），但 TCDM contention 又把 compute path 增加 7,644
cycles（19.6%）。所以大量 DMA 改善最终只剩 0.5% loop 改善。

相对于原来纯计算的 streamer sum 34,688 cycles：

- iDMA loop overhead：19,230 cycles，55.44%；
- XDMA loop overhead：18,969 cycles，54.68%。

branch/barrier micro-calibration 为：

- branch+nop loop：201 cycles / 64 iterations；
- 64 次 hardware barrier loop：263 cycles；
- barrier 相对 branch loop 的增量：62 cycles。

因此 branch + barrier 指令本身约占 263 / 53,657 = 0.49%，不是主要
overhead。XDMA run 中从 `compute_path_sum=46,727` 到 `loop=53,657`
仍有 6,930 cycles 的多核控制/同步间隙；主要问题仍是每 iteration 的
两条执行路径对齐以及 TCDM contention，而不是 CSR 配置。CSR 写已经放在
START 后，隐藏在当前 iteration 内。

日志：

- `run_xdma_local_staging_nolatency_fallthrough.log`
- `run_idma_nolatency_fallthrough.log`

## 5. iDMA 直接注入 XDMA data window

还实现并尝试了更合理的 hybrid：

```text
L3 --iDMA, 4096 B × 8--> XDMA data MMIO
   --XDMA writer {2,256}/{64,512}--> weight ping/pong
```

这样不需要 32 KiB staging buffer，理论上可消除 staging TCDM read
contention。top-level 生成 wrapper 把 `ClusterAddrSpace=16384 KiB` 传给
cluster，并把 `ClusterAddressSpace=0x1000000` 传给 XDMA wrapper，因此
data window 是 `cluster_base + 0xFFC000`。这里不能采用 XDMA wrapper
module declaration 中未覆盖的 1 MiB parameter default。

使用正确的 16 MiB address space，在
`NO_LATENCY + FallThrough=true` 下仍卡在 `snrt_dma_wait_all()`。DM core
trace 停在 DMA status custom instruction loop。

RTL 原因是 `xdma_axi_to_write.sv` 只有在 `aw_valid && w_valid` 同周期为
真时才接受新的 write burst；而当前 iDMA/xbar 路径在 AW 被接受前不会把
首个 W 送到该 slave，形成协议层互等。仅打开 wide-xbar FallThrough 或
去掉 xbar cuts 均不能修复。

日志：

- `run_idma_to_xdma_mmio_nolatency_fallthrough.log`

要使此路径工作，需要把 XDMA AXI write adapter 改成独立接收并缓存 AW，
再等待 W，或在 adapter 前加入能独立缓冲 AW/W 的合规桥。它是下一步最有
价值的 XDMA 方向，因为它有机会保留 28.9% 的 DMA-side 改善，同时避免
local staging 引入的 TCDM read contention。

## 6. 结论

XDMA 本身确实更快地完成了 layout reshape，但在当前单-cluster RTL 上：

- 直接 L3→XDMA 不具备 remote endpoint；
- 可工作的 local staging 版本被额外 TCDM contention 抵消；
- standard `CUT_ALL_AX` 配置下 loop 慢 98 cycles（0.18%）；
- `NO_LATENCY` 性能上界只改善 0.48%。

因此当前不应把 staging XDMA 当成最终优化。真正可能有明显收益的是修正
XDMA data-window AXI adapter 后的 iDMA→XDMA streaming 路径。
