# Shape 1 weight-overlap workload：iDMA cfg 分析与试验

日期：2026-07-23

## 1. 结论摘要

本轮对当前实际使用的 cfg 做了干净的逐项对照：

`target/snitch_cluster/cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson`

最终保留的试验配置为：

```hjson
dma_data_width: 512
dma_axi_req_fifo_depth: 32
dma_req_fifo_depth: 8
narrow_trans: 4
wide_trans: 32

timing: {
    wide_xbar_latency: CUT_ALL_AX
    wide_xbar_fall_through: true
    register_ext_wide: false
}
```

主要结论如下。

1. `dma_axi_req_fifo_depth` 从 16 增加到 32 后，所有关键 cycle 数逐项完全相同。因此这个 workload 在 depth 16 时就没有被 AXI request FIFO 容量限制。
2. `wide_xbar_latency` 从 `CUT_ALL_PORTS` 改成 `CUT_ALL_AX` 只减少 40 个 loop cycles（0.074%），而 `dma_wait_sum` 完全不变。wide xbar 的 W/R/B channel cuts 不是主要瓶颈。
3. 进一步把 `register_ext_wide` 从 `true` 改成 `false`，loop 减少 320 cycles（0.590%），`dma_wait_sum` 减少 285 cycles（0.602%）。外部 wide AXI cut 确实增加了少量固定延迟，但影响很小。
4. 在最好配置上打开 DMA xbar `FallThrough` 后，所有 cycle 字段逐项完全相同。它没有改善当前 workload。
5. 最好结果仍为 `loop=53925`，相对纯 streamer 计算时间 `34688` 仍多 19237 cycles，即 55.46%。所以 cfg 中 FIFO/crossbar pipeline 并不能解释或消除主要 overhead。
6. 当前每个 weight chunk 搬 32768 B。512-bit DMA 的带宽下限是 512 cycles，但最好实测平均为 807.37 cycles/chunk，只有 40.59 B/cycle，即理论 64 B/cycle 的 63.42%。根因更接近“256 个 128 B、每个只有 2 beat 的短 1D burst”所产生的事务/后端开销，而不是 CSR 配置开销或 FIFO 太浅。

`register_ext_wide=false` 和较少的 xbar cuts 会加长组合路径。本轮只验证了 RTL 仿真的功能与 cycle 数，没有做 synthesis timing closure。因此最好仿真配置不能直接等价为最佳物理实现。

## 2. 测试条件

测试 app：

`target/snitch_cluster/sw/apps/snax-versacore-int16x4-shape1-common16-weight-overlap`

固定条件：

- 只测 Shape 1 / Mode 0，`meshRow=4`。
- 只跑 4 个 token；token 在进入 weight-overlap loop 前全部搬好。
- K=1024，按 common dimension 16 切成 64 个 chunk。
- W/V 在 L3 中按每行 `W 64 B + V 64 B` 交错。
- 每个 chunk 只发一个 2D DMA descriptor：
  - `size=128 B`
  - `repeat=256`
  - `src_stride=128 B`
  - `dst_stride=512 B`
- 每个 chunk 的 payload 为 `128 × 256 = 32768 B`。
- accelerator、streamer 和下一次 DMA CSR 均按之前实现进行提前配置。
- 第一个 DMA 在 prologue 中完成；loop 内统计后续 63 个 DMA。
- 每个配置均执行 `rtl-gen`、匹配 cfg 的 VLT rebuild，并直接运行同一个 focused ELF。
- 所有性能 run 均得到 `status=PASS errors=0`。

纯计算参考：

- 双 VersaCore accelerator busy 总和：`33152 cycles`
- streamer busy 总和：`34688 cycles`
- loop 对比采用更完整的纯 streamer 时间 `34688 cycles`

## 3. cfg 参数到 RTL 的实际映射

iDMA 路径可概括为：

```text
DMA CSRs
  -> top-level ND request FIFO
  -> idma_nd_midend（把 ND descriptor 展开成多个 1D request）
  -> idma_backend_rw_axi
  -> cluster wide AXI xbar
  -> external L3 / local TCDM
```

关键源码映射：

- `hw/snitch_cluster/src/snitch_cluster_wrapper.sv.tpl`
  - `dma_data_width -> WideDataWidth`
  - `dma_axi_req_fifo_depth -> DMAAxiReqFifoDepth`
  - `dma_req_fifo_depth -> DMAReqFifoDepth`
  - `wide_trans -> WideMaxMstTrans/WideMaxSlvTrans`
  - `timing.wide_xbar_latency -> WideXbarLatency`
  - `timing.register_ext_wide -> RegisterExtWide`
- `hw/snitch_cluster/src/snitch_cc.sv`
  - `DMAAxiReqFifoDepth -> idma_inst64_top.NumAxInFlight`
  - `DMAReqFifoDepth -> idma_inst64_top.DMAReqFifoDepth`
- `.bender/git/checkouts/idma-26158a88a10f327f/src/frontend/inst64/idma_inst64_top.sv`
  - `DMAReqFifoDepth` 实例化在 ND request 的 `stream_fifo_optimal_wrap`
  - backend 的 `BufferDepth=3` 和 `MemSysDepth=16` 在这里固定，并不是当前 HJSON 选项
- `.bender/git/checkouts/idma-26158a88a10f327f/src/midend/idma_nd_midend.sv`
  - repeat counter 更新 source/destination address
  - 每次 repeat 向 backend 发出一个 1D `burst_req`
- `.bender/git/checkouts/idma-26158a88a10f327f/target/rtl/idma_backend_rw_axi.sv`
  - `NumAxInFlight` 设置 read/write datapath request FIFO 和 AXI coupling 相关深度
- `hw/snitch_cluster/src/snitch_cluster.sv`
  - `wide_trans` 设置 DMA xbar 的 `MaxMstTrans` 和 `MaxSlvTrans`
  - `register_ext_wide` 控制 cluster wide input/output 上两个 `axi_cut` 是否 bypass

## 4. 各 cfg 选项与本 workload 的相关性

| cfg 选项 | 当前/最终值 | 作用 | 与当前问题的关系 |
|---|---:|---|---|
| `dma_data_width` | 512 | DMA AXI beat 和 TCDM wide port 宽度；峰值 64 B/cycle | 直接决定理论带宽。当前已刚好等于计算每周期所需 512 bit。改成 1024 是硬件带宽/面积变更，不是 FIFO 调优 |
| `dma_axi_req_fifo_depth` | 32 | 映射到 iDMA `NumAxInFlight`，控制 AXI/backend 可容纳的并行事务 | 最直接的候选；16→32 实测完全无效，说明实际需要的并发不超过 16，或更早已有 backpressure |
| `dma_req_fifo_depth` | 8 | 顶层 ND descriptor FIFO 深度 | 它只能让多个软件提交的 ND descriptor 排队，不能让单个 ND midend 并行展开它们，也不能合并相邻 burst。当前合并版 loop 每次只 start 一个 descriptor，下一次只预配 CSR、不提前 start，因此 depth 8 已远大于需求 |
| `wide_trans` | 32 | wide AXI xbar 每个 master/slave port 的 outstanding 上限 | 可能成为 `NumAxInFlight` 的下游上限。当前 32 与 FIFO 32 匹配；FIFO 16→32 已无效，所以增加到 64 没有依据 |
| `wide_xbar_latency` | `CUT_ALL_AX` | wide xbar 内各 AXI channel 的 register slices | 有少量控制侧影响；去掉 W/R/B cuts 后 DMA wait 不变 |
| `wide_xbar_fall_through` | `true` | 允许 AW routing decision 直接传到 W channel，使首个 W beat 可与对应 AW 同周期被接受 | 单独从 false→true 后所有 cycle 字段完全不变；当前短 burst 路径没有被 AW/W 首拍耦合限制 |
| `register_ext_wide` | `false` | cluster 外部 wide AXI input/output 的额外完整 `axi_cut` | 有少量固定延迟影响；bypass 后每个 loop DMA 平均约减少 4.52 wait cycles |
| `narrow_trans` | 4 | narrow AXI 网络 outstanding 数 | iDMA payload 走 wide AXI，不是本数据路径 |
| `dma_user_width` | 1 | AXI user sideband 宽度 | 只改变 metadata 宽度，不增加数据吞吐 |
| `dma_id_width_in` | 默认值 | 外部 wide AXI ID 宽度 | 增加 ID 位数不会自动让单 iDMA channel 发出更多事务；不是当前限制 |
| `timing.register_tcdm_cuts` | 未启用/默认 false | 增加 TCDM memory response pipeline latency | 当前已经没有额外 cut；打开只会增加 latency |

### `dma_axi_req_fifo_depth` 与 `wide_trans` 的共同上限

即使把 iDMA `NumAxInFlight` 提高到 64，当前 wide xbar 的 `MaxMstTrans/MaxSlvTrans` 仍是 32，所以下游最多只接受 32 个 outstanding transaction。用户之前试过的

`snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane_bgran2_highgran_search_2.hjson`

使用的是：

```hjson
dma_axi_req_fifo_depth: 64
dma_req_fifo_depth: 8
wide_trans: 32
```

因此 depth 32 以上没有收益是合理的。不过该旧 cfg 同时修改了 sparse interconnect/granularity，不能单独作为 FIFO 的严格 A/B。本轮在同一个 cfg 上完成的 16→32 测试是干净对照，而且结果逐项完全相同。

### 为什么 W/V 两条 descriptor 合成一条仍然明显有效

`dma_req_fifo_depth=8` 只代表前端能够缓存八条完整的 ND descriptor。旧版 W/V 两条 descriptor 可以连续执行 `DMCPYI` 并进入这个 FIFO，但 `idma_inst64_top` 当前只有一个 DMA channel 和一个 ND midend。FIFO head 只有在当前 descriptor 到达 `last` 后才会被 midend 接受并切换到下一条。因此第二条 descriptor 只是排队，不会与第一条 descriptor 并行展开，也不会由 FIFO 自动合并 AXI transaction。

更重要的是，L3 W/V 交错修改同时改变了 backend 看到的 1D request 粒度：

```text
旧布局：
  W descriptor: size=64 B,  repeat=256  -> 256 个 1-beat request
  V descriptor: size=64 B,  repeat=256  -> 256 个 1-beat request
  合计：512 个 1-beat request，2 个 ND descriptor

新交错布局：
  WV descriptor: size=128 B, repeat=256 -> 256 个 2-beat request
  合计：256 个 2-beat request，1 个 ND descriptor
```

两者 payload 都是 32768 B、总数据 beat 数也都是 512，但新布局把 1D/AXI transaction 数量减半。每个 transaction 的地址请求、合法化、metadata、response/completion 等固定成本因此只支付一半；同时还少了一次 ND descriptor 边界和一次软件 start。旧版两个 descriptor 即使都已缓存在 depth=8 FIFO 中，这些处理成本仍然存在。

所以：

- `dma_req_fifo_depth` 回答的是“软件能提前排队多少条 descriptor”；
- W/V 合并回答的是“完成相同 payload 要处理多少条 descriptor、多少个短 1D/AXI transaction”。

两者并不矛盾。若只是把两条 descriptor 包装成一个更高维 descriptor，但 backend 最终仍生成 512 个 64 B request，预期收益只会是少一个 descriptor 边界；当前从约 1316 cycles/chunk 降到约 812 cycles/chunk 的主要收益来自 64 B burst 合并为 128 B burst。

## 5. 仿真结果

| 试验 | AXI FIFO | wide xbar | ext-wide cut | loop | 相对原始 loop | dma loop sum | dma wait sum | streamer sum | loop / streamer |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| E0 原始基线 | 16 | `CUT_ALL_PORTS` | on | 54245 | — | 51174 | 47358 | 34688 | 1.5639× |
| E1 FIFO | 32 | `CUT_ALL_PORTS` | on | 54245 | 0 (0.000%) | 51174 | 47358 | 34688 | 1.5639× |
| E2 xbar cuts | 32 | `CUT_ALL_AX` | on | 54205 | -40 (-0.074%) | 51138 | 47358 | 34688 | 1.5629× |
| E3 ext cut bypass | 32 | `CUT_ALL_AX` | off | 53925 | -320 (-0.590%) | 50864 | 47073 | 34688 | 1.5546× |
| E4 FallThrough | 32 | `CUT_ALL_AX`, FallThrough=1 | off | 53925 | -320 (-0.590%) | 50864 | 47073 | 34688 | 1.5546× |

对应日志：

- E0：`../common16_shape1_weight_overlap_20260723/run_wv_interleaved.log`
- E1：`run_axi_req_fifo32.log`
- E2：`run_fifo32_cut_all_ax_extwide_on.log`
- E3：`run_fifo32_cut_all_ax_extwide_off.log`
- E3 最终 ELF/VLT 组合复验：`run_final_fifo32_cut_all_ax_extwide_off.log`
- E4：`run_fifo32_cut_all_ax_extwide_off_fallthrough_on.log`

### E1：FIFO 16 → 32

E0 和 E1 不只是 loop 相同，`prologue`、`first_dma`、`dma_loop_sum`、`dma_wait_sum`、`barrier_wait_sum` 等字段也逐项完全相同。这排除了仿真噪声下的“小改善”，说明 depth 16 已足够。

### E2：`CUT_ALL_PORTS` → `CUT_ALL_AX`

`dma_wait_sum` 保持 47358 不变，所以移除 W/R/B channel cuts 没有加快 weight DMA 的完成。减少的 40 cycles 来自 `dma_next_prepare_sum`、streamer preconfiguration 等控制路径的细小变化。

### E3：bypass external-wide `axi_cut`

相对 E0：

- loop：`54245 - 53925 = 320 cycles`，改善 0.590%
- DMA loop sum：`51174 - 50864 = 310 cycles`，改善 0.606%
- DMA wait sum：`47358 - 47073 = 285 cycles`，改善 0.602%
- 平均每个 loop DMA wait 减少：`285 / 63 = 4.52 cycles`

这是本轮唯一直接降低 `dma_wait_sum` 的 cfg 修改，但数量级不足以改变整体结论。

### E4：DMA xbar `FallThrough=1`

为了只测 FallThrough，本轮保持 E3 的其他配置不变：

```hjson
dma_axi_req_fifo_depth: 32
wide_trans: 32
wide_xbar_latency: CUT_ALL_AX
register_ext_wide: false
wide_xbar_fall_through: true
```

生成 wrapper 已确认传入 `.WideXbarFallThrough(1)`，narrow xbar 仍保持 `FallThrough=0`。仿真结果与 E3 逐字段完全相同：

- `loop=53925`
- `dma_loop_sum=50864`
- `dma_wait_sum=47073`
- `barrier_wait_sum=8239`
- `status=PASS errors=0`

AXI xbar 中的 `FallThrough` 并不是普通 request FIFO 深度或所有 channel 的空 FIFO bypass；它专门允许 AW channel 的 routing decision 直接影响 W channel，使 crossbar 能在接受 AW 的同一周期接受对应首个 W beat。当前数据说明我们的 DMA copy 没有被这一拍限制。

axi xbar 文档推荐 `CUT_ALL_AX` 时保持 `FallThrough=0`，避免 AW 组合逻辑延伸到 W channel；若要完全组合，则使用 `NO_LATENCY + FallThrough=1`。因此当前打开 FallThrough 没有 cycle 收益，却仍可能增加综合组合路径。队友若观察到明显改善，需要对齐其 `LatencyMode`、external cuts、traffic direction 和 burst pattern 后再比较。

## 6. 为什么理论 512 bit/cycle 仍然不够

从 payload 守恒看，计算和 DMA 的标称速率都为 512 bit/cycle：

- 每个 chunk：32768 B
- 512-bit DMA 理想时间：`32768 / 64 = 512 cycles`
- 原始实测平均 DMA loop 时间：`51174 / 63 = 812.29 cycles`
- 最好实测平均 DMA loop 时间：`50864 / 63 = 807.37 cycles`
- 最好有效带宽：`32768 / 807.37 = 40.59 B/cycle = 324.69 bit/cycle`
- 最好带宽利用率：`40.59 / 64 = 63.42%`

所以“计算消耗 512 bit/cycle，DMA 标称也是 512 bit/cycle”只说明峰值相等。要完全 overlap，还要求 DMA 在整个 chunk 上接近 100% 利用率。当前 descriptor 被 ND midend 展开成 256 个 128 B 的 1D request，每个 request 只有两个 512-bit beat。每个短 request 都有地址、request handoff、AXI transaction 和后端 bookkeeping 的固定成本；outstanding FIFO 能隐藏一部分延迟，但不能把这些短 burst 变成长连续流。

这也和之前的连续 32 KiB 1D DMA benchmark 一致：连续传输约 544 cycles，可达到约 94.1% 峰值；相同 payload 在当前 strided pattern 下需要约 807 cycles。因此主要损失来自短 2-beat sub-transfer pattern，而不是 512-bit datapath 本身。

## 7. 未继续 sweep 的选项及原因

- `dma_axi_req_fifo_depth=64`：16→32 已严格证明无收益；旧试验也表明更大无收益。
- `wide_trans=64`：当前 iDMA depth 为 32，而且 16→32 没有收益；单独增加 xbar 上限不会产生更多有效请求。
- `dma_req_fifo_depth=16`：当前没有多个已 start 的 ND descriptor 排队，增加顶层 descriptor FIFO 不会改变一个 descriptor 内的 256 repeats。
- `NO_LATENCY`：会进一步拉长组合路径，AXI xbar 文档也对互连 crossbar 的组合环路有约束。`CUT_ALL_AX` 已经移除了数据/响应 channel cuts，却没有改善 DMA wait，因此不值得用更高 timing 风险继续追几十 cycles。
- `dma_data_width=1024`：可能通过硬件带宽过配补偿低利用率，但会把 DMA wide port、TCDM super-bank 组织、crossbar、testbench interface 和物理面积都改成另一套架构；它不能回答 512-bit DMA 为什么只有约 63% 利用率。
- 修改 iDMA `BufferDepth=3` 或 `MemSysDepth=16`：这两个值在 `idma_inst64_top.sv` 中固定，不是 cfg 选项。若要继续硬件探索，应新增显式 HJSON 参数后分别 sweep，而不是把它们和现有 cfg 字段混在一起。

## 8. 建议的下一步

优先级高于继续加 FIFO：

1. 设法增大 ND midend 生成的单个 1D burst 长度，或减少每 chunk 的 256 个 repeats。目标是让同样 32 KiB payload 更接近已测得的连续 1D DMA（约 544 cycles）。
2. 在不改变 accelerator bank mapping 的前提下，研究是否能让 destination 中相邻多行连续，或者让 streamer 接受更连续的 weight TCDM layout。
3. 若布局无法改变，再对 iDMA backend 的固定 `BufferDepth` 做独立参数化与 sweep，并通过 AXI trace 统计 AR/AW 接收间隔、R/W beat 利用率以及 backpressure 来源。
4. `register_ext_wide=false` 可用于性能上界仿真，但进入综合前必须比较 timing slack；若 timing 不过，应恢复 external cut，因为它只值约 0.6%。

## 9. 构建说明

所有命令在 `barnard3` 容器中执行，且始终使用同一个 `CFG_OVERRIDE`：

```bash
CFG=cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson

make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=$CFG
make -C target/snitch_cluster/sw/snax/dual-versacore-swiglu all CFG_OVERRIDE=$CFG
make -C target/snitch_cluster/sw/apps/snax-versacore-int16x4-shape1-common16-weight-overlap all CFG_OVERRIDE=$CFG
make -C target/snitch_cluster bin/snitch_cluster.vlt CFG_OVERRIDE=$CFG

target/snitch_cluster/bin/snitch_cluster.vlt \
  target/snitch_cluster/sw/apps/snax-versacore-int16x4-shape1-common16-weight-overlap/build/snax-versacore-int16x4-shape1-common16-weight-overlap.elf
```

第一次从增量 `work-vlt` 构建 E2 时，Verilator 5.034 出现一次 internal fault；清理 VLT build artifacts 后重试成功。它不是 RTL assertion/compile error，最终 E2 和 E3 均成功构建并通过功能仿真。
