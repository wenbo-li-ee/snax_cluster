# common16 resident layout：streamer 维度与 TCDM sparse granularity 下界

日期：2026-07-31

## 1. 范围与基线

本实验只针对以下固定 contract：

- memory layout：`multishape_k8_common16_chunked_20260718/resident_k1024.md`；
- 每次 accelerator invocation 固定处理 16 columns；
- W/V/W2 在任何 shape 开始前全部搬入 8 MiB TCDM，计算阶段不再搬 weight；
- 依次验证 S0 `(8,8,4)`、S1 `(4,8,8)`、S2 `(2,8,16)` 的 Mode0 和 Mode1；
- 修改用户指定 cfg：`snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane_bgran2_highgran_search_2.hjson`。

这里讨论的 `sparse_interconnect_config` access granularity 是 streamer 端口到
64-bank TCDM 的静态可达 bank 同余类，不是 VersaCore 内部的 `granularity_a/b/c_d`
serialization 参数。

## 2. temporal dimension 理论下界

逐端口统计所有 shape/mode 的非退化 temporal loop：

| 端口 | 最坏情况的有效 bounds | 所需维度 |
|---|---|---:|
| A reader | S1 Mode1 `[2,64,2]`；S2 Mode1 `[2,2,32]` | 3 |
| B0/B1 reader | S0/S1 `[4,K/4,N_tiles]` | 3 |
| D0/D1 writer | S1 `[2,4,2]`，S0 Mode0 `[8,2,2]` | 3 |

因此 reader temporal dimensions 可从 `[6,4,4]` 降到 `[3,3,3]`，writer 可从
`[4,4]` 降到 `[3,3]`。不能统一降到 2，否则至少无法表达 S1 Mode1-A、B 的
`row-within-panel / K-row / N-tile`，或 S1 writer 的三个独立地址循环。

该下界专用于 `CHUNK_COLS=16`。若 S2 在一次 invocation 中处理多个 16-column
N tile，Mode1-A 还需要一个 stride=0 的 broadcast temporal dimension，A 将重新需要
4 维；因此本最小 cfg 不承诺支持 `CHUNK_COLS>16`。

## 3. spatial dimension 理论下界

| 端口 | spatial bounds | 下界原因 |
|---|---|---|
| A | `[2,8]` | S2 Mode1 offset 为 `0,8,32,40,...`，不是单一等差数列 |
| B0/B1 | `[2,4]` | 同时表达 panel 内两个 8 B half 与相隔 `panel_span` 的四个 panel |
| D0/D1 | `[1]` | 单 writer channel；AGU 保留一个退化 spatial 维 |

所以 spatial dimension 数已经是最小的 `2/2/2/1/1`，本轮不改。

## 4. sparse TCDM access granularity 理论上界

硬件要求 port 内 channel `i` 的每个有效访问满足：

```text
bank(address) mod granularity == i mod granularity
```

- A：Mode0、Mode1-S0/S1 可支持更大值，但 S2 Mode1 的最小 temporal stride 是
  16 B = 2 banks，且 spatial offset 相对 channel index 只保持 mod-2，因此公共最大值是 2。
- B0/B1：最内层 temporal stride 是 16 B = 2 banks，所以公共最大值也是 2。
- D0/D1：每个 writer 只有一个 input，`width/granularity` 必须至少为 1，所以只能是 1。

最终配置为：

```text
[[16,2], [8,2], [8,2], [1,1], [1,1]]
```

accelerator streamer 端口对每个 bank 的 sparse arbiter input fan-in 贡献从指定 cfg 原来的
`16/1 + 8/2 + 8/2 + 1 + 1 = 26` 降为
`16/2 + 8/2 + 8/2 + 1 + 1 = 18`，减少 `8/26 = 30.77%`。

完整 cluster 生成出的 sparse config 还追加 DMA `[16,1]` 和三个单输入
`[1,1]` 端口。因此完整每-bank arbiter fan-in 是：

```text
旧：26 + 16 + 1 + 1 + 1 = 45
新：18 + 16 + 1 + 1 + 1 = 37
```

即全互连每 bank 实际减少 8 个输入，降幅 `8/45 = 17.78%`。

## 5. 实际修改

### 5.1 cfg

指定 cfg 已修改为：

```text
reader temporal_dim = [3,3,3]
writer temporal_dim = [3,3]
spatial dimensions  = [2,2,2,1,1]  # bounds 不变
sparse config       = [[16,2],[8,2],[8,2],[1,1],[1,1]]
```

RTL 重新生成后，`streamer_csr_addr_map.h` 中五个端口的 temporal bound/stride
数量均为 3。整个 streamer CSR address span 从旧的 `960..1036`（77 个地址）缩为
`960..1022`（63 个地址），正好删除 7 个 temporal dimension 对应的 14 个
bound/stride CSR。

### 5.2 app 与 datagen

- Mode0 S0 writer 将旧 `[1,8,2,2]` 去掉 leading bound=1，编码为 `[8,2,2]`；
- S1 writer 直接使用 `[2,4,2]`；
- S2 common16 writer 使用 `[4,2,1]`；
- A/B/D 的 runtime 数组改成与设计时三维容量一致；
- datagen 同时断言新的 temporal/spatial/sparse contract，防止用错 cfg；
- 该最小 cfg/app 明确限制 `CHUNK_COLS=16`；更粗 S2 invocation 的 Mode1-A
  broadcast 不属于本次下界；
- 在 SW 与 apps Makefile 中加入指定 cfg 文件名的显式路由，使完整 `make sw`
  能构建 dual-versacore library 与目标 app。

## 6. clean build

所有命令都在 `barnard3` 容器的 pixi 环境中执行。原始日志位于：

```text
layout_explore_logs/common16_streamer_min_20260731/
```

| 步骤 | 命令摘要 | 结果 | 墙钟 |
|---|---|---|---:|
| clean | `make -C target/snitch_cluster clean`，再显式 clean 目标 library/app | PASS | 12.39 s |
| RTL | `make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=...search_2.hjson` | PASS | 1:49.79 |
| SW | `make -C target/snitch_cluster sw CFG_OVERRIDE=...search_2.hjson -j16` | PASS | 9.03 s |
| VLT | `make -C target/snitch_cluster bin/snitch_cluster.vlt CFG_OVERRIDE=...search_2.hjson -j16` | PASS | 4:12.18 |

RTL elaboration 明确打印：

```text
temporal_dim readers = [3,3,3]
temporal_dim writers = [3,3]
sparse_interconnect_config = [[16,2],[8,2],[8,2],[1,1],[1,1]]
streamer_csr_num = 63
```

构建仅有仓库既有的 Chisel width/unused-import 和 C `static-in-inline` warning；没有
编译或链接错误。

原始日志：

- `01_make_clean.log`
- `02_rtl_gen.log`
- `03_sw_build.log`
- `04_vlt_build.log`

## 7. 完整硬件仿真

运行的是全搬完再计算的 app：

```text
snax-versacore-int16x4-multishape-k8-common16-chunked.elf
SELECT_SHAPE=-1, CHUNK_COLS=16, RUN_MODE1=1
```

执行顺序得到日志确认：

```text
64 个 Mode0 weight chunks
64 个 Mode1 weight chunks
WEIGHTS_RESIDENT_READY ... no_more_weight_dma=1
S0 Mode0 -> S0 Mode1 -> S1 Mode0 -> S1 Mode1 -> S2 Mode0 -> S2 Mode1
```

第一个 `SHAPE_BEGIN` 之后的 `WEIGHT_DMA_*` 行数为 0，确认不是边搬边算。

weight 搬运摘要：

| Mode | payload | chunks | descriptors | 仿真 cycles | region end |
|---|---:|---:|---:|---:|---:|
| 0 W/V | 2 MiB | 64 | 128 | 1,497,825 | 4,194,304 |
| 1 W2L/W2R | 1 MiB | 64 | 128 | 1,447,507 | 6,291,456 |

这里 summary cycles 包含每 chunk UART 打印等软件开销，不能当作纯 DMA payload
吞吐；本实验的目标是功能与可达性验证。

计算结果：

| Shape | Mode | chunks | status | accel sum | streamer sum | wall sum |
|---|---:|---:|---|---:|---:|---:|
| S0 `(8,8,4)` | 0 | 64 | PASS | 65,920 | 67,456 | 168,546 |
| S0 `(8,8,4)` | 1 | 64 | PASS | 33,152 | 34,304 | 136,285 |
| S1 `(4,8,8)` | 0 | 64 | PASS | 33,152 | 34,688 | 133,626 |
| S1 `(4,8,8)` | 1 | 64 | PASS | 16,768 | 17,920 | 119,959 |
| S2 `(2,8,16)` | 0 | 64 | PASS | 16,896 | 18,432 | 116,287 |
| S2 `(2,8,16)` | 1 | 64 | PASS | 8,704 | 9,856 | 111,884 |

验收统计：

```text
CHUNK_RESULT lines = 384
FAIL/MISMATCH/TIMEOUT/Illegal bank access/Assertion failed = 0
FINAL_RESULT selected_shape=-1 chunk_cols=16 status=PASS total_errors=0
host wall time = 36:36.80
process exit = 0
```

完整仿真日志为 `05_full_run.log`。

## 8. 最终结论

对于固定 common16、全 weight resident 的当前 memory layout，已经由 clean RTL build
和完整硬件仿真验证的最小配置是：

```text
temporal dimension counts: A/B0/B1/D0/D1 = 3/3/3/3/3
spatial dimension counts:  A/B0/B1/D0/D1 = 2/2/2/1/1
sparse access granularity: A/B0/B1/D0/D1 = 2/2/2/1/1
accelerator streamer fan-in contribution per bank: 26 -> 18 (-30.77%)
full-cluster sparse arbiter fan-in per bank:        45 -> 37 (-17.78%)
streamer CSR addresses: 77 -> 63 (-14)
```

“TCDM 连接 granularity 降低”若指 accelerator streamer 部分的硬件连接，结果是
每 bank `26 -> 18`；计入未修改的 DMA/core/system 端口后，完整 arbiter 是
`45 -> 37`。若指 HJSON 中 access-granularity 数值本身，则数值是 `2/2/2/1/1`，其中
数值越大、实际连接越少。A 或 B 再提高到 4 会违反本 layout 的 mod-2 bank 可达性，
writer 由于 width=1 也不能提高。
