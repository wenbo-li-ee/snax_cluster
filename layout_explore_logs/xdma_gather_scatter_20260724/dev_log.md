# XDMA gather/scatter 开发日志（2026-07-24）

## 背景

`common16_resident_k1024_20260718` 的布局中，一个 token 的一个 K8
片段占两个 64-bit bank。XDMA 数据宽度为 512 bit（8 个 64-bit lane），
因此一次需要同时搬运四个 token 才能用满带宽。原 Cartesian AGU 只能用
一个 spatial stride 生成 8 个等距地址，不能选择四个任意 token。

## 硬件设计

- 保留模式 0：原 `base + temporal_offset + spatial_index * stride`。
- 新增模式 1（indexed-lane）：地址变为
  `base + temporal_offset + channel_offset[lane]`。
- 每个 reader/writer AGU 各有一个 mode CSR 和 8 个 lane offset CSR。
- offset 使用字节地址、相对各自 base pointer；两个相邻 offset 表示一个
  16-byte token 片段，8 个 lane 可组成四个任意 token。
- temporal bound/stride 在两种模式都继续工作，因此选好四个 token 后，
  可以用共同的 row/K stride 连续搬运后续片段。
- `has_gather_scatter` 是 XDMA cfg 开关；未打开时不生成新 CSR/逻辑。
  accelerator 与 TCDM 之间的普通 streamer 参数保持默认关闭，不受影响。
- 软件原 `snax_xdma_memcpy_nd()` 每次显式恢复模式 0，防止前一条
  gather/scatter 描述符的 mode 泄漏到旧 stride 任务。

## 验证计划

同一个测试 app 依次验证：

1. 原 stride 模式连续搬运，证明兼容性；
2. gather：从 bank pair `(3,4)、(17,18)、(29,30)、(45,46)` 聚合；
3. scatter：分散到另一组不规则 bank pair；
4. gather→scatter：同一个任务中 source/destination 同时使用 indexed；
5. 每项覆盖 3 个 temporal row，确认 indexed spatial offset 与原 temporal
   stride 可以组合，并检查 scatter 未选择的 bank 仍为 guard 值。

## 开发过程与问题修复

### 1. 识别原 AGU

原 `AddressGenUnit` 先按 `spatialBounds` 把 lane 编号展开成多维坐标，
再累加各维 `coordinate * spatialStride`，所以它确实是 Cartesian AGU。
这部分保留为 mode 0，没有改变原地址序列。

### 2. 第一次 RTL 生成：零宽可选字段未赋值

最初给公共 `AddressGenUnitCfgIO` 增加可选 mode/offset 字段后，未启用功能的
VersaCore streamer 在 elaboration 时报告零宽 `addressMode` 未初始化。
修复方式是在所有关闭分支显式驱动 `addressMode := 0.U`。重新生成后，
普通 `snax_dual_versacore_swiglu_Streamer.sv` 中搜索
`addressMode/channelOffsets` 为 0 个结果，确认 accelerator streamer 的
端口、CSR 和地址生成逻辑没有变化。

### 3. 第二次 RTL 生成：XDMA cfg 参数传播不完整

`XDMACfgIO` 内部会重新构造一份 AGU 参数，第一次修改遗漏了
`hasGatherScatter`，造成控制侧有 8 个 offset、传输 cfg 侧却是 0 个，
elaboration 报 Vec 长度不一致。修复为从 XDMA reader/writer 参数显式传播
该开关；同时把 mode/offset 补入 intra-cluster cfg 和 inter-cluster
serializer/deserializer。offset 跨 cluster 传输时按 64-bit word 编码，
落到本地 AGU 前恢复成 byte offset。

### 4. CSR 与软件接口

生成的 XDMA 地址头文件中 `XDMA_HAS_GATHER_SCATTER=1`，source 和
destination 各有 1 个 mode CSR 与 8 个 offset CSR。软件库增加：

- `snax_xdma_set_src_address_mode(mode, offsets)`
- `snax_xdma_set_dst_address_mode(mode, offsets)`

旧 `snax_xdma_memcpy_nd[_full_addr]()` 每次配置任务时主动把两侧恢复为
stride mode，保证旧程序即使在 gather/scatter 任务之后运行也不会继承状态。

## 构建与闭环结果

使用配置：

`cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson`

在 `barnard3` 容器中完成：

1. `rtl-gen` 成功，XDMA 生成 RTL 包含 mode/offset，普通 accelerator
   streamer 不包含；
2. `snax-xdma-lib` 和 `snax-xdma-gather-scatter.elf` 编译成功；
3. 完整 `bin/snitch_cluster.vlt` 构建成功；
4. Verilator 执行测试 ELF，退出码为 0，输出：

```text
XDMA_STRIDE_MODE PASS
XDMA_GATHER_MODE PASS
XDMA_SCATTER_MODE PASS
XDMA_GATHER_TO_SCATTER PASS
XDMA_GATHER_SCATTER PASS errors=0
```

结论：旧 Cartesian stride、单侧 gather、单侧 scatter、双侧同时
gather→scatter 均已在 RTL 仿真闭环通过。当前 indexed 模式表达的是
“每个 512-bit beat 的 8 个 64-bit lane 各自一个固定 base-relative
offset，并叠加原 temporal offset”；这正好表达一次搬四个任意 16-byte
token，同时仍可用 temporal stride 连续处理多行。
