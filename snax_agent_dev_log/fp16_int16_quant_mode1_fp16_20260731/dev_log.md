# FP16/INT16 量化与 Mode1 FP16 输出开发日志

日期：2026-07-31

## 目标

- SwiGLU Mode0 保持 int16 后处理输出，并在 XDMA reader 增加 FP16→INT16 输入量化。
- SwiGLU Mode1 绕过两路 `rescale_down_32to16`，改为 buffer 后的 INT32→FP16，再交给两路 writer/streamer。
- FP16→INT16 扩展沿用参考 FP16→INT8 的 `inv_scale`、4 个计算 PE、流水线与 ready/valid 行为；由于 16→16 为 1:1，不做跨两个 beat 打包。

## 2026-07-31：代码基线与设计确认

- 目标仓库分支：`quant/dequant`，开始时工作区干净。
- 参考仓库存在用户修改：`StreamParamGen.scala` 与 `snax_versacore_to_simd_cluster.hjson`；本任务只读取，未覆盖。
- 当前 Mode1 数据流确认是两路 `VC output buffer → chunk serializer → rescale32to16 → output assembler → streamer`。
- 计划改为 Mode0 保留上述 rescale；Mode1 在 serializer 后分流到 4 个参考算法一致的 INT32→FP16 PE。
- 目标 XDMA 为 512-bit：每 beat 32 个 FP16；4 PE 时每 beat 需要 8 个 issue 子周期。FP16→INT16 输出元素数不变，`pack=1`。
- 参考 FP16→INT8 使用 native-Chisel FP32×FP16 乘法和 FP32 magic-number RNE；目标分支尚无 `fp-native` 依赖及 FP helper，需要做最小移植。

## 硬件实现

### FP16→INT16 XDMA reader extension

- 新增 `Fp16ToInt16.scala` 与 `FpHelpers.scala`，并从参考工程最小移植
  `fp-native` 依赖。
- 数值路径保持参考 FP16→INT8 的控制语义：每个 FP16 lane 先与 FP32
  `inv_scale` 相乘，再用 magic-number 方法做 round-to-nearest-even，最后做
  对称 INT16 饱和，输出范围为 `[-32767, 32767]`。
- 数据宽度为 512 bit，每 beat 包含 32 个 FP16。实例化 4 个 PE，分 8 个
  subcycle 完成；输入输出均为 16 bit，因此每个输入 beat 精确对应一个输出
  beat，不需要参考 FP16→INT8 的双 beat 收集/打包。
- 扩展带输入缓存、输出缓存、credit/busy 与完整 ready/valid backpressure。
- 修改 `XDMATop.scala` 的资源拼接逻辑，保留并前置 `fp-native` 生成的 package/
  module resource block，避免生成 RTL 时浮点依赖被截断或声明顺序错误。

### Mode1 INT32→FP16

- 最终实现为 `int32_to_fp16.sv`：signed INT32 转 IEEE FP16 的组合算术保持参考
  PE 一致，包括 leading-one detection、归一化、GRS round-to-nearest-even、
  mantissa carry 和溢出转 signed infinity。
- 模块参数化为 4 lane，并在组合转换后加入一个与 `rescale_down_32to16` 相同
  风格的单项弹性输出寄存器：`ready_o = !valid_o || ready_i`。因此模块延迟一拍、
  II=1，支持下游停顿保持以及同拍 pop+push。
- 两个 VersaCore 输出 buffer 后各实例化一个 4-lane SV pipeline。Mode1 两路
  serializer 分别进入转换 pipeline 和对应 writer；两路 rescale 仅在 Mode0
  接收 valid。
- `postproc_busy` 同时覆盖两个 INT32→FP16 pipeline 的输出 valid、buffer、Mode0
  后处理流水线和 writer 前输出寄存器，确保下一次 CSR launch 不会覆盖仍在途
  的 active config。

## cfg 与软件集成

- 在指定 cfg 的 XDMA `reader_extensions` 中加入 `HasFp16ToInt16`：
  `dataWidth=512`、`computeLanes=4`、`fpPipe=1`。
- XDMA 软件库新增启用/关闭 API；启用 API 写入一个 FP32 bit-pattern 形式的
  `inv_scale` CSR，并在未配置该扩展的 cfg 下返回错误。
- SwiGLU 软件库新增 FP16 bit-pattern 精确检查函数。
- 回归程序先由 iDMA 把 ELF 中的 FP16 A 搬到 TCDM staging buffer，再启动本地
  XDMA extension 写入 INT16 A。直接把外部地址交给 local XDMA 会被解释为
  remote-cluster task，因此不能用于这个扩展流。
- datagen 同时输出 `A_fp16`、XDMA INT16 golden，以及 Mode1 两路由 NumPy
  `float16` 生成的 bit-pattern golden。
- Mode0 只配置和等待 Writer0；Writer1 显式 idle。Mode1 配置两个 writer，输出
  缓冲区类型改为 `uint16_t`。
- 指定 cfg 的 reader spatial bounds 已是 A=`[2,8]`、B=`[2,4]`。旧 app 只传
  一个 spatial stride，库函数会越界读取数组并导致 A row 广播。现已为 A、B0、
  B1 都传入 `{8,16}`，恢复 16/8 个 64-bit channel 的连续地址映射。

## RTL 生成与调试记录

1. 首次 Verilator elaboration 报新 PE 的 `clock/reset` 未连接；补齐端口后不再有
   该类新增警告，剩余 PINMISSING/LATCH 均为工程已有警告。
2. 初版直接用 XDMA 读取 ELF 外部地址，task 不结束；确认该接口为 TCDM-local
   extension 后，改成 iDMA staging + local XDMA，转换精确通过。
3. 首轮 Mode0 表现为每 32 个元素重复一组 4-lane 结果。用临时 1.1 GB VCD
   逐级追踪后确认 `buf0_data` 本身已经重复，不是新增转换 PE 或 rescale 的数值
   错误；根因是多维 streamer 的第二个 spatial stride 未由旧软件提供。补齐
   `{8,16}` 后 Mode0 和后续 Mode1 均通过。
4. 调试波形 `sim.vcd` 是本次临时生成文件，完成诊断后已删除，不留在工作区。
5. 2026-08-01 根据微架构审查，将最初的纯组合 Chisel INT32→FP16 PE 替换为
   4-lane SV 弹性流水模块；同时删除旧的生成式 `Int32ToFp16PE.sv` 路径，Bender
   只包含新的 `int32_to_fp16.sv`。

## 验证结果

- `Fp16ToInt16Tester`：2/2 通过。
  - PE 边界值、RNE、对称饱和。
  - 7 个连续 512-bit beat、4 PE 时分复用、随机 backpressure、1:1 输出与 busy
    drain。
- `Int32ToFp16PipelineTest`：1/1 通过，覆盖正负数、tie-to-even、最大有限值、
  overflow、一拍延迟、backpressure 保持和同拍 pop+push。
- `hw/chisel`、`hw/chisel_acc` compile 通过。
- 指定 cfg 的 `rtl-gen` 通过；生成 XDMA RTL 中有且仅有 4 个 FP16→INT16 PE。
- Verilator 模型从最终生成 RTL 构建成功。
- 最终完整 RTL 回归退出码 0：

```text
FP16 -> INT16 XDMA result: PASS, Error: 0
Mode 0 SwiGLU D0: PASS, Error: 0
  M0 Cycles: accel=126, streamer=153, wall=2251
Mode 1 GEMM D0: PASS, Error: 0
Mode 1 GEMM D1: PASS, Error: 0
  M1 Cycles: accel=248, streamer=269, wall=2499
```

## 最终验证清单

- [x] FP16→INT16 PE 边界、RNE、饱和测试。
- [x] 512-bit/4-lane 扩展 backpressure 与连续 beat 测试。
- [x] 目标 cfg 生成 XDMA RTL 与 CSR header。
- [x] Mode0 int16 回归不退化。
- [x] Mode1 两路 FP16 bit pattern 对照软件 golden。
- [x] 完整 RTL 仿真退出码为 0。
