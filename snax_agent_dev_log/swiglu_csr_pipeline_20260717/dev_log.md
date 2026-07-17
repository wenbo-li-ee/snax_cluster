# SwiGLU CSR 流水线配置改造记录

日期：2026-07-17  
最后整理：2026-07-18

## 1. 改造结论

本次改造将 `snax_dual_versacore_swiglu` 的 RW CSR 数量统一为 23，并为
SwiGLU shell 增加了 staging/active 双配置 bank。软件现在可以在当前任务运行时
预写下一任务的 CSR；只有下一次 START 被硬件接受时，这组配置才会原子提交并
成为新的 active 配置。

这个软件可见行为与成熟的单 VersaCore 路径一致：

- 普通 CSR 写入只更新下一任务配置；
- START 是配置提交点；
- START 通过 valid/ready 握手被接受；
- 当前任务运行时使用 START 时锁存的配置，不受后续 CSR 写入影响。

SwiGLU 不是逐行复制单 VersaCore wrapper。它在两个 VersaCore 后面还有 rescale、
SiLU、逐元素乘法和输出重组，所以增加了 shell 级 `active_cfg`，并将 START ready
扩展为“两个 VersaCore 都 ready，且整个后处理流水线已经排空”。

最终在现有 4-lane cfg 和现有 small MoE app 上完成 Verilator 回归：三个 array
shape 的 Mode 0 和 Mode 1 共 6 项全部通过，`total error: 0`，仿真退出码为 0。

## 2. 修改前存在的问题

### 2.1 后处理 CSR 没有 active 配置快照

修改前，Mode、array-shape serializer 和三个 rescale 模块直接读取
`csr_reg_set_i`：

```text
软件写 CSR
    ↓
ReqRspManager 中的 RW CSR 寄存器
    ↓（直接组合连接）
Mode / shape serializer / rescale
```

`ReqRspManager` 中的寄存器会随软件写操作更新。因此，如果软件在当前任务运行时
提前配置下一任务，例如把 Mode 从 SwiGLU 改成 GEMM，当前任务的后处理控制也可能
立即变化。这不满足流水线配置要求。

### 2.2 busy 只反映两个 VersaCore

修改前的 accelerator busy 基本等价于：

```systemverilog
busy = vc0_busy || vc1_busy;
```

两个 VersaCore 不再 busy，只能说明它们的序列化输出已经完成，不能说明下游的
以下模块已经排空：

- 两路 VersaCore 输出 buffer；
- rescale0 和 rescale1；
- SiLU 内部三级流水线；
- element-mul 两个输入 FIFO 和输出寄存器；
- 乘法后的 rescale_mul；
- Writer0/Writer1 前的输出重组寄存器。

如果此时接受下一次 START 并切换 active 配置，上一任务仍在后处理路径中的数据
可能使用新任务的 Mode 或 rescale 参数。

### 2.3 不同 cfg 的 CSR 数量不一致

部分较新的 dual-VersaCore-SwiGLU cfg 已经是 23 个 RW CSR，但七个旧 cfg 仍然是
20。由于 ReqRspManager 将最后一个 RW CSR 作为 START，CSR 总数不一致也意味着
START 地址不一致，不利于统一软件接口。

## 3. 本次改造目标

1. 所有 dual-VersaCore-SwiGLU cfg 统一使用 23 个 RW CSR。
2. 软件能够在任务 N 运行期间写任务 N+1 的 accelerator CSR。
3. 任务 N 的 Mode、shape 和 rescale 参数在任务结束前保持稳定。
4. START 是唯一的 active 配置提交点。
5. START 只有在两个 VersaCore 和本地后处理路径都安全时才被接受。
6. 保持成熟 VersaCore 的 CSR 0–5 锁存语义和 valid/ready 控制方式。
7. 使用现有 cfg、现有 small app 和现有 golden data 验证，不新造一套简化硬件 cfg。

## 4. 统一后的 CSR 映射

| CSR 索引 | 含义 | active 配置使用者 |
|---|---|---|
| 0 | `take_in_new_c` / overwrite accumulation | 两个 VersaCore |
| 1 | temporal accumulation bound | 两个 VersaCore |
| 2 | output bound | 两个 VersaCore |
| 3 | subtraction constants | 两个 VersaCore |
| 4 | array shape | VersaCore + shell serializer |
| 5 | data type | 两个 VersaCore |
| 6 | Mode：0=SwiGLU，1=双 GEMM | shell 后处理选择逻辑 |
| 7–10 | rescale0 参数 | VC0 后处理 |
| 11–14 | rescale1 参数 | VC1 后处理 |
| 15–18 | rescale_mul 参数 | SwiGLU 乘法后处理 |
| 19–21 | 保留 | 当前不使用 |
| 22 | START | ReqRspManager 触发提交 |

总 RW CSR 数固定为 23。START 的 accelerator-local 地址为十六进制 `0x16`，即
十进制索引 22。

## 5. 硬件实现修改

### 5.1 生成器强制要求 23 个 RW CSR

源文件：

`hw/chisel_acc/src/main/scala/snax_acc/versacore/DualVersaCoreSwigluGen.scala`

默认值从 20 改为 23，并增加 elaboration-time 检查：

```scala
val RegRWCount = cfg.obj.get("snax_num_rw_csr")
  .map(_.num.toInt).getOrElse(23)

require(
  RegRWCount == 23,
  s"snax_dual_versacore_swiglu requires exactly 23 RW CSRs, got $RegRWCount"
)
```

这样如果以后新增 cfg 时又写成 20 或其他数量，RTL generation 会立即失败，而不
会生成一套 START 地址和软件库不一致的 RTL。

### 5.2 保留 ReqRspManager 作为 staging bank

没有重新实现一套 CSR manager。SNAX 原有 ReqRspManager 中的 23 个 RW CSR
继续作为 staging/next-job bank：

```text
CSR 0–21 普通写入：更新 staging bank
CSR 22 写 1：产生 csr_reg_set_valid_i，尝试 START 握手
```

ReqRspManager 对普通 CSR 写入不要求 accelerator ready，所以软件可以在当前任务
busy 时写任务 N+1 的参数。只有 START 写入需要等待 shell 的
`csr_reg_set_ready_o`。

### 5.3 shell 新增 active_cfg

在生成的 dual-VersaCore-SwiGLU shell 中增加：

```systemverilog
localparam int unsigned ActiveCfgCount = 19;
logic [ActiveCfgCount-1:0][RegDataWidth-1:0] active_cfg;
```

这里只保存 CSR 0–18：

- CSR 0–18 是当前任务真正需要的配置；
- CSR 19–21 是保留项；
- CSR 22 是 START 事件，不是持续配置。

只有 START valid/ready 同时成立时才更新 `active_cfg`：

```systemverilog
assign launch_fire = csr_reg_set_valid_i && csr_reg_set_ready_o;

always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
        active_cfg <= '0;
    end else if (launch_fire) begin
        active_cfg <= csr_reg_set_i[ActiveCfgCount-1:0];
    end
end
```

因此，任务运行期间的软件写操作只改变 staging bank，不改变 `active_cfg`。

### 5.4 shell 后处理全部改读 active_cfg

以下控制从直接读取 `csr_reg_set_i` 改为读取 `active_cfg`：

- `mode_sel`；
- rescale0 的 input zero-point、multiplier、output zero-point、shift；
- rescale1 的四个参数；
- rescale_mul 的四个参数；
- 根据 array shape 决定有效 chunk 数的 serializer 控制。

例如：

```systemverilog
assign mode_sel = active_cfg[6][0];
assign rescale0_multiplier = active_cfg[8];
assign rescale1_shift = active_cfg[14][7:0];
assign rescale_mul_shift = active_cfg[18][7:0];
```

这一步是“当前任务不受下一任务配置污染”的核心。

### 5.5 VersaCore CSR 0–5 为什么仍在 START 边沿直接采样 staging bank

两个内部 VersaCore 没有改动。成熟 VersaCore 本身已经包含内部 `csrReg`，并在
`io.ctrl.fire` 时锁存 CSR 0–5。因此 dual shell 在 `launch_fire` 同一个时钟沿将
`csr_reg_set_i[0:5]` 送给两个 VersaCore。

不能让 VersaCore 在这个边沿读取刚更新的 `active_cfg`，原因是 SystemVerilog
非阻塞赋值的寄存器新值要在该时钟沿结束后才可见；如果这样连接，VersaCore 会
采到上一任务的 `active_cfg`。

正确的同边沿行为是：

```text
START 握手边沿：
  VC0 内部 csrReg ← staging CSR 0–5
  VC1 内部 csrReg ← staging CSR 0–5
  shell active_cfg ← staging CSR 0–18
```

边沿之后，两个 VersaCore 使用各自内部 csrReg，shell 后处理使用 `active_cfg`，
三者都不再受 staging bank 后续写入影响。

### 5.6 START ready 扩展为端到端安全条件

修改后的控制为：

```systemverilog
assign cores_ready = vc0_ctrl_ready && vc1_ctrl_ready;
assign csr_reg_set_ready_o = cores_ready && !postproc_busy;
assign launch_fire = csr_reg_set_valid_i && csr_reg_set_ready_o;
assign ctrl_valid_to_vc = launch_fire;
```

含义如下：

1. 两个 VersaCore 必须都处于可接受新控制配置的状态；
2. 上一任务在 shell 本地后处理中的数据必须全部排空；
3. 只有满足上述条件，START 才同时启动两个 VersaCore 并提交 shell active 配置。

这与成熟单 VersaCore 的 `ctrl.fire` 语义一致，但 ready 条件更严格，因为融合
SwiGLU shell 比单 VersaCore 多了下游流水线。

### 5.7 postproc_busy 的构成

`postproc_busy` 定义为：

```systemverilog
assign postproc_busy = buf0_valid || buf1_valid ||
                       rescale0_out_valid || rescale1_out_valid ||
                       silu_busy || elem_mul_busy ||
                       rescale_mul_out_valid ||
                       out_assemble_0_valid || out_assemble_1_valid;
```

它覆盖上一任务可能残留数据的所有本地后处理状态。

accelerator 对软件暴露的 busy 也扩展为：

```systemverilog
busy = vc0_busy || vc1_busy || postproc_busy;
```

因此软件看到 busy 清零时，不只是两个 matmul core 结束，而且 shell 到 writer
接口之间也已经排空。

### 5.8 SiLU 增加 busy_o

源文件：

`hw/chisel_acc/src/main/resources/snax_acc/versacore/silu_multilane.sv`

SiLU 有三级 valid 流水，因此 busy 定义为任一级仍持有有效数据：

```systemverilog
assign busy_o = stage_valid[0] || stage_valid[1] || stage_valid[2];
```

只看 `valid_o` 不够，因为数据可能仍在 stage0 或 stage1，还没有到输出端。

### 5.9 element-mul 增加 busy_o

源文件：

`hw/chisel_acc/src/main/resources/snax_acc/versacore/elem_mul_16b.sv`

逐元素乘法器包含两个独立输入 FIFO 和一个输出寄存器，因此：

```systemverilog
assign busy_o = fifo0_valid || fifo1_valid || valid_o;
```

这也能覆盖两路输入暂时不齐、只有一侧 FIFO 持有数据的情况。

### 5.10 为什么不把 A 输入预取 buffer 计入 postproc_busy

共享 A 路径有一个 shell 输入 buffer。它可能在下一任务 START 之前由 streamer
预填。如果把它计入 `postproc_busy`：

```text
下一任务 A 已预填 → postproc_busy=1 → START 不可接受
START 不可接受 → VersaCore 不消费 A → buffer 永远不空
```

这会形成死锁。因此 A 输入预取 buffer 被有意排除在 `postproc_busy` 之外。
它属于下一任务的输入预取状态，不是上一任务的后处理残留状态。

## 6. 与成熟 VersaCore 方法的比较

| 项目 | 成熟单 VersaCore | 改造后的 dual-VersaCore-SwiGLU |
|---|---|---|
| staging CSR | ReqRspManager RW CSR | 相同 |
| START 触发 | 最后一个 RW CSR 写 1 | 相同，固定为 CSR 22 |
| 提交条件 | `ctrl.valid && ctrl.ready` | 相同 valid/ready 语义 |
| 核心配置锁存 | VersaCore 在 `ctrl.fire` 锁存内部 `csrReg` | 两个 VersaCore 同时锁存 CSR 0–5 |
| 当前任务配置隔离 | VersaCore 内部 `csrReg` | VersaCore 内部 `csrReg` + shell `active_cfg` |
| ready 条件 | 单个 VersaCore idle | 两个 VersaCore ready 且后处理排空 |
| shell 后处理快照 | 不需要 | Mode、shape、rescale 必须快照 |
| RW CSR 数 | 常规单 VersaCore 为 7 | 固定为 23 |

结论：软件编程模型和原子提交方法是一致的；SwiGLU 根据融合流水线增加了必要的
active bank 和端到端 drain 条件，属于对成熟方法的扩展，而不是另一套不兼容协议。

## 7. cfg 修改

检查到 14 个 dual-VersaCore-SwiGLU HJSON 配置。最终全部满足：

```hjson
snax_num_rw_csr: 23
```

用于最终功能回归的 cfg 是：

`target/snitch_cluster/cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson`

这个 cfg 原本已经是 23 CSR，测试过程中没有修改其 shape、端口、MAC 数、streamer
拓扑或稀疏互连配置。

七个旧 cfg 原本为 20 CSR，本次改为 23。同时为了让这些旧 cfg 能通过当前生成和
Verilator 流程，还做了两项兼容修复：

- XDMA 字段 `max_mem_size` 更新为当前生成器要求的 `max_mem_size_kiB`，数值仍为
  4096；
- cluster Bender target 列表补上 `snitch_cluster`，使 Verilator 能包含
  `idma_inst64_top`。

这两项不改变 SwiGLU 数学功能或目标 4-lane cfg 的硬件参数。

## 8. 软件回归 app 修改

使用现有 app：

`target/snitch_cluster/sw/apps/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15`

修改前每个任务都是临近 START 才配置 accelerator CSR：

```text
配置 Mode 0 accelerator CSR → START Mode 0
等待并检查 Mode 0
配置 Mode 1 accelerator CSR → START Mode 1
等待并检查 Mode 1
```

修改后特意形成 CSR 流水线：

```text
配置 Mode 0 accelerator CSR
START Mode 0
立即预写完整 Mode 1 accelerator CSR
等待并检查 Mode 0
只配置 Mode 1 streamer CSR
不再重写 Mode 1 accelerator CSR
START Mode 1，提交之前预写的 staging bank
等待并检查 Mode 1
```

关键代码行为：

```c
set_dual_versacore_start();  // 提交并启动 Mode 0

// Mode 0 运行期间预写下一任务
set_dual_versacore_csr(...Mode 1 bounds...);
set_dual_versacore_mode(1);
configure_identity_rescale_for_mode1();
```

之后 Mode 1 路径不调用上述 accelerator 配置函数，只发送下一次 START。

这个测试可以同时证明：

1. Mode 0 golden 仍通过：预写 Mode 1 没有污染当前 active 配置；
2. Mode 1 golden 通过：预写配置确实保存在 staging bank，并在第二次 START 时正确
   提交；
3. 三个 shape 都通过：shape-dependent serializer 也使用了正确的 active shape。

## 9. 软件构建依赖修复

源文件：

`target/snitch_cluster/sw/snax/dual-versacore-swiglu/Makefile`

测试过程中发现，`rtl-gen` 更新以下生成头文件后，SwiGLU 软件库对象不一定自动
重编译：

- `snax_dual_versacore_stationarity.h`；
- `streamer_csr_addr_map.h`。

这会导致新 cfg 的 app 链接旧 cfg 对应的软件库。初次 small app 仿真因此出现两个
VersaCore 一直 busy 的假失败。

Makefile 现已增加：

```make
CFG_HEADERS = $(MK_DIR)include/snax_dual_versacore_stationarity.h \
              $(MK_DIR)include/streamer_csr_addr_map.h

$(OBJS): $(CFG_HEADERS)
```

使用 `make -n -W <header> ...` 验证后，任一生成头文件变化都会触发库对象重编译。

## 10. 验证配置与数据规模

最终功能回归使用现有 cfg：

```text
cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson
```

使用现有 small MoE app：

```text
snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15
```

使用 app 原有数据生成参数：

| 参数 | 数值 |
|---|---:|
| M | 8 |
| K0 | 1024 |
| N0 | 128 |
| K1 | 128 |
| N1 | 512 |

测试覆盖 cfg 中三个 array shape：S0、S1 和 S2，以及每个 shape 的 Mode 0/Mode 1。

## 11. 验证步骤和结果

### 11.1 静态检查

- 14 个 dual-VersaCore-SwiGLU cfg 全部为 23 RW CSR；
- 生成 shell 的 `RegRWCount = 23`；
- ReqRspManager 的 START decode 地址为 `0x16`；
- Mode、rescale 和 shape serializer 的生成 RTL 均读取 `active_cfg`；
- 只有 CSR 0–5 在 START 边沿直接送入 VersaCore，符合其内部锁存语义；
- `git diff --check` 通过。

### 11.2 RTL generation

以下两类配置完成 full `rtl-gen`：

- 原始 336-port dual-VersaCore-SwiGLU 配置；
- 最终使用的 34-port、4-lane 配置。

两者 RTL generation 均通过。

### 11.3 软件构建

- 生成 streamer 和 stationarity 头文件；
- 重建 dual-VersaCore-SwiGLU 软件库；
- 重建 focused small MoE app；
- 软件编译和链接通过；
- 仅存在仓库原有的 `static-in-inline` 警告。

### 11.4 Verilator 构建

最终 4-lane 模型构建结果：

- Verilator 版本：5.034；
- elaborated modules：782；
- generated C++ files：196；
- 构建 wall time：235.858 秒；
- 构建成功，已有 lint 警告为 non-fatal。

### 11.5 focused ELF 仿真

最终结果：

```text
S0 Mode 0: PASS
S0 Mode 1 using pipelined core config: PASS
S1 Mode 0: PASS
S1 Mode 1 using pipelined core config: PASS
S2 Mode 0: PASS
S2 Mode 1 using pipelined core config: PASS

total checks: 6, total error: 0
```

仿真进程退出码为 0。

这不是只验证“原来功能还能跑”。Mode 1 明确不重新配置 accelerator CSR，因此
Mode 1 PASS 是 staging bank 在 START 时正确提交的直接证据；Mode 0 同时 PASS 是
active bank 隔离正确的直接证据。

## 12. 复现命令

在 `barnard3` 容器内执行，并保持所有步骤使用同一个 `CFG_OVERRIDE`：

```sh
make -C target/snitch_cluster rtl-gen \
  CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson

make -B -C target/snitch_cluster \
  /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster/target/snitch_cluster/generated/bender_targets.tmp \
  CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson

make -C target/snitch_cluster/sw/snax/dual-versacore-swiglu clean
make -C target/snitch_cluster/sw/snax/dual-versacore-swiglu all

make -C target/snitch_cluster/sw/apps/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15 clean \
  CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson

make -C target/snitch_cluster/sw/apps/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15 all \
  CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson

make -C target/snitch_cluster bin/snitch_cluster.vlt -j$(nproc) \
  CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson

timeout 300s ./target/snitch_cluster/bin/snitch_cluster.vlt \
  ./target/snitch_cluster/sw/apps/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15/build/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15.elf
```

## 13. 调试过程中发现的问题

### 13.1 旧 XDMA cfg 字段

旧 cfg 使用 `max_mem_size`，当前 `snaxgen.py` 读取 `max_mem_size_kiB`，导致 XDMA
generation 失败。字段更新后 full `rtl-gen` 通过。

### 13.2 stale bender_targets.tmp

`generated/bender_targets.tmp` 没有随 CFG_OVERRIDE 自动刷新时，Verilator 可能继续
使用旧 cluster target。通过强制重建该文件解决。

### 13.3 缺少 snitch_cluster Bender target

七个旧 cfg 的 target 列表没有 `snitch_cluster`，导致 Verilator elaboration 找不到
`idma_inst64_top`。补充 target 后构建通过。

### 13.4 软件库对象没有随生成头文件重建

初次 small app 仿真出现 accelerator timeout，诊断时两个 VersaCore busy、streamer
和 writer 已经 idle。根因不是 CSR 双 bank，而是旧软件库对象使用了与当前 cfg 不
匹配的 stationarity/streamer 配置。clean rebuild 后原始 small app 和流水线 CSR
版本都通过全部 6 项。随后补上 Makefile 依赖，避免再次发生。

## 14. 已验证范围与未验证范围

### 已验证

- 23 CSR 接口和 START 地址生成正确；
- staging CSR 可以在当前任务运行时写入；
- 当前任务不会被下一任务 Mode/rescale/shape 配置污染；
- 下一次 START 能提交之前预写的配置；
- S0/S1/S2 三个 shape 均正确；
- Mode 0 SwiGLU 和 Mode 1 双 GEMM 均通过 golden；
- full RTL generation、软件构建和 Verilator 构建通过；
- focused 4-lane 仿真退出码为 0。

### 尚未完全验证

- FPGA 或硅后运行；
- 形式验证；
- 更多量化参数和非 identity rescale 组合；
- 软件以最小间隔连续发出多个 START 的吞吐测试；
- 336-port 大配置的完整功能仿真。

336-port 模型已经通过 RTL generation、软件构建和 Verilator 构建，但 focused ELF
在 600 秒 wall-clock timeout 前还没有输出 app 结果。因此该项记录为“仿真结果
不确定”，不能算作功能 PASS。最终功能结论来自现有 4-lane cfg 的完整 small-app
回归。

## 15. Review 时建议重点检查

1. `active_cfg` 只在 `launch_fire` 更新；
2. shell-owned Mode、shape serializer 和 rescale 全部读取 `active_cfg`；
3. 两个 VersaCore 的 CSR 0–5 仍在 START 边沿直接锁存 staging bank；
4. `csr_reg_set_ready_o` 同时包含两个 core ready 和 `!postproc_busy`；
5. SiLU 和 element-mul 的 busy 覆盖所有内部 valid/FIFO；
6. A 输入预取 buffer 没有错误地加入 `postproc_busy`；
7. app 的 Mode 1 路径确实没有重新写 accelerator CSR；
8. 重新生成 cfg 后软件库会因配置头文件变化而自动重建；
9. 所有相关 cfg 的 `snax_num_rw_csr` 都为 23。
