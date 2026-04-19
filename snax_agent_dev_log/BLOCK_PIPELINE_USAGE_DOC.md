# Dual VersaCore SwiGLU — Block-Level Pipeline 修改说明与使用指南

## 1. 修改概述

本次修改实现了 Streamer 的 **Reader/Writer 解耦**，使得在 Writer 仍在写回上一块结果时，可以立即重新配置并启动 Reader，开始下一块的计算。核心思路：Writer 按完整 M 维度一次性配置、全程不重启；Reader 每处理完一个 block 就重新配置地址和 bound 后重启。

### 修改文件清单

| 文件 | 性质 | 改动内容 |
|------|------|----------|
| `hw/chisel/src/main/scala/snax/streamer/Streamer.scala` | HW (Chisel) | 新增 `readers_all_done` 信号、Writer start 解耦、CSR ready 放宽、新增 writer_busy RO CSR |
| `util/snaxgen/snaxgen.py` | 生成脚本 | `streamer_csr_num` 计算加 1（writer_busy CSR） |
| `target/.../include/snax-dual-versacore-swiglu-lib.h` | SW 头文件 | 新增 `WRITER_BUSY_CSR` 宏、函数声明 |
| `target/.../src/snax-dual-versacore-swiglu-lib.c` | SW 库 | 新增 `wait_dual_versacore_writer()`、`restart_dual_versacore_readers()` |
| `target/.../src/snax-dual-versacore-swiglu-test.c` | 测试程序 | 改写为 2-block pipeline 测试 |

提交：`bcf51dd6`，分支 `swiglue`

---

## 2. 硬件修改详解（Streamer.scala）

### 2.1 新增 `readers_all_done` 信号

```scala
val readers_all_done = Wire(Bool())
readers_all_done := !reader
  .map(_.io.busy)
  .reduceLeftOption(_ || _)
  .getOrElse(false.B)
```

当所有 Reader 都完成（busy=0）时为 `true`，即使 Writer 仍在运行。

### 2.2 Writer start 与 Reader start 解耦

```scala
// Reader：任何 config_fire 都重启（不变）
reader(i).io.start := streamer_config_fire

// Writer：只在 IDLE 状态的 config_fire 才启动
writer(j).io.start := streamer_config_fire && streamer_ready
```

这样在 Writer 还在跑（状态为 sBUSY）时触发的 config_fire 只会重启 Reader，不会打断 Writer。

### 2.3 CSR 接受条件放宽

```scala
// 原来：只有 IDLE 才接受新配置
csrManager.io.readWriteRegIO.ready := streamer_ready

// 现在：IDLE 或 所有 Reader 完成 都可以接受新配置
csrManager.io.readWriteRegIO.ready := streamer_ready || readers_all_done
```

### 2.4 新增 Writer-only Busy RO CSR

```scala
csrManager.io.readOnlyReg(0) := streamer_busy           // 原有：全局 busy
csrManager.io.readOnlyReg(1) := performance_counter      // 原有：性能计数器
csrManager.io.readOnlyReg(2) := writer                   // 新增：writer-only busy
  .map(_.io.busy)
  .reduceLeftOption(_ || _)
  .getOrElse(false.B)
```

对应 `numReadOnlyReg` 从 2 改为 3。

### 2.5 snaxgen.py 同步修改

```python
streamer_csr_num = (
    2 * num_t_loop_dim
    + num_s_loop_dim
    + 2 * num_data_mover
    + num_configurable_channel
    + address_remapper_csr_num
    + 1  # Performance counter
    + 1  # Busy register
    + 1  # Writer busy register  ← 新增
    + 1  # Start register
)
```

**这个改动非常关键**：它决定了 `snax_csr_mux_demux` 的 `AddrSelOffSet` 参数（62→63）。不改的话，所有加速器 CSR 地址会偏移 1 位，导致仿真挂死。

---

## 3. CSR 地址映射（生成后）

```
STREAMER_START_CSR              = 1019   // RW，写 1 启动 streamer
STREAMER_BUSY_CSR               = 1020   // RO，全局 busy（Reader | Writer）
STREAMER_PERFORMANCE_COUNTER_CSR = 1021  // RO，性能计数器
STREAMER_WRITER_BUSY_CSR        = 1022   // RO，仅 Writer busy（新增）

DUAL_VC_CSR_ADDR_BASE           = 1023   // 加速器 CSR 起始地址（自动偏移）
```

---

## 4. SW 新增 API

### `wait_dual_versacore_writer()`

轮询 `WRITER_BUSY_CSR` 直到 Writer 写回完成。用于最后一个 block 结束后等全部数据写回。

```c
void wait_dual_versacore_writer() {
    while (csrr_ss(WRITER_BUSY_CSR)) {}
}
```

### `restart_dual_versacore_readers(...)`

只重新配置 3 个 Reader（A/B0/B1）的 CSR，然后写 `STREAMER_START_CSR = 1` 触发 Reader 重启。**不重新配置 Writer CSR**，Writer 保持自动地址递增。

函数签名和 `set_dual_versacore_streamer_csr()` 中 Reader 部分完全一致，只是去掉了 Writer 参数。

---

## 5. 使用方法：如何编写 Block Pipeline 测试

### 5.1 基本流程

```
Block 0（第一次启动）
  ├── 配置所有 Streamer CSR（Reader + Writer），Writer 用完整 M 的 bound
  ├── 配置 VersaCore CSR（output_times = N × M_BLOCK）
  ├── set_dual_versacore_streamer_start()    // 启动全部
  ├── set_dual_versacore_start()             // 启动 VC
  └── wait_dual_versacore()                  // 等 VC 计算完

Block 1（后续 block，不等 Writer）
  ├── restart_dual_versacore_readers(...)    // 只配 Reader CSR + start
  ├── set_dual_versacore_csr(...)            // 配 VC CSR
  ├── set_dual_versacore_start()             // 启动 VC
  └── wait_dual_versacore()                  // 等 VC 计算完

... 可以继续更多 block ...

最终收尾
  ├── wait_dual_versacore_writer()           // 等 Writer 全部写完
  └── check_dual_versacore_result(...)       // 检查完整结果
```

### 5.2 关键参数计算

**Writer 配置（Block 0 时一次性设好）**：
- `Dtlbound[2]` = 完整 M（所有 block 的总和），不是单个 block 的 M
- Writer 的地址自动递增，不需要后续 block 再配

**Reader A 地址偏移（Block i）**：
```c
int32_t a_base_block_i = delta_local_a + i * M_BLOCK * Atlstride2;
// Atlstride2 = meshRow × tileSize × K × sizeof(int8) = M方向每 tile 的字节偏移
```

**Reader B0/B1 地址（Weight Stationary）**：
- 如果 B 的 M 方向 stride = 0（weight stationary），则每个 block 的 B 地址不变

**VersaCore output_times**：
```c
set_dual_versacore_csr(1, K, N * M_BLOCK, ...);  // 每个 block 只算 M_BLOCK 个 M tile
```

### 5.3 示例代码（M=20 分 2 块）

```c
const uint32_t M_BLOCK = 10;

// ===== Block 0 =====
// Reader bound: M维度 = M_BLOCK
int32_t Atlbound_b0[] = {Atlbound0, Atlbound1, M_BLOCK, Atlbound3, Atlbound4, Atlbound5};
// Writer bound: M维度 = 完整 M=20
int32_t Dtlbound_all[] = {Dtlbound0, Dtlbound1, M, Dtlbound3};

set_dual_versacore_streamer_csr(
    delta_local_a, ..., Atlbound_b0, ...,   // Reader A
    delta_local_b0, ...,                     // Reader B0
    delta_local_b1, ...,                     // Reader B1
    delta_local_d, ..., Dtlbound_all, ...);  // Writer D（完整 M）

set_dual_versacore_csr(1, K, N * M_BLOCK, ...);
set_dual_versacore_streamer_start();
set_dual_versacore_start();
wait_dual_versacore();

// ===== Block 1（不等 Writer！）=====
int32_t a_block1_base = delta_local_a + M_BLOCK * Atlstride2;

restart_dual_versacore_readers(
    a_block1_base, ...,     // A 新地址
    delta_local_b0, ...,    // B0 不变
    delta_local_b1, ...);   // B1 不变

set_dual_versacore_csr(1, K, N * M_BLOCK, ...);
set_dual_versacore_start();
wait_dual_versacore();

// ===== 收尾 =====
wait_dual_versacore_writer();
check_dual_versacore_result(..., d_data_length);  // 检查完整 M=20
```

---

## 6. 注意事项

### 6.1 改了 Streamer.scala 之后必须

```bash
make clean-vlt clean-generated   # 或 rm -rf bin/snitch_cluster.vlt work-vlt generated
make rtl-gen                     # 重新生成 RTL + CSR 地址映射
make bin/snitch_cluster.vlt      # 重新编译仿真器（~12分钟）
make sw                          # 重新编译软件
```

### 6.2 只改了 SW（.c/.h）

```bash
make sw                          # 只需重新编译软件
# 不用重新 rtl-gen 或 bin/snitch_cluster.vlt
```

### 6.3 CSR 地址偏移问题

新增 Streamer RO CSR 会导致 `AddrSelOffSet` 变化（控制 Streamer/Accelerator CSR 的地址分界），**必须同步修改 `snaxgen.py` 中的 `streamer_csr_num` 计算**。否则加速器的所有 CSR 写入都会指向错误地址。

### 6.4 config_fire 时 CSR 寄存器会被新值覆盖

Block 1 的 `restart_dual_versacore_readers()` 触发 config_fire 时，`csrCfgReg` 会被新的 Reader CSR 值覆盖。但这没关系——Writer 在 start 触发时已经把配置 latch 到自己内部的地址生成器中，不依赖外部 CSR 寄存器。

### 6.5 扩展到更多 block

流程可以自然扩展到 N 个 block：Block 0 正常启动全部，Block 1~N-1 都用 `restart_dual_versacore_readers()` + `set_dual_versacore_start()`，最后一个 block 结束后调用 `wait_dual_versacore_writer()` 收尾。

---

## 7. 验证结果

```
DBG: Block 0 start (M_tile 0~9)
DBG: Block 0 VC done
DBG: Block 1 start (M_tile 10~19)
DBG: Block 1 VC done
DBG: Writer done
Dual VersaCore SwiGLU: PASS, Error: 0.
Workload: M=20 (2 blocks of 10), N=1, K=2
Accelerator cycles: 41
Streamer cycles: 7523
```
