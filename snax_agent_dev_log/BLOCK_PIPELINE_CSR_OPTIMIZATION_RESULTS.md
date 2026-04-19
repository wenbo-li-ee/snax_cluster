# Block Pipeline CSR 重配置优化实验记录

## 实验背景

将 M=20 分成 2 块 (M_BLOCK=10) 进行 block pipeline 计算。Writer 一次性配置全 M=20，不重启。Block 1 只需要重新配置 Reader 并重启。

初始实现（之前的完整重配置）中间 printf 导致 streamer 周期虚高。以下实验均去掉了中间 printf，只保留最后结果输出。

## 实验配置

- M=20, K=2, N=1, meshRow=16, tileSize=8, meshCol=8
- 分 2 块：Block 0 = M_tile 0~9, Block 1 = M_tile 10~19
- Weight stationary: B0/B1 的 M 方向 stride=0，每个 block 的 B 配置完全相同

## 实验结果

| 实验 | 描述 | Block 间 CSR 写次数 | Streamer 周期 | VC 周期 | 结果 |
|------|------|---------------------|---------------|---------|------|
| Baseline | 单块 M=20，无分块 | — | **95** | 84 | PASS |
| A (初版) | 重配全部 Reader CSR (46 写) + 重配 VC CSR (6 写) + printf | 52 + printf | 7523 | 41 | PASS |
| B | 只改 A 基地址 (1 写) + streamer start + VC start，wait_vc 后配置 | 3 | 5732* | 44 | PASS |
| C | 同 B，但 A 基地址在 wait_vc **之前**写入（overlap） | 3 | 5695* | 44 | PASS |
| D | 同 C，但去掉 wait_vc，让 STREAMER_START 和 VC_START 隐式 stall | 3 | 5695* | 41 | PASS |
| **D (无 printf)** | **最终版：同 D，去掉所有中间 printf** | **3** | **110** | **44** | **PASS** |

> *带 printf 的数字包含大量 UART 输出开销（每个 printf ~2000+ 周期），不反映真实硬件性能。

## 关键发现

### 1. printf 是周期杀手
去掉中间的 printf 后，streamer 周期从 5695 降到 **110**。实际 block pipeline 开销仅 **15 个 streamer 周期**（110 - 95 = 15），即 3 次 CSR 写 + 隐式 stall 等待。

### 2. 只需重配 1 个 CSR（A 基地址）
在 weight stationary 场景下，Block 间唯一变化的是 Reader A 的起始地址。ReqRspManager 的内部寄存器 `regs` 保持上次写入的值，只需覆盖变化的那一个，然后写 start 触发 config_fire，所有 reader 都会用更新后的 `regs` 值重启。

原理：
```
csrw_ss(BASE_PTR_READER_0_LOW, new_addr);  // 只更新 regs 中的 1 个值
csrw_ss(STREAMER_START_CSR, 1);            // config_fire → readers 用当前 regs 重启
set_dual_versacore_start();                // VC 用存储的配置重启
```

### 3. 不需要重新配置 VersaCore
VC 的 CSR Manager 同样保持上次的 `regs`。写 `DUAL_VC_START = 1` 会重新 fire 已存储的配置给 VC。因为两个 block 的 K、M_BLOCK、subtraction 等完全相同，不需要重新写 VC CSR。

### 4. 可以在 VC 完成前写入下一块的基地址
非 start 类 CSR 写入在 ReqRspManager 中总是被接受（`req.ready = 1`），不受 streamer busy 状态限制。写入只更新内部 `regs`，不影响正在运行的 reader（它们在 start 时已 latch 了配置）。

### 5. 隐式 stall vs 显式 polling
- 显式 polling (`wait_dual_versacore()`): 循环读 DUAL_VC_BUSY，每次迭代若干周期
- 隐式 stall（直接写 DUAL_VC_START）: Snitch core 冻结在 CSR 写指令，VC 空闲后立即执行

两者周期差异极小（本实验中 C 和 D 相同），但隐式 stall 代码更简洁。

### 6. Reader restart 可以在 VC 完成前发生
`STREAMER_START_CSR` 写入在 `readers_all_done` 时被接受。Readers 通常比 VC 更早完成（它们只负责发数据，VC 还要计算）。重启后的 reader 会开始读 Block 1 的数据，但 VC 还没准备好接收，数据暂存在 FIFO/lockstep buffer 中。VC 重启后直接从 buffer 取数据，实现了数据预取。

## 最终 Block Pipeline 流程（最优版）

```c
// === Block 0: 首次启动全部 ===
set_dual_versacore_streamer_csr(...);  // 配全部 Reader + Writer
set_dual_versacore_csr(...);            // 配 VC
set_dual_versacore_streamer_start();    // 启动全部
set_dual_versacore_start();             // 启动 VC

// === Block 1..N-1: 只改 A 基地址 + 重启 ===
csrw_ss(BASE_PTR_READER_0_LOW, new_a_addr);  // 1 CSR write
csrw_ss(STREAMER_START_CSR, 1);               // 隐式等 readers 完成后重启
set_dual_versacore_start();                   // 隐式等 VC 完成后重启

// === 最后一个 block 结束后 ===
wait_dual_versacore();         // 等 VC 计算完
wait_dual_versacore_writer();  // 等 Writer 写完
```

## 性能总结

对于 M=20, K=2, N=1 的工作负载：
- **单块基线**: 95 streamer 周期
- **2 块 pipeline（优化后）**: 110 streamer 周期
- **分块开销**: 仅 **15 周期**（约 16% overhead）
- **CSR 写开销**: 每个额外 block 只需 3 次 CSR 写
