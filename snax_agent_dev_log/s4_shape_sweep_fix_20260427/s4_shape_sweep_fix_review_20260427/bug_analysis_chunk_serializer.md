# Bug 深度分析：DualVersaCoreSwigluGen Chunk 串行化计数错误

**日期**: 2026-04-27  
**文件**: `hw/chisel_acc/src/main/scala/snax_acc/versacore/DualVersaCoreSwigluGen.scala`  
**关联 Dev Log**: `snax_agent_dev_log/s4_shape_sweep_fix_20260427/`

---

## 背景：硬件数据流架构

在理解这两个 bug 之前，需要先理解 shell wrapper 的数据流结构。

### VersaCore 输出的物理结构

VersaCore（VC）的输出总线宽度是固定的：

```
DataWidthD = 1024 bits = 32 个 int32 元素
```

这是硬件物理宽度，与当前运行的 array_shape 无关。不同 shape 实际输出的有效元素数量不同：

| Shape | meshRow | meshCol | 有效 int32 元素数 |
|-------|---------|---------|------------------|
| S0 | 8 | 4 | 8 × 4 = **32** |
| S1 | 4 | 4 | 4 × 4 = **16** |
| S2 | 2 | 4 | 2 × 4 = **8** |

VC 输出的无效部分（高位）是零或无意义数据。

### PostprocLanes 与 chunk 概念

后处理流水线（rescale → SiLU → ElemMul → Writer）每次只能处理 `PostprocLanes = 4` 个 int32 元素。因此，VC 的 1024-bit 输出必须被切分成若干个 **chunk**，逐个送入后处理流水线：

```
总 chunk 数 NumChunks = ceil(ElemsPerBeat / PostprocLanes)
                      = ceil(32 / 4) = 8
```

切分后，每个 shape 的有效 chunk 数量：

| Shape | 有效元素 | 有效 chunk 数 | 无效 chunk 数 |
|-------|---------|--------------|--------------|
| S0 | 32 | **8** | 0 |
| S1 | 16 | **4** | 4 |
| S2 | 8 | **2** | 6 |

### Writer Streamer 的配置

Writer streamer 由 datagen 生成的 CSR 配置。Writer 的 beat bound（每个 tile 接收多少拍）是**形状感知**的：

| Shape | beats_per_tile（Writer 配置） |
|-------|------------------------------|
| S0 | 8 |
| S1 | **4** |
| S2 | **2** |

Writer 收到足够的 beat 后，`acc2stream_ready` 信号会拉低（quota 满），拒绝接收新数据。

---

## Bug 1：Chunk 串行化器计数硬编码为 NumChunks=8

### 问题代码（修复前）

```sv
localparam int unsigned NumChunks = 8;  // 物理最大值

// chunk_last 逻辑：永远等到 count==7 才触发
assign chunk_last_0 = (NumChunks <= 1) ||
                      (chunk_cnt_0 == NumChunks - 1);  // == 7，固定
```

这里 `chunk_last_0` 决定了什么时候认为当前 tile 的 chunk 串行化完成：只有 counter 到达 7（即发完 8 个 chunk），才触发 `chunk_last_0`，才能通过 `buf0_out_ready` 释放 buffer。

```sv
assign buf0_out_ready = chunk_ser0_ready && chunk_last_0;
```

### S0 为何能正常运行

S0 有 32 个有效元素 = 8 个有效 chunk。Writer 也配置为每 tile 接收 8 拍。

```
chunk_cnt: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7
Writer:    接  接  接  接  接  接  接  接（quota 满，下一 tile 等）
chunk_last: 只在 count=7 时触发 ✓（和 Writer 配置匹配）
```

流水线正常推进，没有死锁。

### S1 的死锁过程（详细）

S1 有 16 个有效元素 = 4 个有效 chunk。但 shell 不感知这一点，仍然按 8 个 chunk 进行串行化。Writer 配置为每 tile 只接收 4 拍：

**第一阶段：前 4 个 chunk 正常流动**

```
chunk_cnt=0: chunk 0（有效数据）→ rescale → ... → acc2stream_ready=1 → Writer 接收，beat_cnt=1
chunk_cnt=1: chunk 1（有效数据）→ ... → Writer 接收，beat_cnt=2
chunk_cnt=2: chunk 2（有效数据）→ ... → Writer 接收，beat_cnt=3
chunk_cnt=3: chunk 3（有效数据）→ ... → Writer 接收，beat_cnt=4
                                          Writer quota 满！acc2stream_ready 拉低 ↓
```

**第二阶段：后 4 个 chunk 造成死锁**

```
chunk_cnt=4: shell 试图推 chunk 4（死数据）→ out_assemble_0 满了（Writer 不接收）
             → acc2stream_0_ready_i=0
             → out_assemble_0_valid 保持高
             → oa0_in_ready = !out_assemble_0_valid = 0  ← 反压到 chunk 输入端
             → chunk_ser0_ready=0（下游满了）
             → buf0_out_ready = chunk_ser0_ready && chunk_last_0 = 0 && ... = 0
             → buf0 无法释放（stuck valid）
```

**第三阶段：反压传播到 VersaCore**

```
buf0_valid=1（永远不释放）且 buf0_out_ready=0
→ buf_can_accept = (!buf0_valid || buf0_out_ready) && ... = (0) && ... = 0
→ vc0_out_d_ready = vc1_out_d_valid && buf_can_accept = 0
→ VersaCore D 输出无法 drain
→ vc0_busy 永远为 1
→ wait_dual_versacore() 检查 busy 信号，永久等待
→ 死锁，模拟超时
```

**信号传播链路图**：

```
Writer quota 满
  └─► acc2stream_ready = 0
        └─► out_assemble_0_valid 无法清除
              └─► oa0_in_ready = 0
                    └─► chunk_ser0_ready = 0
                          └─► buf0_out_ready = 0  (chunk_last_0 虽然还在等，但此路已断)
                                └─► buf0 stuck valid
                                      └─► buf_can_accept = 0
                                            └─► vc0_out_d_ready = 0
                                                  └─► VC D output 无法 drain
                                                        └─► vc0_busy = 1 永远不清
                                                              └─► wait_dual_versacore() 永久挂死
```

### 修复方案

新增 `active_num_chunks()` 函数，根据运行时 CSR 中的 `array_shape_cfg` 返回当前 shape 实际需要的 chunk 数：

```sv
function automatic logic [$clog2(NumChunks + 1)-1:0]
    active_num_chunks(input logic [RegDataWidth-1:0] array_shape_cfg);
    case (array_shape_cfg)
        32'd0: active_num_chunks = 8;  // S0: meshRow=8, meshCol=4 → 32 elems → 8 chunks
        32'd1: active_num_chunks = 4;  // S1: meshRow=4, meshCol=4 → 16 elems → 4 chunks
        32'd2: active_num_chunks = 2;  // S2: meshRow=2, meshCol=4 → 8  elems → 2 chunks
        default: active_num_chunks = NumChunks;
    endcase
endfunction

// chunk_last 改用 active count
assign chunk_last_0 = (NumChunks <= 1) ||
                      (chunk_cnt_0 == active_num_chunks(csr_reg_set_i[4]) - 1);
```

S1 下，chunk_cnt 到达 3（= 4-1）时 chunk_last_0 触发，buf0 立即释放，VC output 正常 drain，死锁消除。

---

## Bug 2：`$clog2(NumChunks)` 位宽无法表示 NumChunks 本身

### 问题的由来

修复 Bug 1 时，agent 最初将 `active_num_chunks` 的函数返回类型写为：

```sv
// 有 bug 的初版返回类型
function automatic logic [$clog2(NumChunks)-1:0]  // $clog2(8) = 3 bits
    active_num_chunks(...);
```

`$clog2(N)` 的语义是"表示 0 到 N-1 所需的最少位数"。具体地：

```
$clog2(8) = 3   → 可表示 0..7（最大值 7）
```

但 S0 需要的 active chunk count = **8**，超出了 3 位的范围。

### 数值截断过程

```
active_num_chunks = 8 = 4'b1000
                        ↓ 截断为 3 位
                    = 3'b000 = 0
```

**S0 的 active_num_chunks 在硬件中变成了 0。**

### 为什么 0 导致死锁（SystemVerilog 运算规则）

关键在于 `chunk_last_0` 的表达式：

```sv
assign chunk_last_0 = chunk_cnt_0 == active_num_chunks(csr_reg_set_i[4]) - 1;
```

分析各信号位宽：
- `chunk_cnt_0`：`logic [$clog2(NumChunks > 1 ? NumChunks : 2)-1:0]` = `logic [2:0]`（3 bits）
- `active_num_chunks(...)` 返回：`logic [2:0]`（3 bits，已截断为 0）
- 字面量 `1`：在 SystemVerilog 中默认为 **32-bit integer**

根据 SystemVerilog LRM，混合位宽表达式中，**较窄的操作数零扩展到最宽操作数的位宽**再进行运算：

```
active_num_chunks(...) - 1
= 3'b000 - 32'd1           （3-bit 零扩展到 32-bit）
= 32'd0  - 32'd1
= 32'hFFFF_FFFF             （无符号下溢，wraps to max uint32）
```

然后比较：

```
chunk_cnt_0 == 32'hFFFF_FFFF
↓（chunk_cnt_0 零扩展到 32-bit）
32'd{0..7} == 32'hFFFF_FFFF
→ 永远为 false
```

`chunk_last_0` 在 S0 下永远为 false。结果与 Bug 1 的死锁链路完全相同：

```
chunk_last_0 永远 false
  └─► buf0_out_ready = chunk_ser0_ready && false = 0
        └─► buf0 stuck valid
              └─► VC D output 无法 drain
                    └─► vc0_busy 永远不清
                          └─► wait_dual_versacore() 永久挂死（S0 回归）
```

### 为什么 S1/S2 在此阶段没有问题

S1 的 active_num_chunks = 4 = `3'b100`，3 位可以无损表示。S2 = 2 = `3'b010`，同样没问题。只有 S0 = 8 超出了 3 位范围，触发截断。

### 修复方案

将返回类型位宽从 `$clog2(NumChunks)` 改为 `$clog2(NumChunks + 1)`：

```sv
// 修复后
function automatic logic [$clog2(NumChunks + 1)-1:0]  // $clog2(9) = 4 bits
    active_num_chunks(...);
```

数值验证：

```
$clog2(NumChunks + 1) = $clog2(9) = 4 bits → 可表示 0..15

S0: active_num_chunks = 8 = 4'b1000 ✓（无截断）
S1: active_num_chunks = 4 = 4'b0100 ✓
S2: active_num_chunks = 2 = 4'b0010 ✓
```

修复后 S0：

```
active_num_chunks(S0) = 4'b1000 = 8
8 - 1 = 7（在 32-bit 上下文中：32'd8 - 32'd1 = 32'd7）
chunk_cnt_0 == 7 → 在 count=7 时触发 ✓
```

流水线正常推进，S0 恢复 PASS。

---

## 两个 Bug 的根本原因对比

| | Bug 1 | Bug 2 |
|--|-------|-------|
| **根本原因** | chunk_last 使用硬编码 NumChunks，不感知运行时 active shape | 函数返回位宽 `$clog2(N)` 无法表示 N 本身 |
| **触发条件** | S1/S2（active chunks < NumChunks） | S0 修复后（active chunks == NumChunks = 8，恰好溢出） |
| **直接症状** | Writer quota 满后 shell 继续推死 chunk，反压传播 | active_num_chunks 截断为 0，chunk_last 永远 false |
| **最终表现** | wait_dual_versacore() 永久挂死 | 同上 |
| **死锁节点** | buf0_out_ready 因 chunk_last=0 无法触发 | buf0_out_ready 因 chunk_last 永远 false 无法触发 |
| **修复位置** | chunk_last 的比较右值，从 NumChunks-1 → active_num_chunks-1 | active_num_chunks 返回位宽，从 $clog2(N) → $clog2(N+1) |

---

## 可复用的设计规则（已写入 Skill）

1. **固定宽度硬件输出的串行化器必须使用运行时 active count，而不是物理最大值。** 即使 VC 输出总线是 1024-bit，也应该在 CSR 配置的 active shape chunk 数时停止，否则多余的 chunk 会填满下游 quota，产生不可恢复的反压死锁。

2. **表示 N 种数值（包括 N 本身）的信号，位宽必须是 `$clog2(N+1)`，而不是 `$clog2(N)`。** `$clog2(N)` 只能表示 0 到 N-1，当值等于 N 时会截断为 0，在混合位宽运算中（特别是减 1 后与 32-bit 字面量比较时）产生极难调试的 always-false 条件。

3. **RTL 死锁的诊断方法**：遇到 `wait_xxx()` 永不返回的情况，从 busy 信号出发，逆向追踪 output handshake 链路（valid/ready），找到第一个"stuck valid + ready=0"的节点，再分析为什么 ready 无法拉高。
