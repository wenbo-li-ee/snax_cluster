# Dual VersaCore SwiGLU 握手接口详细文档

> 本文档逐接口记录 `snax_dual_versacore_swiglu_shell_wrapper.sv` 中所有 valid/ready 握手信号的行为。  
> 所有接口均遵循标准 AXI-Stream / Decoupled 握手协议：  
> **当且仅当 `valid=1` 且 `ready=1` 时，本拍完成一次数据传输（称为"fire"）。**

---

## 总体数据流图

```
TCDM
 │
 ├─ stream2acc_0 (A, 1024bit) ──► [A 锁步缓冲区]
 │                                      │
 │                               ┌──────┴──────┐
 │                               ▼             ▼
 ├─ stream2acc_1 (B0, 8192bit)─► VersaCore_0   VersaCore_1 ◄─ stream2acc_2 (B1, 8192bit)
 │
 │      VersaCore_0 输出 (4096bit)──►[buf0] VersaCore_1 输出 (4096bit)──►[buf1]
 │                                      │                                  │
 │                             chunk_cnt_0 序列化                  chunk_cnt_1 序列化
 │                                      │                                  │
 │                               shifter_6stage                     shifter_2stage
 │                            (6级流水, 算术>>2)                  (2级流水, 算术>>2)
 │                                      │                                  │
 │                                      └──────────┬───────────────────────┘
 │                                                 ▼
 │                                          elem_adder_32b (64 lane)
 │                                                 │
 │                                        out_chunk_cnt 重组
 │                                                 │
 └─ acc2stream_0 (D, 4096bit) ◄─── out_assemble ──┘
```

---

## 一、输入 Streamer → 加速器接口

### 1.1 Reader 0 → A 输入（stream2acc_0）

**信号列表：**

| 方向 | 信号名 | 位宽 | 说明 |
|------|--------|------|------|
| Streamer → Shell | `stream2acc_0_data_i` | 1024 bit | A 矩阵数据（meshRow×tileSize 个 int8） |
| Streamer → Shell | `stream2acc_0_valid_i` | 1 bit | Streamer 断言：本拍数据有效 |
| Shell → Streamer | `stream2acc_0_ready_o` | 1 bit | Shell 断言：可以接收新的 A 拍 |

**握手逻辑：**

```systemverilog
assign stream2acc_0_ready_o = !a_buf_valid;
```

- `ready_o = 1`：当 A 锁步缓冲区（`a_buf_valid=0`）为空时，才接受新的 A 拍。
- `ready_o = 0`：缓冲区有效但尚未被两个 VersaCore 都消费完，暂停接收。
- **此接口不存在组合回环**：`ready` 信号仅依赖寄存器状态 `a_buf_valid`，不依赖 `valid_i`。

**时序图（正常流动）：**
```
clk:       ↑       ↑       ↑       ↑
valid_i:   0       1       1       0
ready_o:   1       1       0       0      ← 第2拍 fire，buf 被锁住
data_i:    X       D0      D1      X
a_buf:     空      D0      D0      ...    ← D0 等待两个 VC 消费完毕后清空
```

---

### 1.2 Reader 1 → B0 输入（stream2acc_1）

**信号列表：**

| 方向 | 信号名 | 位宽 | 说明 |
|------|--------|------|------|
| Streamer → Shell | `stream2acc_1_data_i` | 8192 bit | B0（W）权重数据 |
| Streamer → Shell | `stream2acc_1_valid_i` | 1 bit | 数据有效 |
| Shell → Streamer | `stream2acc_1_ready_o` | 1 bit | 可接收新权重拍 |

**握手逻辑（直通式，无额外缓冲）：**

```systemverilog
assign stream2acc_1_ready_o = vc0_in_b_ready;
assign vc0_in_b_valid       = stream2acc_1_valid_i;
```

- B0 数据直接穿透到 VersaCore_0 的 B 输入端口，中间无寄存器。
- `ready_o` 完全由 VersaCore_0 的反压决定（VersaCore 内部队列满时下降）。
- **此接口的 ready 依赖 VersaCore_0 内部状态**，可能形成组合路径（VersaCore → Shell → Streamer），需注意时序。

---

### 1.3 Reader 2 → B1 输入（stream2acc_2）

**信号列表（与 1.2 对称）：**

| 方向 | 信号名 | 位宽 | 说明 |
|------|--------|------|------|
| Streamer → Shell | `stream2acc_2_data_i` | 8192 bit | B1（V）权重数据 |
| Streamer → Shell | `stream2acc_2_valid_i` | 1 bit | 数据有效 |
| Shell → Streamer | `stream2acc_2_ready_o` | 1 bit | 可接收新权重拍 |

**握手逻辑：**

```systemverilog
assign stream2acc_2_ready_o = vc1_in_b_ready;
assign vc1_in_b_valid       = stream2acc_2_valid_i;
```

- 和 B0 完全对称，直通到 VersaCore_1 的 B 端口。

---

## 二、A 锁步缓冲区（a_buf）——防止 A 被重复消费

A 矩阵由两个 VersaCore **共享**，必须确保每拍 A 数据被 VC0 和 VC1 **各消费一次，且恰好一次**。锁步缓冲区实现此保证。

### 2.1 内部信号

| 信号 | 宽度 | 含义 |
|------|------|------|
| `a_buf_data` | 1024 bit | 锁存的 A 数据 |
| `a_buf_valid` | 1 bit | 缓冲区内有有效 A 数据 |
| `a_buf_sent_0` | 1 bit | VC0 已消费本拍 A |
| `a_buf_sent_1` | 1 bit | VC1 已消费本拍 A |
| `a_fire_0` | 1 bit | 本拍 VC0 的 A 握手成功（= vc0_in_a_valid && vc0_in_a_ready） |
| `a_fire_1` | 1 bit | 本拍 VC1 的 A 握手成功 |

### 2.2 往 VersaCore 的 A 接口

```systemverilog
assign vc0_in_a_valid = a_buf_valid && !a_buf_sent_0;
assign vc1_in_a_valid = a_buf_valid && !a_buf_sent_1;
```

- 只要缓冲区有效且本 VC 尚未消费，就持续置 valid。
- 两个 VC 的 ready 信号相互独立，允许异步消费（先到先得）。

### 2.3 缓冲区释放条件

```systemverilog
// 两路都 fire 或已标记 sent，则清空缓冲区
if ((a_buf_sent_0 || a_fire_0) && (a_buf_sent_1 || a_fire_1))
    a_buf_valid <= 1'b0;
```

- 只有 VC0 和 VC1 **都完成**了对本拍 A 的消费，`a_buf_valid` 才清零。
- 此设计避免了死锁：两个 VC 的 A ready 可不同步，先慢后快均可。

### 2.4 缓冲区加载条件

```systemverilog
if (!a_buf_valid && stream2acc_0_valid_i)  // Streamer 有新数据且缓冲区空
    a_buf_valid <= 1'b1;
    a_buf_data  <= stream2acc_0_data_i;
    a_buf_sent_0 <= 1'b0;
    a_buf_sent_1 <= 1'b0;
```

---

## 三、VersaCore 数据输入接口

两个 VersaCore 使用相同的 Chisel 生成端口，以 VC0 为例说明（VC1 对称）。

### 3.1 A 输入端口

| 方向 | 信号名 | 连接到 | 说明 |
|------|--------|--------|------|
| Shell→VC | `io_versacore_data_in_a_valid` | `vc0_in_a_valid`（来自 a_buf） | A 有效 |
| VC→Shell | `io_versacore_data_in_a_ready` | `vc0_in_a_ready` | VC 可接收 A |
| Shell→VC | `io_versacore_data_in_a_bits` | `a_buf_data`（锁存的共享 A） | A 数据 |

- VC0 和 VC1 的 `a_bits` 都连接到**同一个** `a_buf_data`，实现真正的广播共享。
- 握手独立：VC0 ready 不等待 VC1，反之亦然。

### 3.2 B 输入端口（权重，各自独立）

| 方向 | 信号名（VC0） | 连接到 | 说明 |
|------|--------------|--------|------|
| Shell→VC | `io_versacore_data_in_b_valid` | `stream2acc_1_valid_i`（直通） | B0 有效 |
| VC→Shell | `io_versacore_data_in_b_ready` | `vc0_in_b_ready` → `stream2acc_1_ready_o` | B0 反压 |
| Shell→VC | `io_versacore_data_in_b_bits` | `stream2acc_1_data_i`（直通） | B0 数据 |

B1 ↔ VC1 完全对称。

### 3.3 C 输入端口（累加器，始终为零）

```systemverilog
logic [4096-1:0] tied_c_data;
assign tied_c_data = '0;

// ...
.io_versacore_data_in_c_valid(1'b1),   // 始终有效
.io_versacore_data_in_c_bits (tied_c_data),  // 全零
```

| 信号 | 值 | 说明 |
|------|-----|------|
| `in_c_valid` | **永远 1** | C 一直"准备好"（因为是零） |
| `in_c_bits` | **全 0** | SwiGLU 不需要累加历史输出，每次从零开始 |
| `in_c_ready` | VC 内部反压 | Shell 不使用此信号 |

---

## 四、VersaCore 输出 → 双路独立缓冲区

### 4.1 联合握手逻辑（防止数据偏移）

由于 VC0 和 VC1 必须同步输出（两路输出后续要相加），需要确保 **buf0 和 buf1 在同一拍接收到对应数据**。

| 信号 | 宽度 | 说明 |
|------|------|------|
| `vc0_out_d_data` | 4096 bit | VersaCore_0 的输出 D（路径 0） |
| `vc0_out_d_valid` | 1 bit | VC0 输出有效 |
| `vc0_out_d_ready` | 1 bit | 本级可接收 VC0 输出 |
| `vc1_out_d_data` | 4096 bit | VersaCore_1 的输出 D（路径 1） |
| `vc1_out_d_valid` | 1 bit | VC1 输出有效 |
| `vc1_out_d_ready` | 1 bit | 本级可接收 VC1 输出 |
| `buf_can_accept` | 1 bit | 两个缓冲区都有空间 |
| `buf_fire` | 1 bit | 本拍两路同时 fire |

**关键赋值（联合握手）：**

```systemverilog
assign both_vc_out_valid = vc0_out_d_valid && vc1_out_d_valid;

assign buf_can_accept = (!buf0_valid || buf0_out_ready) &&
                        (!buf1_valid || buf1_out_ready);

// VC0 的 ready 需要 VC1 也有效 + 两个 buf 都有空间
assign vc0_out_d_ready = vc1_out_d_valid && buf_can_accept;
// VC1 的 ready 需要 VC0 也有效 + 两个 buf 都有空间
assign vc1_out_d_ready = vc0_out_d_valid && buf_can_accept;

assign buf_fire = both_vc_out_valid && buf_can_accept;
```

**设计要点：**
- VC0 的 ready 依赖 VC1 的 valid（反之亦然）：只有双方都准备好才同时 fire。
- 这是两路锁步同步的核心，避免 buf0/buf1 时序偏移。
- 如果只有一路 valid（另一路被反压），两路都停止，防止数据错位。

### 4.2 路径 0 缓冲区（buf0）

| 信号 | 宽度 | 说明 |
|------|------|------|
| `buf0_data` | 4096 bit | 锁存的 VC0 输出 |
| `buf0_valid` | 1 bit | buf0 有有效数据 |
| `buf0_out_ready` | 1 bit | 下游（chunk 序列化器路径 0）可以接收 |

```systemverilog
// buf0_out_ready 由下游 chunk serializer 的 ready 驱动
assign buf0_out_ready = shifter0_in_ready && chunk_last_0;
// 只有当前 chunk 是最后一个（chunk_last_0）并且 shifter 接受，才释放 buf0
```

### 4.3 路径 1 缓冲区（buf1）

与 buf0 完全对称，`buf1_out_ready = shifter1_in_ready && chunk_last_1`。

**独立缓冲的必要性：**  
- 6-stage shifter 延迟 = 6 拍，2-stage shifter 延迟 = 2 拍。
- 若只用一个共享缓冲 + 联合 ready，则需要两路 shifter 同步消费，6 vs 2 级差异会导致死锁。
- 独立缓冲 + 独立 chunk 计数 + 独立 ready，让两路各自以自己的节奏消费，避免互相等待。

---

## 五、缓冲区 → Chunk 序列化 → Shifter 输入接口

### 5.1 Chunk 序列化原理

```
ElemsPerBeat = DataWidthD / 32 = 4096 / 32 = 128 个 int32 元素
PostprocLanes = 64
NumChunks = ceil(128 / 64) = 2       ← 每个 VersaCore 输出拍需分 2 拍送入 shifter
```

每个 buf 的 4096bit 输出被切分成 2 个 chunk，每个 chunk 64×32bit = 2048bit。

### 5.2 路径 0：buf0 → shifter_6stage 输入

**信号列表：**

| 信号 | 宽度 | 说明 |
|------|------|------|
| `chunk_cnt_0` | 1 bit | 当前处理到第几个 chunk（0 或 1） |
| `chunk_last_0` | 1 bit | 是否为最后一个 chunk（= chunk_cnt_0 == NumChunks-1 = 1） |
| `shifter0_in_data` | 64×32 bit | 当前 chunk 的数据（从 buf0_data 中切片） |
| `shifter0_in_valid` | 1 bit | 数据有效（= buf0_valid） |
| `shifter0_in_ready` | 1 bit | shifter_6stage 可以接收（由 shifter 内部流水线反压决定） |

**握手逻辑：**

```systemverilog
assign shifter0_in_data[i] = buf0_data[chunk_cnt_0 * PostprocLanes*32 + i*32 +: 32];
assign shifter0_in_valid   = buf0_valid;
assign buf0_out_ready      = shifter0_in_ready && chunk_last_0;
```

- `valid` 直接由 `buf0_valid` 驱动，只要 buf 有数据就持续输出 chunk。
- `buf0_out_ready` 仅在最后一个 chunk 被 shifter 接受时才置 1，此时 buf0 被消费完毕并清空。
- **chunk 计数器更新：**
  ```systemverilog
  if (buf0_valid && shifter0_in_ready) begin
      if (chunk_last_0) chunk_cnt_0 <= '0;          // 重置，buf 被释放
      else              chunk_cnt_0 <= chunk_cnt_0 + 1;
  end
  ```

### 5.3 路径 1：buf1 → shifter_2stage 输入

与路径 0 对称：

```systemverilog
assign shifter1_in_valid = buf1_valid;
assign buf1_out_ready    = shifter1_in_ready && chunk_last_1;
```

---

## 六、Shifter 输出 → 元素加法器接口

### 6.1 shifter_6stage → elem_adder 输入端口 0

| 方向 | 信号名 | 宽度 | 说明 |
|------|--------|------|------|
| 6stage→adder | `shifter0_out_data` | 64×32 bit | 经过算术右移 >>2 的 path0 结果 |
| 6stage→adder | `shifter0_out_valid` | 1 bit | 结果有效 |
| adder→6stage | `shifter0_out_ready` | 1 bit | 加法器可以接收 path0 数据 |

**shifter_6stage 内部延迟：** 6 个寄存器级，valid 需经过 6 拍传播；第 1 拍在 stage0 对数据 >>1，第 2 拍在 stage1 再 >>1（总计 >>2），stages 2\~5 纯寄存器传递。

### 6.2 shifter_2stage → elem_adder 输入端口 1

| 方向 | 信号名 | 宽度 | 说明 |
|------|--------|------|------|
| 2stage→adder | `shifter1_out_data` | 64×32 bit | path1 右移 >>2 结果 |
| 2stage→adder | `shifter1_out_valid` | 1 bit | 结果有效 |
| adder→2stage | `shifter1_out_ready` | 1 bit | 加法器可以接收 path1 数据 |

### 6.3 加法器的联合握手

`elem_adder_32b` **不是简单的两路"各自独立消费"**，而是要求两路 **同时有效** 才能 fire：

```systemverilog
// elem_adder_32b 内部（来自 elem_adder_32b.sv）
logic both_valid;
assign both_valid = valid_i_0 && valid_i_1;
assign fire = both_valid && out_can_accept;

assign ready_o_0 = valid_i_1 && out_can_accept;   // 只有 path1 也准备好才接收 path0
assign ready_o_1 = valid_i_0 && out_can_accept;
```

**重要含义：**
- 即使 shifter0 先完成（2 拍延迟少），adder 也会等到 shifter1 的输出也 valid 后才消费。
- 反之，6-stage 先完成（因为 path1 是 2-stage 更快），adder 等快的那路等慢的。
- **等待期间数据暂存在 shifter 的最后一级输出寄存器中，反压沿流水线向上传递。**

这和 VersaCore 输出的锁步设计一致：两路始终对同一个 tile 的数据配对相加，不会错位。

---

## 七、加法器输出 → 输出重组 → Writer Streamer 接口

### 7.1 加法器 → 输出重组（out_assemble）

| 信号 | 宽度 | 说明 |
|------|------|------|
| `adder_out_data` | 64×32 bit | 加法器输出（一个 chunk 的最终结果） |
| `adder_out_valid` | 1 bit | 加法结果有效 |
| `adder_out_ready` | 1 bit | 重组寄存器可以接收（= !out_assemble_valid） |

**关键逻辑：**

```systemverilog
// 只在 out_assemble 没有待发送的数据时才接受新 chunk
assign adder_out_ready = !out_assemble_valid;
```

- 防止在前一个完整输出拍还未被 Streamer 取走时覆盖 `out_assemble`。
- 这在整个流水线末端形成背压（backpressure）：Streamer 慢则 adder 停，adder 停则 shifter 停，shifter 停则 buf 满，buf 满则 VersaCore 停。

**重组计数器 out_chunk_cnt：**

```systemverilog
if (adder_out_valid && adder_out_ready) begin
    out_assemble[out_chunk_cnt * PostprocLanes * 32 +: PostprocLanes*32] <= adder_out_data;
    if (out_chunk_last) begin
        out_chunk_cnt <= '0;
        out_assemble_valid <= 1'b1;   // 完整的 4096bit 输出拍准备好
    end else begin
        out_chunk_cnt <= out_chunk_cnt + 1;
        out_assemble_valid <= 1'b0;
    end
end
```

- 两个 chunk 写完后，`out_assemble_valid` 才置 1，向 Streamer 发出一个完整的 4096bit 输出拍。

### 7.2 Shell → Writer Streamer（acc2stream_0）

| 方向 | 信号名 | 宽度 | 说明 |
|------|--------|------|------|
| Shell→Streamer | `acc2stream_0_data_o` | 4096 bit | 完整的输出拍（128 × int32） |
| Shell→Streamer | `acc2stream_0_valid_o` | 1 bit | 输出有效（= out_assemble_valid） |
| Streamer→Shell | `acc2stream_0_ready_i` | 1 bit | Streamer 可以写入 TCDM |

```systemverilog
assign acc2stream_0_data_o  = out_assemble;
assign acc2stream_0_valid_o = out_assemble_valid;

// out_assemble 清空条件
else if (out_assemble_valid && acc2stream_0_ready_i) begin
    out_assemble_valid <= 1'b0;
end
```

- 完整 4096bit 输出拍送出后，`out_assemble_valid` 清零，允许下一轮重组开始。

---

## 八、CSR 控制接口握手

### 8.1 写 CSR（配置 VersaCore 启动）

| 方向 | 信号名 | 宽度 | 说明 |
|------|--------|------|------|
| CSR Manager→Shell | `csr_reg_set_i` | 7×32 bit | 7 个 RW CSR 值（配置参数） |
| CSR Manager→Shell | `csr_reg_set_valid_i` | 1 bit | CSR 值有效，请求启动 |
| Shell→CSR Manager | `csr_reg_set_ready_o` | 1 bit | 两个 VC 都准备好接受配置 |

```systemverilog
assign csr_reg_set_ready_o = vc0_ctrl_ready && vc1_ctrl_ready;
assign ctrl_valid_to_vc    = csr_reg_set_valid_i;
```

- **同步要求：两个 VersaCore 必须都 ready** 才说明 Shell 准备好（避免只有一个 VC 接收了配置）。
- 配置被广播到两个 VC 的 `io_ctrl_*` 端口，两者共用同一份 CSR。

### 8.2 只读 CSR（状态查询）

| 信号 | 值 | 含义 |
|------|-----|------|
| `csr_reg_ro_set_o[0]` | `{31'b0, vc0_busy \|\| vc1_busy}` | bit0 = 任一 VC 仍在计算 |
| `csr_reg_ro_set_o[1]` | `max(vc0_perf, vc1_perf)` | 取两路性能计数器中的最大值 |

软件通过轮询 `DUAL_VC_BUSY_CSR` 的 bit0，等待 `0` 后再轮询 `STREAMER_BUSY_CSR`，确保所有数据已写入 TCDM。

---

## 九、完整握手信号一览表

| 接口位置 | valid 信号 | ready 信号 | 数据信号 | 宽度 |
|----------|-----------|-----------|----------|------|
| Reader0(A)→Shell | `stream2acc_0_valid_i` | `stream2acc_0_ready_o` | `stream2acc_0_data_i` | 1024b |
| Reader1(B0)→Shell | `stream2acc_1_valid_i` | `stream2acc_1_ready_o` | `stream2acc_1_data_i` | 8192b |
| Reader2(B1)→Shell | `stream2acc_2_valid_i` | `stream2acc_2_ready_o` | `stream2acc_2_data_i` | 8192b |
| a_buf→VC0(A) | `vc0_in_a_valid` | `vc0_in_a_ready` | `a_buf_data` | 1024b |
| a_buf→VC1(A) | `vc1_in_a_valid` | `vc1_in_a_ready` | `a_buf_data`（共享） | 1024b |
| B0直通→VC0(B) | `vc0_in_b_valid` | `vc0_in_b_ready` | `stream2acc_1_data_i` | 8192b |
| B1直通→VC1(B) | `vc1_in_b_valid` | `vc1_in_b_ready` | `stream2acc_2_data_i` | 8192b |
| VC0输出→buf0 | `vc0_out_d_valid` | `vc0_out_d_ready` | `vc0_out_d_data` | 4096b |
| VC1输出→buf1 | `vc1_out_d_valid` | `vc1_out_d_ready` | `vc1_out_d_data` | 4096b |
| buf0→shifter6 | `shifter0_in_valid` | `shifter0_in_ready` | `shifter0_in_data` | 64×32b |
| buf1→shifter2 | `shifter1_in_valid` | `shifter1_in_ready` | `shifter1_in_data` | 64×32b |
| shifter6→adder(0) | `shifter0_out_valid` | `shifter0_out_ready` | `shifter0_out_data` | 64×32b |
| shifter2→adder(1) | `shifter1_out_valid` | `shifter1_out_ready` | `shifter1_out_data` | 64×32b |
| adder→重组寄存器 | `adder_out_valid` | `adder_out_ready` | `adder_out_data` | 64×32b |
| Shell→Writer(D) | `acc2stream_0_valid_o` | `acc2stream_0_ready_i` | `acc2stream_0_data_o` | 4096b |
| CSR→Shell(配置) | `csr_reg_set_valid_i` | `csr_reg_set_ready_o` | `csr_reg_set_i` | 7×32b |

---

## 十、背压传播路径（Backpressure Chain）

当 Writer Streamer 暂时无法写入 TCDM（`acc2stream_0_ready_i = 0`）时，背压按如下路径向前传播：

```
Streamer 拉低 acc2stream_0_ready_i (= 0)
    → out_assemble_valid 保持 1
    → adder_out_ready = !out_assemble_valid = 0
    → elem_adder_32b 停止接收（both ready 置 0）
    → shifter0_out_ready = 0, shifter1_out_ready = 0
        ├→ shifter_6stage 最后一级反压向前传播（6 级寄存器链逐拍变满）
        └→ shifter_2stage 最后一级反压（2 级）
    → shifter0_in_ready = 0, shifter1_in_ready = 0
    → buf0_out_ready = 0, buf1_out_ready = 0
    → buf0/buf1 保持 valid，不消费 VersaCore 输出
    → vc0_out_d_ready = 0, vc1_out_d_ready = 0（通过 buf_can_accept）
    → VersaCore 输出端反压（VersaCore 内部累加器停止输出）
    → VersaCore 输入端停止接收 A/B（VersaCore 计算暂停）
    → stream2acc_1/2_ready_o 降低（Streamer B 停止读 TCDM）
    → stream2acc_0_ready_o 降低（Streamer A 停止读 TCDM）
```

整条流水线实现**完全端到端背压**，保证不丢数据、不溢出。
