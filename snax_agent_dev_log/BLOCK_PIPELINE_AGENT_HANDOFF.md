# Agent Handoff: Dual VersaCore SwiGLU — Block-Level Pipeline Implementation

## 0. 快速上手

**容器**：`barnard3`（podman，已在运行）  
**工作目录**：`/esat/studscratch/r1015498/Thesis/original_snax/snax_cluster`  
**当前Git分支**：`swiglue`（已push到 `origin/swiglue`）  
**构建命令前缀**：
```bash
podman exec barnard3 bash -lc 'source /pixi/entrypoint.sh; cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster/target/snitch_cluster; <cmd>'
```

**本地参考文档**（读这些，不要去网上找）：
- `/esat/studscratch/r1015498/Thesis/original_snax/DUAL_VERSACORE_CFG_GUIDE_ZH.md` — cfg参数详解
- `/esat/studscratch/r1015498/Thesis/original_snax/DUAL_VERSACORE_HANDSHAKE_DOC_ZH.md` — 握手信号详解
- `/esat/studscratch/r1015498/Thesis/original_snax/DUAL_VERSACORE_SWIGLU_DEVLOG.md` — 开发日志

---

## 1. 当前加速器状态（已完成，仿真PASS）

### 架构

```
TCDM
 ├─ Reader0 (A,  1024bit) ─► [A锁步缓冲] ─►┬─► VersaCore_0 ◄─ Reader1 (B0, 8192bit)
 │                                          └─► VersaCore_1 ◄─ Reader2 (B1, 8192bit)
 │
 │     VC0输出(4096bit)──►[buf0]──chunk_cnt──► shifter_6stage (>>2)──►┐
 │     VC1输出(4096bit)──►[buf1]──chunk_cnt──► shifter_2stage (>>2)──►┤
 │                                                                     ▼
 │                                                          elem_adder_32b (64 lane)
 │                                                                     │
 └─ Writer0 (D, 4096bit) ◄──────────────── [out_assemble 重组4096bit]─┘
```

**黄金模型**：`D = (A @ W >> 2) + (A @ V >> 2)`（算术右移，int8→int32）

### 关键硬件参数
| 参数 | 值 | 含义 |
|------|----|------|
| meshRow | 16 | 单个VersaCore，输出行并行 |
| tileSize | 8 | K维度（内积）并行 |
| meshCol | 8 | 单个VersaCore，输出列并行 |
| DataWidthA | 1024 bit | 16×8×8 bit |
| DataWidthB | 8192 bit | 权重总线 |
| DataWidthD | 4096 bit | 16×8×32 bit |
| PostprocLanes | 64 | 后处理并行lane数 |
| NumChunks | 2 | 每VC输出拍分2拍送shifter |

### 当前测试参数（params.hjson）
```hjson
K: 2    // K维度tile数（实际K = 2×tileSize = 16列）
N: 1    // N维度tile数（实际N = 1×meshCol = 8列）
M: 20   // M维度tile数（实际M = 20×meshRow = 320行）
```
**仿真结果**：PASS, Error=0, 加速器84周期, Streamer 95周期

### 关键文件
```
hw/chisel_acc/src/main/scala/snax_acc/versacore/DualVersaCoreSwigluGen.scala
hw/chisel_acc/src/main/resources/snax_acc/versacore/shifter_6stage.sv
hw/chisel_acc/src/main/resources/snax_acc/versacore/shifter_2stage.sv
hw/chisel_acc/src/main/resources/snax_acc/versacore/elem_adder_32b.sv
target/snitch_cluster/cfg/snax_dual_versacore_swiglu_cluster.hjson
target/snitch_cluster/sw/snax/dual-versacore-swiglu/include/snax-dual-versacore-swiglu-lib.h
target/snitch_cluster/sw/snax/dual-versacore-swiglu/src/snax-dual-versacore-swiglu-lib.c
target/snitch_cluster/sw/apps/snax-dual-versacore-swiglu-test/src/snax-dual-versacore-swiglu-test.c
target/snitch_cluster/sw/apps/snax-dual-versacore-swiglu-test/data/datagen.py
target/snitch_cluster/sw/apps/snax-dual-versacore-swiglu-test/data/params.hjson
```

### CSR地址映射（已生成，勿手改）
```
target/snitch_cluster/sw/snax/dual-versacore-swiglu/include/streamer_csr_addr_map.h
// Reader0(A): 960~976, Reader1(B0): 977~990, Reader2(B1): 991~1004
// Writer0(D): 1005~1018
// STREAMER_START_CSR = 1019
// STREAMER_BUSY_CSR  = 1020  ← 当前：Reader0|Reader1|Reader2|Writer0 的OR
// STREAMER_PERF_CSR  = 1021
// DUAL_VC_CSR_BASE   = 1022
// DUAL_VC_BUSY       = 1029
```

### 当前SW调用流程
```c
// 1. all在一次: 配置所有streamer CSR一次性启动
set_dual_versacore_streamer_csr(...all params...);
// 2. 配置VC: take_in_new_c=1, K轮次, M*N输出次数
set_dual_versacore_csr(1, K, N * M, subtraction, array_shape, data_type);
// 3. 启动
set_dual_versacore_streamer_start();   // STREAMER_START_CSR = 1
set_dual_versacore_start();            // DUAL_VC_START = 1
// 4. 等计算完
wait_dual_versacore();                 // poll DUAL_VC_BUSY == 0
// 5. 等写回完
wait_dual_versacore_streamer();        // poll STREAMER_BUSY_CSR == 0
// 6. 检查结果
check_dual_versacore_result(...)
```

---

## 2. 需要实现的功能：Block-Level Pipeline

### 背景与动机

实际场景中，完整矩阵可能比TCDM（256KB）大，需要分多次DMA搬运（block-by-block）并多次重启读Streamer和VersaCore。

但是，输出D的大小通常放得下（结果是int32，比输入int8大4倍，但MN通常小），因此写Streamer（Writer0）可以**一直跑不重启**，地址自动递增。

**Block-Level Pipeline目标**：在 Writer0 还在写上一块结果到TCDM的时候，立刻重新配置读Streamer（Reader0/1/2）和VersaCore、启动下一块的计算，而不是等Writer0写完再启动。

### 当前为什么无法做到

问题在 `hw/chisel/src/main/scala/snax/streamer/Streamer.scala`，三个耦合点：

```scala
// 1. BUSY时CSR写不进去（FSM: IDLE→BUSY，BUSY期间不接受新配置）
csrManager.io.readWriteRegIO.ready := streamer_ready   // 只有IDLE才接受

// 2. Reader和Writer绑定同一个start信号
reader(i).io.start := streamer_config_fire    // 全部同时start
writer(j).io.start := streamer_config_fire    // 无法独立启动

// 3. STREAMER_BUSY是所有reader+writer的OR，无法区分"只有writer在跑"
streamer_finish := !(reader(0).busy || reader(1).busy || reader(2).busy || writer(0).busy)
```

### 推荐的最小改动方案（只改Streamer.scala + hjson）

**目标**：当所有Reader都完成时（即使Writer还在跑），允许：
1. 写入新的Streamer CSR配置（新的base address、bound等）
2. 只重启Readers，不重启/不打断Writer0
3. 通过新增的 `WRITER_BUSY_CSR`（RO CSR）让SW单独等写回完成

#### 改动1：计算readers_all_done信号（新增~5行）

```scala
// 在 streamer_finish 定义之后新增：
val readers_all_done = Wire(Bool())
readers_all_done := !reader
  .map(_.io.busy)
  .reduceLeftOption(_ || _)
  .getOrElse(false.B)
dontTouch(readers_all_done)
```

#### 改动2：CSR接受条件放宽（改1行）

```scala
// 原来：
csrManager.io.readWriteRegIO.ready := streamer_ready
// 改成（readers全完成 或 真正IDLE 都接受新配置）：
csrManager.io.readWriteRegIO.ready := streamer_ready || readers_all_done
```

**注意**：config_fire发生后，csrCfgReg要正确存储（已有when(streamer_config_fire)，不需要额外改）。

#### 改动3：Writer的start信号解耦（改~3行）

```scala
// Reader start: 任何config_fire都重启（原来不变）
reader(i).io.start := streamer_config_fire

// Writer start: 只在从IDLE出发的config_fire才启动（新逻辑）
// 当streamer_ready=true（IDLE状态）时的config_fire才触发writer start
// 当readers_all_done&&!streamer_ready（只有writer在跑）时的config_fire不触发writer start
val writer_start_first = streamer_config_fire && streamer_ready
writer(j).io.start := writer_start_first
```

#### 改动4：新增Writer-only busy RO CSR

在 `csrManager.io.readOnlyReg(2)` 连接writer(0).io.busy（当前只有readOnlyReg(0)=streamer_busy, readOnlyReg(1)=perf_counter）。

**同步修改**：
- `target/snitch_cluster/cfg/snax_dual_versacore_swiglu_cluster.hjson` 中：
  ```hjson
  snax_num_ro_csr: 3   // 原来是2，新增一个writer_busy
  ```

- 重新运行 `make rtl-gen` 会自动更新 `streamer_csr_addr_map.h`，新CSR地址 = 1022+1+（RW CSR数）... 要注意地址偏移，以下面实际生成为准。

#### 改动5：SW库新增函数

在 `snax-dual-versacore-swiglu-lib.h/c` 中新增：
```c
// 等Writer完成（只等写回）
void wait_dual_versacore_writer();

// 只重新配置并启动Readers（Writer继续运行）
// 参数：新的A/B0/B1 base地址和bound（相对偏移）
void restart_dual_versacore_readers(
    int32_t delta_a, int32_t* Atlbound, int32_t* Atlstride,
    int32_t delta_b0, int32_t* B0tlbound, int32_t* B0tlstride,
    int32_t delta_b1, int32_t* B1tlbound, int32_t* B1tlstride);
```

---

## 3. SW验证设计：M=20分两块，手动模拟Block Pipeline

### 验证场景

当前 M=20, K=2, N=1（一次性跑完）。  
为了验证Block Pipeline，**不改datagen.py**（黄金数据保持完整M=20），但在test.c里手动分两次：

- **Block 0**：M_tile 0~9（前10个meshRow=160行）
- **Block 1**：M_tile 10~19（后10个meshRow=160行）

Writer一直不重启，地址自动递增写完整个D。

### 具体实现思路（在test.c里）

**Block 0 启动：**
```c
// 配置Reader0(A): base=delta_local_a, M_bound=10（只算前10个M tile）
// 配置Reader1(B0): base=delta_local_b0, M_bound=10, B0_stride_M=0（weight stationary）
// 配置Reader2(B1): base=delta_local_b1, M_bound=10
// 配置Writer0(D): base=delta_local_d, M_bound=10（或M=20，让它自动走完）
// 配置VC: output_times = N*10（只算10个M tile）
set_dual_versacore_streamer_csr(...block0_params...);
set_dual_versacore_csr(1, K, N * 10, ...);
set_dual_versacore_streamer_start();
set_dual_versacore_start();
```

**等VC计算完（此时Writer0可能还在搬最后几拍数据）：**
```c
wait_dual_versacore();   // DUAL_VC_BUSY==0
```

**立刻启动Block 1（不等Writer！）：**
```c
// A的起始地址偏移：跳过前10个M tile的A数据
int32_t a_block1_offset = delta_local_a + 10 * K * meshRow * tileSize * 1; // int8
// 配置Reader0(A): base=a_block1_offset, M_bound=10
// B不变（weight stationary，B的M-stride=0）
// Writer不重新配，继续自动走
// 只配readers和VC CSR，然后只start readers（新函数）
restart_dual_versacore_readers(...block1_A_params...);
set_dual_versacore_csr(1, K, N * 10, ...);
set_dual_versacore_start();
```

**等Block 1的Writer写完：**
```c
wait_dual_versacore();
wait_dual_versacore_writer();   // 等Writer0写完Block 1对应输出
```

**检查完整结果：**
```c
check_dual_versacore_result((int8_t*)local_d, (int8_t*)D, d_data_length);
```

### 成功条件

- 仿真输出：`Dual VersaCore SwiGLU: PASS, Error: 0.`（所有M=20的结果正确）
- 打印两次的Block启动和完成信息
- Block 1的readers启动时，Writer0对应Block 0的写回尚未完成（可在仿真trace里验证"pipeline overlap"）

---

## 4. 完整TODO List

### Phase 1：硬件改动（Streamer.scala + cfg）

- [ ] **1.1** 阅读 `hw/chisel/src/main/scala/snax/streamer/Streamer.scala`（约400行），理解FSM和start/busy信号
- [ ] **1.2** 在 `Streamer.scala` 中添加 `readers_all_done`信号
- [ ] **1.3** 修改 `csrManager.io.readWriteRegIO.ready` 逻辑（加OR条件）
- [ ] **1.4** 修改 `writer(j).io.start` 只在 `streamer_ready` 时触发
- [ ] **1.5** 在 `csrManager.io.readOnlyReg(2)` 连接 `writer(0).io.busy`（需要确认Streamer的readOnlyReg数组够用）
- [ ] **1.6** 修改 `snax_dual_versacore_swiglu_cluster.hjson` 中 `snax_num_ro_csr: 3`
- [ ] **1.7** 运行 `make CFG_OVERRIDE=cfg/snax_dual_versacore_swiglu_cluster.hjson rtl-gen`，检查：
  - 生成的 `streamer_csr_addr_map.h` 中有新的CSR地址
  - 生成的 `snax_dual_versacore_swiglu_csrman_wrapper.sv` 中 RegROCount=3
- [ ] **1.8** 运行 `make ... bin/snitch_cluster.vlt` 重新编译仿真器（**耗时约12分钟**）

### Phase 2：SW库改动

- [ ] **2.1** 在 `snax-dual-versacore-swiglu-lib.h/c` 中添加 `wait_dual_versacore_writer()`，读新的WRITER_BUSY CSR地址（从新生成的streamer_csr_addr_map.h中找）
- [ ] **2.2** 添加 `restart_dual_versacore_readers(...)` 函数：只配置Reader0/1/2的CSR，然后写 `STREAMER_START_CSR=1`（此时Writer0继续，不会被打断）
- [ ] **2.3** 运行 `make ... sw`，确认编译无错

### Phase 3：Test App改动（test.c）

- [ ] **3.1** 修改 `snax-dual-versacore-swiglu-test.c`，实现M=20的两块pipeline（Block 0: M_tile 0~9，Block 1: M_tile 10~19）
- [ ] **3.2** 在Block 0的VC完成后，立即配置并启动Block 1（不等Writer！）
- [ ] **3.3** 在Block 1完成后才等Writer全部写完
- [ ] **3.4** 最后用完整的 `D[]` golden（M=20）检查结果
- [ ] **3.5** 打印每个block的启动/完成信息

### Phase 4：验证

- [ ] **4.1** `make ... sw` 重新build
- [ ] **4.2** 运行仿真：`bin/snitch_cluster.vlt ...elf`
- [ ] **4.3** 检查输出：PASS, Error=0
- [ ] **4.4** （可选）确认pipeline确实有overlap：Block 1启动时Block 0的Writer还在running

### Phase 5：收尾

- [ ] **5.1** git add + commit（信息自拟）+ push到 `origin/swiglue`

---

## 5. 注意事项 / 常见陷阱

1. **make clean 用法**：每次改了Streamer.scala后必须 `make clean-vlt clean-generated`，再 `make rtl-gen` 再 `make bin/snitch_cluster.vlt`。

2. **streamer_csr_addr_map.h 是生成文件**：地址会因为 `snax_num_ro_csr` 改变而整体偏移。修改后要检查 `DUAL_VC_CSR_ADDR_BASE`、`DUAL_VC_BUSY` 等地址是否还正确。新的WRITER_BUSY CSR地址在文件末尾找 `READ_ONLY_CSR_` 或类似命名。

3. **Writer start 解耦的关键问题**：`streamer_config_fire` 发生时，如果处于 `readers_all_done && !streamer_ready` 状态，此时 `csrCfgReg` 会被新值覆盖——但这是OK的，因为Writer的CFG（base address等）已经完成，它的地址生成器在跑，不依赖CSR寄存器（已经latched到自己的内部寄存器了）。需要验证这一点：Reader.scala和Writer.scala中start触发后CFG是否从config_fire时的latched寄存器读，而不是从csrCfg wire读。

4. **Block 1的B0/B1地址**：如果B是weight stationary（M方向stride=0），那么B的base address不需要变，B0tlbound/stride和Block 0一样。

5. **datagen.py 黄金数据**：一次性生成M=20的完整golden，两块分开计算后的结果合在一起应该等于这个golden。datagen.py不需要修改。

6. **wait_dual_versacore_writer 的CSR地址**：修改hjson后重新rtl-gen，从新生成的 `streamer_csr_addr_map.h` 找 `STREAMER_BUSY_CSR` 之后的新地址（或者在lib.h里新定义 `WRITER_BUSY_CSR = DUAL_VC_PERFORMANCE_COUNTER + 1` 之类，依实际生成为准）。

7. **仿真器不需要重新编译**（如果只改SW的话）：只改了.c / .h文件，`make sw` 就够了，不用重跑12分钟的 `make bin/snitch_cluster.vlt`。但如果改了Streamer.scala或cfg，就必须重编。
