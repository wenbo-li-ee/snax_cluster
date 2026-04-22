# SwiGLU SiLU 硬件集成用户指南

**日期：** 2026-04-21  
**版本：** 1.0  
**适用项目：** `original_snax/snax_cluster`，分支 `swiglue`

---

## 1. 概述

本指南说明如何在 SNAX 双 VersaCore SwiGLU 加速器中使用真实 SiLU 硬件模块。  
SwiGLU 计算公式为 `SwiGLU(x, W, V) = (x @ W) * SiLU(x @ V)`，其中 SiLU(x) = x · σ(x)。

硬件实现采用分段多项式近似（Q16.11 定点，3 级流水线），精度与浮点 SiLU 基本一致。

---

## 2. 硬件架构：Mode 0 SwiGLU 数据流

```
Reader0(A) ──┬──────────────────────────────────────────────────┐
             │                                                  │
        VersaCore0 ← Reader1(B0/W)       VersaCore1 ← Reader2(B1/V)
             │ int32                              │ int32
        rescale_down_32to16                 rescale_down_32to16
             │ int16                              │ int16
        silu_multilane                      (rescale1_out)
             │ int16  SiLU(x@V)                  │ int16  x@W
             └─────────── elem_mul_16b ───────────┘
                               │ int32
                      rescale_down_32to16 (rescale_mul)
                               │ int16
                Writer0(D0) ── 同时 ── Writer1(D1_mode0 dummy)
```

- **Path 0**：VersaCore0 计算 `A @ W`，rescale → SiLU → elem_mul 输入 0
- **Path 1**：VersaCore1 计算 `A @ V`，rescale → elem_mul 输入 1
- **elem_mul**：逐元素相乘，输出 int32
- **rescale_mul**：int32 → int16 重新量化
- **Writer0**：写入最终 SwiGLU 输出 D0

---

## 3. SiLU 模块参数

| 参数 | 值 | 说明 |
|------|----|------|
| 定点格式 | Q16.11 | 输入输出均为 int16，小数点位于第 11 位 |
| 流水线深度 | 3 级 | partition_detector → horner_stage0 → horner_stage1 |
| 并行通道数 | 32 (PostprocLanes) | 每时钟周期处理 32 个 int16 元素 |
| 分段数 | 见 `silu_out16_balanced_pkg.sv` | 分段多项式近似 |

---

## 4. CSR 地址映射（软件端）

CSR 地址由 `streamer_csr_addr_map.h` 和 `snax-dual-versacore-swiglu-lib.h` 定义。  
**每次重新生成 RTL（`make rtl-gen`）后，地址可能改变，必须重新编译库和 ELF。**

关键 CSR（当前值，基于 STREAMER_WRITER1_BUSY_CSR=1033）：

| 名称 | CSR 地址 | 说明 |
|------|---------|------|
| DUAL_VC_MODE | 1040 | 0=SwiGLU, 1=GEMM |
| DUAL_VC_START | 1053 | 写 1 启动，写 0 无操作（不重新触发）|
| DUAL_VC_BUSY | 1054 | 读 1 表示 VersaCore 仍在计算 |
| STREAMER_START_CSR | 1029 | Streamer 启动 |
| STREAMER_BUSY_CSR | 1030 | Streamer 忙状态 |

---

## 5. 软件使用流程

### 5.1 头文件包含

```c
#include "snax-dual-versacore-swiglu-lib.h"
```

### 5.2 Mode 0 (SwiGLU) 完整调用流程

```c
// 1. 设置 Streamer CSR（输入/输出地址、步幅、边界）
set_dual_versacore_streamer_csr(
    delta_local_a,  Aslstride, Atlbound, Atlstride, ...,
    delta_local_b0, B0slstride, B0tlbound, B0tlstride, ...,
    delta_local_b1, B1slstride, B1tlbound, B1tlstride, ...,
    delta_local_d0, D0slstride, D0tlbound, D0tlstride, ...,
    delta_local_d1_mode0, D1slstride, D1tlbound, D1tlstride, ...);

// 2. 设置 VersaCore 参数
set_dual_versacore_csr(
    1,           // take_in_new_c（清除累加器）
    K,           // a_b_input_times_one_output（每个输出块的输入次数）
    N * M,       // output_times（输出块总数）
    subtraction, // 减法配置（通常为 0）
    array_shape, // 矩阵布局（2 = int16x4）
    data_type    // 0 = int16
);

// 3. 设置模式为 SwiGLU
set_dual_versacore_mode(0);

// 4. 配置重量化参数（shift=0, mult=1 = 恒等变换）
set_dual_versacore_rescale0(0, 1, 0, 0);   // VC0 路径：int32 → int16
set_dual_versacore_rescale1(0, 1, 0, 0);   // VC1 路径：int32 → int16
set_dual_versacore_rescale_mul(0, 1, 0, 0); // elem_mul 后：int32 → int16

// 5. 启动 Streamer，然后启动 VersaCore
set_dual_versacore_streamer_start();
set_dual_versacore_start();

// 6. 等待完成
wait_dual_versacore();         // 等待 VersaCore BUSY=0
wait_dual_versacore_writer();  // 等待 Writer0 写完
```

### 5.3 重量化参数说明

rescale_down_32to16 模块将 int32 量化为 int16：

```
output = clamp((input - input_zp) * multiplier >> shift + output_zp, -32768, 32767)
```

当 `shift=0, multiplier=1, input_zp=0, output_zp=0` 时为恒等截断（保留低 16 位并钳位）。

**注意**：SiLU 模块的输入是 Q16.11 格式的 int16，rescale 的参数必须与实际数值范围匹配，否则 SiLU 输出错误（值域超出 SiLU 的线性区间）。

---

## 6. 内存布局（以 m1-batch 为例）

| 缓冲区 | 起始偏移 | 大小 | 说明 |
|--------|---------|------|------|
| A | 0 | 256 B | 输入矩阵 A（int16）|
| B0 (W) | 256 | 4096 B | 权重矩阵 W（int8x4，b_tile_padded=64B）|
| B1 (V) | 4352 | 4096 B | 权重矩阵 V（int8x4，b_tile_padded=64B）|
| D0 | 8448 | 512 B | SwiGLU 输出（int16）|
| D1_mode0 | 8960 | 512 B | Mode 0 辅助写出（与 D0 相同数据）|

**B tile padding 规则**：B tile 字节数 = max(raw_b_tile, b_channel_footprint)，防止 TCDM 死锁。

---

## 7. 重建流程（修改 RTL 后）

每次修改 RTL 并重新生成后，必须按顺序执行：

```bash
cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster

# 1. RTL 生成（重新生成 shell wrapper 和 streamer_csr_addr_map.h）
make rtl-gen CFG_FILE=target/snitch_cluster/cfg/snax_dual_versacore_int16x4_cluster.hjson

# 2. 重建共享库（CSR 地址发生变化）
~/.pixi/bin/pixi run -- bash -c \
  "cd target/snitch_cluster/sw/snax/dual-versacore-swiglu && make clean && make"

# 3. 重建所有 4 个 app ELF
for app in snax-versacore-int16x4-swiglu-m1-batch snax-versacore-int16x4-swiglu-m1-pingpong \
           snax-versacore-int16x4-swiglu-m4-batch snax-versacore-int16x4-swiglu-m4-pingpong; do
  rm -rf target/snitch_cluster/sw/apps/$app/build
  ~/.pixi/bin/pixi run -- bash -c \
    "cd target/snitch_cluster/sw/apps/$app && make CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_cluster.hjson"
done

# 4. 重建仿真器（修改了 RTL 文件时）
make -C target/snitch_cluster bin/snitch_cluster.vlt

# 5. 运行仿真
PIXI_LIB=".pixi/envs/default/lib"
LD_LIBRARY_PATH="$PIXI_LIB:$LD_LIBRARY_PATH" \
  target/snitch_cluster/bin/snitch_cluster.vlt \
  target/snitch_cluster/sw/apps/snax-versacore-int16x4-swiglu-m1-batch/build/*.elf
```

---

## 8. 常见问题

### Q: 仿真输出 `accel=0, streamer=0, wall=2863`，VersaCore 没有启动

**原因**：共享库 `.o` 文件没有随 `streamer_csr_addr_map.h` 的更新而重新编译。  
**解决**：执行上面第 2 步（make clean && make 共享库）。

### Q: 运行仿真器报 GLIBCXX 版本错误

**解决**：添加 pixi 库路径：
```bash
LD_LIBRARY_PATH="/path/to/snax_cluster/.pixi/envs/default/lib:$LD_LIBRARY_PATH"
```

### Q: Mode 0 输出数值错误（与 golden 不符）

检查：
1. `set_dual_versacore_mode(0)` 是否在 `set_dual_versacore_start()` 之前调用
2. rescale 参数是否与数据动态范围匹配（shift=0 时假设输入已在 int16 范围内）
3. B tile padding 是否正确（`b_tile_padded = max(raw, channel_footprint)`）

### Q: Mode 1 (GEMM) 结果也出错

Mode 1 的 A 输入 = Mode 0 的 D0 输出地址。若 Mode 0 未正确写入，Mode 1 用的是错误输入。  
先确保 Mode 0 PASS 再测试 Mode 1。

---

## 9. SiLU 精度说明

硬件 SiLU 采用分段二次多项式近似 SiLU(x) = x * sigmoid(x)，基于 Q16.11 定点格式。

Python 验证模型位于：
```
/esat/studscratch/r1015498/Thesis/original_snax/silu/pkg/silu_out16_balanced_golden.py
```

在 datagen.py 中，golden 计算使用：
```python
from silu_out16_balanced_golden import silu_out16_balanced_eval_q

# 将 int16 转换为 Q16.11 浮点，计算 SiLU，再转回 int16
silu_out = silu_out16_balanced_eval_q(vc0_int16)
```

---

## 10. 文件清单

| 文件 | 位置 | 说明 |
|------|------|------|
| silu_multilane.sv | silu/rtl/ 和 generated/snax_dual_versacore_swiglu/ | 多通道 SiLU 包装器（CE 背压）|
| silu_top.sv | silu/rtl/ | 单通道 SiLU（3 级流水线，带 CE 输入）|
| partition_detector.sv | silu/rtl/ | 分区检测，带 CE |
| horner_stage.sv | silu/rtl/ | Horner 求值级，带 CE |
| silu_out16_balanced_pkg.sv | silu/pkg/ | 参数包（Q16.11 格式）|
| snax_dual_versacore_swiglu_shell_wrapper.sv | generated/snax_dual_versacore_swiglu/ | 生成的 SwiGLU 外壳（包含 silu_multilane 实例）|
| snax-dual-versacore-swiglu-lib.c/h | sw/snax/dual-versacore-swiglu/ | SW 驱动库 |
| streamer_csr_addr_map.h | sw/snax/dual-versacore-swiglu/include/ | 自动生成的 CSR 地址表 |
