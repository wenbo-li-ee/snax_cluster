# Review: S4 Shape Sweep 修复与直接读取完整验证

**Review 日期**: 2026-04-27  
**Dev Log 路径**: `snax_agent_dev_log/s4_shape_sweep_fix_20260427/`  
**对应 Todo**: `review_log/s4_direct_mode1_review_20260427/todo_s4_shape_sweep_fix.md`

---

## 1. 总体评价

**优秀**。Agent 成功完成了本次任务的所有核心目标：找到并修复了 S1/S2 的硬件 bug（Chisel generator 中 chunk 串行化计数错误），添加了 Mode0 D1 正确性检查，完成了三形状完整验证扫描，并在引入回归后及时自我发现并修复。唯一遗留的是 Route B（`delta_local_a1` 清理），属于有意 defer，且理由充分。

---

## 2. 已完成内容

### Phase 0 - S0 Baseline 验证
- 确认 S0 direct-read 在任务开始时仍然 PASS。
- 记录初始周期：M0 `wall=2859`, M1 `wall=2718`。

### Phase 1 - S1/S2 超时根因诊断与修复

**诊断路径执行规范**，完全按 escalation 顺序进行：

1. 静态检查 `data.h`：无地址重叠、无零 bound、bank 分配正确 → 无问题。
2. 延长超时至 600s 运行：仍然挂死 → 不是 timeout 太短。
3. 添加临时 app 进度打印：定位到 `wait_dual_versacore()` 处挂死，accelerator busy 从未清除。
4. 添加临时 RTL probe：确认 VC 输出和 postproc 都在 fire，但 shell 在写出 32 拍（4 tiles × 8 chunks/tile）之后，`acc_ready=00`，writer quota 已耗尽，还有一个 output buffer 保持 valid 无法 drain——**shell 在对每个 tile 串行化 8 个 chunk（来自 1024-bit 固定宽度 DataWidthD），但 S1 实际只有 4 个 active chunk，datagen 的 writer bound 是正确的 4**。这导致 4 个死 chunk 打满了 writer 的 beat quota，产生 backpressure，VC 输出被阻塞，accelerator 永不完成。

**修复**：在 `DualVersaCoreSwigluGen.scala` 中，让 chunk 串行化器按 `active_num_chunks(array_shape)` 停止，而不是固定使用 `NumChunks=8`。S2 同理（active chunks=2）。

### Phase 2 - Mode0 D1 正确性检查

- 先从源码（`DualVersaCoreSwigluGen.scala`）确认 D1 语义：Mode0 中 `rescale_mul_out_data` 同时驱动 `oa0_in_data` 和 `oa1_in_data`，即 D1 与 D0 接收同一 postprocess 流——D1 确实等于 D0 golden，但这是**经验证的事实，不是预先假设**。
- `datagen.py` 新增 `mode0_d1_golden_padded` 数组。
- C app 在 Mode0 完成后、Mode1 启动前，分别对 D0 和 D1 做对比，打印独立的 PASS/FAIL 行。
- 三个形状均 Mode0 D1 PASS。

### Phase 3 - S0 回归与最终三形状扫描

- 将 `params.hjson` 恢复 `array_shape=0` 后，**发现 S0 回归**：新加的 active chunk-count 信号位宽用了 `$clog2(NumChunks)` bit，当 `NumChunks=8` 时最大可表示 7，S0 的 active count=8 溢出为 0，导致 chunk 串行化立即停止，accelerator 再次挂死。
- Agent **自主发现并修复**：将返回类型改为 `$clog2(NumChunks + 1)` bit，重新运行 `rtl-gen` 和 `bin/snitch_cluster.vlt` 构建。
- 最终三形状全部 PASS（含 Mode0 D0/D1、Mode1 D0/D1）：

| 形状 | Mode0 D0/D1 | Mode1 D0/D1 | M0 wall | M1 wall |
|------|-------------|-------------|---------|---------|
| S0 | PASS/PASS | PASS/PASS | 2896 | 2731 |
| S1 | PASS/PASS | PASS/PASS | 2888 | 2748 |
| S2 | PASS/PASS | PASS/PASS | 2888 | 2752 |

- `params.hjson` 恢复 `array_shape=0`，最终 S0 仍 PASS。
- Skill `versacore-snax-fusion-design` 已更新，记录 active chunk count 规则。

---

## 3. 遗留问题

### 唯一遗留：Route B（`delta_local_a1` 清理）

`delta_local_a1` 在 datagen 中仍作为未使用的兼容性填充存在（占用约 512~1056 字节，具体因 shape 而异）。Agent 明确 defer 并给出理由：移除后所有下游 buffer（W2L/W2R/Mode1 D0/D1）地址都会变化，三个形状的 bank 分配需要重新验证，代价较高。

这是合理的 defer 决定。Route B 清理可在下一轮独立任务中处理，不影响当前验证结论的正确性。

---

## 4. Spec 要求交付物对照

| 要求 | 状态 | 说明 |
|------|------|------|
| S0 direct-read 开始时仍 PASS | ✅ | wall=2859，与上次一致 |
| S1 超时根因分类并修复 | ✅ | shell chunk 串行化 count bug，Chisel generator 修复 |
| S2 超时根因分类并修复 | ✅ | 同 S1，同一根因 |
| S1/S2 修复后 PASS | ✅ | 全通，含 Mode0 D0/D1 |
| Mode0 D1 检查添加 | ✅ | 源码确认语义后添加，三形状均 PASS |
| 最终三形状扫描 | ✅ | S0/S1/S2 全 PASS |
| 周期记录 | ✅ | cycle_comparison.md 完整 |
| params.hjson 恢复 array_shape=0 | ✅ | 已恢复并再次验证 |
| Route B 清理 | ⏸️ Defer | 有意推迟，理由充分 |
| Skill 更新 | ✅ | versacore-snax-fusion-design 已更新 |

---

## 5. 工程质量

- **诊断流程规范**：严格按 escalation 顺序执行，未跳步。
- **RTL 修改有依据**：SW/datagen 路径穷尽后才修改 Chisel，修改范围最小。
- **回归自检**：S0 width bug 是 agent 自己引入并自己发现修复的，没有"交付后回归"。
- **Mode0 D1 语义先确认后编码**：符合 review 对上次任务的要求。
- **临时调试代码清理干净**：RTL probe 和 app progress print 在修复后均已移除。

---

## 6. 下一步建议

当前 S4 系列任务已基本完成。后续可选项：

1. **Route B 清理**（低优先级）：移除 `delta_local_a1` datagen 分配，重新计算三个形状的 W2/Mode1 D buffer bank 分配，验证 S0/S1/S2 仍全通。
2. **跨 workload 验证**：确认 active chunk-count 修复对其他 workload（非 SwiGLU）是否有影响，或检查是否有其他 shell 存在类似 fixed-count 问题。
3. **M=1 以外的 M 维度测试**：目前三个形状均为 M=1，若未来需要 M>1，需要验证布局契约是否成立。
