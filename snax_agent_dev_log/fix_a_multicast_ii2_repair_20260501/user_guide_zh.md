# user_guide_zh.md

创建日期：2026-05-01

## 背景

本次修复对象是 dual-VersaCore int16x4 SwiGLU 加速器的共享 A 输入路径。目标配置和应用保持不变：

- 配置：`snax_dual_versacore_int16x4_multidim_spatial_16x2.hjson`
- 应用：`snax-versacore-int16x4-multishape-16x2`

修复没有修改 VersaCore 内部握手逻辑，例如 `Array.scala`、`Accumulator.scala`、`VersaCore.scala`。所有关键修复都在 shell/wrapper/generator 侧完成。

## 旧的 A multicast 握手

旧 wrapper 对 A 输入使用一个共享寄存槽：

1. streamer 的 A beat 进入 `a_buf_data`。
2. `a_buf_valid` 表示该 A beat 当前有效。
3. `a_buf_sent_0` 和 `a_buf_sent_1` 分别记录这个 A beat 是否已经被 VC0/VC1 接收。
4. 只有两个 VC 都接收后，`a_buf_valid` 才清零。

这个设计的正确性思想是对的：同一个 A beat 必须被两个 VC 各消费一次，代表同一个逻辑 K step。

问题在于旧逻辑的 A reader ready 是：

```text
stream2acc_0_ready_o = !a_buf_valid
```

也就是说，只要槽里还有当前 A beat，即使本周期两个 VC 正好都消费完成，A reader 在同一个周期仍然看不到 ready。下一个 A beat 必须等到下一周期才能进入。结果形成“消费一拍、补充一拍”的结构性交替，吞吐接近 `II=2`。

## 新的 A multicast 握手

新逻辑保留“一个共享 A 槽 + 两个 sent bit”的正确性模型，但允许同周期 refill：

```text
a_buf_done = 当前 A 有效，并且 VC0/VC1 在 sent bit 或本周期 fire 后都已收到
stream2acc_0_ready_o = !a_buf_valid || a_buf_done
```

因此，当当前 A beat 在本周期被两个 VC 都接收完成时，A reader 可以在同一周期送入下一个 A beat。这样去掉了旧设计的 refill bubble。

## atomic dual-consume 如何保证

每个 VC 仍然有独立的 sent bit：

- VC0 只有在 `vc0_in_a_valid && vc0_in_a_ready && vc0_in_b_ready` 时才标记收到当前 A。
- VC1 只有在对应条件成立时才标记收到当前 A。
- A 槽只有在两个 sent bit 都完成后才允许替换为下一个 A。

新设计还把 A 和对应的 B 绑定起来：

- `vc0_in_a_valid` 需要当前 A 槽有效、VC0 尚未收到、且 B0 valid。
- `vc1_in_a_valid` 需要当前 A 槽有效、VC1 尚未收到、且 B1 valid。
- B0/B1 的 ready 只有在同一个 VC 的 A 和 B 都 ready 时才返回给 streamer。

这样可以防止某个 VC 的 B 流先走一步，也防止 A 在缺少对应 B 的情况下进入 VC 内部。VC0 和 VC1 仍然可以独立响应，不会重新引入跨 VC 的 B 串行化。

## 为什么不需要修改 VersaCore 内部

VersaCore 内部已经提供了正确的 ready/valid 反压语义。问题不在内部 Array/Accumulator，而在 shell 侧的共享 A 槽不能同周期 refill，以及外部 B/输出路径在高吞吐下暴露了配对问题。

因此修复放在 wrapper 中：

- wrapper 负责保证同一个 A beat 对两个 VC 的逻辑原子性；
- wrapper 负责让 B 与对应 A 配对；
- VersaCore 内部继续按照原有 ready/valid 接口工作。

## A/B/D 反压如何传播

A 路径：

- 如果 A 槽为空，A reader 可以写入。
- 如果 A 槽满但两个 VC 本周期都完成当前 A，A reader 可以同周期写入下一拍。
- 如果任意 VC 还没有接收当前 A，A reader 被反压。

B 路径：

- B0 只在 VC0 的 A/B 都能接收当前逻辑 step 时前进。
- B1 只在 VC1 的 A/B 都能接收当前逻辑 step 时前进。
- VC0 和 VC1 不互相等待对方的 B ready；它们只共享同一个 A 槽的生命周期。

D/输出路径：

- VC0/VC1 的 D 输出仍进入原有后处理流水。
- Mode0 中，后处理结果要复制给两个 writer。对于 S6 的 `OutChunks == 1` 情况，新 wrapper 加了一个 Mode0 直接输出保持寄存器。
- 该寄存器保存一个后处理 beat，并用两个 sent bit 跟踪 writer0/writer1 是否已经各自接收一次。
- 因此即使两个 writer ready 有偏斜，同一个 Mode0 输出 beat 也不会被某个 writer 重复接收或漏接。
- 当两个 writer 都达到软件配置的输出 quota 后，Mode0 会在 shell 内部 drain 多余的 postprocess beat，避免 writer 已经结束后加速器仍被反压卡住。

## 验证结果

最终在 `barnard3` 容器内完成了完整流程：

```bash
make -C snax_cluster/target/snitch_cluster rtl-gen \
  CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_16x2.hjson

make -C snax_cluster/target/snitch_cluster sw \
  CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_16x2.hjson

make -C snax_cluster/target/snitch_cluster bin/snitch_cluster.vlt -j$(nproc) \
  CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_16x2.hjson
```

并直接运行真实 S6 ELF：

```text
snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-multishape-16x2/build/snax-versacore-int16x4-multishape-16x2.elf
```

最终结果：

```text
S6 multishape 16x2 total checks: 12, total error: 0
```

周期对比：

| Shape | Mode | 修复前 accel cycles | 修复后 accel cycles |
|---|---:|---:|---:|
| S0 | Mode0 | 180236 | 115688 |
| S0 | Mode1 | 90117 | 55877 |
| S1 | Mode0 | 90120 | 59113 |
| S1 | Mode1 | 45061 | 27542 |
| S2 | Mode0 | 45062 | 28246 |
| S2 | Mode1 | 22533 | 12968 |

结论：结构性的 A multicast `II=2` 瓶颈已经被移除，所有 S6 检查保持正确，并且所有测量模式的周期数都显著下降。
