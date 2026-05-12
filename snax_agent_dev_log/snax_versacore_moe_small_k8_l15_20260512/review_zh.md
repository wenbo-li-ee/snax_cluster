# Review：small MoE K8 L15 weights-first app

**日期：** 2026-05-12  
**cfg：** `snax_cluster/target/snitch_cluster/cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson`  
**新 app：** `snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15`  
**模板：** `snax-versacore-int16x4-multishape-k8-8x4-mode1-contiguous-l15`

## 1. 改动目标

这个 app 是给 MoE 场景准备的小尺寸快速验证版。核心目标有两个：

1. **TCDM 上权重放前面，token 放所有权重后面。** MoE 里专家权重空间是确定的，token 数量和路由可能变化，所以这个 layout 更接近真实部署。
2. **尺寸改小，跑得更快。** 输入 token 数仍是 8，但 hidden dimension 从 2048 降到 1024；Mode0 的 W/V 是 `1024 x 128`；Mode1 的 down projection 总体是 `128 x 1024`，拆成两个 `128 x 512` 矩阵。

固定尺寸：

```text
M_total  = 8
K0_total = 1024
N0_total = 128
K1_total = 128
N1_total = 512
```

保持不变的 L15 逻辑：

```text
A padding  = 32 B
B1 color   = 272 B = bank34
W2L color  = 128 B = bank16
M1D0 color = 256 B = bank32
```

## 2. 文件变化

新增目录：

```text
snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15/
```

新增 app 已加入：

```text
snax_cluster/target/snitch_cluster/sw/apps/Makefile
```

在 `CFG_OVERRIDE=cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson` 时会进入 `SUBDIRS`。

## 3. datagen 关键变化

### 3.1 权重优先 placement

旧 app 是 A 放最前。新 app 改成：

```text
B0/W -> B1/V -> W2_left -> W2_right -> A tokens -> Mode0 D0 -> Mode1 D0/D1
```

生成出来的实际 offset：

| Tensor | offset bytes | bank | 说明 |
|---|---:|---:|---|
| B0/W | 0 | 0 | Mode0 第一权重 |
| B1/V | 65808 | 34 | Mode0 第二权重 |
| W2_left | 132224 | 16 | Mode1 左半 down projection |
| W2_right | 165888 | 0 | Mode1 右半 down projection |
| A | 198656 | 0 | token buffer，在所有权重之后 |
| Mode0 D0 | 216064 | 0 | Mode0 输出，也是 Mode1 A |
| Mode1 D0 | 218368 | 32 | Mode1 左输出 |
| Mode1 D1 | 219392 | 32 | Mode1 右输出 |
| End | 235008 | 0 | 总 TCDM 占用 |

### 3.2 小尺寸 tensor 大小

| Tensor | bytes |
|---|---:|
| W/V each | 65536 |
| W2_left/W2_right each | 32768 |
| A padded | 16640 |
| Mode0 D0 | 2048 |
| Mode1 padded output | 16640 |
| Total used | 235008 |

### 3.3 A padding

新的 A row：

```text
有效 A 行 = 1024 * 2 = 2048 B
padding  = 32 B
stride   = 2080 B
row      = 1040 int16
```

也就是每个 token 行前 1024 个 int16 是有效数据，后 16 个 int16 是 padding。

### 3.4 B reader stride 参数化

这是这次除了 placement 和 size 之外最关键的修正。旧 app 里的 B spatial stride 对应旧大尺寸：

```text
Mode0 B sstride[1] = 4096
Mode1 B sstride[1] = 2816
```

新尺寸必须改成从 tile 数计算：

```python
mode0_b_sstride = k0_s0_tiles * 16  # 128 * 16 = 2048
mode1_b_sstride = k1_s0_tiles * 16  # 16 * 16 = 256
```

生成结果：

```text
mode0_B_sstride = { 8, 2048 }
mode1_B_sstride = { 8, 256 }
```

如果不改这个，streamer 会按旧矩阵 layout 读权重，结果很容易错。

## 4. shape 配置

| Shape | K_tiles | N_tiles | K1 | N1 | Mode0 B sstride | Mode1 B sstride |
|---|---:|---:|---:|---:|---|---|
| S0 | 128 | 32 | 16 | 128 | `{8,2048}` | `{8,256}` |
| S1 | 128 | 16 | 16 | 64 | `{8,2048}` | `{8,256}` |
| S2 | 128 | 8 | 16 | 32 | `{8,2048}` | `{8,256}` |

Mode1 输出仍然是 padded contiguous row：

```text
每个 token row = [D0 left 512 int16][D1 right 512 int16][16 int16 padding]
row_bytes      = 2080
```

## 5. 仿真结果

用已有 `target/snitch_cluster/bin/snitch_cluster.vlt` 跑通，没有重新生成 RTL，也没有重建硬件。

```text
total checks: 6
total error : 0
```

| Shape | Mode | 正确性 | accelerator cycles | streamer cycles | wall cycles |
|---|---|---|---:|---:|---:|
| S0 | Mode0 | PASS | 4102 | 4126 | 42311 |
| S0 | Mode1 | PASS | 2180 | 2199 | 235108 |
| S1 | Mode0 | PASS | 2058 | 2082 | 28702 |
| S1 | Mode1 | PASS | 1110 | 1129 | 126607 |
| S2 | Mode0 | PASS | 1037 | 1061 | 22794 |
| S2 | Mode1 | PASS | 544 | 563 | 72594 |

现在 C 程序已经重构成固定跑 `layout_cfgs[0]`、固定跑完 S0/S1/S2、每个 shape 固定跑 Mode0+Mode1，不再依赖 `SELECT_LAYOUT/SELECT_SHAPE/RUN_MODE1` 这些宏选项。

cycle 文件：

```text
review_log/snax_versacore_moe_small_k8_l15_20260512/cycles.md
```

## 6. 给 Hemaia agent 的注意事项

这个 app 证明了：只要 `delta_local_*`、stride、bounds 都一致更新，A/token buffer 不需要放在 TCDM 开头。streamer 只关心 base pointer 和 stride，不关心 tensor 在 TCDM 里的先后顺序。

但这个 app 还不是完整 MoE layer datagen：

- 只有单个专家的 W/V/W2。
- token 数还是固定 8。
- 没有 expert routing table。
- 没有多个专家权重 prefix 的 layout。

完整 MoE datagen 可以沿这个方向扩展：

```text
expert0 weights
expert1 weights
...
all routed token blocks
all intermediate/output blocks
```

每次跑某个 expert 时，只需要选择该 expert 的 weight base 和对应 token block base，再复用当前 streamer shape 公式。性能方面，当前 L15 coloring 是继承来的，已通过正确性和小尺寸仿真，但如果最终 expert 数、token block 位置、矩阵尺寸变化较大，建议重新 sweep `A pad / B1 bank / W2_left bank / Mode1 D0 bank`。
