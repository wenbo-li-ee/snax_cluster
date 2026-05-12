# C 程序伪代码：small MoE K8 L15 app

**对应源码：** `snax_cluster/target/snitch_cluster/sw/apps/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15/src/snax-versacore-int16x4-moe-small-k8-mode1-contiguous-l15.c`

这份 C 程序做的事情可以简化成：

```text
1. 从 data.h 读取 datagen 生成好的 layout / shape / golden 配置
2. DMA core 把 A 和所有 weight 从 L3 搬到 TCDM
3. compute core 对每个 shape 依次跑：
   Mode0: A * W 和 A * V，做 SiLU/Gate，写 D0
   Mode1: 读 D0，乘 W2_left/W2_right，写 padded-contiguous output
4. 每一步和 Python golden 比较
5. 打印 cycle 和 error
```

## 1. 顶层 main

```text
function main():
    err = 0

    if this is core0:
        print app name, number of layouts, number of shapes

    layout_cfg = layout_cfgs[0]

    stage_err = stage_layout_to_tcdm(layout_cfg)
    if stage_err != 0:
        return 1

    if this is not core0:
        return 0

    print "dataset ready"

    for shape_id from 0 to NUM_SHAPES - 1:
        cfg = layout_cfg.shapes[shape_id]
        err += run_shape(layout_cfg.layout_id, cfg)

    print total checks = NUM_SHAPES * 2
    print total error = err
    return err
```

重构后的程序不再用 `SELECT_LAYOUT`、`SELECT_SHAPE`、`RUN_MODE1` 这些宏选项。它固定使用 `layout_cfgs[0]`，也就是 L15；固定跑 S0/S1/S2；每个 shape 固定跑 Mode0 和 Mode1。

## 2. stage_layout_to_tcdm：把 L3 数据搬到 TCDM

```text
function stage_layout_to_tcdm(layout):
    cfg0 = layout.shapes[0]

    local_a   = TCDM_BASE + cfg0.delta_local_a
    local_b0  = TCDM_BASE + cfg0.delta_local_b0
    local_b1  = TCDM_BASE + cfg0.delta_local_b1
    local_w2l = TCDM_BASE + cfg0.delta_local_w2l
    local_w2r = TCDM_BASE + cfg0.delta_local_w2r

    if cfg0.tcdm_end > TCDM_CAPACITY_BYTES:
        print capacity error
        return 1

    if this is core0:
        print layout name, TCDM bytes, A stride, bank info

    if this is DMA core:
        dma_start_cycle = mcycle()

        DMA copy layout.a_data       -> local_a
        DMA copy layout.w_data       -> local_b0
        DMA copy layout.v_data       -> local_b1
        DMA copy layout.w2_left_data -> local_w2l
        DMA copy layout.w2_right_data-> local_w2r

        wait all DMA transfers

        dma_end_cycle = mcycle()
        print DMA staging cycles

    cluster barrier
    return 0
```

注意：虽然 TCDM layout 是 weights-first，但这里 DMA 调用顺序先搬了 A。最终放在哪里由 `delta_local_*` 这些目的地址决定，不由 DMA 调用顺序决定。

D0、Mode1 D0、Mode1 D1 不从 L3 搬进来，它们是 accelerator writer 后面写出来的 TCDM output buffer。

## 3. run_shape：对一个 shape 跑 Mode0 和 Mode1

```text
function run_shape(layout_id, cfg):
    subtraction_setting = make subtraction config

    print "Lx Sy start"

    mode0_err = run_mode0(layout_id, cfg, subtraction_setting)
    if mode0_err != 0:
        return mode0_err

    return run_mode1(layout_id, cfg, subtraction_setting)
```

`cfg` 是 datagen 在 `data.h` 里为某个 shape 生成的 `shape_cfg_t`。里面包含：

```text
array_shape
meshRow / tileSize / meshCol
K_tiles / N_tiles / K1 / N1
A/B/D 的 streamer spatial stride
A/B/D 的 streamer temporal bound 和 stride
A/B/D 的 channel enable
所有 TCDM offset
golden output pointer
```

## 4. Mode0 伪代码

Mode0 做的是：

```text
D0 = SiLU(A * W) * (A * V)
```

C 程序流程：

```text
function run Mode0:
    m0_start = mcycle()

    configure streamer for Mode0:
        A reader:
            base    = cfg.delta_local_a
            stride  = cfg.mode0_A_sstride / tstride / tbound
            channels= cfg.A_channel_en

        B0 reader:
            base    = cfg.delta_local_b0
            stride  = cfg.mode0_B_sstride / tstride / tbound
            channels= cfg.B_channel_en

        B1 reader:
            base    = cfg.delta_local_b1
            stride  = same as B0
            channels= cfg.B_channel_en

        D0 writer:
            base    = cfg.delta_local_d0
            stride  = cfg.D_sstride / cfg.mode0_D_tstride / cfg.mode0_D_tbound
            channels= cfg.D_channel_en

    configure dual VersaCore:
        batch / M loops = 1
        K loops         = cfg.K_tiles
        N loops         = cfg.N_tiles * cfg.M_tiles
        subtraction     = subtraction_setting
        array shape     = cfg.array_shape
        data type       = int16x4

    set accelerator mode = 0
    set rescale0 to identity
    set rescale1 to identity
    set rescale_mul to identity

    start streamer
    start accelerator

    wait until accelerator is not busy
    if timeout:
        return Mode0 accelerator timeout error

    wait until streamer / writer is not busy
    if timeout:
        return Mode0 writer timeout error

    m0_end = mcycle()

    compare local_d0 with cfg.mode0_d0_golden
    print PASS/FAIL
    print accelerator cycles, streamer cycles, wall cycles

    if compare failed:
        return Mode0 correctness error
```

这个阶段只用 D0 writer，不用 D1 writer，所以源码调用的是：

```c
set_dual_versacore_streamer_csr_d0_only(...)
```

## 5. Mode1 伪代码

Mode1 做的是：

```text
left  = D0 * W2_left
right = D0 * W2_right
output row = [left, right, padding]
```

C 程序流程：

```text
function run Mode1:
    m1_start = mcycle()

    zero local_mode1_d padded output buffer
        reason:
            writer only writes left/right real output
            padding elements should remain 0 for golden compare

    configure streamer for Mode1:
        A reader:
            base    = cfg.delta_local_d0
            stride  = cfg.mode1_A_sstride / tstride / tbound
            channels= cfg.A_channel_en
            meaning = read Mode0 D0 as Mode1 input

        B0 reader:
            base    = cfg.delta_local_w2l
            stride  = cfg.mode1_B_sstride / tstride / tbound
            channels= cfg.B_channel_en
            meaning = read W2_left

        B1 reader:
            base    = cfg.delta_local_w2r
            stride  = same as B0
            channels= cfg.B_channel_en
            meaning = read W2_right

        D0 writer:
            base    = cfg.delta_local_mode1_d0
            stride  = cfg.mode1_D_tstride / cfg.mode1_D_tbound
            meaning = write left half

        D1 writer:
            base    = cfg.delta_local_mode1_d1
            stride  = same as D0
            meaning = write right half

    configure dual VersaCore:
        batch / M loops = 1
        K loops         = cfg.K1
        N loops         = cfg.N1 * cfg.M_tiles
        subtraction     = subtraction_setting
        array shape     = cfg.array_shape
        data type       = int16x4

    set accelerator mode = 1
    set rescale0 to identity
    set rescale1 to identity

    start streamer
    start accelerator

    wait until accelerator is not busy
    if timeout:
        return Mode1 accelerator timeout error

    wait until streamer / writers are not busy
    if timeout:
        return Mode1 writer timeout error

    m1_end = mcycle()

    compare local_mode1_d with cfg.mode1_padded_golden
    print PASS/FAIL
    print accelerator cycles, streamer cycles, wall cycles
```

Mode1 用两个 writer，所以源码调用的是：

```c
set_dual_versacore_streamer_csr(...)
```

## 6. wait helper

### wait_accelerator_done

```text
function wait_accelerator_done(layout, shape, mode):
    write DUAL_VC_START = 0
    write DUAL_VC_START = 0

    start = mcycle()

    while DUAL_VC_BUSY is true:
        if mcycle() - start > timeout:
            print accel / streamer / writer busy status
            return 1

    return 0
```

### wait_streamer_done

```text
function wait_streamer_done(layout, shape, mode):
    write STREAMER_START_CSR = 0
    write STREAMER_START_CSR = 0

    start = mcycle()

    while STREAMER_BUSY_CSR is true:
        if mcycle() - start > timeout:
            print streamer / writer busy status
            return 1

    return 0
```

这两个函数主要是防止仿真卡死。如果 streamer 配错、TCDM bank/interconnect 卡住、writer 没结束，就会超时并打印状态。

## 7. compare helper

```text
function check_result_i16_limited(output, golden, num_elements):
    err = 0

    for i in 0 .. num_elements - 1:
        if output[i] != golden[i]:
            if err < 16:
                print mismatch index, output value, golden value
            err += 1

    if err > 16:
        print mismatch was capped

    return err
```

它只打印前 16 个 mismatch，避免仿真 log 被大量错误刷屏。

## 8. 一句话流程图

```text
main
  |
  +-- use layout_cfgs[0]
  |
  +-- stage_layout_to_tcdm
  |     |
  |     +-- DMA A, W, V, W2_left, W2_right into TCDM
  |     +-- barrier
  |
  +-- non-core0 exits
  |
  +-- for each selected shape:
        |
        +-- Mode0:
        |     configure A/B0/B1 readers + D0 writer
        |     configure dual VersaCore mode 0
        |     start, wait, compare D0
        |
        +-- Mode1:
              clear padded output
              configure D0-as-A reader + W2 readers + D0/D1 writers
              configure dual VersaCore mode 1
              start, wait, compare padded output
```

## 9. 最核心的 mental model

```text
datagen.py 决定：
  数据长什么样
  数据放在 TCDM 哪里
  streamer 怎么读写
  正确 golden 是什么

C 程序决定：
  什么时候把数据 DMA 到 TCDM
  什么时候把 datagen 给的 CSR 配进去
  什么时候启动 accelerator
  怎么等待结束
  怎么比较结果
```

所以这个 C 文件本身不重新推导 layout，它只是执行 `data.h` 里已经生成好的 layout recipe。
