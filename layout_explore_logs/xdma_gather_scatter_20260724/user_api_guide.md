# XDMA Gather/Scatter 用户 API 指南

本文说明如何使用 XDMA 的 indexed-lane 地址模式，在一个 512-bit beat 中
gather/scatter 四个任意的 16-byte token，同时保留原 Cartesian stride
模式。

## 1. 使用前提

目标 cluster 配置的 `xdma` 节点必须包含：

```hjson
has_gather_scatter: true
```

修改配置后需要用同一份 `CFG_OVERRIDE` 重新生成 RTL 和软件头文件。生成的
`snax-xdma-addr.h` 应满足：

```c
#define XDMA_HAS_GATHER_SCATTER 1
#define XDMA_ADDR_MODE_STRIDE 0
#define XDMA_ADDR_MODE_INDEXED 1
```

应用程序需要包含：

```c
#include "snax-xdma-lib.h"
```

当前验证配置为：

```text
cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson
```

在该配置中：

```c
XDMA_WIDTH        == 64  // 每个 lane 为 64 bit，即 8 byte
XDMA_SPATIAL_CHAN == 8   // 一个 512-bit beat 有 8 个 lane
```

## 2. 新增 API

```c
int32_t snax_xdma_set_src_address_mode(
    uint32_t mode,
    const uint32_t *channel_offsets);

int32_t snax_xdma_set_dst_address_mode(
    uint32_t mode,
    const uint32_t *channel_offsets);
```

参数含义：

- `mode = XDMA_ADDR_MODE_STRIDE`：使用原 Cartesian spatial stride。
- `mode = XDMA_ADDR_MODE_INDEXED`：使用每个 lane 独立的 offset。
- `channel_offsets`：长度至少为 `XDMA_SPATIAL_CHAN` 的数组；每项是相对
  source/destination base pointer 的 byte offset。
- stride 模式不读取 `channel_offsets`，可以传 `NULL`。

返回值：

- `0`：配置成功。
- `-1`：mode 非法，或者 indexed 模式传入了 `NULL` offsets。
- `-2`：当前生成的 XDMA 没有启用 `has_gather_scatter`，却请求了 indexed
  模式。

建议在应用中检查：

```c
#if !XDMA_HAS_GATHER_SCATTER
#error "This app requires XDMA gather/scatter support"
#endif
```

## 3. 地址语义

stride 模式保持原行为：

```text
address[lane] =
    base_pointer
  + temporal_offset
  + Cartesian_spatial_offset[lane]
```

indexed 模式变为：

```text
address[lane] =
    base_pointer
  + temporal_offset
  + channel_offsets[lane]
```

`channel_offsets` 是相对地址，不是绝对地址。当前每个 lane 访问 8 byte，
因此 `base_pointer + channel_offsets[lane]` 应当按 8 byte 对齐，并且落在
合法的 TCDM 地址范围内。

如果一个 token 占连续两个 64-bit bank，一个 16-byte token 的 offset pair
应写成：

```c
token_offset, token_offset + 8
```

四个任意 token 可以表示为：

```c
static const uint32_t offsets[XDMA_SPATIAL_CHAN] = {
    token0_offset, token0_offset + 8,
    token1_offset, token1_offset + 8,
    token2_offset, token2_offset + 8,
    token3_offset, token3_offset + 8,
};
```

例如选择 bank pair `(3,4)、(17,18)、(29,30)、(45,46)`：

```c
static const uint32_t gather_offsets[8] = {
    3 * 8,  4 * 8,
    17 * 8, 18 * 8,
    29 * 8, 30 * 8,
    45 * 8, 46 * 8,
};
```

lane 和地址的对应关系为：

| XDMA lane | 访问地址 |
| --- | --- |
| 0 | `base + temporal_offset + offsets[0]` |
| 1 | `base + temporal_offset + offsets[1]` |
| ... | ... |
| 7 | `base + temporal_offset + offsets[7]` |

## 4. 必须遵守的调用顺序

正确顺序是：

1. 调用 `snax_xdma_memcpy_nd()` 或 `snax_xdma_memcpy_nd_full_addr()` 配置
   base pointer、temporal bounds/strides 和 channel mask；
2. 调用 source/destination address mode API 覆盖需要 indexed 的一侧；
3. 调用 `snax_xdma_start()`；
4. 使用 `snax_xdma_local_wait()` 或 `snax_xdma_remote_wait()` 等待完成。

不要在 `snax_xdma_memcpy_nd()` 之前设置 indexed mode。为了保证旧 API
兼容，`snax_xdma_memcpy_nd[_full_addr]()` 每次都会主动把 source 和
destination 恢复为 `XDMA_ADDR_MODE_STRIDE`。

```c
// 错误：后面的 memcpy 会覆盖 indexed mode。
snax_xdma_set_src_address_mode(XDMA_ADDR_MODE_INDEXED, src_offsets);
snax_xdma_memcpy_nd(...);

// 正确：先配置描述符，再选择 indexed mode。
snax_xdma_memcpy_nd(...);
snax_xdma_set_src_address_mode(XDMA_ADDR_MODE_INDEXED, src_offsets);
uint32_t task_id = snax_xdma_start();
snax_xdma_local_wait(task_id);
```

## 5. 完整 gather→scatter 示例

下面的例子从四个任意 source token gather，然后在同一个 XDMA 任务中
scatter 到四个任意 destination token。例子连续处理 `rows` 行；每一行使用
同一组 lane offsets，并通过 temporal stride 移动到下一行。

```c
#include <stdint.h>

#include "snax-xdma-lib.h"

#define LANE_BYTES 8u

int xdma_gather_scatter_four_tokens(
    void *src,
    void *dst,
    uint32_t rows,
    uint32_t src_row_stride,
    uint32_t dst_row_stride,
    const uint32_t src_offsets[XDMA_SPATIAL_CHAN],
    const uint32_t dst_offsets[XDMA_SPATIAL_CHAN]) {
#if !XDMA_HAS_GATHER_SCATTER
    (void)src;
    (void)dst;
    (void)rows;
    (void)src_row_stride;
    (void)dst_row_stride;
    (void)src_offsets;
    (void)dst_offsets;
    return -2;
#else
    uint32_t bounds[1] = {rows};
    uint32_t src_strides[1] = {src_row_stride};
    uint32_t dst_strides[1] = {dst_row_stride};

    // 先配置普通描述符。两个 spatial stride 在对应侧启用 indexed 后
    // 不参与该侧的 lane 地址计算，这里仍传入一个合法的 lane stride。
    int32_t rc = snax_xdma_memcpy_nd(
        src,
        dst,
        LANE_BYTES,
        LANE_BYTES,
        1,
        src_strides,
        bounds,
        1,
        dst_strides,
        bounds,
        0xffu,  // 启用 source 的 8 个 lane
        0xffu,  // 启用 destination 的 8 个 lane
        0xffu   // 启用每个 64-bit destination word 的 8 个 byte
    );
    if (rc != 0) return rc;

    // 必须位于 snax_xdma_memcpy_nd() 之后。
    rc = snax_xdma_set_src_address_mode(
        XDMA_ADDR_MODE_INDEXED, src_offsets);
    if (rc != 0) return rc;

    rc = snax_xdma_set_dst_address_mode(
        XDMA_ADDR_MODE_INDEXED, dst_offsets);
    if (rc != 0) return rc;

    uint32_t task_id = snax_xdma_start();
    snax_xdma_local_wait(task_id);
    return 0;
#endif
}
```

调用示例：

```c
static const uint32_t src_offsets[8] = {
    3 * 8,  4 * 8,
    17 * 8, 18 * 8,
    29 * 8, 30 * 8,
    45 * 8, 46 * 8,
};

static const uint32_t dst_offsets[8] = {
    6 * 8,  7 * 8,
    20 * 8, 21 * 8,
    35 * 8, 36 * 8,
    51 * 8, 52 * 8,
};

int rc = xdma_gather_scatter_four_tokens(
    src,
    dst,
    3,    // temporal rows
    512,  // source 相邻 row 的 byte stride
    512,  // destination 相邻 row 的 byte stride
    src_offsets,
    dst_offsets);
```

## 6. Gather-only

source 使用 indexed，destination 保持原 stride：

```c
int32_t rc = snax_xdma_memcpy_nd(...);
if (rc != 0) return rc;

rc = snax_xdma_set_src_address_mode(
    XDMA_ADDR_MODE_INDEXED, src_offsets);
if (rc != 0) return rc;

uint32_t task_id = snax_xdma_start();
snax_xdma_local_wait(task_id);
```

因为 `snax_xdma_memcpy_nd()` 已经把 destination 恢复为 stride，所以无需再
显式设置 destination mode。

## 7. Scatter-only

source 保持原 stride，destination 使用 indexed：

```c
int32_t rc = snax_xdma_memcpy_nd(...);
if (rc != 0) return rc;

rc = snax_xdma_set_dst_address_mode(
    XDMA_ADDR_MODE_INDEXED, dst_offsets);
if (rc != 0) return rc;

uint32_t task_id = snax_xdma_start();
snax_xdma_local_wait(task_id);
```

## 8. 使用原 stride 模式

旧程序不需要修改。直接调用原 API 即可：

```c
int32_t rc = snax_xdma_memcpy_nd(...);
if (rc != 0) return rc;

uint32_t task_id = snax_xdma_start();
snax_xdma_local_wait(task_id);
```

也可以显式恢复某一侧：

```c
snax_xdma_set_src_address_mode(XDMA_ADDR_MODE_STRIDE, NULL);
snax_xdma_set_dst_address_mode(XDMA_ADDR_MODE_STRIDE, NULL);
```

## 9. Temporal stride 与能力边界

indexed 模式只替换 spatial lane 地址生成，原 temporal bounds/strides 仍然
生效。因此：

- 一个任务可以连续处理多行；
- 每一行都会使用相同的 8 个 `channel_offsets`；
- 每一行的共同位移由该侧的 temporal AGU 产生。

当前 API 不是从内存读取 index list 的间接寻址引擎。一个任务执行期间，
8 个 offsets 不会逐 beat 自动换成另一组任意值。如果下一组四个 token
没有共同的 temporal stride，需要等待当前任务完成，然后重新配置 offsets
并启动下一个任务。

此外，调用者需要保证：

- source 和 destination 的 temporal frame 数一致；
- 所有启用 lane 的地址合法且满足对齐要求；
- scatter 的 destination offsets 不发生非预期重叠；
- 修改 offsets/mode CSR 时没有仍在执行的旧 XDMA 任务。

## 10. 本地与远端等待

本地 TCDM 搬运使用：

```c
uint32_t task_id = snax_xdma_start();
snax_xdma_local_wait(task_id);
```

跨 cluster/远端任务使用：

```c
uint32_t task_id = snax_xdma_start();
snax_xdma_remote_wait(task_id);
```

mode 和 offsets 已包含在 XDMA 的 intra/inter-cluster 配置传输中，因此两种
执行路径使用相同的 address mode API。

## 11. 参考实现

- 开发与硬件修改记录：[dev_log.md](./dev_log.md)
- 闭环测试 app：
  [snax-xdma-gather-scatter.c](../../target/snitch_cluster/sw/apps/snax-xdma-gather-scatter/src/snax-xdma-gather-scatter.c)
- API 声明：
  [snax-xdma-lib.h](../../target/snitch_cluster/sw/snax/xdma/include/snax-xdma-lib.h)
- API 实现：
  [snax-xdma-lib.c](../../target/snitch_cluster/sw/snax/xdma/src/snax-xdma-lib.c)
