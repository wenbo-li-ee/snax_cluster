#!/usr/bin/env python3

"""Generate a compact FP16xINT4 closed-loop SwiGLU test.

Both mode-0 matrices use identity INT4 weights, so the FP32 postprocess sees
the FP16 input values directly:

    mode0 = fp16(fp32_silu(A) * fp32(A))

Mode 1 reads that FP16 writeback in place.  Its left weight is identity and its
right weight is 2 * identity:

    mode1_d0 = mode0
    mode1_d1 = fp16(2 * mode0)
"""

import argparse

import hjson
import numpy as np


BREAKPOINTS = np.array([
    -8.0,
    -5.4833984375,
    -2.0146484375,
    -0.94970703125,
    0.943359375,
    1.98291015625,
    6.0,
], dtype=np.float32)

A0 = np.array([
    -877179 / 2**22,
    -2166905 / 2**22,
    -518345 / 2**22,
    13417 / 2**22,
    -501092 / 2**22,
    -2146849 / 2**22,
], dtype=np.float32)

A1 = np.array([
    -3415 / 2**16,
    -10937 / 2**16,
    14823 / 2**16,
    32771 / 2**16,
    50290 / 2**16,
    76289 / 2**16,
], dtype=np.float32)

A2 = np.array([
    -54 / 2**14,
    -229 / 2**14,
    1374 / 2**14,
    3718 / 2**14,
    1412 / 2**14,
    -223 / 2**14,
], dtype=np.float32)


def f32(value):
    return np.float32(value)


def silu_fp32(value):
    x = f32(value)
    if x < BREAKPOINTS[3]:
        if x < BREAKPOINTS[2]:
            segment = 0 if x < BREAKPOINTS[1] else 1
        else:
            segment = 2
    else:
        if x < BREAKPOINTS[5]:
            segment = 3 if x < BREAKPOINTS[4] else 4
        else:
            segment = 5
    stage0 = f32(f32(A2[segment] * x) + A1[segment])
    return f32(f32(stage0 * x) + A0[segment])


def fp16_bits(values):
    return np.asarray(values, dtype=np.float16).view(np.uint16)


def pack_int4(values):
    flat = np.asarray(values, dtype=np.int8).reshape(-1)
    assert len(flat) % 2 == 0
    assert np.all(flat >= -8) and np.all(flat <= 7)
    lo = flat[0::2].astype(np.uint8) & 0x0F
    hi = flat[1::2].astype(np.uint8) & 0x0F
    return lo | (hi << 4)


def identity_weight(scale):
    # VersaCore B order is [meshCol, tileSize].
    weight = np.zeros((8, 8), dtype=np.int8)
    for col in range(8):
        weight[col, col] = scale
    return pack_int4(weight)


def c_u16_array(name, values):
    body = ", ".join(f"0x{int(v):04x}" for v in values)
    return f"static const uint16_t {name}[{len(values)}] = {{ {body} }};"


def c_u8_array(name, values):
    body = ", ".join(f"0x{int(v):02x}" for v in values)
    return f"static const uint8_t {name}[{len(values)}] = {{ {body} }};"


def emit_header(params):
    assert int(params["array_shape"]) == 1

    # All values are exactly representable in FP16 and stay far from overflow.
    # The first row crosses every one of the six SiLU polynomial segments.
    row = np.array(
        [-6.0, -3.0, -1.5, -0.5, 0.5, 1.5, 3.0, 6.0],
        dtype=np.float16,
    )
    input_a = np.vstack([row, row[::-1], row * 0.5, -row]).astype(np.float16)

    mode0_fp16 = np.empty_like(input_a)
    for index, value in np.ndenumerate(input_a):
        x = f32(value)
        mode0_fp16[index] = np.float16(f32(silu_fp32(x) * x))

    mode1_d0 = mode0_fp16.copy()
    mode1_d1 = np.asarray(
        [np.float16(f32(f32(v) * f32(2.0))) for v in mode0_fp16.reshape(-1)],
        dtype=np.float16,
    ).reshape(mode0_fp16.shape)

    identity = identity_weight(1)
    double_identity = identity_weight(2)

    return "\n".join([
        "#include <stdint.h>",
        "",
        "#ifndef SNAX_DUAL_VC_FP16X4_SWIGLU_DATA_H",
        "#define SNAX_DUAL_VC_FP16X4_SWIGLU_DATA_H",
        "",
        "#define ARRAY_SHAPE 1",
        "#define DATA_TYPE 0",
        "#define K_TILES 1",
        "#define OUTPUT_TILES 1",
        "#define OUTPUT_ELEMENTS 32",
        "#define A_DATA_LENGTH 64",
        "#define WEIGHT_DATA_LENGTH 32",
        "",
        "#define DELTA_LOCAL_A 0",
        "#define DELTA_LOCAL_B0 1024",
        "#define DELTA_LOCAL_B1 2048",
        "#define DELTA_LOCAL_MODE0_D0 3072",
        "#define DELTA_LOCAL_W2_LEFT 5120",
        "#define DELTA_LOCAL_W2_RIGHT 6144",
        "#define DELTA_LOCAL_MODE1_D0 7168",
        "#define DELTA_LOCAL_MODE1_D1 7240",
        "",
        c_u16_array("input_a_fp16", fp16_bits(input_a.reshape(-1))),
        c_u8_array("mode0_weight_w", identity),
        c_u8_array("mode0_weight_v", identity),
        c_u8_array("mode1_weight_left", identity),
        c_u8_array("mode1_weight_right", double_identity),
        c_u16_array("mode0_golden_fp16", fp16_bits(mode0_fp16.reshape(-1))),
        c_u16_array("mode1_d0_golden_fp16", fp16_bits(mode1_d0.reshape(-1))),
        c_u16_array("mode1_d1_golden_fp16", fp16_bits(mode1_d1.reshape(-1))),
        "",
        "#endif",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swcfg", required=True)
    args = parser.parse_args()
    with open(args.swcfg, "r", encoding="utf-8") as handle:
        params = hjson.load(handle)
    print(emit_header(params))


if __name__ == "__main__":
    main()
