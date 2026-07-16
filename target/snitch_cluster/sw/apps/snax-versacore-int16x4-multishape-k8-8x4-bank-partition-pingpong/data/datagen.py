#!/usr/bin/env python3

import argparse
import os
import pathlib
import sys

import hjson
import numpy as np

_this_dir = os.path.dirname(os.path.realpath(__file__))
_silu_pkg = os.path.realpath(
    os.path.join(_this_dir, "../../../../../../util/silu_pkg"))
if os.path.isdir(_silu_pkg):
    sys.path.insert(0, _silu_pkg)
else:
    sys.path.insert(0, "/esat/studscratch/r1015498/Thesis/original_snax/silu/pkg")
from silu_out16_balanced_golden import silu_out16_balanced_eval_q  # noqa: E402


SHAPES = [
    ("S0", 0, 8, 8, 4),
    ("S1", 1, 4, 8, 8),
    ("S2", 2, 2, 8, 16),
]

BANKS = 64
BANK_WORD_BYTES = 8
BANK_ROW_BYTES = BANKS * BANK_WORD_BYTES

# One physical bank row is shared by all partitions.  The low address selects
# the bank group; subsequent logical rows advance by 512 B and return to it.
A_BASE = 0                         # banks 0..15
B0_PING_BASE = 16 * BANK_WORD_BYTES   # banks 16..23
B1_PING_BASE = 24 * BANK_WORD_BYTES   # banks 24..31
B0_PONG_BASE = 32 * BANK_WORD_BYTES   # banks 32..39
B1_PONG_BASE = 40 * BANK_WORD_BYTES   # banks 40..47
MODE0_D_BASE = 48 * BANK_WORD_BYTES   # banks 48..63


def c_array(values):
    return "{ " + ", ".join(str(int(v)) for v in values) + " }"


def c_u8_array(values):
    return "{ " + ", ".join(f"0x{int(v) & 0xff:02x}" for v in values) + " }"


def emit_i16(name, values):
    return f"static const int16_t {name}[{len(values)}] = {c_array(values)};"


def emit_u8(name, values):
    return f"static const uint8_t {name}[{len(values)}] = {c_u8_array(values)};"


def rescale_down_32to16(arr_int32, input_zp=0, mult=1, output_zp=0, shift=0):
    result = arr_int32.astype(np.int64) - int(input_zp)
    multiplied = result * np.int64(mult)
    if shift > 0:
        rounding = np.int64(1) << (shift - 1)
        shifted = multiplied + rounding
        adjusted = np.where(result >= 0, shifted + np.int64(1 << 30),
                            shifted - np.int64(1 << 30))
        shifted = np.where(shift > 31, adjusted, shifted) >> shift
    else:
        shifted = multiplied
    out = shifted.astype(np.int32).astype(np.int64) + int(output_zp)
    return np.clip(out, -32768, 32767).astype(np.int16)


def apply_silu(values):
    flat = values.reshape(-1)
    result = np.array(
        [silu_out16_balanced_eval_q(int(x)) for x in flat], dtype=np.int16)
    return result.reshape(values.shape)


def block_gemm(m_tiles, k_tiles, n_tiles, mesh_row, tile_size, mesh_col,
               a_flat, b_flat):
    a = a_flat.astype(np.int32).reshape(
        m_tiles, k_tiles, mesh_row, tile_size)
    b = b_flat.astype(np.int32).reshape(
        n_tiles, k_tiles, mesh_col, tile_size)
    d = np.zeros((m_tiles, n_tiles, mesh_row, mesh_col), dtype=np.int32)
    for mm in range(m_tiles):
        for nn in range(n_tiles):
            d[mm, nn] = np.tensordot(
                a[mm], b[nn], axes=([0, 2], [0, 2]))
    return d


def make_a(m_total, k_total):
    a = np.zeros((m_total, k_total), dtype=np.int16)
    for m in range(m_total):
        for k in range(k_total):
            a[m, k] = ((m * 5 + k * 3) % 11) - 5
    return a


def packed_int4_constant(num_bytes, value):
    assert 0 <= value <= 7
    return np.full(num_bytes, (value << 4) | value, dtype=np.uint8)


def make_goldens(a, k0, n0, k1, n1):
    arrays = []
    names = {}
    for shape_name, _, mesh_row, tile_size, mesh_col in SHAPES:
        k0_tiles = k0 // tile_size
        n0_tiles = n0 // mesh_col
        # The token-striped A image is consumed K-tile first, then token.
        a_stream = a[:mesh_row].reshape(
            mesh_row, k0_tiles, tile_size).transpose(1, 0, 2).reshape(-1)
        b0 = np.ones(n0_tiles * k0_tiles * mesh_col * tile_size,
                     dtype=np.int8)
        b1 = np.full_like(b0, 2)
        vc0 = block_gemm(1, k0_tiles, n0_tiles, mesh_row, tile_size,
                         mesh_col, a_stream, b0)
        vc1 = block_gemm(1, k0_tiles, n0_tiles, mesh_row, tile_size,
                         mesh_col, a_stream, b1)
        mode0_raw = rescale_down_32to16(
            apply_silu(rescale_down_32to16(vc0)) *
            rescale_down_32to16(vc1))
        # Writer stores one logical row per token in a two-bank stripe.
        mode0_token = mode0_raw.transpose(0, 2, 1, 3).reshape(mesh_row, n0)

        k1_tiles = k1 // tile_size
        n1_tiles = n1 // mesh_col
        a1_stream = mode0_token.reshape(
            mesh_row, k1_tiles, tile_size).transpose(1, 0, 2).reshape(-1)
        w2l = np.ones(n1_tiles * k1_tiles * mesh_col * tile_size,
                      dtype=np.int8)
        w2r = np.full_like(w2l, 2)
        d0_raw = rescale_down_32to16(block_gemm(
            1, k1_tiles, n1_tiles, mesh_row, tile_size, mesh_col,
            a1_stream, w2l))
        d1_raw = rescale_down_32to16(block_gemm(
            1, k1_tiles, n1_tiles, mesh_row, tile_size, mesh_col,
            a1_stream, w2r))
        d0_token = d0_raw.transpose(0, 2, 1, 3).reshape(mesh_row, n1)
        d1_token = d1_raw.transpose(0, 2, 1, 3).reshape(mesh_row, n1)

        shape_names = (
            f"{shape_name}_mode0_token_golden",
            f"{shape_name}_mode1_d0_token_golden",
            f"{shape_name}_mode1_d1_token_golden",
        )
        names[shape_name] = shape_names
        arrays.append(emit_i16(shape_names[0], mode0_token.reshape(-1)))
        arrays.append(emit_i16(shape_names[1], d0_token.reshape(-1)))
        arrays.append(emit_i16(shape_names[2], d1_token.reshape(-1)))
    return arrays, names


def mode0_d_pattern(array_shape, mesh_row, mesh_col, n_tiles):
    # Convert accelerator tile order to 16-bank token stripes.  Every logical
    # 8-element token chunk occupies two adjacent banks in one physical row.
    if array_shape == 0:
        return [1, mesh_row, 2, n_tiles // 2], [8, 16, 8, 512]
    if array_shape == 1:
        return [2, mesh_row, 2, n_tiles // 2], [8, 16, 64, 512]
    if array_shape == 2:
        return [4, mesh_row, 2, n_tiles // 2], [8, 32, 64, 512]
    raise ValueError(array_shape)


def shape_cfg(shape, globals_, golden_names):
    name, array_shape, mesh_row, tile_size, mesh_col = shape
    k0 = globals_["k0"]
    n0 = globals_["n0"]
    k1 = globals_["k1"]
    n1 = globals_["n1"]
    k0_tiles = k0 // tile_size
    n0_tiles = n0 // mesh_col
    k1_tiles = k1 // tile_size
    n1_tiles = n1 // mesh_col
    q = mesh_col // 4
    mode0_panel_bytes = (k0 // 8) * 16
    mode1_panel_bytes = (k1 // 8) * 16
    mode0_panel_span = (mode0_panel_bytes // 64) * BANK_ROW_BYTES
    mode1_panel_span = (mode1_panel_bytes // 64) * BANK_ROW_BYTES
    a_channel = {0: 0xffff, 1: 0x00ff, 2: 0x000f}[array_shape]
    b_channel = {0: 0x03, 1: 0x0f, 2: 0x00ff}[array_shape]
    mode0_d_bound, mode0_d_stride = mode0_d_pattern(
        array_shape, mesh_row, mesh_col, n0_tiles)
    mode1_a_sstride = {
        0: [8, 16],
        1: [8, 16],
        2: [8, 32],
    }[array_shape]
    mode1_a_tbound = {
        0: [k1_tiles, n1_tiles, 1, 1],
        1: [2, k1_tiles // 2, n1_tiles, 1],
        2: [2, 2, k1_tiles // 4, n1_tiles],
    }[array_shape]
    mode1_a_tstride = {
        0: [512, 0, 0, 0],
        1: [64, 512, 0, 0],
        2: [16, 64, 512, 0],
    }[array_shape]

    # Reserve the rows occupied by Mode0 D before placing Mode1 writeback.
    # The logical Mode0 row has n0/8 K-tiles, each at a 512-B physical pitch.
    mode1_d0_base = MODE0_D_BASE + (n0 // 8) * BANK_ROW_BYTES
    mode1_d1_base = mode1_d0_base + 8 * BANK_WORD_BYTES
    beats_per_token_tile = mesh_col // 4

    fields = [
        f".array_shape = {array_shape}",
        f".meshRow = {mesh_row}",
        f".tileSize = {tile_size}",
        f".meshCol = {mesh_col}",
        f".K_tiles = {k0_tiles}",
        f".N_tiles = {n0_tiles}",
        f".K1_tiles = {k1_tiles}",
        f".N1_tiles = {n1_tiles}",
        f".q_shape0_cols = {q}",
        f".mode0_panel_bytes = {mode0_panel_bytes}",
        f".mode1_panel_bytes = {mode1_panel_bytes}",
        f".mode0_panel_span = {mode0_panel_span}",
        f".mode1_panel_span = {mode1_panel_span}",
        f".mode0_A_sstride = {c_array([8, 16])}",
        f".mode1_A_sstride = {c_array(mode1_a_sstride)}",
        f".mode0_A_tbound = {c_array([k0_tiles, n0_tiles, 1, 1])}",
        f".mode0_A_tstride = {c_array([512, 0, 0, 0])}",
        f".mode1_A_tbound = {c_array(mode1_a_tbound)}",
        f".mode1_A_tstride = {c_array(mode1_a_tstride)}",
        f".mode0_B_sstride = {c_array([8, mode0_panel_span])}",
        f".mode0_B_tbound = {c_array([4, k0_tiles // 4, 2, n0_tiles // 2])}",
        f".mode0_B_tstride = {c_array([16, 512, 128, q * mode0_panel_span])}",
        f".mode1_B_sstride = {c_array([8, mode1_panel_span])}",
        f".mode1_B_tbound = {c_array([4, k1_tiles // 4, 2, n1_tiles // 2])}",
        f".mode1_B_tstride = {c_array([16, 512, 128, q * mode1_panel_span])}",
        f".D_sstride = {c_array([8])}",
        f".mode0_D_tbound = {c_array(mode0_d_bound)}",
        f".mode0_D_tstride = {c_array(mode0_d_stride)}",
        f".mode1_D_tbound = {c_array([beats_per_token_tile, mesh_row, n1_tiles, 1])}",
        f".mode1_D_tstride = {c_array([512, 8, beats_per_token_tile * 512, 0])}",
        f".A_channel_en = {c_array([a_channel])}",
        f".B_channel_en = {c_array([b_channel])}",
        f".D_channel_en = {c_array([1])}",
        f".delta_local_a = {A_BASE}",
        f".delta_local_b0_ping = {B0_PING_BASE}",
        f".delta_local_b1_ping = {B1_PING_BASE}",
        f".delta_local_b0_pong = {B0_PONG_BASE}",
        f".delta_local_b1_pong = {B1_PONG_BASE}",
        f".delta_local_mode0_d = {MODE0_D_BASE}",
        f".delta_local_mode1_d0 = {mode1_d0_base}",
        f".delta_local_mode1_d1 = {mode1_d1_base}",
        f".mode0_token_golden = {golden_names[name][0]}",
        f".mode1_d0_token_golden = {golden_names[name][1]}",
        f".mode1_d1_token_golden = {golden_names[name][2]}",
    ]
    return "    {\n        " + ",\n        ".join(fields) + "\n    }"


def emit_header(params):
    m_total = int(params["M_total"])
    k0 = int(params["K0_total"])
    n0 = int(params["N0_total"])
    k1 = int(params["K1_total"])
    n1 = int(params["N1_total"])
    assert (m_total, k0, n0, k1, n1) == (8, 2048, 1408, 1408, 1024)
    assert all(k0 % s[3] == 0 and k1 % s[3] == 0 for s in SHAPES)

    a = make_a(m_total, k0)
    mode0_weight_bytes = (k0 // 8) * (n0 // 4) * 16
    mode1_weight_bytes = (k1 // 8) * (n1 // 4) * 16
    arrays = [
        emit_i16("A", a.reshape(-1)),
        emit_u8("W", packed_int4_constant(mode0_weight_bytes, 1)),
        emit_u8("V", packed_int4_constant(mode0_weight_bytes, 2)),
        emit_u8("W2_left", packed_int4_constant(mode1_weight_bytes, 1)),
        emit_u8("W2_right", packed_int4_constant(mode1_weight_bytes, 2)),
    ]
    golden_arrays, golden_names = make_goldens(a, k0, n0, k1, n1)
    arrays.extend(golden_arrays)
    globals_ = {"k0": k0, "n0": n0, "k1": k1, "n1": n1}
    cfgs = ",\n".join(shape_cfg(s, globals_, golden_names) for s in SHAPES)

    return "\n".join([
        "#include <stdint.h>",
        "",
        "#ifndef SNAX_VC_K8_BANK_PARTITION_PINGPONG_DATA_H",
        "#define SNAX_VC_K8_BANK_PARTITION_PINGPONG_DATA_H",
        "",
        "#define NUM_SHAPES 3",
        "#define DATA_TYPE 0",
        "#define RESCALE_INPUT_ZP 0",
        "#define RESCALE_MULTIPLIER 1",
        "#define RESCALE_OUTPUT_ZP 0",
        "#define RESCALE_SHIFT 0",
        "#define SUBTRACTION_A 0",
        "#define SUBTRACTION_B 0",
        "#define SET_ADDR_REMAP_INDEX_A 0",
        "#define SET_ADDR_REMAP_INDEX_B0 0",
        "#define SET_ADDR_REMAP_INDEX_B1 0",
        "#define SET_ADDR_REMAP_INDEX_D0 0",
        "#define SET_ADDR_REMAP_INDEX_D1 0",
        f"#define M_TOTAL {m_total}",
        f"#define K0_TOTAL {k0}",
        f"#define N0_TOTAL {n0}",
        f"#define K1_TOTAL {k1}",
        f"#define N1_TOTAL {n1}",
        f"#define A_DATA_LENGTH {a.nbytes}",
        f"#define MODE0_WEIGHT_DATA_LENGTH {mode0_weight_bytes}",
        f"#define MODE1_WEIGHT_DATA_LENGTH {mode1_weight_bytes}",
        f"#define TCDM_CAPACITY_BYTES {8192 * 1024}",
        "",
        "typedef struct {",
        "    int32_t array_shape, meshRow, tileSize, meshCol;",
        "    int32_t K_tiles, N_tiles, K1_tiles, N1_tiles;",
        "    int32_t q_shape0_cols;",
        "    int32_t mode0_panel_bytes, mode1_panel_bytes;",
        "    int32_t mode0_panel_span, mode1_panel_span;",
        "    int32_t mode0_A_sstride[2], mode1_A_sstride[2];",
        "    int32_t mode0_A_tbound[4], mode0_A_tstride[4];",
        "    int32_t mode1_A_tbound[4], mode1_A_tstride[4];",
        "    int32_t mode0_B_sstride[2];",
        "    int32_t mode0_B_tbound[4], mode0_B_tstride[4];",
        "    int32_t mode1_B_sstride[2];",
        "    int32_t mode1_B_tbound[4], mode1_B_tstride[4];",
        "    int32_t D_sstride[1];",
        "    int32_t mode0_D_tbound[4], mode0_D_tstride[4];",
        "    int32_t mode1_D_tbound[4], mode1_D_tstride[4];",
        "    int32_t A_channel_en[1], B_channel_en[1], D_channel_en[1];",
        "    int32_t delta_local_a;",
        "    int32_t delta_local_b0_ping, delta_local_b1_ping;",
        "    int32_t delta_local_b0_pong, delta_local_b1_pong;",
        "    int32_t delta_local_mode0_d;",
        "    int32_t delta_local_mode1_d0, delta_local_mode1_d1;",
        "    const int16_t *mode0_token_golden;",
        "    const int16_t *mode1_d0_token_golden;",
        "    const int16_t *mode1_d1_token_golden;",
        "} shape_cfg_t;",
        "",
        *arrays,
        "",
        "static const shape_cfg_t shape_cfgs[NUM_SHAPES] = {",
        cfgs,
        "};",
        "",
        "#endif",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swcfg", required=True)
    parser.add_argument("--hwcfg", required=True)
    args = parser.parse_args()
    with open(args.swcfg, "r", encoding="utf-8") as handle:
        params = hjson.load(handle)
    # Parse the selected hardware cfg as an early guard against a stale path.
    with open(args.hwcfg, "r", encoding="utf-8") as handle:
        hwcfg = hjson.load(handle)
    acc = hwcfg["snax_dual_versacore_int16x4_core_template"]["snax_acc_cfg"][0]
    assert acc["granularity_a"] == 1
    assert acc["granularity_b"] == 1
    assert acc["granularity_c_d"] == 1
    print(emit_header(params))


if __name__ == "__main__":
    main()
