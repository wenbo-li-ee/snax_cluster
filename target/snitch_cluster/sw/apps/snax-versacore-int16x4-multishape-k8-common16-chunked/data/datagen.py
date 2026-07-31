#!/usr/bin/env python3

"""Generate resident-weight data for the common-16-column chunk experiment.

Weights are stored in the canonical S0 physical order: one full-K panel for
four logical columns.  Runtime S0/S1/S2 readers group 1/2/4 adjacent panels.
Sparse, panel-dependent weights make chunk/panel base mistakes observable.

The logical workload is W/V=[2048,1024] and W2=[1024,2048].  W2 is split
across the two VersaCores as W2_left/right=[1024,1024].
"""

import argparse
import os
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


def c_array(values):
    return "{ " + ", ".join(str(int(v)) for v in values) + " }"


def emit_i16(name, values):
    return f"static const int16_t {name}[{len(values)}] = {c_array(values)};"


def emit_u8(name, values):
    body = ", ".join(f"0x{int(v) & 0xff:02x}" for v in values)
    return f"static const uint8_t {name}[{len(values)}] = {{ {body} }};"


def pack_int4(values):
    flat = np.asarray(values, dtype=np.int8).reshape(-1)
    assert len(flat) % 2 == 0
    assert np.all(flat >= -8) and np.all(flat <= 7)
    lo = flat[0::2].astype(np.uint8) & 0x0f
    hi = flat[1::2].astype(np.uint8) & 0x0f
    return lo | (hi << 4)


def rescale_down_32to16(values):
    return np.clip(values.astype(np.int64), -32768, 32767).astype(np.int16)


def apply_silu(values):
    flat = values.reshape(-1)
    out = np.array(
        [silu_out16_balanced_eval_q(int(x)) for x in flat], dtype=np.int16)
    return out.reshape(values.shape)


def block_gemm(k_tiles, n_tiles, mesh_row, tile_size, mesh_col,
               a_flat, b_flat):
    a = a_flat.astype(np.int32).reshape(k_tiles, mesh_row, tile_size)
    b = b_flat.astype(np.int32).reshape(n_tiles, k_tiles, mesh_col, tile_size)
    out = np.zeros((n_tiles, mesh_row, mesh_col), dtype=np.int32)
    for n in range(n_tiles):
        out[n] = np.tensordot(a, b[n], axes=([0, 2], [0, 2]))
    return out


def make_a(m_total, k_total):
    a = np.zeros((m_total, k_total), dtype=np.int16)
    for token in range(m_total):
        for k in range(k_total):
            a[token, k] = ((token * 5 + k * 3) % 11) - 5
    return a


def make_sparse_s0_weights(n_panels, k_tiles, seed, sign):
    """One nonzero K tile per output column, varying with the S0 panel."""
    w = np.zeros((n_panels, k_tiles, 4, 8), dtype=np.int8)
    for panel in range(n_panels):
        for col in range(4):
            active_k = (panel * (7 + seed) + col * (11 + seed) + seed) % k_tiles
            for kk in range(8):
                magnitude = 1 + ((panel + col + kk + seed) & 1)
                w[panel, active_k, col, kk] = sign * magnitude
    return w


def regroup_s0_panels(weights, mesh_col):
    q = mesh_col // 4
    n_tiles = weights.shape[0] // q
    k_tiles = weights.shape[1]
    return weights.reshape(n_tiles, q, k_tiles, 4, 8).transpose(
        0, 2, 1, 3, 4).reshape(n_tiles, k_tiles, mesh_col, 8)


def make_goldens(a, w, v, w2l, w2r, k0, n0, k1, n1):
    arrays = []
    names = {}
    for shape_name, _, mesh_row, tile_size, mesh_col in SHAPES:
        k0_tiles = k0 // tile_size
        n0_tiles = n0 // mesh_col
        a_stream = a[:mesh_row].reshape(
            mesh_row, k0_tiles, tile_size).transpose(1, 0, 2).reshape(-1)
        w_shape = regroup_s0_panels(w, mesh_col)
        v_shape = regroup_s0_panels(v, mesh_col)
        vc0 = block_gemm(k0_tiles, n0_tiles, mesh_row, tile_size, mesh_col,
                         a_stream, w_shape.reshape(-1))
        vc1 = block_gemm(k0_tiles, n0_tiles, mesh_row, tile_size, mesh_col,
                         a_stream, v_shape.reshape(-1))
        mode0_raw = rescale_down_32to16(
            apply_silu(rescale_down_32to16(vc0)).astype(np.int32) *
            rescale_down_32to16(vc1).astype(np.int32))
        mode0_token = mode0_raw.transpose(1, 0, 2).reshape(mesh_row, n0)

        k1_tiles = k1 // tile_size
        n1_tiles = n1 // mesh_col
        a1_stream = mode0_token.reshape(
            mesh_row, k1_tiles, tile_size).transpose(1, 0, 2).reshape(-1)
        w2l_shape = regroup_s0_panels(w2l, mesh_col)
        w2r_shape = regroup_s0_panels(w2r, mesh_col)
        d0 = rescale_down_32to16(block_gemm(
            k1_tiles, n1_tiles, mesh_row, tile_size, mesh_col,
            a1_stream, w2l_shape.reshape(-1)))
        d1 = rescale_down_32to16(block_gemm(
            k1_tiles, n1_tiles, mesh_row, tile_size, mesh_col,
            a1_stream, w2r_shape.reshape(-1)))
        d0_token = d0.transpose(1, 0, 2).reshape(mesh_row, n1)
        d1_token = d1.transpose(1, 0, 2).reshape(mesh_row, n1)

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


def shape_cfg(shape, names, k0, n0, k1, n1):
    name, array_shape, mesh_row, tile_size, mesh_col = shape
    a_channel = {0: 0xffff, 1: 0x00ff, 2: 0x000f}[array_shape]
    b_channel = {0: 0x03, 1: 0x0f, 2: 0x00ff}[array_shape]
    fields = [
        f".array_shape = {array_shape}",
        f".meshRow = {mesh_row}",
        f".tileSize = {tile_size}",
        f".meshCol = {mesh_col}",
        f".K0_tiles = {k0 // tile_size}",
        f".N0_tiles = {n0 // mesh_col}",
        f".K1_tiles = {k1 // tile_size}",
        f".N1_tiles = {n1 // mesh_col}",
        f".q_shape0_cols = {mesh_col // 4}",
        f".A_channel_en = {c_array([a_channel])}",
        f".B_channel_en = {c_array([b_channel])}",
        f".D_channel_en = {c_array([1])}",
        f".mode0_token_golden = {names[name][0]}",
        f".mode1_d0_token_golden = {names[name][1]}",
        f".mode1_d1_token_golden = {names[name][2]}",
    ]
    return "    {\n        " + ",\n        ".join(fields) + "\n    }"


def emit_header(params, hwcfg):
    m_total = int(params["M_total"])
    k0 = int(params["K0_total"])
    n0 = int(params["N0_total"])
    k1 = int(params["K1_total"])
    n1 = int(params["N1_total"])
    assert (m_total, k0, n0, k1, n1) == (8, 2048, 1024, 1024, 1024)

    acc = hwcfg["snax_dual_versacore_int16x4_core_template"]["snax_acc_cfg"][0]
    assert acc["sparse_interconnect_config"] == [[16, 2], [8, 2], [8, 2], [1, 1], [1, 1]]
    streamer = hwcfg["snax_dual_versacore_int16x4_streamer_template"]
    assert streamer["data_reader_params"]["spatial_bounds"] == [[2, 8], [2, 4], [2, 4]]
    assert streamer["data_writer_params"]["spatial_bounds"] == [[1], [1]]
    assert streamer["data_reader_params"]["temporal_dim"] == [3, 3, 3]
    assert streamer["data_writer_params"]["temporal_dim"] == [3, 3]

    a = make_a(m_total, k0)
    w = make_sparse_s0_weights(n0 // 4, k0 // 8, seed=1, sign=1)
    v = make_sparse_s0_weights(n0 // 4, k0 // 8, seed=3, sign=1)
    w2l = make_sparse_s0_weights(n1 // 4, k1 // 8, seed=5, sign=1)
    w2r = make_sparse_s0_weights(n1 // 4, k1 // 8, seed=7, sign=-1)
    arrays = [
        emit_i16("A", a.reshape(-1)),
        emit_u8("W", pack_int4(w)),
        emit_u8("V", pack_int4(v)),
        emit_u8("W2_left", pack_int4(w2l)),
        emit_u8("W2_right", pack_int4(w2r)),
    ]
    golden_arrays, names = make_goldens(a, w, v, w2l, w2r, k0, n0, k1, n1)
    arrays.extend(golden_arrays)
    cfgs = ",\n".join(shape_cfg(s, names, k0, n0, k1, n1) for s in SHAPES)

    return "\n".join([
        "#include <stdint.h>",
        "",
        "#ifndef SNAX_VC_K8_COMMON16_CHUNKED_DATA_H",
        "#define SNAX_VC_K8_COMMON16_CHUNKED_DATA_H",
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
        f"#define MODE0_WEIGHT_DATA_LENGTH {pack_int4(w).nbytes}",
        f"#define MODE1_WEIGHT_DATA_LENGTH {pack_int4(w2l).nbytes}",
        "#define TCDM_CAPACITY_BYTES (8192 * 1024)",
        "",
        "typedef struct {",
        "    int32_t array_shape, meshRow, tileSize, meshCol;",
        "    int32_t K0_tiles, N0_tiles, K1_tiles, N1_tiles;",
        "    int32_t q_shape0_cols;",
        "    int32_t A_channel_en[1], B_channel_en[1], D_channel_en[1];",
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
    with open(args.hwcfg, "r", encoding="utf-8") as handle:
        hwcfg = hjson.load(handle)
    print(emit_header(params, hwcfg))


if __name__ == "__main__":
    main()
