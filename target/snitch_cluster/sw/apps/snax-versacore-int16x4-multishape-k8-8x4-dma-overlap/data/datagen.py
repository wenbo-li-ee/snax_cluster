#!/usr/bin/env python3

import argparse
import hjson
import os
import pathlib
import sys

import numpy as np

_this_dir = os.path.dirname(os.path.realpath(__file__))
_silu_pkg = os.path.realpath(os.path.join(_this_dir, "../../../../../../util/silu_pkg"))
if os.path.isdir(_silu_pkg):
    sys.path.insert(0, _silu_pkg)
else:
    sys.path.insert(0, "/esat/studscratch/r1015498/Thesis/original_snax/silu/pkg")
from silu_out16_balanced_golden import silu_out16_balanced_eval_q  # noqa: E402


SHAPE_DIMS = [
    ("S0", 0, 8, 8, 4),
    ("S1", 1, 4, 8, 8),
    ("S2", 2, 2, 8, 16),
]

A_SPATIAL_BOUNDS = [2, 8]
B_SPATIAL_BOUNDS = [2, 4]


def c_array(values):
    return "{ " + ", ".join(str(int(v)) for v in values) + " }"


def c_u8_array(values):
    return "{ " + ", ".join(f"0x{int(v) & 0xff:02x}" for v in values) + " }"


def rescale_down_32to16(arr_int32, input_zp=0, mult=1, output_zp=0, shift=0):
    result = arr_int32.astype(np.int64) - int(input_zp)
    multiplied = result * np.int64(mult)
    if shift > 0:
        shifted_one = np.int64(1) << (shift - 1)
        shifted_data = multiplied + shifted_one
        scaled_32 = np.where(result >= 0,
                             shifted_data + np.int64(1 << 30),
                             shifted_data - np.int64(1 << 30))
        correct_shift = np.where(shift > 31, scaled_32, shifted_data)
        shifted_value = correct_shift >> shift
    else:
        shifted_value = multiplied
    out = shifted_value.astype(np.int32).astype(np.int64) + int(output_zp)
    return np.clip(out, -32768, 32767).astype(np.int16)


def apply_silu_vectorized(arr_int16):
    flat = arr_int16.flatten()
    result = np.array([silu_out16_balanced_eval_q(int(x)) for x in flat], dtype=np.int16)
    return result.reshape(arr_int16.shape)


def block_gemm_int16x4(M, K, N, meshRow, tileSize, meshCol, A_flat, B_flat):
    a = A_flat.astype(np.int32).reshape(M, K, meshRow, tileSize)
    b = B_flat.astype(np.int32).reshape(N, K, meshCol, tileSize)
    d = np.zeros((M, N, meshRow, meshCol), dtype=np.int32)
    for mm in range(M):
        for nn in range(N):
            d[mm, nn] = np.tensordot(a[mm], b[nn], axes=([0, 2], [0, 2]))
    return d.reshape(-1)


def make_physical_a(m_total, k_total):
    data = np.zeros((m_total, k_total), dtype=np.int16)
    for m in range(m_total):
        for k in range(k_total):
            data[m, k] = ((m * 5 + k * 3) % 11) - 5
    return data


def spatial_offsets(bounds, strides):
    out = []
    for i in range(np.prod(bounds)):
        rem = i
        off = 0
        for bound, stride in zip(bounds, strides):
            off += (rem % bound) * stride
            rem //= bound
        out.append(off)
    return out


def streamer_i16_flat(source_i16, m_tiles, k_bound, spatial_bounds, spatial_strides,
                      k_stride, m_stride, channel_en):
    flat = source_i16.reshape(-1)
    offsets = [off for i, off in enumerate(spatial_offsets(spatial_bounds, spatial_strides))
               if (channel_en >> i) & 1]
    out = []
    for mt in range(m_tiles):
        for kt in range(k_bound):
            base = mt * m_stride + kt * k_stride
            for off in offsets:
                byte_addr = base + off
                assert byte_addr % 2 == 0
                elem = byte_addr // 2
                assert 0 <= elem and elem + 4 <= len(flat), (elem, len(flat), byte_addr)
                out.extend(flat[elem:elem + 4])
    return np.array(out, dtype=np.int16)


def packed_int4_constant(num_shape0_tiles, value):
    # Tight S0 layout: one [tileSize=8, meshCol=4] int4 tile is 32 nibbles = 16 B.
    assert 0 <= value <= 7
    packed_byte = (value << 4) | value
    return np.full(num_shape0_tiles * 16, packed_byte, dtype=np.uint8)


def align_up(value, alignment):
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def avoid_same_bank(offset, reference, alignment=16, bank_word_bytes=8, banks=64):
    out = align_up(offset, alignment)
    ref_bank = (reference // bank_word_bytes) % banks
    while (out // bank_word_bytes) % banks == ref_bank:
        out = align_up(out + bank_word_bytes, alignment)
    return out


def emit_i16_array(name, values):
    return f"static const int16_t {name}[{len(values)}] = {c_array(values)};"


def emit_u8_array(name, values):
    return f"static const uint8_t {name}[{len(values)}] = {c_u8_array(values)};"


def build_shape_cfg(shape, globals_, golden_names):
    name, array_shape, mesh_row, tile_size, mesh_col = shape
    k0_total = globals_["k0_total"]
    n0_total = globals_["n0_total"]
    k1_total = globals_["k1_total"]
    n1_total = globals_["n1_total"]
    k0_s0_tiles = globals_["k0_s0_tiles"]
    k1_s0_tiles = globals_["k1_s0_tiles"]
    n0_s0_tiles = globals_["n0_s0_tiles"]
    n1_s0_tiles = globals_["n1_s0_tiles"]
    k0_bytes = globals_["k0_bytes"]

    m_tiles = 1
    k_tiles = k0_total // tile_size
    k1_tiles = k1_total // tile_size
    n0_tiles = n0_total // mesh_col
    n1_tiles = n1_total // mesh_col
    b_k_stride = 16
    mode0_b_n_stride = (mesh_col // 4) * k0_s0_tiles * 16
    mode1_b_n_stride = (mesh_col // 4) * k1_s0_tiles * 16
    a_channel_en = {0: 0xFFFF, 1: 0x00FF, 2: 0x000F}[array_shape]
    b_channel_en = {0: 0x03, 1: 0x0F, 2: 0xFF}[array_shape]
    d_bound0 = 8
    d_stride1 = 64
    mode0_d_m_stride = n0_tiles * d_stride1
    mode1_d_m_stride = n1_tiles * d_stride1

    # Mode1 D writer per-token layout parameters (spec s9)
    beats_per_row = mesh_col // 4
    mode1_token_stride = n1_total * 2  # = 2048 bytes per token

    a_bytes = 8 * k0_total * 2
    w_bytes = k0_s0_tiles * n0_s0_tiles * 16
    mode0_d_bytes = 8 * n0_total * 2
    w2_bytes = k1_s0_tiles * n1_s0_tiles * 16
    mode1_d_bytes = 8 * n1_total * 2
    delta_local_a = 0
    delta_local_b0 = align_up(delta_local_a + a_bytes, 1024)
    delta_local_b1 = align_up(delta_local_b0 + w_bytes, 1024)
    delta_local_d0 = align_up(delta_local_b1 + w_bytes, 1024)
    delta_local_w2l = align_up(delta_local_d0 + mode0_d_bytes, 1024)
    delta_local_w2r = align_up(delta_local_w2l + w2_bytes, 1024)
    delta_local_mode1_d0 = align_up(delta_local_w2r + w2_bytes, 1024)
    delta_local_mode1_d1 = avoid_same_bank(delta_local_mode1_d0 + mode1_d_bytes + 8,
                                           delta_local_mode1_d0)
    tcdm_end = delta_local_mode1_d1 + mode1_d_bytes

    mode1_a_sstride = {
        0: [64, 8],
        1: [8, 16],
        2: [8, 32],
    }[array_shape]
    mode1_a_k_stride = {0: 128, 1: 64, 2: 16}[array_shape]

    fields = [
        f".array_shape = {array_shape}",
        f".meshRow = {mesh_row}",
        f".tileSize = {tile_size}",
        f".meshCol = {mesh_col}",
        f".tokens_used = {mesh_row}",
        f".M_tiles = {m_tiles}",
        f".K_tiles = {k_tiles}",
        f".N_tiles = {n0_tiles}",
        f".K1 = {k1_tiles}",
        f".N1 = {n1_tiles}",
        f".mode0_A_sstride = {c_array([8, k0_bytes])}",
        f".mode1_A_sstride = {c_array(mode1_a_sstride)}",
        f".mode0_B_sstride = {c_array([8, 4096])}",
        f".mode1_B_sstride = {c_array([8, 2816])}",
        f".D_sstride = {c_array([8])}",
        f".mode0_A_tbound = {c_array([k_tiles, n0_tiles, m_tiles, 1, 1, 1])}",
        f".mode0_A_tstride = {c_array([tile_size * 2, 0, mesh_row * k0_bytes, 0, 0, 0])}",
        f".mode1_A_tbound = {c_array([k1_tiles, n1_tiles, m_tiles, 1, 1, 1])}",
        f".mode1_A_tstride = {c_array([mode1_a_k_stride, 0, mode0_d_m_stride, 0, 0, 0])}",
        f".mode0_B_tbound = {c_array([k_tiles, n0_tiles, m_tiles, 1])}",
        f".mode0_B_tstride = {c_array([b_k_stride, mode0_b_n_stride, 0, 0])}",
        f".mode1_B_tbound = {c_array([k1_tiles, n1_tiles, m_tiles, 1])}",
        f".mode1_B_tstride = {c_array([b_k_stride, mode1_b_n_stride, 0, 0])}",
        f".mode0_D_tbound = {c_array([d_bound0, n0_tiles, m_tiles, 1])}",
        f".mode0_D_tstride = {c_array([8, d_stride1, mode0_d_m_stride, 0])}",
        f".mode1_D_tbound = {c_array([beats_per_row, mesh_row, n1_tiles, m_tiles])}",
        f".mode1_D_tstride = {c_array([8, mode1_token_stride, mesh_col * 2, 0])}",
        f".A_channel_en = {c_array([a_channel_en])}",
        f".B_channel_en = {c_array([b_channel_en])}",
        f".D_channel_en = {c_array([0x01])}",
        f".delta_local_a = {delta_local_a}",
        f".delta_local_b0 = {delta_local_b0}",
        f".delta_local_b1 = {delta_local_b1}",
        f".delta_local_d0 = {delta_local_d0}",
        f".delta_local_w2l = {delta_local_w2l}",
        f".delta_local_w2r = {delta_local_w2r}",
        f".delta_local_mode1_d0 = {delta_local_mode1_d0}",
        f".delta_local_mode1_d1 = {delta_local_mode1_d1}",
        f".tcdm_end = {tcdm_end}",
        f".mode0_output_elems = {mesh_row * n0_total}",
        f".mode1_output_elems = {mesh_row * n1_total}",
        f".mode0_d0_golden = {golden_names[name][0]}",
        f".mode1_d0_golden = {golden_names[name][1]}",
        f".mode1_d1_golden = {golden_names[name][2]}",
    ]
    return "    {\n        " + ",\n        ".join(fields) + "\n    }"


def emit_header(params):
    m_total = int(params["M_total"])
    k0_total = int(params["K0_total"])
    n0_total = int(params["N0_total"])
    k1_total = int(params["K1_total"])
    n1_total = int(params["N1_total"])
    assert m_total == 8
    assert k0_total == 2048
    assert n0_total == 1408
    assert k1_total == 1408
    assert n1_total == 1024

    k0_s0_tiles = k0_total // 8
    k1_s0_tiles = k1_total // 8
    n0_s0_tiles = n0_total // 4
    n1_s0_tiles = n1_total // 4
    k0_bytes = k0_total * 2
    physical_a = make_physical_a(m_total, k0_total)

    globals_ = {
        "k0_total": k0_total,
        "n0_total": n0_total,
        "k1_total": k1_total,
        "n1_total": n1_total,
        "k0_s0_tiles": k0_s0_tiles,
        "k1_s0_tiles": k1_s0_tiles,
        "n0_s0_tiles": n0_s0_tiles,
        "n1_s0_tiles": n1_s0_tiles,
        "k0_bytes": k0_bytes,
    }

    arrays = []
    arrays.append(emit_i16_array("A", physical_a.reshape(-1)))
    # Distinct per-tensor constants keep the tight S0 physical layout simple while
    # making B0/B1 and Mode1 left/right swaps visible in the goldens.
    arrays.append(emit_u8_array("W", packed_int4_constant(k0_s0_tiles * n0_s0_tiles, 1)))
    arrays.append(emit_u8_array("V", packed_int4_constant(k0_s0_tiles * n0_s0_tiles, 2)))
    arrays.append(emit_u8_array("W2_left", packed_int4_constant(k1_s0_tiles * n1_s0_tiles, 1)))
    arrays.append(emit_u8_array("W2_right", packed_int4_constant(k1_s0_tiles * n1_s0_tiles, 2)))

    golden_names = {}
    for shape in SHAPE_DIMS:
        name, array_shape, mesh_row, tile_size, mesh_col = shape
        m_shape_tiles = 1
        k_shape_tiles = k0_total // tile_size
        n_shape_tiles = n0_total // mesh_col
        a_channel_en = {0: 0xFFFF, 1: 0x00FF, 2: 0x000F}[array_shape]
        a_flat = streamer_i16_flat(
            physical_a, m_shape_tiles, k_shape_tiles, A_SPATIAL_BOUNDS,
            [8, k0_bytes], tile_size * 2, mesh_row * k0_bytes, a_channel_en)
        b0_flat = np.full(n_shape_tiles * k_shape_tiles * mesh_col * tile_size,
                          1, dtype=np.int8)
        b1_flat = np.full(n_shape_tiles * k_shape_tiles * mesh_col * tile_size,
                          2, dtype=np.int8)
        vc0 = block_gemm_int16x4(
            m_shape_tiles, k_shape_tiles, n_shape_tiles,
            mesh_row, tile_size, mesh_col, a_flat, b0_flat)
        vc1 = block_gemm_int16x4(
            m_shape_tiles, k_shape_tiles, n_shape_tiles,
            mesh_row, tile_size, mesh_col, a_flat, b1_flat)
        vc0_i16 = rescale_down_32to16(vc0)
        vc0_silu = apply_silu_vectorized(vc0_i16)
        vc1_i16 = rescale_down_32to16(vc1)
        mode0 = rescale_down_32to16(vc0_silu.astype(np.int32) * vc1_i16.astype(np.int32))

        k1_tiles = k1_total // tile_size
        n1_tiles_shape = n1_total // mesh_col
        w2_left_flat = np.full(n1_tiles_shape * k1_tiles * mesh_col * tile_size,
                               1, dtype=np.int8)
        w2_right_flat = np.full(n1_tiles_shape * k1_tiles * mesh_col * tile_size,
                                2, dtype=np.int8)
        d_stride1 = 64
        mode0_d_m_stride = n_shape_tiles * d_stride1
        mode1_a_sstride = {
            0: [64, 8],
            1: [8, 16],
            2: [8, 32],
        }[array_shape]
        mode1_a_k_stride = {0: 128, 1: 64, 2: 16}[array_shape]
        mode1_a_flat = streamer_i16_flat(
            mode0, m_shape_tiles, k1_tiles, A_SPATIAL_BOUNDS, mode1_a_sstride,
            mode1_a_k_stride, mode0_d_m_stride, a_channel_en)
        mode1_d0 = rescale_down_32to16(block_gemm_int16x4(
            m_shape_tiles, k1_tiles, n1_tiles_shape, mesh_row, tile_size, mesh_col,
            mode1_a_flat, w2_left_flat))
        mode1_d1 = rescale_down_32to16(block_gemm_int16x4(
            m_shape_tiles, k1_tiles, n1_tiles_shape, mesh_row, tile_size, mesh_col,
            mode1_a_flat, w2_right_flat))

        names = (
            f"{name}_mode0_d0_golden",
            f"{name}_mode1_d0_golden",
            f"{name}_mode1_d1_golden",
        )
        golden_names[name] = names
        arrays.append(emit_i16_array(names[0], mode0))
        # Mode1 D writer uses per-token layout: transpose (M_tiles, N_tiles, meshRow, meshCol)
        # to (M_tiles, meshRow, N_tiles, meshCol) so token k's outputs are contiguous.
        mode1_d0_pertoken = mode1_d0.reshape(
            m_shape_tiles, n1_tiles_shape, mesh_row, mesh_col
        ).transpose(0, 2, 1, 3).reshape(-1)
        mode1_d1_pertoken = mode1_d1.reshape(
            m_shape_tiles, n1_tiles_shape, mesh_row, mesh_col
        ).transpose(0, 2, 1, 3).reshape(-1)
        arrays.append(emit_i16_array(names[1], mode1_d0_pertoken))
        arrays.append(emit_i16_array(names[2], mode1_d1_pertoken))

    shape_cfgs = ",\n".join(build_shape_cfg(shape, globals_, golden_names) for shape in SHAPE_DIMS)
    return "\n".join([
        "#include <stdint.h>",
        "",
        "#ifndef SNAX_VERSACORE_INT16X4_MULTISHAPE_K8_8X4_MODE1_PERTOKEN_DATA_H",
        "#define SNAX_VERSACORE_INT16X4_MULTISHAPE_K8_8X4_MODE1_PERTOKEN_DATA_H",
        "",
        "#define NUM_SHAPES 3",
        "#define DATA_TYPE 0",
        "#define RESCALE_INPUT_ZP 0",
        "#define RESCALE_MULTIPLIER 1",
        "#define RESCALE_OUTPUT_ZP 0",
        "#define RESCALE_SHIFT 0",
        "#define SUBTRACTION_A 0",
        "#define SUBTRACTION_B 0",
        f"#define A_DATA_LENGTH {m_total * k0_total * 2}",
        f"#define B_DATA_LENGTH {k0_s0_tiles * n0_s0_tiles * 16}",
        f"#define W2_DATA_LENGTH {k1_s0_tiles * n1_s0_tiles * 16}",
        "#define TCDM_CAPACITY_BYTES (8192 * 1024)",
        "#define SET_ADDR_REMAP_INDEX_A 0",
        "#define SET_ADDR_REMAP_INDEX_B0 0",
        "#define SET_ADDR_REMAP_INDEX_B1 0",
        "#define SET_ADDR_REMAP_INDEX_D0 0",
        "#define SET_ADDR_REMAP_INDEX_D1 0",
        "",
        "typedef struct {",
        "    uint32_t array_shape, meshRow, tileSize, meshCol, tokens_used;",
        "    uint32_t M_tiles, K_tiles, N_tiles, K1, N1;",
        "    int32_t mode0_A_sstride[2], mode1_A_sstride[2];",
        "    int32_t mode0_B_sstride[2], mode1_B_sstride[2], D_sstride[1];",
        "    int32_t mode0_A_tbound[6], mode0_A_tstride[6], mode1_A_tbound[6], mode1_A_tstride[6];",
        "    int32_t mode0_B_tbound[4], mode0_B_tstride[4], mode1_B_tbound[4], mode1_B_tstride[4];",
        "    int32_t mode0_D_tbound[4], mode0_D_tstride[4], mode1_D_tbound[4], mode1_D_tstride[4];",
        "    int32_t A_channel_en[1], B_channel_en[1], D_channel_en[1];",
        "    int32_t delta_local_a, delta_local_b0, delta_local_b1, delta_local_d0;",
        "    int32_t delta_local_w2l, delta_local_w2r, delta_local_mode1_d0, delta_local_mode1_d1;",
        "    int32_t tcdm_end, mode0_output_elems, mode1_output_elems;",
        "    const int16_t *mode0_d0_golden, *mode1_d0_golden, *mode1_d1_golden;",
        "} shape_cfg_t;",
        "",
        *arrays,
        "",
        "static const shape_cfg_t shape_cfg[NUM_SHAPES] = {",
        shape_cfgs,
        "};",
        "",
        "#endif",
        "",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swcfg", type=pathlib.Path, required=True)
    parser.add_argument("--hwcfg", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with args.swcfg.open() as f:
        params = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hjson.loads(f.read())
    print(emit_header(params))


if __name__ == "__main__":
    main()
