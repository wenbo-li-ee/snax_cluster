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
    ("S1", 1, 4, 16, 4),
    ("S2", 2, 2, 32, 4),
]


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
            data[m, k] = ((m * 3 + k * 2) % 7) - 3
    return data


def shape_a_flat(physical_a, m_tiles, k_tiles, mesh_row, tile_size):
    out = []
    for mt in range(m_tiles):
        for kt in range(k_tiles):
            for r in range(mesh_row):
                for kk in range(tile_size):
                    out.append(physical_a[mt * mesh_row + r, kt * tile_size + kk])
    return np.array(out, dtype=np.int16)


def spatial_offsets(bounds, strides):
    if len(bounds) == 3:
        out = []
        for i2 in range(bounds[2]):
            for i1 in range(bounds[1]):
                for i0 in range(bounds[0]):
                    out.append(i0 * strides[0] + i1 * strides[1] + i2 * strides[2])
        return out
    if len(bounds) == 2:
        out = []
        for i1 in range(bounds[1]):
            for i0 in range(bounds[0]):
                out.append(i0 * strides[0] + i1 * strides[1])
        return out
    raise ValueError(f"unsupported spatial rank {len(bounds)}")


def streamer_i16_flat(source_i16, m_tiles, k_bound, spatial_bounds, spatial_strides,
                      k_stride, m_stride):
    flat = source_i16.reshape(-1)
    offsets = spatial_offsets(spatial_bounds, spatial_strides)
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


def packed_ones(num_tiles):
    # One S0 int4 tile has a 64B streamer footprint. Filling every nibble with
    # 1 makes B layout/order irrelevant while still exercising non-zero math.
    return np.full(num_tiles * 64, 0x11, dtype=np.uint8)


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
    k_total = globals_["k_total"]
    n_total = globals_["n_total"]
    k_s0_tiles = globals_["k_s0_tiles"]
    n_tiles = globals_["n_tiles"]
    k_bytes = globals_["k_bytes"]

    m_tiles = 8 // mesh_row
    k_tiles = k_total // tile_size
    k1 = n_total // tile_size
    n1 = k_total // mesh_col
    b_k_stride = (tile_size // 8) * 64
    b_k_bound = k_tiles
    b_n_stride = k_s0_tiles * 64
    b_channel_en = {0: 0x03, 1: 0x0F, 2: 0xFF}[array_shape]
    d_bound0 = mesh_row * mesh_col // 4
    d_stride1 = d_bound0 * 8
    d_m_stride = n_tiles * d_stride1
    a_bytes = 8 * k_total * 2
    b_bytes = k_s0_tiles * n_tiles * 64
    mode0_d_bytes = 8 * n_total * 2
    w2_bytes = k_s0_tiles * (k_total // 4) * 64
    mode1_d_bytes = 8 * k_total * 2
    delta_local_a = 0
    delta_local_b0 = align_up(delta_local_a + a_bytes, 1024)
    delta_local_b1 = align_up(delta_local_b0 + b_bytes, 1024)
    delta_local_d0 = align_up(delta_local_b1 + b_bytes, 1024)
    delta_local_d1_mode0 = avoid_same_bank(delta_local_d0 + mode0_d_bytes + 8,
                                           delta_local_d0)
    delta_local_w2l = align_up(delta_local_d1_mode0 + mode0_d_bytes, 1024)
    delta_local_w2r = align_up(delta_local_w2l + w2_bytes, 1024)
    delta_local_mode1_d0 = align_up(delta_local_w2r + w2_bytes, 1024)
    delta_local_mode1_d1 = avoid_same_bank(delta_local_mode1_d0 + mode1_d_bytes + 8,
                                           delta_local_mode1_d0)
    mode0_a_sstride = {
        0: [8, k_bytes, 4 * k_bytes],
        1: [8, k_bytes, 16],
        2: [8, 16, k_bytes],
    }[array_shape]

    fields = [
        f".array_shape = {array_shape}",
        f".meshRow = {mesh_row}",
        f".tileSize = {tile_size}",
        f".meshCol = {mesh_col}",
        f".M_tiles = {m_tiles}",
        f".K_tiles = {k_tiles}",
        f".N_tiles = {n_tiles}",
        f".K1 = {k1}",
        f".N1 = {n1}",
        f".mode0_A_sstride = {c_array(mode0_a_sstride)}",
        f".mode1_A_sstride = {c_array([8, 16, 64])}",
        f".B_sstride = {c_array([8, 64])}",
        f".D_sstride = {c_array([8])}",
        f".mode0_A_tbound = {c_array([k_tiles, n_tiles, m_tiles, 1, 1, 1])}",
        f".mode0_A_tstride = {c_array([tile_size * 2, 0, mesh_row * k_bytes, 0, 0, 0])}",
        f".mode1_A_tbound = {c_array([k1, n1, m_tiles, 1, 1, 1])}",
        f".mode1_A_tstride = {c_array([128, 0, d_m_stride, 0, 0, 0])}",
        f".mode0_B_tbound = {c_array([b_k_bound, n_tiles, m_tiles, 1])}",
        f".mode0_B_tstride = {c_array([b_k_stride, b_n_stride, 0, 0])}",
        f".mode1_B_tbound = {c_array([k1, n1, m_tiles, 1])}",
        f".mode1_B_tstride = {c_array([b_k_stride, b_n_stride, 0, 0])}",
        f".mode0_D_tbound = {c_array([d_bound0, n_tiles, m_tiles, 1])}",
        f".mode0_D_tstride = {c_array([8, d_stride1, d_m_stride, 0])}",
        f".mode1_D_tbound = {c_array([d_bound0, n1, m_tiles, 1])}",
        f".mode1_D_tstride = {c_array([8, d_stride1, d_bound0 * 8 * n1, 0])}",
        f".A_channel_en = {c_array([0xFFFF])}",
        f".B_channel_en = {c_array([b_channel_en])}",
        f".D_channel_en = {c_array([0x01])}",
        f".delta_local_a = {delta_local_a}",
        f".delta_local_b0 = {delta_local_b0}",
        f".delta_local_b1 = {delta_local_b1}",
        f".delta_local_d0 = {delta_local_d0}",
        f".delta_local_d1_mode0 = {delta_local_d1_mode0}",
        f".delta_local_w2l = {delta_local_w2l}",
        f".delta_local_w2r = {delta_local_w2r}",
        f".delta_local_mode1_d0 = {delta_local_mode1_d0}",
        f".delta_local_mode1_d1 = {delta_local_mode1_d1}",
        f".mode0_output_elems = {8 * n_total}",
        f".mode1_output_elems = {8 * k_total}",
        f".mode0_d0_golden = {golden_names[name][0]}",
        f".mode0_d1_golden = {golden_names[name][1]}",
        f".mode1_d0_golden = {golden_names[name][2]}",
        f".mode1_d1_golden = {golden_names[name][3]}",
    ]
    return "    {\n        " + ",\n        ".join(fields) + "\n    }"


def emit_header(params):
    m_tiles = int(params["M_tiles"])
    k_s0_tiles = int(params["K_tiles"])
    n_tiles = int(params["N_tiles"])
    assert m_tiles == 1
    assert k_s0_tiles % 4 == 0, "S2 needs K_total divisible by 32"
    assert n_tiles % 8 == 0, "S2 Mode1 needs N_total divisible by 32"

    m_total = m_tiles * 8
    k_total = k_s0_tiles * 8
    n_total = n_tiles * 4
    k_bytes = k_total * 2
    physical_a = make_physical_a(m_total, k_total)

    globals_ = {
        "k_total": k_total,
        "n_total": n_total,
        "k_s0_tiles": k_s0_tiles,
        "n_tiles": n_tiles,
        "k_bytes": k_bytes,
    }

    arrays = []
    arrays.append(emit_i16_array("A", physical_a.reshape(-1)))
    arrays.append(emit_u8_array("W", packed_ones(k_s0_tiles * n_tiles)))
    arrays.append(emit_u8_array("V", packed_ones(k_s0_tiles * n_tiles)))
    w2_padded_s0_k_tiles = k_s0_tiles
    arrays.append(emit_u8_array("W2_left", packed_ones(w2_padded_s0_k_tiles * (k_total // 4))))
    arrays.append(emit_u8_array("W2_right", packed_ones(w2_padded_s0_k_tiles * (k_total // 4))))

    golden_names = {}
    for shape in SHAPE_DIMS:
        name, _, mesh_row, tile_size, mesh_col = shape
        array_shape = shape[1]
        m_shape_tiles = m_total // mesh_row
        k_shape_tiles = k_total // tile_size
        n_shape_tiles = n_total // mesh_col
        mode0_a_sstride = {
            0: [8, k_bytes, 4 * k_bytes],
            1: [8, k_bytes, 16],
            2: [8, 16, k_bytes],
        }[array_shape]
        a_flat = streamer_i16_flat(
            physical_a, m_shape_tiles, k_shape_tiles, [2, 4, 2], mode0_a_sstride,
            tile_size * 2, mesh_row * k_bytes)
        b_flat = np.ones(n_shape_tiles * k_shape_tiles * mesh_col * tile_size, dtype=np.int8)
        vc0 = block_gemm_int16x4(
            m_shape_tiles, k_shape_tiles, n_shape_tiles,
            mesh_row, tile_size, mesh_col, a_flat, b_flat)
        vc1 = block_gemm_int16x4(
            m_shape_tiles, k_shape_tiles, n_shape_tiles,
            mesh_row, tile_size, mesh_col, a_flat, b_flat)
        vc0_i16 = rescale_down_32to16(vc0)
        vc0_silu = apply_silu_vectorized(vc0_i16)
        vc1_i16 = rescale_down_32to16(vc1)
        mode0 = rescale_down_32to16(vc0_silu.astype(np.int32) * vc1_i16.astype(np.int32))

        k1 = n_total // tile_size
        n1 = k_total // mesh_col
        w2_flat = np.ones(n1 * k1 * mesh_col * tile_size, dtype=np.int8)
        d_bound0 = mesh_row * mesh_col // 4
        d_stride1 = d_bound0 * 8
        d_m_stride = n_tiles * d_stride1
        mode1_a_flat = streamer_i16_flat(
            mode0, m_shape_tiles, k1, [2, 4, 2], [8, 16, 64],
            128, d_m_stride)
        mode1_d0 = rescale_down_32to16(block_gemm_int16x4(
            m_shape_tiles, k1, n1, mesh_row, tile_size, mesh_col,
            mode1_a_flat, w2_flat))
        mode1_d1 = mode1_d0.copy()

        names = (
            f"{name}_mode0_d0_golden",
            f"{name}_mode0_d1_golden",
            f"{name}_mode1_d0_golden",
            f"{name}_mode1_d1_golden",
        )
        golden_names[name] = names
        arrays.append(emit_i16_array(names[0], mode0))
        arrays.append(emit_i16_array(names[1], mode0))
        arrays.append(emit_i16_array(names[2], mode1_d0))
        arrays.append(emit_i16_array(names[3], mode1_d1))

    shape_cfgs = ",\n".join(build_shape_cfg(shape, globals_, golden_names) for shape in SHAPE_DIMS)
    return "\n".join([
        "#include <stdint.h>",
        "",
        "#ifndef SNAX_VERSACORE_INT16X4_MULTISHAPE_BATCH_DATA_H",
        "#define SNAX_VERSACORE_INT16X4_MULTISHAPE_BATCH_DATA_H",
        "",
        "#define NUM_SHAPES 3",
        "#define DATA_TYPE 0",
        "#define RESCALE_INPUT_ZP 0",
        "#define RESCALE_MULTIPLIER 1",
        "#define RESCALE_OUTPUT_ZP 0",
        "#define RESCALE_SHIFT 0",
        "#define SUBTRACTION_A 0",
        "#define SUBTRACTION_B 0",
        f"#define A_DATA_LENGTH {m_total * k_total * 2}",
        f"#define B_DATA_LENGTH {k_s0_tiles * n_tiles * 64}",
        f"#define W2_DATA_LENGTH {w2_padded_s0_k_tiles * (k_total // 4) * 64}",
        "#define SET_ADDR_REMAP_INDEX_A 0",
        "#define SET_ADDR_REMAP_INDEX_B0 0",
        "#define SET_ADDR_REMAP_INDEX_B1 0",
        "#define SET_ADDR_REMAP_INDEX_D0 0",
        "#define SET_ADDR_REMAP_INDEX_D1 0",
        "",
        "typedef struct {",
        "    uint32_t array_shape, meshRow, tileSize, meshCol;",
        "    uint32_t M_tiles, K_tiles, N_tiles, K1, N1;",
        "    int32_t mode0_A_sstride[3], mode1_A_sstride[3], B_sstride[2], D_sstride[1];",
        "    int32_t mode0_A_tbound[6], mode0_A_tstride[6], mode1_A_tbound[6], mode1_A_tstride[6];",
        "    int32_t mode0_B_tbound[4], mode0_B_tstride[4], mode1_B_tbound[4], mode1_B_tstride[4];",
        "    int32_t mode0_D_tbound[4], mode0_D_tstride[4], mode1_D_tbound[4], mode1_D_tstride[4];",
        "    int32_t A_channel_en[1], B_channel_en[1], D_channel_en[1];",
        "    int32_t delta_local_a, delta_local_b0, delta_local_b1, delta_local_d0, delta_local_d1_mode0;",
        "    int32_t delta_local_w2l, delta_local_w2r, delta_local_mode1_d0, delta_local_mode1_d1;",
        "    int32_t mode0_output_elems, mode1_output_elems;",
        "    const int16_t *mode0_d0_golden, *mode0_d1_golden, *mode1_d0_golden, *mode1_d1_golden;",
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
