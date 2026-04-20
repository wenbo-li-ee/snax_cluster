#!/usr/bin/env python3

# Data generator for dual VersaCore int16x4 scaled 1/16 batch test
# Mode 0 (SwiGLU): A[8,128] @ W[128,88], V[128,88] → Output0[8,88]
# Mode 1 (GEMM): A'=Output0[8,88] @ W2_left[88,64], W2_right[88,64]
# Mode 1 A input = Mode 0 output (no separate A1 data)

import numpy as np
import argparse
import pathlib
import hjson
import sys
import os
import math

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/"))
from data_utils import format_scalar_definition, format_vector_definition  # noqa E402
from snax_utils import (  # noqa E402
    align_wide_addr,
)  # noqa E402

np.random.seed(42)


def rescale_down_32to16(arr_int32, input_zp, mult, output_zp, shift):
    """Golden model matching RescaleDownPE hardware logic (in=32, out=16)."""
    result = arr_int32.astype(np.int64)
    result = result - int(input_zp)
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


def arithmetic_right_shift_int16(arr_int16, n):
    """SiLU placeholder: arithmetic right shift by n on int16."""
    return (arr_int16.astype(np.int32) >> n).clip(-32768, 32767).astype(np.int16)


def block_gemm_int16x4(M, K, N, meshRow, tileSize, meshCol, A_flat, B_flat,
                        subtraction_a, subtraction_b):
    """
    Block GEMM golden model for int16 x int4 -> int32.
    A: flat array, [M, K, meshRow, tileSize] as int16
    B: flat array, [N, K, meshCol, tileSize] as int4
    Output: [M, N, meshRow, meshCol] as int32
    """
    a = A_flat.astype(np.int32)
    b = B_flat.astype(np.int32)
    a_sub = a.reshape(M, K, meshRow, tileSize) - subtraction_a
    b_sub = b.reshape(N, K, meshCol, tileSize) - subtraction_b

    d = np.zeros((M, N, meshRow, meshCol), dtype=np.int32)
    for mm in range(M):
        for nn in range(N):
            d[mm, nn] = np.tensordot(
                a_sub[mm], b_sub[nn], axes=([0, 2], [0, 2])
            )
    return d.reshape(-1)


def pack_int4(values):
    """Pack int4 values (in range [-8, 7]) into nibble-packed bytes."""
    values = np.array(values, dtype=np.int8)
    assert len(values) % 2 == 0, "int4 array must have even length"
    packed = np.zeros(len(values) // 2, dtype=np.uint8)
    for i in range(0, len(values), 2):
        lo = values[i] & 0x0F
        hi = values[i+1] & 0x0F
        packed[i // 2] = (hi << 4) | lo
    return packed


def pad_b_tiles(packed_bytes, num_tiles, raw_tile_bytes, padded_tile_bytes):
    """Pad each B tile from raw_tile_bytes to padded_tile_bytes."""
    if raw_tile_bytes == padded_tile_bytes:
        return packed_bytes
    result = np.zeros(num_tiles * padded_tile_bytes, dtype=np.uint8)
    for t in range(num_tiles):
        src_start = t * raw_tile_bytes
        dst_start = t * padded_tile_bytes
        result[dst_start:dst_start + raw_tile_bytes] = \
            packed_bytes[src_start:src_start + raw_tile_bytes]
    return result


def gen_channel_enable_CSR(channel_en_CSR, channel_en_bits):
    for i in range(channel_en_bits):
        element_index = i // 32
        bit_position = i % 32
        if element_index < len(channel_en_CSR):
            channel_en_CSR[element_index] |= 1 << (bit_position)
    channel_en_CSR = [int(x) for x in channel_en_CSR][::-1]
    return channel_en_CSR


def emit_header_file(**kwargs):
    emit_str = "#include <stdint.h>\n\n"
    emit_str += emit_dual_versacore_data(**kwargs)
    return emit_str


def emit_dual_versacore_data(**kwargs):
    data_str = []

    M = kwargs["M"]
    K = kwargs["K"]
    N = kwargs["N"]

    # Hardware parameters needed for auto-computing Mode 1 dimensions
    snax_acc_cfg_tmp = kwargs["snax_dual_versacore_int16x4_core_template"]["snax_acc_cfg"][0]
    array_shape_tmp = kwargs["array_shape"]
    data_type_tmp = kwargs["data_type"]
    meshCol_tmp = snax_acc_cfg_tmp["snax_versacore_spatial_unrolling"][data_type_tmp][array_shape_tmp][2]
    tileSize_tmp = snax_acc_cfg_tmp["snax_versacore_spatial_unrolling"][data_type_tmp][array_shape_tmp][1]

    # Auto-compute Mode 1 dimensions for chained SwiGLU
    M1 = M
    K1 = N * meshCol_tmp // tileSize_tmp
    N1 = K * tileSize_tmp // meshCol_tmp
    assert K1 * tileSize_tmp == N * meshCol_tmp, \
        f"K1 constraint failed: K1={K1}, N*meshCol={N*meshCol_tmp}, tileSize={tileSize_tmp}"
    assert N1 * meshCol_tmp == K * tileSize_tmp, \
        f"N1 constraint failed: N1={N1}, K*tileSize={K*tileSize_tmp}, meshCol={meshCol_tmp}"

    data_str += [format_scalar_definition("uint32_t", "M", M)]
    data_str += [format_scalar_definition("uint32_t", "K", K)]
    data_str += [format_scalar_definition("uint32_t", "N", N)]
    data_str += [format_scalar_definition("uint32_t", "M1", M1)]
    data_str += [format_scalar_definition("uint32_t", "K1", K1)]
    data_str += [format_scalar_definition("uint32_t", "N1", N1)]

    array_shape = kwargs["array_shape"]
    data_str += [format_scalar_definition("uint32_t", "array_shape", array_shape)]
    data_type = kwargs["data_type"]
    data_str += [format_scalar_definition("uint32_t", "data_type", data_type)]

    # Hardware parameters
    snax_acc_cfg = kwargs["snax_dual_versacore_int16x4_core_template"]["snax_acc_cfg"][0]
    meshRow = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][0]
    tileSize = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][1]
    meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][2]

    a_len = snax_acc_cfg["snax_versacore_input_a_element_width"][data_type]  # 16
    b_len = snax_acc_cfg["snax_versacore_input_b_element_width"][data_type]  # 4

    a_array_width = snax_acc_cfg["snax_versacore_array_input_a_width"]
    b_array_width = snax_acc_cfg["snax_versacore_array_input_b_width"]
    snax_versacore_serial_c_d_width = snax_acc_cfg["snax_versacore_serial_c_d_width"]

    bankWidth = 64
    out_elem_bits = 16

    granularity_a = snax_acc_cfg.get("granularity_a", 1)
    granularity_b = snax_acc_cfg.get("granularity_b", 1)
    granularity_c_d = snax_acc_cfg.get("granularity_c_d", 1)

    data_str += [format_scalar_definition("uint32_t", "meshRow", meshRow)]
    data_str += [format_scalar_definition("uint32_t", "tileSize", tileSize)]
    data_str += [format_scalar_definition("uint32_t", "meshCol", meshCol)]

    stationary = kwargs["stationary"]
    assert stationary == 0

    # Identity rescale parameters
    rescale_input_zp = 0
    rescale_multiplier = 1
    rescale_output_zp = 0
    rescale_shift = 0

    data_str += [format_scalar_definition("int32_t", "rescale_input_zp", rescale_input_zp)]
    data_str += [format_scalar_definition("uint32_t", "rescale_multiplier", rescale_multiplier)]
    data_str += [format_scalar_definition("int32_t", "rescale_output_zp", rescale_output_zp)]
    data_str += [format_scalar_definition("uint32_t", "rescale_shift", rescale_shift)]

    A_MIN, A_MAX = -3, 3
    B_MIN, B_MAX = -3, 3

    # ===================== B tile padding =====================================
    b_tile_raw = b_len * tileSize * meshCol // 8
    b_bits_needed = meshCol * tileSize * b_len
    B_enabled_channel_CSR_num = int(math.ceil(b_array_width // bankWidth / 32))
    channel_en_B0_bits = int((b_bits_needed // bankWidth + 7) // 8 * 8)
    if channel_en_B0_bits == 0:
        channel_en_B0_bits = 8
    b_channel_footprint_bytes = channel_en_B0_bits * (bankWidth // 8)
    b_tile_padded = max(b_tile_raw, b_channel_footprint_bytes)

    # ===================== A streamer settings (Reader 0) ====================
    data_str += [format_scalar_definition("int32_t", "Aslstride0", int(bankWidth // 8))]

    Atlbound0 = K
    Atlstride0 = int(a_len * tileSize * meshRow // 8)
    Atlbound1 = N
    Atlstride1 = 0
    Atlbound2 = M
    Atlstride2 = int(K * a_len * tileSize * meshRow // 8)

    data_str += [format_scalar_definition("int32_t", "Atlbound0", Atlbound0)]
    data_str += [format_scalar_definition("int32_t", "Atlstride0", Atlstride0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound1", Atlbound1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride1", Atlstride1)]
    data_str += [format_scalar_definition("int32_t", "Atlbound2", Atlbound2)]
    data_str += [format_scalar_definition("int32_t", "Atlstride2", Atlstride2)]
    data_str += [format_scalar_definition("int32_t", "Atlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride3", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound4", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride4", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound5", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride5", 0)]

    # Channel enable A
    A_enabled_channel_CSR_num = int(math.ceil(a_array_width // bankWidth / 32))
    channel_en_A = [0] * A_enabled_channel_CSR_num
    channel_en_A_bits = int(meshRow * tileSize * a_len // bankWidth)
    channel_en_A = gen_channel_enable_CSR(channel_en_A, channel_en_A_bits)
    data_str += [
        "int32_t channel_en_A[] = { " + ", ".join(map(str, channel_en_A)) + " };"
    ]

    a_data_length = M * K * meshRow * tileSize * a_len // 8
    data_str += [format_scalar_definition("int32_t", "a_data_length", a_data_length)]

    # ===================== B0 streamer settings (Reader 1) ====================
    data_str += [format_scalar_definition("int32_t", "B0slstride0", bankWidth // 8)]

    B0tlbound0 = K
    B0tlstride0 = b_tile_padded
    B0tlbound1 = N
    B0tlstride1 = K * b_tile_padded
    B0tlbound2 = M
    B0tlstride2 = 0

    data_str += [format_scalar_definition("int32_t", "B0tlbound0", B0tlbound0)]
    data_str += [format_scalar_definition("int32_t", "B0tlstride0", B0tlstride0)]
    data_str += [format_scalar_definition("int32_t", "B0tlbound1", B0tlbound1)]
    data_str += [format_scalar_definition("int32_t", "B0tlstride1", B0tlstride1)]
    data_str += [format_scalar_definition("int32_t", "B0tlbound2", B0tlbound2)]
    data_str += [format_scalar_definition("int32_t", "B0tlstride2", B0tlstride2)]
    data_str += [format_scalar_definition("int32_t", "B0tlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "B0tlstride3", 0)]

    channel_en_B0 = [0] * B_enabled_channel_CSR_num
    channel_en_B0 = gen_channel_enable_CSR(channel_en_B0, channel_en_B0_bits)
    data_str += [
        "int32_t channel_en_B0[] = { " + ", ".join(map(str, channel_en_B0)) + " };"
    ]

    b0_data_length = K * N * b_tile_padded
    data_str += [format_scalar_definition("int32_t", "b0_data_length", b0_data_length)]

    # ===================== B1 streamer settings (Reader 2) ====================
    data_str += [format_scalar_definition("int32_t", "B1slstride0", bankWidth // 8)]
    data_str += [format_scalar_definition("int32_t", "B1tlbound0", B0tlbound0)]
    data_str += [format_scalar_definition("int32_t", "B1tlstride0", B0tlstride0)]
    data_str += [format_scalar_definition("int32_t", "B1tlbound1", B0tlbound1)]
    data_str += [format_scalar_definition("int32_t", "B1tlstride1", B0tlstride1)]
    data_str += [format_scalar_definition("int32_t", "B1tlbound2", B0tlbound2)]
    data_str += [format_scalar_definition("int32_t", "B1tlstride2", B0tlstride2)]
    data_str += [format_scalar_definition("int32_t", "B1tlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "B1tlstride3", 0)]

    channel_en_B1 = [0] * B_enabled_channel_CSR_num
    channel_en_B1 = gen_channel_enable_CSR(channel_en_B1, channel_en_B0_bits)
    data_str += [
        "int32_t channel_en_B1[] = { " + ", ".join(map(str, channel_en_B1)) + " };"
    ]

    b1_data_length = b0_data_length
    data_str += [format_scalar_definition("int32_t", "b1_data_length", b1_data_length)]

    # ===================== D Writer settings ====================
    d_spatial_bound_0 = 8
    data_str += [format_scalar_definition("int32_t", "D0slstride0", bankWidth // 8)]
    data_str += [format_scalar_definition("int32_t", "D1slstride0", bankWidth // 8)]

    Dtlbound0 = 1
    Dtlstride0 = d_spatial_bound_0 * (bankWidth // 8)
    Dtlbound1 = N
    Dtlstride1 = out_elem_bits * meshRow * meshCol // 8
    Dtlbound2 = M
    Dtlstride2 = N * out_elem_bits * meshRow * meshCol // 8

    assert Dtlstride1 % (bankWidth // 8 * granularity_c_d) == 0, \
        f"Dtlstride1={Dtlstride1} not aligned to {bankWidth // 8 * granularity_c_d}"

    data_str += [format_scalar_definition("int32_t", "Dtlbound0", Dtlbound0)]
    data_str += [format_scalar_definition("int32_t", "Dtlstride0", Dtlstride0)]
    data_str += [format_scalar_definition("int32_t", "Dtlbound1", Dtlbound1)]
    data_str += [format_scalar_definition("int32_t", "Dtlstride1", Dtlstride1)]
    data_str += [format_scalar_definition("int32_t", "Dtlbound2", Dtlbound2)]
    data_str += [format_scalar_definition("int32_t", "Dtlstride2", Dtlstride2)]
    data_str += [format_scalar_definition("int32_t", "Dtlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "Dtlstride3", 0)]

    D_channels_per_writer = 8
    D_enabled_channel_CSR_num = int(math.ceil(D_channels_per_writer / 32))
    channel_en_D0 = gen_channel_enable_CSR([0] * D_enabled_channel_CSR_num, D_channels_per_writer)
    channel_en_D1 = gen_channel_enable_CSR([0] * D_enabled_channel_CSR_num, D_channels_per_writer)
    data_str += [
        "int32_t channel_en_D0[] = { " + ", ".join(map(str, channel_en_D0)) + " };"
    ]
    data_str += [
        "int32_t channel_en_D1[] = { " + ", ".join(map(str, channel_en_D1)) + " };"
    ]

    mode0_output_elems = M * N * meshRow * meshCol
    mode0_d_data_length = mode0_output_elems * out_elem_bits // 8
    data_str += [format_scalar_definition("int32_t", "mode0_d_data_length", mode0_d_data_length)]
    data_str += [format_scalar_definition("int32_t", "mode0_output_elems", mode0_output_elems)]

    # ===================== Base addresses ====================================
    b_channel_footprint = channel_en_B0_bits * (bankWidth // 8)
    d_channel_footprint = D_channels_per_writer * (bankWidth // 8)
    a_channel_footprint = channel_en_A_bits * (bankWidth // 8)

    a_max_temporal = (K - 1) * Atlstride0 + (N - 1) * Atlstride1 + (M - 1) * Atlstride2
    a_access_range = a_max_temporal + a_channel_footprint

    b_max_temporal = max(0, (K - 1) * B0tlstride0) + max(0, (N - 1) * B0tlstride1) \
        + max(0, (M - 1) * B0tlstride2)
    b_access_range = b_max_temporal + b_channel_footprint

    d_max_temporal = max(0, (Dtlbound0 - 1) * Dtlstride0) \
        + max(0, (N - 1) * Dtlstride1) + max(0, (M - 1) * Dtlstride2)
    d_access_range = d_max_temporal + d_channel_footprint

    delta_local_a = 0
    delta_local_a = align_wide_addr(delta_local_a, granularity_a * bankWidth // 8)

    a_data_length_bytes = K * M * (meshRow * tileSize * a_len // 8)
    a_total = max(a_data_length_bytes, a_access_range)
    delta_local_b0 = a_total
    delta_local_b0 = align_wide_addr(delta_local_b0, granularity_b * bankWidth // 8)

    b_data_size = K * N * b_tile_padded
    b_alloc = max(b_data_size, b_access_range)
    delta_local_b1 = delta_local_b0 + b_alloc
    delta_local_b1 = align_wide_addr(delta_local_b1, granularity_b * bankWidth // 8)

    delta_local_d0 = delta_local_b1 + b_alloc
    delta_local_d0 = align_wide_addr(delta_local_d0, granularity_c_d * bankWidth // 8)

    d_alloc = max(mode0_d_data_length, d_access_range)
    delta_local_d1_mode0 = delta_local_d0 + d_alloc
    delta_local_d1_mode0 = align_wide_addr(delta_local_d1_mode0, granularity_c_d * bankWidth // 8)

    data_str += [format_scalar_definition("int32_t", "delta_local_a", delta_local_a)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b0", delta_local_b0)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b1", delta_local_b1)]
    data_str += [format_scalar_definition("int32_t", "delta_local_d0", delta_local_d0)]
    data_str += [format_scalar_definition("int32_t", "delta_local_d1_mode0", delta_local_d1_mode0)]

    # ===================== Mode 1 streamer params ============================
    # Mode 1 A = Mode 0 output (at delta_local_d0)
    # Mode 0 output is [M*meshRow, N*meshCol] int16 = [8, 88] int16
    # In tile layout: [M, N, meshRow, meshCol] written contiguously
    # Mode 1 reads this as A with shape [M1, K1, meshRow, tileSize]
    # where K1*tileSize = N*meshCol = 88 → K1 = 88/8 = 11
    # Mode 1 A tile stride = meshRow * tileSize * 2 = 8*8*2 = 128 bytes
    # But: Mode 0 output is laid out as N output tiles, each meshRow*meshCol*2 bytes
    # The output tiles are contiguous: tile[0] at offset 0, tile[1] at Dtlstride1, etc.
    # For Mode 1, we need A tiles contiguous, which matches since Dtlstride1 = meshRow*meshCol*2 = 64
    # and Mode 1 Atlstride0 = meshRow*tileSize*2 = 128 bytes
    # Wait: meshRow*meshCol = 8*4 = 32 elements per output tile = 64 bytes
    # Mode 1 needs A tiles of meshRow*tileSize = 8*8 = 64 elements = 128 bytes
    # So 2 Mode 0 output tiles = 1 Mode 1 A tile (since tileSize/meshCol = 8/4 = 2)
    # The A reader stride for K dimension = 128 bytes (reads 128 bytes per A tile)
    # Mode 0 output is flat [8, N*meshCol] = [8, 88] in row-major, so consecutive
    # columns are consecutive in memory. The A reader reads [meshRow, tileSize] = [8,8]
    # sub-blocks with stride = tileSize*2 = 16 bytes between elements in same column.

    # Actually, let me reconsider the layout.
    # Mode 0 writer writes output as [M, N, meshRow, meshCol] tiles.
    # Each tile is meshRow*meshCol elements = 8*4 = 32 int16 = 64 bytes.
    # Tiles are written at stride Dtlstride1 = meshRow*meshCol*2 = 64 bytes.
    # So output memory is: tile(0,0) at d0+0, tile(0,1) at d0+64, ..., tile(0,21) at d0+21*64
    # That's contiguous: [8, 88] row-major is 8*88*2 = 1408 bytes total.
    # But tiles are [meshRow, meshCol] = [8,4] layout within each tile.
    # So memory for tile(m,n): row r, col c → element at offset (r*meshCol + c)*2
    # Across tiles: tile(m,n) starts at n*meshRow*meshCol*2 + m*N*meshRow*meshCol*2

    # For Mode 1, the A reader expects [M1, K1, meshRow, tileSize] layout.
    # Each A tile = [meshRow, tileSize] = [8, 8] = 64 elements = 128 bytes.
    # The A reader spatial stride = bankWidth/8 = 8 bytes (between channels).
    # A reader reads meshRow*tileSize*a_len/bankWidth = 8*8*16/64 = 16 channels.
    # Each channel reads 8 bytes = 4 int16 elements.
    # So channel 0 reads addr+0..7, channel 1 reads addr+8..15, etc.

    # The Mode 0 output is laid out as tiles of [8,4], but Mode 1 needs tiles of [8,8].
    # These layouts DON'T match tile-for-tile. However, if Mode 0 output is written
    # flat row-major (which it IS with contiguous tiles), then:
    # Row 0: cols 0-3 (tile 0), cols 4-7 (tile 1), ... cols 84-87 (tile 21)
    # = 88 int16 = 176 bytes per row
    # Mode 1 A tile [8,8] starting at col offset k*8:
    # Row r reads 8 consecutive int16 starting at r*88*2 + k*8*2

    # This is NOT how the tiled A reader works. The A reader expects the data in
    # tile layout [K1, meshRow, tileSize], NOT in row-major matrix layout.

    # Hmm, but the A reader just uses temporal strides. Let's think about what
    # temporal stride pattern achieves reading [8,8] sub-tiles from a flat [8,88] matrix.
    # We need: A reader temporal stride for K = (jump to next K tile) = tileSize * 2 = 16 bytes
    #   because consecutive K tiles are at columns k*tileSize apart, and each column is 2 bytes
    #   But wait - the A data in TCDM from Mode 0 output is [M*meshRow, N*meshCol] = [8, 88]
    #   with row stride = N*meshCol*2 = 88*2 = 176 bytes
    #   The A reader spatial channels read consecutive addresses, so channel i reads
    #   base + i*8 bytes. With 16 channels, that's 128 bytes = 1 A tile [8,8].
    #   Channel 0 reads elements [0..3], channel 1 reads [4..7], channel 2 reads [8..11]...
    #   That covers cols 0..63 if reading 128 bytes contiguously.
    #   But we want [8,8] = rows 0-7, cols 0-7. Row-major means row 0 has cols 0..87,
    #   row 1 starts 176 bytes later. So a contiguous 128-byte read gets row 0 cols 0..63,
    #   NOT a [8,8] tile.

    # So the flat row-major layout from Mode 0 writer does NOT match Mode 1 A reader
    # which expects tile layout. We need to set up the A reader strides to handle
    # the row-major output.

    # Actually, rethinking: the MODE 0 WRITER writes in tile layout [M, N, meshRow, meshCol].
    # Within each tile, elements are laid out as the spatial channels write them.
    # Writer D0 has 8 channels at spatial stride 8 bytes = 64 bytes per tile write.
    # So a [8,4] output tile occupies 64 bytes contiguously, and tiles are at stride 64.
    # This is effectively: meshRow groups of meshCol elements, where each group spans
    # channels. With 8 channels each writing 8 bytes (4 int16), we get:
    # ch0 → addr+0..7 (elements [row0,col0]..[row0,col3])
    # ch1 → addr+8..15 (elements [row1,col0]..[row1,col3])
    # ...
    # ch7 → addr+56..63 (elements [row7,col0]..[row7,col3])
    # So within a tile: row r, col c → offset (r*8 + c*2) ? No...
    # Actually with spatial stride 8: ch_i at addr + i*8
    # Each channel carries meshCol*out_elem_bits/64 = 4*16/64 = 1 word = 8 bytes = 4 int16
    # So ch_i carries row i's data: 4 consecutive int16 for row i.
    # Within ch_i's 8 bytes: elements [row_i, col_0], [row_i, col_1], [row_i, col_2], [row_i, col_3]

    # Multiple tiles are contiguous: tile (m=0,n=0) at D0+0, tile (m=0,n=1) at D0+64, etc.
    # So the full [8,88] output in memory is:
    # tile(0,0): rows 0-7, cols 0-3 (64B at offset 0)
    # tile(0,1): rows 0-7, cols 4-7 (64B at offset 64)
    # ...
    # tile(0,21): rows 0-7, cols 84-87 (64B at offset 21*64=1344)
    # Total: 22*64 = 1408 bytes

    # For Mode 1 A reader (16 channels, [8,8] tiles):
    # tile(k=0): rows 0-7, cols 0-7 → needs data from output tiles (0,0) and (0,1)
    # = 128 bytes from offset 0..127 → this IS contiguous because tile(0,0) is 0-63
    # and tile(0,1) is 64-127.
    # tile(k=1): rows 0-7, cols 8-15 → output tiles (0,2) and (0,3)
    # = 128 bytes from offset 128..255 → also contiguous!

    # So A tiles for Mode 1 are contiguous 128-byte blocks starting from delta_local_d0.
    # K stride = 128 bytes, N stride = 0 (broadcast), M stride = K1*128.

    # Verify: meshCol(Mode0) = 4, tileSize = 8, so each Mode 1 A tile spans
    # tileSize/meshCol = 2 Mode 0 output tiles. They are contiguous so stride = 128 works!

    # N_mode0 = N * meshCol = 22 * 4 = 88 columns in Mode 0 output
    # K1 = N_mode0 / tileSize = 88 / 8 = 11 ✓

    M1_Atlbound0 = K1
    M1_Atlstride0 = int(a_len * tileSize * meshRow // 8)  # 16*8*8/8 = 128
    M1_Atlbound1 = N1
    M1_Atlstride1 = 0
    M1_Atlbound2 = M1
    M1_Atlstride2 = int(K1 * a_len * tileSize * meshRow // 8)

    data_str += [format_scalar_definition("int32_t", "M1_Atlbound0", M1_Atlbound0)]
    data_str += [format_scalar_definition("int32_t", "M1_Atlstride0", M1_Atlstride0)]
    data_str += [format_scalar_definition("int32_t", "M1_Atlbound1", M1_Atlbound1)]
    data_str += [format_scalar_definition("int32_t", "M1_Atlstride1", M1_Atlstride1)]
    data_str += [format_scalar_definition("int32_t", "M1_Atlbound2", M1_Atlbound2)]
    data_str += [format_scalar_definition("int32_t", "M1_Atlstride2", M1_Atlstride2)]

    M1_B0tlbound0 = K1
    M1_B0tlstride0 = b_tile_padded
    M1_B0tlbound1 = N1
    M1_B0tlstride1 = K1 * b_tile_padded
    M1_B0tlbound2 = M1
    M1_B0tlstride2 = 0

    data_str += [format_scalar_definition("int32_t", "M1_B0tlbound0", M1_B0tlbound0)]
    data_str += [format_scalar_definition("int32_t", "M1_B0tlstride0", M1_B0tlstride0)]
    data_str += [format_scalar_definition("int32_t", "M1_B0tlbound1", M1_B0tlbound1)]
    data_str += [format_scalar_definition("int32_t", "M1_B0tlstride1", M1_B0tlstride1)]
    data_str += [format_scalar_definition("int32_t", "M1_B0tlbound2", M1_B0tlbound2)]
    data_str += [format_scalar_definition("int32_t", "M1_B0tlstride2", M1_B0tlstride2)]
    data_str += [format_scalar_definition("int32_t", "M1_B0tlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "M1_B0tlstride3", 0)]

    data_str += [format_scalar_definition("int32_t", "M1_B1tlbound0", M1_B0tlbound0)]
    data_str += [format_scalar_definition("int32_t", "M1_B1tlstride0", M1_B0tlstride0)]
    data_str += [format_scalar_definition("int32_t", "M1_B1tlbound1", M1_B0tlbound1)]
    data_str += [format_scalar_definition("int32_t", "M1_B1tlstride1", M1_B0tlstride1)]
    data_str += [format_scalar_definition("int32_t", "M1_B1tlbound2", M1_B0tlbound2)]
    data_str += [format_scalar_definition("int32_t", "M1_B1tlstride2", M1_B0tlstride2)]
    data_str += [format_scalar_definition("int32_t", "M1_B1tlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "M1_B1tlstride3", 0)]

    M1_Dtlbound0 = 1
    M1_Dtlstride0 = d_spatial_bound_0 * (bankWidth // 8)
    M1_Dtlbound1 = N1
    M1_Dtlstride1 = out_elem_bits * meshRow * meshCol // 8
    M1_Dtlbound2 = M1
    M1_Dtlstride2 = N1 * out_elem_bits * meshRow * meshCol // 8

    data_str += [format_scalar_definition("int32_t", "M1_Dtlbound0", M1_Dtlbound0)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlstride0", M1_Dtlstride0)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlbound1", M1_Dtlbound1)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlstride1", M1_Dtlstride1)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlbound2", M1_Dtlbound2)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlstride2", M1_Dtlstride2)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlstride3", 0)]

    # ===================== Mode 1 memory offsets ============================
    # Mode 1 A = Mode 0 output at delta_local_d0 (no separate A1 data!)
    w2l_data_length = N1 * K1 * b_tile_padded
    w2r_data_length = w2l_data_length

    # Mode 1 access ranges
    m1_a_max_temporal = (K1 - 1) * M1_Atlstride0 + (N1 - 1) * M1_Atlstride1 \
        + (M1 - 1) * M1_Atlstride2
    m1_a_access_range = m1_a_max_temporal + a_channel_footprint

    m1_b_max_temporal = max(0, (K1 - 1) * M1_B0tlstride0) \
        + max(0, (N1 - 1) * M1_B0tlstride1) + max(0, (M1 - 1) * M1_B0tlstride2)
    m1_b_access_range = m1_b_max_temporal + b_channel_footprint

    m1_d_max_temporal = max(0, (M1_Dtlbound0 - 1) * M1_Dtlstride0) \
        + max(0, (N1 - 1) * M1_Dtlstride1) + max(0, (M1 - 1) * M1_Dtlstride2)
    m1_d_access_range = m1_d_max_temporal + d_channel_footprint

    # W2 tiles placed after Mode 0 D1 dummy output
    delta_local_w2l = delta_local_d1_mode0 + d_alloc
    delta_local_w2l = align_wide_addr(delta_local_w2l, granularity_b * bankWidth // 8)

    w2_alloc = max(w2l_data_length, m1_b_access_range)
    delta_local_w2r = delta_local_w2l + w2_alloc
    delta_local_w2r = align_wide_addr(delta_local_w2r, granularity_b * bankWidth // 8)

    mode1_output_elems = M1 * N1 * meshRow * meshCol
    mode1_d_data_length = mode1_output_elems * out_elem_bits // 8
    m1_d_alloc = max(mode1_d_data_length, m1_d_access_range)

    delta_local_mode1_d0 = delta_local_w2r + w2_alloc
    delta_local_mode1_d0 = align_wide_addr(delta_local_mode1_d0, granularity_c_d * bankWidth // 8)

    delta_local_mode1_d1 = delta_local_mode1_d0 + m1_d_alloc
    delta_local_mode1_d1 = align_wide_addr(delta_local_mode1_d1, granularity_c_d * bankWidth // 8)

    data_str += [format_scalar_definition("int32_t", "w2l_data_length", w2l_data_length)]
    data_str += [format_scalar_definition("int32_t", "w2r_data_length", w2r_data_length)]
    data_str += [format_scalar_definition("int32_t", "delta_local_w2l", delta_local_w2l)]
    data_str += [format_scalar_definition("int32_t", "delta_local_w2r", delta_local_w2r)]
    data_str += [format_scalar_definition("int32_t", "delta_local_mode1_d0", delta_local_mode1_d0)]
    data_str += [format_scalar_definition("int32_t", "delta_local_mode1_d1", delta_local_mode1_d1)]
    data_str += [format_scalar_definition("int32_t", "mode1_output_elems", mode1_output_elems)]

    # ===================== Test data generation ==============================
    subtraction_a = 0
    subtraction_b = 0
    data_str += [format_scalar_definition("int8_t", "subtraction_a", subtraction_a)]
    data_str += [format_scalar_definition("int8_t", "subtraction_b", subtraction_b)]

    # Generate A (int16) - small values
    A_int16 = np.random.randint(A_MIN, A_MAX + 1,
                                 size=(M * K * meshRow * tileSize,)).astype(np.int16)

    # Generate W, V (int4)
    W_int4 = np.random.randint(B_MIN, B_MAX + 1,
                                size=(N * K * meshCol * tileSize,)).astype(np.int8)
    V_int4 = np.random.randint(B_MIN, B_MAX + 1,
                                size=(N * K * meshCol * tileSize,)).astype(np.int8)

    W_packed_raw = pack_int4(W_int4)
    V_packed_raw = pack_int4(V_int4)
    num_b_tiles = N * K
    W_packed = pad_b_tiles(W_packed_raw, num_b_tiles, b_tile_raw, b_tile_padded)
    V_packed = pad_b_tiles(V_packed_raw, num_b_tiles, b_tile_raw, b_tile_padded)

    data_str += [format_vector_definition("int16_t", "A", A_int16)]
    data_str += [format_vector_definition("uint8_t", "W", W_packed)]
    data_str += [format_vector_definition("uint8_t", "V", V_packed)]

    # ===================== Mode 0 Golden Model ==============================
    vc0_int32 = block_gemm_int16x4(
        M, K, N, meshRow, tileSize, meshCol,
        A_int16, W_int4, subtraction_a, subtraction_b
    )
    vc1_int32 = block_gemm_int16x4(
        M, K, N, meshRow, tileSize, meshCol,
        A_int16, V_int4, subtraction_a, subtraction_b
    )

    vc0_int16 = rescale_down_32to16(vc0_int32, rescale_input_zp, rescale_multiplier,
                                     rescale_output_zp, rescale_shift)
    vc0_silu = arithmetic_right_shift_int16(vc0_int16, 2)
    vc1_int16 = rescale_down_32to16(vc1_int32, rescale_input_zp, rescale_multiplier,
                                     rescale_output_zp, rescale_shift)
    mul_int32 = vc0_silu.astype(np.int32) * vc1_int16.astype(np.int32)
    mode0_out = rescale_down_32to16(mul_int32, rescale_input_zp, rescale_multiplier,
                                     rescale_output_zp, rescale_shift)

    data_str += [format_vector_definition("int16_t", "mode0_golden", mode0_out)]

    # ===================== Mode 1 (GEMM) ===================================
    # Mode 1 A input = Mode 0 output (mode0_out, shape [M*N*meshRow*meshCol] flat)
    # In tile layout for Mode 1: [M1, K1, meshRow, tileSize]
    # Mode 0 output is [M, N, meshRow, meshCol] in tile layout
    # Since M=1, this is [N, meshRow, meshCol] = [22, 8, 4]
    # Reshaped for Mode 1: [K1, meshRow, tileSize] where K1=11, tileSize=8
    # Each Mode 1 K tile = 2 Mode 0 N tiles (since tileSize/meshCol = 8/4 = 2)
    # The data IS contiguous in memory, so the reshape works.

    # Mode 1 A input = Mode 0 output read contiguously from TCDM.
    # Mode 0 output is [M, N, meshRow, meshCol] tiles written contiguously.
    # The Mode 1 A reader reads 128-byte blocks (16 channels × 8 bytes).
    # Two consecutive [8,4] output tiles (128 bytes) become one [8,8] A tile.
    # The hardware interprets contiguous bytes directly: ch0→row0_left,
    # ch1→row0_right, ch2→row1_left, etc. So a simple reshape of the flat
    # output to [M1, K1, meshRow, tileSize] matches the hardware layout.
    assert K1 * tileSize == N * meshCol, \
        f"Chaining constraint: K1*tileSize={K1*tileSize} != N*meshCol={N*meshCol}"
    mode1_A_flat = mode0_out.reshape(-1)

    # Generate W2 (int4) for Mode 1
    W2_int4 = np.random.randint(B_MIN, B_MAX + 1,
                                 size=(2 * N1 * K1 * meshCol * tileSize,)).astype(np.int8)
    W2_left_int4 = W2_int4[:N1 * K1 * meshCol * tileSize]
    W2_right_int4 = W2_int4[N1 * K1 * meshCol * tileSize:]

    W2_left_packed_raw = pack_int4(W2_left_int4)
    W2_right_packed_raw = pack_int4(W2_right_int4)
    num_m1_b_tiles = N1 * K1
    W2_left_packed = pad_b_tiles(W2_left_packed_raw, num_m1_b_tiles, b_tile_raw, b_tile_padded)
    W2_right_packed = pad_b_tiles(W2_right_packed_raw, num_m1_b_tiles, b_tile_raw, b_tile_padded)

    data_str += [format_vector_definition("uint8_t", "W2_left", W2_left_packed)]
    data_str += [format_vector_definition("uint8_t", "W2_right", W2_right_packed)]

    # Mode 1 golden: mode0_out (as A) @ W2 → int32 → rescale → int16
    golden_d0_int32 = block_gemm_int16x4(
        M1, K1, N1, meshRow, tileSize, meshCol,
        mode1_A_flat, W2_left_int4, subtraction_a, subtraction_b
    )
    golden_d1_int32 = block_gemm_int16x4(
        M1, K1, N1, meshRow, tileSize, meshCol,
        mode1_A_flat, W2_right_int4, subtraction_a, subtraction_b
    )

    mode1_golden_d0 = rescale_down_32to16(golden_d0_int32, rescale_input_zp,
                                           rescale_multiplier, rescale_output_zp, rescale_shift)
    mode1_golden_d1 = rescale_down_32to16(golden_d1_int32, rescale_input_zp,
                                           rescale_multiplier, rescale_output_zp, rescale_shift)

    data_str += [format_vector_definition("int16_t", "mode1_golden_d0", mode1_golden_d0)]
    data_str += [format_vector_definition("int16_t", "mode1_golden_d1", mode1_golden_d1)]

    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_A", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_B0", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_B1", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D0", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D1", 0)]

    data_str = "\n\n".join(data_str)
    return data_str


def main():
    parser = argparse.ArgumentParser(description="Generate data for int16x4 scale16 batch test")
    parser.add_argument(
        "--swcfg", type=pathlib.Path, required=True,
        help="Select param config file",
    )
    parser.add_argument(
        "--hwcfg", type=pathlib.Path, required=True,
        help="Select hardware config file",
    )
    args = parser.parse_args()

    with args.swcfg.open() as f:
        param = hjson.loads(f.read())
    with args.hwcfg.open() as f:
        hw = hjson.loads(f.read())

    merged_config = {**param, **hw}
    print(emit_header_file(**merged_config))


if __name__ == "__main__":
    main()
