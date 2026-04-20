#!/usr/bin/env python3

# Data generator for dual VersaCore int16x4 Shape S2 ping-pong test
# array_shape=2 → (meshRow=2, tileSize=8, meshCol=16)
# B tile double-buffering along N dimension.
# Mode 0 (SwiGLU), Mode 1 (GEMM) with independent A1 data.

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
    return (arr_int16.astype(np.int32) >> n).clip(-32768, 32767).astype(np.int16)


def block_gemm_int16x4(M, K, N, meshRow, tileSize, meshCol, A_flat, B_flat,
                        subtraction_a, subtraction_b):
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
    values = np.array(values, dtype=np.int8)
    assert len(values) % 2 == 0
    packed = np.zeros(len(values) // 2, dtype=np.uint8)
    for i in range(0, len(values), 2):
        lo = values[i] & 0x0F
        hi = values[i+1] & 0x0F
        packed[i // 2] = (hi << 4) | lo
    return packed


def pad_b_tiles(packed_bytes, num_tiles, raw_tile_bytes, padded_tile_bytes):
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
    M1 = kwargs.get("M1", M)
    K1 = kwargs.get("K1", K)
    N1 = kwargs.get("N1", N)
    N_chunk = kwargs["N_chunk"]
    N1_chunk = kwargs["N1_chunk"]

    assert N % N_chunk == 0, f"N={N} not divisible by N_chunk={N_chunk}"
    assert N1 % N1_chunk == 0, f"N1={N1} not divisible by N1_chunk={N1_chunk}"
    num_chunks = N // N_chunk
    num_chunks1 = N1 // N1_chunk

    data_str += [format_scalar_definition("uint32_t", "M", M)]
    data_str += [format_scalar_definition("uint32_t", "K", K)]
    data_str += [format_scalar_definition("uint32_t", "N", N)]
    data_str += [format_scalar_definition("uint32_t", "M1", M1)]
    data_str += [format_scalar_definition("uint32_t", "K1", K1)]
    data_str += [format_scalar_definition("uint32_t", "N1", N1)]
    data_str += [format_scalar_definition("uint32_t", "N_chunk", N_chunk)]
    data_str += [format_scalar_definition("uint32_t", "N1_chunk", N1_chunk)]
    data_str += [format_scalar_definition("uint32_t", "num_chunks", num_chunks)]
    data_str += [format_scalar_definition("uint32_t", "num_chunks1", num_chunks1)]

    array_shape = kwargs["array_shape"]
    data_str += [format_scalar_definition("uint32_t", "array_shape", array_shape)]
    data_type = kwargs["data_type"]
    data_str += [format_scalar_definition("uint32_t", "data_type", data_type)]

    # Hardware parameters
    snax_acc_cfg = kwargs["snax_dual_versacore_int16x4_core_template"]["snax_acc_cfg"][0]
    meshRow = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][0]
    tileSize = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][1]
    meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][2]

    a_len = snax_acc_cfg["snax_versacore_input_a_element_width"][data_type]
    b_len = snax_acc_cfg["snax_versacore_input_b_element_width"][data_type]

    a_array_width = snax_acc_cfg["snax_versacore_array_input_a_width"]
    b_array_width = snax_acc_cfg["snax_versacore_array_input_b_width"]

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

    # ===================== Streamer settings (per-chunk, N=N_chunk) ===========
    # A reader (same for all chunks — A is shared, broadcast over N)
    data_str += [format_scalar_definition("int32_t", "Aslstride0", int(bankWidth // 8))]

    Atlbound0 = K
    Atlstride0 = int(a_len * tileSize * meshRow // 8)
    Atlbound1 = N_chunk  # per-chunk N
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

    # B0 streamer settings (per-chunk: N=N_chunk)
    data_str += [format_scalar_definition("int32_t", "B0slstride0", bankWidth // 8)]

    B0tlbound0 = K
    B0tlstride0 = b_tile_padded
    B0tlbound1 = N_chunk
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

    # Per-chunk B data: N_chunk * K tiles per buffer
    b_chunk_data_length = K * N_chunk * b_tile_padded
    b_full_data_length = K * N * b_tile_padded
    data_str += [format_scalar_definition("int32_t", "b_chunk_data_length", b_chunk_data_length)]
    data_str += [format_scalar_definition("int32_t", "b_full_data_length", b_full_data_length)]

    # B1 same as B0
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

    # D Writer settings (per-chunk: N=N_chunk)
    d_spatial_bound_0 = 8
    data_str += [format_scalar_definition("int32_t", "D0slstride0", bankWidth // 8)]
    data_str += [format_scalar_definition("int32_t", "D1slstride0", bankWidth // 8)]

    Dtlbound0 = 1
    Dtlstride0 = d_spatial_bound_0 * (bankWidth // 8)
    Dtlbound1 = N_chunk
    Dtlstride1 = out_elem_bits * meshRow * meshCol // 8
    Dtlbound2 = M
    Dtlstride2 = N_chunk * out_elem_bits * meshRow * meshCol // 8

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

    # D output: per-chunk offset stride
    d_chunk_bytes = M * N_chunk * meshRow * meshCol * out_elem_bits // 8
    data_str += [format_scalar_definition("int32_t", "d_chunk_bytes", d_chunk_bytes)]

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

    # A: single buffer
    delta_local_a = 0
    delta_local_a = align_wide_addr(delta_local_a, granularity_a * bankWidth // 8)
    a_data_length_bytes = K * M * (meshRow * tileSize * a_len // 8)

    # B0: TWO buffers for ping-pong (each holds N_chunk * K tiles)
    b_chunk_alloc = b_chunk_data_length
    # Add channel footprint for access range
    b_chunk_max_temporal = max(0, (K - 1) * B0tlstride0) + max(0, (N_chunk - 1) * B0tlstride1)
    b_chunk_access_range = b_chunk_max_temporal + b_channel_footprint
    b_chunk_alloc = max(b_chunk_alloc, b_chunk_access_range)

    delta_local_b0_buf0 = a_data_length_bytes
    delta_local_b0_buf0 = align_wide_addr(delta_local_b0_buf0, granularity_b * bankWidth // 8)

    delta_local_b0_buf1 = delta_local_b0_buf0 + b_chunk_alloc
    delta_local_b0_buf1 = align_wide_addr(delta_local_b0_buf1, granularity_b * bankWidth // 8)

    # B1: TWO buffers
    delta_local_b1_buf0 = delta_local_b0_buf1 + b_chunk_alloc
    delta_local_b1_buf0 = align_wide_addr(delta_local_b1_buf0, granularity_b * bankWidth // 8)

    delta_local_b1_buf1 = delta_local_b1_buf0 + b_chunk_alloc
    delta_local_b1_buf1 = align_wide_addr(delta_local_b1_buf1, granularity_b * bankWidth // 8)

    # D0, D1: full output buffers (output from all chunks goes here)
    d_full_max_temporal = max(0, (Dtlbound0 - 1) * Dtlstride0) \
        + max(0, (N - 1) * Dtlstride1) + max(0, (M - 1) * (N * out_elem_bits * meshRow * meshCol // 8))
    d_full_access_range = d_full_max_temporal + d_channel_footprint
    d_full_alloc = max(mode0_d_data_length, d_full_access_range)

    delta_local_d0 = delta_local_b1_buf1 + b_chunk_alloc
    delta_local_d0 = align_wide_addr(delta_local_d0, granularity_c_d * bankWidth // 8)

    delta_local_d1_mode0 = delta_local_d0 + d_full_alloc
    delta_local_d1_mode0 = align_wide_addr(delta_local_d1_mode0, granularity_c_d * bankWidth // 8)

    data_str += [format_scalar_definition("int32_t", "delta_local_a", delta_local_a)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b0_buf0", delta_local_b0_buf0)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b0_buf1", delta_local_b0_buf1)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b1_buf0", delta_local_b1_buf0)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b1_buf1", delta_local_b1_buf1)]
    data_str += [format_scalar_definition("int32_t", "delta_local_d0", delta_local_d0)]
    data_str += [format_scalar_definition("int32_t", "delta_local_d1_mode0", delta_local_d1_mode0)]

    # ===================== Mode 1 streamer params ============================
    # Mode 1 A = Mode 0 output (contiguous at delta_local_d0)
    M1_Atlbound0 = K1
    M1_Atlstride0 = int(a_len * tileSize * meshRow // 8)
    M1_Atlbound1 = N1_chunk
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
    M1_B0tlbound1 = N1_chunk
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
    M1_Dtlbound1 = N1_chunk
    M1_Dtlstride1 = out_elem_bits * meshRow * meshCol // 8
    M1_Dtlbound2 = M1
    M1_Dtlstride2 = N1_chunk * out_elem_bits * meshRow * meshCol // 8

    data_str += [format_scalar_definition("int32_t", "M1_Dtlbound0", M1_Dtlbound0)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlstride0", M1_Dtlstride0)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlbound1", M1_Dtlbound1)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlstride1", M1_Dtlstride1)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlbound2", M1_Dtlbound2)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlstride2", M1_Dtlstride2)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlstride3", 0)]

    m1_d_chunk_bytes = M1 * N1_chunk * meshRow * meshCol * out_elem_bits // 8
    data_str += [format_scalar_definition("int32_t", "m1_d_chunk_bytes", m1_d_chunk_bytes)]

    # ===================== Mode 1 memory offsets ============================
    m1_b_chunk_data_length = N1_chunk * K1 * b_tile_padded
    data_str += [format_scalar_definition("int32_t", "m1_b_chunk_data_length",
                                          m1_b_chunk_data_length)]
    m1_b_full_data_length = N1 * K1 * b_tile_padded
    data_str += [format_scalar_definition("int32_t", "m1_b_full_data_length",
                                          m1_b_full_data_length)]

    # Mode 1 B access range per chunk
    m1_b_chunk_max_temporal = max(0, (K1 - 1) * M1_B0tlstride0) \
        + max(0, (N1_chunk - 1) * M1_B0tlstride1)
    m1_b_chunk_access_range = m1_b_chunk_max_temporal + b_channel_footprint
    m1_b_chunk_alloc = max(m1_b_chunk_data_length, m1_b_chunk_access_range)

    # (Mode 1 memory layout is defined below after A1 generation)

    # ===================== Test data generation ==============================
    subtraction_a = 0
    subtraction_b = 0
    data_str += [format_scalar_definition("int8_t", "subtraction_a", subtraction_a)]
    data_str += [format_scalar_definition("int8_t", "subtraction_b", subtraction_b)]

    A_int16 = np.random.randint(A_MIN, A_MAX + 1,
                                 size=(M * K * meshRow * tileSize,)).astype(np.int16)

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
    # Mode 1 uses independent A1 data (not chained from Mode 0)
    A1_int16 = np.random.randint(A_MIN, A_MAX + 1,
                                  size=(M1 * K1 * meshRow * tileSize,)).astype(np.int16)
    data_str += [format_vector_definition("int16_t", "A1", A1_int16)]

    a1_data_length = M1 * K1 * meshRow * tileSize * a_len // 8
    data_str += [format_scalar_definition("int32_t", "a1_data_length", a1_data_length)]

    # A1 memory: placed after D1 mode0
    m1_a_max_temporal_val = (K1 - 1) * M1_Atlstride0 + (N1_chunk - 1) * M1_Atlstride1 \
        + (M1 - 1) * M1_Atlstride2
    m1_a_access_range_val = m1_a_max_temporal_val + a_channel_footprint
    a1_alloc = max(a1_data_length, m1_a_access_range_val)

    delta_local_a1 = delta_local_d1_mode0 + d_full_alloc
    delta_local_a1 = align_wide_addr(delta_local_a1, granularity_a * bankWidth // 8)
    data_str += [format_scalar_definition("int32_t", "delta_local_a1", delta_local_a1)]

    # W2l: TWO buffers (placed after A1)
    delta_local_w2l_buf0 = delta_local_a1 + a1_alloc
    delta_local_w2l_buf0 = align_wide_addr(delta_local_w2l_buf0, granularity_b * bankWidth // 8)

    delta_local_w2l_buf1 = delta_local_w2l_buf0 + m1_b_chunk_alloc
    delta_local_w2l_buf1 = align_wide_addr(delta_local_w2l_buf1, granularity_b * bankWidth // 8)

    # W2r: TWO buffers
    delta_local_w2r_buf0 = delta_local_w2l_buf1 + m1_b_chunk_alloc
    delta_local_w2r_buf0 = align_wide_addr(delta_local_w2r_buf0, granularity_b * bankWidth // 8)

    delta_local_w2r_buf1 = delta_local_w2r_buf0 + m1_b_chunk_alloc
    delta_local_w2r_buf1 = align_wide_addr(delta_local_w2r_buf1, granularity_b * bankWidth // 8)

    # Mode 1 output
    mode1_output_elems = M1 * N1 * meshRow * meshCol
    mode1_d_data_length = mode1_output_elems * out_elem_bits // 8
    m1_d_full_max_temporal = max(0, (M1_Dtlbound0 - 1) * M1_Dtlstride0) \
        + max(0, (N1 - 1) * M1_Dtlstride1) \
        + max(0, (M1 - 1) * (N1 * out_elem_bits * meshRow * meshCol // 8))
    m1_d_full_access_range = m1_d_full_max_temporal + d_channel_footprint
    m1_d_full_alloc = max(mode1_d_data_length, m1_d_full_access_range)

    delta_local_mode1_d0 = delta_local_w2r_buf1 + m1_b_chunk_alloc
    delta_local_mode1_d0 = align_wide_addr(delta_local_mode1_d0, granularity_c_d * bankWidth // 8)

    delta_local_mode1_d1 = delta_local_mode1_d0 + m1_d_full_alloc
    delta_local_mode1_d1 = align_wide_addr(delta_local_mode1_d1, granularity_c_d * bankWidth // 8)

    data_str += [format_scalar_definition("int32_t", "delta_local_w2l_buf0", delta_local_w2l_buf0)]
    data_str += [format_scalar_definition("int32_t", "delta_local_w2l_buf1", delta_local_w2l_buf1)]
    data_str += [format_scalar_definition("int32_t", "delta_local_w2r_buf0", delta_local_w2r_buf0)]
    data_str += [format_scalar_definition("int32_t", "delta_local_w2r_buf1", delta_local_w2r_buf1)]
    data_str += [format_scalar_definition("int32_t", "delta_local_mode1_d0", delta_local_mode1_d0)]
    data_str += [format_scalar_definition("int32_t", "delta_local_mode1_d1", delta_local_mode1_d1)]
    data_str += [format_scalar_definition("int32_t", "mode1_output_elems", mode1_output_elems)]

    # ===================== Mode 1 data generation ==============================
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

    # Mode 1 golden model: A1 @ W2 -> rescale
    golden_d0_int32 = block_gemm_int16x4(
        M1, K1, N1, meshRow, tileSize, meshCol,
        A1_int16, W2_left_int4, subtraction_a, subtraction_b
    )
    golden_d1_int32 = block_gemm_int16x4(
        M1, K1, N1, meshRow, tileSize, meshCol,
        A1_int16, W2_right_int4, subtraction_a, subtraction_b
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
    parser = argparse.ArgumentParser(description="Generate data for int16x4 scale16 ping-pong test")
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
