#!/usr/bin/env python3

# Data generator for dual VersaCore int16x4 sanity test
# Mode 0 (SwiGLU): output = rescale_mul( rescale0(A@W)>>2 * rescale1(A@V) )
# Mode 1 (GEMM): D0 = rescale0(A1@W2_left), D1 = rescale1(A1@W2_right)
# A = sint16, W/V/W2 = sint4 (packed nibbles)

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
    Uses the same data layout as the working int8x8 block_gemm_golden_model.
    A: flat array, interpreted as [M, K, meshRow, tileSize] as int16
    B: flat array, interpreted as [N, K, meshCol, tileSize] as int4
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
    """Pack int4 values (in range [-8, 7]) into nibble-packed bytes.
    Lower nibble = even element, upper nibble = odd element.
    Returns packed bytes as uint8 array."""
    values = np.array(values, dtype=np.int8)
    assert len(values) % 2 == 0, "int4 array must have even length"
    packed = np.zeros(len(values) // 2, dtype=np.uint8)
    for i in range(0, len(values), 2):
        lo = values[i] & 0x0F      # lower nibble = even element
        hi = values[i+1] & 0x0F    # upper nibble = odd element
        packed[i // 2] = (hi << 4) | lo
    return packed


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

    data_str += [format_scalar_definition("uint32_t", "M", M)]
    data_str += [format_scalar_definition("uint32_t", "K", K)]
    data_str += [format_scalar_definition("uint32_t", "N", N)]

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
    d_array_width = snax_acc_cfg["snax_versacore_array_output_d_width"]
    snax_versacore_serial_c_d_width = snax_acc_cfg["snax_versacore_serial_c_d_width"]

    bankWidth = 64
    out_elem_bits = 16  # int16 output after rescale

    granularity_a = snax_acc_cfg.get("granularity_a", 1)
    granularity_b = snax_acc_cfg.get("granularity_b", 1)
    granularity_c_d = snax_acc_cfg.get("granularity_c_d", 1)

    data_str += [format_scalar_definition("uint32_t", "meshRow", meshRow)]
    data_str += [format_scalar_definition("uint32_t", "tileSize", tileSize)]
    data_str += [format_scalar_definition("uint32_t", "meshCol", meshCol)]

    stationary = kwargs["stationary"]
    assert stationary == 0
    data_str += [format_scalar_definition("uint32_t", "stationary", stationary)]

    # Identity rescale parameters
    rescale_input_zp = 0
    rescale_multiplier = 1
    rescale_output_zp = 0
    rescale_shift = 0

    data_str += [format_scalar_definition("int32_t", "rescale_input_zp", rescale_input_zp)]
    data_str += [format_scalar_definition("uint32_t", "rescale_multiplier", rescale_multiplier)]
    data_str += [format_scalar_definition("int32_t", "rescale_output_zp", rescale_output_zp)]
    data_str += [format_scalar_definition("uint32_t", "rescale_shift", rescale_shift)]

    # Data range: small to avoid overflow
    A_MIN, A_MAX = -3, 3
    B_MIN, B_MAX = -3, 3  # must fit in sint4 [-8, 7]

    # ===================== A streamer settings (Reader 0) ====================
    data_str += [format_scalar_definition("int32_t", "Aslstride0", int(bankWidth // 8))]

    # A element is int16 (2 bytes)
    Atlbound0 = K
    Atlstride0 = int(a_len * tileSize * meshRow // 8)  # bytes per A tile
    Atlbound1 = N
    Atlstride1 = 0  # A broadcast over N
    Atlbound2 = M
    Atlstride2 = int(K * a_len * tileSize * meshRow // 8)

    assert Atlstride0 % (bankWidth // 8 * granularity_a) == 0, \
        f"Atlstride0={Atlstride0} not aligned"
    # stride1=0 always aligned
    if Atlstride2 > 0:
        assert Atlstride2 % (bankWidth // 8 * granularity_a) == 0, \
            f"Atlstride2={Atlstride2} not aligned"

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
    # B elements are int4 (nibble-packed, 2 per byte)
    data_str += [format_scalar_definition("int32_t", "B0slstride0", bankWidth // 8)]

    B0tlbound0 = K
    B0tlstride0 = b_len * tileSize * meshCol // 8  # bytes per B tile
    B0tlbound1 = N
    B0tlstride1 = K * b_len * tileSize * meshCol // 8
    B0tlbound2 = M
    B0tlstride2 = 0  # B broadcast over M

    if B0tlstride0 > 0:
        assert B0tlstride0 % (bankWidth // 8 * granularity_b) == 0, \
            f"B0tlstride0={B0tlstride0} not aligned to {bankWidth // 8 * granularity_b}"

    data_str += [format_scalar_definition("int32_t", "B0tlbound0", B0tlbound0)]
    data_str += [format_scalar_definition("int32_t", "B0tlstride0", B0tlstride0)]
    data_str += [format_scalar_definition("int32_t", "B0tlbound1", B0tlbound1)]
    data_str += [format_scalar_definition("int32_t", "B0tlstride1", B0tlstride1)]
    data_str += [format_scalar_definition("int32_t", "B0tlbound2", B0tlbound2)]
    data_str += [format_scalar_definition("int32_t", "B0tlstride2", B0tlstride2)]
    data_str += [format_scalar_definition("int32_t", "B0tlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "B0tlstride3", 0)]

    # Channel enable B0
    B_enabled_channel_CSR_num = int(math.ceil(b_array_width // bankWidth / 32))
    channel_en_B0 = [0] * B_enabled_channel_CSR_num
    # Number of 64-bit channels needed for B data (round up to nearest 8 for alignment)
    b_bits_needed = meshCol * tileSize * b_len  # e.g. 4*8*4 = 128 bits
    channel_en_B0_bits = int((b_bits_needed // bankWidth + 7) // 8 * 8)
    if channel_en_B0_bits == 0:
        channel_en_B0_bits = 8  # minimum
    channel_en_B0 = gen_channel_enable_CSR(channel_en_B0, channel_en_B0_bits)
    data_str += [
        "int32_t channel_en_B0[] = { " + ", ".join(map(str, channel_en_B0)) + " };"
    ]

    # B data length in bytes (nibble-packed)
    b0_data_length = K * N * tileSize * meshCol * b_len // 8
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
    PostprocLanes = snax_acc_cfg.get("snax_dual_versacore_postproc_lanes", 32)
    ElemsPerBeat_int32 = snax_versacore_serial_c_d_width // 32
    NumChunks = (ElemsPerBeat_int32 + PostprocLanes - 1) // PostprocLanes

    # Writer D: 8 channels, each 64b = 512b = DataWidthOut
    d_spatial_bound_0 = 8
    data_str += [format_scalar_definition("int32_t", "D0slstride0", bankWidth // 8)]
    data_str += [format_scalar_definition("int32_t", "D1slstride0", bankWidth // 8)]

    Dtlbound0 = 1  # No serialization (NumChunks=1 with PostprocLanes=32)
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

    # Channel enable for writers (8 channels each)
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
    # IMPORTANT: Each streamer reader/writer with N enabled channels at spatial
    # stride S occupies N*S bytes of TCDM address space.  Buffers must be spaced
    # by this *channel footprint*, not just the logical data size, to avoid
    # overlapping TCDM bank accesses that cause deadlocks.
    b_channel_footprint = channel_en_B0_bits * (bankWidth // 8)  # 8 * 8 = 64
    d_channel_footprint = D_channels_per_writer * (bankWidth // 8)  # 8 * 8 = 64
    a_channel_footprint = channel_en_A_bits * (bankWidth // 8)  # 16 * 8 = 128

    delta_local_a = 0
    delta_local_a = align_wide_addr(delta_local_a, granularity_a * bankWidth // 8)

    a_total = max(K * M * (meshRow * tileSize * a_len // 8), a_channel_footprint)
    delta_local_b0 = a_total
    delta_local_b0 = align_wide_addr(delta_local_b0, granularity_b * bankWidth // 8)

    b_data_size = K * N * (meshCol * tileSize * b_len // 8)
    b_alloc = max(b_data_size, b_channel_footprint)
    delta_local_b1 = delta_local_b0 + b_alloc
    delta_local_b1 = align_wide_addr(delta_local_b1, granularity_b * bankWidth // 8)

    delta_local_d0 = delta_local_b1 + b_alloc
    delta_local_d0 = align_wide_addr(delta_local_d0, granularity_c_d * bankWidth // 8)

    d_alloc = max(mode0_d_data_length, d_channel_footprint)
    delta_local_d1_mode0 = delta_local_d0 + d_alloc
    delta_local_d1_mode0 = align_wide_addr(delta_local_d1_mode0, granularity_c_d * bankWidth // 8)

    data_str += [format_scalar_definition("int32_t", "delta_local_a", delta_local_a)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b0", delta_local_b0)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b1", delta_local_b1)]
    data_str += [format_scalar_definition("int32_t", "delta_local_d0", delta_local_d0)]
    data_str += [format_scalar_definition("int32_t", "delta_local_d1_mode0", delta_local_d1_mode0)]

    # ===================== Test data generation ==============================
    subtraction_a = 0
    subtraction_b = 0
    data_str += [format_scalar_definition("int8_t", "subtraction_a", subtraction_a)]
    data_str += [format_scalar_definition("int8_t", "subtraction_b", subtraction_b)]

    # Generate A (int16) - small values
    A_int16 = np.random.randint(A_MIN, A_MAX + 1,
                                 size=(M * K * meshRow * tileSize,)).astype(np.int16)

    # Generate W, V (int4) as int8 arrays (values in [-3, 3], well within [-8, 7])
    # Layout: [N, K, meshCol, tileSize] — matches hardware's B port layout
    W_int4 = np.random.randint(B_MIN, B_MAX + 1,
                                size=(N * K * meshCol * tileSize,)).astype(np.int8)
    V_int4 = np.random.randint(B_MIN, B_MAX + 1,
                                size=(N * K * meshCol * tileSize,)).astype(np.int8)

    # Pack int4 weights into nibble-packed bytes for hardware
    W_packed = pack_int4(W_int4)
    V_packed = pack_int4(V_int4)

    data_str += [format_vector_definition("int16_t", "A", A_int16)]
    data_str += [format_vector_definition("uint8_t", "W", W_packed)]
    data_str += [format_vector_definition("uint8_t", "V", V_packed)]

    # ===================== Mode 0 Golden Model ==============================
    C_zeros = np.zeros(M * N * meshRow * meshCol, dtype=np.int32)

    vc0_int32 = block_gemm_int16x4(
        M, K, N, meshRow, tileSize, meshCol,
        A_int16, W_int4, subtraction_a, subtraction_b
    )
    vc1_int32 = block_gemm_int16x4(
        M, K, N, meshRow, tileSize, meshCol,
        A_int16, V_int4, subtraction_a, subtraction_b
    )

    # RescaleDown0 (identity)
    vc0_int16 = rescale_down_32to16(vc0_int32, rescale_input_zp, rescale_multiplier,
                                     rescale_output_zp, rescale_shift)
    # Shifter 6-stage (>>2)
    vc0_silu = arithmetic_right_shift_int16(vc0_int16, 2)
    # RescaleDown1 (identity)
    vc1_int16 = rescale_down_32to16(vc1_int32, rescale_input_zp, rescale_multiplier,
                                     rescale_output_zp, rescale_shift)
    # ElemMul
    mul_int32 = vc0_silu.astype(np.int32) * vc1_int16.astype(np.int32)
    # RescaleMul (identity)
    mode0_out = rescale_down_32to16(mul_int32, rescale_input_zp, rescale_multiplier,
                                     rescale_output_zp, rescale_shift)

    data_str += [format_vector_definition("int16_t", "mode0_golden", mode0_out)]

    # ===================== Mode 1 (GEMM) ===================================
    # Independent data for mode 1 (not chained from mode 0)
    # because meshCol != tileSize for some shapes, so mode0 output can't
    # directly reshape as mode1 A input tiles.
    M1 = 1
    K1 = 1
    N1 = 1

    # Generate independent A1 (int16) for mode 1
    A1_int16 = np.random.randint(A_MIN, A_MAX + 1,
                                  size=(M1 * K1 * meshRow * tileSize,)).astype(np.int16)

    W2_int4 = np.random.randint(B_MIN, B_MAX + 1,
                                 size=(2 * N1 * K1 * meshCol * tileSize,)).astype(np.int8)
    W2_left_int4 = W2_int4[:N1 * K1 * meshCol * tileSize]
    W2_right_int4 = W2_int4[N1 * K1 * meshCol * tileSize:]

    W2_left_packed = pack_int4(W2_left_int4)
    W2_right_packed = pack_int4(W2_right_int4)

    data_str += [format_vector_definition("int16_t", "A1", A1_int16)]
    data_str += [format_vector_definition("uint8_t", "W2_left", W2_left_packed)]
    data_str += [format_vector_definition("uint8_t", "W2_right", W2_right_packed)]

    # Mode 1 golden: A1 (int16) @ W2 (int4) -> int32 -> rescale -> int16
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

    mode1_output_elems = M1 * N1 * meshRow * meshCol
    data_str += [format_scalar_definition("int32_t", "mode1_output_elems", mode1_output_elems)]
    data_str += [format_scalar_definition("uint32_t", "M1", M1)]
    data_str += [format_scalar_definition("uint32_t", "K1", K1)]
    data_str += [format_scalar_definition("uint32_t", "N1", N1)]

    # ===================== Mode 1 Streamer params ============================
    M1_Atlbound0 = K1
    M1_Atlstride0 = int(a_len * tileSize * meshRow // 8)
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
    M1_B0tlstride0 = b_len * tileSize * meshCol // 8
    M1_B0tlbound1 = N1
    M1_B0tlstride1 = K1 * b_len * tileSize * meshCol // 8
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
    M1_Dtlbound3 = 1
    M1_Dtlstride3 = 0

    data_str += [format_scalar_definition("int32_t", "M1_Dtlbound0", M1_Dtlbound0)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlstride0", M1_Dtlstride0)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlbound1", M1_Dtlbound1)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlstride1", M1_Dtlstride1)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlbound2", M1_Dtlbound2)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlstride2", M1_Dtlstride2)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlbound3", M1_Dtlbound3)]
    data_str += [format_scalar_definition("int32_t", "M1_Dtlstride3", M1_Dtlstride3)]

    # Mode 1 memory offsets
    a1_data_length = M1 * K1 * meshRow * tileSize * a_len // 8
    w2l_data_length = N1 * K1 * meshCol * tileSize * b_len // 8
    w2r_data_length = w2l_data_length

    delta_local_a1 = delta_local_d1_mode0 + d_alloc
    delta_local_a1 = align_wide_addr(delta_local_a1, granularity_a * bankWidth // 8)

    delta_local_w2l = delta_local_a1 + max(a1_data_length, a_channel_footprint)
    delta_local_w2l = align_wide_addr(delta_local_w2l, granularity_b * bankWidth // 8)

    delta_local_w2r = delta_local_w2l + max(w2l_data_length, b_channel_footprint)
    delta_local_w2r = align_wide_addr(delta_local_w2r, granularity_b * bankWidth // 8)

    mode1_d_data_length = mode1_output_elems * out_elem_bits // 8

    delta_local_mode1_d0 = delta_local_w2r + max(w2r_data_length, b_channel_footprint)
    delta_local_mode1_d0 = align_wide_addr(delta_local_mode1_d0, granularity_c_d * bankWidth // 8)

    delta_local_mode1_d1 = delta_local_mode1_d0 + max(mode1_d_data_length, d_channel_footprint)
    delta_local_mode1_d1 = align_wide_addr(delta_local_mode1_d1, granularity_c_d * bankWidth // 8)

    data_str += [format_scalar_definition("int32_t", "a1_data_length", a1_data_length)]
    data_str += [format_scalar_definition("int32_t", "w2l_data_length", w2l_data_length)]
    data_str += [format_scalar_definition("int32_t", "w2r_data_length", w2r_data_length)]
    data_str += [format_scalar_definition("int32_t", "delta_local_a1", delta_local_a1)]
    data_str += [format_scalar_definition("int32_t", "delta_local_w2l", delta_local_w2l)]
    data_str += [format_scalar_definition("int32_t", "delta_local_w2r", delta_local_w2r)]
    data_str += [format_scalar_definition("int32_t", "delta_local_mode1_d0", delta_local_mode1_d0)]
    data_str += [format_scalar_definition("int32_t", "delta_local_mode1_d1", delta_local_mode1_d1)]

    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_A", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_B0", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_B1", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D0", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D1", 0)]

    data_str = "\n\n".join(data_str)
    return data_str


def main():
    parser = argparse.ArgumentParser(description="Generate data for int16x4 sanity test")
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
