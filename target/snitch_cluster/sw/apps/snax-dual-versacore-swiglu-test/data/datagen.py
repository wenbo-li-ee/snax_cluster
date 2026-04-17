#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for dual VersaCore SwiGLU test
# Mode 0 (SwiGLU): output = rescale_mul( rescale0(A@W)>>2 * rescale1(A@V) )
# Mode 1 (GEMM): D0 = rescale0(A1@W2_left), D1 = rescale1(A1@W2_right)

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
    block_gemm_golden_model,
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
        shifted_value = correct_shift >> shift  # arithmetic in numpy int64
    else:
        shifted_value = multiplied
    # Truncate to 32 bits then add output_zp
    out = shifted_value.astype(np.int32).astype(np.int64) + int(output_zp)
    return np.clip(out, -32768, 32767).astype(np.int16)


def arithmetic_right_shift_int16(arr_int16, n):
    """SiLU placeholder: arithmetic right shift by n on int16."""
    return (arr_int16.astype(np.int32) >> n).clip(-32768, 32767).astype(np.int16)


def emit_header_file(**kwargs):
    emit_str = "#include <stdint.h>\n\n"
    emit_str += emit_dual_versacore_data(**kwargs)
    return emit_str


def signed_int_range(bits):
    min_val = -(2 ** (bits - 1))
    max_val = 2 ** (bits - 1) - 1
    return min_val, max_val


def gen_channel_enable_CSR(channel_en_CSR, channel_en_bits):
    for i in range(channel_en_bits):
        element_index = i // 32
        bit_position = i % 32
        if element_index < len(channel_en_CSR):
            channel_en_CSR[element_index] |= 1 << (bit_position)
    channel_en_CSR = [int(x) for x in channel_en_CSR][::-1]
    return channel_en_CSR


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
    snax_acc_cfg = kwargs["snax_dual_versacore_swiglu_core_template"]["snax_acc_cfg"][0]
    meshRow = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][0]
    tileSize = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][1]
    meshCol = snax_acc_cfg["snax_versacore_spatial_unrolling"][data_type][array_shape][2]

    a_len = snax_acc_cfg["snax_versacore_input_a_element_width"][data_type]
    b_len = snax_acc_cfg["snax_versacore_input_b_element_width"][data_type]
    c_len = snax_acc_cfg["snax_versacore_input_c_element_width"][data_type]

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
    assert stationary == 0, "Dual VersaCore SwiGLU only supports output stationary (stationary=0)"
    data_str += [format_scalar_definition("uint32_t", "stationary", stationary)]

    # Identity rescale parameters (pass-through for debug)
    rescale_input_zp = 0
    rescale_multiplier = 1
    rescale_output_zp = 0
    rescale_shift = 0

    data_str += [format_scalar_definition("int32_t", "rescale_input_zp", rescale_input_zp)]
    data_str += [format_scalar_definition("uint32_t", "rescale_multiplier", rescale_multiplier)]
    data_str += [format_scalar_definition("int32_t", "rescale_output_zp", rescale_output_zp)]
    data_str += [format_scalar_definition("uint32_t", "rescale_shift", rescale_shift)]

    # Data range: keep small to avoid int16 overflow in mode 0
    A_MIN, A_MAX = -3, 3
    B_MIN, B_MAX = -3, 3

    # ===================== A streamer settings (Reader 0) ====================
    data_str += [format_scalar_definition("int32_t", "Aslstride0", int(bankWidth // 8))]

    # Output stationary: K loop innermost, N middle, M outermost
    Atlbound0 = K
    Atlstride0 = int(a_len * tileSize * meshRow // 8)
    Atlbound1 = N
    Atlstride1 = 0
    Atlbound2 = M
    Atlstride2 = int(K * a_len * tileSize * meshRow // 8)

    assert Atlstride0 % (bankWidth // 8 * granularity_a) == 0
    assert Atlstride1 % (bankWidth // 8 * granularity_a) == 0
    assert Atlstride2 % (bankWidth // 8 * granularity_a) == 0

    data_str += [format_scalar_definition("int32_t", "Atlbound0", Atlbound0)]
    data_str += [format_scalar_definition("int32_t", "Atlstride0", Atlstride0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound1", Atlbound1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride1", Atlstride1)]
    data_str += [format_scalar_definition("int32_t", "Atlbound2", Atlbound2)]
    data_str += [format_scalar_definition("int32_t", "Atlstride2", Atlstride2)]
    # Extra temporal dims (unused, set to 1/0)
    data_str += [format_scalar_definition("int32_t", "Atlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride3", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound4", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride4", 0)]
    data_str += [format_scalar_definition("int32_t", "Atlbound5", 1)]
    data_str += [format_scalar_definition("int32_t", "Atlstride5", 0)]

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
    B0tlstride0 = b_len * tileSize * meshCol // 8
    B0tlbound1 = N
    B0tlstride1 = K * b_len * tileSize * meshCol // 8
    B0tlbound2 = M
    B0tlstride2 = 0

    assert B0tlstride0 % (bankWidth // 8 * granularity_b) == 0
    assert B0tlstride1 % (bankWidth // 8 * granularity_b) == 0
    assert B0tlstride2 % (bankWidth // 8 * granularity_b) == 0

    data_str += [format_scalar_definition("int32_t", "B0tlbound0", B0tlbound0)]
    data_str += [format_scalar_definition("int32_t", "B0tlstride0", B0tlstride0)]
    data_str += [format_scalar_definition("int32_t", "B0tlbound1", B0tlbound1)]
    data_str += [format_scalar_definition("int32_t", "B0tlstride1", B0tlstride1)]
    data_str += [format_scalar_definition("int32_t", "B0tlbound2", B0tlbound2)]
    data_str += [format_scalar_definition("int32_t", "B0tlstride2", B0tlstride2)]

    B_enabled_channel_CSR_num = int(math.ceil(b_array_width // bankWidth / 32))
    channel_en_B0 = [0] * B_enabled_channel_CSR_num
    channel_en_B0_bits = int((meshCol * tileSize * b_len // bankWidth + 7) // 8 * 8)
    channel_en_B0 = gen_channel_enable_CSR(channel_en_B0, channel_en_B0_bits)
    data_str += [
        "int32_t channel_en_B0[] = { " + ", ".join(map(str, channel_en_B0)) + " };"
    ]

    b0_data_length = K * N * tileSize * meshCol * b_len // 8
    data_str += [format_scalar_definition("int32_t", "b0_data_length", b0_data_length)]

    # ===================== B1 streamer settings (Reader 2) ====================
    data_str += [format_scalar_definition("int32_t", "B1slstride0", bankWidth // 8)]

    B1tlbound0 = B0tlbound0
    B1tlstride0 = B0tlstride0
    B1tlbound1 = B0tlbound1
    B1tlstride1 = B0tlstride1
    B1tlbound2 = B0tlbound2
    B1tlstride2 = B0tlstride2

    data_str += [format_scalar_definition("int32_t", "B1tlbound0", B1tlbound0)]
    data_str += [format_scalar_definition("int32_t", "B1tlstride0", B1tlstride0)]
    data_str += [format_scalar_definition("int32_t", "B1tlbound1", B1tlbound1)]
    data_str += [format_scalar_definition("int32_t", "B1tlstride1", B1tlstride1)]
    data_str += [format_scalar_definition("int32_t", "B1tlbound2", B1tlbound2)]
    data_str += [format_scalar_definition("int32_t", "B1tlstride2", B1tlstride2)]

    channel_en_B1 = [0] * B_enabled_channel_CSR_num
    channel_en_B1_bits = channel_en_B0_bits
    channel_en_B1 = gen_channel_enable_CSR(channel_en_B1, channel_en_B1_bits)
    data_str += [
        "int32_t channel_en_B1[] = { " + ", ".join(map(str, channel_en_B1)) + " };"
    ]

    b1_data_length = b0_data_length
    data_str += [format_scalar_definition("int32_t", "b1_data_length", b1_data_length)]

    # ===================== D Writer settings (Writer 0 and Writer 1) ====================
    # Output is int16 now. DataWidthOut = DataWidthD/2.
    # Each writer handles 32 channels (half of 64).
    # PostprocLanes=64, NumChunks=2, output reassembly: 64 lanes * 16b * 2 chunks = 2048b
    # Writer spatial bound = 32 channels
    # d_spatial_bound_0 = 32 (channels per writer)
    # Dtlbound0 = NumChunks = 2 (serialization of 128 int16 elems into 2 beats of 32 channels)

    PostprocLanes = snax_acc_cfg.get("snax_dual_versacore_postproc_lanes", 64)
    ElemsPerBeat_int32 = snax_versacore_serial_c_d_width // 32  # 128
    NumChunks = (ElemsPerBeat_int32 + PostprocLanes - 1) // PostprocLanes  # 2

    # Each output beat from out_assemble is DataWidthOut = 2048 bits = 128 int16 values
    # Writer D0 spatial_bound = 32 channels, each channel = bankWidth=64 bits
    # 32 channels * 64b = 2048b = DataWidthOut
    d_spatial_bound_0 = 32
    data_str += [format_scalar_definition("int32_t", "D0slstride0", bankWidth // 8)]
    data_str += [format_scalar_definition("int32_t", "D1slstride0", bankWidth // 8)]

    # int16 output: Dtlstride2 = meshRow * meshCol * 2 bytes = 256
    Dtlbound0 = 1  # No serialization needed: 32ch*64b=2048b = full beat
    Dtlstride0 = d_spatial_bound_0 * (bankWidth // 8)
    Dtlbound1 = N
    Dtlstride1 = out_elem_bits * meshRow * meshCol // 8  # = 16*8*2 = 256
    Dtlbound2 = M
    Dtlstride2 = N * out_elem_bits * meshRow * meshCol // 8

    assert Dtlstride0 % (bankWidth // 8 * granularity_c_d) == 0, \
        f"Dtlstride0={Dtlstride0} not aligned to {bankWidth // 8 * granularity_c_d}"
    assert Dtlstride1 % (bankWidth // 8 * granularity_c_d) == 0, \
        f"Dtlstride1={Dtlstride1} not aligned to {bankWidth // 8 * granularity_c_d}"
    assert Dtlstride2 % (bankWidth // 8 * granularity_c_d) == 0, \
        f"Dtlstride2={Dtlstride2} not aligned to {bankWidth // 8 * granularity_c_d}"

    data_str += [format_scalar_definition("int32_t", "Dtlbound0", Dtlbound0)]
    data_str += [format_scalar_definition("int32_t", "Dtlstride0", Dtlstride0)]
    data_str += [format_scalar_definition("int32_t", "Dtlbound1", Dtlbound1)]
    data_str += [format_scalar_definition("int32_t", "Dtlstride1", Dtlstride1)]
    data_str += [format_scalar_definition("int32_t", "Dtlbound2", Dtlbound2)]
    data_str += [format_scalar_definition("int32_t", "Dtlstride2", Dtlstride2)]
    data_str += [format_scalar_definition("int32_t", "Dtlbound3", 1)]
    data_str += [format_scalar_definition("int32_t", "Dtlstride3", 0)]

    # Channel enable for Writer 0 and Writer 1 (32 channels each)
    # DataWidthOut = 2048 bits. 2048/64 = 32 channels per writer
    D_channels_per_writer = 32
    D_enabled_channel_CSR_num = int(math.ceil(D_channels_per_writer / 32))
    channel_en_D0 = [0] * D_enabled_channel_CSR_num
    channel_en_D0_bits = D_channels_per_writer
    channel_en_D0 = gen_channel_enable_CSR(channel_en_D0, channel_en_D0_bits)
    data_str += [
        "int32_t channel_en_D0[] = { " + ", ".join(map(str, channel_en_D0)) + " };"
    ]
    channel_en_D1 = [0] * D_enabled_channel_CSR_num
    channel_en_D1 = gen_channel_enable_CSR(channel_en_D1, channel_en_D0_bits)
    data_str += [
        "int32_t channel_en_D1[] = { " + ", ".join(map(str, channel_en_D1)) + " };"
    ]

    # Output data length in bytes (mode 0: int16)
    mode0_output_elems = M * N * meshRow * meshCol
    mode0_d_data_length = mode0_output_elems * out_elem_bits // 8
    data_str += [format_scalar_definition("int32_t", "mode0_d_data_length", mode0_d_data_length)]
    data_str += [format_scalar_definition("int32_t", "mode0_output_elems", mode0_output_elems)]

    # ===================== Base addresses ====================================
    delta_local_a = 0
    delta_local_a = align_wide_addr(delta_local_a, granularity_a * bankWidth // 8)

    delta_local_b0 = K * M * (meshRow * tileSize * a_len // 8)
    delta_local_b0 = align_wide_addr(delta_local_b0, granularity_b * bankWidth // 8)

    delta_local_b1 = delta_local_b0 + K * N * (meshCol * tileSize * b_len // 8)
    delta_local_b1 = align_wide_addr(delta_local_b1, granularity_b * bankWidth // 8)

    # Mode 0 output (D0 only in mode 0)
    delta_local_d0 = delta_local_b1 + K * N * (meshCol * tileSize * b_len // 8)
    delta_local_d0 = align_wide_addr(delta_local_d0, granularity_c_d * bankWidth // 8)

    # Mode 0 uses only Writer 0; Writer 1 is idle. D1 points to a dummy location.
    delta_local_d1_mode0 = delta_local_d0 + mode0_d_data_length
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

    # Full random test (small values to avoid overflow)
    A = np.random.randint(A_MIN, A_MAX + 1, size=(M, K, meshRow, tileSize)).reshape(-1)
    W = np.random.randint(B_MIN, B_MAX + 1, size=(K, N, tileSize, meshCol)).reshape(-1)
    V = np.random.randint(B_MIN, B_MAX + 1, size=(K, N, tileSize, meshCol)).reshape(-1)

    data_str += [format_vector_definition("int8_t", "A", A)]
    data_str += [format_vector_definition("int8_t", "W", W)]
    data_str += [format_vector_definition("int8_t", "V", V)]

    # ===================== Mode 0 Golden Model ==============================
    # Step 1: VC0 = A @ W (int8 x int8 -> int32)
    C_zeros = np.zeros((M, N, meshRow, meshCol)).reshape(-1).astype(np.int32)

    vc0_int32 = block_gemm_golden_model(
        M, K, N, meshRow, tileSize, meshCol,
        A, W, subtraction_a, subtraction_b, C_zeros
    )

    # Step 2: VC1 = A @ V (int8 x int8 -> int32)
    vc1_int32 = block_gemm_golden_model(
        M, K, N, meshRow, tileSize, meshCol,
        A, V, subtraction_a, subtraction_b, C_zeros
    )

    # Step 3: RescaleDown0 (int32 -> int16) with identity params
    vc0_int16 = rescale_down_32to16(vc0_int32, rescale_input_zp, rescale_multiplier,
                                     rescale_output_zp, rescale_shift)

    # Step 4: Shifter 6-stage (SiLU placeholder): arithmetic right shift by 2
    vc0_silu = arithmetic_right_shift_int16(vc0_int16, 2)

    # Step 5: RescaleDown1 (int32 -> int16) with identity params
    vc1_int16 = rescale_down_32to16(vc1_int32, rescale_input_zp, rescale_multiplier,
                                     rescale_output_zp, rescale_shift)

    # Step 6: Element-wise multiply (int16 x int16 -> int32)
    mul_int32 = vc0_silu.astype(np.int32) * vc1_int16.astype(np.int32)

    # Step 7: RescaleDown_mul (int32 -> int16) with identity params
    mode0_out = rescale_down_32to16(mul_int32, rescale_input_zp, rescale_multiplier,
                                     rescale_output_zp, rescale_shift)

    data_str += [format_vector_definition("int16_t", "mode0_golden", mode0_out)]

    # ===================== Mode 1 (GEMM) Golden Model ===========================
    # Step 1: Cast mode 0 output int16 -> int8 (saturating clamp)
    cast_a1 = np.clip(mode0_out.astype(np.int32), -128, 127).astype(np.int8)

    # Step 2: Generate W2 weight matrix, split by columns into left/right halves
    # W2 total: (N_total=2, K1=2, meshCol=8, tileSize=8) -> 256 elements
    # W2_left: first N1=1 tiles, W2_right: last N1=1 tiles
    M1 = M   # = 2
    K1 = N   # mode 0's N becomes mode 1's K (reinterpret output shape)
    N1 = 1   # per VersaCore (total 2 output column tiles across both VCs)
    N_total = 2 * N1  # 2 halves

    W2 = np.random.randint(B_MIN, B_MAX + 1, size=(N_total, K1, meshCol, tileSize)).reshape(-1)
    W2_left = W2[:N1 * K1 * meshCol * tileSize]    # first 128 elements
    W2_right = W2[N1 * K1 * meshCol * tileSize:]   # last 128 elements

    data_str += [format_vector_definition("int8_t", "W2_left", W2_left)]
    data_str += [format_vector_definition("int8_t", "W2_right", W2_right)]

    # Step 3: Compute mode 1 golden (A1 @ W2_left, A1 @ W2_right)
    C_zeros_m1 = np.zeros((M1, N1, meshRow, meshCol)).reshape(-1).astype(np.int32)

    golden_d0_int32 = block_gemm_golden_model(
        M1, K1, N1, meshRow, tileSize, meshCol,
        cast_a1, W2_left, subtraction_a, subtraction_b, C_zeros_m1
    )
    golden_d1_int32 = block_gemm_golden_model(
        M1, K1, N1, meshRow, tileSize, meshCol,
        cast_a1, W2_right, subtraction_a, subtraction_b, C_zeros_m1
    )

    # Step 4: RescaleDown (int32 -> int16) with identity params
    mode1_golden_d0 = rescale_down_32to16(golden_d0_int32, rescale_input_zp,
                                           rescale_multiplier, rescale_output_zp, rescale_shift)
    mode1_golden_d1 = rescale_down_32to16(golden_d1_int32, rescale_input_zp,
                                           rescale_multiplier, rescale_output_zp, rescale_shift)

    data_str += [format_vector_definition("int16_t", "mode1_golden_d0", mode1_golden_d0)]
    data_str += [format_vector_definition("int16_t", "mode1_golden_d1", mode1_golden_d1)]

    mode1_output_elems = M1 * N1 * meshRow * meshCol  # 256 per writer
    data_str += [format_scalar_definition("int32_t", "mode1_output_elems", mode1_output_elems)]

    # Mode 1 dimensions
    data_str += [format_scalar_definition("uint32_t", "M1", M1)]
    data_str += [format_scalar_definition("uint32_t", "K1", K1)]
    data_str += [format_scalar_definition("uint32_t", "N1", N1)]

    # ===================== Mode 1 Streamer params ===============================
    # Reader A (same structure as mode 0 but with M1, K1, N1 dims)
    M1_Atlbound0 = K1
    M1_Atlstride0 = int(a_len * tileSize * meshRow // 8)  # 128
    M1_Atlbound1 = N1
    M1_Atlstride1 = 0
    M1_Atlbound2 = M1
    M1_Atlstride2 = int(K1 * a_len * tileSize * meshRow // 8)  # 256

    data_str += [format_scalar_definition("int32_t", "M1_Atlbound0", M1_Atlbound0)]
    data_str += [format_scalar_definition("int32_t", "M1_Atlstride0", M1_Atlstride0)]
    data_str += [format_scalar_definition("int32_t", "M1_Atlbound1", M1_Atlbound1)]
    data_str += [format_scalar_definition("int32_t", "M1_Atlstride1", M1_Atlstride1)]
    data_str += [format_scalar_definition("int32_t", "M1_Atlbound2", M1_Atlbound2)]
    data_str += [format_scalar_definition("int32_t", "M1_Atlstride2", M1_Atlstride2)]

    # Reader B0 (W2_left)
    M1_B0tlbound0 = K1
    M1_B0tlstride0 = b_len * tileSize * meshCol // 8  # 64
    M1_B0tlbound1 = N1
    M1_B0tlstride1 = K1 * b_len * tileSize * meshCol // 8  # 128 (but N1=1 so doesn't matter)
    M1_B0tlbound2 = M1
    M1_B0tlstride2 = 0  # output stationary: B doesn't move with M

    data_str += [format_scalar_definition("int32_t", "M1_B0tlbound0", M1_B0tlbound0)]
    data_str += [format_scalar_definition("int32_t", "M1_B0tlstride0", M1_B0tlstride0)]
    data_str += [format_scalar_definition("int32_t", "M1_B0tlbound1", M1_B0tlbound1)]
    data_str += [format_scalar_definition("int32_t", "M1_B0tlstride1", M1_B0tlstride1)]
    data_str += [format_scalar_definition("int32_t", "M1_B0tlbound2", M1_B0tlbound2)]
    data_str += [format_scalar_definition("int32_t", "M1_B0tlstride2", M1_B0tlstride2)]

    # Reader B1 (W2_right) — same bounds/strides as B0
    M1_B1tlbound0 = M1_B0tlbound0
    M1_B1tlstride0 = M1_B0tlstride0
    M1_B1tlbound1 = M1_B0tlbound1
    M1_B1tlstride1 = M1_B0tlstride1
    M1_B1tlbound2 = M1_B0tlbound2
    M1_B1tlstride2 = M1_B0tlstride2

    data_str += [format_scalar_definition("int32_t", "M1_B1tlbound0", M1_B1tlbound0)]
    data_str += [format_scalar_definition("int32_t", "M1_B1tlstride0", M1_B1tlstride0)]
    data_str += [format_scalar_definition("int32_t", "M1_B1tlbound1", M1_B1tlbound1)]
    data_str += [format_scalar_definition("int32_t", "M1_B1tlstride1", M1_B1tlstride1)]
    data_str += [format_scalar_definition("int32_t", "M1_B1tlbound2", M1_B1tlbound2)]
    data_str += [format_scalar_definition("int32_t", "M1_B1tlstride2", M1_B1tlstride2)]

    # Writer D0/D1 for mode 1
    M1_Dtlbound0 = 1  # no serialization
    M1_Dtlstride0 = d_spatial_bound_0 * (bankWidth // 8)  # 256
    M1_Dtlbound1 = N1  # 1
    M1_Dtlstride1 = out_elem_bits * meshRow * meshCol // 8  # 256
    M1_Dtlbound2 = M1  # 2
    M1_Dtlstride2 = N1 * out_elem_bits * meshRow * meshCol // 8  # 256
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

    # ===================== Mode 1 memory offsets ================================
    # W2_left and W2_right are loaded at start
    w2l_data_length = N1 * K1 * meshCol * tileSize * b_len // 8  # 128 bytes
    w2r_data_length = w2l_data_length

    delta_local_w2l = delta_local_d1_mode0 + mode0_d_data_length
    delta_local_w2l = align_wide_addr(delta_local_w2l, granularity_b * bankWidth // 8)

    delta_local_w2r = delta_local_w2l + w2l_data_length
    delta_local_w2r = align_wide_addr(delta_local_w2r, granularity_b * bankWidth // 8)

    # Mode 1 outputs (separate from mode 0 to allow independent checking)
    mode1_d_data_length = mode1_output_elems * out_elem_bits // 8  # 512 bytes

    delta_local_mode1_d0 = delta_local_w2r + w2r_data_length
    delta_local_mode1_d0 = align_wide_addr(delta_local_mode1_d0, granularity_c_d * bankWidth // 8)

    delta_local_mode1_d1 = delta_local_mode1_d0 + mode1_d_data_length
    delta_local_mode1_d1 = align_wide_addr(delta_local_mode1_d1, granularity_c_d * bankWidth // 8)

    data_str += [format_scalar_definition("int32_t", "w2l_data_length", w2l_data_length)]
    data_str += [format_scalar_definition("int32_t", "w2r_data_length", w2r_data_length)]
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
    parser = argparse.ArgumentParser(description="Generate data for dual VersaCore SwiGLU test")
    parser.add_argument(
        "--swcfg",
        type=pathlib.Path,
        required=True,
        help="Select param config file",
    )
    parser.add_argument(
        "--hwcfg",
        type=pathlib.Path,
        required=True,
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
