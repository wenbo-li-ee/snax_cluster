#!/usr/bin/env python3

# Copyright 2025 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Data generator for dual VersaCore SwiGLU test
# Golden model: output = (A @ W >> 2) + (A @ V >> 2)
# where W = B0, V = B1, and >> is arithmetic right shift

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


def arithmetic_right_shift_int32(val, shift):
    """Python arithmetic right shift for signed 32-bit integer."""
    # Ensure val is treated as signed 32-bit
    if val >= (1 << 31):
        val -= (1 << 32)
    result = val >> shift
    # Clamp to 32-bit signed range
    if result >= (1 << 31):
        result -= (1 << 32)
    if result < -(1 << 31):
        result = -(1 << 31)
    return result


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

    granularity_a = snax_acc_cfg.get("granularity_a", 1)
    granularity_b = snax_acc_cfg.get("granularity_b", 1)
    granularity_c_d = snax_acc_cfg.get("granularity_c_d", 1)

    data_str += [format_scalar_definition("uint32_t", "meshRow", meshRow)]
    data_str += [format_scalar_definition("uint32_t", "tileSize", tileSize)]
    data_str += [format_scalar_definition("uint32_t", "meshCol", meshCol)]

    stationary = kwargs["stationary"]
    assert stationary == 0, "Dual VersaCore SwiGLU only supports output stationary (stationary=0)"
    data_str += [format_scalar_definition("uint32_t", "stationary", stationary)]

    # Integer data types only for now
    A_MIN, A_MAX = 1, 2
    B_MIN, B_MAX = 1, 2

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

    # Output stationary
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
    # B1 has same structure as B0 but different base address
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

    # ===================== D streamer settings (Writer 0) ====================
    data_str += [format_scalar_definition("int32_t", "Dslstride0", bankWidth // 8)]

    if meshCol * meshRow * c_len >= snax_versacore_serial_c_d_width:
        d_spatial_bound_0 = snax_versacore_serial_c_d_width // bankWidth
    else:
        d_spatial_bound_0 = meshCol * meshRow * c_len // bankWidth

    Dtlbound0 = max(1, meshCol * meshRow * c_len / snax_versacore_serial_c_d_width)
    Dtlstride0 = d_spatial_bound_0 * (bankWidth // 8)
    Dtlbound1 = N
    Dtlstride1 = c_len * meshRow * meshCol // 8
    Dtlbound2 = M
    Dtlstride2 = N * c_len * meshRow * meshCol // 8
    Dtlbound3 = 1
    Dtlstride3 = 0

    assert Dtlstride0 % (bankWidth // 8 * granularity_c_d) == 0
    assert Dtlstride1 % (bankWidth // 8 * granularity_c_d) == 0
    assert Dtlstride2 % (bankWidth // 8 * granularity_c_d) == 0
    assert Dtlstride3 % (bankWidth // 8 * granularity_c_d) == 0

    data_str += [format_scalar_definition("int32_t", "Dtlbound0", Dtlbound0)]
    data_str += [format_scalar_definition("int32_t", "Dtlstride0", Dtlstride0)]
    data_str += [format_scalar_definition("int32_t", "Dtlbound1", Dtlbound1)]
    data_str += [format_scalar_definition("int32_t", "Dtlstride1", Dtlstride1)]
    data_str += [format_scalar_definition("int32_t", "Dtlbound2", Dtlbound2)]
    data_str += [format_scalar_definition("int32_t", "Dtlstride2", Dtlstride2)]
    data_str += [format_scalar_definition("int32_t", "Dtlbound3", Dtlbound3)]
    data_str += [format_scalar_definition("int32_t", "Dtlstride3", Dtlstride3)]

    D_enabled_channel_CSR_num = int(
        math.ceil(snax_versacore_serial_c_d_width // bankWidth / 32)
    )
    channel_en_D = [0] * D_enabled_channel_CSR_num
    channel_en_D_bits = int((meshRow * meshCol * c_len // bankWidth + 7) // 8 * 8)
    channel_en_D = gen_channel_enable_CSR(channel_en_D, channel_en_D_bits)
    data_str += [
        "int32_t channel_en_D[] = { " + ", ".join(map(str, channel_en_D)) + " };"
    ]

    d_data_length = M * N * meshRow * meshCol
    data_str += [
        format_scalar_definition("int32_t", "d_data_length", d_data_length * c_len // 8)
    ]

    # ===================== Base addresses ====================================
    delta_local_a = 0
    delta_local_a = align_wide_addr(delta_local_a, granularity_a * bankWidth // 8)

    delta_local_b0 = K * M * (meshRow * tileSize * a_len // 8)
    delta_local_b0 = align_wide_addr(delta_local_b0, granularity_b * bankWidth // 8)

    delta_local_b1 = delta_local_b0 + K * N * (meshCol * tileSize * b_len // 8)
    delta_local_b1 = align_wide_addr(delta_local_b1, granularity_b * bankWidth // 8)

    delta_local_d = delta_local_b1 + K * N * (meshCol * tileSize * b_len // 8)
    delta_local_d = align_wide_addr(delta_local_d, granularity_c_d * bankWidth // 8)

    data_str += [format_scalar_definition("int32_t", "delta_local_a", delta_local_a)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b0", delta_local_b0)]
    data_str += [format_scalar_definition("int32_t", "delta_local_b1", delta_local_b1)]
    data_str += [format_scalar_definition("int32_t", "delta_local_d", delta_local_d)]

    # ===================== Test data generation ==============================
    subtraction_a = 0
    subtraction_b = 0
    data_str += [format_scalar_definition("int8_t", "subtraction_a", subtraction_a)]
    data_str += [format_scalar_definition("int8_t", "subtraction_b", subtraction_b)]

    # Full random test
    A = np.random.randint(A_MIN, A_MAX + 1, size=(M, K, meshRow, tileSize)).reshape(-1)
    W = np.random.randint(B_MIN, B_MAX + 1, size=(K, N, tileSize, meshCol)).reshape(-1)
    V = np.random.randint(B_MIN, B_MAX + 1, size=(K, N, tileSize, meshCol)).reshape(-1)

    data_str += [format_vector_definition("int8_t", "A", A)]
    data_str += [format_vector_definition("int8_t", "W", W)]
    data_str += [format_vector_definition("int8_t", "V", V)]

    # Golden model: output = (A @ W >> 2) + (A @ V >> 2)
    # Use block_gemm_golden_model with C=0
    C_zeros = np.zeros((M, N, meshRow, meshCol)).reshape(-1).astype(np.int32)

    xW = block_gemm_golden_model(
        M, K, N, meshRow, tileSize, meshCol,
        A, W, subtraction_a, subtraction_b, C_zeros
    )

    xV = block_gemm_golden_model(
        M, K, N, meshRow, tileSize, meshCol,
        A, V, subtraction_a, subtraction_b, C_zeros
    )

    # Arithmetic right shift by 2 for each int32 element
    xW_shifted = np.array([arithmetic_right_shift_int32(int(x), 2) for x in xW], dtype=np.int32)
    xV_shifted = np.array([arithmetic_right_shift_int32(int(x), 2) for x in xV], dtype=np.int32)

    # Element-wise addition
    D = (xW_shifted.astype(np.int64) + xV_shifted.astype(np.int64)).astype(np.int32)

    data_str += [format_vector_definition("int32_t", "D", D)]

    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_A", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_B0", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_B1", 0)]
    data_str += [format_scalar_definition("int32_t", "set_addr_remap_index_D", 0)]

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
