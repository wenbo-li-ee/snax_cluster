#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys

import numpy as np

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../../../../../util/sim/")
)
from data_utils import format_scalar_definition, format_vector_definition  # noqa: E402

MIN = -50
MAX = 50

TYPE_MAP = {
    8: ("int8_t", np.int8),
    16: ("int16_t", np.int16),
    32: ("int32_t", np.int32),
}


def wrap_signed(values, width):
    mask = (1 << width) - 1
    sign_bit = 1 << (width - 1)
    wrapped = np.bitwise_and(values.astype(np.int64), mask)
    return ((wrapped ^ sign_bit) - sign_bit).astype(np.int64)


def pack_lanes(values, width, num_pe):
    words_per_iter = (num_pe * width + 63) // 64
    packed_words = []
    mask = (1 << width) - 1

    for base in range(0, len(values), num_pe):
        beat = 0
        lanes = values[base:base + num_pe]
        for lane, value in enumerate(lanes):
            beat |= (int(value) & mask) << (lane * width)
        for word_idx in range(words_per_iter):
            packed_words.append(np.uint64((beat >> (64 * word_idx)) & ((1 << 64) - 1)))

    return np.array(packed_words, dtype=np.uint64), words_per_iter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--input-width", type=int, required=True, choices=sorted(TYPE_MAP))
    parser.add_argument("--output-width", type=int, required=True, choices=sorted(TYPE_MAP))
    parser.add_argument("--num-pe", type=int, required=True)
    args = parser.parse_args()

    if args.length % args.num_pe != 0:
        raise ValueError("length must be divisible by num_pe")

    input_c_type, _ = TYPE_MAP[args.input_width]
    output_c_type, _ = TYPE_MAP[args.output_width]

    a = wrap_signed(np.random.randint(MIN, MAX, args.length, dtype=np.int64), args.input_width)
    b = wrap_signed(np.random.randint(MIN, MAX, args.length, dtype=np.int64), args.input_width)

    out_add = wrap_signed(a + b, args.output_width)
    out_mul = wrap_signed(a * b, args.output_width)

    a_packed, input_words_per_iter = pack_lanes(a, args.input_width, args.num_pe)
    b_packed, _ = pack_lanes(b, args.input_width, args.num_pe)
    out_add_packed, output_words_per_iter = pack_lanes(out_add, args.output_width, args.num_pe)
    out_mul_packed, _ = pack_lanes(out_mul, args.output_width, args.num_pe)

    loop_iter = args.length // args.num_pe

    f_str = "\n\n".join(
        [
            "#include <stdint.h>",
            format_scalar_definition("uint32_t", "EWU_NUM_PE", args.num_pe),
            format_scalar_definition("uint32_t", "EWU_INPUT_WIDTH", args.input_width),
            format_scalar_definition("uint32_t", "EWU_OUTPUT_WIDTH", args.output_width),
            format_scalar_definition("uint32_t", "DATA_LEN", args.length),
            format_scalar_definition("uint32_t", "LOOP_ITER", loop_iter),
            format_scalar_definition("uint32_t", "INPUT_WORDS_PER_ITER", input_words_per_iter),
            format_scalar_definition("uint32_t", "OUTPUT_WORDS_PER_ITER", output_words_per_iter),
            format_scalar_definition("uint32_t", "INPUT_WORDS", len(a_packed)),
            format_scalar_definition("uint32_t", "OUTPUT_WORDS", len(out_add_packed)),
            format_vector_definition(input_c_type, "A", a.astype(np.int64)),
            format_vector_definition(input_c_type, "B", b.astype(np.int64)),
            format_vector_definition(output_c_type, "OUT_ADD", out_add.astype(np.int64)),
            format_vector_definition(output_c_type, "OUT_MUL", out_mul.astype(np.int64)),
            format_vector_definition("uint64_t", "A_PACKED", a_packed.astype(np.uint64)),
            format_vector_definition("uint64_t", "B_PACKED", b_packed.astype(np.uint64)),
            format_vector_definition("uint64_t", "OUT_ADD_PACKED", out_add_packed.astype(np.uint64)),
            format_vector_definition("uint64_t", "OUT_MUL_PACKED", out_mul_packed.astype(np.uint64)),
        ]
    )
    print(f_str + "\n")


if __name__ == "__main__":
    sys.exit(main())
