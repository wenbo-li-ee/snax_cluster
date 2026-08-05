#!/usr/bin/env python3
"""
Golden reference for the `silu_out16_balanced_pkg.sv` hardware configuration.

Purpose
-------
This file provides a standalone Python implementation of the exact fixed-point
input -> output datapath for the following SiLU approximation configuration:

- input        : signed 16-bit, frac 11
- breakpoints  : signed 16-bit, frac 11
- output       : signed 16-bit, frac 11
- stage0       : signed 14-bit, frac 12
- stage1_mul   : signed 15-bit, frac 11

The intent is to support hardware work:

- generate golden vectors for RTL simulation
- compare a DUT output against a software bit-true reference
- inspect intermediate values when debugging mismatches

This implementation is intentionally self-contained:

- no imports from the rest of `silu_fit_hardware`
- no external parameter files
- all constants are copied into this file so it can be dropped into a simple
  testbench flow without additional dependencies

Relationship to the RTL
-----------------------
The arithmetic here matches the intended datapath structure:

    t0 = a2 * x + a1
    y  = t0 * x + a0

using the same fixed-point rules as the RTL:

- signed two's-complement integers
- arithmetic right shifts for frac alignment
- saturation on overflow
- no rounding-to-nearest; all requantization uses truncation

The segment coefficients and breakpoints match:

- `silu_out16_balanced_pkg.sv`

The eight fitted polynomials cover the closed interval [-8, 6]. Inputs below
-8 bypass to zero, while inputs above 6 bypass the polynomial and return the
input code unchanged.

Recommended use
---------------
For hardware verification, the most useful function is:

    silu_out16_balanced_eval_q(x_q: int) -> int

where:

- `x_q` is the signed integer fixed-point input code in Q16.11
- return value is the signed integer fixed-point output code in Q16.11

If you want more visibility into the pipeline, use:

    silu_out16_balanced_eval_q_debug(x_q: int) -> dict

which returns the final output plus intermediate values such as:

- selected segment
- stage0 output
- stage1 multiply result

Quick example
-------------
    >>> x_q = 2048   # 1.0 in Q16.11
    >>> y_q = silu_out16_balanced_eval_q(x_q)
    >>> y_real = fixed_to_real(y_q, OUTPUT_FRAC)
    >>> print(y_q, y_real)

"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Top-level fixed-point format constants
# ---------------------------------------------------------------------------

NUM_SEGMENTS = 8
NUM_BREAKPOINTS = 9

INPUT_WIDTH = 16
INPUT_FRAC = 11

BREAKPOINT_WIDTH = 16
BREAKPOINT_FRAC = 11

OUTPUT_WIDTH = 16
OUTPUT_FRAC = 11

A0_WIDTH = 30
A0_FRAC = 22

A1_WIDTH = 22
A1_FRAC = 16

A2_WIDTH = 18
A2_FRAC = 14

STAGE0_WIDTH = 14
STAGE0_FRAC = 12

STAGE1_MUL_WIDTH = 15
STAGE1_MUL_FRAC = 11


# ---------------------------------------------------------------------------
# Quantized breakpoints and coefficients
# These values were produced from the selected balanced configuration.
# Coefficient order is ascending: [a0, a1, a2].
# ---------------------------------------------------------------------------

BREAKPOINTS_Q = [
    -16384,
    -12782,
    -9219,
    -4105,
    -1951,
    1938,
    4080,
    8644,
    12288,
]

SEG_A0_Q = [
    -686136,
    -1539297,
    -2150798,
    -516228,
    13532,
    -506623,
    -2117952,
    -1771311,
]

SEG_A1_Q = [
    -2561,
    -6946,
    -10761,
    14890,
    32771,
    50423,
    75931,
    73908,
]

SEG_A2_Q = [
    -38,
    -128,
    -222,
    1381,
    3717,
    1400,
    -206,
    -163,
]


# ---------------------------------------------------------------------------
# Small fixed-point helpers
# ---------------------------------------------------------------------------

def max_signed(width: int) -> int:
    """Largest signed integer representable in `width` bits."""
    return (1 << (width - 1)) - 1


def min_signed(width: int) -> int:
    """Smallest signed integer representable in `width` bits."""
    return -(1 << (width - 1))


def saturate_signed(value: int, width: int) -> int:
    """Saturate an integer into the signed range representable by `width` bits."""
    if value > max_signed(width):
        return max_signed(width)
    if value < min_signed(width):
        return min_signed(width)
    return value


def align_frac(value: int, src_frac: int, dst_frac: int) -> int:
    """
    Convert an integer code from one fixed-point frac to another.

    This matches the RTL behavior:
    - if the destination has fewer fractional bits, use arithmetic right shift
    - if the destination has more fractional bits, use left shift
    - truncation only, no rounding-to-nearest
    """
    frac_delta = dst_frac - src_frac
    if frac_delta >= 0:
        return value << frac_delta
    return value >> (-frac_delta)


def fixed_to_real(value_q: int, frac: int) -> float:
    """Convert a signed fixed-point integer code to a Python float."""
    return float(value_q) / float(1 << frac)


def real_to_fixed_trunc(value: float, frac: int) -> int:
    """
    Helper for ad-hoc experiments only.

    This is not needed for the golden path itself, but is convenient if you want
    to feed a real-valued sample into the Q-format datapath manually.
    """
    scaled = value * float(1 << frac)
    if scaled >= 0:
        return int(scaled)
    return -int(-scaled)


# ---------------------------------------------------------------------------
# Segment selection
# ---------------------------------------------------------------------------

def segment_index_from_x(x_q: int) -> int:
    """
    Return the selected piecewise segment index from the quantized input code.

    This matches the tree structure used in the existing SV package.
    """
    if x_q < BREAKPOINTS_Q[4]:
        if x_q < BREAKPOINTS_Q[2]:
            return 0 if x_q < BREAKPOINTS_Q[1] else 1
        return 2 if x_q < BREAKPOINTS_Q[3] else 3
    if x_q < BREAKPOINTS_Q[6]:
        return 4 if x_q < BREAKPOINTS_Q[5] else 5
    return 6 if x_q < BREAKPOINTS_Q[7] else 7


# ---------------------------------------------------------------------------
# Horner datapath for the chosen balanced configuration
# ---------------------------------------------------------------------------

def eval_stage0(x_q: int, a2_q: int, a1_q: int) -> int:
    """
    Evaluate the first Horner stage:

        t0 = a2 * x + a1

    Formats:
    - a2:       18/14
    - x:        16/11
    - a1:       22/16
    - t0:       14/12
    """
    raw_product = a2_q * x_q
    mul_requant = align_frac(raw_product, A2_FRAC + INPUT_FRAC, STAGE0_FRAC)
    mul_sat = saturate_signed(mul_requant, STAGE0_WIDTH)

    mul_for_add = align_frac(mul_sat, STAGE0_FRAC, STAGE0_FRAC)
    a1_for_add = align_frac(a1_q, A1_FRAC, STAGE0_FRAC)
    raw_sum = mul_for_add + a1_for_add
    add_sat = saturate_signed(raw_sum, STAGE0_WIDTH)

    # Stage0 output format equals stage0 add format for this configuration.
    return saturate_signed(align_frac(add_sat, STAGE0_FRAC, STAGE0_FRAC), STAGE0_WIDTH)


def eval_stage1(x_q: int, t0_q: int, a0_q: int) -> tuple[int, int]:
    """
    Evaluate the second Horner stage:

        y = t0 * x + a0

    Returns:
    - stage1_mul_q: quantized second-stage multiply output in 15/11
    - y_q: final output in 16/11
    """
    raw_product = t0_q * x_q
    mul_requant = align_frac(raw_product, STAGE0_FRAC + INPUT_FRAC, STAGE1_MUL_FRAC)
    stage1_mul_q = saturate_signed(mul_requant, STAGE1_MUL_WIDTH)

    mul_for_add = align_frac(stage1_mul_q, STAGE1_MUL_FRAC, OUTPUT_FRAC)
    a0_for_add = align_frac(a0_q, A0_FRAC, OUTPUT_FRAC)
    raw_sum = mul_for_add + a0_for_add
    add_sat = saturate_signed(raw_sum, OUTPUT_WIDTH)

    # Final stage output format equals top-level output format.
    y_q = saturate_signed(align_frac(add_sat, OUTPUT_FRAC, OUTPUT_FRAC), OUTPUT_WIDTH)
    return stage1_mul_q, y_q


# ---------------------------------------------------------------------------
# Public golden reference functions
# ---------------------------------------------------------------------------

def silu_out16_balanced_eval_q(x_q: int) -> int:
    """
    Bit-true fixed-point SiLU approximation for the balanced out16 configuration.

    Input:
    - `x_q`: signed integer code in Q16.11

    Output:
    - signed integer code in Q16.11

    Notes:
    - `x_q` is interpreted as already-quantized hardware input data
    - values outside the 16-bit signed range are first saturated into 16 bits
    - the returned value is the exact golden output integer code for this config
    """
    x_q = saturate_signed(int(x_q), INPUT_WIDTH)
    if x_q < BREAKPOINTS_Q[0]:
        return 0
    if x_q > BREAKPOINTS_Q[-1]:
        return x_q

    seg_idx = segment_index_from_x(x_q)
    t0_q = eval_stage0(x_q, SEG_A2_Q[seg_idx], SEG_A1_Q[seg_idx])
    _stage1_mul_q, y_q = eval_stage1(x_q, t0_q, SEG_A0_Q[seg_idx])
    return y_q


def silu_out16_balanced_eval_q_debug(x_q: int) -> dict:
    """
    Debug-oriented variant of the golden reference.

    Returns a dictionary containing:
    - input code and real value
    - selected segment
    - selected coefficients
    - stage0 output
    - stage1 multiply output
    - final output

    This is useful when comparing an RTL waveform against software.
    """
    x_q_sat = saturate_signed(int(x_q), INPUT_WIDTH)
    seg_idx = segment_index_from_x(x_q_sat)
    a0_q = SEG_A0_Q[seg_idx]
    a1_q = SEG_A1_Q[seg_idx]
    a2_q = SEG_A2_Q[seg_idx]

    t0_q = eval_stage0(x_q_sat, a2_q, a1_q)
    stage1_mul_q, polynomial_y_q = eval_stage1(x_q_sat, t0_q, a0_q)

    if x_q_sat < BREAKPOINTS_Q[0]:
        bypass = "zero"
        y_q = 0
    elif x_q_sat > BREAKPOINTS_Q[-1]:
        bypass = "input"
        y_q = x_q_sat
    else:
        bypass = None
        y_q = polynomial_y_q

    return {
        "x_q": x_q_sat,
        "x_real": fixed_to_real(x_q_sat, INPUT_FRAC),
        "segment_index": seg_idx,
        "bypass": bypass,
        "breakpoints_q": list(BREAKPOINTS_Q),
        "coeff_q": {
            "a0": a0_q,
            "a1": a1_q,
            "a2": a2_q,
        },
        "stage0_q": t0_q,
        "stage0_real": fixed_to_real(t0_q, STAGE0_FRAC),
        "stage1_mul_q": stage1_mul_q,
        "stage1_mul_real": fixed_to_real(stage1_mul_q, STAGE1_MUL_FRAC),
        "polynomial_y_q": polynomial_y_q,
        "y_q": y_q,
        "y_real": fixed_to_real(y_q, OUTPUT_FRAC),
    }


def silu_out16_balanced_eval_real(x_real: float) -> float:
    """
    Convenience wrapper for quick software inspection from a real-valued input.

    This is *not* how hardware feeds the datapath, but it is handy during debug.
    The input is first quantized into Q16.11 using truncation toward zero, then
    passed through the exact integer datapath, and finally converted back to float.
    """
    x_q = saturate_signed(real_to_fixed_trunc(float(x_real), INPUT_FRAC), INPUT_WIDTH)
    y_q = silu_out16_balanced_eval_q(x_q)
    return fixed_to_real(y_q, OUTPUT_FRAC)


if __name__ == "__main__":
    # Tiny smoke demo for manual use from the shell:
    #
    #   python silu_out16_balanced_golden.py
    #
    demo_inputs_q = [-4096, -2048, -1024, 0, 1024, 2048, 4096]
    for x_q in demo_inputs_q:
        dbg = silu_out16_balanced_eval_q_debug(x_q)
        print(
            f"x_q={dbg['x_q']:6d} "
            f"x_real={dbg['x_real']: .6f} "
            f"seg={dbg['segment_index']} "
            f"stage0_q={dbg['stage0_q']:6d} "
            f"stage1_mul_q={dbg['stage1_mul_q']:6d} "
            f"y_q={dbg['y_q']:6d} "
            f"y_real={dbg['y_real']: .6f}"
        )
