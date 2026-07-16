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

LAYOUTS = [
    # V00: copied baseline, used to reproduce the existing MoE-small L15 app.
    {"id": 0, "name": "v00_l15_baseline_pad32_b1bank34", "a_pad": 32,
     "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
    # V01: remove TCDM token padding while keeping the same weight/base colors.
    {"id": 1, "name": "v01_compact_token_no_pad_b1bank34", "a_pad": 0,
     "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
    # V02: keep the baseline A stride, move B1 to a high clean bank phase.
    {"id": 2, "name": "v02_pad32_b1bank52", "a_pad": 32,
     "b1_color": 416, "w2l_color": 128, "m1d0_color": 256},
    # V03: combine a larger A row phase with the high B1 phase.
    {"id": 3, "name": "v03_pad56_b1bank52", "a_pad": 56,
     "b1_color": 416, "w2l_color": 128, "m1d0_color": 256},
    # V04: smaller padding than baseline, to test whether MoE-small needs 32B.
    {"id": 4, "name": "v04_pad16_b1bank34", "a_pad": 16,
     "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
    # Batch 2: A stride phase sweep around the successful 2080B row.
    {"id": 5, "name": "v05_pad08_b1bank34", "a_pad": 8,
     "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
    {"id": 6, "name": "v06_pad24_b1bank34", "a_pad": 24,
     "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
    {"id": 7, "name": "v07_pad40_b1bank34", "a_pad": 40,
     "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
    {"id": 8, "name": "v08_pad48_b1bank34", "a_pad": 48,
     "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
    {"id": 9, "name": "v09_pad64_b1bank34", "a_pad": 64,
     "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
    # Batch 3: B1 bank-color sweep at the best-known A stride.
    {"id": 10, "name": "v10_pad32_b1bank18", "a_pad": 32,
     "b1_color": 144, "w2l_color": 128, "m1d0_color": 256},
    {"id": 11, "name": "v11_pad32_b1bank26", "a_pad": 32,
     "b1_color": 208, "w2l_color": 128, "m1d0_color": 256},
    {"id": 12, "name": "v12_pad32_b1bank42", "a_pad": 32,
     "b1_color": 336, "w2l_color": 128, "m1d0_color": 256},
    {"id": 13, "name": "v13_pad32_b1bank58", "a_pad": 32,
     "b1_color": 464, "w2l_color": 128, "m1d0_color": 256},
    {"id": 14, "name": "v14_pad32_b1bank62", "a_pad": 32,
     "b1_color": 496, "w2l_color": 128, "m1d0_color": 256},
    # Batch 4: W2 left/right bank-color sweep for Mode1.
    {"id": 15, "name": "v15_pad32_w2lbank08", "a_pad": 32,
     "b1_color": 272, "w2l_color": 64, "w2r_color": 64, "m1d0_color": 256},
    {"id": 16, "name": "v16_pad32_w2lbank16", "a_pad": 32,
     "b1_color": 272, "w2l_color": 128, "w2r_color": 128, "m1d0_color": 256},
    {"id": 17, "name": "v17_pad32_w2lbank24", "a_pad": 32,
     "b1_color": 272, "w2l_color": 192, "w2r_color": 192, "m1d0_color": 256},
    {"id": 18, "name": "v18_pad32_w2lbank40", "a_pad": 32,
     "b1_color": 272, "w2l_color": 320, "w2r_color": 320, "m1d0_color": 256},
    {"id": 19, "name": "v19_pad32_w2lbank56", "a_pad": 32,
     "b1_color": 272, "w2l_color": 448, "w2r_color": 448, "m1d0_color": 256},
    # Batch 5: Mode0/Mode1 output base bank sweep.
    {"id": 20, "name": "v20_pad32_dbank08_m1bank32", "a_pad": 32,
     "b1_color": 272, "w2l_color": 128, "d0_color": 64, "m1d0_color": 256},
    {"id": 21, "name": "v21_pad32_dbank16_m1bank40", "a_pad": 32,
     "b1_color": 272, "w2l_color": 128, "d0_color": 128, "m1d0_color": 320},
    {"id": 22, "name": "v22_pad32_dbank24_m1bank48", "a_pad": 32,
     "b1_color": 272, "w2l_color": 128, "d0_color": 192, "m1d0_color": 384},
    {"id": 23, "name": "v23_pad32_dbank40_m1bank08", "a_pad": 32,
     "b1_color": 272, "w2l_color": 128, "d0_color": 320, "m1d0_color": 64},
    {"id": 24, "name": "v24_pad32_dbank56_m1bank24", "a_pad": 32,
     "b1_color": 272, "w2l_color": 128, "d0_color": 448, "m1d0_color": 192},
    # Batch 6: Larger A stride phases for conflict-class mapping.
    {"id": 25, "name": "v25_pad72_b1bank34", "a_pad": 72,
     "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
    {"id": 26, "name": "v26_pad80_b1bank34", "a_pad": 80,
     "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
    {"id": 27, "name": "v27_pad88_b1bank34", "a_pad": 88,
     "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
    {"id": 28, "name": "v28_pad96_b1bank34", "a_pad": 96,
     "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
    {"id": 29, "name": "v29_pad128_b1bank34", "a_pad": 128,
     "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
    # Batch 7: Combined A stride and B1 high-color points.
    {"id": 30, "name": "v30_pad08_b1bank52", "a_pad": 8,
     "b1_color": 416, "w2l_color": 128, "m1d0_color": 256},
    {"id": 31, "name": "v31_pad24_b1bank52", "a_pad": 24,
     "b1_color": 416, "w2l_color": 128, "m1d0_color": 256},
    {"id": 32, "name": "v32_pad40_b1bank52", "a_pad": 40,
     "b1_color": 416, "w2l_color": 128, "m1d0_color": 256},
    {"id": 33, "name": "v33_pad48_b1bank52", "a_pad": 48,
     "b1_color": 416, "w2l_color": 128, "m1d0_color": 256},
    {"id": 34, "name": "v34_pad64_b1bank52", "a_pad": 64,
     "b1_color": 416, "w2l_color": 128, "m1d0_color": 256},
    # Batch 8: B0/B1 separation sweep.
    {"id": 35, "name": "v35_pad32_b0bank08_b1bank34", "a_pad": 32,
     "b0_color": 64, "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
    {"id": 36, "name": "v36_pad32_b0bank16_b1bank52", "a_pad": 32,
     "b0_color": 128, "b1_color": 416, "w2l_color": 128, "m1d0_color": 256},
    {"id": 37, "name": "v37_pad32_b0bank24_b1bank42", "a_pad": 32,
     "b0_color": 192, "b1_color": 336, "w2l_color": 128, "m1d0_color": 256},
    {"id": 38, "name": "v38_pad32_b0bank32_b1bank58", "a_pad": 32,
     "b0_color": 256, "b1_color": 464, "w2l_color": 128, "m1d0_color": 256},
    {"id": 39, "name": "v39_pad32_b0bank48_b1bank18", "a_pad": 32,
     "b0_color": 384, "b1_color": 144, "w2l_color": 128, "m1d0_color": 256},
    # Batch 9: Mixed low-footprint and Mode1-color candidates.
    {"id": 40, "name": "v40_pad00_w2lbank40_m1bank08", "a_pad": 0,
     "b1_color": 272, "w2l_color": 320, "w2r_color": 320, "m1d0_color": 64},
    {"id": 41, "name": "v41_pad16_w2lbank40_m1bank08", "a_pad": 16,
     "b1_color": 272, "w2l_color": 320, "w2r_color": 320, "m1d0_color": 64},
    {"id": 42, "name": "v42_pad24_w2lbank56_m1bank24", "a_pad": 24,
     "b1_color": 416, "w2l_color": 448, "w2r_color": 448, "m1d0_color": 192},
    {"id": 43, "name": "v43_pad32_w2lbank56_m1bank24", "a_pad": 32,
     "b1_color": 416, "w2l_color": 448, "w2r_color": 448, "m1d0_color": 192},
    {"id": 44, "name": "v44_pad40_w2lbank08_m1bank40", "a_pad": 40,
     "b1_color": 272, "w2l_color": 64, "w2r_color": 64, "m1d0_color": 320},
    # Batch 10: Final confirmation around the best-looking cycle points.
    {"id": 45, "name": "v45_pad32_b1bank34_confirm", "a_pad": 32,
     "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
    {"id": 46, "name": "v46_pad32_b1bank52_confirm", "a_pad": 32,
     "b1_color": 416, "w2l_color": 128, "m1d0_color": 256},
    {"id": 47, "name": "v47_pad24_b1bank52_confirm", "a_pad": 24,
     "b1_color": 416, "w2l_color": 128, "m1d0_color": 256},
    {"id": 48, "name": "v48_pad40_b1bank34_confirm", "a_pad": 40,
     "b1_color": 272, "w2l_color": 128, "m1d0_color": 256},
    {"id": 49, "name": "v49_pad56_b1bank52_confirm", "a_pad": 56,
     "b1_color": 416, "w2l_color": 128, "m1d0_color": 256},
    # Shape-specific campaign batch 1: independent S0/S1/S2 A staging strides.
    {"id": 50, "name": "v50_shapeA_s0p32_s1p00_s2p56", "a_pad": 32,
     "shape_a_pads": [32, 0, 56], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256},
    {"id": 51, "name": "v51_shapeA_s0p32_s1p00_s2p80", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256},
    {"id": 52, "name": "v52_shapeA_s0p32_s1p16_s2p56", "a_pad": 32,
     "shape_a_pads": [32, 16, 56], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256},
    {"id": 53, "name": "v53_shapeA_s0p40_s1p00_s2p80", "a_pad": 40,
     "shape_a_pads": [40, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256},
    {"id": 54, "name": "v54_shapeA_s0p32_s1p24_s2p96", "a_pad": 32,
     "shape_a_pads": [32, 24, 96], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256},
    # Shape-specific campaign batch 2: local refinement around V51.
    {"id": 55, "name": "v55_shapeA_s0p32_s1p04_s2p80", "a_pad": 32,
     "shape_a_pads": [32, 4, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256},
    {"id": 56, "name": "v56_shapeA_s0p32_s1p08_s2p80", "a_pad": 32,
     "shape_a_pads": [32, 8, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256},
    {"id": 57, "name": "v57_shapeA_s0p32_s1p12_s2p80", "a_pad": 32,
     "shape_a_pads": [32, 12, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256},
    {"id": 58, "name": "v58_shapeA_s0p32_s1p00_s2p72", "a_pad": 32,
     "shape_a_pads": [32, 0, 72], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256},
    {"id": 59, "name": "v59_shapeA_s0p32_s1p00_s2p88", "a_pad": 32,
     "shape_a_pads": [32, 0, 88], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256},
    # Shape-specific campaign batch 3: S2 aligned padding sweep.
    {"id": 60, "name": "v60_shapeA_s0p32_s1p00_s2p64", "a_pad": 32,
     "shape_a_pads": [32, 0, 64], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256},
    {"id": 61, "name": "v61_shapeA_s0p32_s1p00_s2p96", "a_pad": 32,
     "shape_a_pads": [32, 0, 96], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256},
    {"id": 62, "name": "v62_shapeA_s0p32_s1p00_s2p104", "a_pad": 32,
     "shape_a_pads": [32, 0, 104], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256},
    {"id": 63, "name": "v63_shapeA_s0p32_s1p00_s2p112", "a_pad": 32,
     "shape_a_pads": [32, 0, 112], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256},
    {"id": 64, "name": "v64_shapeA_s0p32_s1p00_s2p128", "a_pad": 32,
     "shape_a_pads": [32, 0, 128], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256},
    # Shape-specific campaign batch 4: combine best shape-A pads with output bank phases.
    {"id": 65, "name": "v65_shapeA_32_0_80_d8_m132", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "d0_color": 64, "m1d0_color": 256},
    {"id": 66, "name": "v66_shapeA_32_0_80_d16_m140", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "d0_color": 128, "m1d0_color": 320},
    {"id": 67, "name": "v67_shapeA_32_0_80_d24_m148", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "d0_color": 192, "m1d0_color": 384},
    {"id": 68, "name": "v68_shapeA_32_0_104_d16_m140", "a_pad": 32,
     "shape_a_pads": [32, 0, 104], "b1_color": 416, "w2l_color": 128,
     "d0_color": 128, "m1d0_color": 320},
    {"id": 69, "name": "v69_shapeA_32_0_104_d24_m148", "a_pad": 32,
     "shape_a_pads": [32, 0, 104], "b1_color": 416, "w2l_color": 128,
     "d0_color": 192, "m1d0_color": 384},
    # Shape-specific campaign batch 5: B0/B1 separation on the V51 shape pads.
    {"id": 70, "name": "v70_shapeA_32_0_80_b0bank08_b1bank34", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b0_color": 64, "b1_color": 272,
     "w2l_color": 128, "m1d0_color": 256},
    {"id": 71, "name": "v71_shapeA_32_0_80_b0bank16_b1bank52", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b0_color": 128, "b1_color": 416,
     "w2l_color": 128, "m1d0_color": 256},
    {"id": 72, "name": "v72_shapeA_32_0_80_b0bank24_b1bank42", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b0_color": 192, "b1_color": 336,
     "w2l_color": 128, "m1d0_color": 256},
    {"id": 73, "name": "v73_shapeA_32_0_80_b0bank32_b1bank58", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b0_color": 256, "b1_color": 464,
     "w2l_color": 128, "m1d0_color": 256},
    {"id": 74, "name": "v74_shapeA_32_0_80_b0bank48_b1bank18", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b0_color": 384, "b1_color": 144,
     "w2l_color": 128, "m1d0_color": 256},
    # Shape-specific campaign batch 6: S0 Mode0-D repack for linear Mode1-A reads.
    {"id": 75, "name": "v75_shapeA_32_0_80_s0linearD", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True},
    {"id": 76, "name": "v76_shapeA_32_0_104_s0linearD", "a_pad": 32,
     "shape_a_pads": [32, 0, 104], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True},
    {"id": 77, "name": "v77_shapeA_16_0_80_s0linearD", "a_pad": 16,
     "shape_a_pads": [16, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True},
    {"id": 78, "name": "v78_shapeA_48_0_80_s0linearD", "a_pad": 48,
     "shape_a_pads": [48, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True},
    {"id": 79, "name": "v79_shapeA_32_0_128_s0linearD", "a_pad": 32,
     "shape_a_pads": [32, 0, 128], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True},
    # Shape-specific campaign batch 7: add S1 16-wide Mode0-D paneling.
    {"id": 80, "name": "v80_shapeA_32_0_80_s0linear_s1panel16", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s1_mode1_linear16": True},
    {"id": 81, "name": "v81_shapeA_32_0_104_s0linear_s1panel16", "a_pad": 32,
     "shape_a_pads": [32, 0, 104], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s1_mode1_linear16": True},
    {"id": 82, "name": "v82_shapeA_32_0_80_s1panel16_only", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s1_mode1_linear16": True},
    {"id": 83, "name": "v83_shapeA_32_8_80_s0linear_s1panel16", "a_pad": 32,
     "shape_a_pads": [32, 8, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s1_mode1_linear16": True},
    {"id": 84, "name": "v84_shapeA_32_0_128_s0linear_s1panel16", "a_pad": 32,
     "shape_a_pads": [32, 0, 128], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s1_mode1_linear16": True},
    # Shape-specific campaign batch 8: A-granularity-2 compatible row phases.
    {"id": 85, "name": "v85_shapeA_32_0_64_s0linear_s1panel16", "a_pad": 32,
     "shape_a_pads": [32, 0, 64], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s1_mode1_linear16": True},
    {"id": 86, "name": "v86_shapeA_32_0_96_s0linear_s1panel16", "a_pad": 32,
     "shape_a_pads": [32, 0, 96], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s1_mode1_linear16": True},
    {"id": 87, "name": "v87_shapeA_32_0_112_s0linear_s1panel16", "a_pad": 32,
     "shape_a_pads": [32, 0, 112], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s1_mode1_linear16": True},
    {"id": 88, "name": "v88_shapeA_32_16_80_s0linear_s1panel16", "a_pad": 32,
     "shape_a_pads": [32, 16, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s1_mode1_linear16": True},
    {"id": 89, "name": "v89_shapeA_64_0_80_s0linear_s1panel16", "a_pad": 64,
     "shape_a_pads": [64, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s1_mode1_linear16": True},
    # Shape-specific campaign batch 9: S2 Mode0-D panels for coarser Mode1-A.
    {"id": 90, "name": "v90_shapeA_32_0_80_s0_s1_s2lin4", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s1_mode1_linear16": True, "s2_mode1_linear4": True},
    {"id": 91, "name": "v91_shapeA_32_0_80_s0_s1_s2lin8", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s1_mode1_linear16": True, "s2_mode1_linear8": True},
    {"id": 92, "name": "v92_shapeA_32_0_80_s0_s1_s2lin16", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s1_mode1_linear16": True, "s2_mode1_linear16": True},
    {"id": 93, "name": "v93_shapeA_32_0_96_s0_s1_s2lin16", "a_pad": 32,
     "shape_a_pads": [32, 0, 96], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s1_mode1_linear16": True, "s2_mode1_linear16": True},
    {"id": 94, "name": "v94_shapeA_32_0_80_s0_s2lin16", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s2_mode1_linear16": True},
    # Shape-specific campaign batch 10: final Pareto confirmation points.
    {"id": 95, "name": "v95_pareto_practical_Agran2", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True},
    {"id": 96, "name": "v96_pareto_Agran2_s2pad96", "a_pad": 32,
     "shape_a_pads": [32, 0, 96], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True},
    {"id": 97, "name": "v97_pareto_s0_s1_headroom", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s1_mode1_linear16": True},
    {"id": 98, "name": "v98_pareto_s0_s2lin16", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s2_mode1_linear16": True},
    {"id": 99, "name": "v99_pareto_s0_s1_s2lin16", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "s0_mode1_linear": True,
     "s1_mode1_linear16": True, "s2_mode1_linear16": True},
    # 2026-07-10 joint layout/granularity campaign, batch 1.
    # The global/L3 A image and final Mode1 output remain per-token with the
    # same row stride.  a_panel128 only changes the staged TCDM compute image.
    {"id": 100, "name": "v100_joint_A16_B8_fullpanel_p32_0_80", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 0, "w2l_color": 128,
     "w2r_color": 0, "m1d0_color": 256, "a_panel128": True,
     "b_panel64": True, "s0_mode1_linear": True,
     "s1_mode1_linear16": True, "s2_mode1_linear16": True,
     "target_granularity": [16, 8, 8]},
    {"id": 101, "name": "v101_joint_A4_B8_s2lin4_p32_0_80", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 0, "w2l_color": 128,
     "w2r_color": 0, "m1d0_color": 256, "a_panel128": True,
     "b_panel64": True, "s0_mode1_linear": True,
     "s1_mode1_linear16": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 8, 8]},
    {"id": 102, "name": "v102_joint_A8_B8_s2lin8_p32_0_80", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 0, "w2l_color": 128,
     "w2r_color": 0, "m1d0_color": 256, "a_panel128": True,
     "b_panel64": True, "s0_mode1_linear": True,
     "s1_mode1_linear16": True, "s2_mode1_linear8": True,
     "target_granularity": [8, 8, 8]},
    {"id": 103, "name": "v103_joint_A2_B8_isolate_Bpanel", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 0, "w2l_color": 128,
     "w2r_color": 0, "m1d0_color": 256, "b_panel64": True,
     "s0_mode1_linear": True, "target_granularity": [2, 8, 8]},
    {"id": 104, "name": "v104_joint_A16_B2_isolate_Apanel", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "a_panel128": True,
     "s0_mode1_linear": True, "s1_mode1_linear16": True,
     "s2_mode1_linear16": True, "target_granularity": [16, 2, 2]},
    # Joint campaign batch 2: shape-specific input-A panel pitches.
    {"id": 105, "name": "v105_joint_A4_B2_shapePanel_128_64_32", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "a_panel_granularity": 4,
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 106, "name": "v106_joint_A4_B2_s2pad64", "a_pad": 32,
     "shape_a_pads": [32, 0, 64], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "a_panel_granularity": 4,
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 107, "name": "v107_joint_A4_B2_s2pad96", "a_pad": 32,
     "shape_a_pads": [32, 0, 96], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "a_panel_granularity": 4,
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 108, "name": "v108_joint_A4_B2_m1bank40", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 320, "a_panel_granularity": 4,
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 109, "name": "v109_joint_A8_B2_shapePanel_128_64_64", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "a_panel_granularity": 8,
     "s0_mode1_linear": True, "s2_mode1_linear8": True,
     "target_granularity": [8, 2, 2]},
    # Joint campaign batch 3: A4-legal input-panel bank phases.
    {"id": 110, "name": "v110_joint_A4_Apanel_bank16", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 111, "name": "v111_joint_A4_Apanel_bank32", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 256, "m1d0_color": 256, "a_panel_granularity": 4,
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 112, "name": "v112_joint_A4_Apanel_bank48", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 384, "m1d0_color": 256, "a_panel_granularity": 4,
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 113, "name": "v113_joint_A4_Apanel_bank16_B1bank34", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 272, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 114, "name": "v114_joint_A4_Apanel_bank32_B1bank34", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 272, "w2l_color": 128,
     "a_color": 256, "m1d0_color": 256, "a_panel_granularity": 4,
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    # Joint campaign batch 4: A4-legal Mode0-D / Mode1-A base phases.
    {"id": 115, "name": "v115_joint_A4_D0bank04", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "d0_color": 32, "m1d0_color": 256,
     "a_panel_granularity": 4, "s0_mode1_linear": True,
     "s2_mode1_linear4": True, "target_granularity": [4, 2, 2]},
    {"id": 116, "name": "v116_joint_A4_D0bank08", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "d0_color": 64, "m1d0_color": 256,
     "a_panel_granularity": 4, "s0_mode1_linear": True,
     "s2_mode1_linear4": True, "target_granularity": [4, 2, 2]},
    {"id": 117, "name": "v117_joint_A4_D0bank12", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "d0_color": 96, "m1d0_color": 256,
     "a_panel_granularity": 4, "s0_mode1_linear": True,
     "s2_mode1_linear4": True, "target_granularity": [4, 2, 2]},
    {"id": 118, "name": "v118_joint_A4_D0bank16", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "d0_color": 128, "m1d0_color": 256,
     "a_panel_granularity": 4, "s0_mode1_linear": True,
     "s2_mode1_linear4": True, "target_granularity": [4, 2, 2]},
    {"id": 119, "name": "v119_joint_A4_D0bank32", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "d0_color": 256, "m1d0_color": 256,
     "a_panel_granularity": 4, "s0_mode1_linear": True,
     "s2_mode1_linear4": True, "target_granularity": [4, 2, 2]},
    # Joint campaign batch 5: shape-specific B4 panels.
    {"id": 120, "name": "v120_joint_A2_B4_B1phase52", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "b_shape_panel_granularity": 4,
     "s0_mode1_linear": True, "target_granularity": [2, 4, 4]},
    {"id": 121, "name": "v121_joint_A2_B4_B1phase48", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 384, "w2l_color": 128,
     "m1d0_color": 256, "b_shape_panel_granularity": 4,
     "s0_mode1_linear": True, "target_granularity": [2, 4, 4]},
    {"id": 122, "name": "v122_joint_A2_B4_B1phase32", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 256, "w2l_color": 128,
     "m1d0_color": 256, "b_shape_panel_granularity": 4,
     "s0_mode1_linear": True, "target_granularity": [2, 4, 4]},
    {"id": 123, "name": "v123_joint_A2_B4_B1phase16", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 128, "w2l_color": 128,
     "m1d0_color": 256, "b_shape_panel_granularity": 4,
     "s0_mode1_linear": True, "target_granularity": [2, 4, 4]},
    {"id": 124, "name": "v124_joint_A4_B4_combined", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256,
     "a_panel_granularity": 4, "b_shape_panel_granularity": 4,
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 4, 4]},
    # Joint campaign batch 6: legal non-contiguous S0 A4 spatial panels.
    {"id": 125, "name": "v125_joint_A4_S0tokenStride48", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [48, 16, 16],
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 126, "name": "v126_joint_A4_S0tokenStride80", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [80, 16, 16],
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 127, "name": "v127_joint_A4_S0tokenStride112", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [112, 16, 16],
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 128, "name": "v128_joint_A4_S0tokenStride144", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [144, 16, 16],
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 129, "name": "v129_joint_A4_S0tokenStride176", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [176, 16, 16],
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    # Joint campaign batch 7: legal non-contiguous S1 A4 spatial panels.
    {"id": 130, "name": "v130_joint_A4_S1tokenStride48", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 48, 16],
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 131, "name": "v131_joint_A4_S1tokenStride80", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 80, 16],
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 132, "name": "v132_joint_A4_S1tokenStride112", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 112, 16],
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 133, "name": "v133_joint_A4_S1tokenStride144", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 144, 16],
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 134, "name": "v134_joint_A4_S1tokenStride176", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 176, 16],
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    # Joint campaign batch 8: legal non-contiguous S2 A4 spatial panels.
    {"id": 135, "name": "v135_joint_A4_S2tokenStride48", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 16, 48],
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 136, "name": "v136_joint_A4_S2tokenStride80", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 16, 80],
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 137, "name": "v137_joint_A4_S2tokenStride112", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 16, 112],
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 138, "name": "v138_joint_A4_S2tokenStride144", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 16, 144],
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    {"id": 139, "name": "v139_joint_A4_S2tokenStride176", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 16, 176],
     "s0_mode1_linear": True, "s2_mode1_linear4": True,
     "target_granularity": [4, 2, 2]},
    # Joint campaign batch 9: sparse S2 Mode0-D / Mode1-A panels.
    {"id": 140, "name": "v140_joint_A4_S2in176_M1stride48", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 16, 176],
     "s0_mode1_linear": True, "s2_mode1_token_stride": 48,
     "target_granularity": [4, 2, 2]},
    {"id": 141, "name": "v141_joint_A4_S2in176_M1stride80", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 16, 176],
     "s0_mode1_linear": True, "s2_mode1_token_stride": 80,
     "target_granularity": [4, 2, 2]},
    {"id": 142, "name": "v142_joint_A4_S2in176_M1stride112", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 16, 176],
     "s0_mode1_linear": True, "s2_mode1_token_stride": 112,
     "target_granularity": [4, 2, 2]},
    {"id": 143, "name": "v143_joint_A4_S2in176_M1stride144", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 16, 176],
     "s0_mode1_linear": True, "s2_mode1_token_stride": 144,
     "target_granularity": [4, 2, 2]},
    {"id": 144, "name": "v144_joint_A4_S2in176_M1stride176", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 16, 176],
     "s0_mode1_linear": True, "s2_mode1_token_stride": 176,
     "target_granularity": [4, 2, 2]},
    # Joint campaign batch 10: extend the monotonic S2 intermediate sweep.
    {"id": 145, "name": "v145_joint_A4_S2in176_M1stride208", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 16, 176],
     "s0_mode1_linear": True, "s2_mode1_token_stride": 208,
     "target_granularity": [4, 2, 2]},
    {"id": 146, "name": "v146_joint_A4_S2in176_M1stride240", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 16, 176],
     "s0_mode1_linear": True, "s2_mode1_token_stride": 240,
     "target_granularity": [4, 2, 2]},
    {"id": 147, "name": "v147_joint_A4_S2in176_M1stride272", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 16, 176],
     "s0_mode1_linear": True, "s2_mode1_token_stride": 272,
     "target_granularity": [4, 2, 2]},
    {"id": 148, "name": "v148_joint_A4_S2in176_M1stride304", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 16, 176],
     "s0_mode1_linear": True, "s2_mode1_token_stride": 304,
     "target_granularity": [4, 2, 2]},
    {"id": 149, "name": "v149_joint_A4_S2in176_M1stride336", "a_pad": 32,
     "shape_a_pads": [32, 0, 80], "b1_color": 416, "w2l_color": 128,
     "a_color": 128, "m1d0_color": 256, "a_panel_granularity": 4,
     "a_panel_token_strides": [16, 16, 176],
     "s0_mode1_linear": True, "s2_mode1_token_stride": 336,
     "target_granularity": [4, 2, 2]},
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


def make_logical_a(m_total, k_total):
    data = np.zeros((m_total, k_total), dtype=np.int16)
    for m in range(m_total):
        for k in range(k_total):
            data[m, k] = ((m * 5 + k * 3) % 11) - 5
    return data


def make_padded_a(logical_a, row_stride_bytes):
    row_elems = row_stride_bytes // 2
    out = np.zeros((logical_a.shape[0], row_elems), dtype=np.int16)
    out[:, :logical_a.shape[1]] = logical_a
    return out.reshape(-1)


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


def writer_i16_image(source_i16, tbound, tstride, total_elems):
    flat = source_i16.reshape(-1)
    out = np.zeros(total_elems, dtype=np.int16)
    src = 0
    for i3 in range(tbound[3]):
        for i2 in range(tbound[2]):
            for i1 in range(tbound[1]):
                for i0 in range(tbound[0]):
                    byte_addr = (i0 * tstride[0] + i1 * tstride[1] +
                                 i2 * tstride[2] + i3 * tstride[3])
                    assert byte_addr % 2 == 0
                    elem = byte_addr // 2
                    assert elem + 4 <= total_elems, (elem, total_elems, tbound, tstride)
                    assert src + 4 <= len(flat), (src, len(flat), tbound)
                    out[elem:elem + 4] = flat[src:src + 4]
                    src += 4
    assert src == len(flat), (src, len(flat), tbound)
    return out


def packed_int4_constant(num_shape0_tiles, value):
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


def colored_offset(offset, color_bytes=0, alignment=1024):
    return align_up(offset, alignment) + int(color_bytes)


def mode0_memory_plan(array_shape, layout, default_elems, k1_tiles=16):
    if array_shape == 0 and layout.get("s0_mode1_linear", False):
        return {
            "mode1_a_sstride": [8, 16],
            "mode1_a_k_stride": 128,
            "mode0_d_tbound": [8, 2, k1_tiles, 1],
            "mode0_d_tstride": [16, 8, 128, 0],
            "mode0_check_elems": default_elems,
        }
    if array_shape == 1 and layout.get("s1_mode1_linear16", False):
        return {
            "mode1_a_sstride": [8, 16],
            "mode1_a_k_stride": 128,
            "mode0_d_tbound": [8, 16, 1, 1],
            "mode0_d_tstride": [8, 128, 0, 0],
            "mode0_check_elems": 992,
        }
    if array_shape == 2 and layout.get("s2_mode1_token_stride", 0):
        token_stride = int(layout["s2_mode1_token_stride"])
        pitch = align_up(token_stride + 16, 32)
        return {
            "mode1_a_sstride": [8, token_stride],
            "mode1_a_k_stride": pitch,
            "mode0_d_tbound": [2, 2, k1_tiles, 1],
            "mode0_d_tstride": [8, token_stride, pitch, 0],
            "mode0_check_elems": (
                (k1_tiles - 1) * pitch + token_stride + 16) // 2,
        }
    if array_shape == 2 and layout.get("s2_mode1_linear16", False):
        return {
            "mode1_a_sstride": [8, 16],
            "mode1_a_k_stride": 128,
            "mode0_d_tbound": [4, 16, 1, 1],
            "mode0_d_tstride": [8, 128, 0, 0],
            "mode0_check_elems": 976,
        }
    if array_shape == 2 and layout.get("s2_mode1_linear8", False):
        return {
            "mode1_a_sstride": [8, 16],
            "mode1_a_k_stride": 64,
            "mode0_d_tbound": [4, 16, 1, 1],
            "mode0_d_tstride": [8, 64, 0, 0],
            "mode0_check_elems": 496,
        }
    if array_shape == 2 and layout.get("s2_mode1_linear4", False):
        return {
            "mode1_a_sstride": [8, 16],
            "mode1_a_k_stride": 32,
            "mode0_d_tbound": [4, 16, 1, 1],
            "mode0_d_tstride": [8, 32, 0, 0],
            "mode0_check_elems": default_elems,
        }
    return None


def emit_i16_array(name, values):
    return f"static const int16_t {name}[{len(values)}] = {c_array(values)};"


def emit_u8_array(name, values):
    return f"static const uint8_t {name}[{len(values)}] = {c_u8_array(values)};"


def input_a_panel_pitch(array_shape, layout):
    granularity = layout.get(
        "a_panel_granularity", 16 if layout.get("a_panel128", False) else 1)
    if granularity <= 1:
        return 0
    tokens = {0: 8, 1: 4, 2: 2}[array_shape]
    token_stride = layout.get("a_panel_token_strides", [16, 16, 16])[array_shape]
    extent = (tokens - 1) * token_stride + 16
    pitch_overrides = layout.get("a_panel_pitches", [0, 0, 0])
    override = int(pitch_overrides[array_shape])
    if override:
        assert override >= extent
        assert override % (granularity * 8) == 0
        return override
    return align_up(extent, granularity * 8)


def input_a_panel_token_stride(array_shape, layout):
    return layout.get("a_panel_token_strides", [16, 16, 16])[array_shape]


def input_b_panel_pitch(array_shape, layout):
    granularity = int(layout.get("b_shape_panel_granularity", 0))
    if granularity <= 1:
        return 0
    active_words = {0: 2, 1: 4, 2: 8}[array_shape]
    return max(active_words, granularity) * 8


def place_tensors(globals_, layout):
    shape_a_pads = layout.get("shape_a_pads", [layout["a_pad"]] * len(SHAPE_DIMS))
    shape_a_strides = [globals_["k0_bytes"] + pad for pad in shape_a_pads]
    panel_pitches = [input_a_panel_pitch(s[1], layout) for s in SHAPE_DIMS]
    if any(panel_pitches):
        shape_a_local_bytes = [globals_["k0_s0_tiles"] * pitch
                               for pitch in panel_pitches]
    else:
        shape_a_local_bytes = [globals_["m_total"] * stride
                               for stride in shape_a_strides]
    mode0_d_bytes = globals_["m_total"] * globals_["n0_total"] * 2
    if layout.get("s2_mode1_token_stride", 0):
        s2_pitch = align_up(int(layout["s2_mode1_token_stride"]) + 16, 32)
        mode0_d_bytes = max(
            mode0_d_bytes,
            (globals_["k1_s0_tiles"] - 1) * s2_pitch
            + int(layout["s2_mode1_token_stride"]) + 16)
    mode1_padded_d_bytes = globals_["m_total"] * max(shape_a_strides)

    # MoE deployment keeps expert weights in a deterministic prefix of TCDM and
    # places the variable token buffer after all expert weights.
    if layout.get("b_shape_panel_granularity", 0):
        delta_local_b0 = []
        delta_local_b1 = []
        delta_local_w2l = []
        delta_local_w2r = []
        weight_cursor = 0
        for _, array_shape, _, _, mesh_col in SHAPE_DIMS:
            pitch = input_b_panel_pitch(array_shape, layout)
            w_bytes = (globals_["k0_s0_tiles"] *
                       (globals_["n0_total"] // mesh_col) * pitch)
            w2_bytes = (globals_["k1_s0_tiles"] *
                        (globals_["n1_total"] // mesh_col) * pitch)
            b0 = colored_offset(weight_cursor, layout.get("b0_color", 0))
            b1 = colored_offset(b0 + w_bytes, layout.get("b1_color", 0))
            w2l = colored_offset(b1 + w_bytes, layout.get("w2l_color", 0))
            w2r = colored_offset(w2l + w2_bytes, layout.get("w2r_color", 0))
            delta_local_b0.append(b0)
            delta_local_b1.append(b1)
            delta_local_w2l.append(w2l)
            delta_local_w2r.append(w2r)
            weight_cursor = w2r + w2_bytes
        delta_local_a_base = colored_offset(weight_cursor,
                                            layout.get("a_color", 0))
    else:
        b_panel_pitch = 64 if layout.get("b_panel64", False) else 16
        w_bytes = globals_["k0_s0_tiles"] * globals_["n0_s0_tiles"] * b_panel_pitch
        w2_bytes = globals_["k1_s0_tiles"] * globals_["n1_s0_tiles"] * b_panel_pitch
        b0 = colored_offset(0, layout.get("b0_color", 0))
        b1 = colored_offset(b0 + w_bytes, layout.get("b1_color", 0))
        w2l = colored_offset(b1 + w_bytes, layout.get("w2l_color", 0))
        w2r = colored_offset(w2l + w2_bytes, layout.get("w2r_color", 0))
        delta_local_b0 = [b0] * len(SHAPE_DIMS)
        delta_local_b1 = [b1] * len(SHAPE_DIMS)
        delta_local_w2l = [w2l] * len(SHAPE_DIMS)
        delta_local_w2r = [w2r] * len(SHAPE_DIMS)
        delta_local_a_base = colored_offset(w2r + w2_bytes,
                                            layout.get("a_color", 0))
    delta_local_a_by_shape = []
    cursor = delta_local_a_base
    shape_a_banks = layout.get("shape_a_base_banks", [None] * len(SHAPE_DIMS))
    for array_shape, local_bytes in enumerate(shape_a_local_bytes):
        alignment = panel_pitches[array_shape] if panel_pitches[array_shape] else 16
        desired_bank = shape_a_banks[array_shape]
        if desired_bank is None:
            cursor = align_up(cursor, alignment)
        else:
            # A4 requires a base-bank residue of zero modulo four.  Keep the
            # allocation minimally 32-byte aligned, then select the requested
            # absolute TCDM bank phase without changing panel strides.
            desired_bank = int(desired_bank)
            assert 0 <= desired_bank < 64 and desired_bank % 4 == 0
            cursor = align_up(cursor, 32)
            while (cursor // 8) % 64 != desired_bank:
                cursor += 32
        delta_local_a_by_shape.append(cursor)
        cursor += local_bytes
    delta_local_d0 = colored_offset(cursor, layout.get("d0_color", 0))
    delta_local_mode1_d0 = colored_offset(delta_local_d0 + mode0_d_bytes, layout.get("m1d0_color", 0))
    delta_local_mode1_d1 = delta_local_mode1_d0 + globals_["n1_total"] * 2

    return {
        "delta_local_a": delta_local_a_base,
        "delta_local_a_by_shape": delta_local_a_by_shape,
        "delta_local_b0": delta_local_b0,
        "delta_local_b1": delta_local_b1,
        "delta_local_d0": delta_local_d0,
        "delta_local_w2l": delta_local_w2l,
        "delta_local_w2r": delta_local_w2r,
        "delta_local_mode1_d0": delta_local_mode1_d0,
        "delta_local_mode1_d1": delta_local_mode1_d1,
        "tcdm_end": delta_local_mode1_d0 + mode1_padded_d_bytes,
    }


def build_shape_cfg(shape, globals_, golden_names, layout, placement):
    name, array_shape, mesh_row, tile_size, mesh_col = shape
    k0_total = globals_["k0_total"]
    n0_total = globals_["n0_total"]
    k1_total = globals_["k1_total"]
    n1_total = globals_["n1_total"]
    k0_s0_tiles = globals_["k0_s0_tiles"]
    k1_s0_tiles = globals_["k1_s0_tiles"]
    shape_a_pads = layout.get("shape_a_pads", [layout["a_pad"]] * len(SHAPE_DIMS))
    a_row_stride = globals_["k0_bytes"] + shape_a_pads[array_shape]
    a_panel_pitch = input_a_panel_pitch(array_shape, layout)
    a_panel_token_stride = input_a_panel_token_stride(array_shape, layout)

    m_tiles = 1
    k_tiles = k0_total // tile_size
    k1_tiles = k1_total // tile_size
    n0_tiles = n0_total // mesh_col
    n1_tiles = n1_total // mesh_col
    shape_b_panel_pitch = input_b_panel_pitch(array_shape, layout)
    if shape_b_panel_pitch:
        b_k_stride = shape_b_panel_pitch
        mode0_b_n_stride = k0_s0_tiles * shape_b_panel_pitch
        mode1_b_n_stride = k1_s0_tiles * shape_b_panel_pitch
        mode0_b_spatial = [8, 16]
        mode1_b_spatial = [8, 16]
    elif layout.get("b_panel64", False):
        b_k_stride = 64
        mode0_b_n_stride = k0_s0_tiles * 64
        mode1_b_n_stride = k1_s0_tiles * 64
        mode0_b_spatial = [8, 16]
        mode1_b_spatial = [8, 16]
    else:
        b_k_stride = 16
        mode0_b_n_stride = (mesh_col // 4) * k0_s0_tiles * 16
        mode1_b_n_stride = (mesh_col // 4) * k1_s0_tiles * 16
        mode0_b_spatial = [8, k0_s0_tiles * 16]
        mode1_b_spatial = [8, k1_s0_tiles * 16]
    a_channel_en = {0: 0xFFFF, 1: 0x00FF, 2: 0x000F}[array_shape]
    b_channel_en = {0: 0x03, 1: 0x0F, 2: 0xFF}[array_shape]
    d_stride1 = 64
    mode0_d_m_stride = n0_tiles * d_stride1
    d_bound0 = 8
    beats_per_row = mesh_col // 4
    mode1_token_stride = a_row_stride
    mode1_a_sstride = {0: [64, 8], 1: [8, 16], 2: [8, 32]}[array_shape]
    mode1_a_k_stride = {0: 128, 1: 64, 2: 16}[array_shape]
    mode0_d_tbound = [d_bound0, n0_tiles, m_tiles, 1]
    mode0_d_tstride = [8, d_stride1, mode0_d_m_stride, 0]
    mode0_check_elems = mesh_row * n0_total
    plan = mode0_memory_plan(
        array_shape, layout, mode0_check_elems, k1_tiles)
    if plan is not None:
        mode1_a_sstride = plan["mode1_a_sstride"]
        mode1_a_k_stride = plan["mode1_a_k_stride"]
        mode0_d_tbound = plan["mode0_d_tbound"]
        mode0_d_tstride = plan["mode0_d_tstride"]
        mode0_check_elems = plan["mode0_check_elems"]

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
        f".mode0_A_sstride = {c_array([8, a_panel_token_stride] if a_panel_pitch else [8, a_row_stride])}",
        f".mode1_A_sstride = {c_array(mode1_a_sstride)}",
        f".mode0_B_sstride = {c_array(mode0_b_spatial)}",
        f".mode1_B_sstride = {c_array(mode1_b_spatial)}",
        f".D_sstride = {c_array([8])}",
        f".mode0_A_tbound = {c_array([k_tiles, n0_tiles, m_tiles, 1, 1, 1])}",
        f".mode0_A_tstride = {c_array([a_panel_pitch if a_panel_pitch else tile_size * 2, 0, mesh_row * a_row_stride, 0, 0, 0])}",
        f".mode1_A_tbound = {c_array([k1_tiles, n1_tiles, m_tiles, 1, 1, 1])}",
        f".mode1_A_tstride = {c_array([mode1_a_k_stride, 0, mode0_d_m_stride, 0, 0, 0])}",
        f".mode0_B_tbound = {c_array([k_tiles, n0_tiles, m_tiles, 1])}",
        f".mode0_B_tstride = {c_array([b_k_stride, mode0_b_n_stride, 0, 0])}",
        f".mode1_B_tbound = {c_array([k1_tiles, n1_tiles, m_tiles, 1])}",
        f".mode1_B_tstride = {c_array([b_k_stride, mode1_b_n_stride, 0, 0])}",
        f".mode0_D_tbound = {c_array(mode0_d_tbound)}",
        f".mode0_D_tstride = {c_array(mode0_d_tstride)}",
        f".mode1_D_tbound = {c_array([beats_per_row, mesh_row, n1_tiles, m_tiles])}",
        f".mode1_D_tstride = {c_array([8, mode1_token_stride, mesh_col * 2, 0])}",
        f".A_channel_en = {c_array([a_channel_en])}",
        f".B_channel_en = {c_array([b_channel_en])}",
        f".D_channel_en = {c_array([0x01])}",
        f".delta_local_a = {placement['delta_local_a_by_shape'][array_shape]}",
        f".delta_local_b0 = {placement['delta_local_b0'][array_shape]}",
        f".delta_local_b1 = {placement['delta_local_b1'][array_shape]}",
        f".delta_local_d0 = {placement['delta_local_d0']}",
        f".delta_local_w2l = {placement['delta_local_w2l'][array_shape]}",
        f".delta_local_w2r = {placement['delta_local_w2r'][array_shape]}",
        f".delta_local_mode1_d0 = {placement['delta_local_mode1_d0']}",
        f".delta_local_mode1_d1 = {placement['delta_local_mode1_d1']}",
        f".tcdm_end = {placement['tcdm_end']}",
        f".mode0_output_elems = {mode0_check_elems}",
        f".mode1_output_elems = {mesh_row * n1_total}",
        f".mode1_output_row_stride_bytes = {a_row_stride}",
        f".mode1_padded_output_elems = {mesh_row * (a_row_stride // 2)}",
        f".a_data = {golden_names[name][2]}",
        f".a_data_length = {globals_['m_total'] * a_row_stride}",
        f".a_row_stride = {a_row_stride}",
        f".a_panel_pitch = {a_panel_pitch}",
        f".a_panel_token_stride = {a_panel_token_stride if a_panel_pitch else 0}",
        f".w_data = W_g{layout.get('b_shape_panel_granularity')}_S{array_shape}" if shape_b_panel_pitch else f".w_data = {'W_panel64' if layout.get('b_panel64', False) else 'W'}",
        f".v_data = V_g{layout.get('b_shape_panel_granularity')}_S{array_shape}" if shape_b_panel_pitch else f".v_data = {'V_panel64' if layout.get('b_panel64', False) else 'V'}",
        f".w2_left_data = W2_left_g{layout.get('b_shape_panel_granularity')}_S{array_shape}" if shape_b_panel_pitch else f".w2_left_data = {'W2_left_panel64' if layout.get('b_panel64', False) else 'W2_left'}",
        f".w2_right_data = W2_right_g{layout.get('b_shape_panel_granularity')}_S{array_shape}" if shape_b_panel_pitch else f".w2_right_data = {'W2_right_panel64' if layout.get('b_panel64', False) else 'W2_right'}",
        f".b_data_length = {k_tiles * n0_tiles * shape_b_panel_pitch if shape_b_panel_pitch else (globals_['k0_s0_tiles'] * globals_['n0_s0_tiles'] * (64 if layout.get('b_panel64', False) else 16))}",
        f".w2_data_length = {k1_tiles * n1_tiles * shape_b_panel_pitch if shape_b_panel_pitch else (globals_['k1_s0_tiles'] * globals_['n1_s0_tiles'] * (64 if layout.get('b_panel64', False) else 16))}",
        f".mode0_d0_golden = {golden_names[name][0]}",
        f".mode1_padded_golden = {golden_names[name][1]}",
    ]
    return "        {\n            " + ",\n            ".join(fields) + "\n        }"


def build_golden_arrays(logical_a, globals_, layout):
    k0_total = globals_["k0_total"]
    n0_total = globals_["n0_total"]
    k1_total = globals_["k1_total"]
    n1_total = globals_["n1_total"]
    k0_bytes = globals_["k0_bytes"]
    layout_id = layout["id"]
    a_row_stride = k0_bytes + layout["a_pad"]
    arrays = []
    names_by_shape = {}
    for shape in SHAPE_DIMS:
        name, array_shape, mesh_row, tile_size, mesh_col = shape
        shape_a_pads = layout.get("shape_a_pads", [layout["a_pad"]] * len(SHAPE_DIMS))
        shape_a_row_stride = k0_bytes + shape_a_pads[array_shape]
        m_shape_tiles = 1
        k_shape_tiles = k0_total // tile_size
        k1_tiles = k1_total // tile_size
        n_shape_tiles = n0_total // mesh_col
        a_channel_en = {0: 0xFFFF, 1: 0x00FF, 2: 0x000F}[array_shape]
        a_flat = streamer_i16_flat(
            logical_a, m_shape_tiles, k_shape_tiles, A_SPATIAL_BOUNDS,
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
        mode0_memory = mode0
        plan = mode0_memory_plan(
            array_shape, layout, len(mode0), k1_tiles)
        if plan is not None:
            mode0_memory = writer_i16_image(
                mode0, plan["mode0_d_tbound"], plan["mode0_d_tstride"],
                plan["mode0_check_elems"])

        n1_tiles_shape = n1_total // mesh_col
        w2_left_flat = np.full(n1_tiles_shape * k1_tiles * mesh_col * tile_size,
                               1, dtype=np.int8)
        w2_right_flat = np.full(n1_tiles_shape * k1_tiles * mesh_col * tile_size,
                                2, dtype=np.int8)
        d_stride1 = 64
        mode0_d_m_stride = n_shape_tiles * d_stride1
        mode1_a_sstride = {0: [64, 8], 1: [8, 16], 2: [8, 32]}[array_shape]
        mode1_a_k_stride = {0: 128, 1: 64, 2: 16}[array_shape]
        if plan is not None:
            mode1_a_sstride = plan["mode1_a_sstride"]
            mode1_a_k_stride = plan["mode1_a_k_stride"]
        mode1_a_flat = streamer_i16_flat(
            mode0_memory, m_shape_tiles, k1_tiles, A_SPATIAL_BOUNDS, mode1_a_sstride,
            mode1_a_k_stride, mode0_d_m_stride, a_channel_en)
        mode1_d0 = rescale_down_32to16(block_gemm_int16x4(
            m_shape_tiles, k1_tiles, n1_tiles_shape, mesh_row, tile_size, mesh_col,
            mode1_a_flat, w2_left_flat))
        mode1_d1 = rescale_down_32to16(block_gemm_int16x4(
            m_shape_tiles, k1_tiles, n1_tiles_shape, mesh_row, tile_size, mesh_col,
            mode1_a_flat, w2_right_flat))

        names = (
            f"L{layout_id}_{name}_mode0_d0_golden",
            f"L{layout_id}_{name}_mode1_padded_golden",
            f"A_row_stride_{shape_a_row_stride}",
        )
        names_by_shape[name] = names
        arrays.append(emit_i16_array(names[0], mode0_memory))
        mode1_d0_pertoken = mode1_d0.reshape(
            m_shape_tiles, n1_tiles_shape, mesh_row, mesh_col
        ).transpose(0, 2, 1, 3).reshape(-1)
        mode1_d1_pertoken = mode1_d1.reshape(
            m_shape_tiles, n1_tiles_shape, mesh_row, mesh_col
        ).transpose(0, 2, 1, 3).reshape(-1)
        mode1_combined = np.concatenate([
            mode1_d0_pertoken.reshape(mesh_row, n1_total),
            mode1_d1_pertoken.reshape(mesh_row, n1_total),
        ], axis=1).reshape(-1)
        row_elems = shape_a_row_stride // 2
        mode1_padded = np.zeros((mesh_row, row_elems), dtype=np.int16)
        mode1_padded[:, :n1_total * 2] = mode1_combined.reshape(mesh_row,
                                                                n1_total * 2)
        arrays.append(emit_i16_array(names[1], mode1_padded.reshape(-1)))
    return arrays, names_by_shape


def bank_coverage_for_shape(array_shape, row_stride):
    channel_en = {0: 0xFFFF, 1: 0x00FF, 2: 0x000F}[array_shape]
    offsets = [off for i, off in enumerate(spatial_offsets(A_SPATIAL_BOUNDS, [8, row_stride]))
               if (channel_en >> i) & 1]
    banks = sorted({(off // 8) % 64 for off in offsets})
    return len(banks), banks


def emit_header(params):
    m_total = int(params["M_total"])
    k0_total = int(params["K0_total"])
    n0_total = int(params["N0_total"])
    k1_total = int(params["K1_total"])
    n1_total = int(params["N1_total"])
    assert m_total == 8
    assert n0_total == k1_total
    assert k0_total == 2 * n1_total
    assert k0_total % 8 == 0
    assert k1_total % 8 == 0
    assert n0_total % 16 == 0
    assert n1_total % 16 == 0

    k0_s0_tiles = k0_total // 8
    k1_s0_tiles = k1_total // 8
    n0_s0_tiles = n0_total // 4
    n1_s0_tiles = n1_total // 4
    k0_bytes = k0_total * 2
    logical_a = make_logical_a(m_total, k0_total)
    globals_ = {
        "m_total": m_total,
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
    a_arrays = {}
    all_a_pads = set()
    for layout in LAYOUTS:
        all_a_pads.add(layout["a_pad"])
        all_a_pads.update(layout.get("shape_a_pads", []))
    for a_pad in sorted(all_a_pads):
        row_stride = k0_bytes + a_pad
        name = f"A_row_stride_{row_stride}"
        a_arrays[a_pad] = (name, m_total * row_stride)
        arrays.append(emit_i16_array(name, make_padded_a(logical_a, row_stride)))

    arrays.append(emit_u8_array("W", packed_int4_constant(k0_s0_tiles * n0_s0_tiles, 1)))
    arrays.append(emit_u8_array("V", packed_int4_constant(k0_s0_tiles * n0_s0_tiles, 2)))
    arrays.append(emit_u8_array("W2_left", packed_int4_constant(k1_s0_tiles * n1_s0_tiles, 1)))
    arrays.append(emit_u8_array("W2_right", packed_int4_constant(k1_s0_tiles * n1_s0_tiles, 2)))

    layout_inits = []
    layout_notes = []
    b_data_lengths = {16: k0_s0_tiles * n0_s0_tiles * 16,
                      64: k0_s0_tiles * n0_s0_tiles * 64}
    w2_data_lengths = {16: k1_s0_tiles * n1_s0_tiles * 16,
                       64: k1_s0_tiles * n1_s0_tiles * 64}
    if any(layout.get("b_panel64", False) for layout in LAYOUTS):
        arrays.append(emit_u8_array(
            "W_panel64",
            packed_int4_constant(k0_s0_tiles * n0_s0_tiles * 4, 1)))
        arrays.append(emit_u8_array(
            "V_panel64",
            packed_int4_constant(k0_s0_tiles * n0_s0_tiles * 4, 2)))
        arrays.append(emit_u8_array(
            "W2_left_panel64",
            packed_int4_constant(k1_s0_tiles * n1_s0_tiles * 4, 1)))
        arrays.append(emit_u8_array(
            "W2_right_panel64",
            packed_int4_constant(k1_s0_tiles * n1_s0_tiles * 4, 2)))
    shape_b_granularities = sorted({
        int(layout.get("b_shape_panel_granularity", 0))
        for layout in LAYOUTS
        if int(layout.get("b_shape_panel_granularity", 0)) > 1
    })
    for granularity in shape_b_granularities:
        for _, array_shape, _, _, mesh_col in SHAPE_DIMS:
            pitch = max({0: 2, 1: 4, 2: 8}[array_shape], granularity) * 8
            w_len = k0_s0_tiles * (n0_total // mesh_col) * pitch
            w2_len = k1_s0_tiles * (n1_total // mesh_col) * pitch
            arrays.append(emit_u8_array(
                f"W_g{granularity}_S{array_shape}",
                packed_int4_constant(w_len // 16, 1)))
            arrays.append(emit_u8_array(
                f"V_g{granularity}_S{array_shape}",
                packed_int4_constant(w_len // 16, 2)))
            arrays.append(emit_u8_array(
                f"W2_left_g{granularity}_S{array_shape}",
                packed_int4_constant(w2_len // 16, 1)))
            arrays.append(emit_u8_array(
                f"W2_right_g{granularity}_S{array_shape}",
                packed_int4_constant(w2_len // 16, 2)))
    for layout in LAYOUTS:
        golden_arrays, golden_names = build_golden_arrays(logical_a, globals_, layout)
        arrays.extend(golden_arrays)
        placement = place_tensors(globals_, layout)
        shape_cfgs = ",\n".join(
            build_shape_cfg(shape, globals_, golden_names, layout, placement)
            for shape in SHAPE_DIMS)
        a_name, a_len = a_arrays[layout["a_pad"]]
        panel_pitch = 64 if layout.get("b_panel64", False) else 16
        weight_suffix = "_panel64" if panel_pitch == 64 else ""
        layout_inits.append("\n".join([
            "    {",
            f"        .layout_id = {layout['id']},",
            f"        .name = \"{layout['name']}\",",
            f"        .a_row_stride = {k0_bytes + layout['a_pad']},",
            f"        .a_data = {a_name},",
            f"        .a_data_length = {a_len},",
            f"        .w_data = W{weight_suffix},",
            f"        .v_data = V{weight_suffix},",
            f"        .w2_left_data = W2_left{weight_suffix},",
            f"        .w2_right_data = W2_right{weight_suffix},",
            f"        .b_data_length = {b_data_lengths[panel_pitch]},",
            f"        .w2_data_length = {w2_data_lengths[panel_pitch]},",
            "        .shapes = {",
            shape_cfgs,
            "        }",
            "    }",
        ]))

        coverage = []
        for _, array_shape, _, _, _ in SHAPE_DIMS:
            count, banks = bank_coverage_for_shape(array_shape, k0_bytes + layout["a_pad"])
            coverage.append(f"S{array_shape}:{count} banks {banks}")
        layout_notes.append(
            f"// L{layout['id']} {layout['name']}: A row stride "
            f"{k0_bytes + layout['a_pad']} B; "
            f"A bank {placement['delta_local_a'] // 8 % 64}; "
            f"B0 bank {placement['delta_local_b0'][0] // 8 % 64}; "
            f"B1 bank {placement['delta_local_b1'][0] // 8 % 64}; "
            f"W2L bank {placement['delta_local_w2l'][0] // 8 % 64}; "
            f"D0 bank {placement['delta_local_d0'] // 8 % 64}; "
            + "; ".join(coverage))

    return "\n".join([
        "#include <stdint.h>",
        "",
        "#ifndef SNAX_VERSACORE_INT16X4_MOE_SMALL_K8_MODE1_CONTIGUOUS_L15_DATA_H",
        "#define SNAX_VERSACORE_INT16X4_MOE_SMALL_K8_MODE1_CONTIGUOUS_L15_DATA_H",
        "",
        f"#define NUM_LAYOUTS {len(LAYOUTS)}",
        "#define NUM_SHAPES 3",
        "#define DATA_TYPE 0",
        "#define RESCALE_INPUT_ZP 0",
        "#define RESCALE_MULTIPLIER 1",
        "#define RESCALE_OUTPUT_ZP 0",
        "#define RESCALE_SHIFT 0",
        "#define SUBTRACTION_A 0",
        "#define SUBTRACTION_B 0",
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
        "    int32_t mode1_output_row_stride_bytes, mode1_padded_output_elems;",
        "    const int16_t *a_data;",
        "    int32_t a_data_length, a_row_stride, a_panel_pitch, a_panel_token_stride;",
        "    const uint8_t *w_data, *v_data, *w2_left_data, *w2_right_data;",
        "    int32_t b_data_length, w2_data_length;",
        "    const int16_t *mode0_d0_golden, *mode1_padded_golden;",
        "} shape_cfg_t;",
        "",
        "typedef struct {",
        "    uint32_t layout_id;",
        "    const char *name;",
        "    int32_t a_row_stride;",
        "    const int16_t *a_data;",
        "    int32_t a_data_length;",
        "    const uint8_t *w_data, *v_data, *w2_left_data, *w2_right_data;",
        "    int32_t b_data_length, w2_data_length;",
        "    shape_cfg_t shapes[NUM_SHAPES];",
        "} layout_cfg_t;",
        "",
        *layout_notes,
        "",
        *arrays,
        "",
        "static const layout_cfg_t layout_cfgs[NUM_LAYOUTS] = {",
        ",\n".join(layout_inits),
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
