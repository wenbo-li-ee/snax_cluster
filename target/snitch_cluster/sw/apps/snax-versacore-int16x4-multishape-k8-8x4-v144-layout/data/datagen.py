#!/usr/bin/env python3

"""Emit the full-size workload using only the validated V144 layout."""

import importlib.util
import pathlib


_source = (pathlib.Path(__file__).resolve().parents[2]
           / "snax-versacore-int16x4-moe-small-k8-highgran-layout-search"
           / "data" / "datagen.py")
_spec = importlib.util.spec_from_file_location("v144_layout_datagen", _source)
_datagen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_datagen)

_base = {
    "id": 144,
    "name": "v144_full_joint_A4_S2in176_M1stride176",
    "a_pad": 32,
    "shape_a_pads": [32, 0, 80],
    "b1_color": 416,
    "w2l_color": 128,
    "a_color": 128,
    "m1d0_color": 256,
    "a_panel_granularity": 4,
    "a_panel_token_strides": [16, 16, 176],
    "s0_mode1_linear": True,
    "s2_mode1_token_stride": 176,
    "target_granularity": [4, 2, 2],
}

_final = dict(_base)
_final.update({
    "id": 270,
    "name": "v270_full_A4_V144_S1in144_pitch576_bank56",
    "a_panel_token_strides": [16, 144, 176],
    "a_panel_pitches": [0, 576, 0],
    "shape_a_base_banks": [None, 56, None],
})
_datagen.LAYOUTS = [_final]


if __name__ == "__main__":
    _datagen.main()
