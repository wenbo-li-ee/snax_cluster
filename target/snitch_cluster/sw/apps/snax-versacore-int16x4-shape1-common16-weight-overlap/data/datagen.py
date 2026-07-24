#!/usr/bin/env python3

"""Generate common-16 data with W/V interleaved by one 512-bit DMA beat."""

import argparse
import importlib.util
import os

import hjson
import numpy as np


THIS_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DATAGEN = os.path.realpath(os.path.join(
    THIS_DIR,
    "../../snax-versacore-int16x4-multishape-k8-common16-chunked/"
    "data/datagen.py",
))


def load_base_datagen():
    spec = importlib.util.spec_from_file_location(
        "common16_base_datagen", BASE_DATAGEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_wv_interleaved(base, params):
    k0 = int(params["K0_total"])
    n0 = int(params["N0_total"])
    w = base.pack_int4(base.make_sparse_s0_weights(
        n0 // 4, k0 // 8, seed=1, sign=1))
    v = base.pack_int4(base.make_sparse_s0_weights(
        n0 // 4, k0 // 8, seed=3, sign=1))
    assert w.nbytes == v.nbytes
    assert w.nbytes % 64 == 0
    return np.stack(
        (w.reshape(-1, 64), v.reshape(-1, 64)), axis=1).reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--swcfg", required=True)
    parser.add_argument("--hwcfg", required=True)
    args = parser.parse_args()

    with open(args.swcfg, "r", encoding="utf-8") as handle:
        params = hjson.load(handle)
    with open(args.hwcfg, "r", encoding="utf-8") as handle:
        hwcfg = hjson.load(handle)

    base = load_base_datagen()
    header = base.emit_header(params, hwcfg)
    wv = base.emit_u8("WV_interleaved", make_wv_interleaved(base, params))
    marker = "static const shape_cfg_t shape_cfgs[NUM_SHAPES] = {"
    assert marker in header
    print(header.replace(marker, wv + "\n\n" + marker, 1))


if __name__ == "__main__":
    main()
