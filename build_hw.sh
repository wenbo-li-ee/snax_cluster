#!/bin/bash
export PATH=/pixi/.pixi/envs/default/bin:$PATH
cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster
make CFG_OVERRIDE=cfg/snax_versacore_to_cluster.hjson -C target/snitch_cluster bin/snitch_cluster.vlt -j4 2>&1
