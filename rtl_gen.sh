#!/bin/bash
export PATH=/pixi/.pixi/envs/default/bin:$PATH
cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster
echo "=== Running RTL generation ==="
make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=cfg/snax_versacore_to_cluster.hjson 2>&1
echo "=== RTL generation exit code: $? ==="
echo "=== Generated files ==="
ls -la target/snitch_cluster/generated/ 2>&1
