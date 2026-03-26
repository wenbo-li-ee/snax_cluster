#!/bin/bash
export PATH=/pixi/.pixi/envs/default/bin:$PATH
cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster

echo "=== Step 1: RTL Generation ==="
make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=cfg/snax_versacore_to_cluster.hjson 2>&1
RTL_EXIT=$?
echo "=== RTL Generation exit code: $RTL_EXIT ==="

if [ $RTL_EXIT -ne 0 ]; then
    echo "RTL generation failed!"
    exit 1
fi

echo "=== Step 2: Hardware Build ==="
make CFG_OVERRIDE=cfg/snax_versacore_to_cluster.hjson -C target/snitch_cluster bin/snitch_cluster.vlt -j$(nproc) 2>&1
BUILD_EXIT=$?
echo "=== Hardware Build exit code: $BUILD_EXIT ==="

exit $BUILD_EXIT
