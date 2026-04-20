#!/bin/bash
# Run all 12 SwiGLU experiments (4 apps × 3 shapes)
# Usage: bash run_all_experiments.sh

set -e

SNAX_ROOT="/esat/studscratch/r1015498/Thesis/original_snax/snax_cluster"
TARGET_DIR="$SNAX_ROOT/target/snitch_cluster"
APPS_DIR="$TARGET_DIR/sw/apps"
PIXI="$HOME/.pixi/bin/pixi"
PIXI_LIB="$SNAX_ROOT/.pixi/envs/default/lib"
CFG="cfg/snax_dual_versacore_int16x4_cluster.hjson"
RESULTS_DIR="$TARGET_DIR/experiment_results"

mkdir -p "$RESULTS_DIR"

APPS=(
    "snax-versacore-int16x4-swiglu-m1-batch"
    "snax-versacore-int16x4-swiglu-m1-pingpong"
    "snax-versacore-int16x4-swiglu-m4-batch"
    "snax-versacore-int16x4-swiglu-m4-pingpong"
)

SHAPES=(0 1 2)

for app in "${APPS[@]}"; do
    for shape in "${SHAPES[@]}"; do
        echo "============================================"
        echo "Running: $app shape=$shape"
        echo "============================================"

        PARAMS="$APPS_DIR/$app/data/params.hjson"

        # Update array_shape in params.hjson
        sed -i "s/array_shape: [0-9]*/array_shape: $shape/" "$PARAMS"

        # Clean and rebuild
        rm -f "$APPS_DIR/$app/data/data.h"
        rm -rf "$APPS_DIR/$app/build"

        cd "$SNAX_ROOT"
        $PIXI run -- bash -c "cd target/snitch_cluster/sw/apps/$app && make CFG_OVERRIDE=$CFG" 2>&1 | tail -5

        if [ ! -f "$APPS_DIR/$app/build/$app.elf" ]; then
            echo "ERROR: Build failed for $app shape=$shape"
            continue
        fi

        # Run simulation
        cd "$TARGET_DIR"
        OUTFILE="$RESULTS_DIR/${app}_shape${shape}.log"
        echo "Simulating..."
        LD_LIBRARY_PATH="$PIXI_LIB:$LD_LIBRARY_PATH" ./bin/snitch_cluster.vlt \
            "$APPS_DIR/$app/build/$app.elf" 2>&1 | tee "$OUTFILE"

        echo ""
        echo "Result saved to: $OUTFILE"
        echo ""
    done
done

echo "============================================"
echo "All experiments complete. Results in: $RESULTS_DIR"
echo "============================================"
