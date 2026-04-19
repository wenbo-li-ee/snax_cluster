# SNAX Thesis Skills Development Summary

## Execution Record

This document records the successful execution of the complete SNAX build workflow on commit 2cbac6af, establishing the foundation for dual-VersaCore development.

### Date
April 8, 2026

### Successful Workflow Execution

**All steps completed successfully inside the barnard3 container:**

1. ✅ **RTL Generation** (18-20 seconds)
   - `make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=cfg/snax_versacore_dse_cluster.hjson`
   - Generated VersaCore RTL, streamer wrappers, CSR managers
   - Verified: No module not-found errors, Chisel compilation succeeded

2. ✅ **Software Build** (1-2 minutes)
   - `make -C target/snitch_cluster sw CFG_OVERRIDE=cfg/snax_versacore_dse_cluster.hjson`
   - Compiled all apps including snax-versacore-dse-matmul-profile
   - Verified: All ELF files generated, no linker errors

3. ✅ **Simulator Build** (12-15 minutes)
   - `make -C target/snitch_cluster bin/snitch_cluster.vlt -j4 CFG_OVERRIDE=cfg/snax_versacore_dse_cluster.hjson`
   - Verilator elaboration and C++ compilation successful
   - Binary size: ~50-100 MB as expected

4. ✅ **ELF Execution** (~1 minute)
   - `./target/snitch_cluster/bin/snitch_cluster.vlt ./target/snitch_cluster/sw/apps/snax-versacore-dse-matmul-profile/build/snax-versacore-dse-matmul-profile.elf`
   - Output: "SNAX GEMM Matmul: PASS"
   - Exit code: 0

**Total time**: ~30-40 minutes end-to-end

### Critical Learning: Container Environment

**Root cause of initial failures**: Pixi environment not activated in container.

**Solution**: `source /pixi/entrypoint.sh` must be the first command inside the container, before any other build commands.

**Proven command pattern for container execution:**
```bash
podman exec -it barnard3 bash -lc '
source /pixi/entrypoint.sh
cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster
<your-build-commands>
'
```

## Skills Created

### 1. claude-snax-thesis (Enhanced)
- **Location**: `/esat/studscratch/r1015498/Thesis/original_snax/skills/claude-snax-thesis/`
- **Purpose**: General SNAX cluster development workflow
- **Status**: Existing; minor improvements added based on proven workflow
- **Key reference**: SKILL.md contains comprehensive SNAX workflow guide

### 2. snax-dual-versacore-development (NEW)
- **Location**: `/esat/studscratch/r1015498/Thesis/original_snax/skills/snax-dual-versacore-development/`
- **Purpose**: Specific guidance for dual-VersaCore accelerator development
- **Contents**:
  - **SKILL.md**: Main skill definition with architecture overview and build workflow
  - **references/container-workflow.md**: Detailed container setup and command patterns (extracted from BARNARD3_CONTAINER_WORKFLOW.md)
  - **references/architecture.md**: Hardware architecture, module structure, CSR layout, implementation guidance
  - **agents/openai.yaml**: Agent configuration

### 3. Updated Documentation
- **SNAX_DUAL_VERSACORE_SKILL.md** (workspace root): Captured workflow and architecture decisions
- **BARNARD3_CONTAINER_WORKFLOW.md** (existing): Reference document for container operations

## Key Files Modified/Created

### Skill Files
```
skills/
├── snax-dual-versacore-development/
│   ├── SKILL.md                           (NEW)
│   ├── agents/openai.yaml                 (NEW)
│   └── references/
│       ├── container-workflow.md          (NEW)
│       └── architecture.md                (NEW)
└── README.md
```

### Reference Documentation
```
/esat/studscratch/r1015498/Thesis/original_snax/
├── SNAX_DUAL_VERSACORE_SKILL.md          (NEW)
├── BARNARD3_CONTAINER_WORKFLOW.md        (existing, reference)
└── build_workflow.sh                     (NEW, test script)
```

## Architecture Decisions Recorded

### 1. Computation Completion Semantics
- **Decision**: Separate `computation_done` from `writeback_done`
- **Rationale**: Enable pipelining between compute and post-processing stages
- **Implementation**: Intermediate buffer captures full output on computation complete signal

### 2. Intermediate Buffer Design
- **Purpose**: Bridge between stationary dataflow output and post-processing
- **Key feature**: Backpressure support to prevent deadlock
- **Benefit**: Reduces lane requirements in post-processing stage

### 3. Streamer Organization
- **Input A** (shared): Single streamer for both VersaCores
- **Input B, C** (per-core): Separate streamers for each VersaCore
- **Output** (combined): Single streamer from adder results
- **Synchronization**: Both VersaCores start only when all inputs ready

### 4. Post-Processing Pipeline
- **Stage 1**: Per-core 2-stage shifter (placeholder for actual post-processor)
- **Stage 2**: Element-wise adder combining outputs
- **Design**: Fully pipelinable, no control dependencies

## Container Workflow Proven Pattern

### Prerequisites
- Container: `barnard3` running `ghcr.io/kuleuven-micas/snax:main`
- Mount: `/esat/studscratch/r1015498/Thesis/original_snax` (same path host and container)
- Config: `cfg/snax_versacore_dse_cluster.hjson` (exists and validated)

### Execution Pattern Verified
```bash
podman exec -it barnard3 bash -lc '
source /pixi/entrypoint.sh          # MUST be first
cd /esat/.../snax_cluster           # Navigate
make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=...
make -C target/snitch_cluster sw CFG_OVERRIDE=...
make -C target/snitch_cluster bin/snitch_cluster.vlt -j4 CFG_OVERRIDE=...
./target/snitch_cluster/bin/snitch_cluster.vlt <elf>
'
```

### Success Metrics
- All make targets complete without errors
- Generated files appear in expected locations
- ELF execution produces PASS message with correct output
- Total execution time: 30-40 minutes depending on cache state

## Next Steps for Future Development

### For Next Agent

Use the `snax-dual-versacore-development` skill when:
1. Implementing dual-VersaCore wrapper (Chisel)
2. Writing intermediate buffer RTL (SystemVerilog)
3. Implementing post-processing modules
4. Modifying config for new streamer/CSR parameters
5. Testing complete pipeline

**Provided Resources**:
- Complete architecture specification in skill references
- Proven container/build workflow patterns
- HJSON config baseline
- Software API hints for CSR/streamer access

### Files That Need Implementation

| File | Type | Purpose |
|------|------|---------|
| `VersACoreGen.scala` | Chisel | Dual-core instantiation and wiring |
| `VersACoreIntermediateBuffer.sv` | SV | Register buffer module |
| `VersACoreShifter.sv` | SV | Per-core shifter pipeline |
| `VersACoreElementwiseAdd.sv` | SV | Element-wise addition module |
| `snax_versacore_dse_cluster.hjson` | Config | Dual-core parameters, streamer setup |
| `snax-versacore-dse-lib.h` | C header | New CSR offsets, helper functions |
| Software golden models | Python | Update datagen.py for dual inputs |

### Estimated Effort

- RTL generation + implementation: 2-3 days
- Software API and CSR mappings: 1 day
- Integration testing and debugging: 2-3 days
- **Total**: 5-7 days for complete working dual-core accelerator

## Communication for Next Agent

### Recommended Prompt

> You are extending SNAX cluster development to build a dual-VersaCore accelerator with post-processing pipeline. Reference the `snax-dual-versacore-development` skill and supporting documentation in `/esat/studscratch/r1015498/Thesis/original_snax/skills/`.
>
> **Architecture** (from skill references):
> - Two independent VersaCores (shared input A, separate B/C)
> - Intermediate register buffer capturing full output on K-complete
> - Post-processing: shifter pipeline + element-wise adder
>
> **Baseline**: Commit 2cbac6af (clean, tested single-core)
> **Build Pattern** (proven working):
> ```bash
> source /pixi/entrypoint.sh  # FIRST in container
> make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=cfg/snax_versacore_dse_cluster.hjson
> make -C target/snitch_cluster sw CFG_OVERRIDE=cfg/snax_versacore_dse_cluster.hjson
> make -C target/snitch_cluster bin/snitch_cluster.vlt -j4 CFG_OVERRIDE=cfg/snax_versacore_dse_cluster.hjson
> ./target/snitch_cluster/bin/snitch_cluster.vlt <elf>
> ```
> **Container Entry**: `podman exec -it barnard3 bash -lc 'source /pixi/entrypoint.sh; cd /esat/.../snax_cluster; <commands>'`
>
> Success criteria: All build steps complete, ELF runs and produces PASS message.

## Lessons Learned

1. **Container environment is critical**: `source /pixi/entrypoint.sh` is non-negotiable first step
2. **Build order matters**: Always rtl-gen → sw → bin/vlt → simulation
3. **Config consistency**: Use same `CFG_OVERRIDE` throughout entire workflow
4. **Timing insight**: Simulator build is the longest step; parallelization essential
5. **Backpressure design**: Critical for dual-core synchronization and avoiding deadlock
6. **Intermediate buffer role**: Bridges dataflow styles and enables pipelining

## Documentation Maintenance

- Update SNAX_DUAL_VERSACORE_SKILL.md if new container requirements discovered
- Keep container-workflow.md synchronized with BARNARD3_CONTAINER_WORKFLOW.md
- Architecture.md should be updated as implementation progresses to track actual vs. planned design
- Skills README should track new skills as they are created
