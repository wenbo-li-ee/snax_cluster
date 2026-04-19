# Agent Prompt: Dual VersaCore SwiGLU Accelerator Implementation

---

## ⚠️ CRITICAL: ALL BUILD COMMANDS MUST RUN INSIDE THE CONTAINER

**You cannot run `make`, `verilator`, `sbt`, `bender`, or any SNAX build command on the host directly. They are only available inside the `barnard3` podman container.**

Enter the container interactively:
```bash
podman start -ai barnard3
```
Then immediately initialize the environment — **this must be the very first command every session**:
```bash
source /pixi/entrypoint.sh
cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster
```

Or run a single command non-interactively from the host:
```bash
podman exec -it barnard3 bash -lc '
source /pixi/entrypoint.sh
cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster
<your command here>
'
```

File editing (read/write source files) can be done on the host outside the container. Only build/simulation commands need the container.

---

You are implementing a new SNAX hardware accelerator from scratch. The workspace is `/esat/studscratch/r1015498/Thesis/original_snax`, and all work must be done in SNAX's existing Chisel + SystemVerilog build system.

**Before starting, read these files for context and workflow:**
- `skills/snax-cluster-workflow/SKILL.md`
- `skills/snax-dual-versacore-development/SKILL.md`
- `skills/versacore-snax-fusion-design/SKILL.md`
- `SNAX_DUAL_VERSACORE_SKILL.md`
- `COMPREHENSIVE_AGENT_HANDOFF.md`
- `DUAL_VERSACORE_SWIGLU_TASK.md` ← full specification document

**The detailed specification is in `DUAL_VERSACORE_SWIGLU_TASK.md`. Read it completely before writing any code.** Below is a compact summary of what to implement, and the exact deliverables.

---

## Summary: What to Build

A new SNAX accelerator that computes the first phase of SwiGLU: `(Swish(xW) ⊙ xV)`. In this phase Swish and ⊙ are placeholder modules; the goal is to verify the dataflow end-to-end.

### Architecture

```
Shared A (Streamer 0) ──┬──────────────────────┐
                        │                      │
                   VersaCore 0 ← B0 (S1)   VersaCore 1 ← B1 (S2)
                        │                      │
                     out_d                   out_d   (both serial, 32-bit elements)
                        └──────┬───────────────┘
                     [Large Intermediate Buffer]
                        /                     \
               [Shifter-0: 6-stage]    [Shifter-1: 2-stage]
             (>>> 1 in stage 1+2,      (>>> 1 in stage 1+2)
              stages 3-6 pure reg)
                        \                     /
                    [Element-wise 32-bit Adder]
                                 │
                         Output (Streamer 3)
```

Key points:
- Output stationary only; no C (partial sum) input; both VersaCores share config CSRs
- 3 input streamers (A shared, B0, B1) + 1 output streamer = 4 total
- Intermediate buffer stores the complete output of both VersaCores before post-processing
- Configurable number of post-processing lanes (shifter/adder units) via cfg parameter `snax_dual_versacore_postproc_lanes`
- Full backpressure: output streamer stall propagates backwards through adder → shifters → buffer → VersaCore
- Finish signal exposed to software as a read-only busy CSR

---

## Deliverables (Files to Create)

### Hardware

1. **`snax_cluster/target/snitch_cluster/cfg/snax_dual_versacore_swiglu_cluster.hjson`**
   - New cluster config based on `snax_versacore_dse_cluster.hjson`
   - Name: `snax_dual_versacore_swiglu_cluster`
   - 4 streamers: 3 readers (A, B0, B1) + 1 writer (output)
   - New field: `snax_dual_versacore_postproc_lanes: 64` (or appropriate value)
   - No C/partial-sum reader in streamer template

2. **`snax_cluster/hw/chisel_acc/src/main/scala/snax_acc/versacore/DualVersaCoreSwigluGen.scala`**
   - New Chisel generator `object DualVersaCoreSwigluGen` instantiating two VersaCores + buffer + shifters + adder
   - Generates `snax_dual_versacore_swiglu_shell_wrapper.sv` (SNAX integration wrapper)
   - Generates C stationarity header to the new lib path

3. **`snax_cluster/hw/chisel_acc/src/main/resources/snax_acc/versacore/shifter_6stage.sv`**
   - 6-stage pipelined arithmetic right-shifter by 2 bits total, parameterized DATA_WIDTH and NUM_LANES
   - Stage 1+2: shift-right-1 each; stages 3–6: pure register pipeline
   - Full valid/ready handshake with backpressure per stage

4. **`snax_cluster/hw/chisel_acc/src/main/resources/snax_acc/versacore/shifter_2stage.sv`**
   - 2-stage pipelined arithmetic right-shifter by 2 bits total
   - Stage 1+2: shift-right-1 each
   - Full valid/ready handshake

5. **`snax_cluster/hw/chisel_acc/src/main/resources/snax_acc/versacore/elem_adder_32b.sv`**
   - Element-wise 32-bit adder, NUM_LANES outputs
   - Two input ports (path 0 + path 1) with joint valid/ready handshake
   - 1-cycle register on output

6. **`snax_cluster/Bender.yml`** — Add new target `snax_dual_versacore_swiglu` listing all generated SV files

### Software

7. **`snax_cluster/target/snitch_cluster/sw/snax/dual-versacore-swiglu/include/snax-dual-versacore-swiglu-lib.h`**
8. **`snax_cluster/target/snitch_cluster/sw/snax/dual-versacore-swiglu/src/snax-dual-versacore-swiglu-lib.c`**
   - Functions to configure all streamers, configure both VersaCores, start, poll, check result,  read perf counter
   - CSR offsets derived from generated `streamer_csr_addr_map.h`

9. **`snax_cluster/target/snitch_cluster/sw/apps/snax-dual-versacore-swiglu-test/Makefile`**
10. **`snax_cluster/target/snitch_cluster/sw/apps/snax-dual-versacore-swiglu-test/data/datagen.py`**
    - Generates int8 test inputs A, W, V and int32 golden output
    - Golden model: `output = (A @ W >> 2) + (A @ V >> 2)` (arithmetic right shift)
11. **`snax_cluster/target/snitch_cluster/sw/apps/snax-dual-versacore-swiglu-test/data/params.hjson`**
12. **`snax_cluster/target/snitch_cluster/sw/apps/snax-dual-versacore-swiglu-test/data/Makefile`**
13. **`snax_cluster/target/snitch_cluster/sw/apps/snax-dual-versacore-swiglu-test/src/snax-dual-versacore-swiglu-test.c`**
    - DMA data in, configure, start, poll, DMA out, compare with golden

### Documentation

14. **`DUAL_VERSACORE_SWIGLU_DEVLOG.md`** (in repo root `/esat/studscratch/r1015498/Thesis/original_snax/`)
    - Running development log: record every change, every bug, root cause, and fix as you go
    - Format: dated entries, each with: what changed, error encountered, analysis, resolution

15. **Update skills** at the end of the task:
    - `skills/snax-dual-versacore-development/SKILL.md`
    - `skills/snax-cluster-workflow/SKILL.md` (if new workflow quirks found)

---

## Build Commands (in container)

```bash
# Enter container
podman start -ai barnard3
# Inside container, always initialize pixi first:
source /pixi/entrypoint.sh
cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster

# Step 1: RTL generation
make clean
make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=cfg/snax_dual_versacore_swiglu_cluster.hjson

# Step 2: Software build
make -C target/snitch_cluster sw CFG_OVERRIDE=cfg/snax_dual_versacore_swiglu_cluster.hjson

# Step 3: Simulator build
make -C target/snitch_cluster bin/snitch_cluster.vlt -j$(nproc) CFG_OVERRIDE=cfg/snax_dual_versacore_swiglu_cluster.hjson

# Step 4: Run simulation
./target/snitch_cluster/bin/snitch_cluster.vlt \
  ./target/snitch_cluster/sw/apps/snax-dual-versacore-swiglu-test/build/snax-dual-versacore-swiglu-test.elf
```

From host without entering interactive container:
```bash
podman exec -it barnard3 bash -lc '
source /pixi/entrypoint.sh
cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster
<command here>
'
```

---

## Iteration Protocol

1. Start with RTL: get all hardware files written, then run `rtl-gen`.
2. Fix all RTL generation errors (Chisel compilation, SV parsing, missing module references) before moving to SW.
3. Write SW after RTL generation succeeds.
4. Build simulator after SW compiles.
5. Run simulation and compare with golden.
6. If results do not match: add `printf` debug in C, re-examine waveforms conceptually, trace CSR address mapping issues, check shifter/adder bit arithmetic.
7. Keep iterating until `PASS` is printed with exit code 0.
8. Never skip a step by assuming it works — verify each step explicitly.

**Do not stop for bugs.** Analyze, fix, and continue.

---

## Reference Code Locations

When implementing, study these existing files:

| What to implement | Reference to study |
|---|---|
| New cfg syntax | `target/snitch_cluster/cfg/snax_versacore_dse_cluster.hjson` |
| Chisel generator pattern | `hw/chisel_acc/src/main/scala/snax_acc/versacore/VersaCoreGen.scala` |
| Shell wrapper SV generation | `VersaCoreGen.scala` lines 170–280 |
| Streamer template in cfg | `snax_versacore_streamer_template` section in DSE cluster cfg |
| SW library structure | `target/snitch_cluster/sw/snax/versacore-dse/` |
| App structure | `target/snitch_cluster/sw/apps/snax-versacore-dse-matmul-profile/` |
| Datagen pattern | `snax-versacore-dse-matmul-profile/data/datagen.py` |
| Bender.yml target format | Lines 360–374 of `Bender.yml` |

---

## Success Criteria

The task is complete when:
1. `rtl-gen` completes with no errors
2. `sw` build produces a valid ELF
3. `bin/snitch_cluster.vlt` links successfully
4. ELF runs and prints `PASS` with exit code 0
5. The golden model comparison confirms `output = (A@W >> 2) + (A@V >> 2)` matches hardware output
6. `DUAL_VERSACORE_SWIGLU_DEVLOG.md` contains a complete record of the development process
7. Relevant skill files have been updated with lessons learned
