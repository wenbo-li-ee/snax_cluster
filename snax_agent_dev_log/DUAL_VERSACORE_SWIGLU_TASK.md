# Dual VersaCore SwiGLU Pre-computation Accelerator — Task Specification

**Date**: April 12, 2026  
**Author**: Handoff document for next agent  
**Workspace**: `/esat/studscratch/r1015498/Thesis/original_snax`  
**Repository root**: `snax_cluster/`

---

## 1. Goal and Background

### What We Are Building

An accelerator that computes the SwiGLU down-projection pre-computation:

```
output = (Swish(x · W) ⊙ (x · V))
```

where:
- `x` is the input matrix (shared)
- `W`, `V` are two weight matrices
- `⊙` is element-wise multiplication

### Scope for This Phase — Data Flow Only

**Swish** and the element-wise `⊙` are **placeholder modules only** in this phase. The goal is to get the complete data path wired up and verified end-to-end with simulation. The placeholder post-processing blocks are:

- **Path 0 (xW)**: A **6-stage pipeline right-shifter** (shift right 2 bits total: stage 1 shifts right 1 bit, stage 2 shifts right 1 bit, stages 3–6 are pure register stages). This represents the future Swish computation path.
- **Path 1 (xV)**: A **2-stage pipeline right-shifter** (each stage shifts right 1 bit, total 2 bits). This represents the simpler path.
- **Final**: An **element-wise 32-bit adder** combining the two shifted results.

The intentional pipeline depth asymmetry between path 0 (6-stage) and path 1 (2-stage) exists to prove that two paths with different pipeline depths can be aligned by the intermediate buffer + flow control, and to provide headroom for the actual Swish computation in a later phase.

---

## 2. Full Architecture Description

```
Shared Input A (Streamer 0, reader)
         |
         +-------------------+
         |                   |
         v                   v
Input B0 (Streamer 1) →  [VersaCore 0]    Input B1 (Streamer 2) → [VersaCore 1]
                               |                                         |
                               | out_d (serial, 32-bit per element)      | out_d (serial, 32-bit per element)
                               +------------------+-----------------------+
                                                  |
                                         [Large Intermediate Buffer]
                                           (stores all VersaCore outputs before post-proc)
                                         /                              \
                                        v                                v
                              [Shifter 0 — 6-stage]            [Shifter 1 — 2-stage]
                           stages 1-2: shift-right-1-bit       stages 1-2: shift-right-1-bit
                           stages 3-6: pure register
                                        \                                /
                                         v                              v
                                         [Element-wise 32-bit Adder]
                                                      |
                                            Output (Streamer 3, writer)
```

### Key Design Rules

1. **Two VersaCores share input A**. Both cores use the same configuration (same shape/type CSRs).
2. **No C (partial sum) input**. Neither VersaCore uses a C accumulator. The `take_in_new_c` CSR field should always be set to 0 for this accelerator.
3. **Three input streamers + one output streamer** (4 streamers total). Streamer 0 = A (shared). Streamer 1 = B for VersaCore 0. Streamer 2 = B for VersaCore 1. Streamer 3 = output.
4. **Output stationary dataflow only** for this accelerator (no weight/input stationary).
5. **Both VersaCores start simultaneously**, only after data from all three input streamers is ready. Use appropriate synchronization (see §4).
6. **Intermediate buffer**:
   - Sized to hold the *entire* output of both VersaCores (all elements, full parallel width).
   - VersaCore writes via its existing parallel-to-serial converter → the buffer receives serial beats and reconstructs full parallel.
   - Buffer acts as a FIFO of depth 1 (one full capture). It exerts backpressure to VersaCore when full (downstream has not consumed it yet).
   - Finish signal to software: if no output, finish = computation done; if output, finish = entire result has been loaded into the intermediate buffer AND drained by post-processing is a separate concern (post-proc runs downstream).
7. **Post-processing lanes are configurable** in the cfg:
   - E.g., VersaCore outputs 128×32-bit = 4096 bits. If you configure 64 adder lanes, the post-proc needs 2 cycles to process them.
   - The number of shifter-0 lanes, shifter-1 lanes, and adder lanes must all be equal (they process in lock-step).
   - Number of lanes ≤ total VersaCore output elements. Must divide evenly.
8. **Backpressure chain**: output streamer stall → adder back-pressures shifters → shifters back-pressure buffer → buffer back-pressures VersaCore output.
9. **Finish signal exposed to software** via a read-only CSR: computation done (VersaCore finished k-steps and result loaded into buffer).

---

## 3. Filing Strategy — Where to Put New Files

### New cfg file

Create a new file (do NOT overwrite the old DSE cluster cfg):

```
snax_cluster/target/snitch_cluster/cfg/snax_dual_versacore_swiglu_cluster.hjson
```

Base it on `snax_versacore_dse_cluster.hjson`. Key differences:
- `name: snax_dual_versacore_swiglu_cluster`
- `bender_target` must include `snax_dual_versacore_swiglu` (new target)
- Remove the C streamer from `snax_versacore_streamer_template`
- Add config field for post-processing lane count: `snax_dual_versacore_postproc_lanes`
- Adjust TCDM ports and streamer config for 4 streamers (3 in + 1 out)
- Single accelerator block entry under `snax_acc_cfg` with `snax_acc_name: snax_dual_versacore_swiglu`

### New Chisel / SV accelerator

The new top-level accelerator generator should be added as a new `object` (main) in chisel_acc. Suggested approach:

1. **New Chisel file**: `snax_cluster/hw/chisel_acc/src/main/scala/snax_acc/versacore/DualVersaCoreSwigluGen.scala`  
   - Contains `object DualVersaCoreSwigluGen` with a `main` that:
     - Instantiates **two** `VersaCore` modules using the same `SpatialArrayParam`
     - Instantiates the intermediate buffer, two shifters, adder
     - Generates `snax_dual_versacore_swiglu_shell_wrapper.sv` (the SNAX shell wrapper)
     - Generates the C header file to the new lib directory

2. **New SV compute units** (hand-written, placed in `hw/chisel_acc/src/main/resources/snax_acc/versacore/` or a new subfolder):
   - `shifter_6stage.sv` — 6-stage pipelined right-shifter, parameterized DATA_WIDTH (32-bit elements), NUM_LANES
   - `shifter_2stage.sv` — 2-stage pipelined right-shifter, same params
   - `elem_adder_32b.sv` — combinational or 1-cycle element-wise adder, NUM_LANES outputs

   These SV modules are instantiated by Chisel using `BlackBox` (with hand-written SV resource files).

3. **Intermediate buffer** can be a Chisel `Module` (no need for SV): a register with valid/ready handshake, parameterized by total width = `2 × VersaCore_array_output_width`.

### Bender.yml entry

Add a new target `snax_dual_versacore_swiglu` in `snax_cluster/Bender.yml`, listing all generated files (analogous to the existing `snax_versacore` target):

```yaml
- target: snax_dual_versacore_swiglu
  files:
    # Level 0
    - target/snitch_cluster/generated/snax_dual_versacore_swiglu/VersaCore.sv        # re-used
    - target/snitch_cluster/generated/snax_dual_versacore_swiglu/snax_dual_versacore_swiglu_shell_wrapper.sv
    - target/snitch_cluster/generated/snax_dual_versacore_swiglu/shifter_6stage.sv   # new SV
    - target/snitch_cluster/generated/snax_dual_versacore_swiglu/shifter_2stage.sv   # new SV
    - target/snitch_cluster/generated/snax_dual_versacore_swiglu/elem_adder_32b.sv   # new SV
    - target/snitch_cluster/generated/snax_dual_versacore_swiglu/snax_dual_versacore_swiglu_reqrspman_ReqRspManager.sv
    - target/snitch_cluster/generated/snax_dual_versacore_swiglu/snax_dual_versacore_swiglu_Streamer.sv
    # Level 1
    - target/snitch_cluster/generated/snax_dual_versacore_swiglu/snax_dual_versacore_swiglu_csrman_wrapper.sv
    - target/snitch_cluster/generated/snax_dual_versacore_swiglu/snax_dual_versacore_swiglu_streamer_wrapper.sv
    # Level 2
    - target/snitch_cluster/generated/snax_dual_versacore_swiglu/snax_dual_versacore_swiglu_wrapper.sv
```

Also add a corresponding XDMA entry (copy/adapt the `snax_versacore_dse_cluster_xdma` entry if the new cluster also uses an XDMA core, or remove XDMA if not needed).

### New software library

Create:
```
snax_cluster/target/snitch_cluster/sw/snax/dual-versacore-swiglu/
├── include/
│   └── snax-dual-versacore-swiglu-lib.h
└── src/
    └── snax-dual-versacore-swiglu-lib.c
```

The library must expose:
- `set_dual_versacore_streamer_csr(...)` — configures 3 input streamers + 1 output streamer CSRs
- `set_dual_versacore_csr(...)` — configures both VersaCores (same config shared)
- `set_dual_versacore_streamer_start()` — starts streamers
- `set_dual_versacore_start()` — starts accelerator
- `wait_dual_versacore_and_streamer()` — polls finish
- `read_dual_versacore_perf_counter()` — reads perf CSR
- Result check helper function

The CSR address offset layout will be derived from the generated `streamer_csr_addr_map.h` (auto-generated by rtl-gen). The accelerator CSR layout for the new wrapper:
- CSR 0 (RW): `take_in_new_c = 0` (always)
- CSR 1 (RW): `a_b_input_times_one_output` = K (number of K-steps per output tile)
- CSR 2 (RW): `output_times` = M × N (number of output tiles)
- CSR 3 (RW): `subtraction_constant`
- CSR 4 (RW): `array_shape`
- CSR 5 (RW): `data_type`
- CSR 6 (RW): start (write 1 to trigger)
- CSR 7 (RO): busy
- CSR 8 (RO): performance counter

Both VersaCores use the same configuration registers. Map both to the same CSR range in the wrapper.

### New software app

Create:
```
snax_cluster/target/snitch_cluster/sw/apps/snax-dual-versacore-swiglu-test/
├── Makefile
├── data/
│   ├── Makefile
│   ├── datagen.py
│   └── params.hjson
└── src/
    └── snax-dual-versacore-swiglu-test.c
```

---

## 4. VersaCore Synchronization for Two Cores

Look at `VersaCore.scala` lines ~65–105. Each VersaCore starts when `io.ctrl.fire` (= `ctrl.valid && ctrl.ready && cstate === sIDLE`).

For two VersaCores with shared input A:
- The single A streamer sends data to both VersaCores simultaneously (broadcast via wire fork).
- B0 goes to VersaCore 0, B1 goes to VersaCore 1.
- In the shell wrapper, apply the configuration to both VersaCores at the same clock edge. Since the shell wrapper has one `csr_reg_set_valid_i` signal, gate both `io_ctrl_valid` lines with the same single ready signal: only assert `valid` for both VersaCores when both report `ready`. i.e., `ctrl_valid_to_vc = csr_reg_set_valid_i && vc0_ctrl_ready && vc1_ctrl_ready`.
- For input A: broadcast `stream2acc_A_data_i`, `stream2acc_A_valid_i` to both VersaCores. The A streamer's `ready` is the AND of both VersaCores' A ready signals: `stream2acc_A_ready_o = vc0_in_a_ready && vc1_in_a_ready`.

---

## 5. Intermediate Buffer Design

The intermediate buffer sits between the two VersaCore `out_d` ports and the two shifter inputs.

```
vc0.out_d  (serial, DataWidthD bits wide) → |               |
vc1.out_d  (serial, DataWidthD bits wide) → | Buffer Module | → shifter_0_in (parallel, full output width)
                                             |               | → shifter_1_in (parallel, full output width)
```

Simplest correct implementation:
- Buffer depth = 1 row of `(2 × arrayOutputDWidth)` bits.
- VersaCore 0's `D_p2s` serializes its outputs. Buffer collects all serial beats until it has a full parallel word and then asserts `out_valid` downstream.
- Buffer acts as a decoupled register: if `out_ready` (from downstream) is low, it holds. Exerts backpressure: asserts `in_ready = false` when `buf_valid && !out_ready`.
- Both VersaCore 0 and VersaCore 1 outputs are collected into the same buffer (concatenated: `{vc1_out_flat, vc0_out_flat}`).

**Important**: The buffer must collect serial beats from both VersaCores together. Both VersaCores run in lock-step (same K, same shape), so their output beats arrive at the same time. Buffer collects them in pairs.

---

## 6. Post-Processing Block

### 6-Stage Shifter (SV `shifter_6stage.sv`)

```systemverilog
module shifter_6stage #(
    parameter int unsigned DATA_WIDTH = 32,
    parameter int unsigned NUM_LANES  = 64
)(
    input  logic                             clk_i,
    input  logic                             rst_ni,
    input  logic [NUM_LANES-1:0][DATA_WIDTH-1:0] data_i,
    input  logic                             valid_i,
    output logic                             ready_o,
    output logic [NUM_LANES-1:0][DATA_WIDTH-1:0] data_o,
    output logic                             valid_o,
    input  logic                             ready_i
);
```

Behavior:
- Stage 1: `d[1] = d[0] >>> 1` (arithmetic shift right 1) + register
- Stage 2: `d[2] = d[1] >>> 1` (total == `d[0] >>> 2`) + register
- Stages 3–6: pure register delay (data passes through unchanged)
- Full pipeline valid/ready handshake (every stage has its own `valid`/`ready` register)
- No stall between stages unless downstream is not ready (backpressure propagates)

### 2-Stage Shifter (SV `shifter_2stage.sv`)

Same port signature as `shifter_6stage.sv` but only 2 stages:
- Stage 1: `d[1] = d[0] >>> 1` + register
- Stage 2: `d[2] = d[1] >>> 1` (total == `d[0] >>> 2`) + register

Both shifters produce the same numerical result (`>>> 2`) but the 6-stage version has 4 extra cycles of latency. These extra latency cycles are why the intermediate buffer and the pipeline architecture is needed: the faster (2-stage) path must wait for the slower (6-stage) path before the adder can fire.

**Alignment**: The adder should fire when BOTH shifted outputs are valid simultaneously. Since stage counts differ (6 vs. 2), naturally the 6-stage output valid arrives 4 cycles later. The adder simply AND's both valid signals and exerts independent backpressure. No extra FIFO or alignment logic is needed because:
1. Both paths launch from the same buffer read beat.
2. The buffer holds until the downstream adder is ready.
3. The adder's `ready_i_0` is wired to `shifter_6stage.ready_o`, and the 6-stage shifter will stall if the adder is not ready. The 2-stage path, being faster, will wait at the adder's `ready_i_1` until 6-stage is also valid.

### Element-wise Adder (SV `elem_adder_32b.sv`)

```systemverilog
module elem_adder_32b #(
    parameter int unsigned NUM_LANES = 64
)(
    input  logic                             clk_i,
    input  logic                             rst_ni,
    input  logic [NUM_LANES-1:0][31:0]       data_i_0,
    input  logic                             valid_i_0,
    output logic                             ready_o_0,
    input  logic [NUM_LANES-1:0][31:0]       data_i_1,
    input  logic                             valid_i_1,
    output logic                             ready_o_1,
    output logic [NUM_LANES-1:0][31:0]       data_o,
    output logic                             valid_o,
    input  logic                             ready_i
);
```

Behavior:
- Fires when `valid_i_0 && valid_i_1 && ready_i` (joint handshake)
- `data_o[i] = data_i_0[i] + data_i_1[i]` (32-bit, ignore overflow)
- 1-cycle register on output (optional but recommended)
- `ready_o_0 = ready_o_1 = ready_i` (or registered version)

---

## 7. Expected Software Test Logic

### Golden Model

For a matmul of shape M×K (A), K×N (W), K×N (V), with integer int8 elements:

```python
xW = A @ W   # int32 result, shape M×N
xV = A @ V   # int32 result, shape M×N

# Both shifted right 2 bits (placeholder for Swish / rescale)
xW_shifted = xW >> 2  # arithmetic right shift
xV_shifted = xV >> 2

# Element-wise add (placeholder for Swish ⊙ x)
output = xW_shifted + xV_shifted  # int32, shape M×N
```

**Important**: In datagen.py, compute this golden model and emit it as the reference `D[]` array. The software test compares the simulator output against this golden.

### Test C program flow

```c
// DMA: load A, W (as B0), V (as B1) into TCDM

// Configure streamer CSRs for:
//   - Streamer 0 (A): M×K tiles, temporal loops
//   - Streamer 1 (B0 = W): K×N tiles
//   - Streamer 2 (B1 = V): K×N tiles  
//   - Streamer 3 (output): M×N output tiles

// Configure accelerator CSRs:
//   - take_in_new_c = 0
//   - a_b_input_times_one_output = K (for output stationary)
//   - output_times = M * N
//   - subtraction_constant = 0
//   - array_shape, data_type

// Start streamer, start accelerator

// Poll until done

// DMA output back to L3 (or check in TCDM directly)

// Compare with golden model
```

---

## 8. Build and Iteration Workflow

### Container Setup

```bash
podman start -ai barnard3
# OR, from host:
podman exec -it barnard3 bash -lc 'source /pixi/entrypoint.sh; cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster; <commands>'
```

**Always source pixi first inside container:**
```bash
source /pixi/entrypoint.sh
```

### RTL Generation

```bash
make clean  # Per project preference: always clean before regenerating
make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=cfg/snax_dual_versacore_swiglu_cluster.hjson
```

Expected outputs in:
```
target/snitch_cluster/generated/snax_dual_versacore_swiglu/
target/snitch_cluster/generated/snax_dual_versacore_swiglu_xdma/  (if xdma enabled)
target/snitch_cluster/sw/snax/dual-versacore-swiglu/include/snax_dual_versacore_stationarity.h  (generated header)
```

### Software Build

```bash
make -C target/snitch_cluster sw CFG_OVERRIDE=cfg/snax_dual_versacore_swiglu_cluster.hjson
```

Check ELF:
```bash
ls target/snitch_cluster/sw/apps/snax-dual-versacore-swiglu-test/build/*.elf
```

### Hardware Build

```bash
make -C target/snitch_cluster bin/snitch_cluster.vlt -j$(nproc) CFG_OVERRIDE=cfg/snax_dual_versacore_swiglu_cluster.hjson
```

### Run Simulation

```bash
./target/snitch_cluster/bin/snitch_cluster.vlt \
  ./target/snitch_cluster/sw/apps/snax-dual-versacore-swiglu-test/build/snax-dual-versacore-swiglu-test.elf
```

Expected: `PASS` with exit code 0.

---

## 9. Agent Responsibilities

1. **Implement all hardware** (Chisel + SV) as described above.
2. **Create the new cfg** file.
3. **Update Bender.yml** with the new target.
4. **Create all software** (lib header, lib implementation, datagen.py, app main.c, Makefiles).
5. **Iterate the build loop** until simulation output matches the golden model.
6. **Debug independently**: when build fails, read error messages carefully, trace back to the root cause, fix, and retry. Never ask for help; exhaust all available resources (skill files, reference code, error traces) first.
7. **Keep a running development log** in a new file:
   ```
   /esat/studscratch/r1015498/Thesis/original_snax/DUAL_VERSACORE_SWIGLU_DEVLOG.md
   ```
   The log must include:
   - What was changed and why
   - Every bug encountered, with error message, root cause analysis, and fix
   - Timing and iteration count
   - Any unexpected behavior from the framework
8. **Update the relevant skill files** at the end:
   - `skills/snax-dual-versacore-development/SKILL.md` — add any new architectural insights, discovered pitfalls, verified patterns
   - `skills/snax-cluster-workflow/SKILL.md` — update if any workflow quirks are found
   - `skills/versacore-snax-fusion-design/SKILL.md` — if any VersaCore integration patterns are discovered

---

## 10. Key File Paths Reference

| File | Purpose |
|------|---------|
| `target/snitch_cluster/cfg/snax_versacore_dse_cluster.hjson` | **REFERENCE** (do not edit) |
| `target/snitch_cluster/cfg/snax_dual_versacore_swiglu_cluster.hjson` | **NEW** cfg to create |
| `hw/chisel_acc/src/main/scala/snax_acc/versacore/VersaCore.scala` | VersaCore hardware (read-only reference) |
| `hw/chisel_acc/src/main/scala/snax_acc/versacore/VersaCoreGen.scala` | VersaCore generator (read-only reference) |
| `hw/chisel_acc/src/main/scala/snax_acc/versacore/DualVersaCoreSwigluGen.scala` | **NEW** dual-core generator |
| `hw/chisel_acc/src/main/resources/snax_acc/versacore/shifter_6stage.sv` | **NEW** SV module |
| `hw/chisel_acc/src/main/resources/snax_acc/versacore/shifter_2stage.sv` | **NEW** SV module |
| `hw/chisel_acc/src/main/resources/snax_acc/versacore/elem_adder_32b.sv` | **NEW** SV module |
| `Bender.yml` | Add new target `snax_dual_versacore_swiglu` |
| `target/snitch_cluster/sw/snax/dual-versacore-swiglu/include/snax-dual-versacore-swiglu-lib.h` | **NEW** lib header |
| `target/snitch_cluster/sw/snax/dual-versacore-swiglu/src/snax-dual-versacore-swiglu-lib.c` | **NEW** lib source |
| `target/snitch_cluster/sw/apps/snax-dual-versacore-swiglu-test/` | **NEW** test app directory |
| `target/snitch_cluster/sw/apps/snax-versacore-dse-matmul-profile/` | **REFERENCE** existing app |
| `target/snitch_cluster/sw/snax/versacore-dse/` | **REFERENCE** existing library |
| `DUAL_VERSACORE_SWIGLU_DEVLOG.md` | **NEW** development log |

---

## 11. Important Constraints and Warnings

1. **Do NOT modify** `snax_versacore_dse_cluster.hjson` or any existing test app — all work is new file creation or additive.
2. **The C (partial sum) input** is deliberately removed in this new accelerator. The streamer template for the new cfg should have **2 readers** per VersaCore side (but since A is shared: 1 A reader + 1 B0 reader + 1 B1 reader = 3 readers total), and **1 writer** (output). Compare carefully with the reference config which has 3 readers (A, B, C) + 1 read-writer (D).
3. **Lane count parallelism**: The `snax_dual_versacore_postproc_lanes` config parameter controls how many 32-bit shifter/adder units are instantiated in parallel. For example, if VersaCore outputs 128 int32 results and `postproc_lanes = 64`, the post-proc takes 2 cycles per buffer output.
4. **snax_library_name in cfg**: Set it to `dual-versacore-swiglu` so the generator knows where to emit the C header.
5. **SV BlackBox in Chisel**: When using Chisel `BlackBox` to wrap the hand-written SV modules (shifter, adder), use `HasBlackBoxResource` so the SV file is automatically copied to the `generated/` directory by the Chisel build.
6. **Performance counter**: Both VersaCores should contribute to a single performance counter (use VersaCore 0's counter as the primary, or max of both).

---

## 12. Skills to Read Before Starting

Read these skill files for context:
- `skills/snax-cluster-workflow/SKILL.md`
- `skills/snax-dual-versacore-development/SKILL.md`
- `skills/versacore-snax-fusion-design/SKILL.md`
- `SNAX_DUAL_VERSACORE_SKILL.md` (root-level overview)
- `COMPREHENSIVE_AGENT_HANDOFF.md` (root-level context)
