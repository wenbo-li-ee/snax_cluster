# Agent Task: Dual VersaCore SwiGLU — Mode 0/1 + RescaleDown Integration

You are working on a SNAX cluster accelerator project. Your task is to extend the existing Dual VersaCore SwiGLU accelerator with mode switching (SwiGLU vs plain GEMM), RescaleDown modules, a real element-wise multiplier, and a second writer streamer. Then validate everything with a full end-to-end SW test.

---

## CRITICAL: Environment and Workflow

### Container — ALL build/sim commands must run inside the container

```bash
# All make/sim commands use this prefix:
podman exec barnard3 bash -lc 'source /pixi/entrypoint.sh; cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster/target/snitch_cluster; <YOUR_COMMAND>'
```

Do NOT run `make` or `bin/snitch_cluster.vlt` directly in the host shell.

### Git branch

Work exclusively on branch `swiglue`:
```bash
cd /esat/studscratch/r1015498/Thesis/original_snax/snax_cluster
git checkout swiglue
```

Base state: commit `6c7cf48a` — "perf: minimal CSR reconfiguration for block pipeline"

### Standard build order (when HW or RTL changes)

Always use `CFG_OVERRIDE=cfg/snax_dual_versacore_swiglu_cluster.hjson`.

```bash
# 1. RTL generation (when hjson, Scala generators, or SV resources change)
make CFG_OVERRIDE=cfg/snax_dual_versacore_swiglu_cluster.hjson rtl-gen

# 2. Software build (when SW or generated headers change)
make clean   # per user preference, always clean before sw build
make CFG_OVERRIDE=cfg/snax_dual_versacore_swiglu_cluster.hjson sw

# 3. Hardware build (Verilator compile, when RTL changes)
make CFG_OVERRIDE=cfg/snax_dual_versacore_swiglu_cluster.hjson bin/snitch_cluster.vlt

# 4. Simulation
bin/snitch_cluster.vlt --spm-size 1048576 \
  sw/apps/snax-dual-versacore-swiglu-test/build/snax-dual-versacore-swiglu-test.bin \
  2>&1 | grep -E "SwiGLU|PASS|FAIL|Error|EXIT|cycle"
```

The Verilator build takes ~12 minutes. Only rebuild `bin/snitch_cluster.vlt` when RTL actually changes.

### SNAX workflow reference

Read `/esat/studscratch/r1015498/Thesis/original_snax/skills/snax-cluster-workflow/SKILL.md` before starting.

---

## CRITICAL: Dev Log — Read First, Write Throughout

### Read existing logs before starting

Go through ALL files in `/esat/studscratch/r1015498/Thesis/original_snax/snax_agent_dev_log/` before touching any code. Key files:
- `DUAL_VERSACORE_SWIGLU_DEVLOG.md` — previous architecture decisions and verified functionality
- `BLOCK_PIPELINE_CSR_OPTIMIZATION_RESULTS.md` — known pitfalls with snaxgen.py AddrSelOffSet
- `DUAL_VERSACORE_CFG_GUIDE_ZH.md` — existing parameter documentation

### Write your dev log as you work

Create and continuously update: `/esat/studscratch/r1015498/Thesis/original_snax/snax_agent_dev_log/RESCALE_MODE_INTEGRATION_DEVLOG.md`

Record as you go (not only at the end):
- What you changed and why
- RTL gen / build / sim outcomes
- Every bug you hit: symptom → root cause → fix
- Cycle counts and correctness results per experiment

**Context window warning:** This task is long. After ~30 tool calls, re-read your own devlog to recover context before continuing.

### Update skills on bugs found

When you find and fix a bug, update the relevant skill:
- `/esat/studscratch/r1015498/Thesis/original_snax/skills/snax-cluster-workflow/SKILL.md`
- `/esat/studscratch/r1015498/Thesis/original_snax/skills/snax-dual-versacore-development/SKILL.md`

### Write user configuration guide (Chinese) as you work

Update (or create) `/esat/studscratch/r1015498/Thesis/original_snax/snax_agent_dev_log/DUAL_VERSACORE_CFG_GUIDE_ZH.md` throughout development. This is a guide for end users, written in Chinese. It must cover:
- All CSR parameters (old and new), what they mean, valid values
- MODE CSR (0=SwiGLU, 1=GEMM)
- RescaleDown parameters: input_zp, multiplier, output_zp, shift — meaning, how to compute, and the identity/debug values (all zero except multiplier=1)
- Dual writer hjson configuration: spatial_bounds, num_channel, sparse_interconnect_config
- Full SW call sequence for mode 0 and mode 1
- Complete SwiGLU 3-step flow
- Constraint table (all parameter interdependencies)

---

## Detailed Technical Specification

Full spec: `/esat/studscratch/r1015498/Thesis/original_snax/snax_agent_dev_log/RESCALE_MODE_INTEGRATION_SPEC.md`

Summary of changes below.

---

### Task 1: RescaleDown modules (int32 → int16, registered output)

**New SV resource file:** `hw/chisel_acc/src/main/resources/snax_acc/versacore/rescale_down_32to16.sv`

The accelerator currently computes int32 outputs. We need to scale them down to int16 before further processing.

Interface:
```sv
module rescale_down_32to16 #(parameter int unsigned NUM_LANES = 64)(
    input  logic clk_i, rst_ni,
    input  logic signed [31:0] input_zp,
    input  logic        [31:0] multiplier,
    input  logic signed [31:0] output_zp,
    input  logic        [7:0]  shift,
    input  logic [NUM_LANES-1:0][31:0] data_i,
    input  logic valid_i,
    output logic ready_o,
    output logic [NUM_LANES-1:0][15:0] data_o,
    output logic valid_o,
    input  logic ready_i
);
```

Per-lane combinational logic — implement exactly as `RescaleDownPE(in=32, out=16)` from:
`snax_cluster/hw/chisel/src/main/scala/snax/DataPathExtension/RescaleDown.scala`

Key steps (all in signed arithmetic):
1. `zero_compensated = data_i - input_zp`
2. `multiplied = zero_compensated * {1'b0, multiplier}` (int64)
3. `shifted_one = 1 << (shift - 1)` (rounding addend)
4. `shifted_data = multiplied + shifted_one`
5. `scaled_32 = (zero_compensated >= 0) ? shifted_data + (1<<30) : shifted_data - (1<<30)`
6. `correct_shift = (shift > 31) ? scaled_32 : shifted_data`
7. `shifted_value = correct_shift >>> shift` (arithmetic right shift)
8. `result = shifted_value[31:0] + output_zp`
9. Clamp to int16 range: `max(-32768, min(32767, result))`

Registered output (1-cycle latency, backpressure):
```sv
assign ready_o = !valid_o || ready_i;
// always_ff: when ready_o is high, latch valid_i and data
```

**Identity (no-op) parameter values for debug:** `input_zp=0, multiplier=1, output_zp=0, shift=0`
With these values: output = clamp(input, -32768, 32767) — passes through any int32 that fits in int16.

Three instances needed in the wrapper:
- `u_rescale0` — after chunk_ser0, before shifter_6stage (VC0 path)
- `u_rescale1` — after chunk_ser1 (VC1 path)
- `u_rescale_mul` — after elem_mul_16b (mode 0 final output)

---

### Task 2: Real element-wise multiplier (int16 × int16 → int32)

**New SV resource file:** `hw/chisel_acc/src/main/resources/snax_acc/versacore/elem_mul_16b.sv`

Model the interface and handshake exactly on `elem_adder_32b.sv` (already in the same resource directory). Change:
- Input widths: `[NUM_LANES-1:0][15:0]` (int16 signed) for both ports
- Output width: `[NUM_LANES-1:0][31:0]` (int32 signed)
- Operation: `$signed(data_i_0[l]) * $signed(data_i_1[l])` per lane

Keep the joint handshake and registered output structure identical to `elem_adder_32b`.

---

### Task 3: SiLU placeholder — shifter_6stage with DATA_WIDTH=16

The file `shifter_6stage.sv` already exists and has a `DATA_WIDTH` parameter.

In `DualVersaCoreSwigluGen.scala`, change the instantiation from:
```sv
shifter_6stage #(.DATA_WIDTH(32), .NUM_LANES(PostprocLanes)) u_shifter_6stage (...)
```
to:
```sv
shifter_6stage #(.DATA_WIDTH(16), .NUM_LANES(PostprocLanes)) u_shifter_6stage (...)
```

The shifter still performs arithmetic right shift by 2 across 6 pipeline stages. At DATA_WIDTH=16 it operates on int16 values.

Remove `shifter_2stage` entirely — do not instantiate it and remove it from the resource copy list.

---

### Task 4: Mode CSR + data path mux

Add `MODE` as `csr_reg_set_i[6]` (new 7th RW CSR, `snax_num_rw_csr` increases from 7 to 20):
- `mode_sel = csr_reg_set_i[6][0]`
- `0` = Mode 0: SwiGLU
- `1` = Mode 1: plain GEMM

**Mode 0 data flow:**
```
chunk_ser0 → u_rescale0(int32→int16) → u_shifter_6stage(SiLU, 16b) ──► ElemMul input 0
chunk_ser1 → u_rescale1(int32→int16) ─────────────────────────────►  ElemMul input 1
                                              u_elem_mul_16b(int16→int32)
                                                        │
                                              u_rescale_mul(int32→int16)
                                                        │
                                            out_assemble0 → acc2stream_0 → Writer0
acc2stream_1 : valid = 0 (Writer1 idle)
```

**Mode 1 data flow:**
```
chunk_ser0 → u_rescale0(int32→int16) → out_assemble0 → acc2stream_0 → Writer0
chunk_ser1 → u_rescale1(int32→int16) → out_assemble1 → acc2stream_1 → Writer1
```

ElemMul and RescaleMul are still instantiated in mode 1, but their inputs receive `valid=0` so they stay idle.

The `mode_sel` mux is purely combinational and statically held during one computation.

---

### Task 5: Second writer streamer + second output port

**Shell wrapper new port:**
```sv
output logic [DataWidthOut-1:0] acc2stream_1_data_o,
output logic                   acc2stream_1_valid_o,
input  logic                   acc2stream_1_ready_i,
```
where `DataWidthOut = DataWidthD / 2` = 2048 bits (PostprocLanes=64 lanes × 16 bits × NumChunks=2 chunks reassembled).

**out_assemble** width: change from `DataWidthD` (4096b) to `DataWidthOut` (2048b) throughout.

---

### Task 6: Updated RW CSR count — snax_num_rw_csr = 20

The 20 RW CSRs in order (csr_reg_set_i indices 0–19):

| Index | Name |
|-------|------|
| 0 | OVERWRITE_ACCUM |
| 1 | ACCUM_BOUND |
| 2 | OUTPUT_BOUND |
| 3 | SUBTRACTIONS |
| 4 | ARRAY_SHAPE_CFG |
| 5 | DATA_TYPE_CFG |
| 6 | MODE (0=SwiGLU, 1=GEMM) |
| 7–10 | RESCALE0: input_zp, multiplier, output_zp, shift |
| 11–14 | RESCALE1: input_zp, multiplier, output_zp, shift |
| 15–18 | RESCALE_MUL: input_zp, multiplier, output_zp, shift |
| 19 | START (was index 6) |

---

## Files to Modify

| File | Change |
|------|--------|
| `hw/chisel_acc/src/main/resources/snax_acc/versacore/rescale_down_32to16.sv` | **NEW** |
| `hw/chisel_acc/src/main/resources/snax_acc/versacore/elem_mul_16b.sv` | **NEW** |
| `hw/chisel_acc/src/main/scala/snax_acc/versacore/DualVersaCoreSwigluGen.scala` | Major rewrite of post-proc section, new ports, new CSRs, mode mux, resource copy list |
| `hw/chisel/src/main/scala/snax/streamer/Streamer.scala` | `numReadOnlyReg` 3→4; add Writer1 busy RO CSR |
| `util/snaxgen/snaxgen.py` | `+1` → `+2` for writer-busy RO CSR count |
| `target/snitch_cluster/cfg/snax_dual_versacore_swiglu_cluster.hjson` | `snax_num_rw_csr: 20`; dual writer params; updated sparse_interconnect_config |
| `target/snitch_cluster/sw/snax/dual-versacore-swiglu/include/snax-dual-versacore-swiglu-lib.h` | New CSR defines (MODE, RESCALE0–2), DUAL_VC_START moved to +19 |
| `target/snitch_cluster/sw/snax/dual-versacore-swiglu/src/snax-dual-versacore-swiglu-lib.c` | New setter functions, dual-writer streamer config |
| `target/snitch_cluster/sw/apps/snax-dual-versacore-swiglu-test/data/datagen.py` | Updated golden model, W₂ generation, int16 strides, new delta offsets |
| `target/snitch_cluster/sw/apps/snax-dual-versacore-swiglu-test/src/snax-dual-versacore-swiglu-test.c` | Complete 3-step SwiGLU flow: mode0 → SW cast → mode1 → verify |

---

## hjson Diff Summary

```hjson
// snax_num_rw_csr: 7  →  20

// data_writer_params: 1 writer (64ch) → 2 writers (32ch each)
data_writer_params:
{
    spatial_bounds:  [[ 32 ], [ 32 ]]
    temporal_dim:    [ 4, 4 ]
    num_channel:     [ 32, 32 ]
    fifo_depth:      [ 1, 1 ]
    configurable_channel: [ 1, 1 ]
    tcdm_logic_word_size: [[ 256, 128, 64 ], [ 256, 128, 64 ]]
}

// sparse_interconnect_config: [64,16] → [32,8] + [32,8]
sparse_interconnect_config:
[
    [ 16,  4 ]   // Reader A   (unchanged)
    [ 128, 8 ]   // Reader B0  (unchanged)
    [ 128, 8 ]   // Reader B1  (unchanged)
    [ 32,  8 ]   // Writer D0
    [ 32,  8 ]   // Writer D1
]
// snax_tcdm_ports: 336 — UNCHANGED (16+128+128+32+32 = 336)
```

---

## snaxgen.py Diff Summary

```python
# Find this line in util/snaxgen/snaxgen.py:
streamer_csr_num += 1  # Writer busy register

# Change to:
streamer_csr_num += 2  # Two writer busy registers (Writer0 + Writer1)
```

**Why this matters:** This controls `AddrSelOffSet` in the generated `snax_csr_mux_demux`. If the count is wrong by 1, all accelerator CSR writes are offset by 1, causing silent misrouting or simulation hang. This was the root cause of a previous bug documented in `BLOCK_PIPELINE_CSR_OPTIMIZATION_RESULTS.md`.

---

## Streamer.scala Diff Summary

```scala
// In Streamer.scala, find numReadOnlyReg
// Change:  numReadOnlyReg = 3
// To:      numReadOnlyReg = 4

// Add Writer1 busy to readOnlyReg(3):
readOnlyReg(3) := writers(1).io.streamStatus  // or equivalent Writer1 busy signal
// (Follow the same pattern as readOnlyReg(2) which is the Writer0 busy OR signal)
```

---

## datagen.py: Golden Model Pipeline

Use identity rescale parameters during initial testing (`input_zp=0, multiplier=1, output_zp=0, shift=0`). With shift=0, the RescaleDownPE formula reduces to: `output = clamp(input - 0) * 1 + 0 = clamp(input, -32768, 32767)`. So any int32 value fitting in int16 passes unchanged.

```python
def rescale_down_32to16(arr_int32, input_zp, mult, output_zp, shift):
    """Golden model matching RescaleDownPE hardware logic."""
    result = arr_int32.astype(np.int64)
    result = result - input_zp
    multiplied = result * np.int64(mult)
    if shift > 0:
        shifted_one = np.int64(1) << (shift - 1)
        shifted_data = multiplied + shifted_one
        scaled_32 = np.where(result >= 0, shifted_data + (1 << 30), shifted_data - (1 << 30))
        correct_shift = np.where(shift > 31, scaled_32, shifted_data)
        shifted_value = correct_shift >> shift  # arithmetic in numpy int64
    else:
        shifted_value = multiplied
    out = shifted_value.astype(np.int32) + output_zp
    return np.clip(out, -32768, 32767).astype(np.int16)

def arithmetic_right_shift_int16(arr_int16, n):
    """SiLU placeholder: arithmetic right shift by n on int16."""
    return (arr_int16.astype(np.int32) >> n).clip(-32768, 32767).astype(np.int16)
```

Mode 0 golden flow:
```python
vc0_int32  = block_gemm_golden(A, W)                    # int8×int8→int32
vc0_int16  = rescale_down_32to16(vc0_int32, ...)         # int32→int16
vc0_silu   = arithmetic_right_shift_int16(vc0_int16, 2)  # SiLU placeholder

vc1_int32  = block_gemm_golden(A, V)                    # int8×int8→int32
vc1_int16  = rescale_down_32to16(vc1_int32, ...)         # int32→int16

mul_int32  = vc0_silu.astype(np.int32) * vc1_int16.astype(np.int32)  # int16×int16→int32
mode0_out  = rescale_down_32to16(mul_int32, ...)          # int32→int16
```

Mode 1 golden flow:
```python
# SW quantize int16 → int8
mode1_a = np.clip(mode0_out.astype(np.int32), -128, 127).astype(np.int8)

# Mode 1: A @ W2_left, A @ W2_right
d0_int32 = block_gemm_golden(mode1_a, W2_left)
d1_int32 = block_gemm_golden(mode1_a, W2_right)
d0_golden = rescale_down_32to16(d0_int32, ...)   # int32→int16
d1_golden = rescale_down_32to16(d1_int32, ...)   # int32→int16
```

**int16 output stride:** `Dtlstride2 = meshRow * meshCol * 2 = 16 * 8 * 2 = 256` bytes (was 512 for int32). Update all Dtlstride calculations and assertions in datagen.py.

---

## Suggested Workload Parameters

Start small to verify correctness, then expand.

**Mode 0 standalone test first:**
- `M=2, K=2, N=2`, identity rescale params
- Expected: `PASS, Error: 0`

**Full SwiGLU end-to-end:**
- Mode 0: `M=4, K=2, N=2`
- Mode 1: input is `[64, 16]` (mode0 output cast to int8), W₂ is `[16, 16]`
  - `N=2` for mode 1 so W₂ splits into 2 halves of 8 columns each
- Both modes must pass

**Note on W₂ dimensions:**
- Mode 0 output shape: `[M*meshRow, N*meshCol]` = `[64, 16]` for M=4, N=2
- This cast to int8 becomes mode 1's A matrix
- W₂ shape: `[16, N1*meshCol]`, split by column: `W2_left[:, :8]`, `W2_right[:, 8:]`
- So mode 1 N=2 (each VersaCore handles N1/2=1 column tile)

---

## Bug Triage Playbook

| Symptom | Most Likely Cause | Fix |
|---------|-------------------|-----|
| Simulation hangs, no output | Wrong `AddrSelOffSet` in `snax_csr_mux_demux` — snaxgen.py `+N` is off | Verify `snaxgen.py` writer-busy count, rerun `rtl-gen` |
| All outputs = 0 or garbage | Wrong CSR base address → accelerator never actually starts or receives wrong config | Check `DUAL_VC_CSR_ADDR_BASE` in generated `streamer_csr_addr_map.h` |
| `PASS` but wrong values | Golden model mismatch — check int16 overflow/arithmetic in datagen.py | Add numpy dtype assertions throughout datagen.py |
| Verilator compile error: port mismatch | Shell wrapper port width in Scala-generated SV doesn't match instantiated module | Check `DataWidthOut` computation in DualVersaCoreSwigluGen.scala |
| RTL gen fails: `$$` related Scala error | String interpolation: inside Scala `s"""..."""`, the symbol `$` must be written as `$$` | Fix `$$clog2` etc. in Scala string literals |
| Mode 1 Writer1 never fires | acc2stream_1 valid not driven in mode 1 path | Check mode mux logic in out_assemble1 |
| Block pipeline broken after changes | `numReadOnlyReg` or `snaxgen.py` count mismatch | Recheck Streamer.scala and snaxgen.py together |

---

## Verification Criteria

The final simulation must produce:
```
Dual VersaCore SwiGLU: PASS, Error: 0.
EXIT_CODE: 0
```

covering:
- Mode 0 correctness (SwiGLU result matches golden)
- Mode 1 correctness (both D0 and D1 match golden)
- Full 3-step flow without hangs

---

## Reference Files (read before editing)

| Path | Purpose |
|------|---------|
| `hw/chisel_acc/src/main/resources/snax_acc/versacore/elem_adder_32b.sv` | Joint handshake pattern to copy for `elem_mul_16b.sv` |
| `hw/chisel/src/main/scala/snax/DataPathExtension/RescaleDown.scala` | `RescaleDownPE` logic to implement in `rescale_down_32to16.sv` |
| `hw/chisel_acc/src/main/resources/snax_acc/versacore/shifter_6stage.sv` | Already has `DATA_WIDTH` param; only change instantiation to 16 |
| `hw/chisel_acc/src/main/scala/snax_acc/versacore/DualVersaCoreSwigluGen.scala` | Current full Scala generator (on swiglue branch) |
| `target/snitch_cluster/cfg/snax_dual_versacore_swiglu_cluster.hjson` | Current cluster config (on swiglue branch) |
| `snax_agent_dev_log/BLOCK_PIPELINE_CSR_OPTIMIZATION_RESULTS.md` | snaxgen.py AddrSelOffSet pitfall history |
| `snax_agent_dev_log/DUAL_VERSACORE_SWIGLU_DEVLOG.md` | Prior architecture decisions |
| `skills/snax-cluster-workflow/SKILL.md` | SNAX build/sim workflow reference |
| `snax_agent_dev_log/RESCALE_MODE_INTEGRATION_SPEC.md` | Full technical specification for this task |
