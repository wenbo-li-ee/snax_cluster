# Review: S4 Shape Sweep Fix And Direct-Read Validation

**Review Date**: 2026-04-27  
**Dev Log Path**: `snax_agent_dev_log/s4_shape_sweep_fix_20260427/`  
**Todo Reference**: `review_log/s4_direct_mode1_review_20260427/todo_s4_shape_sweep_fix.md`

---

## 1. Summary

**Excellent execution.** The agent completed all core objectives: found and fixed a real hardware bug in the Chisel generator (chunk serializer count mismatch), added Mode0 D1 correctness checking with semantics confirmed from source, completed the full three-shape validation sweep, and self-detected and fixed a width regression introduced during the fix. The only remaining item is Route B (`delta_local_a1` cleanup), which was explicitly and correctly deferred.

---

## 2. What Was Accomplished

### Phase 0 - S0 Baseline
- Confirmed S0 direct-read still passes at task start.
- Recorded cycles: M0 `wall=2859`, M1 `wall=2718`.

### Phase 1 - S1/S2 Root Cause Diagnosis And Fix

Diagnosis escalation followed the plan exactly:

1. **Static `data.h` inspection**: no buffer overlap, no zero bounds, bank assignments correct — no datagen error visible.
2. **600s timeout run**: still hangs — not a too-short timeout.
3. **Temporary app progress prints**: localized hang to `wait_dual_versacore()` — accelerator busy never clears, before the writer drain wait.
4. **Temporary generated RTL probes**: VC output and postproc output fired. After 32 accepted writer beats, `acc_ready=00`, one output buffer remained valid with no drain possible.

**Root cause**: The generated shell (`DualVersaCoreSwigluGen.scala`) serialized all 8 chunks from the fixed 1024-bit VersaCore DataWidthD output for every tile. S1 only has 4 active chunks per tile (`meshRow*meshCol/PostprocLanes = 4`). The datagen writer bounds were correctly shape-aware (4 beats), but the shell kept emitting 4 extra inactive zero chunks after the real data. These filled the writer's beat quota, creating backpressure that blocked VC output and permanently stalled `wait_dual_versacore()`.

**Fix**: Modified `DualVersaCoreSwigluGen.scala` to stop chunk serialization at `active_num_chunks(array_shape)` (`S0=8`, `S1=4`, `S2=2`) instead of always using the fixed `NumChunks=8`. Same root cause applies to S2 (2 active chunks). After fix, `rtl-gen` regenerated the shell, temporary probes and debug prints were removed.

### Phase 2 - Mode0 D1 Correctness Check

- **Semantics confirmed from source first**: `DualVersaCoreSwigluGen.scala` drives both `oa0_in_data` and `oa1_in_data` from the same `rescale_mul_out_data` stream in Mode0 — D1 is intentionally a duplicate of D0 in this shell integration. This is a verified fact, not an assumption.
- `datagen.py` now emits `mode0_d1_golden_padded`.
- C app compares `local_d1_mode0` against golden after Mode0 completes and before Mode1 starts, printing separate Mode0 D0/D1 pass/fail lines.
- Mode0 D1 PASS for all three shapes.

### Phase 3 - S0 Restore And Final Three-Shape Sweep

- After restoring `params.hjson` to `array_shape=0`, **S0 regressed** (timeout before first print). Root cause: `active_num_chunks` return width was `$clog2(NumChunks)` bits — for `NumChunks=8` this is 3 bits, which cannot represent 8. S0's active count of 8 wrapped to 0, causing the chunk serializer to emit nothing and stall.
- **Agent self-detected and fixed**: widened return type to `$clog2(NumChunks + 1)` bits. Re-ran `rtl-gen` and rebuilt `bin/snitch_cluster.vlt`.
- Final three-shape sweep after width fix:

| Shape | Mode0 D0/D1 | Mode1 D0/D1 | M0 wall | M1 wall |
|-------|-------------|-------------|---------|---------|
| S0 | PASS/PASS | PASS/PASS | 2896 | 2731 |
| S1 | PASS/PASS | PASS/PASS | 2888 | 2748 |
| S2 | PASS/PASS | PASS/PASS | 2888 | 2752 |

- `params.hjson` restored to `array_shape=0`, final S0 re-simulated: PASS.
- Skill `versacore-snax-fusion-design` updated with the active chunk-count rule.

---

## 3. What Is Not Done / Deferred

### Only Remaining: Route B (`delta_local_a1` cleanup)

`delta_local_a1` is still allocated in `datagen.py` as unused compatibility padding. The agent explicitly deferred this with a sound justification: removing it shifts all downstream buffer addresses (W2L, W2R, Mode1 D0, Mode1 D1) across all three shapes, requiring a full bank assignment recomputation and a new three-shape simulation sweep. Given that the active chunk-count fix already required a full sweep, adding another full sweep in the same session was not warranted.

This deferral is correct. Route B is safe to address as a standalone follow-up task.

---

## 4. Spec Deliverables Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| S0 direct-read passes at task start | ✅ | wall=2859, matches prior session |
| S1 timeout root cause classified and fixed | ✅ | Shell chunk serializer count bug in Chisel generator |
| S2 timeout root cause classified and fixed | ✅ | Same root cause as S1 |
| S1 and S2 pass after fix | ✅ | All four output channels PASS |
| Mode0 D1 check added with confirmed semantics | ✅ | Source verified before golden written |
| Final three-shape sweep completed | ✅ | S0/S1/S2 all PASS |
| Cycle counts recorded | ✅ | `cycle_comparison.md` complete with all runs |
| `params.hjson` restored to array_shape=0 | ✅ | Final S0 re-verified after restore |
| Route B `delta_local_a1` cleanup | ⏸️ Deferred | Justified; standalone follow-up task |
| Skill updated | ✅ | `versacore-snax-fusion-design` updated |

---

## 5. Engineering Quality Notes

- **Diagnosis escalation was textbook**: static inspection → longer timeout → app prints → RTL probes. No steps skipped, no premature RTL modifications.
- **RTL modification was justified**: SW/datagen paths exhausted before touching Chisel. The fix was minimal and targeted.
- **Self-detected regression**: the `$clog2(NumChunks)` width bug was introduced and caught by the agent itself during the S0 restore step — no silent regressions shipped.
- **D1 semantics confirmed before coding**: the previous review flagged this risk; the agent handled it correctly.
- **Debug artifacts cleaned up**: temporary RTL probes and app progress prints were removed before final validation runs.
- **Skill updated with reusable rule**: the fixed chunk-count width contract is now documented for future VersaCore shell generators.

---

## 6. Recommended Next Steps (Optional)

The S4 task series is now functionally complete. Optional follow-up items in priority order:

1. **Route B cleanup** (low priority, standalone): remove `delta_local_a1` datagen allocation, recompute W2/Mode1 D buffer addresses and bank assignments for all three shapes, validate S0/S1/S2 still pass.
2. **Cross-workload audit**: verify the active chunk-count fix does not affect other shells that might have the same fixed-serializer pattern. Check whether any non-SwiGLU VersaCore shell uses `DualVersaCoreSwigluGen.scala` patterns.
3. **M > 1 dimension test**: all three shapes currently use `M=1`. If future workloads require `M>1`, the Mode0 D / Mode1 A layout contract should be re-validated.
