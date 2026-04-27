# Design Notes

## Direct-Read State

- Mode1 A currently reads Mode0 D0 directly from `delta_local_d0`.
- The previous direct-read validation only proves S0 behavior; S1/S2 still require diagnosis.

## Mode0 D1 Semantics

- Do not assume Mode0 D1 equals D0.
- Determine Mode0 D1 meaning from `datagen.py`, C writer configuration, and dual-VersaCore SwiGLU source/RTL before adding a correctness check.
- Source check: in Mode0, `DualVersaCoreSwigluGen.scala` assigns `oa0_in_data` and `oa1_in_data` from `rescale_mul_out_data`, with both valids driven by `rescale_mul_out_valid`. Thus this shell integration intentionally duplicates the Mode0 postprocessed output stream to both writer outputs. The final check should still compare D1 explicitly rather than relying on Mode1 side effects.

## S1 Static Layout

- S1 has half the Mode0 output footprint of S0 because `meshRow=4`, `meshCol=4`, and `beats_per_tile=4`.
- Mode0 D0/D1 writer banks are staggered (`0` and `34`), so the prior same-bank D0/D1 hazard is not present in the emitted base addresses.

## Runtime Chunk Count Fix

- The generated shell previously serialized `NumChunks = (DataWidthD / 32 + PostprocLanes - 1) / PostprocLanes`, which is 8 for the fixed 1024-bit VC output.
- S1 only has `meshRow * meshCol / PostprocLanes = 4` real postproc chunks per tile, and S2 only has 2.
- Datagen writer bounds were shape-aware, but the shell kept emitting inactive zero chunks after the real chunks. That filled the writer quota and then backpressured the accelerator before `wait_dual_versacore()` could complete.
- Fix direction: the shell serializer should stop at the active chunk count selected by `array_shape`.
- The active chunk count must use one more representable value than the chunk counter range: when `NumChunks=8`, S0's count is exactly `8`, so the SV helper uses `$clog2(NumChunks + 1)` bits.

## Route B Cleanup

- Deferred `delta_local_a1` cleanup. The final fix touched the generated shell contract and required a full S0/S1/S2 validation sweep; removing A1 now would shift W2 and Mode1 D buffer addresses across all shapes and would require another full layout and simulation pass.
