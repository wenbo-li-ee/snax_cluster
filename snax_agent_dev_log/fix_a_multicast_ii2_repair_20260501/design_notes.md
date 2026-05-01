# design_notes.md

Created: 2026-05-01

## Starting Point

- The old shared-A wrapper uses one registered slot plus per-VC sent bits. It can accept a streamer A beat only when the slot is empty, so even perfect dual consumption leaves an accept/consume alternation and a structural II=2.
- A cannot be naively broadcast to two independent ready/valid consumers. The shared-A contract requires one logical A beat to be consumed by both VersaCores exactly once at the same K step.
- The repair remains on the shell/generator side. It does not alter `Array.scala`, `Accumulator.scala`, or `VersaCore.scala`.

## Chosen Repair Direction

- Initial FIFO and elastic-A-only attempts removed the structural A refill bubble but exposed a shell-level output overrun (`2817` writer beats for S0 Mode0, where the streamer is programmed for `2816`).
- Quota/drain attempts proved invalid because they either discarded real compute work or shifted output data.
- Grouped A/B issue and an explicit Mode0 `rescale1` delay were also falsified: they still corrupted S0 Mode0 output beginning at index 172.
- Current direction is an elastic version of the original A replay slot, plus VC-local B pairing. A advances only when both VCs have consumed the current shared beat; each B stream advances only when its matching VC has A valid and both the VC A/B ready signals are high.
- This keeps the shared-A correctness rule local to the wrapper, avoids a naive independent A broadcast, avoids a cross-VC B serialization policy, and does not modify VersaCore internals.

## Final Design

### A/B input side

- The wrapper keeps the original single shared-A replay slot and per-VC sent bits.
- The structural II=2 bubble is removed by allowing same-cycle refill:
  - `stream2acc_0_ready_o = !a_buf_valid || a_buf_done`
  - when both VCs have accepted the old A beat and a new A beat is valid, the slot loads the new A beat immediately.
- A is only presented to a VC when that VC's matching B stream is valid.
- Each B stream handshakes only when the matching A and B sides of that same VC are both ready. This prevents B from advancing ahead of the shared A step without serializing VC0 and VC1 against each other.

### Output side

- The faster input path exposed two output issues in Mode0:
  - the postprocess path can produce one extra finite-burst beat beyond the streamer writer quota;
  - when `OutChunks == 1`, a direct duplicated output can be accepted by one writer before the other unless the shell tracks per-writer delivery.
- The final wrapper derives `output_quota_beats = output_times * active_num_chunks(array_shape)`.
- In Mode0 after both output quotas close, the shell drains `rescale_mul` locally so the accelerator can complete without asking inactive/full writers for another beat.
- For the S6 `OutChunks == 1` case, Mode0 uses a small direct-output holding register with per-writer sent bits:
  - one `rescale_mul` payload is captured;
  - writer0 and writer1 each see that payload until they accept it exactly once;
  - the register can refill in the same cycle the previous duplicated beat is delivered to both writers.
- Mode1 keeps independent direct output behavior.

### Boundaries

- No internal VersaCore handshake files were modified.
- The repair is localized to the dual-VersaCore SwiGLU shell generator plus the SiLU wrapper valid cleanup.
- The app and cfg were not forked; the existing S6 integration target was used end-to-end.
