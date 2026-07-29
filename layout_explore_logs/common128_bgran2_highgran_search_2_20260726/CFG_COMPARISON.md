# XDMA 128-column overlap test with bgran2/highgran cfg

Date: 2026-07-26

## Scope

- App: `snax-versacore-int16x4-shape1-common16-weight-overlap`
- Shape: shape 1, `(meshRow, tileSize, meshCol) = (4, 8, 8)`
- Mode: mode 0 only
- Tokens: 4
- Weight chunk: 128 columns, 8 loop iterations
- Weight DMA path: interleaved W/V in L3, iDMA staging plus local XDMA reshape
- CSR policy: accelerator configured once; streamer and DMA invariant CSRs
  configured once; only the next iteration's addresses are updated in the loop

The only requested experiment change was the hardware cfg:

`snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane_bgran2_highgran_search_2.hjson`

The comparison baseline used:

`snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane.hjson`

## Relevant cfg differences

The new cfg is a bundle of changes, so this run cannot attribute the result to
one option in isolation:

- `dma_axi_req_fifo_depth`: 32 -> 64
- `dma_axi_to_tcdm_buf_depth`: absent -> 3
- `dma_xbar_fall_through`: absent -> true
- `wide_trans`: 32 -> 64
- `wide_xbar_latency`: `CUT_ALL_AX` -> `CUT_ALL_PORTS`
- `wide_xbar_fall_through: true` is removed
- `register_ext_wide`: false -> true
- B0/B1 sparse access granularity: 1 -> 2
- XDMA `has_gather_scatter: true` is removed, so the generated hardware uses
  Cartesian-stride mode only and has a shorter CSR map

## Build compatibility fixes

Two compatibility fixes were required to generate and build this cfg:

1. The XDMA deserializer now drives the zero-width compatibility
   `addressMode` field with zero when gather/scatter is disabled. Without this,
   Chisel/FIRRTL reports an uninitialized sink.
2. The workload data generator accepts both B granularity 1 and 2. The data
   layout itself is unchanged; B0/B1 bases and spatial lanes preserve the bank
   parity required by granularity 2.

After RTL generation, the VLT model was rebuilt from scratch. The generated
`snax-xdma-addr.h` changed because gather/scatter was disabled, so
`snax-xdma-lib.o` also had to be rebuilt before relinking the app.

An initial run accidentally linked the stale XDMA object compiled for the old
CSR map. It produced 3562 errors and an apparently faster loop. That run is
invalid and was replaced by the correct PASS log. This is important because the
old object still lets XDMA complete, but writes several configuration values to
the wrong CSR offsets.

## Valid simulation result

The final simulation passes with zero errors.

| Metric (cycles) | Old cfg | New cfg | Delta |
|---|---:|---:|---:|
| prologue | 14674 | 14985 | +311 |
| first DMA | 10424 | 10710 | +286 |
| measured for-loop | 41261 | 41289 | +28 |
| accelerator sum | 39605 | 39613 | +8 |
| streamer sum | 39797 | 39805 | +8 |
| compute-path / ideal overlap | 40379 | 40403 | +24 |
| DMA loop sum | 29439 | 29547 | +108 |
| streamer preconfiguration | 818 | 804 | -14 |
| prepared DMA start | 130 | 138 | +8 |
| next-address preparation | 990 | 1755 | +765 |
| DMA wait | 28226 | 27550 | -676 |
| barrier wait | 40 | 40 | 0 |
| loop sync/control overhead | 882 | 886 | +4 |

## Interpretation

The new cfg improves the accumulated DMA wait by 676 cycles, or about 2.40%.
However, the measured next-iteration address-preparation region grows by 765
cycles. Therefore the complete DMA-side loop time increases by 108 cycles, and
the end-to-end for-loop changes only from 41261 to 41289 cycles (+0.07%).

The compute/streamer critical path is essentially unchanged. The final loop is
still compute-bound at about 40403 cycles, and its synchronization/control
overhead remains about 886 cycles. Thus this cfg does not improve the valid
end-to-end 128-column result, although its lower DMA wait confirms a small DMA
transport improvement.

## Raw logs

- New cfg: `run_xdma_chunk128_addr_only_csr.log`
- Old cfg baseline:
  `../common_granularity_sweep_20260723/run_xdma_chunk128_addr_only_csr.log`
