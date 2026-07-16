# Full-size multishape V144 layout app

This app runs the workload recorded in
`snax_agent_dev_log/multishape_k8_4lane_mode1_contiguous_l15_20260507`:

```text
M=8, K0=2048, N0=1408, K1=1408, N1=1024
```

It emits one full-workload optimized V144-family layout,
`v270_full_A4_V144_S1in144_pitch576_bank56`, and is meant for:

```text
cfg/snax_dual_versacore_int16x4_multidim_spatial_k8_8x4_4lane_a4_b2_layout_v144.hjson
```

The L3 input and final Mode1 output remain per-token closed-loop rows.  Their
shape-specific row strides are S0/S1/S2 = 4128/4096/4176 bytes.  Mode0 input A
is packed into shape-specific compute panels.  S1 uses a 144-byte token stride,
a 576-byte K pitch, and TCDM base bank 56.  S2 keeps a 176-byte token stride in
both its Mode0 input and Mode0-D/Mode1-A intermediate panel.

The C execution and checking harness is shared with the high-granularity
layout-search app; this app owns its workload parameters and single-layout
generated `data.h`.
