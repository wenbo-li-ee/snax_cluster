# SiLU 8-Polynomial Q16.11 Candidate Reference

This directory is the handoff point for the candidate SiLU configuration with:

- eight quadratic polynomial segments over `[-8, 6]`
- left bypass `x < -8 -> y = 0`
- right bypass `x > 6 -> y = x`
- ten total regions

Status: **reference only; not integrated into the current six-segment RTL**.

The float breakpoint optimization improves the float-domain MSE, but direct
quantization to the current narrow Horner datapath is not yet better than the
existing six-polynomial configuration. Run hardware-aware re-optimization before
selecting this candidate as the final RTL implementation.

The canonical machine-readable values are in [params.json](params.json).

## Source and optimization setup

The candidate was generated from:

```text
/users/students/r1015498/thesis/silu/silu_fit/test_outputs_v2_8poly
```

using the same settings as `test_outputs_v2`, except that the total partition
count was changed from 8 to 10:

```text
interval                  = [-8, 6]
total_partitions          = 10
polynomial_segments       = 8
degree                    = 2
overlap alpha             = 0.25
optimization grid         = 8192
fit samples per segment   = 256
maximum iterations        = 50
learning rate             = 0.05
finite-difference step    = 0.001
minimum iterations        = 5
early-stop patience       = 8
```

The run stopped after iteration 24 with optimizer loss:

```text
2.8166116572520225e-06
```

## Datapath format

```text
input / breakpoint / output = signed 16/11
a0                          = signed 30/22
a1                          = signed 22/16
a2                          = signed 18/14
Horner stage 0 output       = signed 14/12
Horner stage 1 multiply     = signed 15/11
rounding                    = arithmetic-shift truncation
overflow                    = saturation
```

The fixed shifts remain:

```text
Horner 0 product: >>> 13
Horner 0 a1:      >>> 4
Horner 1 product: >>> 12
Horner 1 a0:      >>> 11
```

## Breakpoints

| Index | Float breakpoint | Q16.11 code | Quantized real value |
|---:|---:|---:|---:|
| 0 | -8.0000000000 | -16384 | -8.0000000000 |
| 1 | -6.2416102478 | -12782 | -6.2412109375 |
| 2 | -4.5017608155 | -9219 | -4.5014648438 |
| 3 | -2.0045867117 | -4105 | -2.0043945312 |
| 4 | -0.9529750780 | -1951 | -0.9526367188 |
| 5 | 0.9463931251 | 1938 | 0.9462890625 |
| 6 | 1.9923392396 | 4080 | 1.9921875000 |
| 7 | 4.2211661553 | 8644 | 4.2207031250 |
| 8 | 6.0000000000 | 12288 | 6.0000000000 |

Internal breakpoint equality selects the segment beginning at that breakpoint.
The endpoint values `-8` and `6` remain in the polynomial range; bypass uses
strict comparisons.

## Coefficients

Coefficient order is ascending powers of `x`:

```text
y = a0 + a1*x + a2*x^2
```

| Segment | Nominal range | Float `[a0, a1, a2]` | Quantized `[a0_q, a1_q, a2_q]` |
|---:|---|---|---|
| 0 | `[-8, -6.2416102478)` | `[-0.1635877270, -0.0390864971, -0.0023751750]` | `[-686136, -2561, -38]` |
| 1 | `[-6.2416102478, -4.5017608155)` | `[-0.3669971787, -0.1059916859, -0.0078728669]` | `[-1539297, -6946, -128]` |
| 2 | `[-4.5017608155, -2.0045867117)` | `[-0.5127903586, -0.1642100782, -0.0135625465]` | `[-2150798, -10761, -222]` |
| 3 | `[-2.0045867117, -0.9529750780)` | `[-0.1230785600, 0.2272075214, 0.0843183108]` | `[-516228, 14890, 1381]` |
| 4 | `[-0.9529750780, 0.9463931251)` | `[0.0032263914, 0.5000543380, 0.2268826564]` | `[13532, 32771, 3717]` |
| 5 | `[0.9463931251, 1.9923392396)` | `[-0.1207884272, 0.7694023972, 0.0854985118]` | `[-506623, 50423, 1400]` |
| 6 | `[1.9923392396, 4.2211661553)` | `[-0.5049593144, 1.1586231064, -0.0126179201]` | `[-2117952, 75931, -206]` |
| 7 | `[4.2211661553, 6]` | `[-0.4223134774, 1.1277508669, -0.0099948476]` | `[-1771311, 73908, -163]` |

## Current evaluation

Float-domain dense evaluation over `[-8, 6]`:

```text
MSE             = 2.816636434039502e-06
MAE             = 1.2193001901888332e-03
maximum error   = 3.750569058497133e-03
```

Direct quantization to the current Q16.11 balanced datapath:

```text
MSE             = 7.376766058255559e-06
MAE             = 2.248463300250590e-03
maximum error   = 6.383613627418027e-03
overflow count  = 0
saturation count = 0
```

The current six-polynomial configuration remains the better fixed-point result
under this datapath. This eight-polynomial candidate should therefore be used as
the starting point for an eight-segment hardware-aware optimization, not as a
drop-in replacement package.

