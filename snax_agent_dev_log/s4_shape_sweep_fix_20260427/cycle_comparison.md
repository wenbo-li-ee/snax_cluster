# Cycle Comparison

| Shape | Stage | Result | Cycles | Notes |
| --- | --- | --- | --- | --- |
| S0 | Initial direct-read baseline | PASS | M0 accel=133 streamer=157 wall=2859; M1 accel=251 streamer=271 wall=2718 | Before touching S1/S2. |
| S1 | Direct-read long diagnostic before fix | TIMEOUT | none printed | 600s timeout, no app output. |
| S1 | After active chunk-count fix | PASS | M0 accel=133 streamer=149 wall=2888; M1 accel=254 streamer=266 wall=2748 | Mode0 D0/D1 and Mode1 D0/D1 all PASS. |
| S2 | After active chunk-count fix | PASS | M0 accel=133 streamer=145 wall=2888; M1 accel=256 streamer=264 wall=2752 | Mode0 D0/D1 and Mode1 D0/D1 all PASS. |
| S0 | First final run after active chunk-count fix | TIMEOUT | none printed | Regression from active chunk count width; fixed by widening return type. |
| S0 | Final post-width-fix sweep | PASS | M0 accel=133 streamer=157 wall=2896; M1 accel=251 streamer=271 wall=2731 | Mode0 D0/D1 and Mode1 D0/D1 all PASS. |
| S1 | Final post-width-fix sweep | PASS | M0 accel=133 streamer=149 wall=2888; M1 accel=254 streamer=266 wall=2748 | Mode0 D0/D1 and Mode1 D0/D1 all PASS. |
| S2 | Final post-width-fix sweep | PASS | M0 accel=133 streamer=145 wall=2888; M1 accel=256 streamer=264 wall=2752 | Mode0 D0/D1 and Mode1 D0/D1 all PASS. |
| S0 | Final restored state | PASS | M0 accel=133 streamer=157 wall=2896; M1 accel=251 streamer=271 wall=2731 | `params.hjson` restored to shape 0 and rebuilt. |

## Final Layout Facts

| Shape | meshRow/tileSize/meshCol | K1/N1 | beats/tile | Mode0 D bounds/strides | Mode1 A bounds/strides | Buffer start banks |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | 8/8/4 | 4/16 | 8 | `{8,8,1,1}` / `{8,64,512,0}` | `{4,16,1}` / `{128,0,512}` | A 0, B0 0, B1 0, D0 0, D1m0 2, A1 4, W2L 4, W2R 4, M1D0 4, M1D1 6 |
| S1 | 4/16/4 | 2/32 | 4 | `{4,8,1,1}` / `{8,32,256,0}` | `{2,32,1}` / `{128,0,256}` | A 0, B0 0, B1 0, D0 0, D1m0 34, A1 4, W2L 36, W2R 36, M1D0 36, M1D1 38 |
| S2 | 2/32/4 | 1/64 | 2 | `{2,8,1,1}` / `{8,16,128,0}` | `{1,64,1}` / `{128,0,128}` | A 0, B0 0, B1 0, D0 0, D1m0 18, A1 36, W2L 52, W2R 52, M1D0 52, M1D1 54 |
