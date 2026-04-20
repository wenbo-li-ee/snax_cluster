# Chained SwiGLU Experiment Results
# 4 apps × 3 array_shapes = 12 experiments, ALL PASS

## Configuration
- Workload: SwiGLU = Mode 0 (gate) → Mode 1 (projection), chained via TCDM
- K=8, N=8, data_type=0 (int16x4)
- Mode 1 A input = Mode 0 D0 output (zero-copy)

## Results

### M=1 Batch
| Shape | meshRow | tileSize | meshCol | K1 | N1 | M0 Status | M0 accel | M0 streamer | M1 Status | M1 accel | M1 streamer |
|-------|---------|----------|---------|----|----|-----------|----------|-------------|-----------|----------|-------------|
| 0     | 8       | 8        | 4       | 4  | 16 | PASS      | 133      | 146         | PASS      | 133      | 138         |
| 1     | 4       | 8        | 8       | 8  | 8  | PASS      | 133      | 146         | PASS      | 133      | 138         |
| 2     | 2       | 8        | 16      | 16 | 4  | PASS      | 132      | 145         | PASS      | 133      | 137         |

### M=1 Pingpong (N_chunk=1, N1_chunk=1)
| Shape | meshRow | tileSize | meshCol | K1 | N1 | M0 Status | M0 accel | M0 streamer | M1 Status | M1 accel | M1 streamer |
|-------|---------|----------|---------|----|----|-----------|----------|-------------|-----------|----------|-------------|
| 0     | 8       | 8        | 4       | 4  | 16 | PASS      | 21       | 34          | PASS      | 12       | 17          |
| 1     | 4       | 8        | 8       | 8  | 8  | PASS      | 21       | 34          | PASS      | 21       | 26          |
| 2     | 2       | 8        | 16      | 16 | 4  | PASS      | 20       | 33          | PASS      | 37       | 41          |

### M=4 Batch
| Shape | meshRow | tileSize | meshCol | K1 | N1 | M0 Status | M0 accel | M0 streamer | M1 Status | M1 accel | M1 streamer |
|-------|---------|----------|---------|----|----|-----------|----------|-------------|-----------|----------|-------------|
| 0     | 8       | 8        | 4       | 4  | 16 | PASS      | 513      | 539         | PASS      | 513      | 531         |
| 1     | 4       | 8        | 8       | 8  | 8  | PASS      | 513      | 539         | PASS      | 513      | 531         |
| 2     | 2       | 8        | 16      | 16 | 4  | PASS      | 517      | 530         | PASS      | 517      | 522         |

### M=4 Pingpong (N_chunk=1, N1_chunk=1)
| Shape | meshRow | tileSize | meshCol | K1 | N1 | M0 Status | M0 accel | M0 streamer | M1 Status | M1 accel | M1 streamer |
|-------|---------|----------|---------|----|----|-----------|----------|-------------|-----------|----------|-------------|
| 0     | 8       | 8        | 4       | 4  | 16 | PASS      | 69       | 82          | PASS      | 36       | 41          |
| 1     | 4       | 8        | 8       | 8  | 8  | PASS      | 69       | 82          | PASS      | 69       | 74          |
| 2     | 2       | 8        | 16      | 16 | 4  | PASS      | 69       | 82          | PASS      | 133      | 138         |

## Notes
- Pingpong cycle counts are for the LAST chunk only (perf counter resets each invocation)
- Batch cycle counts are for the entire computation
- See pingpong_datagen_dataflow.md for detailed mechanism explanation
