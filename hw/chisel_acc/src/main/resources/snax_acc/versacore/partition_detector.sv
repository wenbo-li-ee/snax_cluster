module partition_detector (
  input  silu_out16_balanced_pkg::input_t       x_in,
  output silu_out16_balanced_pkg::seg_idx_t     seg_idx_out,
  output logic                            below_fit_range_out,
  output logic                            above_fit_range_out
);

  import silu_out16_balanced_pkg::*;

  always_comb begin
    // Eight total partitions: two bypass regions surrounding the six fitted
    // polynomial segments.  The boundary points themselves remain in the
    // fitted range, so only strict comparisons select a bypass partition.
    below_fit_range_out = (x_in < BREAKPOINTS_Q[0]);
    above_fit_range_out = (x_in > BREAKPOINTS_Q[NUM_BREAKPOINTS-1]);
    seg_idx_out          = segment_index_from_x(x_in);
  end

endmodule
