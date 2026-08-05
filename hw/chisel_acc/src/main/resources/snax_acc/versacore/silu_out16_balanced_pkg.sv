package silu_out16_balanced_pkg;

  localparam int NUM_SEGMENTS = 8;
  localparam int NUM_BREAKPOINTS = 9;
  localparam int SEG_IDX_WIDTH = 3;
  localparam int PIPELINE_LATENCY = 1;

  localparam int INPUT_WIDTH = 16;
  localparam int INPUT_FRAC  = 11;
  localparam int BREAKPOINT_WIDTH = 16;
  localparam int BREAKPOINT_FRAC  = 11;

  localparam int OUTPUT_WIDTH = 16;
  localparam int OUTPUT_FRAC  = 11;

  localparam int A0_WIDTH = 30;
  localparam int A0_FRAC  = 22;
  localparam int A1_WIDTH = 22;
  localparam int A1_FRAC  = 16;
  localparam int A2_WIDTH = 18;
  localparam int A2_FRAC  = 14;

  localparam int STAGE0_WIDTH = 14;
  localparam int STAGE0_FRAC  = 12;
  localparam int STAGE1_MUL_WIDTH = 15;
  localparam int STAGE1_MUL_FRAC  = 11;

  typedef logic signed [INPUT_WIDTH-1:0] input_t;
  typedef logic signed [BREAKPOINT_WIDTH-1:0] breakpoint_t;
  typedef logic [SEG_IDX_WIDTH-1:0] seg_idx_t;
  typedef logic signed [OUTPUT_WIDTH-1:0] output_t;
  typedef logic signed [A0_WIDTH-1:0] a0_t;
  typedef logic signed [A1_WIDTH-1:0] a1_t;
  typedef logic signed [A2_WIDTH-1:0] a2_t;
  typedef logic signed [STAGE0_WIDTH-1:0] stage0_t;

  localparam breakpoint_t BREAKPOINTS_Q[NUM_BREAKPOINTS] = '{
    -16'sd16384,
    -16'sd12782,
    -16'sd9219,
    -16'sd4105,
    -16'sd1951,
     16'sd1938,
     16'sd4080,
     16'sd8644,
     16'sd12288
  };

  localparam a0_t SEG_A0_Q[NUM_SEGMENTS] = '{
    -30'sd686136,
    -30'sd1539297,
    -30'sd2150798,
    -30'sd516228,
     30'sd13532,
    -30'sd506623,
    -30'sd2117952,
    -30'sd1771311
  };

  localparam a1_t SEG_A1_Q[NUM_SEGMENTS] = '{
    -22'sd2561,
    -22'sd6946,
    -22'sd10761,
     22'sd14890,
     22'sd32771,
     22'sd50423,
     22'sd75931,
     22'sd73908
  };

  localparam a2_t SEG_A2_Q[NUM_SEGMENTS] = '{
    -18'sd38,
    -18'sd128,
    -18'sd222,
     18'sd1381,
     18'sd3717,
     18'sd1400,
    -18'sd206,
    -18'sd163
  };

  function automatic seg_idx_t segment_index_from_x(input input_t x_q);
    begin
      // Three-level balanced decision tree for the seven internal boundaries.
      // Equality selects the segment that starts at the boundary.
      if (x_q < BREAKPOINTS_Q[4]) begin
        if (x_q < BREAKPOINTS_Q[2]) begin
          segment_index_from_x = (x_q < BREAKPOINTS_Q[1]) ? seg_idx_t'(0) : seg_idx_t'(1);
        end else begin
          segment_index_from_x = (x_q < BREAKPOINTS_Q[3]) ? seg_idx_t'(2) : seg_idx_t'(3);
        end
      end else begin
        if (x_q < BREAKPOINTS_Q[6]) begin
          segment_index_from_x = (x_q < BREAKPOINTS_Q[5]) ? seg_idx_t'(4) : seg_idx_t'(5);
        end else begin
          segment_index_from_x = (x_q < BREAKPOINTS_Q[7]) ? seg_idx_t'(6) : seg_idx_t'(7);
        end
      end
    end
  endfunction

endpackage
