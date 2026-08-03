package silu_out16_balanced_pkg;

  localparam int NUM_SEGMENTS = 6;
  localparam int NUM_BREAKPOINTS = 7;
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
    -16'sd11230,
    -16'sd4126,
    -16'sd1945,
     16'sd1932,
     16'sd4061,
     16'sd12288
  };

  localparam a0_t SEG_A0_Q[NUM_SEGMENTS] = '{
    -30'sd877179,
    -30'sd2166905,
    -30'sd518345,
     30'sd13417,
    -30'sd501092,
    -30'sd2146849
  };

  localparam a1_t SEG_A1_Q[NUM_SEGMENTS] = '{
    -22'sd3415,
    -22'sd10937,
     22'sd14823,
     22'sd32771,
     22'sd50290,
     22'sd76289
  };

  localparam a2_t SEG_A2_Q[NUM_SEGMENTS] = '{
    -18'sd54,
    -18'sd229,
     18'sd1374,
     18'sd3718,
     18'sd1412,
    -18'sd223
  };

  function automatic seg_idx_t segment_index_from_x(input input_t x_q);
    begin
      if (x_q < BREAKPOINTS_Q[3]) begin
        if (x_q < BREAKPOINTS_Q[2]) begin
          segment_index_from_x = (x_q < BREAKPOINTS_Q[1]) ? seg_idx_t'(0) : seg_idx_t'(1);
        end else begin
          segment_index_from_x = seg_idx_t'(2);
        end
      end else begin
        if (x_q < BREAKPOINTS_Q[5]) begin
          segment_index_from_x = (x_q < BREAKPOINTS_Q[4]) ? seg_idx_t'(3) : seg_idx_t'(4);
        end else begin
          segment_index_from_x = seg_idx_t'(5);
        end
      end
    end
  endfunction

endpackage
