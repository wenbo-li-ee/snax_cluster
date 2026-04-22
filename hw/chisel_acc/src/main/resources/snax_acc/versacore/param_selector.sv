module param_selector (
  input  silu_out16_balanced_pkg::seg_idx_t seg_idx,
  output silu_out16_balanced_pkg::a0_t      a0_q,
  output silu_out16_balanced_pkg::a1_t      a1_q,
  output silu_out16_balanced_pkg::a2_t      a2_q
);

  import silu_out16_balanced_pkg::*;

  always_comb begin
    unique case (seg_idx)
      seg_idx_t'(0): begin
        a0_q = SEG_A0_Q[0];
        a1_q = SEG_A1_Q[0];
        a2_q = SEG_A2_Q[0];
      end
      seg_idx_t'(1): begin
        a0_q = SEG_A0_Q[1];
        a1_q = SEG_A1_Q[1];
        a2_q = SEG_A2_Q[1];
      end
      seg_idx_t'(2): begin
        a0_q = SEG_A0_Q[2];
        a1_q = SEG_A1_Q[2];
        a2_q = SEG_A2_Q[2];
      end
      seg_idx_t'(3): begin
        a0_q = SEG_A0_Q[3];
        a1_q = SEG_A1_Q[3];
        a2_q = SEG_A2_Q[3];
      end
      seg_idx_t'(4): begin
        a0_q = SEG_A0_Q[4];
        a1_q = SEG_A1_Q[4];
        a2_q = SEG_A2_Q[4];
      end
      default: begin
        a0_q = SEG_A0_Q[5];
        a1_q = SEG_A1_Q[5];
        a2_q = SEG_A2_Q[5];
      end
    endcase
  end

endmodule
