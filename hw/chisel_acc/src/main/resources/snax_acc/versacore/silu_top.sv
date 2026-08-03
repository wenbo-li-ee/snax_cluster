module silu_top (
  input  logic                                clk,
  input  logic                                rst_n,
  input  silu_out16_balanced_pkg::input_t     x_in,
  input  logic                                valid_in,
  input  logic                                ce,
  output silu_out16_balanced_pkg::output_t    y_out,
  output silu_out16_balanced_pkg::seg_idx_t   seg_idx_out,
  output logic                                valid_out
);

  import silu_out16_balanced_pkg::*;

  seg_idx_t seg_idx_comb;
  logic     below_fit_range_comb;
  logic     above_fit_range_comb;
  a0_t      a0_sel;
  a1_t      a1_sel;
  a2_t      a2_sel;
  stage0_t  t0_comb;
  output_t  polynomial_comb;
  output_t  y_comb;

  // Partition detection, parameter selection, and both Horner operations are
  // deliberately combinational.  The only datapath register is at the output.
  partition_detector u_partition_detector (
    .x_in               (x_in),
    .seg_idx_out         (seg_idx_comb),
    .below_fit_range_out (below_fit_range_comb),
    .above_fit_range_out (above_fit_range_comb)
  );

  param_selector u_param_selector (
    .seg_idx (seg_idx_comb),
    .a0_q    (a0_sel),
    .a1_q    (a1_sel),
    .a2_q    (a2_sel)
  );

  horner_stage #(
    .OP_A_WIDTH   (A2_WIDTH),
    .OP_A_FRAC    (A2_FRAC),
    .OP_B_WIDTH   (INPUT_WIDTH),
    .OP_B_FRAC    (INPUT_FRAC),
    .ADDEND_WIDTH (A1_WIDTH),
    .ADDEND_FRAC  (A1_FRAC),
    .MUL_WIDTH    (STAGE0_WIDTH),
    .MUL_FRAC     (STAGE0_FRAC),
    .ADD_WIDTH    (STAGE0_WIDTH),
    .ADD_FRAC     (STAGE0_FRAC),
    .OUT_WIDTH    (STAGE0_WIDTH),
    .OUT_FRAC     (STAGE0_FRAC)
  ) u_horner_stage0 (
    .op_a_in   (a2_sel),
    .op_b_in   (x_in),
    .addend_in (a1_sel),
    .out_q     (t0_comb)
  );

  horner_stage #(
    .OP_A_WIDTH   (STAGE0_WIDTH),
    .OP_A_FRAC    (STAGE0_FRAC),
    .OP_B_WIDTH   (INPUT_WIDTH),
    .OP_B_FRAC    (INPUT_FRAC),
    .ADDEND_WIDTH (A0_WIDTH),
    .ADDEND_FRAC  (A0_FRAC),
    .MUL_WIDTH    (STAGE1_MUL_WIDTH),
    .MUL_FRAC     (STAGE1_MUL_FRAC),
    .ADD_WIDTH    (OUTPUT_WIDTH),
    .ADD_FRAC     (OUTPUT_FRAC),
    .OUT_WIDTH    (OUTPUT_WIDTH),
    .OUT_FRAC     (OUTPUT_FRAC)
  ) u_horner_stage1 (
    .op_a_in   (t0_comb),
    .op_b_in   (x_in),
    .addend_in (a0_sel),
    .out_q     (polynomial_comb)
  );

  // The six fitted polynomials cover the closed interval [-8, 6].  Outside
  // that interval, select the intended SiLU asymptote before the output flop.
  always_comb begin
    if (below_fit_range_comb) begin
      y_comb = output_t'(0);
    end else if (above_fit_range_comb) begin
      y_comb = output_t'(x_in);
    end else begin
      y_comb = polynomial_comb;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      y_out       <= '0;
      seg_idx_out <= '0;
      valid_out   <= 1'b0;
    end else if (ce) begin
      y_out       <= y_comb;
      seg_idx_out <= seg_idx_comb;
      valid_out   <= valid_in;
    end
  end

endmodule
