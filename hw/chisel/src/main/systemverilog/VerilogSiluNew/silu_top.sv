module silu_top (
  input  logic                            clk,
  input  logic                            rst_n,
  input  silu_hp32_q22_pkg::input_t       x_in,
  input  logic                            valid_in,
  output silu_hp32_q22_pkg::output_t      y_out,
  output silu_hp32_q22_pkg::seg_idx_t     seg_idx_out,
  output logic                            valid_out
);

  import silu_hp32_q22_pkg::*;

  input_t   x_p0;
  input_t   x_p1;
  seg_idx_t seg_p0;
  seg_idx_t seg_p1;
  seg_idx_t seg_p2;
  logic     valid_p0;

  a0_t      a0_sel;
  a1_t      a1_sel;
  a2_t      a2_sel;
  a0_t      a0_p1;
  stage0_t  t0_p1;
  logic     valid_t0;

  partition_detector u_partition_detector (
    .clk        (clk),
    .rst_n      (rst_n),
    .x_in       (x_in),
    .valid_in   (valid_in),
    .x_out      (x_p0),
    .seg_idx_out(seg_p0),
    .valid_out  (valid_p0)
  );

  param_selector u_param_selector (
    .seg_idx(seg_p0),
    .a0_q   (a0_sel),
    .a1_q   (a1_sel),
    .a2_q   (a2_sel)
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
    .clk      (clk),
    .rst_n    (rst_n),
    .op_a_in  (a2_sel),
    .op_b_in  (x_p0),
    .addend_in(a1_sel),
    .valid_in (valid_p0),
    .out_q    (t0_p1),
    .valid_out(valid_t0)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      x_p1   <= '0;
      a0_p1  <= '0;
      seg_p1 <= '0;
      seg_p2 <= '0;
    end else begin
      x_p1   <= x_p0;
      a0_p1  <= a0_sel;
      seg_p1 <= seg_p0;
      seg_p2 <= seg_p1;
    end
  end

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
    .clk      (clk),
    .rst_n    (rst_n),
    .op_a_in  (t0_p1),
    .op_b_in  (x_p1),
    .addend_in(a0_p1),
    .valid_in (valid_t0),
    .out_q    (y_out),
    .valid_out(valid_out)
  );

  assign seg_idx_out = seg_p2;

endmodule
