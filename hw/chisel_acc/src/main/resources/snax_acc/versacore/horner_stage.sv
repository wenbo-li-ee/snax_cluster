module horner_stage #(
  parameter int OP_A_WIDTH   = 8,
  parameter int OP_A_FRAC    = 4,
  parameter int OP_B_WIDTH   = 8,
  parameter int OP_B_FRAC    = 4,
  parameter int ADDEND_WIDTH = 8,
  parameter int ADDEND_FRAC  = 4,
  parameter int MUL_WIDTH    = 16,
  parameter int MUL_FRAC     = 8,
  parameter int ADD_WIDTH    = 16,
  parameter int ADD_FRAC     = 8,
  parameter int OUT_WIDTH    = 16,
  parameter int OUT_FRAC     = 8
) (
  input  logic                           clk,
  input  logic                           rst_n,
  input  logic                           ce,
  input  logic signed [OP_A_WIDTH-1:0]   op_a_in,
  input  logic signed [OP_B_WIDTH-1:0]   op_b_in,
  input  logic signed [ADDEND_WIDTH-1:0] addend_in,
  input  logic                           valid_in,
  output logic signed [OUT_WIDTH-1:0]    out_q,
  output logic                           valid_out
);

  import silu_out16_balanced_pkg::*;

  localparam int MUL_FULL_WIDTH   = OP_A_WIDTH + OP_B_WIDTH;
  localparam int MUL_SHIFT        = OP_A_FRAC + OP_B_FRAC - MUL_FRAC;
  localparam int MUL_ALIGN_WIDTH  = MUL_FULL_WIDTH + ((MUL_SHIFT < 0) ? (-MUL_SHIFT) : 0);

  localparam int MUL_TO_ADD_SHIFT   = MUL_FRAC - ADD_FRAC;
  localparam int MUL_TO_ADD_WIDTH   = MUL_WIDTH + ((MUL_TO_ADD_SHIFT < 0) ? (-MUL_TO_ADD_SHIFT) : 0);
  localparam int ADDEND_SHIFT       = ADDEND_FRAC - ADD_FRAC;
  localparam int ADDEND_ALIGN_WIDTH = ADDEND_WIDTH + ((ADDEND_SHIFT < 0) ? (-ADDEND_SHIFT) : 0);
  localparam int ADD_OPERAND_WIDTH  = (MUL_TO_ADD_WIDTH > ADDEND_ALIGN_WIDTH) ? MUL_TO_ADD_WIDTH : ADDEND_ALIGN_WIDTH;
  localparam int SUM_WIDTH          = ADD_OPERAND_WIDTH + 1;

  localparam int OUT_SHIFT       = ADD_FRAC - OUT_FRAC;
  localparam int OUT_ALIGN_WIDTH = ADD_WIDTH + ((OUT_SHIFT < 0) ? (-OUT_SHIFT) : 0);

  logic signed [OP_A_WIDTH-1:0]        op_a_int;
  logic signed [OP_B_WIDTH-1:0]        op_b_int;
  logic signed [ADDEND_WIDTH-1:0]      addend_int;
  logic signed [MUL_FULL_WIDTH-1:0]    raw_product_int;
  logic signed [MUL_ALIGN_WIDTH-1:0]   mul_requant_int;
  logic signed [MUL_WIDTH-1:0]         mul_sat_int;
  logic signed [ADD_OPERAND_WIDTH-1:0] mul_for_add_int;
  logic signed [ADD_OPERAND_WIDTH-1:0] addend_for_add_int;
  logic signed [SUM_WIDTH-1:0]         raw_sum_int;
  logic signed [ADD_WIDTH-1:0]         add_sat_int;
  logic signed [OUT_ALIGN_WIDTH-1:0]   out_aligned_int;
  logic signed [OUT_WIDTH-1:0]         out_sat_int;
  logic signed [OUT_WIDTH-1:0] out_next;

  function automatic logic signed [MUL_ALIGN_WIDTH-1:0] align_mul_product(
    input logic signed [MUL_FULL_WIDTH-1:0] value
  );
    logic signed [MUL_ALIGN_WIDTH-1:0] value_ext;
    begin
      value_ext = {{(MUL_ALIGN_WIDTH-MUL_FULL_WIDTH){value[MUL_FULL_WIDTH-1]}}, value};
      if (MUL_SHIFT >= 0) begin
        align_mul_product = value_ext >>> MUL_SHIFT;
      end else begin
        align_mul_product = value_ext <<< (-MUL_SHIFT);
      end
    end
  endfunction

  function automatic logic signed [ADD_OPERAND_WIDTH-1:0] align_mul_for_add(
    input logic signed [MUL_WIDTH-1:0] value
  );
    logic signed [ADD_OPERAND_WIDTH-1:0] value_ext;
    begin
      value_ext = {{(ADD_OPERAND_WIDTH-MUL_WIDTH){value[MUL_WIDTH-1]}}, value};
      if (MUL_TO_ADD_SHIFT >= 0) begin
        align_mul_for_add = value_ext >>> MUL_TO_ADD_SHIFT;
      end else begin
        align_mul_for_add = value_ext <<< (-MUL_TO_ADD_SHIFT);
      end
    end
  endfunction

  function automatic logic signed [ADD_OPERAND_WIDTH-1:0] align_addend_for_add(
    input logic signed [ADDEND_WIDTH-1:0] value
  );
    logic signed [ADD_OPERAND_WIDTH-1:0] value_ext;
    begin
      value_ext = {{(ADD_OPERAND_WIDTH-ADDEND_WIDTH){value[ADDEND_WIDTH-1]}}, value};
      if (ADDEND_SHIFT >= 0) begin
        align_addend_for_add = value_ext >>> ADDEND_SHIFT;
      end else begin
        align_addend_for_add = value_ext <<< (-ADDEND_SHIFT);
      end
    end
  endfunction

  function automatic logic signed [OUT_ALIGN_WIDTH-1:0] align_output_frac(
    input logic signed [ADD_WIDTH-1:0] value
  );
    logic signed [OUT_ALIGN_WIDTH-1:0] value_ext;
    begin
      value_ext = {{(OUT_ALIGN_WIDTH-ADD_WIDTH){value[ADD_WIDTH-1]}}, value};
      if (OUT_SHIFT >= 0) begin
        align_output_frac = value_ext >>> OUT_SHIFT;
      end else begin
        align_output_frac = value_ext <<< (-OUT_SHIFT);
      end
    end
  endfunction

  function automatic logic signed [MUL_WIDTH-1:0] saturate_mul(
    input logic signed [MUL_ALIGN_WIDTH-1:0] value
  );
    logic signed [MUL_WIDTH-1:0] max_val;
    logic signed [MUL_WIDTH-1:0] min_val;
    logic signed [MUL_ALIGN_WIDTH-1:0] max_ext;
    logic signed [MUL_ALIGN_WIDTH-1:0] min_ext;
    begin
      max_val = {1'b0, {(MUL_WIDTH-1){1'b1}}};
      min_val = {1'b1, {(MUL_WIDTH-1){1'b0}}};
      max_ext = {{(MUL_ALIGN_WIDTH-MUL_WIDTH){max_val[MUL_WIDTH-1]}}, max_val};
      min_ext = {{(MUL_ALIGN_WIDTH-MUL_WIDTH){min_val[MUL_WIDTH-1]}}, min_val};
      if (value > max_ext) begin
        saturate_mul = max_val;
      end else if (value < min_ext) begin
        saturate_mul = min_val;
      end else begin
        saturate_mul = value[MUL_WIDTH-1:0];
      end
    end
  endfunction

  function automatic logic signed [ADD_WIDTH-1:0] saturate_add(
    input logic signed [SUM_WIDTH-1:0] value
  );
    logic signed [ADD_WIDTH-1:0] max_val;
    logic signed [ADD_WIDTH-1:0] min_val;
    logic signed [SUM_WIDTH-1:0] max_ext;
    logic signed [SUM_WIDTH-1:0] min_ext;
    begin
      max_val = {1'b0, {(ADD_WIDTH-1){1'b1}}};
      min_val = {1'b1, {(ADD_WIDTH-1){1'b0}}};
      max_ext = {{(SUM_WIDTH-ADD_WIDTH){max_val[ADD_WIDTH-1]}}, max_val};
      min_ext = {{(SUM_WIDTH-ADD_WIDTH){min_val[ADD_WIDTH-1]}}, min_val};
      if (value > max_ext) begin
        saturate_add = max_val;
      end else if (value < min_ext) begin
        saturate_add = min_val;
      end else begin
        saturate_add = value[ADD_WIDTH-1:0];
      end
    end
  endfunction

  function automatic logic signed [OUT_WIDTH-1:0] saturate_out(
    input logic signed [OUT_ALIGN_WIDTH-1:0] value
  );
    logic signed [OUT_WIDTH-1:0] max_val;
    logic signed [OUT_WIDTH-1:0] min_val;
    logic signed [OUT_ALIGN_WIDTH-1:0] max_ext;
    logic signed [OUT_ALIGN_WIDTH-1:0] min_ext;
    begin
      max_val = {1'b0, {(OUT_WIDTH-1){1'b1}}};
      min_val = {1'b1, {(OUT_WIDTH-1){1'b0}}};
      max_ext = {{(OUT_ALIGN_WIDTH-OUT_WIDTH){max_val[OUT_WIDTH-1]}}, max_val};
      min_ext = {{(OUT_ALIGN_WIDTH-OUT_WIDTH){min_val[OUT_WIDTH-1]}}, min_val};
      if (value > max_ext) begin
        saturate_out = max_val;
      end else if (value < min_ext) begin
        saturate_out = min_val;
      end else begin
        saturate_out = value[OUT_WIDTH-1:0];
      end
    end
  endfunction

  always_comb begin
    op_a_int           = op_a_in;
    op_b_int           = op_b_in;
    addend_int         = addend_in;
    raw_product_int    = op_a_int * op_b_int;
    mul_requant_int    = align_mul_product(raw_product_int);
    mul_sat_int        = saturate_mul(mul_requant_int);
    mul_for_add_int    = align_mul_for_add(mul_sat_int);
    addend_for_add_int = align_addend_for_add(addend_int);
    raw_sum_int        = $signed({mul_for_add_int[ADD_OPERAND_WIDTH-1], mul_for_add_int}) +
                         $signed({addend_for_add_int[ADD_OPERAND_WIDTH-1], addend_for_add_int});
    add_sat_int        = saturate_add(raw_sum_int);
    out_aligned_int    = align_output_frac(add_sat_int);
    out_sat_int        = saturate_out(out_aligned_int);
    out_next           = out_sat_int;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_q     <= '0;
      valid_out <= 1'b0;
    end else if (ce) begin
      out_q     <= out_next;
      valid_out <= valid_in;
    end
  end

endmodule
