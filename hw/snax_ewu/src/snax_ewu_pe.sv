// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

//-------------------------------
// Element-wise unit processing element
//-------------------------------
module snax_ewu_pe #(
  parameter int unsigned DataWidth       = 32,
  parameter int unsigned AddOutDataWidth = 32,
  parameter int unsigned MulOutDataWidth = 64
)(
  input  logic                                clk_i,
  input  logic                                rst_ni,
  input  logic [DataWidth-1:0]                a_i,
  input  logic                                a_valid_i,
  output logic                                a_ready_o,
  input  logic [DataWidth-1:0]                b_i,
  input  logic                                b_valid_i,
  output logic                                b_ready_o,
  output logic [((AddOutDataWidth > MulOutDataWidth) ? AddOutDataWidth : MulOutDataWidth)-1:0] c_o,
  output logic                                c_valid_o,
  input  logic                                c_ready_i,
  input  logic                                acc_ready_i,
  input  logic [1:0]                          op_config_i
);

  //-------------------------------
  // Local parameters
  //-------------------------------
  localparam int unsigned AddOp        = 0;
  localparam int unsigned MulOp        = 1;
  localparam int unsigned OutDataWidth = (AddOutDataWidth > MulOutDataWidth) ? AddOutDataWidth :
      MulOutDataWidth;

  //-------------------------------
  // Wires and combinational logic
  //-------------------------------
  logic signed [AddOutDataWidth-1:0] add_result;
  logic signed [MulOutDataWidth-1:0] mul_result;
  logic signed [OutDataWidth-1:0]    result_wide;

  logic input_success;

  assign input_success = (a_valid_i && a_ready_o) && (b_valid_i && b_ready_o);

  always_comb begin
    add_result = $signed(a_i) + $signed(b_i);
    mul_result = $signed(a_i) * $signed(b_i);

    case (op_config_i)
      MulOp: result_wide = $signed(mul_result);
      default: result_wide = $signed(add_result);
    endcase
  end

  //-------------------------------
  // Assignments
  //-------------------------------
  assign a_ready_o = acc_ready_i && c_ready_i && (a_valid_i && b_valid_i);
  assign b_ready_o = acc_ready_i && c_ready_i && (a_valid_i && b_valid_i);
  assign c_valid_o = input_success;
  assign c_o       = result_wide;

endmodule
