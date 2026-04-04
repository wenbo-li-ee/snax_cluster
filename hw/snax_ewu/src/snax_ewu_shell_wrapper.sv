// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

//-------------------------------
// Accelerator wrapper
//-------------------------------
module snax_ewu_shell_wrapper #(
  parameter int unsigned RegRWCount       = 3,
  parameter int unsigned RegROCount       = 2,
  parameter int unsigned NumPE            = 4,
  parameter int unsigned DataWidth        = 32,
  parameter int unsigned AddOutDataWidth  = 32,
  parameter int unsigned MulOutDataWidth  = 64,
  parameter int unsigned RegDataWidth     = 32,
  parameter int unsigned RegAddrWidth     = 32
)(
  //-------------------------------
  // Clocks and reset
  //-------------------------------
  input  logic clk_i,
  input  logic rst_ni,

  //-------------------------------
  // Accelerator ports
  //-------------------------------
  output logic [((((NumPE*((AddOutDataWidth > MulOutDataWidth) ? AddOutDataWidth : MulOutDataWidth)) + 63) / 64) * 64)-1:0]
      acc2stream_0_data_o,
  output logic acc2stream_0_valid_o,
  input  logic acc2stream_0_ready_i,

  input  logic [((((NumPE*DataWidth) + 63) / 64) * 64)-1:0] stream2acc_0_data_i,
  input  logic stream2acc_0_valid_i,
  output logic stream2acc_0_ready_o,

  input  logic [((((NumPE*DataWidth) + 63) / 64) * 64)-1:0] stream2acc_1_data_i,
  input  logic stream2acc_1_valid_i,
  output logic stream2acc_1_ready_o,

  //-------------------------------
  // CSR manager ports
  //-------------------------------
  input  logic [RegRWCount-1:0][RegDataWidth-1:0] csr_reg_set_i,
  input  logic                                    csr_reg_set_valid_i,
  output logic                                    csr_reg_set_ready_o,
  output logic [RegROCount-1:0][RegDataWidth-1:0] csr_reg_ro_set_o
);

  localparam int unsigned OutDataWidth = (AddOutDataWidth > MulOutDataWidth) ? AddOutDataWidth :
      MulOutDataWidth;
  localparam int unsigned StreamInDataWidth  = (((NumPE * DataWidth) + 63) / 64) * 64;
  localparam int unsigned StreamOutDataWidth = (((NumPE * OutDataWidth) + 63) / 64) * 64;

  logic [NumPE-1:0][DataWidth-1:0]    a_split;
  logic [NumPE-1:0][DataWidth-1:0]    b_split;
  logic [NumPE-1:0][OutDataWidth-1:0] c_split;

  logic [NumPE-1:0] a_ready;
  logic [NumPE-1:0] b_ready;
  logic [NumPE-1:0] result_valid;

  logic       acc_output_success;
  logic       acc_ready;
  logic [1:0] csr_op_config;

  always_comb begin
    acc2stream_0_data_o = '0;

    for (int i = 0; i < NumPE; i++) begin
      a_split[i] = stream2acc_0_data_i[i*DataWidth+:DataWidth];
      b_split[i] = stream2acc_1_data_i[i*DataWidth+:DataWidth];
      acc2stream_0_data_o[i*OutDataWidth+:OutDataWidth] = c_split[i];
    end

    stream2acc_0_ready_o = &a_ready;
    stream2acc_1_ready_o = &b_ready;
    acc2stream_0_valid_o = &result_valid;
  end

  assign acc_output_success = acc2stream_0_valid_o && acc2stream_0_ready_i;

  snax_ewu_csr #(
    .RegRWCount   ( RegRWCount   ),
    .RegROCount   ( RegROCount   ),
    .RegDataWidth ( RegDataWidth )
  ) i_snax_ewu_csr (
    .clk_i                ( clk_i               ),
    .rst_ni               ( rst_ni              ),
    .csr_reg_set_i        ( csr_reg_set_i       ),
    .csr_reg_set_valid_i  ( csr_reg_set_valid_i ),
    .csr_reg_set_ready_o  ( csr_reg_set_ready_o ),
    .csr_reg_ro_set_o     ( csr_reg_ro_set_o    ),
    .acc_output_success_i ( acc_output_success  ),
    .acc_ready_o          ( acc_ready           ),
    .csr_op_config_o      ( csr_op_config       )
  );

  for (genvar i = 0; i < NumPE; i++) begin : gen_ewu_pes
    snax_ewu_pe #(
      .DataWidth       ( DataWidth       ),
      .AddOutDataWidth ( AddOutDataWidth ),
      .MulOutDataWidth ( MulOutDataWidth )
    ) i_snax_ewu_pe (
      .clk_i       ( clk_i                ),
      .rst_ni      ( rst_ni               ),
      .a_i         ( a_split[i]           ),
      .a_valid_i   ( stream2acc_0_valid_i ),
      .a_ready_o   ( a_ready[i]           ),
      .b_i         ( b_split[i]           ),
      .b_valid_i   ( stream2acc_1_valid_i ),
      .b_ready_o   ( b_ready[i]           ),
      .c_o         ( c_split[i]           ),
      .c_valid_o   ( result_valid[i]      ),
      .c_ready_i   ( acc2stream_0_ready_i ),
      .acc_ready_i ( acc_ready            ),
      .op_config_i ( csr_op_config        )
    );
  end

endmodule
