// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Small post-processing block placed after VersaCore D2S.
// Each element is shifted right by 1 bit per stage for 2 stages total.

module VersaCorePostD2SShiftRightTwo #(
    parameter int unsigned DataWidth = 32,
    parameter int unsigned ElementWidth = 32
) (
    input  logic                 clk_i,
    input  logic                 rst_ni,
    input  logic [DataWidth-1:0] in_data_i,
    input  logic                 in_valid_i,
    output logic                 in_ready_o,
    output logic [DataWidth-1:0] out_data_o,
    output logic                 out_valid_o,
    input  logic                 out_ready_i
);

  localparam int unsigned NrElements = DataWidth / ElementWidth;

  logic [DataWidth-1:0] stage1_data_q;
  logic [DataWidth-1:0] stage2_data_q;
  logic                 stage1_valid_q;
  logic                 stage2_valid_q;
  logic                 stage1_ready;
  logic                 stage2_ready;

  function automatic [DataWidth-1:0] shift_elements_right_one(
      input logic [DataWidth-1:0] data_i
  );
    logic signed [ElementWidth-1:0] element;
    for (int idx = 0; idx < NrElements; idx++) begin
      element = data_i[idx * ElementWidth +: ElementWidth];
      shift_elements_right_one[idx * ElementWidth +: ElementWidth] = element >>> 1;
    end
  endfunction

  assign stage2_ready = !stage2_valid_q || out_ready_i;
  assign stage1_ready = !stage1_valid_q || stage2_ready;
  assign in_ready_o   = stage1_ready;

  assign out_data_o  = stage2_data_q;
  assign out_valid_o = stage2_valid_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      stage1_data_q  <= '0;
      stage2_data_q  <= '0;
      stage1_valid_q <= 1'b0;
      stage2_valid_q <= 1'b0;
    end else begin
      if (stage2_ready) begin
        stage2_valid_q <= stage1_valid_q;
        if (stage1_valid_q) begin
          stage2_data_q <= shift_elements_right_one(stage1_data_q);
        end
      end

      if (stage1_ready) begin
        stage1_valid_q <= in_valid_i;
        if (in_valid_i) begin
          stage1_data_q <= shift_elements_right_one(in_data_i);
        end
      end
    end
  end

endmodule
