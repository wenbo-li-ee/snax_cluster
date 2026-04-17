// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Element-wise 16-bit multiplier with NUM_LANES outputs
// Two input ports (int16) with joint valid/ready handshake
// Output: int32 (int16 x int16 -> int32)
// 1-cycle registered output
// Based on elem_adder_32b.sv structure

module elem_mul_16b #(
    parameter int unsigned NUM_LANES = 64
)(
    input  logic                         clk_i,
    input  logic                         rst_ni,

    // Path 0 input (int16 per lane)
    input  logic [NUM_LANES-1:0][15:0]   data_i_0,
    input  logic                         valid_i_0,
    output logic                         ready_o_0,

    // Path 1 input (int16 per lane)
    input  logic [NUM_LANES-1:0][15:0]   data_i_1,
    input  logic                         valid_i_1,
    output logic                         ready_o_1,

    // Output port (int32 per lane)
    output logic [NUM_LANES-1:0][31:0]   data_o,
    output logic                         valid_o,
    input  logic                         ready_i
);

    // Joint handshake: fire only when both inputs valid and output can accept
    logic both_valid;
    logic out_can_accept;
    logic fire;

    assign both_valid    = valid_i_0 && valid_i_1;
    assign out_can_accept = !valid_o || ready_i;
    assign fire          = both_valid && out_can_accept;

    // Backpressure: each input ready only when both paths valid and output can accept
    assign ready_o_0 = valid_i_1 && out_can_accept;
    assign ready_o_1 = valid_i_0 && out_can_accept;

    // Combinational multiplication (signed int16 x signed int16 -> signed int32)
    logic [NUM_LANES-1:0][31:0] product;
    generate
        for (genvar l = 0; l < NUM_LANES; l++) begin : gen_mul
            assign product[l] = $signed(data_i_0[l]) * $signed(data_i_1[l]);
        end
    endgenerate

    // Registered output
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_o <= 1'b0;
        end else begin
            if (out_can_accept) begin
                valid_o <= fire;
                if (fire) begin
                    data_o <= product;
                end
            end
        end
    end

endmodule
