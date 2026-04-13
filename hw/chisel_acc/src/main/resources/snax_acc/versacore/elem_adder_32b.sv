// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Element-wise 32-bit adder with NUM_LANES outputs
// Two input ports (path 0 + path 1) with joint valid/ready handshake
// 1-cycle registered output

module elem_adder_32b #(
    parameter int unsigned NUM_LANES = 64
)(
    input  logic                         clk_i,
    input  logic                         rst_ni,

    // Path 0 input (from shifter 0)
    input  logic [NUM_LANES-1:0][31:0]   data_i_0,
    input  logic                         valid_i_0,
    output logic                         ready_o_0,

    // Path 1 input (from shifter 1)
    input  logic [NUM_LANES-1:0][31:0]   data_i_1,
    input  logic                         valid_i_1,
    output logic                         ready_o_1,

    // Output port
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

    // Combinational addition
    logic [NUM_LANES-1:0][31:0] sum;
    generate
        for (genvar l = 0; l < NUM_LANES; l++) begin : gen_add
            assign sum[l] = data_i_0[l] + data_i_1[l];
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
                    data_o <= sum;
                end
            end
        end
    end

endmodule
