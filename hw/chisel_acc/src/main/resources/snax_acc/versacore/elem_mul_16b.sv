// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Element-wise 16-bit multiplier with NUM_LANES outputs
// Two input ports (int16) with independent 1-deep input buffering
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
    output logic                         busy_o,
    input  logic                         ready_i
);

    logic [NUM_LANES-1:0][15:0] fifo0_data, fifo1_data;
    logic fifo0_valid, fifo1_valid;
    logic out_can_accept;
    logic fire;

    assign out_can_accept = !valid_o || ready_i;
    assign fire          = fifo0_valid && fifo1_valid && out_can_accept;
    assign busy_o        = fifo0_valid || fifo1_valid || valid_o;

    // Each input can be accepted independently. If this stage fires, the buffer
    // can also be refilled in the same cycle.
    assign ready_o_0 = !fifo0_valid || fire;
    assign ready_o_1 = !fifo1_valid || fire;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            fifo0_valid <= 1'b0;
            fifo1_valid <= 1'b0;
        end else begin
            if (ready_o_0) begin
                fifo0_valid <= valid_i_0;
                if (valid_i_0) begin
                    fifo0_data <= data_i_0;
                end
            end
            if (ready_o_1) begin
                fifo1_valid <= valid_i_1;
                if (valid_i_1) begin
                    fifo1_data <= data_i_1;
                end
            end
        end
    end

    // Combinational multiplication (signed int16 x signed int16 -> signed int32)
    logic [NUM_LANES-1:0][31:0] product;
    generate
        for (genvar l = 0; l < NUM_LANES; l++) begin : gen_mul
            assign product[l] = $signed(fifo0_data[l]) * $signed(fifo1_data[l]);
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
