// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Multi-lane SiLU wrapper with a single elastic output register stage.
// All partition and Horner arithmetic is combinational before that register.

module silu_multilane #(
    parameter int unsigned NUM_LANES = 64
)(
    input  logic                             clk_i,
    input  logic                             rst_ni,

    input  logic [NUM_LANES-1:0][15:0]       data_i,
    input  logic                             valid_i,
    output logic                             ready_o,

    output logic [NUM_LANES-1:0][15:0]       data_o,
    output logic                             valid_o,
    output logic                             busy_o,
    input  logic                             ready_i
);

    // A valid output can be replaced in the same cycle in which it is consumed.
    // This retains one-result-per-cycle throughput with one cycle of latency.
    assign ready_o = !valid_o || ready_i;
    assign busy_o  = valid_o;

    logic [NUM_LANES-1:0][15:0] silu_y_out;
    logic [NUM_LANES-1:0]       silu_valid_out;

    genvar l;
    generate
        for (l = 0; l < NUM_LANES; l++) begin : gen_silu_lanes
            silu_top u_silu (
                .clk         (clk_i),
                .rst_n       (rst_ni),
                .x_in        (data_i[l]),
                .valid_in    (valid_i),
                .ce          (ready_o),
                .y_out       (silu_y_out[l]),
                .seg_idx_out (),
                .valid_out   (silu_valid_out[l])
            );
        end
    endgenerate

    // Every lane receives the same CE and valid, so their valid bits are equal.
    assign valid_o = silu_valid_out[0];
    assign data_o  = silu_y_out;

endmodule
