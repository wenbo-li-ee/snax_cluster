// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Multi-lane SiLU wrapper with valid/ready backpressure.
// Implements silu_out16_balanced (Q16.11 piecewise polynomial) over NUM_LANES lanes.
// Pipeline depth: 3 stages, matching silu_top (partition_detector + horner_stage0 + horner_stage1).
// Clock-enable backpressure: each stage has a CE signal derived from stage_ready.
// Interface is identical to shifter_6stage (without DATA_WIDTH parameter).

module silu_multilane #(
    parameter int unsigned NUM_LANES = 64
)(
    input  logic                             clk_i,
    input  logic                             rst_ni,

    // Input port
    input  logic [NUM_LANES-1:0][15:0]       data_i,
    input  logic                             valid_i,
    output logic                             ready_o,

    // Output port
    output logic [NUM_LANES-1:0][15:0]       data_o,
    output logic                             valid_o,
    input  logic                             ready_i
);

    // -----------------------------------------------------------------------
    // 3-stage backpressure (matches silu_top pipeline depth)
    // stage_valid[0] = stage 0 (partition_detector)
    // stage_valid[1] = stage 1 (horner_stage0 + intermediate regs)
    // stage_valid[2] = stage 2 (horner_stage1)
    // -----------------------------------------------------------------------
    logic stage_valid [3];
    logic stage_ready [3];

    // Output interface
    assign ready_o        = !stage_valid[0] || stage_ready[0];
    assign valid_o        = stage_valid[2];
    assign stage_ready[2] = ready_i;

    // Ready propagation (backward)
    assign stage_ready[1] = !stage_valid[2] || stage_ready[2];
    assign stage_ready[0] = !stage_valid[1] || stage_ready[1];

    // CE derivation: stage can advance when it can drain
    logic ce [3];
    assign ce[0] = ready_o;         // stage 0 can accept when input fires
    assign ce[1] = stage_ready[0];  // stage 1 can advance when stage 0 can drain
    assign ce[2] = stage_ready[1];  // stage 2 can advance when stage 1 can drain

    // Stage valid tracking
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            stage_valid[0] <= 1'b0;
            stage_valid[1] <= 1'b0;
            stage_valid[2] <= 1'b0;
        end else begin
            if (ce[0]) stage_valid[0] <= valid_i;
            if (ce[1]) stage_valid[1] <= stage_valid[0];
            if (ce[2]) stage_valid[2] <= stage_valid[1];
        end
    end

    // -----------------------------------------------------------------------
    // NUM_LANES silu_top instances
    // valid_in only fires when the input is actually accepted (ready_o is asserted)
    // data_o is taken from silu_top output (stage 2 result)
    // -----------------------------------------------------------------------
    logic [NUM_LANES-1:0][15:0] silu_y_out;

    genvar l;
    generate
        for (l = 0; l < NUM_LANES; l++) begin : gen_silu_lanes
            silu_top u_silu (
                .clk        (clk_i),
                .rst_n      (rst_ni),
                .x_in       (data_i[l]),
                .valid_in   (valid_i && ready_o),
                .ce0        (ce[0]),
                .ce1        (ce[1]),
                .ce2        (ce[2]),
                .y_out      (silu_y_out[l]),
                .seg_idx_out(),
                .valid_out  ()
            );
        end
    endgenerate

    assign data_o = silu_y_out;

endmodule
