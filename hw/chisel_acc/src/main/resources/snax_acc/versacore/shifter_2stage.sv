// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// 2-stage pipelined arithmetic right-shifter
// Stage 1: arithmetic right shift by 1 bit
// Stage 2: arithmetic right shift by 1 bit (total >>> 2)
// Full valid/ready handshake with backpressure per stage

module shifter_2stage #(
    parameter int unsigned DATA_WIDTH = 32,
    parameter int unsigned NUM_LANES  = 64
)(
    input  logic                                    clk_i,
    input  logic                                    rst_ni,

    // Input port
    input  logic [NUM_LANES-1:0][DATA_WIDTH-1:0]    data_i,
    input  logic                                    valid_i,
    output logic                                    ready_o,

    // Output port
    output logic [NUM_LANES-1:0][DATA_WIDTH-1:0]    data_o,
    output logic                                    valid_o,
    input  logic                                    ready_i
);

    // Pipeline registers for 2 stages
    logic [NUM_LANES-1:0][DATA_WIDTH-1:0] stage_data [2];
    logic                                 stage_valid [2];

    // Ready signals
    logic stage_ready [2];

    // Input interface
    assign ready_o = !stage_valid[0] || stage_ready[0];

    // Output interface
    assign data_o  = stage_data[1];
    assign valid_o = stage_valid[1];
    assign stage_ready[1] = ready_i;

    // Stage 0 ready: can accept if stage 1 empty or stage 1 can move
    assign stage_ready[0] = !stage_valid[1] || stage_ready[1];

    // Compute shifted data
    logic [NUM_LANES-1:0][DATA_WIDTH-1:0] shift_in_0;
    logic [NUM_LANES-1:0][DATA_WIDTH-1:0] shift_in_1;

    // Stage 0: arithmetic right shift by 1
    generate
        for (genvar l = 0; l < NUM_LANES; l++) begin : gen_shift0
            assign shift_in_0[l] = $signed(data_i[l]) >>> 1;
        end
    endgenerate

    // Stage 1: arithmetic right shift by 1 (on stage 0 output)
    generate
        for (genvar l = 0; l < NUM_LANES; l++) begin : gen_shift1
            assign shift_in_1[l] = $signed(stage_data[0][l]) >>> 1;
        end
    endgenerate

    // Pipeline stage logic
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            stage_valid[0] <= 1'b0;
            stage_valid[1] <= 1'b0;
        end else begin
            // Stage 0: shift right 1
            if (ready_o) begin
                stage_valid[0] <= valid_i;
                if (valid_i) begin
                    stage_data[0] <= shift_in_0;
                end
            end

            // Stage 1: shift right 1 (total shift = 2)
            if (stage_ready[0]) begin
                stage_valid[1] <= stage_valid[0];
                if (stage_valid[0]) begin
                    stage_data[1] <= shift_in_1;
                end
            end
        end
    end

endmodule
