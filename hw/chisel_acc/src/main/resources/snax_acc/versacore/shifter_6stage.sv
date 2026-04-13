// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// 6-stage pipelined arithmetic right-shifter
// Stages 1-2: arithmetic right shift by 1 bit each (total >>> 2)
// Stages 3-6: pure register pipeline (latency padding)
// Full valid/ready handshake with backpressure per stage

module shifter_6stage #(
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

    // Pipeline registers for 6 stages
    logic [NUM_LANES-1:0][DATA_WIDTH-1:0] stage_data [6];
    logic                                 stage_valid [6];

    // Ready signals propagate backwards
    logic stage_ready [6];

    // Stage 0 input interface
    assign ready_o = !stage_valid[0] || stage_ready[0];

    // Stage 5 output interface
    assign data_o  = stage_data[5];
    assign valid_o = stage_valid[5];
    assign stage_ready[5] = ready_i;

    // Ready for stages 0..4: can accept if next stage empty or next stage can move
    genvar s;
    generate
        for (s = 0; s < 5; s++) begin : gen_ready
            assign stage_ready[s] = !stage_valid[s+1] || stage_ready[s+1];
        end
    endgenerate

    // Compute shifted data for stages 0 and 1
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
            for (int i = 0; i < 6; i++) begin
                stage_valid[i] <= 1'b0;
            end
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

            // Stages 2-5: pure register pipeline (pass through)
            for (int i = 2; i < 6; i++) begin
                if (stage_ready[i-1]) begin
                    stage_valid[i] <= stage_valid[i-1];
                    if (stage_valid[i-1]) begin
                        stage_data[i] <= stage_data[i-1];
                    end
                end
            end
        end
    end

endmodule
