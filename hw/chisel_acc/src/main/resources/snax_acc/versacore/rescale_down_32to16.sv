// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// RescaleDown: int32 -> int16, per-lane quantization with registered output
// Matches RescaleDownPE logic from RescaleDown.scala (in=32, out=16)
// Identity params: input_zp=0, multiplier=1, output_zp=0, shift=0

module rescale_down_32to16 #(
    parameter int unsigned NUM_LANES = 64
)(
    input  logic                         clk_i,
    input  logic                         rst_ni,

    // Rescale parameters (shared across all lanes)
    input  logic signed [31:0]           input_zp,
    input  logic        [31:0]           multiplier,
    input  logic signed [31:0]           output_zp,
    input  logic        [7:0]            shift,

    // Input port (int32 per lane)
    input  logic [NUM_LANES-1:0][31:0]   data_i,
    input  logic                         valid_i,
    output logic                         ready_o,

    // Output port (int16 per lane)
    output logic [NUM_LANES-1:0][15:0]   data_o,
    output logic                         valid_o,
    input  logic                         ready_i
);

    // Combinational rescale logic per lane
    logic [NUM_LANES-1:0][15:0] rescaled;

    generate
        for (genvar l = 0; l < NUM_LANES; l++) begin : gen_rescale

            // Step 1: zero_compensated = data_i - input_zp
            logic signed [31:0] zero_compensated;
            assign zero_compensated = $signed(data_i[l]) - input_zp;

            // Step 2: multiplied = zero_compensated * {1'b0, multiplier} (int64)
            logic signed [63:0] multiplied;
            assign multiplied = zero_compensated * $signed({1'b0, multiplier});

            // Step 3: shifted_one = 1 << (shift - 1)
            logic signed [63:0] shifted_one;
            assign shifted_one = 64'sd1 <<< (shift - 8'd1);

            // Step 4: shifted_data = multiplied + shifted_one
            logic signed [63:0] shifted_data;
            assign shifted_data = multiplied + shifted_one;

            // Step 5: scaled_32 = (zero_compensated >= 0) ? shifted_data + (1<<30) : shifted_data - (1<<30)
            logic signed [63:0] scaled_32;
            assign scaled_32 = (zero_compensated >= 0) ?
                               (shifted_data + 64'sd1073741824) :
                               (shifted_data - 64'sd1073741824);

            // Step 6: correct_shift = (shift > 31) ? scaled_32 : shifted_data
            logic signed [63:0] correct_shift;
            assign correct_shift = (shift > 8'd31) ? scaled_32 : shifted_data;

            // Step 7: shifted_value = correct_shift >>> shift (arithmetic right shift)
            logic signed [63:0] shifted_value;
            assign shifted_value = correct_shift >>> shift;

            // Step 8: result = shifted_value[31:0] + output_zp
            logic signed [31:0] result;
            assign result = shifted_value[31:0] + output_zp;

            // Step 9: Clamp to int16 range [-32768, 32767]
            always_comb begin
                if (result > 32'sd32767)
                    rescaled[l] = 16'sd32767;
                else if (result < -32'sd32768)
                    rescaled[l] = -16'sd32768;
                else
                    rescaled[l] = result[15:0];
            end

        end
    endgenerate

    // Registered output with backpressure
    assign ready_o = !valid_o || ready_i;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_o <= 1'b0;
        end else begin
            if (ready_o) begin
                valid_o <= valid_i;
                if (valid_i) begin
                    data_o <= rescaled;
                end
            end
        end
    end

endmodule
