// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Four-lane (parameterizable) signed INT32 -> IEEE-754 FP16 conversion stage.
// The conversion arithmetic is combinational and bit-identical to the
// reference Int32ToFp16PE.  A one-entry elastic output register supplies a
// one-cycle pipeline stage and ready/valid backpressure, matching the
// rescale_down_32to16 interface style.

module int32_to_fp16 #(
    parameter int unsigned NUM_LANES = 4
)(
    input  logic                         clk_i,
    input  logic                         rst_ni,

    input  logic [NUM_LANES-1:0][31:0]   data_i,
    input  logic                         valid_i,
    output logic                         ready_o,

    output logic [NUM_LANES-1:0][15:0]   data_o,
    output logic                         valid_o,
    input  logic                         ready_i
);

    logic [NUM_LANES-1:0][15:0] converted;

    function automatic logic [15:0] convert_lane(
        input logic signed [31:0] value
    );
        logic sign;
        logic [31:0] abs_value;
        logic [31:0] mag_norm;
        logic [9:0]  fraction;
        logic        guard_bit;
        logic        round_bit;
        logic        sticky_bit;
        logic        increment;
        logic [10:0] frac_plus;
        logic        mant_overflow;
        logic [9:0]  frac_rounded;
        logic [5:0]  exp_pre_wide;
        logic [5:0]  exp_rounded;
        int unsigned msb_index;
        int unsigned shift_amt;

        begin
            sign = value[31];
            abs_value = sign ? (~$unsigned(value) + 32'd1) : $unsigned(value);
            convert_lane = {sign, 15'd0};

            if (abs_value != 32'd0) begin
                // Priority encode the highest set bit. Later iterations
                // overwrite earlier ones, leaving floor(log2(abs_value)).
                msb_index = 0;
                for (int i = 0; i < 32; i++) begin
                    if (abs_value[i]) msb_index = i;
                end

                // Normalize the leading one to bit 31. Bit 31 is implicit in
                // FP16; bits 30:21 become the ten stored fraction bits.
                shift_amt = 31 - msb_index;
                mag_norm = abs_value << shift_amt;
                fraction = mag_norm[30:21];
                guard_bit = mag_norm[20];
                round_bit = mag_norm[19];
                sticky_bit = |mag_norm[18:0];

                // IEEE round-to-nearest-even using guard/round/sticky bits.
                increment = guard_bit &&
                            (round_bit || sticky_bit || fraction[0]);
                frac_plus = {1'b0, fraction} + increment;
                mant_overflow = frac_plus[10];
                frac_rounded = mant_overflow ? 10'd0 : frac_plus[9:0];

                // Non-zero INT32 values always map to a normal FP16 value or
                // infinity; no subnormal output case is possible.
                exp_pre_wide = msb_index + 6'd15;
                exp_rounded = exp_pre_wide + mant_overflow;
                if (exp_rounded >= 6'd31)
                    convert_lane = {sign, 5'b11111, 10'd0};
                else
                    convert_lane = {sign, exp_rounded[4:0], frac_rounded};
            end
        end
    endfunction

    always_comb begin
        for (int lane = 0; lane < NUM_LANES; lane++) begin
            converted[lane] = convert_lane($signed(data_i[lane]));
        end
    end

    // One-entry elastic pipeline register. It may pop the old result and push
    // a new result in the same cycle when the downstream is ready.
    assign ready_o = !valid_o || ready_i;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_o <= 1'b0;
        end else if (ready_o) begin
            valid_o <= valid_i;
            if (valid_i) data_o <= converted;
        end
    end

endmodule
