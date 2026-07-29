// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// FP16 x INT4 Dual VersaCore SwiGLU shell.
//
// Mode 0:
//   VC0 FP32 -> FP32 SiLU -> FP32 elementwise multiply <- VC1 FP32
//            -> FP32-to-FP16 -> Writer 0
// Mode 1:
//   VC0 FP32 -> FP32-to-FP16 -> Writer 0
//   VC1 FP32 -> FP32-to-FP16 -> Writer 1
//
// Floating-point addition, multiplication, and narrowing are implemented with
// the repository's fp_add and fp_mul primitives.  The narrowing extension uses
// FP32 * 1.0 with an FP16 output format, so rounding is performed by fp_mul.

module silu_fp32_multilane #(
    parameter int unsigned NUM_LANES = 4
) (
    input  logic                              clk_i,
    input  logic                              rst_ni,
    input  logic [NUM_LANES-1:0][31:0]        data_i,
    input  logic                              valid_i,
    output logic                              ready_o,
    output logic [NUM_LANES-1:0][31:0]        data_o,
    output logic                              valid_o,
    output logic                              busy_o,
    input  logic                              ready_i
);

    // The values below are the original Q-format breakpoints and coefficients
    // converted exactly to IEEE-754 FP32.
    localparam logic [31:0] BP1 = 32'hc0af7800; // -5.4833984375
    localparam logic [31:0] BP2 = 32'hc000f000; // -2.0146484375
    localparam logic [31:0] BP3 = 32'hbf732000; // -0.94970703125
    localparam logic [31:0] BP4 = 32'h3f718000; //  0.943359375
    localparam logic [31:0] BP5 = 32'h3ffdd000; //  1.98291015625

    function automatic logic fp32_lt(
        input logic [31:0] lhs,
        input logic [31:0] rhs
    );
        logic lhs_zero;
        logic rhs_zero;
        begin
            lhs_zero = (lhs[30:0] == 31'b0);
            rhs_zero = (rhs[30:0] == 31'b0);
            if (lhs_zero && rhs_zero)
                fp32_lt = 1'b0;
            else if (lhs[31] != rhs[31])
                fp32_lt = lhs[31];
            else if (!lhs[31])
                fp32_lt = lhs[30:0] < rhs[30:0];
            else
                fp32_lt = lhs[30:0] > rhs[30:0];
        end
    endfunction

    function automatic logic [2:0] segment_index(input logic [31:0] x);
        begin
            if (fp32_lt(x, BP3)) begin
                if (fp32_lt(x, BP2))
                    segment_index = fp32_lt(x, BP1) ? 3'd0 : 3'd1;
                else
                    segment_index = 3'd2;
            end else begin
                if (fp32_lt(x, BP5))
                    segment_index = fp32_lt(x, BP4) ? 3'd3 : 3'd4;
                else
                    segment_index = 3'd5;
            end
        end
    endfunction

    function automatic logic [31:0] coeff_a0(input logic [2:0] seg);
        begin
            case (seg)
                3'd0: coeff_a0 = 32'hbe5627b0;
                3'd1: coeff_a0 = 32'hbf0441e4;
                3'd2: coeff_a0 = 32'hbdfd1920;
                3'd3: coeff_a0 = 32'h3b51a400;
                3'd4: coeff_a0 = 32'hbdf4ac80;
                default: coeff_a0 = 32'hbf030884;
            endcase
        end
    endfunction

    function automatic logic [31:0] coeff_a1(input logic [2:0] seg);
        begin
            case (seg)
                3'd0: coeff_a1 = 32'hbd557000;
                3'd1: coeff_a1 = 32'hbe2ae400;
                3'd2: coeff_a1 = 32'h3e679c00;
                3'd3: coeff_a1 = 32'h3f000300;
                3'd4: coeff_a1 = 32'h3f447200;
                default: coeff_a1 = 32'h3f950080;
            endcase
        end
    endfunction

    function automatic logic [31:0] coeff_a2(input logic [2:0] seg);
        begin
            case (seg)
                3'd0: coeff_a2 = 32'hbb580000;
                3'd1: coeff_a2 = 32'hbc650000;
                3'd2: coeff_a2 = 32'h3dabc000;
                3'd3: coeff_a2 = 32'h3e686000;
                3'd4: coeff_a2 = 32'h3db08000;
                default: coeff_a2 = 32'hbc5f0000;
            endcase
        end
    endfunction

    logic stage0_valid;
    logic stage1_valid;
    logic stage2_valid;
    logic stage0_ready;
    logic stage1_ready;
    logic stage2_ready;

    logic [NUM_LANES-1:0][31:0] x_q0;
    logic [NUM_LANES-1:0][31:0] a0_q0;
    logic [NUM_LANES-1:0][31:0] a1_q0;
    logic [NUM_LANES-1:0][31:0] a2_q0;
    logic [NUM_LANES-1:0][31:0] x_q1;
    logic [NUM_LANES-1:0][31:0] a0_q1;
    logic [NUM_LANES-1:0][31:0] t0_q1;
    logic [NUM_LANES-1:0][31:0] y_q2;

    logic [NUM_LANES-1:0][31:0] stage0_mul;
    logic [NUM_LANES-1:0][31:0] stage0_add;
    logic [NUM_LANES-1:0][31:0] stage1_mul;
    logic [NUM_LANES-1:0][31:0] stage1_add;

    assign stage2_ready = !stage2_valid || ready_i;
    assign stage1_ready = !stage1_valid || stage2_ready;
    assign stage0_ready = !stage0_valid || stage1_ready;
    assign ready_o      = stage0_ready;
    assign valid_o      = stage2_valid;
    assign busy_o       = stage0_valid || stage1_valid || stage2_valid;
    assign data_o       = y_q2;

    for (genvar lane = 0; lane < NUM_LANES; lane++) begin : gen_silu_fp32_lane
        fp_mul #(
            .FpFormat_a  (fpnew_pkg_snax::FP32),
            .FpFormat_b  (fpnew_pkg_snax::FP32),
            .FpFormat_out(fpnew_pkg_snax::FP32)
        ) i_stage0_mul (
            .operand_a_i(a2_q0[lane]),
            .operand_b_i(x_q0[lane]),
            .result_o   (stage0_mul[lane])
        );

        fp_add #(
            .FpFormat_a  (fpnew_pkg_snax::FP32),
            .FpFormat_b  (fpnew_pkg_snax::FP32),
            .FpFormat_out(fpnew_pkg_snax::FP32)
        ) i_stage0_add (
            .operand_a_i(stage0_mul[lane]),
            .operand_b_i(a1_q0[lane]),
            .result_o   (stage0_add[lane])
        );

        fp_mul #(
            .FpFormat_a  (fpnew_pkg_snax::FP32),
            .FpFormat_b  (fpnew_pkg_snax::FP32),
            .FpFormat_out(fpnew_pkg_snax::FP32)
        ) i_stage1_mul (
            .operand_a_i(t0_q1[lane]),
            .operand_b_i(x_q1[lane]),
            .result_o   (stage1_mul[lane])
        );

        fp_add #(
            .FpFormat_a  (fpnew_pkg_snax::FP32),
            .FpFormat_b  (fpnew_pkg_snax::FP32),
            .FpFormat_out(fpnew_pkg_snax::FP32)
        ) i_stage1_add (
            .operand_a_i(stage1_mul[lane]),
            .operand_b_i(a0_q1[lane]),
            .result_o   (stage1_add[lane])
        );
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            stage0_valid <= 1'b0;
            stage1_valid <= 1'b0;
            stage2_valid <= 1'b0;
        end else begin
            if (stage0_ready) begin
                stage0_valid <= valid_i;
                if (valid_i) begin
                    for (int lane = 0; lane < NUM_LANES; lane++) begin
                        x_q0[lane]  <= data_i[lane];
                        a0_q0[lane] <= coeff_a0(segment_index(data_i[lane]));
                        a1_q0[lane] <= coeff_a1(segment_index(data_i[lane]));
                        a2_q0[lane] <= coeff_a2(segment_index(data_i[lane]));
                    end
                end
            end
            if (stage1_ready) begin
                stage1_valid <= stage0_valid;
                if (stage0_valid) begin
                    t0_q1 <= stage0_add;
                    x_q1  <= x_q0;
                    a0_q1 <= a0_q0;
                end
            end
            if (stage2_ready) begin
                stage2_valid <= stage1_valid;
                if (stage1_valid)
                    y_q2 <= stage1_add;
            end
        end
    end

endmodule

module elem_mul_fp32_multilane #(
    parameter int unsigned NUM_LANES = 4
) (
    input  logic                              clk_i,
    input  logic                              rst_ni,
    input  logic [NUM_LANES-1:0][31:0]        data_i_0,
    input  logic                              valid_i_0,
    output logic                              ready_o_0,
    input  logic [NUM_LANES-1:0][31:0]        data_i_1,
    input  logic                              valid_i_1,
    output logic                              ready_o_1,
    output logic [NUM_LANES-1:0][31:0]        data_o,
    output logic                              valid_o,
    output logic                              busy_o,
    input  logic                              ready_i
);

    logic stage_ready;
    logic [NUM_LANES-1:0][31:0] product;

    assign stage_ready = !valid_o || ready_i;
    assign ready_o_0   = stage_ready && valid_i_1;
    assign ready_o_1   = stage_ready && valid_i_0;
    assign busy_o      = valid_o;

    for (genvar lane = 0; lane < NUM_LANES; lane++) begin : gen_elem_mul_fp32_lane
        fp_mul #(
            .FpFormat_a  (fpnew_pkg_snax::FP32),
            .FpFormat_b  (fpnew_pkg_snax::FP32),
            .FpFormat_out(fpnew_pkg_snax::FP32)
        ) i_fp32_mul (
            .operand_a_i(data_i_0[lane]),
            .operand_b_i(data_i_1[lane]),
            .result_o   (product[lane])
        );
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_o <= 1'b0;
        end else if (stage_ready) begin
            valid_o <= valid_i_0 && valid_i_1;
            if (valid_i_0 && valid_i_1)
                data_o <= product;
        end
    end

endmodule

module fp32_to_fp16_multilane #(
    parameter int unsigned NUM_LANES = 4
) (
    input  logic                              clk_i,
    input  logic                              rst_ni,
    input  logic [NUM_LANES-1:0][31:0]        data_i,
    input  logic                              valid_i,
    output logic                              ready_o,
    output logic [NUM_LANES-1:0][15:0]        data_o,
    output logic                              valid_o,
    output logic                              busy_o,
    input  logic                              ready_i
);

    logic [NUM_LANES-1:0][15:0] narrowed;

    assign ready_o = !valid_o || ready_i;
    assign busy_o  = valid_o;

    for (genvar lane = 0; lane < NUM_LANES; lane++) begin : gen_fp32_to_fp16_lane
        fp_mul #(
            .FpFormat_a  (fpnew_pkg_snax::FP32),
            .FpFormat_b  (fpnew_pkg_snax::FP32),
            .FpFormat_out(fpnew_pkg_snax::FP16)
        ) i_fp32_to_fp16 (
            .operand_a_i(data_i[lane]),
            .operand_b_i(32'h3f800000),
            .result_o   (narrowed[lane])
        );
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            valid_o <= 1'b0;
        end else if (ready_o) begin
            valid_o <= valid_i;
            if (valid_i)
                data_o <= narrowed;
        end
    end

endmodule

module @TAG_NAME@_shell_wrapper #(
    parameter int unsigned RegRWCount    = @REG_RW_COUNT@,
    parameter int unsigned RegROCount    = @REG_RO_COUNT@,
    parameter int unsigned DataWidthA    = @DATA_WIDTH_A@,
    parameter int unsigned DataWidthB    = @DATA_WIDTH_B@,
    parameter int unsigned DataWidthD    = @DATA_WIDTH_D@,
    parameter int unsigned DataWidthOut  = @DATA_WIDTH_OUT@,
    parameter int unsigned PostprocLanes = @POSTPROC_LANES@,
    parameter int unsigned RegDataWidth  = 32,
    parameter int unsigned RegAddrWidth  = 32
) (
    input  logic                      clk_i,
    input  logic                      rst_ni,

    output logic [DataWidthOut-1:0]   acc2stream_0_data_o,
    output logic                      acc2stream_0_valid_o,
    input  logic                      acc2stream_0_ready_i,
    output logic [DataWidthOut-1:0]   acc2stream_1_data_o,
    output logic                      acc2stream_1_valid_o,
    input  logic                      acc2stream_1_ready_i,

    input  logic [DataWidthA-1:0]     stream2acc_0_data_i,
    input  logic                      stream2acc_0_valid_i,
    output logic                      stream2acc_0_ready_o,
    input  logic [DataWidthB-1:0]     stream2acc_1_data_i,
    input  logic                      stream2acc_1_valid_i,
    output logic                      stream2acc_1_ready_o,
    input  logic [DataWidthB-1:0]     stream2acc_2_data_i,
    input  logic                      stream2acc_2_valid_i,
    output logic                      stream2acc_2_ready_o,

    input  logic [RegRWCount-1:0][RegDataWidth-1:0] csr_reg_set_i,
    input  logic                                    csr_reg_set_valid_i,
    output logic                                    csr_reg_set_ready_o,
    output logic [RegROCount-1:0][RegDataWidth-1:0] csr_reg_ro_set_o
);

    localparam int unsigned ActiveCfgCount = 19;
    localparam int unsigned ElemsPerBeat = DataWidthD / 32;
    localparam int unsigned NumChunks =
        (ElemsPerBeat + PostprocLanes - 1) / PostprocLanes;

    logic [ActiveCfgCount-1:0][RegDataWidth-1:0] active_cfg;
    logic mode_sel;
    logic cores_ready;
    logic postproc_busy;
    logic launch_fire;
    logic ctrl_valid_to_vc;

    logic vc0_in_a_ready;
    logic vc0_in_a_valid;
    logic vc0_in_b_ready;
    logic vc0_in_b_valid;
    logic vc0_in_c_ready;
    logic [DataWidthD-1:0] vc0_out_d_data;
    logic vc0_out_d_valid;
    logic vc0_out_d_ready;
    logic vc0_ctrl_ready;
    logic vc0_busy;
    logic [31:0] vc0_perf_counter;

    logic vc1_in_a_ready;
    logic vc1_in_a_valid;
    logic vc1_in_b_ready;
    logic vc1_in_b_valid;
    logic vc1_in_c_ready;
    logic [DataWidthD-1:0] vc1_out_d_data;
    logic vc1_out_d_valid;
    logic vc1_out_d_ready;
    logic vc1_ctrl_ready;
    logic vc1_busy;
    logic [31:0] vc1_perf_counter;

    logic [DataWidthD-1:0] tied_c_data;
    assign tied_c_data = '0;
    assign mode_sel = active_cfg[6][0];

    // Shared A skid buffer.  Each token is retained until both cores consume it.
    logic [DataWidthA-1:0] a_buf_data;
    logic a_buf_valid;
    logic a_buf_sent_0;
    logic a_buf_sent_1;
    logic a_fire_0;
    logic a_fire_1;
    logic a_buf_done;

    assign vc0_in_a_valid = a_buf_valid && !a_buf_sent_0;
    assign vc1_in_a_valid = a_buf_valid && !a_buf_sent_1;
    assign a_fire_0 = vc0_in_a_valid && vc0_in_a_ready;
    assign a_fire_1 = vc1_in_a_valid && vc1_in_a_ready;
    assign a_buf_done = a_buf_valid &&
                        (a_buf_sent_0 || a_fire_0) &&
                        (a_buf_sent_1 || a_fire_1);
    assign stream2acc_0_ready_o = !a_buf_valid || a_buf_done;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            a_buf_valid  <= 1'b0;
            a_buf_sent_0 <= 1'b0;
            a_buf_sent_1 <= 1'b0;
        end else if (!a_buf_valid || a_buf_done) begin
            a_buf_valid <= stream2acc_0_valid_i;
            if (stream2acc_0_valid_i)
                a_buf_data <= stream2acc_0_data_i;
            a_buf_sent_0 <= 1'b0;
            a_buf_sent_1 <= 1'b0;
        end else begin
            if (a_fire_0) a_buf_sent_0 <= 1'b1;
            if (a_fire_1) a_buf_sent_1 <= 1'b1;
        end
    end

    assign stream2acc_1_ready_o = vc0_in_b_ready;
    assign vc0_in_b_valid = stream2acc_1_valid_i;
    assign stream2acc_2_ready_o = vc1_in_b_ready;
    assign vc1_in_b_valid = stream2acc_2_valid_i;

    assign cores_ready = vc0_ctrl_ready && vc1_ctrl_ready;
    assign csr_reg_set_ready_o = cores_ready && !postproc_busy;
    assign launch_fire = csr_reg_set_valid_i && csr_reg_set_ready_o;
    assign ctrl_valid_to_vc = launch_fire;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni)
            active_cfg <= '0;
        else if (launch_fire)
            active_cfg <= csr_reg_set_i[ActiveCfgCount-1:0];
    end

    VersaCore inst_VersaCore_0 (
        .clock(clk_i),
        .reset(~rst_ni),
        .io_versacore_data_in_a_ready(vc0_in_a_ready),
        .io_versacore_data_in_a_valid(vc0_in_a_valid),
        .io_versacore_data_in_a_bits(a_buf_data),
        .io_versacore_data_in_b_ready(vc0_in_b_ready),
        .io_versacore_data_in_b_valid(vc0_in_b_valid),
        .io_versacore_data_in_b_bits(stream2acc_1_data_i),
        .io_versacore_data_in_c_ready(vc0_in_c_ready),
        .io_versacore_data_in_c_valid(1'b1),
        .io_versacore_data_in_c_bits(tied_c_data),
        .io_versacore_data_out_d_ready(vc0_out_d_ready),
        .io_versacore_data_out_d_valid(vc0_out_d_valid),
        .io_versacore_data_out_d_bits(vc0_out_d_data),
        .io_ctrl_ready(vc0_ctrl_ready),
        .io_ctrl_valid(ctrl_valid_to_vc),
        .io_ctrl_bits_fsmCfg_take_in_new_c(csr_reg_set_i[0]),
        .io_ctrl_bits_fsmCfg_temporal_accumulation_times(csr_reg_set_i[1]),
        .io_ctrl_bits_fsmCfg_output_times(csr_reg_set_i[2]),
        .io_ctrl_bits_fsmCfg_subtraction_constant_i(csr_reg_set_i[3]),
        .io_ctrl_bits_arrayCfg_arrayShapeCfg(csr_reg_set_i[4]),
        .io_ctrl_bits_arrayCfg_dataTypeCfg(csr_reg_set_i[5]),
        .io_busy_o(vc0_busy),
        .io_performance_counter(vc0_perf_counter)
    );

    VersaCore inst_VersaCore_1 (
        .clock(clk_i),
        .reset(~rst_ni),
        .io_versacore_data_in_a_ready(vc1_in_a_ready),
        .io_versacore_data_in_a_valid(vc1_in_a_valid),
        .io_versacore_data_in_a_bits(a_buf_data),
        .io_versacore_data_in_b_ready(vc1_in_b_ready),
        .io_versacore_data_in_b_valid(vc1_in_b_valid),
        .io_versacore_data_in_b_bits(stream2acc_2_data_i),
        .io_versacore_data_in_c_ready(vc1_in_c_ready),
        .io_versacore_data_in_c_valid(1'b1),
        .io_versacore_data_in_c_bits(tied_c_data),
        .io_versacore_data_out_d_ready(vc1_out_d_ready),
        .io_versacore_data_out_d_valid(vc1_out_d_valid),
        .io_versacore_data_out_d_bits(vc1_out_d_data),
        .io_ctrl_ready(vc1_ctrl_ready),
        .io_ctrl_valid(ctrl_valid_to_vc),
        .io_ctrl_bits_fsmCfg_take_in_new_c(csr_reg_set_i[0]),
        .io_ctrl_bits_fsmCfg_temporal_accumulation_times(csr_reg_set_i[1]),
        .io_ctrl_bits_fsmCfg_output_times(csr_reg_set_i[2]),
        .io_ctrl_bits_fsmCfg_subtraction_constant_i(csr_reg_set_i[3]),
        .io_ctrl_bits_arrayCfg_arrayShapeCfg(csr_reg_set_i[4]),
        .io_ctrl_bits_arrayCfg_dataTypeCfg(csr_reg_set_i[5]),
        .io_busy_o(vc1_busy),
        .io_performance_counter(vc1_perf_counter)
    );

    // Joint output capture keeps both branches element-aligned.
    logic [DataWidthD-1:0] buf0_data;
    logic [DataWidthD-1:0] buf1_data;
    logic buf0_valid;
    logic buf1_valid;
    logic buf0_out_ready;
    logic buf1_out_ready;
    logic buf_can_accept;
    logic buf_fire;

    assign buf_can_accept =
        (!buf0_valid || buf0_out_ready) && (!buf1_valid || buf1_out_ready);
    assign vc0_out_d_ready = vc1_out_d_valid && buf_can_accept;
    assign vc1_out_d_ready = vc0_out_d_valid && buf_can_accept;
    assign buf_fire = vc0_out_d_valid && vc1_out_d_valid && buf_can_accept;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            buf0_valid <= 1'b0;
            buf1_valid <= 1'b0;
        end else begin
            if (!buf0_valid || buf0_out_ready) begin
                buf0_valid <= buf_fire;
                if (buf_fire) buf0_data <= vc0_out_d_data;
            end
            if (!buf1_valid || buf1_out_ready) begin
                buf1_valid <= buf_fire;
                if (buf_fire) buf1_data <= vc1_out_d_data;
            end
        end
    end

    logic [$clog2(NumChunks > 1 ? NumChunks : 2)-1:0] chunk_cnt_0;
    logic [$clog2(NumChunks > 1 ? NumChunks : 2)-1:0] chunk_cnt_1;
    logic chunk_last_0;
    logic chunk_last_1;
    logic [PostprocLanes-1:0][31:0] chunk_ser0_data;
    logic [PostprocLanes-1:0][31:0] chunk_ser1_data;
    logic chunk_ser0_valid;
    logic chunk_ser1_valid;
    logic chunk_ser0_ready;
    logic chunk_ser1_ready;

    assign chunk_last_0 = (NumChunks <= 1) || (chunk_cnt_0 == NumChunks - 1);
    assign chunk_last_1 = (NumChunks <= 1) || (chunk_cnt_1 == NumChunks - 1);
    assign chunk_ser0_valid = buf0_valid;
    assign chunk_ser1_valid = buf1_valid;
    assign buf0_out_ready = chunk_ser0_ready && chunk_last_0;
    assign buf1_out_ready = chunk_ser1_ready && chunk_last_1;

    always_comb begin
        for (int lane = 0; lane < PostprocLanes; lane++) begin
            int idx0;
            int idx1;
            idx0 = chunk_cnt_0 * PostprocLanes + lane;
            idx1 = chunk_cnt_1 * PostprocLanes + lane;
            chunk_ser0_data[lane] =
                (idx0 < ElemsPerBeat) ? buf0_data[idx0*32 +: 32] : '0;
            chunk_ser1_data[lane] =
                (idx1 < ElemsPerBeat) ? buf1_data[idx1*32 +: 32] : '0;
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            chunk_cnt_0 <= '0;
            chunk_cnt_1 <= '0;
        end else begin
            if (chunk_ser0_valid && chunk_ser0_ready)
                chunk_cnt_0 <= chunk_last_0 ? '0 : chunk_cnt_0 + 1'b1;
            if (chunk_ser1_valid && chunk_ser1_ready)
                chunk_cnt_1 <= chunk_last_1 ? '0 : chunk_cnt_1 + 1'b1;
        end
    end

    logic [PostprocLanes-1:0][31:0] silu_out_data;
    logic silu_in_valid;
    logic silu_in_ready;
    logic silu_out_valid;
    logic silu_out_ready;
    logic silu_busy;

    assign silu_in_valid = chunk_ser0_valid && !mode_sel;

    silu_fp32_multilane #(
        .NUM_LANES(PostprocLanes)
    ) i_silu_fp32 (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .data_i(chunk_ser0_data),
        .valid_i(silu_in_valid),
        .ready_o(silu_in_ready),
        .data_o(silu_out_data),
        .valid_o(silu_out_valid),
        .busy_o(silu_busy),
        .ready_i(silu_out_ready)
    );

    logic [PostprocLanes-1:0][31:0] elem_mul_out_data;
    logic elem_mul_in1_valid;
    logic elem_mul_in1_ready;
    logic elem_mul_out_valid;
    logic elem_mul_out_ready;
    logic elem_mul_busy;

    assign elem_mul_in1_valid = chunk_ser1_valid && !mode_sel;

    elem_mul_fp32_multilane #(
        .NUM_LANES(PostprocLanes)
    ) i_elem_mul_fp32 (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .data_i_0(silu_out_data),
        .valid_i_0(silu_out_valid),
        .ready_o_0(silu_out_ready),
        .data_i_1(chunk_ser1_data),
        .valid_i_1(elem_mul_in1_valid),
        .ready_o_1(elem_mul_in1_ready),
        .data_o(elem_mul_out_data),
        .valid_o(elem_mul_out_valid),
        .busy_o(elem_mul_busy),
        .ready_i(elem_mul_out_ready)
    );

    logic [PostprocLanes-1:0][31:0] fp16ext0_in_data;
    logic [PostprocLanes-1:0][31:0] fp16ext1_in_data;
    logic fp16ext0_in_valid;
    logic fp16ext1_in_valid;
    logic fp16ext0_in_ready;
    logic fp16ext1_in_ready;
    logic fp16ext0_busy;
    logic fp16ext1_busy;

    always_comb begin
        fp16ext0_in_data  = mode_sel ? chunk_ser0_data : elem_mul_out_data;
        fp16ext0_in_valid = mode_sel ? chunk_ser0_valid : elem_mul_out_valid;
        fp16ext1_in_data  = chunk_ser1_data;
        fp16ext1_in_valid = mode_sel && chunk_ser1_valid;

        if (mode_sel) begin
            chunk_ser0_ready = fp16ext0_in_ready;
            chunk_ser1_ready = fp16ext1_in_ready;
            elem_mul_out_ready = 1'b1;
        end else begin
            chunk_ser0_ready = silu_in_ready;
            chunk_ser1_ready = elem_mul_in1_ready;
            elem_mul_out_ready = fp16ext0_in_ready;
        end
    end

    fp32_to_fp16_multilane #(
        .NUM_LANES(PostprocLanes)
    ) i_writer0_fp32_to_fp16 (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .data_i(fp16ext0_in_data),
        .valid_i(fp16ext0_in_valid),
        .ready_o(fp16ext0_in_ready),
        .data_o(acc2stream_0_data_o),
        .valid_o(acc2stream_0_valid_o),
        .busy_o(fp16ext0_busy),
        .ready_i(acc2stream_0_ready_i)
    );

    fp32_to_fp16_multilane #(
        .NUM_LANES(PostprocLanes)
    ) i_writer1_fp32_to_fp16 (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .data_i(fp16ext1_in_data),
        .valid_i(fp16ext1_in_valid),
        .ready_o(fp16ext1_in_ready),
        .data_o(acc2stream_1_data_o),
        .valid_o(acc2stream_1_valid_o),
        .busy_o(fp16ext1_busy),
        .ready_i(acc2stream_1_ready_i)
    );

    assign postproc_busy = buf0_valid || buf1_valid || silu_busy ||
                           elem_mul_busy || fp16ext0_busy || fp16ext1_busy;

    assign csr_reg_ro_set_o[0] =
        {31'b0, vc0_busy || vc1_busy || postproc_busy};
    assign csr_reg_ro_set_o[1] =
        (vc0_perf_counter > vc1_perf_counter) ?
        vc0_perf_counter : vc1_perf_counter;

endmodule
