module VerilogSiluNew #(
    parameter int UserCsrNum = 1,
    parameter int DataWidth  = 32
) (
    input  logic                     clk_i,
    input  logic                     rst_ni,
    output logic                     ext_data_i_ready,
    input  logic                     ext_data_i_valid,
    input  logic [DataWidth-1:0]     ext_data_i_bits,
    input  logic                     ext_data_o_ready,
    output logic                     ext_data_o_valid,
    output logic [DataWidth-1:0]     ext_data_o_bits,
    input  logic [31:0]              ext_csr_i_0,
    input  logic                     ext_start_i,
    output logic                     ext_busy_o
);

  import silu_hp32_q22_pkg::*;

  localparam int NumElement = DataWidth / 32;

  initial begin
    if ((DataWidth % 32) != 0) begin
      $error("DataWidth (%0d) must be a multiple of 32", DataWidth);
    end
  end

  logic [2:0] _unused_ok;
  assign _unused_ok = {clk_i ^ rst_ni, ext_start_i, ext_csr_i_0[0] ^ UserCsrNum[0]};

  logic                     core_valid_in;
  logic                     core_valid_out;
  logic                     wait_for_core;
  logic [DataWidth-1:0]     output_buf_q;
  logic [NumElement-1:0]    lane_valid_out;
  logic [DataWidth-1:0]     lane_output_bits;

  assign core_valid_in     = ext_data_i_valid && ext_data_i_ready;
  assign ext_data_i_ready  = !(wait_for_core || ext_data_o_valid);
  assign ext_busy_o        = wait_for_core || ext_data_o_valid;
  assign ext_data_o_bits   = output_buf_q;

  genvar i;
  generate
    for (i = 0; i < NumElement; i++) begin : g_lane
      input_t   lane_x_in;
      output_t  lane_y_out;
      seg_idx_t lane_seg_out;

      assign lane_x_in = input_t'(ext_data_i_bits[i*32 + 16 +: 16]);

      silu_top u_silu_top (
        .clk        (clk_i),
        .rst_n      (rst_ni),
        .x_in       (lane_x_in),
        .valid_in   (core_valid_in),
        .y_out      (lane_y_out),
        .seg_idx_out(lane_seg_out),
        .valid_out  (lane_valid_out[i])
      );

      assign lane_output_bits[i*32 +: 32] = lane_y_out;
    end
  endgenerate

  assign core_valid_out = lane_valid_out[0];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wait_for_core    <= 1'b0;
      ext_data_o_valid <= 1'b0;
      output_buf_q     <= '0;
    end else begin
      if (core_valid_in) begin
        wait_for_core <= 1'b1;
      end

      if (core_valid_out) begin
        output_buf_q     <= lane_output_bits;
        ext_data_o_valid <= 1'b1;
        wait_for_core    <= 1'b0;
      end

      if (ext_data_o_valid && ext_data_o_ready) begin
        ext_data_o_valid <= 1'b0;
      end
    end
  end

endmodule
