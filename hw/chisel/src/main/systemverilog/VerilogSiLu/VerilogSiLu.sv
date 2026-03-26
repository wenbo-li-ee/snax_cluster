module VerilogSiLu #(
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

  localparam int NumElement = DataWidth / 32;
  localparam logic [31:0] SIG_ONE_CODE  = 32'd33554432;   // 2^25
  localparam logic [31:0] SIG_HALF_CODE = 32'd16777216;   // 2^24
  localparam logic [31:0] BREAK_CODE    = 32'd134217728;  // 2^27

  initial begin
    if ((DataWidth % 32) != 0) begin
      $error("DataWidth (%0d) must be a multiple of 32", DataWidth);
    end
  end

  // Pure combinational pass-through handshake, as required.
  assign ext_data_i_ready = ext_data_o_ready;
  assign ext_data_o_valid = ext_data_i_valid;
  assign ext_busy_o       = 1'b0;

  // Keep compatibility-only inputs intentionally unused.
  logic [2:0] _unused_ok;
  assign _unused_ok = {clk_i ^ rst_ni, ext_start_i, ext_csr_i_0[0] ^ UserCsrNum[0]};

  genvar i;
  generate
    for (i = 0; i < NumElement; i++) begin : g_lane
      logic signed [31:0] x_code;
      logic        [31:0] abs_code;
      logic        [31:0] s_pos_code;
      logic        [31:0] s_code;
      logic        [63:0] abs_sq;
      logic signed [63:0] mul_full;
      logic signed [63:0] z_shifted;
      logic signed [31:0] z_clamped;

      always_comb begin
        x_code = ext_data_i_bits[i*32 +: 32];

        if (x_code[31]) begin
          abs_code = -x_code;
        end else begin
          abs_code = x_code;
        end

        if (abs_code >= BREAK_CODE) begin
          s_pos_code = SIG_ONE_CODE;
        end else begin
          abs_sq     = abs_code * abs_code;
          s_pos_code = SIG_HALF_CODE + (abs_code >> 2) - (abs_sq >> 30);
        end

        if (x_code >= 0) begin
          s_code = s_pos_code;
        end else begin
          s_code = SIG_ONE_CODE - s_pos_code;
        end

        mul_full  = x_code * $signed({1'b0, s_code});
        z_shifted = mul_full >>> 25;

        if (z_shifted > $signed(64'sd2147483647)) begin
          z_clamped = 32'sh7fffffff;
        end else if (z_shifted < $signed(-64'sd2147483648)) begin
          z_clamped = 32'sh80000000;
        end else begin
          z_clamped = z_shifted[31:0];
        end
      end

      assign ext_data_o_bits[i*32 +: 32] = z_clamped;
    end
  endgenerate

endmodule
