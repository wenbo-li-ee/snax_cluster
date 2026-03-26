module VerilogMinusOne #(
    parameter int UserCsrNum = 1,
    parameter int DataWidth = 512
) (
    input  logic clk_i,
    input  logic rst_ni,
    output logic ext_data_i_ready,
    input  logic ext_data_i_valid,
    input  logic [DataWidth-1:0] ext_data_i_bits,
    input  logic ext_data_o_ready,
    output logic ext_data_o_valid,
    output logic [DataWidth-1:0] ext_data_o_bits,
    input  logic [31:0]ext_csr_i_0,
    input  logic ext_start_i,
    output logic ext_busy_o
);

    localparam int ElementWidth = 32;
    localparam int NumElement = DataWidth / ElementWidth;

    assign ext_data_o_valid = ext_data_i_valid;
    assign ext_data_i_ready = ext_data_o_ready;

    genvar i;
    generate
        for (i = 0; i < NumElement; i = i + 1) begin : g_sub_one
            assign ext_data_o_bits[i*ElementWidth +: ElementWidth] =
                ext_data_i_bits[i*ElementWidth +: ElementWidth] - ElementWidth'(1);
        end
    endgenerate

    assign ext_busy_o = 1'b0;
    logic _unused;
    assign _unused = clk_i ^ rst_ni ^ ext_csr_i_0[0] ^ ext_start_i;

endmodule
