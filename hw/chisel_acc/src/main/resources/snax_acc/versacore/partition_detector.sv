module partition_detector (
  input  logic                            clk,
  input  logic                            rst_n,
  input  logic                            ce,
  input  silu_out16_balanced_pkg::input_t       x_in,
  input  logic                            valid_in,
  output silu_out16_balanced_pkg::input_t       x_out,
  output silu_out16_balanced_pkg::seg_idx_t     seg_idx_out,
  output logic                            valid_out
);

  import silu_out16_balanced_pkg::*;

  seg_idx_t seg_idx_next;

  always_comb begin
    seg_idx_next = segment_index_from_x(x_in);
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      x_out       <= '0;
      seg_idx_out <= '0;
      valid_out   <= 1'b0;
    end else if (ce) begin
      x_out       <= x_in;
      seg_idx_out <= seg_idx_next;
      valid_out   <= valid_in;
    end
  end

endmodule
