// Copyright 2026 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

//-------------------------------
// Simple set of CSRs for the
// element-wise unit accelerator
//-------------------------------
module snax_ewu_csr #(
  parameter int unsigned RegRWCount   = 3,
  parameter int unsigned RegROCount   = 2,
  parameter int unsigned RegDataWidth = 32
)(
  //-------------------------------
  // Clocks and reset
  //-------------------------------
  input  logic                                    clk_i,
  input  logic                                    rst_ni,
  //-------------------------------
  // Register RW from CSR manager
  //-------------------------------
  input  logic [RegRWCount-1:0][RegDataWidth-1:0] csr_reg_set_i,
  input  logic                                    csr_reg_set_valid_i,
  output logic                                    csr_reg_set_ready_o,
  //-------------------------------
  // Register RO to CSR manager
  //-------------------------------
  output logic [RegROCount-1:0][RegDataWidth-1:0] csr_reg_ro_set_o,
  //-------------------------------
  // Direct register control signals
  //-------------------------------
  input  logic                                    acc_output_success_i,
  output logic                                    acc_ready_o,
  output logic               [1:0]                csr_op_config_o
);

  logic [RegRWCount-1:0][RegDataWidth-1:0] csr_reg_rw_set;
  logic [RegDataWidth-1:0]                 csr_reg_rw_len;
  logic                                    csr_reg_set_req_success;

  logic                    reg_ro_busy;
  logic [RegDataWidth-1:0] reg_ro_perf_counter;
  logic [RegDataWidth-1:0] len_counter;
  logic                    len_counter_finish;

  assign csr_reg_set_ready_o   = 1'b1;
  assign csr_reg_set_req_success = csr_reg_set_valid_i && csr_reg_set_ready_o;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < RegRWCount; i++) begin
        csr_reg_rw_set[i] <= '0;
      end
    end else if (csr_reg_set_req_success) begin
      csr_reg_rw_set <= csr_reg_set_i;
    end
  end

  assign csr_op_config_o   = csr_reg_rw_set[0][1:0];
  assign csr_reg_rw_len    = csr_reg_rw_set[1];
  assign acc_ready_o       = reg_ro_busy;
  assign len_counter_finish = (len_counter == (csr_reg_rw_len - 1));

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      len_counter         <= '0;
      reg_ro_busy         <= 1'b0;
      reg_ro_perf_counter <= '0;
    end else begin
      if (len_counter_finish && acc_output_success_i) begin
        reg_ro_busy <= 1'b0;
      end else if (csr_reg_set_req_success) begin
        reg_ro_busy <= 1'b1;
      end

      if (csr_reg_set_req_success) begin
        reg_ro_perf_counter <= 1;
      end else if (reg_ro_busy && !len_counter_finish) begin
        reg_ro_perf_counter <= reg_ro_perf_counter + 1;
      end

      if (len_counter_finish && acc_output_success_i) begin
        len_counter <= '0;
      end else if (acc_output_success_i) begin
        len_counter <= len_counter + 1;
      end
    end
  end

  assign csr_reg_ro_set_o[0] = {{(RegDataWidth-1){1'b0}}, reg_ro_busy};
  assign csr_reg_ro_set_o[1] = reg_ro_perf_counter;

endmodule
