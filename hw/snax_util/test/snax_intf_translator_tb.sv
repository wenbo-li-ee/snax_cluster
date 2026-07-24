package snax_intf_translator_tb_pkg;
  typedef struct packed {
    logic [4:0]  id;
    logic [31:0] data_op;
    logic [63:0] data_arga;
    logic [63:0] data_argb;
  } acc_req_t;

  typedef struct packed {
    logic [4:0]  id;
    logic        error;
    logic [63:0] data;
  } acc_rsp_t;

  typedef struct packed {
    logic [31:0] addr;
    logic [31:0] data;
    logic        write;
  } csr_req_t;

  typedef struct packed {
    logic [31:0] data;
  } csr_rsp_t;
endpackage

module snax_intf_translator_tb;
  import riscv_instr::*;
  import csr_snax_def::*;
  import snax_intf_translator_tb_pkg::*;

  logic clk = 0;
  logic rst_n = 0;
  always #5 clk = ~clk;

  acc_req_t req;
  logic qvalid;
  logic qready;
  acc_rsp_t rsp;
  logic pvalid;
  logic pready;
  csr_req_t csr_req;
  logic acc_req_valid, acc_req_ready;
  logic top_req_valid, top_req_ready;
  csr_rsp_t acc_rsp, top_rsp;
  logic acc_rsp_valid, acc_rsp_ready;
  logic top_rsp_valid, top_rsp_ready;

  snax_intf_translator #(
    .acc_req_t(acc_req_t),
    .acc_rsp_t(acc_rsp_t),
    .csr_req_t(csr_req_t),
    .csr_rsp_t(csr_rsp_t),
    .NumOutstandingLoads(4),
    .CsrAddrOffset(32'h3c0)
  ) dut (
    .clk_i(clk), .rst_ni(rst_n),
    .snax_req_i(req), .snax_qvalid_i(qvalid), .snax_qready_o(qready),
    .snax_resp_o(rsp), .snax_pvalid_o(pvalid), .snax_pready_i(pready),
    .snax_csr_req_o(csr_req),
    .snax_csr_req_acc_valid_o(acc_req_valid),
    .snax_csr_req_acc_ready_i(acc_req_ready),
    .snax_csr_req_top_valid_o(top_req_valid),
    .snax_csr_req_top_ready_i(top_req_ready),
    .snax_csr_rsp_acc_i(acc_rsp),
    .snax_csr_rsp_acc_valid_i(acc_rsp_valid),
    .snax_csr_rsp_acc_ready_o(acc_rsp_ready),
    .snax_csr_rsp_top_i(top_rsp),
    .snax_csr_rsp_top_valid_i(top_rsp_valid),
    .snax_csr_rsp_top_ready_o(top_rsp_ready)
  );

  task automatic idle_inputs;
    req = '0;
    qvalid = 0;
    pready = 1;
    acc_rsp = '0;
    acc_rsp_valid = 0;
    top_rsp = '0;
    top_rsp_valid = 0;
  endtask

  task automatic set_read(input logic [4:0] id, input logic [11:0] addr);
    req = '0;
    req.id = id;
    req.data_op = CSRRS;
    req.data_argb = addr;
    qvalid = 1;
  endtask

  task automatic set_write(input logic [4:0] id, input logic [11:0] addr);
    req = '0;
    req.id = id;
    req.data_op = 32'h0000_1073;
    req.data_argb = addr;
    req.data_arga = 1;
    qvalid = 1;
  endtask

  task automatic expect_rsp(input logic [4:0] id, input logic [31:0] data);
    #1;
    if (!pvalid || rsp.id != id || rsp.data != data) begin
      $fatal(1, "response mismatch: valid=%0d id=%0d data=%0d expected id=%0d data=%0d",
             pvalid, rsp.id, rsp.data, id, data);
    end
  endtask

  task automatic push_delayed_read(input logic [4:0] id, input logic [11:0] addr);
    @(negedge clk);
    set_read(id, addr);
    acc_rsp_valid = 0;
    top_rsp_valid = 0;
    #1;
    if (!qready) $fatal(1, "read ID%0d unexpectedly backpressured", id);
    @(posedge clk);
  endtask

  task automatic accept_acc_rsp(input logic [4:0] id, input logic [31:0] data);
    @(negedge clk);
    qvalid = 0;
    acc_rsp.data = data;
    acc_rsp_valid = 1;
    pready = 1;
    expect_rsp(id, data);
    if (!acc_rsp_ready) $fatal(1, "ACC response ID%0d not ready", id);
    @(posedge clk);
    @(negedge clk);
    acc_rsp_valid = 0;
  endtask

  initial begin
    acc_req_ready = 1;
    top_req_ready = 1;
    idle_inputs();
    repeat (2) @(posedge clk);
    rst_n = 1;

    // Preserve the zero-latency path for an immediate response.
    @(negedge clk);
    set_read(5'd1, 12'h3ff);
    acc_rsp.data = 32'd101;
    acc_rsp_valid = 1;
    pready = 1;
    if (!qready) $fatal(1, "immediate read was not accepted");
    expect_rsp(5'd1, 32'd101);
    @(posedge clk);
    @(negedge clk);
    idle_inputs();
    if (!dut.rsp_fifo_empty) $fatal(1, "immediate response left a stale FIFO ID");

    // Reproduce the system sequence: read ID15, held response, write ID0.
    set_read(5'd15, 12'h3ff);
    acc_rsp.data = 32'd8;
    acc_rsp_valid = 1;
    pready = 0;
    expect_rsp(5'd15, 32'd8);
    @(posedge clk);
    @(negedge clk);
    set_write(5'd0, 12'h3fd);
    pready = 1;
    expect_rsp(5'd15, 32'd8);
    @(posedge clk);
    @(negedge clk);
    idle_inputs();
    set_read(5'd22, 12'h3ff);
    acc_rsp.data = 32'd9;
    acc_rsp_valid = 1;
    expect_rsp(5'd22, 32'd9);
    @(posedge clk);
    @(negedge clk);
    idle_inputs();

    // Delayed and consecutive responses preserve FIFO order.
    push_delayed_read(5'd3, 12'h3ff);
    accept_acc_rsp(5'd3, 32'd103);
    push_delayed_read(5'd4, 12'h3ff);
    push_delayed_read(5'd5, 12'h3ff);
    accept_acc_rsp(5'd4, 32'd104);
    accept_acc_rsp(5'd5, 32'd105);

    // Top-level scheduler CSR uses the same response-ID tracking.
    @(negedge clk);
    set_read(5'd6, CSR_SNAX_READ_TASK_READY_QUEUE);
    top_rsp.data = 32'd106;
    top_rsp_valid = 1;
    pready = 0;
    #1;
    if (!top_req_valid || acc_req_valid) $fatal(1, "top-level request misrouted");
    expect_rsp(5'd6, 32'd106);
    @(posedge clk);
    @(negedge clk);
    set_write(5'd0, 12'h3fd);
    pready = 1;
    expect_rsp(5'd6, 32'd106);
    @(posedge clk);
    @(negedge clk);
    idle_inputs();

    // A full response-ID FIFO must backpressure another read request.
    push_delayed_read(5'd7, 12'h3ff);
    push_delayed_read(5'd8, 12'h3ff);
    push_delayed_read(5'd9, 12'h3ff);
    push_delayed_read(5'd10, 12'h3ff);
    @(negedge clk);
    set_read(5'd11, 12'h3ff);
    #1;
    if (qready || acc_req_valid) $fatal(1, "read accepted while response-ID FIFO full");
    @(posedge clk);

    @(negedge clk);
    acc_rsp.data = 32'd107;
    acc_rsp_valid = 1;
    pready = 1;
    expect_rsp(5'd7, 32'd107);
    if (qready) $fatal(1, "full FIFO accepted read in the pop cycle");
    @(posedge clk);
    @(negedge clk);
    acc_rsp_valid = 0;
    #1;
    if (!qready || !acc_req_valid) $fatal(1, "blocked read not released after FIFO pop");
    @(posedge clk);
    @(negedge clk);
    qvalid = 0;

    accept_acc_rsp(5'd8, 32'd108);
    accept_acc_rsp(5'd9, 32'd109);
    accept_acc_rsp(5'd10, 32'd110);
    accept_acc_rsp(5'd11, 32'd111);

    if (!dut.rsp_fifo_empty) $fatal(1, "FIFO not empty after draining all responses");
    $display("TEST_PASS: translator handshake and response-ID matrix passed");
    $finish;
  end
endmodule
