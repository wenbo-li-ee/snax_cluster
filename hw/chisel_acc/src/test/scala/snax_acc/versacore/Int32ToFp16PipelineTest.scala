// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snax_acc.versacore

import chisel3._
import chisel3.util._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

private class Int32ToFp16Sv(lanes: Int)
    extends BlackBox(Map("NUM_LANES" -> lanes))
    with HasBlackBoxResource {
  override def desiredName = "int32_to_fp16"
  val io = IO(new Bundle {
    val clk_i   = Input(Clock())
    val rst_ni  = Input(Bool())
    val data_i  = Input(UInt((lanes * 32).W))
    val valid_i = Input(Bool())
    val ready_o = Output(Bool())
    val data_o  = Output(UInt((lanes * 16).W))
    val valid_o = Output(Bool())
    val ready_i = Input(Bool())
  })
  addResource("/snax_acc/versacore/int32_to_fp16.sv")
}

private class Int32ToFp16PipelineHarness(lanes: Int = 4) extends Module {
  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(UInt((lanes * 32).W)))
    val out = Decoupled(UInt((lanes * 16).W))
  })

  val dut = Module(new Int32ToFp16Sv(lanes))
  dut.io.clk_i   := clock
  dut.io.rst_ni  := !reset.asBool
  dut.io.data_i  := io.in.bits
  dut.io.valid_i := io.in.valid
  io.in.ready    := dut.io.ready_o
  io.out.bits    := dut.io.data_o
  io.out.valid   := dut.io.valid_o
  dut.io.ready_i := io.out.ready
}

class Int32ToFp16PipelineTest extends AnyFlatSpec with ChiselScalatestTester {
  private def pack32(values: Seq[Int]): BigInt =
    values.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (value, lane)) =>
      acc | (BigInt(value.toLong & 0xffffffffL) << (32 * lane))
    }

  private def pack16(values: Seq[Int]): BigInt =
    values.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (value, lane)) =>
      acc | (BigInt(value & 0xffff) << (16 * lane))
    }

  private val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  behavior of "int32_to_fp16 SV elastic pipeline"

  it should "convert four lanes with one-cycle latency and hold/pop-push under backpressure" in {
    test(new Int32ToFp16PipelineHarness()).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val batches = Seq(
        Seq(0, 1, -1, 2048) -> Seq(0x0000, 0x3c00, 0xbc00, 0x6800),
        Seq(2049, 2051, 65504, 65519) -> Seq(0x6800, 0x6802, 0x7bff, 0x7bff),
        Seq(65520, -65520, Int.MaxValue, Int.MinValue) -> Seq(0x7c00, 0xfc00, 0x7c00, 0xfc00)
      )

      dut.io.in.valid.poke(false.B)
      dut.io.out.ready.poke(false.B)
      dut.clock.step()
      dut.io.out.valid.expect(false.B)

      // Empty output register accepts the first batch; it becomes visible
      // exactly after the accepting edge.
      dut.io.in.bits.poke(pack32(batches(0)._1).U)
      dut.io.in.valid.poke(true.B)
      dut.io.in.ready.expect(true.B)
      dut.io.out.valid.expect(false.B)
      dut.clock.step()
      dut.io.out.valid.expect(true.B)
      dut.io.out.bits.expect(pack16(batches(0)._2).U)

      // With downstream stalled, ready must fall and both valid/data must hold.
      dut.io.in.bits.poke(pack32(batches(1)._1).U)
      dut.io.in.ready.expect(false.B)
      dut.clock.step(3)
      dut.io.out.valid.expect(true.B)
      dut.io.out.bits.expect(pack16(batches(0)._2).U)

      // Pop the old batch and push the second batch on the same edge.
      dut.io.out.ready.poke(true.B)
      dut.io.in.ready.expect(true.B)
      dut.clock.step()
      dut.io.out.valid.expect(true.B)
      dut.io.out.bits.expect(pack16(batches(1)._2).U)

      // Continuous-ready operation accepts and produces one four-lane batch
      // per cycle (II=1).
      dut.io.in.bits.poke(pack32(batches(2)._1).U)
      dut.clock.step()
      dut.io.out.valid.expect(true.B)
      dut.io.out.bits.expect(pack16(batches(2)._2).U)

      dut.io.in.valid.poke(false.B)
      dut.clock.step()
      dut.io.out.valid.expect(false.B)
    }
  }
}
