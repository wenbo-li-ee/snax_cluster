package snax.readerWriter

import chisel3._

import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class AddressGenUnitTester extends AnyFlatSpec with ChiselScalatestTester {

  println(
    getVerilogString(new AddressGenUnit(AddressGenUnitParam()))
  )

  "AddressGenUnit: continuous fetch with first temporal loop disabled" should " pass" in test(
    new AddressGenUnit(
      AddressGenUnitParam(
        spatialBounds     = List(8),
        temporalDimension = 2,
        numChannel        = 8,
        outputBufferDepth = 2,
        tcdmSize          = 128
      )
    )
  )
    .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) { dut =>
      dut.io.cfg.ptr.poke(0x1000.U)
      dut.io.cfg.spatialStrides(0).poke(8)
      dut.io.cfg.temporalStrides(0).poke(64)
      dut.io.cfg.temporalStrides(1).poke(64)
      dut.io.cfg.temporalBounds(0).poke(0)
      dut.io.cfg.temporalBounds(1).poke(16)

      dut.io.start.poke(true)
      dut.clock.step()
      dut.io.start.poke(false)
      for (i <- 0 until 16) {
        dut.clock.step()
      }

      dut.io.addr.foreach(_.ready.poke(true.B))
      for (i <- 0 until 48) {
        dut.clock.step()
      }

    }

  "AddressGenUnit: continuous fetch with first temporal loop enabled" should " pass" in test(
    new AddressGenUnit(
      AddressGenUnitParam(
        spatialBounds     = List(8),
        temporalDimension = 2,
        numChannel        = 8,
        outputBufferDepth = 2,
        tcdmSize          = 128
      )
    )
  )
    .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) { dut =>
      dut.io.cfg.ptr.poke(0x1000.U)
      dut.io.cfg.spatialStrides(0).poke(8)
      dut.io.cfg.temporalStrides(0).poke(64)
      dut.io.cfg.temporalStrides(1).poke(256)
      dut.io.cfg.temporalBounds(0).poke(4)
      dut.io.cfg.temporalBounds(1).poke(4)

      dut.io.start.poke(true)
      dut.clock.step()
      dut.io.start.poke(false)
      for (i <- 0 until 16) {
        dut.clock.step()
      }

      dut.io.addr.foreach(_.ready.poke(true.B))
      for (i <- 0 until 48) {
        dut.clock.step()
      }
    }

  "AddressGenUnit: two-dimensional spatial token layout" should " pass" in test(
    new AddressGenUnit(
      AddressGenUnitParam(
        spatialBounds     = List(2, 4),
        temporalDimension = 1,
        numChannel        = 8,
        outputBufferDepth = 2,
        tcdmSize          = 8192
      )
    )
  ) { dut =>
    val base     = 0x1000
    val expected = Seq(0, 8, 512, 520, 1024, 1032, 1536, 1544)

    dut.io.cfg.ptr.poke(base.U)
    dut.io.cfg.spatialStrides(0).poke(8.U)
    dut.io.cfg.spatialStrides(1).poke(512.U)
    dut.io.cfg.temporalStrides(0).poke(2048.U)
    dut.io.cfg.temporalBounds(0).poke(1.U)
    dut.io.cfg.addressRemapIndex.poke(0.U)
    dut.io.addr.foreach(_.ready.poke(true.B))

    dut.io.start.poke(true.B)
    dut.clock.step()
    dut.io.start.poke(false.B)
    while (!dut.io.addr.head.valid.peek().litToBoolean) {
      dut.clock.step()
    }
    dut.io.addr.zip(expected).foreach { case (addr, offset) =>
      addr.valid.expect(true.B)
      addr.bits.expect((base + offset).U)
    }
  }

  "AddressGenUnit: continuous 1D fetch with memory remapped to non-interleaved in superbank" should " pass" in test(
    new AddressGenUnit(
      AddressGenUnitParam(
        spatialBounds     = List(8),
        temporalDimension = 2,
        numChannel        = 8,
        outputBufferDepth = 2,
        tcdmSize          = 128,
        tcdmPhysWordSize  = 256,
        tcdmLogicWordSize = Seq(256, 128, 64)
      )
    )
  )
    .withAnnotations(Seq(WriteVcdAnnotation, VerilatorBackendAnnotation)) { dut =>
      dut.clock.setTimeout(0)
      dut.io.cfg.ptr.poke(0x0.U)
      dut.io.cfg.spatialStrides(0).poke(8)
      dut.io.cfg.temporalStrides(0).poke(64)
      dut.io.cfg.temporalStrides(1).poke(0)
      dut.io.cfg.temporalBounds(0).poke(2048)
      dut.io.cfg.temporalBounds(1).poke(1)
      dut.io.cfg.addressRemapIndex.poke(2)

      dut.io.start.poke(true)
      dut.clock.step()
      dut.io.start.poke(false)
      for (i <- 0 until 16) {
        dut.clock.step()
      }

      dut.io.addr.foreach(_.ready.poke(true.B))
      for (i <- 0 until 2048) {
        dut.clock.step()
      }
    }
}

object AddressGenUnitEmitter extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new AddressGenUnit(
      AddressGenUnitParam(
        spatialBounds     = List(8),
        temporalDimension = 2,
        numChannel        = 8,
        outputBufferDepth = 8,
        tcdmSize          = 128,
        tcdmPhysWordSize  = 256,
        tcdmLogicWordSize = Seq(256, 128, 64)
      )
    ),
    args = Array("--target-dir", "generated/xdma")
  )
}
