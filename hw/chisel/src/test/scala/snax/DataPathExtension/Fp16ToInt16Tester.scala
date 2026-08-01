// Copyright 2025 KU Leuven.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

package snax.DataPathExtension

import scala.util.Random

import chisel3._
import chiseltest._
import chiseltest.simulator.VerilatorFlags
import org.scalatest.flatspec.AnyFlatSpec

/** Bit-exact PE and streaming tests for the 1:1 FP16 -> INT16 xDMA quantizer. */
class Fp16ToInt16Tester extends AnyFlatSpec with ChiselScalatestTester {

  private def f16bitsToF32(h: Int): Float = {
    val sign = if ((h & 0x8000) != 0) -1.0 else 1.0
    val exp  = (h >> 10) & 0x1f
    val mant = h & 0x3ff
    val v =
      if (exp == 0) sign * mant * math.pow(2, -24)
      else if (exp == 0x1f) if (mant == 0) sign * Double.PositiveInfinity else Double.NaN
      else sign * (1024 + mant) * math.pow(2, exp - 25)
    v.toFloat
  }

  private def f32ToF16bits(f: Float): Int = {
    val bits = java.lang.Float.floatToIntBits(f)
    val sign = (bits >>> 16) & 0x8000
    val rawe = (bits >>> 23) & 0xff
    val mant = bits & 0x7fffff
    if (rawe == 0xff) return sign | 0x7c00 | (if (mant != 0) 0x200 else 0)
    val exp = rawe - 127 + 15
    if (exp >= 0x1f) return sign | 0x7c00
    if (exp <= 0) {
      if (exp < -10) return sign
      val m       = mant | 0x800000
      val shift   = 14 - exp
      val half    = m >>> shift
      val rem     = m & ((1 << shift) - 1)
      val halfway = 1 << (shift - 1)
      return sign | (half + (if (rem > halfway || (rem == halfway && (half & 1) == 1)) 1 else 0))
    }
    var h   = sign | (exp << 10) | (mant >>> 13)
    val rem = mant & 0x1fff
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1)) h += 1
    h
  }

  private def f32bits(f: Float): BigInt =
    BigInt(java.lang.Float.floatToIntBits(f).toLong & 0xffffffffL)

  private def quantRef(h: Int, invScaleBits: BigInt): Int = {
    val x = f16bitsToF32(h).toDouble
    val s = java.lang.Float.intBitsToFloat(invScaleBits.toInt).toDouble
    val v = math.max(-32768.0, math.min(32768.0, x * s))
    math.max(-32767, math.min(32767, math.rint(v).toInt))
  }

  private def sampleFp16(rng: Random): Int =
    f32ToF16bits((rng.between(-2048.0, 2048.0) + rng.nextInt(8) * 0.125).toFloat)

  private def packFp16(lanes: Seq[Int]): BigInt =
    lanes.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (h, i)) => acc | (BigInt(h & 0xffff) << (16 * i)) }

  private def lanesOf(beat: BigInt): Seq[Int] =
    (0 until 32).map(i => ((beat >> (16 * i)) & 0xffff).toInt)

  private def packInt16(lanes: Seq[Int]): BigInt =
    lanes.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (v, i)) => acc | (BigInt(v & 0xffff) << (16 * i)) }

  private val flags = VerilatorFlags(Seq("--build-jobs", "1"))

  behavior of "Fp16ToInt16"

  it should "quantize one lane with RNE and symmetric INT16 saturation" in {
    test(new Fp16ToInt16PE).withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
      val rng = new Random(0xF1616)
      val edges = Seq(
        0x0000, 0x8000, 0x3c00, 0xbc00, 0x3800, 0xb800,
        f32ToF16bits(32767.0f), f32ToF16bits(-32767.0f), 0x7bff, 0xfbff, 0x0400
      )
      val scales = Seq(f32bits(1.0f), f32bits(0.5f), f32bits(16.0f), f32bits(32767.0f))
      for (scale <- scales; h <- edges ++ Seq.fill(400)(sampleFp16(rng))) {
        dut.io.in.poke(h.U)
        dut.io.inv_scale.poke(scale.U)
        dut.clock.step()
        val hw = dut.io.out.peek().litValue.toShort.toInt
        val sw = quantRef(h, scale)
        assert(hw == sw, f"in=0x$h%04x scale=0x${scale.toLong}%08x HW=$hw SW=$sw")
      }
    }
  }

  it should "preserve one output beat per input beat with four time-multiplexed PEs and backpressure" in {
    val rng        = new Random(0xB1616)
    val scale      = f32bits(16.0f)
    val inputBeats = Seq.fill(7)(packFp16(Seq.fill(32)(sampleFp16(rng))))
    val expected   = inputBeats.map(b => packInt16(lanesOf(b).map(h => quantRef(h, scale))))
    var actual     = Vector.empty[BigInt]

    test(new DataPathExtensionHarness(new HasFp16ToInt16(16, 16, 512, computeLanes = 4, fpPipe = 1)))
      .withAnnotations(Seq(VerilatorBackendAnnotation, flags)) { dut =>
        dut.io.csr_i(0).poke(scale.U)
        dut.io.enable_i.poke(true.B)
        dut.io.data_i.valid.poke(false.B)
        dut.io.data_o.ready.poke(false.B)
        dut.io.start_i.poke(true.B)
        dut.clock.step()
        dut.io.start_i.poke(false.B)

        var threads = new chiseltest.internal.TesterThreadList(Seq())
        threads = threads.fork {
          for (beat <- inputBeats) {
            dut.io.data_i.bits.poke(beat.U)
            dut.io.data_i.valid.poke(true.B)
            while (!dut.io.data_i.ready.peekBoolean()) dut.clock.step()
            dut.clock.step()
          }
          dut.io.data_i.valid.poke(false.B)
        }
        threads = threads.fork {
          for (i <- expected.indices) {
            while (!dut.io.data_o.valid.peekBoolean()) dut.clock.step()
            if ((i & 1) == 0) dut.clock.step(3)
            actual :+= dut.io.data_o.bits.peekInt()
            dut.io.data_o.ready.poke(true.B)
            dut.clock.step()
            dut.io.data_o.ready.poke(false.B)
          }
        }
        threads.joinAndStep()
        dut.io.data_o.ready.poke(true.B)
        var timeout = 0
        while (dut.io.busy_o.peekBoolean() && timeout < 400) { dut.clock.step(); timeout += 1 }
        assert(!dut.io.busy_o.peekBoolean(), "busy_o did not drain")
      }

    assert(actual == expected, "1:1 FP16->INT16 stream result mismatch")
  }
}
