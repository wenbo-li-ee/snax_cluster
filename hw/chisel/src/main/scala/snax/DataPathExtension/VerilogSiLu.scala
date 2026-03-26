package snax.DataPathExtension

import chisel3._

class HasVerilogSiLu(dataWidth: Int = 512) extends HasDataPathExtension {
  implicit val extensionParam:          DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "VerilogSiluNew",
      userCsrNum = 1,
      dataWidth  = dataWidth
    )
  def instantiate(clusterName: String): SystemVerilogDataPathExtension =
    Module(
      new SystemVerilogDataPathExtension(
        topmodule = "VerilogSiluNew",
        filelist = Seq(
          "src/main/systemverilog/VerilogSiluNew/silu_hp32_q22_pkg.sv",
          "src/main/systemverilog/VerilogSiluNew/horner_stage.sv",
          "src/main/systemverilog/VerilogSiluNew/param_selector.sv",
          "src/main/systemverilog/VerilogSiluNew/partition_detector.sv",
          "src/main/systemverilog/VerilogSiluNew/silu_top.sv",
          "src/main/systemverilog/VerilogSiluNew/VerilogSiluNew.sv"
        )
      ) {
        override def desiredName = clusterName + namePostfix
      }
    )
}
