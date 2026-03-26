package snax.DataPathExtension

import chisel3._

class HasVerilogMinusOne(dataWidth: Int = 512) extends HasDataPathExtension {
  implicit val extensionParam:          DataPathExtensionParam =
    new DataPathExtensionParam(
      moduleName = "VerilogMinusOne",
      userCsrNum = 1,
      dataWidth  = dataWidth
    )
  def instantiate(clusterName: String): SystemVerilogDataPathExtension =
    Module(
      new SystemVerilogDataPathExtension(
        topmodule = "VerilogMinusOne",
        dir       = "src/main/systemverilog/VerilogMinusOne"
      ) {
        override def desiredName = clusterName + namePostfix
      }
    )
}
