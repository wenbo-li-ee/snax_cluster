#!/usr/bin/env python3

# Copyright 2026 KU Leuven.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path("/esat/studscratch/r1015498/Thesis/original_snax/snax_cluster")
CFG_DIR = ROOT / "target/snitch_cluster/cfg/generated"
RESULTS_PATH = ROOT / "target/snitch_cluster/generated/ewu_width_results.json"
WIDTHS = (8, 16, 32)
NUM_PE = 4
LENGTH = 64


def channels_for(width: int) -> int:
    return math.ceil(NUM_PE * width / 64)


def cfg_text(input_width: int, output_width: int) -> str:
    input_channels = channels_for(input_width)
    output_channels = channels_for(output_width)
    tcdm_ports = input_channels * 2 + output_channels

    return f"""// Copyright 2026 KU Leuven.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

{{
    nr_s1_quadrant: 1,
    s1_quadrant: {{
        nr_clusters: 1,
    }},

    cluster: {{
        name: "snax_ewu_cluster",
        bender_target: ["snax_ewu_cluster", "sparse_interconnect"],
        boot_addr: 4096,
        cluster_base_addr: 268435456,
        cluster_base_offset: 4194304,
        cluster_base_hartid: 0,
        addr_width: 48,
        data_width: 64,
        user_width: 3,
        tcdm: {{
            size: 128,
            banks: 32,
            sparse_interconnect: true,
        }},
        cluster_periph_size: 64,
        zero_mem_size: 64,
        dma_data_width: 512,
        dma_axi_req_fifo_depth: 3,
        dma_req_fifo_depth: 3,
        observable_pin_width: 8,
        narrow_trans: 4,
        wide_trans: 32,
        dma_user_width: 1,
        enable_debug: false,
        vm_support: false,
        sram_cfg_expose: true,
        sram_cfg_fields: {{
            ema: 3,
            emaw: 2,
            emas: 1
        }},

        timing: {{
            lat_comp_fp32: 3,
            lat_comp_fp64: 3,
            lat_comp_fp16: 2,
            lat_comp_fp16_alt: 2,
            lat_comp_fp8: 1,
            lat_comp_fp8_alt: 1,
            lat_noncomp: 1,
            lat_conv: 1,
            lat_sdotp: 2,
            fpu_pipe_config: "BEFORE"
            narrow_xbar_latency: "CUT_ALL_PORTS",
            wide_xbar_latency: "CUT_ALL_PORTS",
            register_core_req: true,
            register_core_rsp: true,
            register_offload_req: true,
            register_offload_rsp: true,
            register_ext_narrow: true,
            register_ext_wide: true,
        }},
        hives: [
            {{
                icache: {{
                    size: 8,
                    sets: 2,
                    cacheline: 256
                }},
                cores: [
                    {{ $ref: "#/snax_ewu_core_template" }},
                    {{ $ref: "#/dma_core_template" }},
                ]
            }}
        ]
    }},
    dram: {{
        address: 2147483648,
        length: 2147483648
    }},
    peripherals: {{
        clint: {{
            address: 4294901760,
            length: 4096
        }},
    }},
    snax_ewu_core_template: {{
        isa: "rv32ima",
        xssr: false,
        xfrep: false,
        xdma: false,
        xf16: false,
        xf16alt: false,
        xf8: false,
        xf8alt: false,
        xfdotp: false,
        xfvec: false,
        snax_acc_cfg: [{{
            snax_acc_name: "snax_ewu",
            bender_target: ["snax_ewu"],
            snax_tcdm_ports: {tcdm_ports},
            sparse_interconnect_config: [[{tcdm_ports}, 1]],
            snax_num_rw_csr: 3,
            snax_num_ro_csr: 2,
            snax_shell_wrapper_params: {{
                NumPE: {NUM_PE},
                DataWidth: {input_width},
                AddOutDataWidth: {output_width},
                MulOutDataWidth: {output_width},
            }},
            snax_streamer_cfg: {{ $ref: "#/snax_ewu_streamer_template" }}
        }}],
        snax_use_custom_ports: false,
        num_int_outstanding_loads: 1,
        num_int_outstanding_mem: 4,
        num_fp_outstanding_loads: 4,
        num_fp_outstanding_mem: 4,
        num_sequencer_instructions: 16,
        num_dtlb_entries: 1,
        num_itlb_entries: 1,
    }},
    dma_core_template: {{
        isa: "rv32ima",
        xdma: true
        xssr: false
        xfrep: false
        xf16: false,
        xf16alt: false,
        xf8: false,
        xf8alt: false,
        xfdotp: false,
        xfvec: false,
        num_int_outstanding_loads: 1,
        num_int_outstanding_mem: 4,
        num_fp_outstanding_loads: 4,
        num_fp_outstanding_mem: 4,
        num_sequencer_instructions: 16,
        num_dtlb_entries: 1,
        num_itlb_entries: 1,
    }},
    snax_ewu_streamer_template: {{
        data_reader_params: {{
            spatial_bounds: [[{input_channels}], [{input_channels}]],
            temporal_dim: [1, 1],
            num_channel: [{input_channels}, {input_channels}],
            fifo_depth: [8, 8],
        }},

        data_writer_params: {{
            spatial_bounds: [[{output_channels}]],
            temporal_dim: [1],
            num_channel: [{output_channels}],
            fifo_depth: [8],
        }},

        snax_library_name: "snax-ewu",
    }}
}}
"""


def run_cmd(cmd, log_path):
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc.stdout


def parse_sim(output: str) -> dict:
    errors = re.search(r"Number of errors:\s*(\d+)", output)
    add_cycles = re.search(r"EWU add cycles:\s*(\d+)", output)
    mul_cycles = re.search(r"EWU mul cycles:\s*(\d+)", output)
    return {
        "errors": int(errors.group(1)) if errors else None,
        "add_cycles": int(add_cycles.group(1)) if add_cycles else None,
        "mul_cycles": int(mul_cycles.group(1)) if mul_cycles else None,
    }


def main():
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for input_width in WIDTHS:
        for output_width in WIDTHS:
            cfg_rel = pathlib.Path("cfg/generated") / f"snax_ewu_i{input_width}_o{output_width}.hjson"
            cfg_path = ROOT / "target/snitch_cluster" / cfg_rel
            cfg_path.write_text(cfg_text(input_width, output_width))

            combo = {
                "input_width": input_width,
                "output_width": output_width,
                "input_channels": channels_for(input_width),
                "output_channels": channels_for(output_width),
                "tcdm_ports": channels_for(input_width) * 2 + channels_for(output_width),
                "cfg": str(cfg_rel),
                "status": "running",
            }
            print(f"=== {input_width} -> {output_width} ===", flush=True)

            try:
                stem = f"i{input_width}_o{output_width}"
                run_cmd(
                    ["make", "-C", "target/snitch_cluster", "rtl-gen", f"CFG_OVERRIDE={cfg_rel}"],
                    ROOT / "target/snitch_cluster/generated" / f"ewu_{stem}_rtl_gen.log",
                )
                run_cmd(
                    ["make", "-C", "target/snitch_cluster", "sw", f"CFG_OVERRIDE={cfg_rel}"],
                    ROOT / "target/snitch_cluster/generated" / f"ewu_{stem}_sw.log",
                )
                run_cmd(
                    [
                        "make",
                        "-C",
                        "target/snitch_cluster/sw/apps/snax-ewu",
                        "clean",
                    ],
                    ROOT / "target/snitch_cluster/generated" / f"ewu_{stem}_app_clean.log",
                )
                run_cmd(
                    [
                        "make",
                        "-C",
                        "target/snitch_cluster/sw/apps/snax-ewu",
                        "all",
                        f"CFG_OVERRIDE={cfg_rel}",
                        f"INPUT_WIDTH={input_width}",
                        f"OUTPUT_WIDTH={output_width}",
                        f"NUM_PE={NUM_PE}",
                        f"LENGTH={LENGTH}",
                    ],
                    ROOT / "target/snitch_cluster/generated" / f"ewu_{stem}_app.log",
                )
                run_cmd(
                    [
                        "make",
                        "-C",
                        "target/snitch_cluster",
                        "bin/snitch_cluster.vlt",
                        "-j4",
                        f"CFG_OVERRIDE={cfg_rel}",
                    ],
                    ROOT / "target/snitch_cluster/generated" / f"ewu_{stem}_hw.log",
                )
                sim_output = run_cmd(
                    [
                        "./target/snitch_cluster/bin/snitch_cluster.vlt",
                        "./target/snitch_cluster/sw/apps/snax-ewu/build/snax-ewu.elf",
                    ],
                    ROOT / "target/snitch_cluster/generated" / f"ewu_{stem}_sim.log",
                )
                combo.update(parse_sim(sim_output))
                combo["status"] = "pass" if combo["errors"] == 0 else "functional_fail"
            except subprocess.CalledProcessError as exc:
                combo["status"] = "build_fail"
                combo["failed_cmd"] = exc.cmd
                combo["failure_excerpt"] = "\n".join(exc.output.splitlines()[-20:])

            results.append(combo)
            RESULTS_PATH.write_text(json.dumps(results, indent=2))
            print(json.dumps(combo, indent=2), flush=True)

    failures = [result for result in results if result["status"] != "pass"]
    print(json.dumps({"results_path": str(RESULTS_PATH), "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
