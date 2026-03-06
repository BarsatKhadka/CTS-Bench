import glob
import os
import random
import json
import subprocess
from datetime import datetime
from openlane.flows import SequentialFlow
from openlane.steps import Yosys, OpenROAD, Odb, Misc


class FullPlacementFlow(SequentialFlow):
    Steps = [
        Yosys.JsonHeader,
        Yosys.Synthesis,
        Misc.LoadBaseSDC,
        OpenROAD.Floorplan,
        OpenROAD.TapEndcapInsertion,
        OpenROAD.IOPlacement,
        Odb.SetPowerConnections,
        OpenROAD.GeneratePDN,
        OpenROAD.GlobalPlacement,
        OpenROAD.RepairDesign,
        OpenROAD.DetailedPlacement,
        OpenROAD.STAPrePNR
    ]


MY_PDK_ROOT = "/home/rain/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af"
os.environ["PDK_ROOT"] = MY_PDK_ROOT
LIB_PATH = f"{MY_PDK_ROOT}/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"


def extract_timing_paths(run_dir, design_name):
    final_dir = os.path.join(run_dir, "final")

    def find_file(subdir, ext):
        d = os.path.join(final_dir, subdir)
        if not os.path.exists(d):
            return None
        for f in sorted(os.listdir(d)):
            if f.endswith(ext) and os.path.isfile(os.path.join(d, f)):
                return os.path.join(d, f)
        return None

    netlist = find_file("nl", ".v") or find_file("pnl", ".v")
    sdc_file = find_file("sdc", ".sdc")
    spef_file = find_file("spef", ".spef")

    if not netlist or not sdc_file:
        print(f"Missing netlist or SDC in {final_dir}")
        return None

    print(f"  Netlist: {netlist}")
    print(f"  SDC:     {sdc_file}")
    print(f"  SPEF:    {spef_file}")

    pdk_base = f"{MY_PDK_ROOT}/sky130A/libs.ref/sky130_fd_sc_hd"
    tech_lef = f"{MY_PDK_ROOT}/sky130A/libs.ref/sky130_fd_sc_hd/techlef/sky130_fd_sc_hd__nom.tlef"
    cell_lef = f"{MY_PDK_ROOT}/sky130A/libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.lef"

    output_csv = os.path.join(run_dir, "timing_paths.csv")
    tcl_path = os.path.join(run_dir, "extract_paths.tcl")
    spef_cmd = f"read_spef {spef_file}" if spef_file else ""

    tcl = """
read_lef {tech_lef}
read_lef {cell_lef}
read_liberty {lib}
read_verilog {netlist}
link_design {design}
read_sdc {sdc}
{spef}

set fp [open "{output}" w]
puts $fp "launch_flop,capture_flop,slack"

set paths [find_timing_paths -path_delay max \
    -group_count 50000 \
    -unique_paths_to_endpoint]

puts "Found [llength $paths] setup paths"

foreach path $paths {{
    set sp [get_property $path startpoint]
    set ep [get_property $path endpoint]
    if {{[get_property $sp is_port] || [get_property $ep is_port]}} continue
    set sp_name [join [lrange [split [get_property $sp full_name] "/"] 0 end-1] "/"]
    set ep_name [join [lrange [split [get_property $ep full_name] "/"] 0 end-1] "/"]
    puts $fp "$sp_name,$ep_name,[get_property $path slack]"
}}

set hold_paths [find_timing_paths -path_delay min \
    -group_count 50000 \
    -unique_paths_to_endpoint]

puts "Found [llength $hold_paths] hold paths"

foreach path $hold_paths {{
    set sp [get_property $path startpoint]
    set ep [get_property $path endpoint]
    if {{[get_property $sp is_port] || [get_property $ep is_port]}} continue
    set sp_name [join [lrange [split [get_property $sp full_name] "/"] 0 end-1] "/"]
    set ep_name [join [lrange [split [get_property $ep full_name] "/"] 0 end-1] "/"]
    puts $fp "$sp_name,$ep_name,[get_property $path slack]"
}}

close $fp
puts "Done writing timing paths"
exit
""".format(
        tech_lef=tech_lef,
        cell_lef=cell_lef,
        lib=LIB_PATH,
        netlist=netlist,
        design=design_name,
        sdc=sdc_file,
        spef=spef_cmd,
        output=output_csv
    )
    with open(tcl_path, "w") as f:
        f.write(tcl)

    print(f"  Running: openroad -exit {tcl_path}")

    result = subprocess.run(
        ["openroad", "-exit", tcl_path],
        capture_output=True, text=True
    )

    print(f"  Return code: {result.returncode}")
    print(f"  STDOUT (last 3000 chars):\n{result.stdout[-3000:]}")
    print(f"  STDERR (last 3000 chars):\n{result.stderr[-3000:]}")

    if result.returncode != 0:
        print("STA extraction failed.")
        return None

    if os.path.exists(output_csv):
        n = sum(1 for _ in open(output_csv)) - 1
        print(f"Extracted {n} flop pairs -> {output_csv}")
    else:
        print("Output CSV was not created.")
        return None

    return output_csv


def run_single_experiment(design_name, clock_period, clock_port):
    if design_name in ["aes", "sha256", "ethmac"]:
        verilog_files = glob.glob(f"./designs/{design_name}/rtl/*.v")
    elif design_name in ["picorv32", "zipdiv"]:
        verilog_files = [f"./designs/{design_name}/rtl/{design_name}.v"]

    io_mode = random.choice([0, 1])
    core_util = random.randint(40, 70)
    min_required_density = (core_util / 100.0) + 0.05
    pl_density = round(min_required_density + random.uniform(0.0, 0.20), 2)
    if pl_density > 0.99:
        pl_density = 0.99

    fp_ratio = random.choice([0.7, 1.0, 1.4, 2.0])
    synth_strategy = random.choice([
        "AREA 0", "AREA 1", "AREA 2",
        "DELAY 0", "DELAY 1", "DELAY 2", "DELAY 3", "DELAY 4"
    ])
    time_driven = random.choice([True, False])
    routability_driven = random.choice([True, False])

    config = {
        "DESIGN_NAME": design_name,
        "VERILOG_FILES": verilog_files,
        "CLOCK_PERIOD": clock_period,
        "CLOCK_PORT": clock_port,
        "FALLBACK_SDC_FILE": "./designs/base.sdc",
        "SYNTH_STRATEGY": synth_strategy,
        "FP_ASPECT_RATIO": fp_ratio,
        "FP_CORE_UTIL": core_util,
        "FP_IO_MODE": io_mode,
        "PL_TARGET_DENSITY": pl_density,
        "PL_RANDOM_GLB_PLACEMENT": True,
        "PL_TIME_DRIVEN": time_driven,
        "PL_ROUTABILITY_DRIVEN": routability_driven
    }

    tag = f"{design_name}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    flow = FullPlacementFlow(config=config, design_dir=".", pdk="sky130A")
    flow.start(tag=tag)

    run_dir = f"./runs/{tag}"

    with open("latest_run.txt", "w") as f:
        f.write(tag + "\n")

    stats = {
        "design_name": design_name,
        "aspect_ratio": fp_ratio,
        "core_util": core_util,
        "density": pl_density,
        "synth_strategy": synth_strategy,
        "io_mode": io_mode,
        "time_driven": int(time_driven),
        "routability_driven": int(routability_driven)
    }
    with open(os.path.join(run_dir, "placement_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    with open("latest_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # Extract timing paths right after placement
    print("\n--- Extracting timing paths ---")
    extract_timing_paths(run_dir, design_name)

    print(f"Done: {tag}")


run_single_experiment("picorv32", 5.0, "clk")