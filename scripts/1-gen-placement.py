import glob
import os
import random
import json

from sympy import ff
from openlane.flows import SequentialFlow
from openlane.steps import Yosys, OpenROAD, Odb, Misc 
from datetime import datetime



# The full pipeline from RTL to Placement
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

def run_single_experiment(design_name, clock_period, clock_port):
    # getting all verilog files for the config
    if design_name in ["aes", "sha256", "ethmac"]:
        verilog_files = glob.glob(f"./designs/{design_name}/rtl/*.v")
    elif design_name == "picorv32":
        verilog_files = [f"./designs/{design_name}/rtl/{design_name}.v"]

    #randomizing some parameters
    io_mode = random.choice([0, 1]) 

    core_util = random.randint(40, 70)
    min_required_density = (core_util / 100.0) + 0.05
    pl_density = round(min_required_density + random.uniform(0.0, 0.20), 2)
    if pl_density > 0.99:
        pl_density = 0.99

    SAFE_RATIOS = [0.7, 1.0, 1.4, 2.0]
    fp_ratio = random.choice(SAFE_RATIOS)

    synth_strategies = [
        "AREA 0", "AREA 1", "AREA 2",
        "DELAY 0", "DELAY 1", "DELAY 2", "DELAY 3", "DELAY 4"
    ]
    synth_strategy = random.choice(synth_strategies)
    time_driven_bool = random.choice([True, False])
    routability_driven_bool = random.choice([True, False])

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
        "PL_TIME_DRIVEN": time_driven_bool,
        "PL_ROUTABILITY_DRIVEN": routability_driven_bool
    }

    flow = FullPlacementFlow(
        config=config,
        design_dir=".",
        pdk="sky130A"
    )
    
    # Each run gets a unique tag
    tag = f"{design_name}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    flow.start(tag=tag)

    with open("latest_run.txt", "w") as f:
        f.write(tag + "\n")
        
    print(f"📄 Saved run tag to latest_run.txt: {tag}")
    print(f"{fp_ratio}")
    
    stats = {
        "design_name": design_name,
        "aspect_ratio": fp_ratio,
        "core_util": core_util,
        "density": pl_density,
        "synth_strategy": synth_strategy,
        "io_mode": io_mode,
        "time_driven": int(time_driven_bool),        # Save as 1/0
        "routability_driven": int(routability_driven_bool) # Save as 1/0
    }

    with open("latest_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

# run_single_experiment("picorv32", 5.0, "clk")
# run_single_experiment("aes", 7.0, "clk")
# run_single_experiment("sha256", 24.0, "clk")
run_single_experiment("ethmac", 10.0, "wb_clk_i")