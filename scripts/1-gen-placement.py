import glob
import os
import random
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
    pl_density = round(random.uniform(0.55, 0.85), 2)
    io_mode = random.choice([0, 1]) 
    core_util = random.randint(45, 65)

    config = {
        "DESIGN_NAME": design_name,
        "VERILOG_FILES": verilog_files,
        "CLOCK_PERIOD": clock_period,
        "CLOCK_PORT": clock_port,
        "FALLBACK_SDC_FILE": "./designs/base.sdc",
        
        
        "FP_CORE_UTIL": core_util,
        "FP_IO_MODE": io_mode,      
        "PL_TARGET_DENSITY": pl_density,
        "PL_RANDOM_GLB_PLACEMENT": True,  
        "PL_TIME_DRIVEN": random.choice([True, False]),
        "PL_ROUTABILITY_DRIVEN": random.choice([True, False]),
    }

    flow = FullPlacementFlow(
        config=config,
        design_dir=".",
        pdk="sky130A"
    )
    
    # Each run gets a unique tag
    tag = f"{design_name}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    flow.start(tag=tag)

run_single_experiment("picorv32", 10.0, "clk")
# run_single_experiment("aes", 20.0, "clk")
# run_single_experiment("sha256", 24.0, "clk")
# run_single_experiment("ethmac", 70.0, "wb_clk_i")