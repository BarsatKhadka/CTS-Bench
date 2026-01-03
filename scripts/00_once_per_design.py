import glob
import os
from tkinter import Misc
from openlane.flows import SequentialFlow
from openlane.steps import Yosys, OpenROAD, Odb, Misc 


# Generated Once per DESIGN , never varied. 
class once_per_design(SequentialFlow):
    Steps = [
        Yosys.JsonHeader,
        #Synthesis of a design is deterministic and does not need to be repeated.
        Yosys.Synthesis,

        #SDC files remain the same
        Misc.LoadBaseSDC,

        #Floorplan remains the same. Because If its not , then the experiment turns into a floorplan + placement problem
        OpenROAD.Floorplan,
        
        #Insert and tap and tie cells on the design to ensure chip's funcionality
        OpenROAD.TapEndcapInsertion,

        #Making IO placements fixed
        OpenROAD.IOPlacement,

        #Same Power Distribution Network
        Odb.SetPowerConnections,
        OpenROAD.GeneratePDN,

        # Sanity Check. Checks before we start the heavy experiments
        OpenROAD.STAPrePNR

    ]


MY_PDK_ROOT = "/home/rain/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af"
os.environ["PDK_ROOT"] = MY_PDK_ROOT

def once_per_design_flow(design_name, clock_period , clock_port):

    if design_name == "aes" or design_name == "sha256" or design_name == "ethmac":
        # GRAB ALL FILES in the rtl folder
        verilog_files = glob.glob(f"./designs/{design_name}/rtl/*.v")
    elif design_name == "picorv32":
        verilog_files = [f"./designs/{design_name}/rtl/{design_name}.v"]


    config = {
    "DESIGN_NAME": design_name,
    "VERILOG_FILES": verilog_files,

    #openlane takes these parameters and sets env in sdc file
    "CLOCK_PERIOD": clock_period ,
    "CLOCK_PORT": clock_port,

    "FALLBACK_SDC_FILE": "./designs/base.sdc",  #Point to our own sdc
    "FP_CORE_UTIL": 50,       
    "FP_ASPECT_RATIO": 1,     # Square 
   

    #Everything else remains default till placement. Since the pdk and tools are fixed , rest of config will not vary either.
}

    flow = once_per_design(
        config=config,
        design_dir=".",
        pdk="sky130A"
    )
    flow.start(tag=design_name+"base")


once_per_design_flow("picorv32" , 10.0 , "clk")
once_per_design_flow("aes" , 20.0 , "clk")
once_per_design_flow("sha256" , 24.0 , "clk")
once_per_design_flow("ethmac" , 70 , "wb_clk_i")
