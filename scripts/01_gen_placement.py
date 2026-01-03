import os
from pathlib import Path as SysPath
from openlane.common import Path    # Required for OpenLane 2
from openlane.flows import SequentialFlow
from openlane.steps import OpenROAD
from openlane.state import State, DesignFormat


MY_PDK_ROOT = "/home/rain/.volare/volare/sky130/versions/0fe599b2afb6708d281543108caf8310912f54af"
os.environ["PDK_ROOT"] = MY_PDK_ROOT

#loading the base snapshot as State to continue placement from there 
def load_snapshot(base_tag):
    final_dir = SysPath(f"./runs/{base_tag}/final")
    
    state_data = {}
    folder_map = {
        "odb": DesignFormat.ODB,
        "def": DesignFormat.DEF,
        "nl":  DesignFormat.NETLIST,           
        "pnl": DesignFormat.POWERED_NETLIST,   
        "json_h": DesignFormat.JSON_HEADER,    
        "sdc": DesignFormat.SDC,
        "sdf": DesignFormat.SDF,               
        "spef": DesignFormat.SPEF,             
    }

    #goes through each folder in the final directory of the base and maps the files to state data 
    for folder, fmt in folder_map.items():
        target = final_dir / folder
        if target.exists():
            files = [f for f in target.glob("*") if f.is_file()]
            if files:
                state_data[fmt] = Path(str(files[0].resolve()))
                print(f"   [FOUND] {folder.upper()}: {files[0].name}")

    if not state_data:
        raise ValueError("Snapshot empty!")
        
    return State(state_data)

class PlacementFlow(SequentialFlow):
    #run placement steps onwards from the loaded base snapshot
    Steps = [
        OpenROAD.GlobalPlacement,
        OpenROAD.RepairDesign,  
        OpenROAD.STAMidPNR,             
        OpenROAD.DetailedPlacement,
       
    ]

def run_placement_from_base(DESIGN, clock_period, clock_port):
    BASE = DESIGN + "base"
    try:
        base_state = load_snapshot(BASE)
        config = {
            "DESIGN_NAME": DESIGN,
            "PDK": "sky130A",
            "STD_CELL_LIBRARY": "sky130_fd_sc_hd",
            "SDC_FILE": "./designs/base.sdc",
            "FP_CORE_UTIL": 50,
            "CLOCK_PERIOD": clock_period,
            "CLOCK_PORT": clock_port,
        }

        flow = PlacementFlow(config=config, design_dir=".")
        
        flow.start(
            tag=DESIGN + "placement_from_base",
            with_initial_state=base_state   #this is where we load the base snapshot
        )
        
    except Exception as e:
        print(f"\nCRITICAL FAIL: {e}")

if __name__ == "__main__":
    run_placement_from_base("picorv32", 10.0, "clk")
    run_placement_from_base("aes", 20.0, "clk")
    run_placement_from_base("sha256", 24.0, "clk")
    run_placement_from_base("ethmac", 70, "wb_clk_i")