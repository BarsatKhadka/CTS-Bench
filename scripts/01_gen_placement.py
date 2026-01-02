import os
from openlane.flows import SequentialFlow
from openlane.steps import Yosys





# 1. Flow until placement. After Placement we run cts 10 times on the same placement.
class flow_till_placement(SequentialFlow):
    Steps = [
        
    ]

# 2. Config 
config = {
    "DESIGN_NAME": "picorv32",
    "VERILOG_FILES": ["./designs/src/picorv32a/picorv32a.v"],
    "CLOCK_PORT": "clk",
    "CLOCK_PERIOD": 16.0
}

# 3. Run 
def run_test():
    print("--- Testing Yosys Synthesis Link ---")
    
    flow = flow_till_placement(
        config=config,
        design_dir=".",
        pdk="sky130A"
    )
    
    flow.start()
    print("Success")

if __name__ == "__main__":
    run_test()