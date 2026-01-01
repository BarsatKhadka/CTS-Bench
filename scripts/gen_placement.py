import os
from openlane.flows import SequentialFlow
from openlane.steps import Yosys

# 1. Test Flow that does ONLY Synthesis
class SynthesisOnlyFlow(SequentialFlow):
    Steps = [
        Yosys.Synthesis
    ]

# 2. Config 
config = {
    "DESIGN_NAME": "picorv32",
    "VERILOG_FILES": ["./designs/src/picorv32a.v"],
    "CLOCK_PORT": "clk",
    "CLOCK_PERIOD": 10.0
}

# 3. Run 
def run_test():
    print("--- Testing Yosys Synthesis Link ---")
    
    flow = SynthesisOnlyFlow(
        config=config,
        design_dir=".",
        pdk="sky130A"
    )
    
    flow.start()
    print("Success")

if __name__ == "__main__":
    run_test()