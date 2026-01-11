from extract_placement_def_to_dict import process_design
import random

FILENAME = "picorv32_run_20260107_145745" 

design_data = process_design(FILENAME, clock_port="clk")

sample_keys = random.sample(list(design_data.keys()), 10)

for k in sample_keys:
    print(k, design_data[k])