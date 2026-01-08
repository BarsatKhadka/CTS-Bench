import re

filename = "picorv32_run_20260107_145745"
design_name = filename.split("_")[0]

def load_def():
    path = f"./runs/{filename}/11-openroad-detailedplacement/{design_name}.def"
    with open(path, "r") as f:
        return f.read()

#the whole def file loaded
def_text = load_def()

clock_port = "clk"

#this will parse all the instances of the clock net from the DEF file
def get_all_clock_consuming_flops(def_text, clock_name):

    pattern = rf'-\s+{re.escape(clock_name)}\s+\(\s+PIN\s+{re.escape(clock_name)}\s+\).*?;'
    match = re.search(pattern, def_text, re.DOTALL)
    
    if match:
        return match.group(0)
    return KeyError("Clock net not found in DEF file.")


raw_block = get_all_clock_consuming_flops(def_text, clock_port)

# print("--- Strict Clock Net Block ---")
# print(raw_block)

instances = re.findall(r'\(\s+((?!PIN)\S+)\s+CLK\s+\)', raw_block)
print(len(instances))


# print(f"Total connections found: {len(instances)}")
# print(f"HashSet (unique instances) size: {len(flop_set)}")




# print(def_text)