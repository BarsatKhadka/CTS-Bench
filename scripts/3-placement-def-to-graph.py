import re

filename = "picorv32_run_20260107_145745"
design_name = filename.split("_")[0]

def load_file(path):
    with open(path, "r") as f:
        return f.read()

#the whole def file loaded
def_text = load_file(path = f"./runs/{filename}/11-openroad-detailedplacement/{design_name}.def")


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

all_flops = re.findall(r'\(\s+((?!PIN)\S+)\s+CLK\s+\)', raw_block)


first_entry = all_flops[0]

print(first_entry)


import re

flop_data = {}
lines = def_text.splitlines()

# Using a set of target strings for exact matching in the COMPONENTS section
targets = {f"- {flop} " for flop in all_flops}

#find coordinates for each flop
for line in lines:
    if "PLACED" in line:
        for target in targets:
            if target in line:
                # 1. Extract the name from the target string
                clean_name = target.strip("- ").strip()
                
                # 2. Extract X and Y using regex
                coord_match = re.search(r'\(\s+(\d+)\s+(\d+)\s+\)', line)
                if coord_match:
                    x = int(coord_match.group(1))
                    y = int(coord_match.group(2))
                    
                    # 3. Store the tuple of integers
                    flop_data[clean_name] = {
                        "coords": (x, y),
                        "is_flip_flop": True
                    }
                
                break 


d_pattern = rf'-\s+\S+\s+[^;]*?\(\s+{re.escape(first_entry)}\s+D\s+\)[^;]*?;'
d_match = re.search(d_pattern, def_text, re.DOTALL)
d_line = " ".join(d_match.group(0).split())

q_pattern = rf'-\s+\S+\s+[^;]*?\(\s+{re.escape(first_entry)}\s+Q\s+\)[^;]*?;'
q_match = re.search(q_pattern, def_text, re.DOTALL)
q_line = " ".join(q_match.group(0).split())

d_instances = re.findall(r'\(\s+(\S+)\s+\S+\s+\)', d_line)
q_instances = re.findall(r'\(\s+(\S+)\s+\S+\s+\)', q_line)


fan_in = [inst for inst in d_instances if inst != first_entry and re.match(r'_\d+_', inst)]
fan_out = [inst for inst in q_instances if inst != first_entry and re.match(r'_\d+_', inst)]

neighbor_targets = set(fan_in + fan_out)
logic_map = {}

# Re-scanning the lines for these specific target names
for line in lines:
    if "PLACED" in line:
        for gate_name in neighbor_targets:
            # For eg: We look for "- _09865_ " to ensure an exact match
            if f"- {gate_name} " in line:
                coord_match = re.search(r'\(\s+(\d+)\s+(\d+)\s+\)', line)
                if coord_match:
                    x = int(coord_match.group(1))
                    y = int(coord_match.group(2))
                    
                    logic_map[gate_name] = {
                        "coords": (x, y),
                        "is_flip_flop": False
                    }
                break 



print(logic_map)

#for every flipflop and logic , extract their saif toggle count 
SAIF_FILE_LOCATION = f"./runs/{filename}/{design_name}.saif"
SAIF_FILE = load_file(SAIF_FILE_LOCATION)


targets = set(logic_map.keys()) | set(flop_data.keys())

print(f"--- Extracting Data-Only Toggle Counts for {len(targets)} Instances ---")

# Pre-compile regex for speed
start_pattern = re.compile(r'\(INSTANCE\s+([a-zA-Z0-9_]+)')
tc_pattern = re.compile(r'\(TC\s+(\d+)\)')

# 2. Scan the SAIF file string once
for match in start_pattern.finditer(SAIF_FILE):
    gate_name = match.group(1)
    
    # Only process if this instance is in our target list
    if gate_name in targets:
        
        # --- A. ISOLATE THE BLOCK ---
        start_index = match.start()
        current_index = start_index
        balance = 0
        
        # Find the closing parenthesis for this instance block
        while current_index < len(SAIF_FILE):
            char = SAIF_FILE[current_index]
            if char == '(':
                balance += 1
            elif char == ')':
                balance -= 1
                if balance == 0:
                    break
            current_index += 1
            
        # Get the full text block
        full_block = SAIF_FILE[start_index : current_index + 1]
        
        # --- B. EXTRACT MAX TC (IGNORING CLOCK) ---
        max_tc = 0
        
        for line in full_block.splitlines():
            # 1. Look for a Toggle Count on this line
            tc_match = tc_pattern.search(line)
            if tc_match:
                # 2. SAFETY CHECK: Ignore Clock lines
                if "CLK" in line.upper() or "CLOCK" in line.upper():
                    continue 
                
                # 3. Find Max
                tc_val = int(tc_match.group(1))
                if tc_val > max_tc:
                    max_tc = tc_val
        
        # --- C. UPDATE THE DICTIONARIES ---
        # We don't know which dict it belongs to, so we try both.
        # This is safe because keys are unique across the design.
        if gate_name in logic_map:
            logic_map[gate_name]['toggle_count'] = max_tc
        elif gate_name in flop_data:
            flop_data[gate_name]['toggle_count'] = max_tc


# print(d_line)
# print(q_line)


    # print(q_line)


# nets_start = def_text.find("NETS")
# nets_end = def_text.find("END NETS")
# nets_section = def_text[nets_start:nets_end]

# print(nets_section)



# for name, data in flop_data.items():
#     coords = data["coords"]
#     print(f"Flop: {name:10} | Coordinates: X={coords[0]:<8} Y={coords[1]:<8} |  is_flip_flop: {data['is_flip_flop']}")


# print(def_text)