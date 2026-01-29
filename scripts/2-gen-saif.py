import os
import subprocess
import sys


if len(sys.argv) < 2:
    print("Error: You must provide the Run Tag as an argument!")
    print("Usage: python 2-gen-saif.py <RUN_TAG>")
    sys.exit(1)

FILENAME = sys.argv[1]  # Takes the argument passed from Bash

DESIGN_CONFIG = {
    "picorv32": {
        "tb_file": "testbench.v",
        "vcd_file": "testbench.vcd",
        "needs_firmware": True
    },
    "aes": {
        "tb_file": "tb_aes.v",
        "vcd_file": "tb_aes.vcd", 
        "needs_firmware": False 
    },
        "ethmac": {
        "tb_file": "tb_eth_top.v",    
        "vcd_file": "testbench.vcd",  # 
        "needs_firmware": False
    } ,
    "sha256": {
        "tb_file": "tb_sha256.v",
        "vcd_file": "tb_sha256.vcd",
        "needs_firmware": False
    }
}



# paths
PROJECT_ROOT = os.getcwd()
RUN_DIR = os.path.join(PROJECT_ROOT, "runs", FILENAME)
PLACEMENT_DIR = os.path.join(RUN_DIR, "11-openroad-detailedplacement")

DESIGN_NAME = FILENAME.split("_")[0]
DESIGN_SRC_DIR = os.path.join(PROJECT_ROOT, "designs", DESIGN_NAME)


# (Required for full testbench)
FIRMWARE_PATH = os.path.join(DESIGN_SRC_DIR, "firmware", "firmware.hex")

current_config = DESIGN_CONFIG[DESIGN_NAME]
NETLIST_PATH = os.path.join(PLACEMENT_DIR, f"{DESIGN_NAME}.nl.v")
TESTBENCH_PATH = os.path.join(DESIGN_SRC_DIR, "tb", current_config["tb_file"])
PRIMITIVES_PATH = os.path.join(PROJECT_ROOT , "designs" , "primitives.v")
SKY130_PATH = os.path.join(PROJECT_ROOT,  "designs", "sky130_fd_sc_hd.v")
WAVE2SAIF_PATH = os.path.join(PROJECT_ROOT,  "wave2saif") 

# # Output files (Saved inside the placement folder for organization)
SIM_EXEC = os.path.join(RUN_DIR, "sim_gate.out")
VCD_FILE = os.path.join(RUN_DIR, current_config["vcd_file"])
SAIF_FILE = os.path.join(RUN_DIR, f"{DESIGN_NAME}.saif")



def run_command(cmd, cwd=None):
    """Helper to run shell commands and print output"""
    print(f"[RUNNING]: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)
        print(result.stdout) # Uncomment for verbose logs
    except subprocess.CalledProcessError as e:
        print(f"[ERROR]: Command failed!\n{e.stderr}")
        sys.exit(1)



# Define include directories based on your file structure
TB_INCLUDE = os.path.join(DESIGN_SRC_DIR, "tb")
RTL_INCLUDE = os.path.join(DESIGN_SRC_DIR, "rtl") 
TB_COP_PATH = os.path.join(DESIGN_SRC_DIR, "tb", "tb_cop.v")

# Missing module definitions
ETH_COP_PATH = os.path.join(DESIGN_SRC_DIR, "rtl", "eth_cop.v") # Defines eth_cop


# Start with the base command components
iverilog_cmd = [
    "iverilog",
    "-o", SIM_EXEC,
    "-DFUNCTIONAL",
    "-DUNIT_DELAY=#1"
]

# Add include paths ONLY for ethmac to keep other designs robust
if DESIGN_NAME == "ethmac":
    iverilog_cmd.extend([
        "-I", TB_INCLUDE, 
        "-I", RTL_INCLUDE
    ])

# append if ethmac
iverilog_cmd.extend([
    TESTBENCH_PATH,   
    NETLIST_PATH,     
    PRIMITIVES_PATH,  
    SKY130_PATH       
])


# 1. Compile with Iverilog
run_command(iverilog_cmd)


# Make  wave2saif is executable
if os.path.exists(WAVE2SAIF_PATH):
    subprocess.run(["chmod", "+x", WAVE2SAIF_PATH])
else:
    print(f"[ERROR] wave2saif not found at {WAVE2SAIF_PATH}")
    sys.exit(1)


#  Run vvp to get vcd
vvp_cmd = ["vvp", SIM_EXEC , "+vcd"]


if current_config["needs_firmware"]:
    if not os.path.exists(FIRMWARE_PATH):
        print(f"❌ Error: Firmware not found at {FIRMWARE_PATH}")
        sys.exit(1)
    vvp_cmd.append(f"+firmware={FIRMWARE_PATH}")

run_command(vvp_cmd, cwd=RUN_DIR)

# VCD to SAIF
wave2saif_cmd = [
    WAVE2SAIF_PATH,
    "-o", SAIF_FILE,
    VCD_FILE
]
run_command(wave2saif_cmd)


