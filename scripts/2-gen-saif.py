import os
import subprocess
import sys


# Change this to the specific run folder you are processing
FILENAME = "picorv32_run_20260107_145745" 

# paths
PROJECT_ROOT = os.getcwd()
RUN_DIR = os.path.join(PROJECT_ROOT, "runs", FILENAME)
PLACEMENT_DIR = os.path.join(RUN_DIR, "11-openroad-detailedplacement")

DESIGN_NAME = FILENAME.split("_")[0]
DESIGN_SRC_DIR = os.path.join(PROJECT_ROOT, "designs", DESIGN_NAME)


# (Required for full testbench)
FIRMWARE_PATH = os.path.join(DESIGN_SRC_DIR, "firmware", "firmware.hex")


NETLIST_PATH = os.path.join(PLACEMENT_DIR, f"{DESIGN_NAME}.nl.v")
TESTBENCH_PATH = os.path.join(DESIGN_SRC_DIR, "tb", "testbench.v") 
PRIMITIVES_PATH = os.path.join(PROJECT_ROOT , "designs" , "primitives.v")
SKY130_PATH = os.path.join(PROJECT_ROOT,  "designs", "sky130_fd_sc_hd.v")
WAVE2SAIF_PATH = os.path.join(PROJECT_ROOT,  "wave2saif") 

# # Output files (Saved inside the placement folder for organization)
SIM_EXEC = os.path.join(RUN_DIR, "sim_gate.out")
VCD_FILE = os.path.join(RUN_DIR, "testbench.vcd") 
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



if not os.path.exists(FIRMWARE_PATH):
    print(f"[ERROR] Firmware not found at: {FIRMWARE_PATH}")
    sys.exit(1)

iverilog_cmd = [
    "iverilog",
    "-o", SIM_EXEC,
    "-DFUNCTIONAL",
    "-DUNIT_DELAY=#1",
    TESTBENCH_PATH,
    NETLIST_PATH,
    PRIMITIVES_PATH,
    SKY130_PATH
]

# 1. Compile with Iverilog
run_command(iverilog_cmd)


# Make  wave2saif is executable
if os.path.exists(WAVE2SAIF_PATH):
    subprocess.run(["chmod", "+x", WAVE2SAIF_PATH])
else:
    print(f"[ERROR] wave2saif not found at {WAVE2SAIF_PATH}")
    sys.exit(1)


#  Run vvp to get vcd
vvp_cmd = ["vvp", SIM_EXEC , "+vcd", f"+firmware={FIRMWARE_PATH}"]
run_command(vvp_cmd, cwd=RUN_DIR)

# VCD to SAIF
wave2saif_cmd = [
    WAVE2SAIF_PATH,
    "-o", SAIF_FILE,
    VCD_FILE
]
run_command(wave2saif_cmd)


