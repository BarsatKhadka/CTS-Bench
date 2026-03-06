import os
import shutil
import subprocess
import csv
import json
from xxlimited import new

# --- CONFIGURATION ---
DATASET_ROOT = "../dataset_with_def"
LOG_FILE = os.path.join(DATASET_ROOT, "experiment_log.csv")
KEPT_FILES_DIR = os.path.join(DATASET_ROOT, "placement_files")

VENV_PYTHON = "./venv/bin/python3"
NUM_ITERATIONS = 122

DESIGN_CONFIG = {
    "aes":      {"clock_period": 7.0, "clock_port": "clk"},
    "picorv32": {"clock_period": 5.0, "clock_port": "clk"},
    "sha256":   {"clock_period": 9.0, "clock_port": "clk"},
    "ethmac":   {"clock_period": 9.0, "clock_port": "wb_clk_i"},
    "zipdiv":   {"clock_period": 5.0, "clock_port": "i_clk"},
}



def setup_directories():
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(KEPT_FILES_DIR, exist_ok=True)

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            header = [
                "run_id", "placement_id", "design_name",
                "aspect_ratio", "core_util", "density", "synth_strategy",
                "io_mode", "time_driven", "routability_driven",
                "cts_max_wire", "cts_buf_dist", "cts_cluster_size", "cts_cluster_dia",
                "skew_setup", "skew_hold",
                "setup_slack", "hold_slack",
                "setup_tns", "hold_tns",
                "setup_vio_count", "hold_vio_count",
                "power_total", "wirelength", "utilization",
                "clock_buffers", "clock_inverters", "timing_repair_buffers",
                "def_path", "saif_path", "timing_path_csv"
            ]
            writer.writerow(header)

def get_next_run_id():
    if not os.path.exists(LOG_FILE):
        return 0
    max_id = -1
    try:
        with open(LOG_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and row[0].isdigit():
                    current_id = int(row[0])
                    if current_id > max_id:
                        max_id = current_id
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 0
    return max_id + 1

def get_latest_run_tag():
    try:
        with open("latest_run.txt", "r") as f:
            return f.readline().strip()
    except FileNotFoundError:
        return None

def save_essential_files(placement_id, design_name):
    """
    Copy DEF, SAIF, and timing_paths.csv to a permanent location.
    Returns paths to the saved files.
    """
    run_dir = os.path.join("runs", placement_id)
    save_dir = os.path.join(KEPT_FILES_DIR, placement_id)
    os.makedirs(save_dir, exist_ok=True)

    # Source paths
    def_src = os.path.join(run_dir, "11-openroad-detailedplacement", f"{design_name}.def")
    saif_src = os.path.join(run_dir, f"{design_name}.saif")
    timing_src = os.path.join(run_dir, "timing_paths.csv")

    # Destination paths
    def_dst = os.path.join(save_dir, f"{design_name}.def")
    saif_dst = os.path.join(save_dir, f"{design_name}.saif")
    timing_dst = os.path.join(save_dir, "timing_paths.csv")

    saved_paths = {"def_path": "", "saif_path": "", "timing_path": ""}

    for src, dst, key in [
        (def_src, def_dst, "def_path"),
        (saif_src, saif_dst, "saif_path"),
        (timing_src, timing_dst, "timing_path")
    ]:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            saved_paths[key] = dst
            print(f"  Saved: {dst}")
        else:
            print(f"  Warning: {src} not found")

    return saved_paths

def delete_run_directory(placement_id):
    """Delete the entire run directory after saving essential files."""
    run_dir = os.path.join("runs", placement_id)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
        print(f"  Deleted: {run_dir}")

def log_data_to_csv(run_id, placement_id, saved_paths):
    # 1. Load Placement Stats
    pl_stats = {}
    try:
        with open("latest_stats.json", "r") as f:
            pl_stats = json.load(f)
    except FileNotFoundError:
        print("latest_stats.json not found.")
        return

    # 2. Load Parsed Metrics from dataset.json
    metric_map = {}
    dataset_path = os.path.join("runs", placement_id, "dataset.json")

    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r") as f:
                data_list = json.load(f)
                for entry in data_list:
                    metric_map[entry['id']] = entry['metrics']
        except Exception as e:
            print(f"Error reading dataset.json: {e}")
            return
    else:
        print(f"Warning: dataset.json not found for {placement_id}")
        return

    # 3. Check existing IDs
    existing_ids = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].isdigit():
                    existing_ids.add(int(row[0]))

    # 4. Load CTS knobs and write rows
    base_cts_dir = os.path.join("runs", placement_id, "CTS-experiments")
    current_db_id = run_id

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        for i in range(1, 11):
            cts_id = f"CTS-{i}"
            knob_file = os.path.join(base_cts_dir, cts_id, "knobs.json")

            if current_db_id in existing_ids:
                print(f"  Skipping Run ID {current_db_id} (exists)")
                current_db_id += 1
                continue

            cts_knobs = {}
            if os.path.exists(knob_file):
                with open(knob_file, "r") as kf:
                    cts_knobs = json.load(kf)

            m = metric_map.get(cts_id, {})

            row = [
                current_db_id,
                placement_id,
                pl_stats.get("design_name", ""),
                pl_stats.get("aspect_ratio", ""),
                pl_stats.get("core_util", ""),
                pl_stats.get("density", ""),
                pl_stats.get("synth_strategy", ""),
                pl_stats.get("io_mode", ""),
                pl_stats.get("time_driven", ""),
                pl_stats.get("routability_driven", ""),
                cts_knobs.get("CTS_CLK_MAX_WIRE_LENGTH", ""),
                cts_knobs.get("CTS_DISTANCE_BETWEEN_BUFFERS", ""),
                cts_knobs.get("CTS_SINK_CLUSTERING_SIZE", ""),
                cts_knobs.get("CTS_SINK_CLUSTERING_MAX_DIAMETER", ""),
                m.get("skew_setup", ""),
                m.get("skew_hold", ""),
                m.get("setup_slack", ""),
                m.get("hold_slack", ""),
                m.get("setup_tns", ""),
                m.get("hold_tns", ""),
                m.get("setup_vio_count", ""),
                m.get("hold_vio_count", ""),
                m.get("power_total", ""),
                m.get("wirelength", ""),
                m.get("utilization", ""),
                m.get("clock_buffers", ""),
                m.get("clock_inverters", ""),
                m.get("timing_repair_buffers", ""),
                saved_paths["def_path"],
                saved_paths["saif_path"],
                saved_paths["timing_path"],
            ]

            writer.writerow(row)
            current_db_id += 1

    print(f"  Logged {current_db_id - run_id} entries for {placement_id}")

def run_pipeline():
    setup_directories()

    if not os.path.exists(VENV_PYTHON):
        print(f"Error: VENV not found at {VENV_PYTHON}")
        return

    for i in range(NUM_ITERATIONS):
        print(f"\n{'='*60}")
        print(f"Iteration {i+1}/{NUM_ITERATIONS}")
        print(f"{'='*60}")

        # 1. Generate Placement
        print("1. Generating Placement...")
        subprocess.run(["python3", "scripts/1-gen-placement.py"], check=True)
        placement_id = get_latest_run_tag()
        if not placement_id:
            print("Could not get placement ID. Skipping.")
            continue

        # 2. Generate SAIF
        print("2. Generating SAIF...")
        subprocess.run(["python3", "scripts/2-gen-saif.py", placement_id], check=True)

        # 3. Run CTS Swarm (skip graph building)
        print("3. Running CTS Swarm...")
        subprocess.run(["python3", "scripts/5-run-cts.py", placement_id], check=True)

        # 4. Parse Reports
        print("4. Parsing Reports...")
        subprocess.run(["python3", "scripts/6-parse-cts-reports.py", placement_id], check=True)

        # 5. Load design name for file paths
        design_name = ""
        try:
            with open("latest_stats.json", "r") as f:
                design_name = json.load(f).get("design_name", "")
        except FileNotFoundError:
            Exception("latest_stats.json not found. Cannot determine design name for file paths.")     

        # 6. Save essential files (DEF, SAIF, timing_paths.csv)
        print("5. Saving essential files...")
        saved_paths = save_essential_files(placement_id, design_name)

        # 7. Log to CSV (must happen before deletion since it reads from run dir)
        print("6. Logging to database...")
        next_id = get_next_run_id()
        log_data_to_csv(next_id, placement_id, saved_paths)

        # 8. Delete run directory to free disk space
        print("7. Cleaning up...")
        delete_run_directory(placement_id)

        print(f"Done: {placement_id}")

if __name__ == "__main__":
    run_pipeline()